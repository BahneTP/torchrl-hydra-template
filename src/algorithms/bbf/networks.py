"""Networks for BBF (Schwarzer et al. 2023).

One module, :class:`BBFNetwork`, owns every learned component:

    encoder            Impala-CNN ResNet (Espeholt et al. 2018), width-scaled.
    transition_model   Latent-space dynamics for the SPR rollout.
    projection         Linear(flat_latent -> hidden_dim); also the first layer
                       of the Q-head, as in SPR (Schwarzer et al. 2021).
    predictor          Linear(hidden_dim -> hidden_dim), online branch only.
    advantage / value  Dueling, distributional (C51) heads.

Mirrors ``spr_networks.py`` of the official JAX release
(google-research/bigger_better_faster): ReLU activations, max-pool 3x3/2 per
stage, two residual blocks per stage, per-sample min-max renormalisation of
latents, xavier-uniform kernels with zero biases (the official initializer;
this also sets the scale of shrink-and-perturb noise and freshly reset
heads), and ``forward()`` returning scalar Q-values so the module drops into
``torchrl.modules.QValueActor`` unchanged.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.transfer_learning import ResNet18Encoder, configure_encoder_transfer


def init_xavier(module: nn.Module) -> None:
    """Official BBF initialisation: xavier-uniform kernels, zero biases
    (flax ``nn.initializers.xavier_uniform()`` + default zero bias init)."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def renormalize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalisation over all non-batch dims (BBF's
    ``renormalize`` in the official code). Keeps latent scale bounded, which
    stabilises the high replay-ratio regime."""
    flat = x.flatten(1)
    mn = flat.min(dim=-1, keepdim=True).values
    mx = flat.max(dim=-1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + 1e-5)
    return flat.view_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(h))
        return x + h


class ImpalaStage(nn.Module):
    """Conv -> max-pool(3x3, stride 2) -> residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.pool(self.conv(x)))


class ImpalaCNN(nn.Module):
    """Impala-CNN encoder; BBF scales ``dims`` by ``width_scale=4``."""

    def __init__(
        self,
        in_channels: int,
        dims: Sequence[int] = (16, 32, 32),
        width_scale: int = 4,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        widths = [int(d * width_scale) for d in dims]
        stages = []
        prev = in_channels
        for w in widths:
            stages.append(ImpalaStage(prev, w, num_blocks))
            prev = w
        self.stages = nn.Sequential(*stages)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.stages(x))


class TransitionModel(nn.Module):
    """Latent dynamics for SPR: z_{t+1} = f(z_t, a_t).

    The action enters as a one-hot map concatenated channel-wise, followed by
    two 3x3 convs with ReLU (``ConvTMCell`` in the official code). The output
    is renormalised like the encoder output so rolled-out latents stay in the
    same value range as encoded ones.
    """

    def __init__(self, latent_channels: int, num_actions: int, renorm: bool = True) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.renorm = renorm
        self.conv1 = nn.Conv2d(latent_channels + num_actions, latent_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b, _, h, w = latent.shape
        onehot = F.one_hot(action.long(), self.num_actions).float()
        action_map = onehot.view(b, self.num_actions, 1, 1).expand(b, self.num_actions, h, w)
        x = torch.cat([latent, action_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return renormalize(x) if self.renorm else x


class SpatialSelfAttentionProbe(nn.Module):
    """Self-attention over spatial tokens, preserving the BBF latent shape."""

    def __init__(self, channels: int = 128, height: int = 11, width: int = 11) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.position_embedding = nn.Parameter(torch.zeros(1, height * width, channels))
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.mlp_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.apply(init_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.height * self.width or tokens.shape[2] != self.channels:
            raise ValueError(
                "SpatialSelfAttentionProbe expects "
                f"{self.channels}x{self.height}x{self.width}, got {tuple(x.shape[1:])}."
            )
        tokens = tokens + self.position_embedding
        attended, _ = self.attention(
            self.attention_norm(tokens),
            self.attention_norm(tokens),
            self.attention_norm(tokens),
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return tokens.transpose(1, 2).reshape(x.shape[0], self.channels, self.height, self.width)


class _ResNetLayerProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.projection.apply(init_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (11, 11):
            x = F.interpolate(x, size=(11, 11), mode="bilinear", align_corners=False)
        return self.projection(x)


class BBFResNet18TransferEncoder(nn.Module):
    """ResNet18 transfer encoder that preserves BBF's 128x11x11 latent interface."""

    _LAYER_CHANNELS = (64, 128, 256, 512)

    def __init__(
        self,
        *,
        input_channels: int,
        weights: str | None,
        transfer_mode: str,
        freeze_encoder_bn: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        transfer_layer_mix: bool,
        mix_layers: Sequence[int] | None,
        attentive_probe_type: str,
    ) -> None:
        super().__init__()
        if attentive_probe_type != "self_attention":
            raise ValueError("BBF attentive probing currently supports only self_attention.")
        self.transfer_mode = transfer_mode
        self.transfer_layer_mix = transfer_layer_mix
        self.base = ResNet18Encoder(
            input_channels=input_channels,
            weights=weights,
            variant="resnet_layer2",
            output_mode="layer_mix" if transfer_layer_mix else "single_layer",
            mix_layers=mix_layers,
        )
        configure_encoder_transfer(
            self.base,
            transfer_mode=transfer_mode,
            freeze_encoder_bn=freeze_encoder_bn,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        if transfer_layer_mix:
            layers = tuple(int(layer) for layer in (mix_layers or range(1, 5)))
            self.mix_layers = layers
            self.projectors = nn.ModuleList(
                [_ResNetLayerProjector(self._LAYER_CHANNELS[layer - 1]) for layer in layers]
            )
            self.mix_logits = nn.Parameter(torch.zeros(len(layers)))
            if transfer_mode == "attentive_probe":
                self.spatial_probe = SpatialSelfAttentionProbe()
            else:
                self.spatial_probe = nn.Identity()
        else:
            self.mix_layers = (2,)
            self.projectors = None
            self.mix_logits = None
            self.spatial_probe = (
                SpatialSelfAttentionProbe() if transfer_mode == "attentive_probe" else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.base(x)
        if not self.transfer_layer_mix:
            return self.spatial_probe(latent)
        assert isinstance(latent, list)
        assert self.projectors is not None
        assert self.mix_logits is not None
        weights = self.mix_logits.softmax(dim=0).to(dtype=latent[0].dtype, device=latent[0].device)
        projected = [
            projector(item) * weights[index]
            for index, (projector, item) in enumerate(zip(self.projectors, latent, strict=True))
        ]
        return self.spatial_probe(torch.stack(projected, dim=0).sum(dim=0))

    def layer_mix_metrics(self) -> dict[str, float]:
        if self.mix_logits is None:
            return {}
        weights = self.mix_logits.softmax(dim=0).detach().cpu()
        return {
            f"transfer_layer_mix/resnet_layer_{layer:02d}": float(weight)
            for layer, weight in zip(self.mix_layers, weights, strict=True)
        }


class BBFNetwork(nn.Module):
    """Encoder + SPR heads + dueling distributional Q-head.

    ``forward(pixels)`` returns scalar Q-values (B, A) — the expectation of
    the C51 distribution — so the module can be wrapped by ``QValueActor``
    for greedy action selection. Training uses the finer-grained methods
    (``encode``, ``q_logits``, ``project``, ``predict``, ``transition_model``).
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        *,
        dims: Sequence[int] = (16, 32, 32),
        width_scale: int = 4,
        blocks_per_stage: int = 2,
        hidden_dim: int = 2048,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        dueling: bool = True,
        renorm: bool = True,
        encoder_type: str = "impala",
        resnet18_weights: str | None = None,
        transfer_mode: str = "none",
        transfer_layer_mix: bool = False,
        resnet18_mix_layers: Sequence[int] | None = None,
        attentive_probe_type: str = "self_attention",
        freeze_encoder_bn: bool = False,
        lora_rank: int = 1,
        lora_alpha: float = 2.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.dueling = dueling
        self.renorm = renorm
        self.encoder_type = encoder_type
        if encoder_type == "impala":
            self.encoder = ImpalaCNN(
                obs_shape[0], dims=dims, width_scale=width_scale, num_blocks=blocks_per_stage
            )
        elif encoder_type == "resnet18":
            self.encoder = BBFResNet18TransferEncoder(
                input_channels=obs_shape[0],
                weights=resnet18_weights,
                transfer_mode=transfer_mode,
                freeze_encoder_bn=freeze_encoder_bn,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                transfer_layer_mix=transfer_layer_mix,
                mix_layers=resnet18_mix_layers,
                attentive_probe_type=attentive_probe_type,
            )
        else:
            raise ValueError(f"Unsupported BBF encoder_type={encoder_type!r}")
        with torch.no_grad():
            latent = self.encoder(torch.zeros(1, *obs_shape))
        self.latent_shape = tuple(latent.shape[1:])
        flat_dim = int(math.prod(self.latent_shape))

        self.transition_model = TransitionModel(self.latent_shape[0], num_actions, renorm=renorm)
        self.projection = nn.Linear(flat_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, num_actions * num_atoms)
        self.value = nn.Linear(hidden_dim, num_atoms)
        if encoder_type == "impala":
            self.apply(init_xavier)
        else:
            self.transition_model.apply(init_xavier)
            self.projection.apply(init_xavier)
            self.predictor.apply(init_xavier)
            self.advantage.apply(init_xavier)
            self.value.apply(init_xavier)
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def resetable_parameter_names(self) -> set[str]:
        names = {"transition_model."}
        names.update(
            f"encoder.{name}"
            for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad
        )
        return names

    def layer_mix_metrics(self) -> dict[str, float]:
        if hasattr(self.encoder, "layer_mix_metrics"):
            return self.encoder.layer_mix_metrics()
        return {}

    # --- representation ------------------------------------------------

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels (B, 4, 84, 84) in [0, 1] -> renormalised spatial latent."""
        z = self.encoder(pixels)
        return renormalize(z) if self.renorm else z

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        """Spatial latent -> hidden_dim vector (first Q-head layer, pre-ReLU).

        Used both as the Q-head trunk and as the SPR projection (the SPR
        design: tie the projection to the value function's representation).
        As in the official release, the SPR branches (``predict`` and the
        target's ``encode_project``) consume this raw linear output; only the
        Q-head applies a ReLU on top (``q_logits``).
        """
        return self.projection(latent.flatten(1))

    def predict(self, projection: torch.Tensor) -> torch.Tensor:
        """SPR predictor head (online branch only)."""
        return self.predictor(projection)

    # --- value head ------------------------------------------------------

    def q_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """Spatial latent -> C51 logits (B, A, num_atoms)."""
        h = F.relu(self.project(latent))
        adv = self.advantage(h).view(-1, self.num_actions, self.num_atoms)
        if not self.dueling:
            return adv
        val = self.value(h).view(-1, 1, self.num_atoms)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def q_values_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) * self.support).sum(-1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Scalar Q-values for action selection (QValueActor interface)."""
        squeeze = pixels.dim() == 3
        if squeeze:  # unbatched env step
            pixels = pixels.unsqueeze(0)
        q = self.q_values_from_logits(self.q_logits(self.encode(pixels)))
        return q.squeeze(0) if squeeze else q


class SACBBFNetwork(BBFNetwork):
    """BBF backbone with an additional discrete policy head for SAC-BBF."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        flat_dim = int(math.prod(self.latent_shape))
        self.policy_projection = nn.Linear(flat_dim, self.projection.out_features)
        self.predict_policy = nn.Linear(self.projection.out_features, self.projection.out_features)
        self.policy = nn.Linear(self.projection.out_features, self.num_actions)
        self._log_alpha = nn.Parameter(torch.zeros(()))
        self.policy_projection.apply(init_xavier)
        self.predict_policy.apply(init_xavier)
        self.policy.apply(init_xavier)

    def entropy_scale(self) -> torch.Tensor:
        return self._log_alpha.exp()

    def policy_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.policy_projection(latent.flatten(1)))
        return self.policy(h)

    def policy_logits(self, pixels: torch.Tensor) -> torch.Tensor:
        squeeze = pixels.dim() == 3
        if squeeze:
            pixels = pixels.unsqueeze(0)
        logits = self.policy_logits_from_latent(self.encode(pixels))
        return logits.squeeze(0) if squeeze else logits

    def sac_project(self, latent: torch.Tensor) -> torch.Tensor:
        flat = latent.flatten(1)
        return torch.cat([self.project(latent), self.policy_projection(flat)], dim=-1)

    def sac_predict(self, latent: torch.Tensor) -> torch.Tensor:
        flat = latent.flatten(1)
        return torch.cat(
            [
                self.predict(self.project(latent)),
                self.predict_policy(self.policy_projection(flat)),
            ],
            dim=-1,
        )
