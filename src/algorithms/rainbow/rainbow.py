"""Rainbow: Combining Improvements in Deep Reinforcement Learning.

Hessel et al. (2018), https://arxiv.org/abs/1710.02298

Rainbow combines six independent extensions of DQN (Mnih et al. 2015). This
class extends ``DQNAlgorithm`` and overrides only what those extensions
require; every override is commented with the paper that introduced it.
Each extension is a toggle (``dueling``, ``noisy``, ``double_dqn``,
``distributional``, ``prioritized``) so ablations stay a config change, not a
code change — set any of them to ``False`` to fall back to vanilla DQN
behaviour for that axis.

``configs/experiment/rainbow/atari100k.yaml`` configures this same class as Data-Efficient
Rainbow (van Hasselt et al. 2019), the Atari-100k preset: longer multi-step,
more frequent target updates, and the paper's smaller encoder
(``encoder_type="data_efficient"``).
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Callable, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from torchrl.data import (
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import EnvBase
from torchrl.envs.transforms import MultiStepTransform
from torchrl.envs.transforms.rb_transforms import _multi_step_func
from torchrl.modules import (
    ConvNet,
    DistributionalQValueActor,
    DuelingCnnDQNet,
    EGreedyModule,
    MLP,
    NoisyLinear,
    QValueActor,
)
from torchrl.objectives import DistributionalDQNLoss, DQNLoss, HardUpdate

from src.algorithms.dqn.dqn import DQNAlgorithm
from src.components.transfer_learning import AttentionWeightedPoolingProbe, AttentiveProbe
from src.components.transfer_learning import DINOv2ViTS14Encoder
from src.components.transfer_learning import LeWMViTTiny14Encoder
from src.components.transfer_learning import ResNet18Encoder, ResNet18Variant
from src.components.transfer_learning import SingleQueryAttentiveProbe
from src.components.transfer_learning import configure_encoder_transfer
from src.components.exploration import FixedEpsilonGreedy

# Conv encoder shapes. "dqn" follows the BBF/Dopamine Atari encoder with
# Flax/JAX-style SAME padding; "data_efficient" is the smaller 2-layer encoder
# from Data-Efficient Rainbow (van Hasselt et al. 2019), tuned for the 100k-frame
# Atari-100k budget.
_ENCODER_CNN_KWARGS: dict[str, dict] = {
    "dqn": {
        "same_padding": True,
        "num_cells": [32, 64, 64],
        "kernel_sizes": [8, 4, 3],
        "strides": [4, 2, 1],
        "activation_class": nn.ReLU,
    },
    "data_efficient": {
        "num_cells": [32, 64],
        "kernel_sizes": [5, 5],
        "strides": [5, 5],
        "activation_class": nn.ReLU,
    },
}

class _FlattenFeatures(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 1:
            return x
        return x.flatten(1)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class _SamePadConv2d(nn.Module):
    """Conv2d with Flax/JAX-style SAME padding for Atari encoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2:]
        out_h = math.ceil(in_h / self.stride[0])
        out_w = math.ceil(in_w / self.stride[1])
        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - in_h, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - in_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h or pad_w:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class _SamePaddingConvNet(nn.Module):
    """Nature-DQN CNN with Flax/JAX SAME padding."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_cells: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        activation_class: type[nn.Module] = nn.ReLU,
        activation_kwargs: dict | list[dict] | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(num_cells, kernel_sizes, strides)
        ):
            layers.append(
                _SamePadConv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            if isinstance(activation_kwargs, list):
                kwargs = activation_kwargs[i] if i < len(activation_kwargs) else {}
            else:
                kwargs = activation_kwargs or {}
            layers.append(activation_class(**kwargs))
            current_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        out = self.layers(x).flatten(1)
        if batch_shape:
            return out.reshape(*batch_shape, out.shape[-1])
        return out.reshape(out.shape[-1])


class InclusiveDoneMultiStepTransform(MultiStepTransform):
    """TorchRL MultiStepTransform with n-step bootstrap terminals.

    TorchRL's transform keeps the original done keys and exposes a separate
    ``nonterminal`` key. DQN losses consume ``next.done`` though, so a terminal
    on the last reward inside the n-step target would otherwise still bootstrap.
    """

    def _inv_call(self, tensordict: TensorDict) -> TensorDict | None:
        if not self._validated:
            self._validate()

        total_cat = self._append_tensordict(tensordict)
        if total_cat.shape[-1] <= self.n_steps:
            return None

        out = _multi_step_func(
            total_cat,
            done_key=self.done_key,
            done_keys=self.done_keys,
            reward_keys=self.reward_keys,
            mask_key=self.mask_key,
            n_steps=self.n_steps,
            gamma=self.gamma,
        )
        out = out[..., : -self.n_steps]
        for done_key in self.done_keys:
            existing = out.get(("next", done_key), default=None)
            if existing is None:
                continue
            inclusive_done = self._inclusive_done(total_cat, done_key)[..., : -self.n_steps]
            value = inclusive_done
            while value.ndim < existing.ndim:
                value = value.unsqueeze(-1)
            out.set(("next", done_key), value.expand_as(existing).to(existing.dtype))
        return out

    def _inclusive_done(self, tensordict: TensorDict, done_key: str) -> torch.Tensor:
        done = tensordict.get(("next", done_key)).bool()
        if done.shape != tensordict.shape:
            if done.shape[-1] == 1 and done.shape[:-1] == tensordict.shape:
                done = done.squeeze(-1)
            else:
                done = done.reshape(tensordict.shape)
        padded = F.pad(done.to(torch.int8), (0, self.n_steps - 1), value=0)
        return padded.unfold(-1, self.n_steps, 1).bool().any(dim=-1)


class _TransferRainbowQNet(nn.Module):
    """Rainbow head on top of a spatial transfer-learning encoder."""

    def __init__(
        self,
        *,
        obs_shape: tuple[int, ...],
        num_actions: int,
        hidden_dim: int,
        distributional: bool,
        num_atoms: int,
        dueling: bool,
        layer_class: type[nn.Module],
        layer_kwargs: dict | None,
        encoder_type: str,
        weights: str | None,
        variant: ResNet18Variant,
        dinov2_output_block: int,
        lewm_output_block: int,
        transfer_layer_mix: bool,
        resnet18_mix_layers: Sequence[int] | None,
        dinov2_mix_blocks: Sequence[int] | None,
        lewm_mix_blocks: Sequence[int] | None,
        transfer_mode: str,
        attentive_probe_type: str,
        freeze_encoder_bn: bool,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.distributional = distributional
        self.dueling = dueling
        self.transfer_layer_mix = transfer_layer_mix
        if encoder_type == "resnet18":
            self.encoder = ResNet18Encoder(
                input_channels=obs_shape[0],
                weights=weights,
                variant=variant,
                output_mode="layer_mix" if transfer_layer_mix else "single_layer",
                mix_layers=resnet18_mix_layers,
            )
            self._mix_metric_prefix = "transfer_layer_mix/resnet_layer"
            self._mix_indices = tuple(int(layer) for layer in (resnet18_mix_layers or range(1, 5)))
        elif encoder_type == "dinov2_vits14":
            self.encoder = DINOv2ViTS14Encoder(
                input_channels=obs_shape[0],
                weights=weights,
                output_block=dinov2_output_block,
                output_mode="layer_mix" if transfer_layer_mix else "single_block",
                mix_blocks=dinov2_mix_blocks,
            )
            self._mix_metric_prefix = "transfer_layer_mix/dinov2_block"
            self._mix_indices = tuple(int(block) for block in (dinov2_mix_blocks or range(1, 13)))
        elif encoder_type == "lewm_vit_tiny14":
            self.encoder = LeWMViTTiny14Encoder(
                input_channels=obs_shape[0],
                weights=weights,
                output_block=lewm_output_block,
                output_mode="layer_mix" if transfer_layer_mix else "single_block",
                mix_blocks=lewm_mix_blocks,
            )
            self._mix_metric_prefix = "transfer_layer_mix/lewm_block"
            self._mix_indices = tuple(int(block) for block in (lewm_mix_blocks or range(1, 13)))
        else:
            raise ValueError(f"Unsupported transfer encoder_type={encoder_type!r}")
        configure_encoder_transfer(
            self.encoder,
            transfer_mode=transfer_mode,
            freeze_encoder_bn=freeze_encoder_bn,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        with torch.no_grad():
            latent = self.encoder(torch.zeros(1, *obs_shape))
        kwargs = layer_kwargs or {}
        if transfer_layer_mix:
            latents = list(latent)
            self.mix_logits = nn.Parameter(torch.zeros(len(latents)))
            self.projection = nn.ModuleList(
                [
                    self._make_projection(
                        spatial_latent=item,
                        hidden_dim=hidden_dim,
                        transfer_mode=transfer_mode,
                        attentive_probe_type=attentive_probe_type,
                        layer_class=layer_class,
                        layer_kwargs=kwargs,
                    )
                    for item in latents
                ]
            )
        else:
            self.mix_logits = None
            self.projection = self._make_projection(
                spatial_latent=latent,
                hidden_dim=hidden_dim,
                transfer_mode=transfer_mode,
                attentive_probe_type=attentive_probe_type,
                layer_class=layer_class,
                layer_kwargs=kwargs,
            )
        head_out = num_actions * num_atoms if distributional else num_actions
        self.advantage = layer_class(hidden_dim, head_out, **kwargs)
        self.value = None
        if dueling:
            value_out = num_atoms if distributional else 1
            self.value = layer_class(hidden_dim, value_out, **kwargs)

    def _make_projection(
        self,
        *,
        spatial_latent: torch.Tensor,
        hidden_dim: int,
        transfer_mode: str,
        attentive_probe_type: str,
        layer_class: type[nn.Module],
        layer_kwargs: dict,
    ) -> nn.Module:
        if transfer_mode == "attentive_probe":
            probe_kwargs = {
                "in_channels": int(spatial_latent.shape[1]),
                "out_features": hidden_dim,
                "num_tokens": int(spatial_latent.flatten(2).shape[-1]),
            }
            if attentive_probe_type == "self_attention":
                return AttentiveProbe(**probe_kwargs)
            if attentive_probe_type == "single_query":
                return SingleQueryAttentiveProbe(**probe_kwargs)
            if attentive_probe_type == "attention_weighted_pooling":
                return AttentionWeightedPoolingProbe(**probe_kwargs)
            raise ValueError(f"Unsupported attentive_probe_type={attentive_probe_type!r}")
        in_features = int(spatial_latent.flatten(1).shape[-1])
        return nn.Sequential(
            nn.Flatten(),
            layer_class(in_features, hidden_dim, **layer_kwargs),
        )

    def _project(self, latent: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if not self.transfer_layer_mix:
            return self.projection(latent)
        assert isinstance(self.projection, nn.ModuleList)
        assert self.mix_logits is not None
        latents = list(latent)
        weights = self.mix_logits.softmax(dim=0).to(dtype=latents[0].dtype, device=latents[0].device)
        projected = [
            projection(item) * weights[index]
            for index, (projection, item) in enumerate(zip(self.projection, latents, strict=True))
        ]
        return torch.stack(projected, dim=0).sum(dim=0)

    def layer_mix_metrics(self) -> dict[str, float]:
        if self.mix_logits is None:
            return {}
        weights = self.mix_logits.softmax(dim=0).detach().cpu()
        return {
            f"{self._mix_metric_prefix}_{index:02d}": float(weight)
            for index, weight in zip(self._mix_indices, weights, strict=True)
        }

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        h = F.relu(self._project(self.encoder(pixels)))
        if self.distributional:
            adv = self.advantage(h).view(-1, self.num_actions, self.num_atoms)
            if self.dueling and self.value is not None:
                value = self.value(h).view(-1, 1, self.num_atoms)
                logits = value + adv - adv.mean(dim=1, keepdim=True)
            else:
                logits = adv
            return logits.transpose(1, 2)

        q_values = self.advantage(h)
        if self.dueling and self.value is not None:
            value = self.value(h)
            q_values = value + q_values - q_values.mean(dim=1, keepdim=True)
        return q_values


class _CnnRainbowQNet(nn.Module):
    """Rainbow head for explicit layer classes that TorchRL cannot lazy-build."""

    def __init__(
        self,
        *,
        obs_shape: tuple[int, ...],
        cnn_kwargs: dict,
        same_padding: bool = False,
        num_actions: int,
        hidden_dim: int,
        distributional: bool,
        num_atoms: int,
        dueling: bool,
        layer_class: type[nn.Module],
        layer_kwargs: dict | None,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.distributional = distributional
        self.dueling = dueling
        if same_padding:
            self.encoder = _SamePaddingConvNet(
                in_channels=obs_shape[0],
                **cnn_kwargs,
            )
        else:
            self.encoder = ConvNet(**cnn_kwargs)
        with torch.no_grad():
            latent = self.encoder(torch.zeros(1, *obs_shape))
        kwargs = layer_kwargs or {}
        self.projection = nn.Sequential(
            _FlattenFeatures(),
            layer_class(int(latent.flatten(1).shape[-1]), hidden_dim, **kwargs),
        )
        head_out = num_actions * num_atoms if distributional else num_actions
        self.advantage = layer_class(hidden_dim, head_out, **kwargs)
        self.value = None
        if dueling:
            value_out = num_atoms if distributional else 1
            self.value = layer_class(hidden_dim, value_out, **kwargs)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.projection(self.encoder(pixels)))
        if self.distributional:
            adv = self.advantage(h).view(-1, self.num_actions, self.num_atoms)
            if self.dueling and self.value is not None:
                value = self.value(h).view(-1, 1, self.num_atoms)
                logits = value + adv - adv.mean(dim=1, keepdim=True)
            else:
                logits = adv
            return logits.transpose(1, 2)

        q_values = self.advantage(h)
        if self.dueling and self.value is not None:
            value = self.value(h)
            q_values = value + q_values - q_values.mean(dim=1, keepdim=True)
        return q_values


class RainbowAlgorithm(DQNAlgorithm):
    """DQN + double Q-learning + dueling + PER + multi-step + C51 + noisy nets."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "pixels",
        lr: float = 1e-4,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.99,
        batch_size: int = 32,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_eval: float = 0.001,
        annealing_frames: int = 250_000,
        frames_per_batch: int = 4,
        init_random_frames: int = 20_000,
        max_frames_per_traj: int = -1,
        num_updates: int = 4,
        hard_update_freq: int = 8_000,
        replay_capacity: int = 1_000_000,
        encoder_type: Literal["dqn", "data_efficient", "resnet18", "dinov2_vits14", "lewm_vit_tiny14"] = "dqn",
        hidden_dim: int = 512,
        resnet18_weights: str | None = None,
        resnet18_variant: ResNet18Variant = "resnet_layer3_reduced",
        dinov2_weights: str | None = "models/dinov2_vits14_pretrain.pth",
        dinov2_output_block: int = 3,
        lewm_weights: str | None = "models/lewm_pusht_weights.pt",
        lewm_output_block: int = 7,
        transfer_layer_mix: bool = False,
        resnet18_mix_layers: Sequence[int] | None = None,
        dinov2_mix_blocks: Sequence[int] | None = None,
        lewm_mix_blocks: Sequence[int] | None = None,
        transfer_mode: Literal[
            "none",
            "full_finetune",
            "linear_probe",
            "attentive_probe",
            "lora",
        ] = "none",
        encoder_lr: float | None = None,
        adapter_lr: float | None = None,
        probe_lr: float | None = None,
        attentive_probe_type: Literal[
            "self_attention",
            "single_query",
            "attention_weighted_pooling",
        ] = "self_attention",
        encoder_lr_scale: float | None = None,
        freeze_encoder_bn: bool = False,
        lora_rank: int = 1,
        lora_alpha: float = 2.0,
        lora_dropout: float = 0.0,
        # --- Wang et al. (2016), "Dueling Network Architectures for Deep RL" ---
        dueling: bool = True,
        # --- Fortunato et al. (2018), "Noisy Networks for Exploration" ---------
        noisy: bool = True,
        noisy_std: float = 0.1,
        eval_noise: bool = True,
        # --- van Hasselt et al. (2016), "Deep RL with Double Q-learning" -------
        # Only takes effect when `distributional=False`: `DistributionalDQNLoss`
        # always selects the next action with the online network and evaluates
        # it with the target network internally, so it is unconditionally
        # "double" regardless of this flag.
        double_dqn: bool = True,
        # --- Bellemare et al. (2017), "A Distributional Perspective on RL" -----
        distributional: bool = True,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # --- Schaul et al. (2016), "Prioritized Experience Replay" -------------
        prioritized: bool = True,
        prb_alpha: float = 0.5,
        prb_beta_start: float = 0.4,
        prb_beta_end: float = 1.0,
        prb_beta_frames: int = 100_000,
        prb_eps: float = 1e-6,
        # --- Multi-step returns (Sutton 1988; used in Rainbow) -----------------
        n_steps: int = 3,
    ) -> None:
        # DQNAlgorithm's `network`/`replay_buffer` factory defaults are stored
        # but never invoked: `setup()` below is a full override that builds
        # both directly, since Rainbow's architecture/buffer are intrinsically
        # coupled to the toggles above (TD-MPC2 precedent, see algorithm
        # README's "Documented deviations").
        super().__init__(
            device,
            obs_key=obs_key,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            eps_start=eps_start,
            eps_end=eps_end,
            annealing_frames=annealing_frames,
            frames_per_batch=frames_per_batch,
            init_random_frames=init_random_frames,
            max_frames_per_traj=max_frames_per_traj,
            num_updates=num_updates,
            hard_update_freq=hard_update_freq,
        )
        self.replay_capacity = replay_capacity
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.resnet18_weights = resnet18_weights
        self.resnet18_variant = resnet18_variant
        self.dinov2_weights = dinov2_weights
        self.dinov2_output_block = dinov2_output_block
        self.lewm_weights = lewm_weights
        self.lewm_output_block = lewm_output_block
        self.transfer_layer_mix = transfer_layer_mix
        self.resnet18_mix_layers = resnet18_mix_layers
        self.dinov2_mix_blocks = dinov2_mix_blocks
        self.lewm_mix_blocks = lewm_mix_blocks
        self.transfer_mode = transfer_mode
        self.encoder_lr = encoder_lr
        self.adapter_lr = adapter_lr
        self.probe_lr = probe_lr
        self.attentive_probe_type = attentive_probe_type
        self.encoder_lr_scale = encoder_lr_scale
        self.freeze_encoder_bn = freeze_encoder_bn
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.dueling = dueling
        self.noisy = noisy
        self.noisy_std = noisy_std
        self.eval_noise = eval_noise
        self.eps_eval = eps_eval
        self.double_dqn = double_dqn
        self.distributional = distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.prioritized = prioritized
        self.prb_alpha = prb_alpha
        self.prb_beta_start = prb_beta_start
        self.prb_beta_end = prb_beta_end
        self.prb_beta_frames = prb_beta_frames
        self.prb_eps = prb_eps
        self.n_steps = n_steps
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        self.action_spec = action_spec
        num_actions = int(action_spec.space.n)
        proof_env.close()

        # 1. Q-network. Dueling (Wang et al. 2016) splits the head into a
        #    state-value and an action-advantage stream; noisy layers
        #    (Fortunato et al. 2018) replace the dense head only — the conv
        #    encoder stays plain, matching the paper. Distributional
        #    (Bellemare et al. 2017) reshapes the output to
        #    [*, num_atoms, num_actions] so raw Q-values become per-atom logits.
        layer_class = NoisyLinear if self.noisy else nn.Linear
        layer_kwargs = (
            {
                "std_init": self.noisy_std,
            }
            if self.noisy
            else None
        )
        out_features = (self.num_atoms, num_actions) if self.distributional else num_actions
        out_features_value = (self.num_atoms, 1) if self.distributional else 1
        if self.encoder_type in {"resnet18", "dinov2_vits14", "lewm_vit_tiny14"}:
            weights = (
                self.resnet18_weights
                if self.encoder_type == "resnet18"
                else self.dinov2_weights
                if self.encoder_type == "dinov2_vits14"
                else self.lewm_weights
            )
            q_net = _TransferRainbowQNet(
                obs_shape=obs_shape,
                num_actions=num_actions,
                hidden_dim=self.hidden_dim,
                distributional=self.distributional,
                num_atoms=self.num_atoms,
                dueling=self.dueling,
                layer_class=layer_class,
                layer_kwargs=layer_kwargs,
                encoder_type=self.encoder_type,
                weights=weights,
                variant=self.resnet18_variant,
                dinov2_output_block=self.dinov2_output_block,
                lewm_output_block=self.lewm_output_block,
                transfer_layer_mix=self.transfer_layer_mix,
                resnet18_mix_layers=self.resnet18_mix_layers,
                dinov2_mix_blocks=self.dinov2_mix_blocks,
                lewm_mix_blocks=self.lewm_mix_blocks,
                transfer_mode=self.transfer_mode,
                attentive_probe_type=self.attentive_probe_type,
                freeze_encoder_bn=self.freeze_encoder_bn,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )
        else:
            cnn_kwargs = dict(_ENCODER_CNN_KWARGS[self.encoder_type])
            same_padding = bool(cnn_kwargs.pop("same_padding", False))
            if same_padding:
                q_net = _CnnRainbowQNet(
                    obs_shape=obs_shape,
                    cnn_kwargs=cnn_kwargs,
                    same_padding=same_padding,
                    num_actions=num_actions,
                    hidden_dim=self.hidden_dim,
                    distributional=self.distributional,
                    num_atoms=self.num_atoms,
                    dueling=self.dueling,
                    layer_class=layer_class,
                    layer_kwargs=layer_kwargs,
                )
            elif self.dueling:
                q_net = DuelingCnnDQNet(
                    out_features=out_features,
                    out_features_value=out_features_value,
                    cnn_kwargs=cnn_kwargs,
                    mlp_kwargs={
                        "num_cells": [self.hidden_dim],
                        "layer_class": layer_class,
                        "layer_kwargs": layer_kwargs,
                    },
                )
            else:
                cnn = ConvNet(**cnn_kwargs)
                with torch.no_grad():
                    cnn_out = cnn(torch.zeros(1, *obs_shape))
                mlp = MLP(
                    in_features=cnn_out.shape[-1],
                    out_features=out_features,
                    num_cells=[self.hidden_dim],
                    activation_class=nn.ReLU,
                    layer_class=layer_class,
                    layer_kwargs=layer_kwargs,
                )
                q_net = nn.Sequential(cnn, mlp)
        q_net = q_net.to(self.device)
        # DuelingCnnDQNet's advantage/value heads are LazyLinear internally
        # (their input size depends on the conv output, which isn't known
        # until a forward pass); materialize them now so the loss module's
        # functional parameter conversion below doesn't see uninitialized
        # parameters.
        with torch.no_grad():
            q_net(torch.zeros(1, *obs_shape, device=self.device))
        if self.noisy:
            _sample_noisy_linear_on_forward(q_net)

        # 2. Actor wrapper.
        if self.distributional:
            support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
            self.q_actor = DistributionalQValueActor(
                module=q_net,
                support=support,
                spec=action_spec,
                in_keys=[self.obs_key],
            ).to(self.device)
        else:
            self.q_actor = QValueActor(
                module=q_net,
                spec=action_spec,
                in_keys=[self.obs_key],
            ).to(self.device)

        # 3. Exploration. DER keeps epsilon-greedy on top of NoisyNet.
        self.greedy_module = EGreedyModule(
            spec=action_spec,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.annealing_frames,
            device=self.device,
        )
        if self.noisy:
            self.q_actor.train()
        self._explore_policy = TensorDictSequential(
            self.q_actor,
            self.greedy_module,
            _SqueezeUnbatchedActionModule(),
        )

        # 4. Replay buffer. Prioritized sampling (Schaul et al. 2016) biases
        #    sampling toward high-TD-error transitions; the importance-sampling
        #    exponent beta is annealed 0.4 -> 1.0 in step() below, following
        #    the paper. Multi-step returns (as used in Rainbow; n-step
        #    bootstrapping traces to Sutton 1988) are applied at write time via
        #    `MultiStepTransform`, which is unbiased by collector-batch
        #    boundaries (unlike the collector-side `MultiStep` postproc).
        storage = LazyTensorStorage(max_size=self.replay_capacity, device="cpu")
        transform = InclusiveDoneMultiStepTransform(n_steps=self.n_steps, gamma=self.gamma) if self.n_steps > 1 else None
        if self.prioritized:
            self.replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=self.prb_alpha,
                beta=self.prb_beta_start,
                eps=self.prb_eps,
                storage=storage,
                transform=transform,
            )
        else:
            self.replay_buffer = TensorDictReplayBuffer(storage=storage, transform=transform)

        # 5. Loss. `DistributionalDQNLoss` computes the C51 categorical
        #    projection (Bellemare et al. 2017) and always uses double-DQN
        #    action selection internally. Otherwise plain `DQNLoss` with the
        #    `double_dqn` toggle (van Hasselt et al. 2016).
        if self.distributional:
            self.loss_module = DistributionalDQNLoss(
                self.q_actor, gamma=self.gamma, delay_value=True
            )
        else:
            self.loss_module = DQNLoss(
                value_network=self.q_actor,
                loss_function="l2",
                delay_value=True,
                double_dqn=self.double_dqn,
            )
            self.loss_module.make_value_estimator(gamma=self.gamma)
        self.loss_module = self.loss_module.to(self.device)
        self.target_updater = HardUpdate(
            self.loss_module, value_network_update_interval=self.hard_update_freq
        )
        self.optimizer = self._make_optimizer()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        if self.encoder_type not in {"resnet18", "dinov2_vits14", "lewm_vit_tiny14"}:
            return torch.optim.Adam(
                self.q_actor.parameters(),
                lr=self.lr,
                eps=self.adam_eps,
                weight_decay=self.weight_decay,
            )

        encoder_params = []
        adapter_params = []
        probe_params = []
        head_params = []
        for name, parameter in self.q_actor.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".encoder.input_adapter." in name or ".encoder.reducer." in name:
                adapter_params.append(parameter)
            elif self.probe_lr is not None and (
                ".projection." in name or name.startswith("projection.")
            ):
                probe_params.append(parameter)
            elif ".encoder." in name:
                encoder_params.append(parameter)
            else:
                head_params.append(parameter)

        groups = []
        if encoder_params:
            encoder_lr = (
                self.encoder_lr
                if self.encoder_lr is not None
                else self.lr * (1.0 if self.encoder_lr_scale is None else self.encoder_lr_scale)
            )
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if adapter_params:
            groups.append(
                {
                    "params": adapter_params,
                    "lr": self.lr if self.adapter_lr is None else self.adapter_lr,
                }
            )
        if probe_params:
            groups.append({"params": probe_params, "lr": self.probe_lr})
        if head_params:
            groups.append({"params": head_params, "lr": self.lr})
        return torch.optim.Adam(
            groups,
            lr=self.lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

    def summary_metrics(self) -> dict[str, float]:
        for module in self.q_actor.modules():
            metrics_fn = getattr(module, "layer_mix_metrics", None)
            if metrics_fn is not None:
                metrics = metrics_fn()
                if metrics:
                    return metrics
        return {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch) -> dict[str, float]:
        batch = batch.reshape(-1)
        _squeeze_policy_singletons(batch)
        _canonicalize_one_hot_action(batch, self.action_spec)
        self.replay_buffer.extend(batch)
        self._collected_frames += batch.numel()

        if self._collected_frames < self.init_random_frames:
            return {"train/epsilon": 1.0}
        self.greedy_module.step(batch.numel())

        losses = torch.zeros(self.num_updates, device=self.device)
        for j in range(self.num_updates):
            sample = self.replay_buffer.sample(self.batch_size).to(self.device)
            _canonicalize_one_hot_action(sample, self.action_spec)
            # MultiStepTransform writes "steps_to_next_obs" with shape [B]
            # instead of [B, 1]; DQNLoss/DistributionalDQNLoss broadcast it
            # directly against [B, 1]-shaped reward/terminated, so a bare [B]
            # silently mis-broadcasts into [B, B]. Align the trailing dim.
            steps_key = "steps_to_next_obs"
            if steps_key in sample.keys() and sample.get(steps_key).dim() == 1:
                sample.set(steps_key, sample.get(steps_key).unsqueeze(-1))
            loss = self.loss_module(sample)["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.target_updater.step()

            if self.prioritized:
                # Schaul et al. (2016): re-prioritize sampled transitions from
                # the per-sample TD error the loss module wrote into `sample`.
                self.replay_buffer.update_tensordict_priority(sample)
                self._anneal_prb_beta()

            losses[j] = loss.detach()

        return {
            "train/q_loss": losses.mean().item(),
            "train/epsilon": float(self.greedy_module.eps) if self.greedy_module else 0.0,
        }

    def _anneal_prb_beta(self) -> None:
        """Linearly anneal the PER importance-sampling exponent (Schaul et al. 2016)."""
        if self.prb_beta_frames <= 0:
            return
        fraction = min(1.0, self._collected_frames / self.prb_beta_frames)
        self.replay_buffer.sampler.beta = (
            self.prb_beta_start + (self.prb_beta_end - self.prb_beta_start) * fraction
        )

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self):
        # Keep the network in eval mode for non-noisy modules, but preserve the
        # DER/Dopamine eval-noise option by leaving NoisyLinear layers stochastic.
        self.q_actor.eval()
        _set_noisy_linear_training(self.q_actor, self.eval_noise)
        return TensorDictSequential(
            self.q_actor,
            FixedEpsilonGreedy(self.action_spec, self.eps_eval),
            _SqueezePolicySingletonsModule(),
        )


def _sample_noisy_linear_on_forward(module: nn.Module) -> None:
    def forward_with_fresh_noise(layer: NoisyLinear, input: torch.Tensor) -> torch.Tensor:
        if not layer.training:
            return F.linear(input, layer.weight_mu, layer.bias_mu)
        epsilon_in = layer._scale_noise(layer.in_features)
        epsilon_out = layer._scale_noise(layer.out_features)
        weight = layer.weight_mu + layer.weight_sigma * epsilon_out.outer(epsilon_in)
        bias = None
        if layer.bias_mu is not None:
            bias = layer.bias_mu + layer.bias_sigma * epsilon_out
        return F.linear(input, weight, bias)

    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.forward = MethodType(forward_with_fresh_noise, child)


def _set_noisy_linear_training(module: nn.Module, training: bool) -> None:
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.train(training)


def _squeeze_policy_singletons(batch) -> None:
    """Keep collector output shapes stable before writing them to replay."""
    for key in ("action", "action_value"):
        value = batch.get(key, default=None)
        if value is not None and value.dim() > 2 and value.shape[-2] == 1:
            batch.set(key, value.squeeze(-2))


def _canonicalize_one_hot_action(batch, action_spec) -> None:
    action = batch.get("action", default=None)
    if action is None or action.dim() == 0:
        return
    spec_shape = tuple(getattr(action_spec, "shape", ()))
    if not spec_shape:
        return
    num_actions = int(spec_shape[-1])
    if num_actions <= 1 or action.shape[-1] != num_actions:
        return
    indices = action.argmax(dim=-1)
    canonical = F.one_hot(indices, num_classes=num_actions).to(dtype=action.dtype)
    batch.set("action", canonical)


class _SqueezePolicySingletonsModule(TensorDictModuleBase):
    """Normalize policy output shapes before the collector stacks them."""

    def __init__(self) -> None:
        self.in_keys = []
        self.out_keys = []
        super().__init__()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        _squeeze_policy_singletons(tensordict)
        return tensordict


class _SqueezeUnbatchedActionModule(TensorDictModuleBase):
    """Match the unbatched action spec shape during collection."""

    def __init__(self) -> None:
        self.in_keys = []
        self.out_keys = []
        super().__init__()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get("action", default=None)
        if (
            action is not None
            and len(tensordict.batch_size) == 0
            and action.dim() > 1
            and action.shape[0] == 1
        ):
            tensordict.set("action", action.squeeze(0))
        return tensordict
