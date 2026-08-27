"""Transfer-learning building blocks for Atari frame-stack encoders."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


ResNet18Variant = Literal[
    "resnet_full",
    "resnet_layer1",
    "resnet_layer1_reduced",
    "resnet_layer2",
    "resnet_layer2_reduced",
    "resnet_layer3",
    "resnet_layer3_flattened",
    "resnet_layer3_reduced",
    "resnet_layer4",
    "resnet_layer4_reduced",
]

InitializerName = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
]


def apply_initializer(module: nn.Module, initializer: InitializerName) -> None:
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return
    if initializer == "xavier_uniform":
        nn.init.xavier_uniform_(module.weight)
    elif initializer == "xavier_normal":
        nn.init.xavier_normal_(module.weight)
    elif initializer == "kaiming_uniform":
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    elif initializer == "kaiming_normal":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif initializer == "orthogonal":
        nn.init.orthogonal_(module.weight)
    else:
        raise NotImplementedError(f"Unsupported initializer: {initializer}")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class AttentiveProbe(nn.Module):
    """Lightweight self-attention probe over spatial encoder features."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_features: int,
        initializer: InitializerName = "xavier_uniform",
        num_tokens: int = 36,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError("AttentiveProbe in_channels must be divisible by num_heads.")
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, in_channels))
        self.attention_norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels),
        )
        self.value = nn.Linear(num_tokens * in_channels, out_features)
        nn.init.normal_(self.position_embedding, std=0.02)
        apply_initializer(self.mlp[0], initializer)
        apply_initializer(self.mlp[2], initializer)
        apply_initializer(self.value, initializer)

    def forward(self, spatial_latent: torch.Tensor) -> torch.Tensor:
        tokens = spatial_latent.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.num_tokens or tokens.shape[2] != self.in_channels:
            raise ValueError(
                "AttentiveProbe expects spatial features with "
                f"{self.in_channels} channels and {self.num_tokens} tokens, got "
                f"{tokens.shape[2]} channels and {tokens.shape[1]} tokens."
            )
        tokens = tokens + self.position_embedding
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return self.value(tokens.reshape(tokens.shape[0], -1))


class SingleQueryAttentiveProbe(nn.Module):
    """Cross-attention probe with one learned query token."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_features: int,
        initializer: InitializerName = "xavier_uniform",
        num_tokens: int = 36,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError("SingleQueryAttentiveProbe in_channels must be divisible by num_heads.")
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, in_channels))
        self.query = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.token_norm = nn.LayerNorm(in_channels)
        self.query_norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.value = nn.Linear(in_channels, out_features)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        apply_initializer(self.value, initializer)

    def forward(self, spatial_latent: torch.Tensor) -> torch.Tensor:
        tokens = spatial_latent.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.num_tokens or tokens.shape[2] != self.in_channels:
            raise ValueError(
                "SingleQueryAttentiveProbe expects spatial features with "
                f"{self.in_channels} channels and {self.num_tokens} tokens, got "
                f"{tokens.shape[2]} channels and {tokens.shape[1]} tokens."
            )
        tokens = tokens + self.position_embedding
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(
            self.query_norm(query),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )
        return self.value(pooled.squeeze(1))


class AttentionWeightedPoolingProbe(nn.Module):
    """Attention-weighted spatial pooling probe."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_features: int,
        initializer: InitializerName = "xavier_uniform",
        num_tokens: int = 36,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, in_channels))
        self.score = nn.Linear(in_channels, 1)
        self.value = nn.Linear(in_channels, out_features)
        nn.init.normal_(self.position_embedding, std=0.02)
        apply_initializer(self.score, initializer)
        apply_initializer(self.value, initializer)

    def forward(self, spatial_latent: torch.Tensor) -> torch.Tensor:
        tokens = spatial_latent.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.num_tokens or tokens.shape[2] != self.in_channels:
            raise ValueError(
                "AttentionWeightedPoolingProbe expects spatial features with "
                f"{self.in_channels} channels and {self.num_tokens} tokens, got "
                f"{tokens.shape[2]} channels and {tokens.shape[1]} tokens."
            )
        tokens = tokens + self.position_embedding
        weights = self.score(tokens).softmax(dim=1)
        pooled = (tokens * weights).sum(dim=1)
        return self.value(pooled)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for a frozen linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        initializer: InitializerName = "xavier_uniform",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / rank
        apply_initializer(self.lora_down, initializer)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


class LoRAConv2d(nn.Module):
    """Low-rank adapter wrapper for a frozen 2D convolution."""

    def __init__(
        self,
        base: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        initializer: InitializerName = "xavier_uniform",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if base.groups != 1:
            raise ValueError("LoRAConv2d only supports groups=1 convolutions.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_down = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / rank
        apply_initializer(self.lora_down, initializer)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


class DINOv2PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DINOv2LayerScale(nn.Module):
    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DINOv2Attention(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(x)


class DINOv2MLP(nn.Module):
    def __init__(self, dim: int = 384, hidden_dim: int = 1536) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DINOv2Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(384)
        self.attn = DINOv2Attention()
        self.ls1 = DINOv2LayerScale()
        self.norm2 = nn.LayerNorm(384)
        self.mlp = DINOv2MLP()
        self.ls2 = DINOv2LayerScale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class LeWMPatchEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 192, kernel_size=14, stride=14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class LeWMEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 257, 192))
        self.patch_embeddings = LeWMPatchEmbeddings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = self.patch_embeddings(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self._position_embedding(height, width).to(dtype=x.dtype, device=x.device)
        return x

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        patch_h = height // 14
        patch_w = width // 14
        cls_pos = self.position_embeddings[:, :1]
        patch_pos = self.position_embeddings[:, 1:]
        source_size = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, source_size, source_size, 192).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(patch_h, patch_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, patch_h * patch_w, 192)
        return torch.cat([cls_pos, patch_pos], dim=1)


class LeWMSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(192, 192)
        self.key = nn.Linear(192, 192)
        self.value = nn.Linear(192, 192)
        self.num_heads = 3
        self.head_dim = 64
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        query = self.query(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        return (attention @ value).transpose(1, 2).reshape(batch, tokens, 192)


class LeWMAttentionOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(192, 192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class LeWMAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = LeWMSelfAttention()
        self.output = LeWMAttentionOutput()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(x))


class LeWMIntermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(192, 768)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(x))


class LeWMOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(768, 192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class LeWMBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = LeWMAttention()
        self.intermediate = LeWMIntermediate()
        self.output = LeWMOutput()
        self.layernorm_before = nn.LayerNorm(192, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(192, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layernorm_before(x))
        x = x + self.output(self.intermediate(self.layernorm_after(x)))
        return x


class LeWMEncoderLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([LeWMBlock() for _ in range(12)])


class LeWMViTTiny14Encoder(nn.Module):
    """LeWM PushT ViT-Tiny/14 encoder adapted to Atari frame stacks."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        weights: str | None = "models/lewm_pusht_weights.pt",
        output_block: int = 7,
        output_mode: str = "single_block",
        mix_blocks: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if output_block < 1 or output_block > 12:
            raise ValueError("LeWM output_block must be in [1, 12].")
        if output_mode not in {"single_block", "layer_mix"}:
            raise ValueError("LeWM output_mode must be 'single_block' or 'layer_mix'.")
        self.output_block = output_block
        self.output_mode = output_mode
        self.mix_blocks = _validate_indices(mix_blocks or range(1, 13), minimum=1, maximum=12)
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)
        _init_input_adapter(self.input_adapter, input_channels)
        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.embeddings = LeWMEmbeddings()
        self.encoder = LeWMEncoderLayers()
        self.layernorm = nn.LayerNorm(192, eps=1e-12)
        self.reducer = None
        self.output_channels = 192
        self._load_weights(weights)

    def _load_weights(self, weights: str | None) -> None:
        if weights is None or str(weights).lower() in {"", "none", "false"}:
            return
        state = torch.load(Path(weights), map_location="cpu")
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        incompatible = self.load_state_dict(encoder_state, strict=False)
        unexpected = set(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        allowed_missing = {
            "input_adapter.weight",
            "input_adapter.bias",
            "input_mean",
            "input_std",
            "reducer.weight",
            "reducer.bias",
        }
        if unexpected or missing - allowed_missing:
            raise RuntimeError(
                "Unexpected LeWM checkpoint mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = self.input_adapter(x)
        x = (x - self.input_mean) / self.input_std
        x = self.embeddings(x)
        if self.output_mode == "layer_mix":
            outputs = []
            mix_blocks = set(self.mix_blocks)
            for block_index, block in enumerate(self.encoder.layer, start=1):
                x = block(x)
                if block_index in mix_blocks:
                    outputs.append(self._patch_tokens(x, height, width))
            return outputs

        for block in self.encoder.layer[: self.output_block]:
            x = block(x)
        return self._patch_tokens(x, height, width)

    def _patch_tokens(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patch_sequence = self.layernorm(x)[:, 1:]
        patch_h = height // 14
        patch_w = width // 14
        return patch_sequence.transpose(1, 2).reshape(x.shape[0], 192, patch_h, patch_w)


class DINOv2ViTS14Encoder(nn.Module):
    """DINOv2 ViT-S/14 adapted to Atari frame stacks via a 4->3 input adapter."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        weights: str | None = "models/dinov2_vits14_pretrain.pth",
        output_block: int = 3,
        output_mode: str = "single_block",
        mix_blocks: Sequence[int] | None = None,
        initializer: InitializerName = "xavier_uniform",
    ) -> None:
        super().__init__()
        if output_block < 1 or output_block > 12:
            raise ValueError("DINOv2 output_block must be in [1, 12].")
        if output_mode not in {"single_block", "layer_mix"}:
            raise ValueError("DINOv2 output_mode must be 'single_block' or 'layer_mix'.")
        self.output_block = output_block
        self.output_mode = output_mode
        self.mix_blocks = _validate_indices(mix_blocks or range(1, 13), minimum=1, maximum=12)
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)
        _init_input_adapter(self.input_adapter, input_channels)
        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.patch_embed = DINOv2PatchEmbed()
        self.blocks = nn.ModuleList([DINOv2Block() for _ in range(12)])
        self.norm = nn.LayerNorm(384)
        self.reducer = None
        self.output_channels = 384
        self._load_weights(weights)

    def _load_weights(self, weights: str | None) -> None:
        if weights is None or str(weights).lower() in {"", "none", "false"}:
            return
        state = torch.load(Path(weights), map_location="cpu")
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = set(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        allowed_missing = {
            "input_adapter.weight",
            "input_adapter.bias",
            "input_mean",
            "input_std",
            "reducer.weight",
            "reducer.bias",
        }
        if unexpected or missing - allowed_missing:
            raise RuntimeError(
                "Unexpected DINOv2 checkpoint mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        patch_h = height // 14
        patch_w = width // 14
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        source_size = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, source_size, source_size, 384).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(patch_h, patch_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, patch_h * patch_w, 384)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = self.input_adapter(x)
        x = (x - self.input_mean) / self.input_std
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self._position_embedding(height, width).to(dtype=x.dtype, device=x.device)
        if self.output_mode == "layer_mix":
            outputs = []
            mix_blocks = set(self.mix_blocks)
            for block_index, block in enumerate(self.blocks, start=1):
                x = block(x)
                if block_index in mix_blocks:
                    outputs.append(self._patch_tokens(x, height, width))
            return outputs

        for block in self.blocks[: self.output_block]:
            x = block(x)
        return self._patch_tokens(x, height, width)

    def _patch_tokens(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patch_sequence = self.norm(x)[:, 1:]
        patch_h = height // 14
        patch_w = width // 14
        return patch_sequence.transpose(1, 2).reshape(x.shape[0], 384, patch_h, patch_w)


class ResNet18Encoder(nn.Module):
    """ResNet-18 trunk adapted for stacked Atari grayscale frames."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        weights: str | None = None,
        variant: ResNet18Variant = "resnet_layer3_reduced",
        output_mode: str = "single_layer",
        mix_layers: Sequence[int] | None = None,
        initializer: InitializerName = "xavier_uniform",
    ) -> None:
        super().__init__()
        if output_mode not in {"single_layer", "layer_mix"}:
            raise ValueError("ResNet18 output_mode must be 'single_layer' or 'layer_mix'.")
        resolved_weights = _resolve_resnet18_weights(weights)
        backbone = resnet18(weights=resolved_weights)
        if resolved_weights is None:
            mean = torch.zeros(backbone.conv1.in_channels)
            std = torch.ones(backbone.conv1.in_channels)
        else:
            transforms = resolved_weights.transforms()
            mean = torch.as_tensor(transforms.mean)
            std = torch.as_tensor(transforms.std)

        self.input_adapter = _make_input_adapter(input_channels, backbone.conv1.in_channels)
        self.register_buffer("input_mean", mean.view(1, backbone.conv1.in_channels, 1, 1))
        self.register_buffer("input_std", std.view(1, backbone.conv1.in_channels, 1, 1))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.output_mode = output_mode
        self.mix_layers = _validate_indices(mix_layers or range(1, 5), minimum=1, maximum=4)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if variant == "resnet_full":
            self.layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
            self.reducer = None
            self.output_channels = 512
        elif variant in {"resnet_layer1", "resnet_layer1_reduced"}:
            self.layers = nn.Sequential(self.layer1)
            self.reducer = None
            self.output_channels = 64
        elif variant in {"resnet_layer2", "resnet_layer2_reduced"}:
            self.layers = nn.Sequential(self.layer1, self.layer2)
            self.reducer = None
            self.output_channels = 128
        elif variant in {"resnet_layer3", "resnet_layer3_flattened", "resnet_layer3_reduced"}:
            self.layers = nn.Sequential(self.layer1, self.layer2, self.layer3)
            self.reducer = None
            self.output_channels = 256
        elif variant in {"resnet_layer4", "resnet_layer4_reduced"}:
            self.layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
            self.reducer = None
            self.output_channels = 512
        else:
            raise ValueError(f"Unsupported resnet18 variant {variant!r}.")
        self.variant = variant
        self._freeze_batch_norm = False
        if self.reducer is not None:
            apply_initializer(self.reducer, initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)
        x = (x - self.input_mean) / self.input_std
        if self.output_mode == "layer_mix":
            x = self.stem(x)
            outputs = []
            selected = set(self.mix_layers)
            for layer_index, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4), start=1):
                x = layer(x)
                if layer_index in selected:
                    outputs.append(x)
            return outputs
        x = self.layers(self.stem(x))
        if self.reducer is not None:
            x = self.reducer(x)
        return x

    def train(self, mode: bool = True) -> "ResNet18Encoder":
        super().train(mode)
        if self._freeze_batch_norm:
            _set_batch_norm_eval(self)
        return self


def apply_lora_adapters(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    initializer: InitializerName = "xavier_uniform",
    excluded_child_names: frozenset[str] = frozenset(),
) -> int:
    """Recursively replace Linear/Conv2d children with frozen LoRA wrappers.

    ``excluded_child_names`` provides an opt-in escape hatch for architectural
    adapters which should remain ordinary, fully trainable layers.  The empty
    default preserves the legacy transfer-learning behaviour.
    """

    replacements = 0
    for name, child in list(module.named_children()):
        if name in excluded_child_names:
            continue
        if isinstance(child, (LoRALinear, LoRAConv2d)):
            continue
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout, initializer=initializer),
            )
            replacements += 1
        elif isinstance(child, nn.Conv2d):
            setattr(
                module,
                name,
                LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout, initializer=initializer),
            )
            replacements += 1
        else:
            replacements += apply_lora_adapters(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                initializer=initializer,
                excluded_child_names=excluded_child_names,
            )
    return replacements


def configure_encoder_transfer(
    encoder: nn.Module,
    *,
    transfer_mode: str,
    freeze_encoder_bn: bool,
    lora_rank: int = 1,
    lora_alpha: float = 2.0,
    lora_dropout: float = 0.0,
    train_input_adapter_without_lora: bool = False,
) -> None:
    if transfer_mode not in {"none", "full_finetune", "linear_probe", "attentive_probe", "lora"}:
        raise ValueError(f"Unsupported transfer_mode={transfer_mode!r}")
    if transfer_mode == "lora":
        replacements = apply_lora_adapters(
            encoder,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            excluded_child_names=(
                frozenset({"input_adapter"})
                if train_input_adapter_without_lora
                else frozenset()
            ),
        )
        if replacements == 0:
            raise ValueError("LoRA transfer mode found no encoder Linear or Conv2d layers.")
        for name, parameter in encoder.named_parameters():
            parameter.requires_grad = ".lora_" in name or name.startswith("lora_")
        if train_input_adapter_without_lora:
            for parameter in encoder.input_adapter.parameters():
                parameter.requires_grad = True
    if transfer_mode in {"linear_probe", "attentive_probe"}:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        for parameter in encoder.input_adapter.parameters():
            parameter.requires_grad = True
        if getattr(encoder, "reducer", None) is not None:
            for parameter in encoder.reducer.parameters():
                parameter.requires_grad = True
    if freeze_encoder_bn:
        encoder._freeze_batch_norm = True
        freeze_batch_norm(encoder)


def freeze_batch_norm(module: nn.Module) -> None:
    _set_batch_norm_eval(module)
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            for parameter in child.parameters():
                parameter.requires_grad = False


def _set_batch_norm_eval(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def _resolve_resnet18_weights(weights: str | None) -> ResNet18_Weights | None:
    if weights is None or str(weights).lower() in {"", "none", "false"}:
        return None
    if str(weights).lower() in {"default", "imagenet", "imagenet1k"}:
        return ResNet18_Weights.DEFAULT
    return ResNet18_Weights[weights]


def _make_input_adapter(input_channels: int, output_channels: int) -> nn.Conv2d:
    adapter = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    _init_input_adapter(adapter, input_channels)
    return adapter


def _validate_indices(indices: Sequence[int], *, minimum: int, maximum: int) -> tuple[int, ...]:
    values = tuple(int(index) for index in indices)
    if not values:
        raise ValueError("Layer mix index list must not be empty.")
    invalid = [index for index in values if index < minimum or index > maximum]
    if invalid:
        raise ValueError(f"Layer mix indices must be in [{minimum}, {maximum}], got {invalid}.")
    return values


def _init_input_adapter(adapter: nn.Conv2d, input_channels: int) -> None:
    with torch.no_grad():
        adapter.weight.fill_(1.0 / input_channels)
        if adapter.bias is not None:
            adapter.bias.zero_()
