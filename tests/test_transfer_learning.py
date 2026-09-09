"""Transfer-learning configuration contracts shared by DER and BBF."""

from __future__ import annotations

import torch
from torch import nn

from src.algorithms.bbf.bbf import BBFAlgorithm
from src.algorithms.bbf.networks import BBFResNet18TransferEncoder
from src.components.transfer_learning import (
    LoRAConv2d,
    LoRALinear,
    configure_encoder_transfer,
)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_adapter = nn.Conv2d(4, 3, kernel_size=1)
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3), nn.Flatten(), nn.Linear(72, 4))


def test_lora_legacy_path_still_wraps_input_adapter():
    encoder = _TinyEncoder()

    configure_encoder_transfer(
        encoder,
        transfer_mode="lora",
        freeze_encoder_bn=False,
        lora_rank=2,
        lora_alpha=4.0,
    )

    assert isinstance(encoder.input_adapter, LoRAConv2d)
    assert isinstance(encoder.backbone[0], LoRAConv2d)
    assert isinstance(encoder.backbone[2], LoRALinear)


def test_bbf_lora_path_keeps_input_adapter_dense_and_trainable():
    encoder = _TinyEncoder()

    configure_encoder_transfer(
        encoder,
        transfer_mode="lora",
        freeze_encoder_bn=False,
        lora_rank=2,
        lora_alpha=4.0,
        train_input_adapter_without_lora=True,
    )

    assert type(encoder.input_adapter) is nn.Conv2d
    assert all(parameter.requires_grad for parameter in encoder.input_adapter.parameters())
    assert isinstance(encoder.backbone[0], LoRAConv2d)
    assert isinstance(encoder.backbone[2], LoRALinear)


def test_bbf_optimizer_assigns_dense_input_adapter_its_own_lr():
    algorithm = BBFAlgorithm(device=torch.device("cpu"), adapter_lr=1.0e-4, encoder_lr=1.0e-7)
    network = nn.Module()
    network.encoder = nn.Module()
    network.encoder.input_adapter = nn.Conv2d(4, 3, kernel_size=1)
    network.encoder.backbone = LoRAConv2d(
        nn.Conv2d(3, 8, kernel_size=3), rank=2, alpha=4.0, dropout=0.0
    )
    network.head = nn.Linear(8, 2)
    algorithm.network = network

    optimizer = algorithm._make_optimizer()
    adapter_parameter_ids = {id(parameter) for parameter in network.encoder.input_adapter.parameters()}
    adapter_groups = [
        group
        for group in optimizer.param_groups
        if any(id(parameter) in adapter_parameter_ids for parameter in group["params"])
    ]

    assert len(adapter_groups) == 2  # weight decay and no-decay (bias)
    assert all(group["lr"] == 1.0e-4 for group in adapter_groups)


def test_bbf_linear_projection_uses_probe_lr():
    algorithm = BBFAlgorithm(
        device=torch.device("cpu"), transfer_mode="linear_probe", probe_lr=1.0e-6
    )
    algorithm.network = nn.Module()
    algorithm.network.projection = nn.Linear(8, 4)
    algorithm.network.head = nn.Linear(4, 2)

    optimizer = algorithm._make_optimizer()
    projection_ids = {id(parameter) for parameter in algorithm.network.projection.parameters()}
    projection_groups = [
        group
        for group in optimizer.param_groups
        if any(id(parameter) in projection_ids for parameter in group["params"])
    ]

    assert all(group["lr"] == 1.0e-6 for group in projection_groups)


def test_bbf_attentive_layer_mix_has_one_probe_per_layer():
    encoder = BBFResNet18TransferEncoder(
        input_channels=4,
        weights=None,
        transfer_mode="attentive_probe",
        freeze_encoder_bn=True,
        lora_rank=1,
        lora_alpha=2.0,
        lora_dropout=0.0,
        transfer_layer_mix=True,
        linear_probe_conv=False,
        mix_layers=(1, 2, 3, 4),
        attentive_probe_type="self_attention",
    )

    assert encoder.spatial_probes is not None
    assert len(encoder.spatial_probes) == 4
    assert len({id(probe) for probe in encoder.spatial_probes}) == 4
    assert encoder(torch.randn(2, 4, 84, 84)).shape == (2, 128, 11, 11)


def test_bbf_attentive_layer_mix_probes_use_probe_lr():
    algorithm = BBFAlgorithm(
        device=torch.device("cpu"), transfer_mode="attentive_probe", probe_lr=1.0e-6
    )
    network = nn.Module()
    network.encoder = nn.Module()
    network.encoder.spatial_probes = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
    network.head = nn.Linear(8, 2)
    algorithm.network = network

    optimizer = algorithm._make_optimizer()
    probe_parameter_ids = {
        id(parameter) for parameter in network.encoder.spatial_probes.parameters()
    }
    probe_groups = [
        group
        for group in optimizer.param_groups
        if any(id(parameter) in probe_parameter_ids for parameter in group["params"])
    ]

    assert probe_groups
    assert all(group["lr"] == 1.0e-6 for group in probe_groups)
