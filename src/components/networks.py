"""Actor-critic network factories with cleanRL-style orthogonal initialisation.

Shared by on-policy actor-critic algorithms (PPO, and reusable by A2C & co).
Each factory takes ``(obs_shape, action_dim)`` positionally and keeps the
rest as keyword-only args, so a Hydra ``_partial_`` config can pre-bind the
kwargs while the algorithm's ``setup()`` supplies the runtime shape and
action count (same convention as ``src/networks.py``).

Initialisation follows cleanRL / "The 37 Implementation Details of PPO":
orthogonal weights with gain sqrt(2) for hidden layers, 0.01 for the policy
output layer, 1.0 for the value output layer; all biases zero.
"""
from __future__ import annotations

import math
from typing import Sequence, Type

import torch
import torch.nn as nn
from tensordict.nn import AddStateIndependentNormalScale
from torchrl.modules import MLP, ConvNet


def orthogonal_init_(
    module: nn.Module,
    *,
    hidden_gain: float = math.sqrt(2),
    final_gain: float | None = None,
    bias_const: float = 0.0,
) -> nn.Module:
    """Orthogonally initialise all Linear/Conv layers of ``module`` in place.

    Every ``nn.Linear`` / ``nn.Conv2d`` gets orthogonal weights with
    ``hidden_gain`` and constant ``bias_const`` biases. If ``final_gain`` is
    given, the *last* such layer (the output head) uses it instead —
    cleanRL uses 0.01 for policy heads and 1.0 for value heads.
    """
    layers = [m for m in module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    for i, layer in enumerate(layers):
        gain = hidden_gain
        if final_gain is not None and i == len(layers) - 1:
            gain = final_gain
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, bias_const)
    return module


def make_normal_mlp_actor(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
    init_log_std: float = 0.0,
) -> nn.Module:
    """MLP actor for a Normal policy with state-independent log-std.

    Returns ``MLP -> AddStateIndependentNormalScale``: the MLP predicts the
    mean (``loc``) only; the scale is a free learnable parameter shared
    across states (cleanRL: ``std = exp(logstd)`` with ``logstd`` init 0).
    The module outputs a ``(loc, scale)`` tuple for ``out_keys=["loc",
    "scale"]``.
    """
    mlp = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=int(action_dim),
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    orthogonal_init_(mlp, final_gain=0.01)
    return nn.Sequential(
        mlp,
        AddStateIndependentNormalScale(int(action_dim), init_value=init_log_std),
    )


def make_mlp_value(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP state-value critic V(s) mapping the flattened observation to a scalar.

    ``action_dim`` is unused — kept for signature parity with the actor factory.
    """
    del action_dim  # signature parity with actor factory
    mlp = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    return orthogonal_init_(mlp, final_gain=1.0)


def make_nature_cnn_trunk(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    out_features: int = 512,
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """Nature-DQN ConvNet -> Linear trunk shared by actor and critic heads.

    Maps ``[C, H, W]`` pixels to ``out_features`` activated features (the
    final linear layer is followed by the activation, as in cleanRL's Atari
    agent). ``action_dim`` is unused — signature parity with head factories.
    """
    del action_dim  # signature parity with head factories
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=list(num_cells_cnn),
        kernel_sizes=list(kernel_sizes),
        strides=list(strides),
    )
    with torch.no_grad():
        cnn_out = cnn(torch.zeros(1, *obs_shape))
    mlp = MLP(
        in_features=cnn_out.shape[-1],
        out_features=out_features,
        num_cells=[],
        activate_last_layer=True,
        activation_class=activation_class,
    )
    return orthogonal_init_(nn.Sequential(cnn, mlp))


def make_categorical_head(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells: Sequence[int] = (),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """Policy head mapping trunk features to action logits (final gain 0.01)."""
    mlp = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=int(num_actions),
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    return orthogonal_init_(mlp, final_gain=0.01)


def make_value_head(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int] = (),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """Value head mapping trunk features to a scalar V(s) (final gain 1.0)."""
    del action_dim  # signature parity with make_categorical_head
    mlp = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    return orthogonal_init_(mlp, final_gain=1.0)
