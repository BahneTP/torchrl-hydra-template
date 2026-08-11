from __future__ import annotations

import math

import torch

from src.algorithms.rainbow.rainbow import GoogleDERNoisyLinear


def test_google_der_noisy_linear_initialization() -> None:
    in_features = 16
    out_features = 8
    std_init = 0.5
    seed = 7

    torch.manual_seed(seed)
    expected_weight_mu = torch.empty(out_features, in_features)
    torch.nn.init.xavier_uniform_(expected_weight_mu)
    expected_bias_mu = torch.empty(out_features)
    torch.nn.init.uniform_(
        expected_bias_mu,
        -1 / math.sqrt(in_features),
        1 / math.sqrt(in_features),
    )

    torch.manual_seed(seed)
    layer = GoogleDERNoisyLinear(in_features, out_features, std_init=std_init)

    expected_sigma = std_init / math.sqrt(in_features)
    torch.testing.assert_close(layer.weight_mu, expected_weight_mu)
    torch.testing.assert_close(layer.bias_mu, expected_bias_mu)
    torch.testing.assert_close(
        layer.weight_sigma,
        torch.full_like(layer.weight_sigma, expected_sigma),
    )
    torch.testing.assert_close(
        layer.bias_sigma,
        torch.full_like(layer.bias_sigma, expected_sigma),
    )
