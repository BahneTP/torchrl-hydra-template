from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.algorithms.rainbow.rainbow import _batch_normalized_priority_loss


def test_priority_weights_are_normalized_by_batch_maximum() -> None:
    loss = torch.tensor([1.0, 2.0])
    priority_weight = torch.tensor([0.2, 0.4])

    weighted_loss = _batch_normalized_priority_loss(loss, priority_weight)

    torch.testing.assert_close(weighted_loss, torch.tensor(1.25))


def test_der_uses_fixed_per_beta() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path("configs").resolve()),
    ):
        cfg = compose(config_name="train", overrides=["experiment=der/atari100k"])

    assert cfg.algorithm.prb_alpha == 0.5
    assert cfg.algorithm.prb_beta == 0.5
    assert "prb_beta_start" not in cfg.algorithm
    assert "prb_beta_end" not in cfg.algorithm
    assert "prb_beta_frames" not in cfg.algorithm
