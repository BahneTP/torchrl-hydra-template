#!/usr/bin/env python3
"""Plot final ResNet-18 mixed-layer weights across James Bond seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLOTS_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
from boxplots_vertical import create_vertical_boxplots  # noqa: E402

DEFAULT_RUNS = REPO_ROOT / (
    "logs/mixedlayerfusion/der_resnet18_mlf_linear_20260808_155703/"
    "all/Jamesbond"
)
DEFAULT_WANDB = REPO_ROOT / "logs/wandb/wandb"
NUM_LAYERS = 4


def load_seed_weights(seed_dir: Path, wandb_root: Path) -> list[float]:
    run_metadata = seed_dir / "checkpoints/wandb_run.json"
    if not run_metadata.is_file():
        raise SystemExit(f"missing W&B run metadata: {run_metadata}")

    run_id = json.loads(run_metadata.read_text())["id"]
    matches = sorted(wandb_root.glob(f"run-*-{run_id}/files/wandb-summary.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected one local W&B summary for run {run_id}, found {len(matches)}"
        )

    summary = json.loads(matches[0].read_text())
    weights = []
    for layer in range(1, NUM_LAYERS + 1):
        key = f"transfer_layer_mix/resnet_layer_{layer:02d}"
        if key not in summary:
            raise SystemExit(f"missing metric {key!r} in {matches[0]}")
        weights.append(float(summary[key]))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="?", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--wandb-root", type=Path, default=DEFAULT_WANDB)
    parser.add_argument(
        "--output", type=Path, default=PLOTS_DIR / "resnet18_mixed_layer_weights.pdf"
    )
    args = parser.parse_args()

    seed_dirs = [args.runs / f"seed_{seed}" for seed in range(1, 6)]
    weights_by_seed = [
        load_seed_weights(seed_dir, args.wandb_root) for seed_dir in seed_dirs
    ]
    weights_by_layer = [
        [seed_weights[layer] for seed_weights in weights_by_seed]
        for layer in range(NUM_LAYERS)
    ]

    output = create_vertical_boxplots(
        [str(layer) for layer in range(1, NUM_LAYERS + 1)],
        weights_by_layer,
        args.output,
        xlabel="ResNet-18 Layer",
        ylabel="Final normalized layer weight",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
