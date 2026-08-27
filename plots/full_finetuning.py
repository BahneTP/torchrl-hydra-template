#!/usr/bin/env python3
"""Compare selected full-finetuning configurations on James Bond."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOTS_DIR))
from boxplots_vertical import create_vertical_boxplots  # noqa: E402


def main() -> None:
    selections = [
        (
            "ResNet-18\nLayer 2",
            Path(
                "/home/bthiehl/torchrl-hydra-template/logs/fullfinetuning/"
                "der_resnet18_layer2_20260811_025944/results.csv"
            ),
            "resnet18_full_layer2_lr_1e_6",
        ),
        (
            "ResNet-18\nLayer 4",
            Path(
                "/home/bthiehl/torchrl-hydra-template/logs/fullfinetuning/"
                "der_resnet18_layer4_20260809_001602/results.csv"
            ),
            "resnet18_full_layer4_lr_1e_8",
        ),
        (
            "DINOv2\nBlock 7",
            Path(
                "/home/bthiehl/torchrl-hydra-template/logs/fullfinetuning/"
                "der_dinov2_block7_20260814_191820/results.csv"
            ),
            "dinov2_full_block7_lr_1e_3",
        ),
        (
            "DINOv2\nBlock 12",
            Path(
                "/home/bthiehl/torchrl-hydra-template/logs/fullfinetuning/"
                "der_dinov2_block12_20260809_001637/results.csv"
            ),
            "dinov2_full_block12_lr_1e_4",
        ),
    ]

    labels: list[str] = []
    values: list[list[float]] = []
    for label, results, variant in selections:
        df = pd.read_csv(results)
        selected = df[
            (df["algorithm"] == "der")
            & (df["game"].str.casefold() == "jamesbond")
            & (df["variant"] == variant)
            & (df["exit_code"] == 0)
            & df["final_return_mean"].notna()
        ].sort_values("seed")

        if len(selected) != 5 or selected["seed"].nunique() != 5:
            raise SystemExit(
                f"expected five successful, unique seeds for {variant}, "
                f"found {len(selected)} rows and {selected['seed'].nunique()} seeds"
            )
        labels.append(label)
        values.append(selected["final_return_mean"].astype(float).tolist())

    output = create_vertical_boxplots(
        labels,
        values,
        PLOTS_DIR / "full_finetuning.pdf",
        xlabel="Encoder Depth",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
