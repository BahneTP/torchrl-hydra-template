#!/usr/bin/env python3
"""Compare the selected learning rate for three attentive probe variants."""

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
            "Single Query",
            Path(
                "/home/bthiehl/torchrl-hydra-template/logs/resnetlayer/attentive_lrsweep/"
                "der_resnet18_attentive_single_query_layer2_probe_lr_sweep_"
                "jamesbond_20260812_130752/results.csv"
            ),
            "resnet18_attentive_single_query_layer2_probe_lr_1e_4",
        ),
        (
            "Self-Attention",
            Path(
                "/home/bthiehl/torchrl-hydra-template/logs/resnetlayer/attentive_lrsweep/"
                "der_resnet18_attentive_self_attention_layer2_probe_lr_sweep_"
                "jamesbond_20260812_130734/results.csv"
            ),
            "resnet18_attentive_self_attention_layer2_probe_lr_1e_6",
        ),
        (
            "Attention-Weighted Pooling",
            Path(
                "/home/bthiehl/torchrl-hydra-template/logs/resnetlayer/attentive_lrsweep/"
                "der_resnet18_attentive_attention_weighted_pooling_layer2_probe_lr_"
                "sweep_jamesbond_20260812_131048/results.csv"
            ),
            "resnet18_attentive_attention_weighted_pooling_layer2_probe_lr_1e_4",
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
        PLOTS_DIR / "attention_methods.pdf",
        xlabel="Attention Mechanism",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
