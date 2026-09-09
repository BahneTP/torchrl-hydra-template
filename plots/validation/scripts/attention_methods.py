#!/usr/bin/env python3
"""Compare the selected learning rate for three attentive probe variants."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = VALIDATION_DIR.parent
PLAIN_RESULTS = PLOTS_DIR.parent / (
    "logs/plain_algorithms/der/atari100k_20260810_181403/results.csv"
)
sys.path.insert(0, str(PLOTS_DIR))
from boxplots_vertical import create_vertical_boxplots  # noqa: E402
from atari_scores import get_atari_reference_scores  # noqa: E402


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

    plain = pd.read_csv(PLAIN_RESULTS)
    mean_reward = plain[
        (plain["algorithm"] == "der")
        & (plain["game"].str.casefold() == "jamesbond")
        & (plain["exit_code"] == 0)
    ]["final_return_mean"].mean()
    random_reward, human_reward = get_atari_reference_scores("Jamesbond")

    output = create_vertical_boxplots(
        labels,
        values,
        VALIDATION_DIR / "attention_methods.pdf",
        xlabel="Attention Mechanism",
        mean_reward=float(mean_reward),
        random_reward=random_reward,
        human_reward=human_reward,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
