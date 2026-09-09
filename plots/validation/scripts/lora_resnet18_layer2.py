#!/usr/bin/env python3
"""Plot the ResNet-18 layer-2 LoRA rank/alpha sweep."""

from pathlib import Path
import sys

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = VALIDATION_DIR.parent
REPO_ROOT = PLOTS_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
from atari_scores import get_atari_reference_scores  # noqa: E402
from boxplots_vertical import create_vertical_boxplots  # noqa: E402

RESULTS = REPO_ROOT / (
    "logs/lora/der_resnet18_layer2_lora_rank_alpha_sweep_jamesbond_"
    "20260817_134400/results.csv"
)
PLAIN_RESULTS = REPO_ROOT / (
    "logs/plain_algorithms/der/atari100k_20260810_181403/results.csv"
)
COMBINATIONS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]


def main() -> None:
    data = pd.read_csv(RESULTS)
    labels, values = [], []
    for rank, alpha in COMBINATIONS:
        variant = f"resnet18_lora_layer2_r{rank}_a{alpha}"
        selected = data[
            (data["variant"] == variant)
            & (data["game"].str.casefold() == "jamesbond")
            & (data["exit_code"] == 0)
            & data["final_return_mean"].notna()
        ].sort_values("seed")
        if len(selected) != 5 or selected["seed"].nunique() != 5:
            raise SystemExit(f"expected five successful seeds for {variant}")
        labels.append(f"{rank}/{alpha}")
        values.append(selected["final_return_mean"].astype(float).tolist())

    plain = pd.read_csv(PLAIN_RESULTS)
    baseline = plain[
        (plain["algorithm"] == "der")
        & (plain["game"].str.casefold() == "jamesbond")
        & (plain["exit_code"] == 0)
    ]["final_return_mean"].mean()
    random_reward, human_reward = get_atari_reference_scores("Jamesbond")
    output = create_vertical_boxplots(
        labels,
        values,
        VALIDATION_DIR / "lora_resnet18_layer2_rank_alpha.pdf",
        xlabel="Rank/Alpha",
        mean_reward=float(baseline),
        random_reward=random_reward,
        human_reward=human_reward,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
