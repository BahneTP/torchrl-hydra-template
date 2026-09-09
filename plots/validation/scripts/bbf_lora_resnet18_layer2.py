#!/usr/bin/env python3
"""Plot the BBF ResNet-18 layer-2 LoRA rank/alpha sweep (bar chart)."""

from pathlib import Path
import sys

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = VALIDATION_DIR.parent
REPO_ROOT = PLOTS_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
from atari_scores import get_atari_reference_scores  # noqa: E402
from bars_vertical import create_vertical_bars  # noqa: E402

RESULTS = REPO_ROOT / (
    "logs/bbf-testing/lora/"
    "lora_rank_alpha_sweep_all_lr_1e_4_with_shrink_perturb_jamesbond_seed1_"
    "20260825_185809/results.csv"
)
R4_A8_RESULTS = REPO_ROOT / (
    "logs/bbf-testing/lora/"
    "lora_lr_sweep_with_shrink_perturb_jamesbond_seed1_20260824_151147/"
    "results.csv"
)
PLAIN_RESULTS = REPO_ROOT / (
    "logs/plain_algorithms/bbf/atari100k_jamesbond_5seeds_20260906_143047/"
    "results.csv"
)
COMBINATIONS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]


def main() -> None:
    data = pd.read_csv(RESULTS)
    r4_a8_data = pd.read_csv(R4_A8_RESULTS)

    labels, values = [], []
    for rank, alpha in COMBINATIONS:
        if (rank, alpha) == (4, 8):
            variant = "resnet18_layer2_lora_lr_1e_8_with_shrink_perturb"
            source = r4_a8_data
        else:
            variant = (
                f"resnet18_layer2_lora_r{rank}_a{alpha}_all_lr_1e_4_"
                "with_shrink_perturb"
            )
            source = data
        selected = source[
            (source["variant"] == variant)
            & (source["game"].str.casefold() == "jamesbond")
            & (source["exit_code"] == 0)
            & source["final_return_mean"].notna()
        ].sort_values("seed")
        if len(selected) != 1 or selected["seed"].nunique() != 1:
            raise SystemExit(f"expected one successful seed for {variant}")
        labels.append(f"{rank}/{alpha}")
        values.append(selected["final_return_mean"].astype(float).tolist())

    plain = pd.read_csv(PLAIN_RESULTS)
    baseline = plain[
        (plain["algorithm"] == "bbf")
        & (plain["game"].str.casefold() == "jamesbond")
        & (plain["exit_code"] == 0)
    ]["final_return_mean"].mean()
    random_reward, human_reward = get_atari_reference_scores("Jamesbond")
    output = create_vertical_bars(
        labels,
        values,
        VALIDATION_DIR / "bbf_lora_resnet18_layer2_rank_alpha.pdf",
        xlabel="Rank/Alpha",
        mean_reward=float(baseline),
        random_reward=random_reward,
        human_reward=human_reward,
        figure_height=2.2,
        ymin=1,
        bar_color="#CCE5FF",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
