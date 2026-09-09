#!/usr/bin/env python3
"""Plot final BBF seed distributions for one ResNet-18 encoder row."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent
PLOTS_DIR = TEST_DIR.parent
REPO_ROOT = PLOTS_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
from atari_scores import get_atari_reference_scores  # noqa: E402
from boxplots_vertical_triple import create_vertical_boxplots_triple  # noqa: E402

GAMES = ("Assault", "BankHeist", "RoadRunner")
METHODS = (
    "Full Fine-Tuning",
    "Attentive Probing",
    "Linear Probing",
    "Linear MLF",
    "Attentive MLF",
    "LoRA",
)
PLAIN_BBF = {
    "Assault": REPO_ROOT
    / "logs/plain_algorithms/bbf/atari100k_assault_5seeds_20260906_143104",
    "BankHeist": REPO_ROOT
    / "logs/plain_algorithms/bbf/atari100k_bankheist_5seeds_20260906_143349",
    "RoadRunner": REPO_ROOT
    / "logs/plain_algorithms/bbf/atari100k_roadrunner_5seeds_20260906_143410",
}
RUNS = {
    "Full Fine-Tuning": REPO_ROOT
    / "logs/bbf_tl_final/full_3games_5seeds_20260827_232107",
    "Attentive Probing": REPO_ROOT
    / "logs/bbf_tl_final/attentive_3games_5seeds_20260828_131657",
    "Linear Probing": REPO_ROOT
    / "logs/bbf_tl_final/linear_3games_5seeds_20260827_232221",
    "Linear MLF": REPO_ROOT
    / "logs/bbf_tl_final/bbf_resnet18_mlf_linear_all_3games_20260821_234445",
    "LoRA": REPO_ROOT
    / "logs/bbf_tl_final/lora_r1_a2_3games_5seeds_20260828_132402",
}
ATTENTIVE_MLF = {
    "Assault": REPO_ROOT
    / "logs/bbf_tl_final/mlf_attentive_lr_1e_4_assault_20260831_122144",
    "BankHeist": REPO_ROOT
    / "logs/bbf_tl_final/mlf_attentive_lr_1e_4_bankheist_20260831_122403",
    "RoadRunner": REPO_ROOT
    / "logs/bbf_tl_final/mlf_attentive_lr_1e_4_roadrunner_20260831_122904",
}


def seeds_for(method: str, game: str) -> tuple[int, ...]:
    if method == "Linear MLF" and game == "RoadRunner":
        return (1, 2, 3, 4)
    return (1, 2, 3, 4, 5)


def final_values(run_directory: Path, game: str, seeds: tuple[int, ...]) -> list[float]:
    results = pd.read_csv(run_directory / "results.csv")
    selected = results[
        (results["algorithm"] == "bbf")
        & (results["game"].str.casefold() == game.casefold())
        & (results["exit_code"] == 0)
        & results["final_return_mean"].notna()
        & results["seed"].isin(seeds)
    ].sort_values("seed")
    if len(selected) != len(seeds) or selected["seed"].nunique() != len(seeds):
        raise SystemExit(
            f"expected {len(seeds)} successful seeds for "
            f"{run_directory.name} on {game}"
        )
    return selected["final_return_mean"].astype(float).tolist()


def main() -> None:
    values: list[list[list[list[float]]]] = []
    baselines: list[list[float]] = []
    mean_rewards: list[float] = []
    random_rewards: list[float] = []
    human_rewards: list[float] = []

    for game in GAMES:
        game_values = []
        for method in METHODS:
            run_directory = ATTENTIVE_MLF[game] if method == "Attentive MLF" else RUNS[method]
            game_values.append(
                [final_values(run_directory, game, seeds_for(method, game))]
            )
        baseline = final_values(PLAIN_BBF[game], game, seeds_for("BBF", game))
        random_reward, human_reward = get_atari_reference_scores(game)
        values.append(game_values)
        baselines.append(baseline)
        mean_rewards.append(float(pd.Series(baseline).mean()))
        random_rewards.append(random_reward)
        human_rewards.append(human_reward)

    output = create_vertical_boxplots_triple(
        GAMES,
        METHODS,
        ("ResNet-18",),
        values,
        baselines,
        TEST_DIR / "bbf_encoder_boxplots.pdf",
        mean_rewards=mean_rewards,
        random_rewards=random_rewards,
        human_rewards=human_rewards,
        baseline_label="BBF",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
