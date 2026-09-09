#!/usr/bin/env python3
"""Plot final seed distributions for three games, encoders, and methods."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent
PLOTS_DIR = TEST_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
from atari_scores import get_atari_reference_scores  # noqa: E402
from boxplots_vertical_triple import create_vertical_boxplots_triple  # noqa: E402
from encoder_learning_curves import (  # noqa: E402
    ENCODERS,
    GAMES,
    METHODS,
    PLAIN_DER,
    seeds_for,
)

TRANSFER_METHODS = METHODS[:-1]
DISPLAY_METHODS = (
    "Full Fine-Tuning",
    "Attentive Probing",
    "Linear Probing",
    "Linear MLF",
    "Attentive MLF",
    "LoRA",
)


def final_values(run_directory: Path, game: str) -> list[float]:
    seeds = seeds_for(run_directory, game)
    results = pd.read_csv(run_directory / "results.csv")
    selected = results[
        (results["algorithm"] == "der")
        & (results["game"].str.casefold() == game.casefold())
        & (results["exit_code"] == 0)
        & results["final_return_mean"].notna()
        & results["seed"].isin(seeds)
    ].sort_values("seed")
    if len(selected) != len(seeds) or selected["seed"].nunique() != len(seeds):
        raise SystemExit(
            f"expected {len(seeds)} successful, unique values for "
            f"{run_directory.name} on {game}, found {len(selected)} rows and "
            f"{selected['seed'].nunique()} seeds"
        )
    return selected["final_return_mean"].astype(float).tolist()


def main() -> None:
    values: list[list[list[list[float]]]] = []
    baselines: list[list[float]] = []
    mean_rewards: list[float] = []
    random_rewards: list[float] = []
    human_rewards: list[float] = []

    for game in GAMES:
        game_values: list[list[list[float]]] = []
        for method_index in range(len(TRANSFER_METHODS)):
            method_values = [
                final_values(run_directories[method_index], game)
                for run_directories in ENCODERS.values()
            ]
            game_values.append(method_values)
        random_reward, human_reward = get_atari_reference_scores(game)
        baseline = final_values(PLAIN_DER, game)
        values.append(game_values)
        baselines.append(baseline)
        mean_rewards.append(float(pd.Series(baseline).mean()))
        random_rewards.append(random_reward)
        human_rewards.append(human_reward)

    output = create_vertical_boxplots_triple(
        GAMES,
        DISPLAY_METHODS,
        list(ENCODERS),
        values,
        baselines,
        TEST_DIR / "encoder_boxplots_triple.pdf",
        mean_rewards=mean_rewards,
        random_rewards=random_rewards,
        human_rewards=human_rewards,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
