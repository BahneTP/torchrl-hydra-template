#!/usr/bin/env python3
"""Plot BBF transfer-learning curves averaged across three Atari games."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent
PLOTS_DIR = TEST_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from atari_scores import get_atari_reference_scores  # noqa: E402
from bbf_encoder_boxplots import (  # noqa: E402
    ATTENTIVE_MLF,
    GAMES,
    METHODS,
    PLAIN_BBF,
    RUNS,
    final_values,
    seeds_for,
)
from learning_curves import create_learning_curves  # noqa: E402
from wandb_history import load_evaluation_curve  # noqa: E402

LABELS = (
    "BBF",
    "LoRA",
    "Full Fine-Tuning",
    "Attentive MLF",
    "Linear Probing",
    "Linear MLF",
    "Attentive Probing",
)
EVALUATION_STEPS = tuple(range(10_000, 100_001, 10_000))


def run_for(method: str, game: str) -> Path:
    if method == "BBF":
        return PLAIN_BBF[game]
    if method == "Attentive MLF":
        return ATTENTIVE_MLF[game]
    return RUNS[method]


def load_curve(
    method: str, game: str, *, refresh: bool
) -> tuple[list[int], list[float], list[float]]:
    run_directory = run_for(method, game)
    seeds = seeds_for(method, game)
    steps, mean, std = load_evaluation_curve(
        run_directory,
        game,
        seeds=seeds,
        refresh=refresh,
        evaluation_steps=EVALUATION_STEPS,
    )
    try:
        final_index = steps.index(100_000)
    except ValueError as error:
        raise SystemExit(
            f"curve {run_directory.name} on {game} has no point at 100K"
        ) from error
    final = np.asarray(final_values(run_directory, game, seeds), dtype=float)
    mean[final_index] = float(final.mean())
    std[final_index] = (
        float(final.std(ddof=1)) if final.size > 1 else 0.0
    )
    return steps, mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download W&B histories again instead of using cached CSV files",
    )
    args = parser.parse_args()

    references = {
        game: get_atari_reference_scores(game)
        for game in GAMES
    }
    common_steps: list[int] | None = None
    means: list[list[float]] = []
    stds: list[list[float]] = []

    for method in LABELS:
        game_means: list[np.ndarray] = []
        game_stds: list[np.ndarray] = []
        for game in GAMES:
            steps, raw_mean, raw_std = load_curve(
                method, game, refresh=args.refresh
            )
            if common_steps is None:
                common_steps = steps
            elif steps != common_steps:
                raise SystemExit(
                    f"evaluation steps differ for {method} on {game}"
                )
            random_reward, human_reward = references[game]
            scale = human_reward - random_reward
            game_means.append((np.asarray(raw_mean) - random_reward) / scale)
            game_stds.append(np.asarray(raw_std) / abs(scale))

        # Normalize within each game before averaging, matching the Mean HNS
        # calculation in the final column of bbf_encoder_boxplots.pdf.
        means.append(np.mean(game_means, axis=0).tolist())
        stds.append(
            (np.sqrt(np.sum(np.square(game_stds), axis=0)) / len(GAMES)).tolist()
        )

    if common_steps is None:
        raise SystemExit("no evaluation curves were loaded")

    baseline_game_hns: list[float] = []
    for game in GAMES:
        baseline = np.asarray(
            final_values(PLAIN_BBF[game], game, seeds_for("BBF", game)),
            dtype=float,
        )
        random_reward, human_reward = references[game]
        baseline_game_hns.append(
            (float(baseline.mean()) - random_reward)
            / (human_reward - random_reward)
        )
    baseline_mean_hns = float(np.mean(baseline_game_hns))

    output = create_learning_curves(
        common_steps,
        LABELS,
        means,
        stds,
        TEST_DIR / "bbf_encoder_learning_curves.pdf",
        # Inputs are already Mean HNS. This makes the right axis equal to
        # Mean HNS / Plain-BBF baseline Mean HNS.
        mean_reward=baseline_mean_hns,
        random_reward=0.0,
        human_reward=1.0,
        legend_outside_left=True,
        figure_width=8.5,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
