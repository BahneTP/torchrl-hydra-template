#!/usr/bin/env python3
"""Plot DER transfer-learning curves averaged across three Atari games."""

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
from encoder_learning_curves import (  # noqa: E402
    ENCODERS,
    GAMES,
    METHODS,
    PLAIN_DER,
    final_evaluation,
    load_curve,
    seeds_for,
)
from learning_curves_triple_encoders import (  # noqa: E402
    create_learning_curves_triple_encoders,
)


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
    means: list[list[list[float]]] = []
    stds: list[list[list[float]]] = []

    for encoder, run_directories in ENCODERS.items():
        encoder_means: list[list[float]] = []
        encoder_stds: list[list[float]] = []
        for method, run_directory in zip(METHODS, run_directories):
            game_means: list[np.ndarray] = []
            game_stds: list[np.ndarray] = []
            for game in GAMES:
                steps, raw_mean, raw_std = load_curve(
                    run_directory, game, refresh=args.refresh
                )
                if common_steps is None:
                    common_steps = steps
                elif steps != common_steps:
                    raise SystemExit(
                        f"evaluation steps differ for {encoder}/{method} on {game}"
                    )
                random_reward, human_reward = references[game]
                scale = human_reward - random_reward
                game_means.append((np.asarray(raw_mean) - random_reward) / scale)
                game_stds.append(np.asarray(raw_std) / abs(scale))

            # This matches the final column of encoder_boxplots_triple.pdf:
            # normalize within each game first, then average the three HNS values.
            encoder_means.append(np.mean(game_means, axis=0).tolist())
            # Propagate the independent across-seed uncertainties of the three
            # equally weighted game means.
            encoder_stds.append(
                (np.sqrt(np.sum(np.square(game_stds), axis=0)) / len(GAMES)).tolist()
            )
        means.append(encoder_means)
        stds.append(encoder_stds)

    if common_steps is None:
        raise SystemExit("no evaluation curves were loaded")

    baseline_game_hns: list[float] = []
    for game in GAMES:
        baseline_mean, _ = final_evaluation(
            PLAIN_DER, game, seeds_for(PLAIN_DER, game)
        )
        random_reward, human_reward = references[game]
        baseline_game_hns.append(
            (baseline_mean - random_reward) / (human_reward - random_reward)
        )
    baseline_mean_hns = float(np.mean(baseline_game_hns))

    output = create_learning_curves_triple_encoders(
        common_steps,
        list(ENCODERS),
        METHODS,
        means,
        stds,
        TEST_DIR / "der_encoder_learning_curves.pdf",
        # Curves are already Mean HNS. These synthetic references preserve them
        # on the left and map the right axis to Mean HNS / DER baseline Mean HNS.
        mean_reward=baseline_mean_hns,
        random_reward=0.0,
        human_reward=1.0,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
