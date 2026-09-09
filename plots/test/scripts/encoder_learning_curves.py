#!/usr/bin/env python3
"""Compare transfer methods across three encoders for three Atari games."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parent
PLOTS_DIR = TEST_DIR.parent
REPO_ROOT = PLOTS_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
from atari_scores import get_atari_reference_scores  # noqa: E402
from learning_curves_triple_encoders import (  # noqa: E402
    create_learning_curves_triple_encoders,
)
from wandb_history import load_evaluation_curve  # noqa: E402

GAMES = ("Assault", "BankHeist", "RoadRunner")
PLAIN_DER = REPO_ROOT / "logs/plain_algorithms/der/atari100k_20260810_181403"
METHODS = (
    "Full Fine-Tuning",
    "Attentive Probing",
    "Linear Probing",
    "Linear MLF",
    "Attentive MLF",
    "LoRA",
    "DER",
)
ENCODERS = {
    "ResNet-18": (
        REPO_ROOT
        / "logs/fullfinetuning/"
        "der_resnet18_layer2_full_lr_1e_6_3games_20260816_222225",
        REPO_ROOT
        / "logs/resnetlayer/"
        "der_resnet18_attentive_self_attention_layer2_probe_lr_1e_6_4games_"
        "20260817_112519",
        REPO_ROOT
        / "logs/resnetlayer/"
        "der_resnet18_linear_layer2_3games_20260810_195914",
        REPO_ROOT
        / "logs/mixedlayerfusion/"
        "der_resnet18_mlf_linear_all_3games_20260811_004533",
        REPO_ROOT
        / "logs/mixedlayerfusion/"
        "der_resnet18_mlf_self_attention_all_probe_lr_1e_6_3games_"
        "20260822_053515",
        REPO_ROOT
        / "logs/lora/"
        "der_resnet18_layer2_lora_r4_a8_lr_1e_7_3games_20260819_221713",
        PLAIN_DER,
    ),
    "DINOv2": (
        REPO_ROOT
        / "logs/fullfinetuning/"
        "der_dinov2_block7_full_lr_1e_3_3games_20260817_114234",
        REPO_ROOT
        / "logs/dinoblocks/"
        "der_dinov2_attentive_self_attention_block7_probe_lr_1e_6_4games_"
        "20260817_112612",
        REPO_ROOT
        / "logs/dinoblocks/der_dinov2_linear_block7_3games_20260817_114321",
        REPO_ROOT
        / "logs/mixedlayerfusion/"
        "der_dinov2_mlf_linear_all_3games_20260812_012715",
        REPO_ROOT
        / "logs/mixedlayerfusion/"
        "der_dinov2_mlf_self_attention_all_probe_lr_1e_6_3games_"
        "20260821_190909",
        REPO_ROOT
        / "logs/lora/"
        "der_dinov2_block7_lora_r4_a8_lr_1e_3_3games_20260821_002825",
        PLAIN_DER,
    ),
    "LeWorldModel": (
        REPO_ROOT
        / "logs/lewm/experiment/"
        "der_lewm_full_block7_lr_1e_3_3games_20260820_180024",
        REPO_ROOT
        / "logs/lewm/experiment/"
        "der_lewm_attentive_self_attention_block7_probe_lr_1e_6_3games_"
        "20260821_002919",
        REPO_ROOT
        / "logs/lewm/experiment/"
        "der_lewm_linear_block7_probe_lr_1e_4_3games_20260820_180845",
        REPO_ROOT
        / "logs/lewm/experiment/"
        "der_lewm_mlf_linear_all_probe_lr_1e_4_3games_20260821_192128",
        REPO_ROOT
        / "logs/lewm/experiment/"
        "der_lewm_mlf_attentive_self_attention_all_probe_lr_1e_6_3games_"
        "20260821_003715",
        REPO_ROOT
        / "logs/lewm/experiment/"
        "der_lewm_lora_block7_r4_a8_lr_1e_3_3games_20260821_192656",
        PLAIN_DER,
    ),
}


def seeds_for(run_directory: Path, game: str) -> tuple[int, ...]:
    if run_directory == PLAIN_DER and game == "BankHeist":
        return (1, 3, 4, 5, 6)
    return (1, 2, 3, 4, 5)


def final_evaluation(
    run_directory: Path, game: str, seeds: tuple[int, ...]
) -> tuple[float, float]:
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
            f"expected {len(seeds)} successful, unique final evaluations for "
            f"{run_directory.name} on {game}, found {len(selected)} rows and "
            f"{selected['seed'].nunique()} seeds"
        )
    values = selected["final_return_mean"].to_numpy(dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def load_curve(
    run_directory: Path, game: str, *, refresh: bool
) -> tuple[list[int], list[float], list[float]]:
    seeds = seeds_for(run_directory, game)
    steps, mean, std = load_evaluation_curve(
        run_directory,
        game,
        seeds=seeds,
        refresh=refresh,
        evaluation_steps=tuple(range(10_000, 100_001, 10_000)),
    )
    try:
        final_index = steps.index(100_000)
    except ValueError as error:
        raise SystemExit(
            f"curve {run_directory.name} on {game} has no point at 100K"
        ) from error
    mean[final_index], std[final_index] = final_evaluation(
        run_directory, game, seeds
    )
    return steps, mean, std


def plot_game(game: str, *, refresh: bool) -> Path:
    common_steps: list[int] | None = None
    means: list[list[list[float]]] = []
    stds: list[list[list[float]]] = []
    for encoder, run_directories in ENCODERS.items():
        encoder_means: list[list[float]] = []
        encoder_stds: list[list[float]] = []
        for method, run_directory in zip(METHODS, run_directories):
            steps, mean, std = load_curve(run_directory, game, refresh=refresh)
            if common_steps is None:
                common_steps = steps
            elif steps != common_steps:
                raise SystemExit(
                    f"evaluation steps differ for {encoder}/{method} on {game}"
                )
            encoder_means.append(mean)
            encoder_stds.append(std)
        means.append(encoder_means)
        stds.append(encoder_stds)

    if common_steps is None:
        raise SystemExit(f"no curves loaded for {game}")
    baseline_mean, _ = final_evaluation(
        PLAIN_DER, game, seeds_for(PLAIN_DER, game)
    )
    random_reward, human_reward = get_atari_reference_scores(game)
    output = create_learning_curves_triple_encoders(
        common_steps,
        list(ENCODERS),
        METHODS,
        means,
        stds,
        TEST_DIR / f"{game.casefold()}_encoder_learning_curves.pdf",
        mean_reward=baseline_mean,
        random_reward=random_reward,
        human_reward=human_reward,
    )
    print(f"wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download W&B histories again instead of using cached CSV files",
    )
    args = parser.parse_args()
    for game in GAMES:
        plot_game(game, refresh=args.refresh)


if __name__ == "__main__":
    main()
