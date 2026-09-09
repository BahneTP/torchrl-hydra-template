#!/usr/bin/env python3
"""Plot the selected single-seed BBF transfer variants as learning curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = VALIDATION_DIR.parent
REPO_ROOT = PLOTS_DIR.parent
PLAIN_RESULTS = REPO_ROOT / (
    "logs/plain_algorithms/bbf/atari100k_jamesbond_5seeds_20260906_143047/"
    "results.csv"
)
sys.path.insert(0, str(PLOTS_DIR))
from atari_scores import get_atari_reference_scores  # noqa: E402
from learning_curves import create_learning_curves  # noqa: E402
from wandb_history import load_evaluation_curve  # noqa: E402

GAME = "Jamesbond"
SEEDS = (1,)
EVALUATION_STEPS = tuple(range(10_000, 100_001, 10_000))
SELECTIONS = [
    (
        "Full Fine-Tuning",
        REPO_ROOT
        / "logs/bbf-testing/full/full_all_lr_sweep_with_shrink_perturb_jamesbond_"
        "seed1_20260824_132255/all_lr_1e_5",
        "resnet18_layer2_full_all_lr_1e_5_with_shrink_perturb",
    ),
    (
        "Attentive Probing",
        REPO_ROOT
        / "logs/bbf-testing/attentive/attentive_all_lr_sweep_with_shrink_perturb_"
        "jamesbond_seed1_20260825_103132/all_lr_1e_5",
        "resnet18_layer2_attentive_all_lr_1e_5_with_shrink_perturb",
    ),
    (
        "Linear Probing",
        REPO_ROOT
        / "logs/bbf-testing/linear/linear_all_lr_sweep_with_shrink_perturb_"
        "jamesbond_seed1_20260825_110640/all_lr_1e_4",
        "resnet18_layer2_linear_all_lr_1e_4_with_shrink_perturb",
    ),
    (
        "LoRA",
        REPO_ROOT
        / "logs/bbf-testing/lora/lora_rank_alpha_sweep_all_lr_1e_4_with_shrink_"
        "perturb_jamesbond_seed1_20260825_185809/r1_a2",
        "resnet18_layer2_lora_r1_a2_all_lr_1e_4_with_shrink_perturb",
    ),
    (
        "Linear MLF",
        REPO_ROOT
        / "logs/bbf-testing/mlf_linear_probe_lr_sweep_with_shrink_perturb_"
        "jamesbond_seed1_20260829_190956/probe_lr_1e_4",
        "resnet18_mlf_linear_probe_lr_1e_4_with_shrink_perturb",
    ),
    (
        "Attentive MLF",
        REPO_ROOT
        / "logs/bbf-testing/mlf_attentive_probe_lr_sweep_with_shrink_perturb_"
        "jamesbond_seed1_20260830_144732/probe_lr_1e_4",
        "resnet18_mlf_attentive_probe_lr_1e_4_with_shrink_perturb",
    ),
]


def _results_csv_for(run_directory: Path) -> Path:
    """Walk up from a variant subdirectory to the results.csv that covers it."""
    candidate = run_directory
    while not (candidate / "results.csv").is_file():
        if candidate == candidate.parent:
            raise SystemExit(f"no results.csv found above {run_directory}")
        candidate = candidate.parent
    return candidate / "results.csv"


def final_evaluation(run_directory: Path, variant: str) -> tuple[float, float]:
    results = pd.read_csv(_results_csv_for(run_directory))
    selected = results[
        (results["algorithm"] == "bbf")
        & (results["game"].str.casefold() == GAME.casefold())
        & (results["variant"] == variant)
        & (results["seed"] == 1)
        & (results["exit_code"] == 0)
        & results["final_return_mean"].notna()
    ]
    if len(selected) != 1:
        raise SystemExit(
            f"expected exactly one successful seed-1 row for {variant}, "
            f"found {len(selected)}"
        )
    value = float(selected.iloc[0]["final_return_mean"])
    return value, 0.0


def load_curve(
    run_directory: Path, variant: str, *, refresh: bool
) -> tuple[list[int], list[float], list[float]]:
    steps, mean, std = load_evaluation_curve(
        run_directory,
        GAME,
        seeds=SEEDS,
        refresh=refresh,
        evaluation_steps=EVALUATION_STEPS,
    )
    try:
        final_index = steps.index(100_000)
    except ValueError as error:
        raise SystemExit(
            f"curve {run_directory.name} on {GAME} has no point at 100K"
        ) from error
    mean[final_index], std[final_index] = final_evaluation(run_directory, variant)
    return steps, mean, std


def final_evaluation_baseline(run_directory: Path, seeds: tuple[int, ...]) -> tuple[float, float]:
    results = pd.read_csv(run_directory / "results.csv")
    selected = results[
        (results["algorithm"] == "bbf")
        & (results["game"].str.casefold() == GAME.casefold())
        & (results["exit_code"] == 0)
        & results["final_return_mean"].notna()
        & results["seed"].isin(seeds)
    ].sort_values("seed")
    if len(selected) != len(seeds) or selected["seed"].nunique() != len(seeds):
        raise SystemExit(
            f"expected {len(seeds)} successful, unique final evaluations for "
            f"plain BBF baseline, found {len(selected)} rows and "
            f"{selected['seed'].nunique()} seeds"
        )
    values = selected["final_return_mean"].to_numpy(dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def load_baseline_curve(
    run_directory: Path, *, refresh: bool
) -> tuple[list[int], list[float], list[float]]:
    seeds = (1, 2, 3, 4, 5)
    steps, mean, std = load_evaluation_curve(
        run_directory,
        GAME,
        seeds=seeds,
        refresh=refresh,
        evaluation_steps=EVALUATION_STEPS,
    )
    try:
        final_index = steps.index(100_000)
    except ValueError as error:
        raise SystemExit(
            f"baseline curve on {GAME} has no point at 100K"
        ) from error
    mean[final_index], std[final_index] = final_evaluation_baseline(run_directory, seeds)
    return steps, mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download W&B histories again instead of using cached CSV files",
    )
    args = parser.parse_args()

    labels: list[str] = []
    means: list[list[float]] = []
    stds: list[list[float]] = []
    common_steps: list[int] | None = None
    for label, run_directory, variant in SELECTIONS:
        steps, mean, std = load_curve(run_directory, variant, refresh=args.refresh)
        if common_steps is None:
            common_steps = steps
        elif steps != common_steps:
            raise SystemExit(f"evaluation steps differ for {label}")
        labels.append(label)
        means.append(mean)
        stds.append(std)

    plain_directory = PLAIN_RESULTS.parent
    baseline_steps, baseline_mean, baseline_std = load_baseline_curve(
        plain_directory, refresh=args.refresh
    )
    if common_steps is None:
        common_steps = baseline_steps
    elif baseline_steps != common_steps:
        raise SystemExit("evaluation steps differ for BBF baseline")
    labels.append("BBF")
    means.append(baseline_mean)
    stds.append(baseline_std)

    if common_steps is None:
        raise SystemExit("no evaluation curves were loaded")

    plain = pd.read_csv(PLAIN_RESULTS)
    mean_reward = plain[
        (plain["algorithm"] == "bbf")
        & (plain["game"].str.casefold() == GAME.casefold())
        & (plain["exit_code"] == 0)
    ]["final_return_mean"].mean()
    random_reward, human_reward = get_atari_reference_scores(GAME)

    output = create_learning_curves(
        common_steps,
        labels,
        means,
        stds,
        VALIDATION_DIR / "bbf_variants.pdf",
        xlabel="Environment Steps",
        mean_reward=float(mean_reward),
        random_reward=random_reward,
        human_reward=human_reward,
        legend_outside_left=True,
        figure_width=8.5,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
