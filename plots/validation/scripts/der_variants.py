#!/usr/bin/env python3
"""Plot the best validation-selected DER transfer variants on James Bond."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = VALIDATION_DIR.parent
REPO_ROOT = PLOTS_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
from atari_scores import get_atari_reference_scores  # noqa: E402
from learning_curves_double_encoders import (  # noqa: E402
    create_learning_curves_double_encoders,
)
from wandb_history import load_evaluation_curve  # noqa: E402

GAME = "Jamesbond"
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
# Each entry: (run_directory, variant, seeds). Selected as the best-performing
# hyperparameters found in the corresponding validation plot for that encoder.
ENCODERS = {
    "ResNet-18": (
        (
            REPO_ROOT / "logs/fullfinetuning/der_resnet18_layer2_20260811_025944/lr_1e_6",
            "resnet18_full_layer2_lr_1e_6",
        ),
        (
            REPO_ROOT
            / "logs/resnetlayer/attentive_lrsweep/"
            "der_resnet18_attentive_self_attention_layer2_probe_lr_sweep_"
            "jamesbond_20260812_130734/probe_lr_1e_6",
            "resnet18_attentive_self_attention_layer2_probe_lr_1e_6",
        ),
        (
            REPO_ROOT / "logs/resnetlayer/der_resnet18_linear_20260808_152354/layer_2",
            "resnet18_linear_layer2",
        ),
        (
            REPO_ROOT / "logs/mixedlayerfusion/der_resnet18_mlf_linear_20260808_155703/all",
            "resnet18_mlf_linear_all",
        ),
        (
            REPO_ROOT
            / "logs/mixedlayerfusion/"
            "der_resnet18_dinov2_mlf_self_attention_probe_lr_1e_6_jamesbond_"
            "20260818_233533/resnet18/all",
            "resnet18_mlf_self_attention_all_probe_lr_1e_6",
        ),
        (
            REPO_ROOT
            / "logs/lora/der_resnet18_layer2_lora_rank_alpha_sweep_jamesbond_"
            "20260817_134400/r4_a8",
            "resnet18_lora_layer2_r4_a8",
        ),
        (PLAIN_DER, "plain"),
    ),
    "DINOv2": (
        (
            REPO_ROOT / "logs/fullfinetuning/der_dinov2_block7_20260814_191820/lr_1e_3",
            "dinov2_full_block7_lr_1e_3",
        ),
        (
            REPO_ROOT
            / "logs/dinoblocks/"
            "der_dinov2_attentive_self_attention_block7_probe_lr_1e_6_4games_"
            "20260817_112612",
            "dinov2_attentive_self_attention_block7_probe_lr_1e_6",
        ),
        (
            REPO_ROOT / "logs/dinoblocks/der_dinov2_linear_20260808_224430/block_7",
            "dinov2_linear_block7",
        ),
        (
            REPO_ROOT / "logs/mixedlayerfusion/der_dinov2_mlf_linear_20260808_155743/all",
            "dinov2_mlf_linear_all",
        ),
        (
            REPO_ROOT
            / "logs/mixedlayerfusion/"
            "der_resnet18_dinov2_mlf_self_attention_probe_lr_1e_6_jamesbond_"
            "20260818_233533/dinov2/all",
            "dinov2_mlf_self_attention_all_probe_lr_1e_6",
        ),
        (
            REPO_ROOT
            / "logs/lora/der_dinov2_block7_lora_rank_alpha_sweep_jamesbond_"
            "20260818_110728/r4_a8",
            "dinov2_lora_block7_r4_a8",
        ),
        (PLAIN_DER, "plain"),
    ),
}
SEEDS = (1, 2, 3, 4, 5)
EVALUATION_STEPS = tuple(range(10_000, 100_001, 10_000))


def final_evaluation(run_directory: Path, variant: str) -> tuple[float, float]:
    results = pd.read_csv(_results_csv_for(run_directory))
    selected = results[
        (results["algorithm"] == "der")
        & (results["game"].str.casefold() == GAME.casefold())
        & (results["variant"] == variant)
        & (results["exit_code"] == 0)
        & results["final_return_mean"].notna()
        & results["seed"].isin(SEEDS)
    ].sort_values("seed")
    if len(selected) != len(SEEDS) or selected["seed"].nunique() != len(SEEDS):
        raise SystemExit(
            f"expected {len(SEEDS)} successful, unique final evaluations for "
            f"{variant}, found {len(selected)} rows and "
            f"{selected['seed'].nunique()} seeds"
        )
    values = selected["final_return_mean"].to_numpy(dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def _results_csv_for(run_directory: Path) -> Path:
    """Walk up from a variant subdirectory to the results.csv that covers it."""
    candidate = run_directory
    while not (candidate / "results.csv").is_file():
        if candidate == candidate.parent:
            raise SystemExit(f"no results.csv found above {run_directory}")
        candidate = candidate.parent
    return candidate / "results.csv"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download W&B histories again instead of using cached CSV files",
    )
    args = parser.parse_args()

    random_reward, human_reward = get_atari_reference_scores(GAME)

    common_steps: list[int] | None = None
    means: list[list[list[float]]] = []
    stds: list[list[list[float]]] = []
    for encoder, runs in ENCODERS.items():
        encoder_means: list[list[float]] = []
        encoder_stds: list[list[float]] = []
        for method, (run_directory, variant) in zip(METHODS, runs):
            steps, raw_mean, raw_std = load_curve(
                run_directory, variant, refresh=args.refresh
            )
            if common_steps is None:
                common_steps = steps
            elif steps != common_steps:
                raise SystemExit(
                    f"evaluation steps differ for {encoder}/{method}"
                )
            encoder_means.append(raw_mean)
            encoder_stds.append(raw_std)
        means.append(encoder_means)
        stds.append(encoder_stds)

    if common_steps is None:
        raise SystemExit("no evaluation curves were loaded")

    baseline_mean, _ = final_evaluation(PLAIN_DER, "plain")

    output = create_learning_curves_double_encoders(
        common_steps,
        list(ENCODERS),
        METHODS,
        means,
        stds,
        VALIDATION_DIR / "der_variants.pdf",
        mean_reward=baseline_mean,
        random_reward=random_reward,
        human_reward=human_reward,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
