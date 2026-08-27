#!/usr/bin/env python3
"""Compare linear DINOv2 mixed-layer fusion variants on James Bond."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLOTS_DIR))
from boxplots_vertical import create_vertical_boxplots  # noqa: E402

DEFAULT_RESULTS = Path(
    "/home/bthiehl/torchrl-hydra-template/logs/mixedlayerfusion/"
    "der_dinov2_mlf_linear_20260808_155743/results.csv"
)

SELECTIONS = [
    ("First 5 Blocks", "dinov2_mlf_linear_first5"),
    ("Last 5 Blocks", "dinov2_mlf_linear_last5"),
    ("Blocks 3, 7 & 11", "dinov2_mlf_linear_block3_7_11"),
    ("All Blocks", "dinov2_mlf_linear_all"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--game", default="Jamesbond")
    parser.add_argument(
        "--output", type=Path, default=PLOTS_DIR / "dinov2_mixed_layer_fusion.pdf"
    )
    args = parser.parse_args()

    results = args.results / "results.csv" if args.results.is_dir() else args.results
    df = pd.read_csv(results)

    labels: list[str] = []
    values: list[list[float]] = []
    for label, variant in SELECTIONS:
        selected = df[
            (df["algorithm"] == "der")
            & (df["game"].str.casefold() == args.game.casefold())
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

    output = create_vertical_boxplots(
        labels,
        values,
        args.output,
        xlabel="Fused DINOv2 Blocks",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
