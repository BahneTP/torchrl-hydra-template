#!/usr/bin/env python3
"""Compare linear mixed-layer fusion variants on James Bond."""

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
    "der_resnet18_mlf_linear_20260808_155703/results.csv"
)

SELECTIONS = [
    ("Layers 1 & 4", "resnet18_mlf_linear_layer1_4"),
    ("Layers 2 & 3", "resnet18_mlf_linear_layer2_3"),
    ("All Layers", "resnet18_mlf_linear_all"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--game", default="Jamesbond")
    parser.add_argument(
        "--output", type=Path, default=PLOTS_DIR / "mixed_layer_fusion.pdf"
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
        xlabel="Fused ResNet-18 Layers",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
