#!/usr/bin/env python3
"""Plot the twelve DINOv2 block-depth experiments."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLOTS_DIR))
from boxplots_vertical import create_vertical_boxplots  # noqa: E402

DEFAULT_RESULTS = Path(
    "/home/bthiehl/torchrl-hydra-template/logs/dinoblocks/"
    "der_dinov2_linear_20260808_224430/results.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--game", default="Jamesbond")
    parser.add_argument("--output", type=Path, default=PLOTS_DIR / "dinov2_blocks.pdf")
    args = parser.parse_args()

    results = args.results / "results.csv" if args.results.is_dir() else args.results
    df = pd.read_csv(results)
    pattern = re.compile(r"dinov2_linear_block(\d+)$")
    selected = df[
        (df["game"].str.casefold() == args.game.casefold())
        & (df["exit_code"] == 0)
        & df["variant"].str.match(pattern)
        & df["final_return_mean"].notna()
    ].copy()
    selected["depth"] = selected["variant"].str.extract(pattern).astype(int)
    grouped = {
        int(depth): group.sort_values("seed")["final_return_mean"].astype(float).tolist()
        for depth, group in selected.groupby("depth")
    }
    depths = list(range(1, 13))
    missing = [depth for depth in depths if depth not in grouped]
    if missing:
        raise SystemExit(f"missing DINOv2 blocks: {missing}")

    output = create_vertical_boxplots(
        [str(depth) for depth in depths],
        [grouped[depth] for depth in depths],
        args.output,
        xlabel="DINOv2 Block",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
