#!/usr/bin/env python3
"""Plot the four ResNet-18 layer-depth experiments."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = VALIDATION_DIR.parent
sys.path.insert(0, str(PLOTS_DIR))
from boxplots_vertical import create_vertical_boxplots  # noqa: E402
from atari_scores import get_atari_reference_scores  # noqa: E402

DEFAULT_RESULTS = Path(
    "/home/bthiehl/torchrl-hydra-template/logs/resnetlayer/"
    "der_resnet18_linear_20260808_152354/results.csv"
)
PLAIN_RESULTS = Path(
    "/home/bthiehl/torchrl-hydra-template/logs/plain_algorithms/der/"
    "atari100k_20260810_181403/results.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--game", default="Jamesbond")
    parser.add_argument("--output", type=Path, default=VALIDATION_DIR / "resnet18_layers.pdf")
    args = parser.parse_args()

    results = args.results / "results.csv" if args.results.is_dir() else args.results
    df = pd.read_csv(results)
    pattern = re.compile(r"resnet18_linear_layer(\d+)$")
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
    depths = list(range(1, 5))
    missing = [depth for depth in depths if depth not in grouped]
    if missing:
        raise SystemExit(f"missing ResNet-18 layers: {missing}")
    plain = pd.read_csv(PLAIN_RESULTS)
    mean_reward = plain[
        (plain["algorithm"] == "der")
        & (plain["game"].str.casefold() == args.game.casefold())
        & (plain["exit_code"] == 0)
    ]["final_return_mean"].mean()
    random_reward, human_reward = get_atari_reference_scores(args.game)

    output = create_vertical_boxplots(
        [str(depth) for depth in depths],
        [grouped[depth] for depth in depths],
        args.output,
        xlabel="ResNet-18 Layer",
        mean_reward=float(mean_reward),
        random_reward=random_reward,
        human_reward=human_reward,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
