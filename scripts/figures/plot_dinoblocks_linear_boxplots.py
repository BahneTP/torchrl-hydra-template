#!/usr/bin/env python3
"""Boxplots for local DINOv2 block results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS = {
    "linear": Path("logs/dinoblocks/der_dinov2_linear_20260808_224430/results.csv"),
    "attentive": Path("logs/dinoblocks/der_dinov2_attentive_20260808_224440/results.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=sorted(DEFAULT_RESULTS), default="linear")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--game", default="Jamesbond")
    parser.add_argument("--min-seeds", type=int, default=1)
    return parser.parse_args()


def load_block_returns(
    results: Path, probe: str, game: str, min_seeds: int
) -> dict[int, list[float]]:
    block_re = re.compile(rf"dinov2_{probe}_block(\d+)$")
    df = pd.read_csv(results)
    df = df[
        (df["algorithm"] == "der")
        & (df["game"] == game)
        & (df["exit_code"] == 0)
        & df["variant"].str.match(block_re)
        & df["final_return_mean"].notna()
    ].copy()
    df["block"] = df["variant"].str.extract(block_re).astype(int)

    block_returns: dict[int, list[float]] = {}
    for block, group in sorted(df.groupby("block"), key=lambda item: item[0]):
        values = group.sort_values("seed")["final_return_mean"].astype(float).tolist()
        if len(values) >= min_seeds:
            block_returns[int(block)] = values
    return block_returns


def plot(block_returns: dict[int, list[float]], probe: str, game: str, out: Path) -> None:
    if not block_returns:
        raise SystemExit("no matching finished runs found")

    blocks = list(block_returns)
    values = [block_returns[block] for block in blocks]
    means = [float(np.mean(v)) for v in values]
    counts = [len(v) for v in values]

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    positions = np.arange(1, len(blocks) + 1)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.6},
        whiskerprops={"color": "#4b5563", "linewidth": 1.2},
        capprops={"color": "#4b5563", "linewidth": 1.2},
        boxprops={"facecolor": "#8ecae6", "edgecolor": "#1f2937", "linewidth": 1.2},
    )

    rng = np.random.default_rng(7)
    for x, vals in zip(positions, values):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            s=22,
            color="#023047",
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )

    ax.plot(
        positions,
        means,
        color="#fb8500",
        marker="o",
        linewidth=1.6,
        markersize=4,
        label="Mean",
        zorder=4,
    )

    ax.set_title(f"DINOv2 {probe.title()} Blocks on {game}")
    ax.set_xlabel("DINOv2 output block")
    ax.set_ylabel("Raw eval return")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(block) for block in blocks])
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    ymax = max(max(v) for v in values)
    ax.set_ylim(bottom=0, top=ymax * 1.08)

    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "block": blocks,
            "n": counts,
            "mean": means,
            "std": [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in values],
            "min": [float(np.min(v)) for v in values],
            "median": [float(np.median(v)) for v in values],
            "max": [float(np.max(v)) for v in values],
        }
    )
    summary.to_csv(out.with_suffix(".csv"), index=False)


def main() -> None:
    args = parse_args()
    results = args.results or DEFAULT_RESULTS[args.probe]
    out = args.out or Path(f"logs/analysis/dinoblocks_{args.probe}_jamesbond_boxplot")
    block_returns = load_block_returns(results, args.probe, args.game, args.min_seeds)
    plot(block_returns, args.probe, args.game, out)
    print(f"wrote {out}.png/.pdf/.svg and {out}.csv")


if __name__ == "__main__":
    main()
