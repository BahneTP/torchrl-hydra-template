#!/usr/bin/env python3
"""Boxplots for local DINOv2 MLF linear results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS = Path(
    "logs/mixedlayerfusion/der_dinov2_mlf_linear_20260808_155743/results.csv"
)
DEFAULT_OUT = Path("logs/analysis/dinov2_mlf_linear_jamesbond_boxplot")
ORDER = [
    ("dinov2_mlf_linear_all", "all"),
    ("dinov2_mlf_linear_first5", "first5"),
    ("dinov2_mlf_linear_last5", "last5"),
    ("dinov2_mlf_linear_block3_7_11", "3/7/11"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--game", default="Jamesbond")
    parser.add_argument("--min-seeds", type=int, default=1)
    return parser.parse_args()


def load_returns(results: Path, game: str, min_seeds: int) -> tuple[list[str], list[list[float]]]:
    df = pd.read_csv(results)
    df = df[
        (df["algorithm"] == "der")
        & (df["game"] == game)
        & (df["exit_code"] == 0)
        & df["final_return_mean"].notna()
    ].copy()

    labels: list[str] = []
    values: list[list[float]] = []
    for variant, label in ORDER:
        group = df[df["variant"] == variant].sort_values("seed")
        returns = group["final_return_mean"].astype(float).tolist()
        if len(returns) >= min_seeds:
            labels.append(label)
            values.append(returns)
    return labels, values


def plot(labels: list[str], values: list[list[float]], game: str, out: Path) -> None:
    if not values:
        raise SystemExit("no matching finished runs found")

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

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    positions = np.arange(1, len(labels) + 1)

    ax.boxplot(
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

    ax.set_title(f"DINOv2 MLF Linear on {game}")
    ax.set_xlabel("MLF block group")
    ax.set_ylabel("Raw eval return")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
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
            "group": labels,
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
    labels, values = load_returns(args.results, args.game, args.min_seeds)
    plot(labels, values, args.game, args.out)
    print(f"wrote {args.out}.png/.pdf/.svg and {args.out}.csv")


if __name__ == "__main__":
    main()
