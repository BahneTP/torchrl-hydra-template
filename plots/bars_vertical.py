"""Shared vertical-bar-chart renderer (bar version of ``boxplots_vertical``)."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def create_vertical_bars(
    labels: Sequence[str],
    values: Sequence[Sequence[float]],
    output: str | Path,
    *,
    xlabel: str,
    ylabel: str = "Raw evaluation return",
    mean_reward: float | None = None,
    random_reward: float | None = None,
    human_reward: float | None = None,
    font: str | Path | None = None,
    figure_width: float | None = None,
    figure_height: float = 3,
    ymin: float | None = None,
    bar_color: str = "#FFCC99",
) -> Path:
    """Render labeled seed distributions as bars (mean height) to a tight PDF."""
    if not labels or len(labels) != len(values):
        raise ValueError("labels and values must be non-empty and have equal lengths")
    if any(not group for group in values):
        raise ValueError("every bar requires at least one value")
    if mean_reward is not None and mean_reward <= 0:
        raise ValueError("mean_reward must be positive")
    use_hns = random_reward is not None or human_reward is not None
    if use_hns and (random_reward is None or human_reward is None):
        raise ValueError("random_reward and human_reward must be provided together")
    if use_hns and human_reward == random_reward:
        raise ValueError("human_reward and random_reward must differ")

    plotted_values = values
    if use_hns:
        scale = human_reward - random_reward
        plotted_values = [
            [(reward - random_reward) / scale for reward in group]
            for group in values
        ]

    font_path = Path(font or Path(__file__).parent / "fonts" / "NewCM08-Regular.otf")
    font_path = font_path.expanduser().resolve()
    if not font_path.is_file():
        raise FileNotFoundError(f"font file not found: {font_path}")
    font_manager.fontManager.addfont(font_path)
    font_family = font_manager.FontProperties(fname=font_path).get_name()

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.family": font_family,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    positions = np.arange(1, len(labels) + 1)
    means = [float(np.mean(group)) for group in plotted_values]
    fig_width = figure_width or max(6.5, 0.52 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, figure_height), constrained_layout=True)

    ax.bar(
        positions,
        means,
        width=0.58,
        facecolor=bar_color,
        edgecolor="#000000",
        linewidth=1.5,
        zorder=2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Human-Normalized Score" if use_hns else ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis="y", color="#d4d4d8", linestyle=":", linewidth=1.0)
    ax.set_axisbelow(True)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if mean_reward is not None:
        if use_hns:
            scale = human_reward - random_reward
            ratio_axis = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda hns: (hns * scale + random_reward) / mean_reward,
                    lambda ratio: (ratio * mean_reward - random_reward) / scale,
                ),
            )
        else:
            ratio_axis = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda reward: reward / mean_reward,
                    lambda ratio: ratio * mean_reward,
                ),
            )
        ratio_axis.set_ylabel("Baseline Normalized Return")

    output_path = Path(output).expanduser().resolve().with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path
