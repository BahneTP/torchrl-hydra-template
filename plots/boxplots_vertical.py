"""Shared vertical-boxplot renderer."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def create_vertical_boxplots(
    labels: Sequence[str],
    values: Sequence[Sequence[float]],
    output: str | Path,
    *,
    xlabel: str,
    ylabel: str = "Raw evaluation return",
    font: str | Path | None = None,
) -> Path:
    """Render labeled seed distributions and their means to a tight PDF."""
    if not labels or len(labels) != len(values):
        raise ValueError("labels and values must be non-empty and have equal lengths")
    if any(not group for group in values):
        raise ValueError("every boxplot requires at least one value")

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
    means = [float(np.mean(group)) for group in values]
    fig_width = max(6.5, 0.52 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 3), constrained_layout=True)

    ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        vert=True,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#f26b6b", "linewidth": 2.0},
        whiskerprops={"color": "#000000", "linewidth": 1.5},
        capprops={"color": "#000000", "linewidth": 1.5},
        boxprops={
            "facecolor": "#CCE5FF",
            "edgecolor": "#000000",
            "linewidth": 1.5,
        },
    )

    rng = np.random.default_rng(7)
    for position, group in zip(positions, values):
        jitter = rng.uniform(-0.10, 0.10, size=len(group))
        ax.scatter(
            np.full(len(group), position) + jitter,
            group,
            s=38,
            facecolor="#FFCC99",
            edgecolor="#000000",
            linewidth=0.8,
            alpha=0.85,
            zorder=3,
        )

    ax.scatter(
        positions,
        means,
        s=76,
        facecolor="white",
        edgecolor="black",
        linewidth=1.8,
        zorder=5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis="y", color="#d4d4d8", linestyle=":", linewidth=1.0)
    ax.set_axisbelow(True)

    output_path = Path(output).expanduser().resolve().with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path
