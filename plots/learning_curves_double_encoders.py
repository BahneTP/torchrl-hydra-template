"""Render two encoder panels for one game with shared outer y-axes."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np


def _format_step(value: float, _position: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def create_learning_curves_double_encoders(
    steps: Sequence[float],
    encoders: Sequence[str],
    labels: Sequence[str],
    means: Sequence[Sequence[Sequence[float]]],
    stds: Sequence[Sequence[Sequence[float]]],
    output: str | Path,
    *,
    mean_reward: float,
    random_reward: float,
    human_reward: float,
    xlabel: str = "Environment Steps",
    font: str | Path | None = None,
) -> Path:
    """Render one game's curves for two encoders on a shared HNS scale."""
    if len(encoders) != 2:
        raise ValueError("exactly two encoders are required")
    if not labels or any(len(values) != len(labels) for values in means):
        raise ValueError("each encoder needs one mean curve per label")
    if any(len(values) != len(labels) for values in stds):
        raise ValueError("each encoder needs one standard-deviation curve per label")
    if len(means) != 2 or len(stds) != 2:
        raise ValueError("means and stds must contain exactly two encoders")
    if mean_reward <= 0:
        raise ValueError("mean_reward must be positive")
    if human_reward == random_reward:
        raise ValueError("human_reward and random_reward must differ")

    x = np.asarray(steps, dtype=float)
    if x.ndim != 1 or x.size == 0 or np.any(np.diff(x) <= 0):
        raise ValueError("steps must be non-empty and strictly increasing")

    font_path = Path(font or Path(__file__).parent / "fonts/NewCM08-Regular.otf")
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

    colors = {
        "DER": "#000000",
        "Attentive MLF": "#D62728",
        "Linear MLF": "#E69F00",
        "Attentive Probing": "#005A7A",
        "Linear Probing": "#56B4E9",
        "Full Fine-Tuning": "#A67C52",
        "LoRA": "#666666",
    }
    unknown_labels = [label for label in labels if label not in colors]
    if unknown_labels:
        raise ValueError(f"no curve color configured for labels: {unknown_labels}")
    scale = human_reward - random_reward
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 3.6), sharey=True, constrained_layout=True
    )
    line_min = float("inf")
    line_max = float("-inf")
    for ax, encoder, encoder_means, encoder_stds in zip(
        axes, encoders, means, stds
    ):
        for label, raw_mean, raw_std in zip(labels, encoder_means, encoder_stds):
            mean = np.asarray(raw_mean, dtype=float)
            std = np.asarray(raw_std, dtype=float)
            if mean.shape != x.shape or std.shape != x.shape:
                raise ValueError(f"curve {label!r} for {encoder} does not match steps")
            color = colors[label]
            hns_mean = (mean - random_reward) / scale
            hns_std = std / abs(scale)
            line_min = min(line_min, float(hns_mean.min()))
            line_max = max(line_max, float(hns_mean.max()))
            ax.plot(x, hns_mean, color=color, linewidth=2, label=label, zorder=3)
            ax.fill_between(
                x,
                hns_mean - hns_std,
                hns_mean + hns_std,
                color=color,
                alpha=0.1,
                linewidth=0,
                zorder=2,
            )
        ax.set_title(encoder)
        ax.set_xlim(10_000, 100_000)
        ax.set_xticks(np.arange(20_000, 100_001, 20_000))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_step))
        ax.grid(color="#d4d4d8", linestyle=":", linewidth=1)
        ax.set_axisbelow(True)

    line_range = line_max - line_min
    padding = 0.05 * line_range if line_range > 0 else 0.05
    axes[0].set_ylim(line_min - padding, line_max + padding)
    axes[0].set_ylabel("Human-Normalized Score")
    for ax in axes[1:]:
        ax.tick_params(axis="y", which="both", left=True, labelleft=False)
    fig.supxlabel(xlabel)

    ratio_axes = [
        ax.secondary_yaxis(
            "right",
            functions=(
                lambda hns: (hns * scale + random_reward) / mean_reward,
                lambda ratio: (ratio * mean_reward - random_reward) / scale,
            ),
        )
        for ax in axes
    ]
    for ratio_axis in ratio_axes[:-1]:
        ratio_axis.tick_params(axis="y", which="both", right=True, labelright=False)
    ratio_axes[-1].set_ylabel("Baseline Normalized Return")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=10,
    )

    output_path = Path(output).expanduser().resolve().with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path
