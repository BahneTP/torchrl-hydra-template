"""Shared renderer for validation learning curves."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np


def _format_step(value: float, _position: float) -> str:
    """Format environment steps compactly, for example 10K instead of 10000."""
    absolute = abs(value)
    if absolute >= 1_000_000:
        scaled = value / 1_000_000
        suffix = "M"
    elif absolute >= 1_000:
        scaled = value / 1_000
        suffix = "K"
    else:
        return f"{value:g}"
    return f"{scaled:g}{suffix}"


def create_learning_curves(
    steps: Sequence[float],
    labels: Sequence[str],
    means: Sequence[Sequence[float]],
    stds: Sequence[Sequence[float]],
    output: str | Path,
    *,
    mean_reward: float,
    random_reward: float,
    human_reward: float,
    xlabel: str = "Environment Steps",
    font: str | Path | None = None,
    legend_outside_right: bool = False,
    legend_outside_left: bool = False,
    figure_width: float = 6.5,
) -> Path:
    """Render raw-score learning curves as HNS means with standard-deviation bands."""
    if not labels or len(labels) != len(means) or len(labels) != len(stds):
        raise ValueError("labels, means, and stds must be non-empty and equally sized")
    if mean_reward <= 0:
        raise ValueError("mean_reward must be positive")
    if human_reward == random_reward:
        raise ValueError("human_reward and random_reward must differ")

    x = np.asarray(steps, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("steps must be a non-empty, finite one-dimensional sequence")
    if np.any(np.diff(x) <= 0):
        raise ValueError("steps must be strictly increasing")

    mean_arrays = [np.asarray(values, dtype=float) for values in means]
    std_arrays = [np.asarray(values, dtype=float) for values in stds]
    for label, mean, std in zip(labels, mean_arrays, std_arrays):
        if mean.shape != x.shape or std.shape != x.shape:
            raise ValueError(f"curve {label!r} does not have one value per step")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError(f"curve {label!r} contains non-finite values")
        if np.any(std < 0):
            raise ValueError(f"curve {label!r} contains negative standard deviations")

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

    scale = human_reward - random_reward
    colors = {
        "DER": "#000000",
        "BBF": "#000000",
        "Attentive MLF": "#D62728",
        "Linear MLF": "#E69F00",
        "Attentive Probing": "#005A7A",
        "Self-Attention Probe": "#005A7A",
        "Linear Probing": "#56B4E9",
        "Linear Probe": "#56B4E9",
        "Full Fine-Tuning": "#A67C52",
        "LoRA": "#666666",
    }
    unknown_labels = [label for label in labels if label not in colors]
    if unknown_labels:
        raise ValueError(f"no curve color configured for labels: {unknown_labels}")
    fig, ax = plt.subplots(figsize=(figure_width, 3), constrained_layout=True)
    for label, raw_mean, raw_std in zip(labels, mean_arrays, std_arrays):
        color = colors[label]
        hns_mean = (raw_mean - random_reward) / scale
        hns_std = raw_std / abs(scale)
        ax.plot(x, hns_mean, color=color, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(
            x,
            hns_mean - hns_std,
            hns_mean + hns_std,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Human-Normalized Score")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_step))
    ax.grid(color="#d4d4d8", linestyle=":", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.margins(x=0)
    if legend_outside_right and legend_outside_left:
        raise ValueError("legend cannot be placed outside both left and right")
    if legend_outside_left:
        ax.legend(
            frameon=False,
            loc="center right",
            bbox_to_anchor=(-0.14, 0.5),
            borderaxespad=0,
        )
    elif legend_outside_right:
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.14, 0.5),
            borderaxespad=0,
        )
    else:
        ax.legend(frameon=False, loc="best")

    ratio_axis = ax.secondary_yaxis(
        "right",
        functions=(
            lambda hns: (hns * scale + random_reward) / mean_reward,
            lambda ratio: (ratio * mean_reward - random_reward) / scale,
        ),
    )
    ratio_axis.set_ylabel("Baseline Normalized Return")

    output_path = Path(output).expanduser().resolve().with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path
