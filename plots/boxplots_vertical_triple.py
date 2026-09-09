"""Grouped horizontal boxplots for three encoders and three games."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def create_vertical_boxplots_triple(
    games: Sequence[str],
    methods: Sequence[str],
    encoders: Sequence[str],
    values: Sequence[Sequence[Sequence[Sequence[float]]]],
    baselines: Sequence[Sequence[float]],
    output: str | Path,
    *,
    mean_rewards: Sequence[float],
    random_rewards: Sequence[float],
    human_rewards: Sequence[float],
    baseline_label: str = "DER",
    font: str | Path | None = None,
) -> Path:
    """Render one compact axis per encoder and data column."""
    if len(games) != 3 or len(values) != 3 or len(baselines) != 3:
        raise ValueError("exactly three games are required")
    if not encoders:
        raise ValueError("at least one encoder is required")
    if not methods:
        raise ValueError("at least one transfer method is required")
    if len(mean_rewards) != 3 or len(random_rewards) != 3 or len(human_rewards) != 3:
        raise ValueError("reward references must contain three entries")
    if any(reward <= 0 for reward in mean_rewards):
        raise ValueError("mean rewards must be positive")
    for game_values in values:
        if len(game_values) != len(methods):
            raise ValueError("each game needs one group per method")
        if any(len(method_values) != len(encoders) for method_values in game_values):
            raise ValueError("each method needs one value group per encoder")

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
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    mean_hns_by_pair: dict[tuple[int, int], float] = {}
    for encoder_index in range(len(encoders)):
        for method_index in range(len(methods)):
            game_means = []
            for game_index in range(3):
                scale = human_rewards[game_index] - random_rewards[game_index]
                group = values[game_index][method_index][encoder_index]
                game_means.append(
                    float(
                        np.mean(
                            [
                                (value - random_rewards[game_index]) / scale
                                for value in group
                            ]
                        )
                    )
                )
            mean_hns_by_pair[(encoder_index, method_index)] = float(
                np.mean(game_means)
            )

    sorted_methods = [
        sorted(
            range(len(methods)),
            key=lambda method_index: mean_hns_by_pair[
                (encoder_index, method_index)
            ],
            reverse=True,
        )
        for encoder_index in range(len(encoders))
    ]

    column_bounds: list[tuple[float, float]] = []
    for game_index, game_values in enumerate(values):
        game_normalized_values = [
            value / mean_rewards[game_index]
            for method_values in game_values
            for encoder_values in method_values
            for value in encoder_values
        ]
        game_normalized_values.extend(
            value / mean_rewards[game_index]
            for value in baselines[game_index]
        )
        lower = min(game_normalized_values)
        upper = max(game_normalized_values)
        padding = 0.04 * (upper - lower) if upper > lower else 0.05
        column_bounds.append((lower - padding, upper + padding))

    baseline_game_means = []
    for game_index, baseline in enumerate(baselines):
        scale = human_rewards[game_index] - random_rewards[game_index]
        baseline_game_means.append(
            float(
                np.mean(
                    [
                        (value - random_rewards[game_index]) / scale
                        for value in baseline
                    ]
                )
            )
        )
    baseline_mean_hns = float(np.mean(baseline_game_means))
    normalized_mean_hns_by_pair = {
        pair: mean_hns / baseline_mean_hns
        for pair, mean_hns in mean_hns_by_pair.items()
    }
    all_normalized_mean_hns = [*normalized_mean_hns_by_pair.values(), 1.0]
    mean_upper = max(0.0, max(all_normalized_mean_hns))
    mean_padding = 0.06 * mean_upper if mean_upper > 0 else 0.05
    mean_lower = 0.0 if len(encoders) == 1 else 0.60
    mean_bounds = (mean_lower, mean_upper + mean_padding)

    encoder_colors = ("#CCE5FF",) * len(encoders)
    positions = np.arange(len(methods), 0, -1, dtype=float)
    num_encoder_rows = len(encoders)
    figure_height = 3.8 if num_encoder_rows == 1 else 2.5 * num_encoder_rows + 0.7
    fig, axes = plt.subplots(
        num_encoder_rows + 1,
        4,
        figsize=(7.8, figure_height),
        sharex="col",
        constrained_layout=False,
        squeeze=False,
        gridspec_kw={
            "width_ratios": (0.9, 0.9, 0.9, 0.45),
            "height_ratios": (*([1] * num_encoder_rows), 0.20),
        },
    )
    rng = np.random.default_rng(7)

    for game_index, game in enumerate(games):
        ax = axes[-1, game_index]
        baseline_group = [
            value / mean_rewards[game_index]
            for value in baselines[game_index]
        ]
        _draw_box(ax, 1, baseline_group, encoder_colors[0], rng)
        ax.set_xlim(*column_bounds[game_index])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([1])
        ax.set_yticklabels([baseline_label] if game_index == 0 else [])
        if game_index > 0:
            ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.grid(axis="x", color="#d4d4d8", linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)

    baseline_mean_ax = axes[-1, 3]
    baseline_mean_ax.barh(
        [1],
        [1.0],
        height=0.34,
        color="#FFCCCC",
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )
    baseline_mean_ax.axvline(mean_lower, color="black", linewidth=0.8, zorder=2)
    baseline_mean_ax.set_xlim(*mean_bounds)
    baseline_mean_ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    baseline_mean_ax.set_ylim(0.5, 1.5)
    baseline_mean_ax.set_yticks([])
    baseline_mean_ax.grid(
        axis="x", color="#d4d4d8", linestyle=":", linewidth=0.8
    )
    baseline_mean_ax.set_axisbelow(True)

    for encoder_index, encoder in enumerate(encoders):
        axes_row = encoder_index
        order = sorted_methods[encoder_index]
        ordered_labels = [methods[method_index] for method_index in order]
        for game_index, game in enumerate(games):
            ax = axes[axes_row, game_index]
            for position, method_index in zip(positions, order):
                group = [
                    value / mean_rewards[game_index]
                    for value in values[game_index][method_index][encoder_index]
                ]
                _draw_box(
                    ax,
                    position,
                    group,
                    encoder_colors[encoder_index],
                    rng,
                )
            ax.set_xlim(*column_bounds[game_index])
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.set_ylim(0.5, len(methods) + 0.5)
            ax.set_yticks(positions)
            if game_index == 0:
                ax.set_yticklabels(ordered_labels)
                ax.set_ylabel(encoder, fontweight="bold", labelpad=6)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", left=False, right=False)
            ax.grid(axis="x", color="#d4d4d8", linestyle=":", linewidth=0.8)
            ax.set_axisbelow(True)
            if encoder_index == 0:
                scale = human_rewards[game_index] - random_rewards[game_index]
                hns_axis = ax.secondary_xaxis(
                    "top",
                    functions=(
                        lambda ratio, game_index=game_index, scale=scale: (
                            ratio * mean_rewards[game_index]
                            - random_rewards[game_index]
                        )
                        / scale,
                        lambda hns, game_index=game_index, scale=scale: (
                            hns * scale + random_rewards[game_index]
                        )
                        / mean_rewards[game_index],
                    ),
                )
                hns_axis.xaxis.set_major_formatter(FormatStrFormatter("%g"))
                hns_axis.tick_params(axis="x", labelsize=9)

        mean_ax = axes[axes_row, 3]
        ordered_means = [
            normalized_mean_hns_by_pair[(encoder_index, method_index)]
            for method_index in order
        ]
        mean_ax.barh(
            positions,
            ordered_means,
            height=0.34,
            color="#FFCCCC",
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        mean_ax.axvline(mean_lower, color="black", linewidth=0.8, zorder=2)
        mean_ax.set_xlim(*mean_bounds)
        mean_ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        mean_ax.set_ylim(0.5, len(methods) + 0.5)
        mean_ax.set_yticks(positions)
        mean_ax.set_yticklabels([])
        mean_ax.tick_params(axis="y", which="both", left=False, right=False)
        mean_ax.grid(axis="x", color="#d4d4d8", linestyle=":", linewidth=0.8)
        mean_ax.set_axisbelow(True)
        if encoder_index == 0:
            mean_hns_axis = mean_ax.secondary_xaxis(
                "top",
                functions=(
                    lambda normalized: normalized * baseline_mean_hns,
                    lambda mean_hns: mean_hns / baseline_mean_hns,
                ),
            )
            mean_hns_axis.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            mean_hns_axis.tick_params(axis="x", labelsize=9)
    if num_encoder_rows == 1:
        bottom_margin, top_margin = 0.15, 0.80
        game_label_y, top_axis_label_y, bottom_axis_label_y = 0.985, 0.90, 0.025
    else:
        bottom_margin, top_margin = 0.08, 0.88
        game_label_y, top_axis_label_y, bottom_axis_label_y = 0.955, 0.92, 0.02
    fig.subplots_adjust(
        left=0.23,
        right=0.98,
        bottom=bottom_margin,
        top=top_margin,
        wspace=0,
        hspace=0,
    )
    for column, game in enumerate(games):
        bounds = axes[0, column].get_position()
        fig.text(
            (bounds.x0 + bounds.x1) / 2,
            game_label_y,
            game,
            ha="center",
            va="top",
            fontsize=12,
        )
    mean_bounds_position = axes[0, 3].get_position()
    fig.text(
        (mean_bounds_position.x0 + mean_bounds_position.x1) / 2,
        game_label_y,
        "Mean HNS",
        ha="center",
        va="top",
        fontsize=12,
    )
    first_bounds = axes[0, 0].get_position()
    fourth_bounds = axes[0, 3].get_position()
    fig.text(
        (first_bounds.x0 + fourth_bounds.x1) / 2,
        top_axis_label_y,
        "Human-Normalized Score",
        ha="center",
        fontsize=11,
    )
    first_bottom_bounds = axes[-1, 0].get_position()
    fourth_bottom_bounds = axes[-1, 3].get_position()
    fig.text(
        (first_bottom_bounds.x0 + fourth_bottom_bounds.x1) / 2,
        bottom_axis_label_y,
        "Baseline Normalized Return",
        ha="center",
        fontsize=11,
    )
    output_path = Path(output).expanduser().resolve().with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def _draw_box(
    ax: plt.Axes,
    position: float,
    group: Sequence[float],
    color: str,
    rng: np.random.Generator,
) -> None:
    ax.boxplot(
        [group],
        positions=[position],
        widths=0.34,
        vert=False,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#f26b6b", "linewidth": 1.7},
        whiskerprops={"color": "#000000", "linewidth": 1.2},
        capprops={"color": "#000000", "linewidth": 1.2},
        boxprops={"facecolor": color, "edgecolor": "#000000", "linewidth": 1.2},
    )
    jitter = rng.uniform(-0.055, 0.055, size=len(group))
    ax.scatter(
        group,
        np.full(len(group), position) + jitter,
        s=22,
        facecolor="#FFCC99",
        edgecolor="#000000",
        linewidth=0.6,
        alpha=0.85,
        zorder=3,
    )
    ax.scatter(
        [float(np.mean(group))],
        [position],
        s=44,
        facecolor="white",
        edgecolor="black",
        linewidth=1.3,
        zorder=5,
    )
