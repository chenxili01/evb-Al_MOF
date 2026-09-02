"""Generate the Figure 3d boundary-layer effects as a full-linewidth 1x4 panel.

This presentation variant reuses the Figure 3d calculation and styling, with
the four reaction steps arranged in one row across the 6.30-inch manuscript
linewidth specified in figsettings.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_figure3d as figure3d


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EFFECTS_CSV = DATA_DIR / "figure3d_boundary_minus_inner_effects.csv"
PAIRED_EFFECTS_CSV = DATA_DIR / "figure3d_paired_boundary_layer_effects.csv"
FULL_LINEWIDTH_IN = 6.30
FIGURE_HEIGHT_IN = 1.75


# The full-linewidth layout leaves room for larger, presentation-ready text
# and symbols than the compact 2x2 source figure.
figure3d.matplotlib.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 5.0,
    }
)


def draw_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    paired_table: pd.DataFrame,
    reaction: str,
    y_limits: tuple[float, float],
    effect_column: str,
    paired_column: str,
) -> None:
    """Draw one enlarged-text, enlarged-marker reaction panel."""
    subset = table[table["reaction"] == reaction].set_index("system_id").reindex(figure3d.SYSTEM_ORDER)
    x = np.arange(len(figure3d.SYSTEM_ORDER))
    effect = subset[effect_column].to_numpy(float)
    low = subset["ci95_low"].to_numpy(float)
    high = subset["ci95_high"].to_numpy(float)
    raw_pairs = paired_table[paired_table["reaction"] == reaction]
    for system_index, system_id in enumerate(figure3d.SYSTEM_ORDER):
        system_pairs = raw_pairs[raw_pairs["system_id"] == system_id]
        for pair_label in figure3d.PAIR_MAP:
            values = system_pairs.loc[
                system_pairs["pair"] == pair_label, paired_column
            ].to_numpy(float)
            ax.scatter(
                system_index + figure3d.PAIR_X_OFFSETS[pair_label],
                values,
                s=70,
                color=figure3d.PAIR_COLORS[pair_label],
                alpha=0.8,
                linewidths=0.1,
                zorder=1,
            )
    ax.errorbar(
        x,
        effect,
        yerr=np.vstack((effect - low, high - effect)),
        fmt="D",
        markersize=0.0,
        color=figure3d.EFFECT_COLOR,
        markerfacecolor="none",
        markeredgecolor=figure3d.EFFECT_COLOR,
        markeredgewidth=0.9,
        elinewidth=1.0,
        capsize=1.0,
        capthick=1.0,
        alpha=0.00,
        zorder=2,
    )
    #ax.axhline(0, color="#8c8c8c", linewidth=0.9, zorder=0)
    ax.text(
        0.045,
        0.94,
        figure3d.REACTION_LABELS[reaction],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x, [figure3d.SYSTEM_DISPLAY[system] for system in figure3d.SYSTEM_ORDER])
    ax.tick_params(pad=2.0)


def draw_figure(
    table: pd.DataFrame,
    paired_table: pd.DataFrame,
    output_base: Path,
    y_label: str = r"$\Delta\Delta G_{\mathrm{i}}$ (kcal/mol)",
    effect_column: str = "effect_boundary_minus_inner",
    paired_column: str = "paired_boundary_minus_inner",
) -> None:
    """Draw all four reaction effects in one full-linewidth row."""
    y_limits = figure3d.shared_y_limits(table, paired_table, paired_column)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FULL_LINEWIDTH_IN, FIGURE_HEIGHT_IN),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    for axis, reaction in zip(axes, figure3d.REACTION_ORDER, strict=True):
        draw_panel(
            axis, table, paired_table, reaction, y_limits, effect_column, paired_column
        )
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    fig.text(0.012, 0.55, y_label, rotation=90, va="center", fontsize=8)
    fig.text(0.52, 0.035, "System", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.20, top=0.97)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def draw_pair_legend(output_base: Path) -> None:
    """Save a larger, separately placeable a/b/c/d position legend."""
    fig, ax = plt.subplots(figsize=(2.80, 0.50))
    ax.set_axis_off()
    handles = [
        Line2D([], [], marker="o", linestyle="", color=figure3d.PAIR_COLORS[pair], markersize=6)
        for pair in figure3d.PAIR_MAP
    ]
    fig.legend(
        handles,
        list(figure3d.PAIR_MAP),
        title="Pair",
        loc="center",
        frameon=False,
        ncol=4,
        fontsize=7,
        title_fontsize=7,
        handletextpad=0.35,
        columnspacing=0.80,
        borderaxespad=0.0,
    )
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    effects = pd.read_csv(EFFECTS_CSV, dtype={"system_id": str})
    paired_effects = pd.read_csv(PAIRED_EFFECTS_CSV, dtype={"system_id": str})
    draw_figure(
        effects,
        paired_effects,
        OUTPUT_DIR / "figure3d_boundary_minus_inner_effects_1x4",
    )
    draw_pair_legend(OUTPUT_DIR / "figure3d_pair_legend_1x4")


if __name__ == "__main__":
    main()
