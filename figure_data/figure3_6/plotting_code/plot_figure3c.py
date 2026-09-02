"""Generate the standalone Figure 3c panel for a three-panel main figure.

Figure 3c summarizes site-resolved EVB-minus-molecular reaction-energy shifts.
Points are individual sites, while error bars are the standard deviation across
the four inner or four boundary sites; they are not FEP uncertainties.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POINTS_CSV = DATA_DIR / "figure3c_site_points.csv"
SUMMARY_CSV = DATA_DIR / "figure3c_inner_boundary_summary.csv"


REACTION_ORDER = ["reaction1", "reaction2", "reaction3", "reaction4"]
REACTION_LABELS = {
    "reaction1": r"$\mathrm{CO_2}$ adsorption",
    "reaction2": r"$^{*}\mathrm{COOH}$ formation",
    "reaction3": "C-O cleavage",
    "reaction4": "CO desorption",
}
SYSTEM_ORDER = ["01", "02", "03", "04"]
SYSTEM_DISPLAY = {"01": "I", "02": "II", "03": "III", "04": "IV"}
POSITION_SPECS = {
    "inside": {"label": "Inner", "color": "#386cb0", "offset": -0.13},
    "surface": {"label": "Boundary", "color": "#fdb462", "offset": 0.13},
}

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "axes.labelsize": 7,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return individual reaction shifts and inner/boundary mean±SD summaries."""
    data = load_merged_data()
    points = data[
        ["system_id", "site_id", "reaction", "mof_position", "delta_reaction"]
    ].copy()
    points = points[points["reaction"].astype(str).isin(REACTION_ORDER)].copy()
    points["system_id"] = points["system_id"].astype(str)
    points["reaction"] = points["reaction"].astype(str)
    points["site_id"] = points["site_id"].astype(str)
    points = points.sort_values(["reaction", "system_id", "mof_position", "site_id"])

    summary = (
        points.groupby(["reaction", "system_id", "mof_position"], observed=False)["delta_reaction"]
        .agg(mean="mean", sd="std", count="count")
        .reset_index()
    )
    summary["uncertainty"] = "SD across four sites"
    summary = summary.sort_values(["reaction", "system_id", "mof_position"])
    return points, summary


def draw_reaction_panel(ax: plt.Axes, points: pd.DataFrame, summary: pd.DataFrame, reaction: str) -> None:
    rng = np.random.default_rng(42 + REACTION_ORDER.index(reaction))
    reaction_points = points[points["reaction"] == reaction]
    reaction_summary = summary[summary["reaction"] == reaction]
    reaction_values = reaction_points["delta_reaction"].to_numpy(dtype=float)
    y_min = float(np.nanmin(reaction_values))
    y_max = float(np.nanmax(reaction_values))
    if y_max - y_min < 4.0:
        y_mid = 0.5 * (y_min + y_max)
        y_min = y_mid - 2.0
        y_max = y_mid + 2.0
    y_pad = 0.04 * (y_max - y_min)

    for position, spec in POSITION_SPECS.items():
        subset = reaction_points[reaction_points["mof_position"] == position]
        x = subset["system_id"].map({system: index for index, system in enumerate(SYSTEM_ORDER)}).to_numpy(float)
        x += spec["offset"] + rng.uniform(-0.035, 0.035, len(subset))
        ax.scatter(
            x, subset["delta_reaction"], s=22, color=spec["color"], alpha=0.52,
            edgecolor="white", linewidth=0.25, zorder=2, label=spec["label"],
        )

        stats = reaction_summary[reaction_summary["mof_position"] == position]
        stats = stats.set_index("system_id").reindex(SYSTEM_ORDER).reset_index()
        ax.errorbar(
            np.arange(len(SYSTEM_ORDER)) + spec["offset"], stats["mean"], yerr=stats["sd"],
            fmt="D", markersize=4.4, color="black", markerfacecolor=spec["color"],
            markeredgecolor="black", markeredgewidth=0.45, elinewidth=0.8,
            capsize=2.0, capthick=0.8, zorder=3,
        )

    ax.axhline(0, color="#8c8c8c", linewidth=0.7, zorder=0)
    ax.text(
        0.97, 0.06, REACTION_LABELS[reaction], transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8,
    )
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks(range(len(SYSTEM_ORDER)), [SYSTEM_DISPLAY[system] for system in SYSTEM_ORDER])
    ax.tick_params(pad=1.5)


def draw_figure(points: pd.DataFrame, summary: pd.DataFrame, output_base: Path) -> None:
    """Draw a Figure2d-style summary at 75% of the 6.30-inch text width."""
    fig, axes = plt.subplots(
        2, 2, figsize=(4.725, 2.70), sharex=True, sharey=False,
        gridspec_kw={"wspace": 0.08, "hspace": 0.10},
    )
    axes = axes.ravel()
    for index, reaction in enumerate(REACTION_ORDER):
        draw_reaction_panel(axes[index], points, summary, reaction)

    for axis in (axes[0], axes[1]):
        axis.tick_params(labelbottom=False)
    for axis in (axes[1], axes[3]):
        axis.yaxis.tick_right()
        axis.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
    handles, labels = axes[0].get_legend_handles_labels()
    unique_handles = dict(zip(labels, handles))
    fig.legend(
        unique_handles.values(), unique_handles.keys(), title="Site position",
        loc="upper left", bbox_to_anchor=(0.11, 0.99), frameon=False,
        fontsize=6, title_fontsize=7, ncol=2, handletextpad=0.35, columnspacing=0.9,
    )
    fig.text(0.045, 0.49, r"$\Delta\Delta G$ (kcal mol$^{-1}$)", rotation=90, va="center", fontsize=7)
    fig.text(0.52, 0.045, "System", ha="center", fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.92, bottom=0.18, top=0.78)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = pd.read_csv(POINTS_CSV, dtype={"system_id": str, "site_id": str})
    summary = pd.read_csv(SUMMARY_CSV, dtype={"system_id": str})
    draw_figure(points, summary, OUTPUT_DIR / "figure3c_inner_boundary_reaction_shifts")


if __name__ == "__main__":
    main()
