"""Generate Figure 3d: direct boundary-minus-inner reaction-energy effects.

Each matched-pair effect is DeltaDeltaG_Ba - DeltaDeltaG_Ia (and equivalently
for b, c, and d). The 95% interval bootstraps these four paired effects; it
describes site-to-site sampling uncertainty, not FEP uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EFFECTS_CSV = DATA_DIR / "figure3d_boundary_minus_inner_effects.csv"
PAIRED_EFFECTS_CSV = DATA_DIR / "figure3d_paired_boundary_layer_effects.csv"


REACTION_ORDER = ["reaction1", "reaction2", "reaction3", "reaction4"]
REACTION_LABELS = {
    "reaction1": r"$\mathrm{CO_2}$ adsorption",
    "reaction2": r"$^{*}\mathrm{COOH}$ formation",
    "reaction3": "C-OH cleavage",
    "reaction4": "CO desorption",
}
SYSTEM_ORDER = ["01", "02", "03", "04"]
SYSTEM_DISPLAY = {"01": "I", "02": "II", "03": "III", "04": "IV"}
BOOTSTRAP_SAMPLES = 20_000
EFFECT_COLOR = "#4d4d4d"
#PAIR_COLORS = {
#    "Ba-Ia": "#4d4d4d",
#    "Bb-Ib": "#1b9e77",
#    "Bc-Ic": "#d95f02",
#    "Bd-Id": "#7570b3",
#}

PAIR_COLORS = {
    "Ba-Ia": "#0000f7",
    "Bb-Ib": "#f4331e",
    "Bc-Ic": "#ffb700",
    "Bd-Id": "#00a878",
}
PAIR_X_OFFSETS = {"Ba-Ia": -0.15, "Bb-Ib": -0.05, "Bc-Ic": 0.05, "Bd-Id": 0.15}
PAIR_MAP = {
    "Ba-Ia": ("05", "01"),
    "Bb-Ib": ("06", "02"),
    "Bc-Ic": ("07", "03"),
    "Bd-Id": ("08", "04"),
}


def configure_pairs(pair_map: dict[str, tuple[str, str]]) -> None:
    """Configure a four-pair direct-effect figure using the Figure 3b palette."""
    global PAIR_MAP, PAIR_COLORS, PAIR_X_OFFSETS
    PAIR_MAP = pair_map
    #colors = ("#4d4d4d", "#1b9e77", "#d95f02", "#7570b3")
    #colors = ["#009E73", "#E69F00", "#7B61A8", "#00A6D6"]
    #colors = ["#173c91", "#ffb700", "#B1FFF3", "#FD5972"]
    colors = [
    "#0000f7",  # vivid blue
    "#f4331e",  # vivid red
    "#ffb700",  # golden yellow
    "#00a878",  # teal green
]
    offsets = (-0.15, -0.05, 0.05, 0.15)
    PAIR_COLORS = dict(zip(pair_map, colors, strict=True))
    PAIR_X_OFFSETS = dict(zip(pair_map, offsets, strict=True))

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def bootstrap_effect(paired_effects: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Return the paired mean effect and its percentile 95% bootstrap interval."""
    effect = float(paired_effects.mean())
    rng = np.random.default_rng(seed)
    bootstrap_means = rng.choice(
        paired_effects, size=(BOOTSTRAP_SAMPLES, len(paired_effects)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return effect, float(low), float(high)


def build_effect_tables(
    effect_column: str = "effect_boundary_minus_inner",
    paired_column: str = "paired_boundary_minus_inner",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate matched a/b/c/d boundary-layer effects for every step/system."""
    data = load_merged_data()
    data = data.copy()
    data["site_id"] = data["site_id"].astype(str)
    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for reaction_index, reaction in enumerate(REACTION_ORDER):
        for system_index, system_id in enumerate(SYSTEM_ORDER):
            subset = data[
                (data["reaction"].astype(str) == reaction)
                & (data["system_id"].astype(str) == system_id)
            ]
            paired_effects: list[float] = []
            for pair_label, (inner_site, boundary_site) in PAIR_MAP.items():
                first_values = subset.loc[subset["site_id"] == inner_site, "delta_reaction"].to_numpy(float)
                second_values = subset.loc[subset["site_id"] == boundary_site, "delta_reaction"].to_numpy(float)
                if len(first_values) != 1 or len(second_values) != 1:
                    raise ValueError(f"Missing matched {pair_label} pair for {reaction}/{system_id}")
                pair_effect = float(first_values[0] - second_values[0])
                paired_effects.append(pair_effect)
                first_label, second_label = (
                    pair_label.split("-", 1) if "-" in pair_label else (f"B{pair_label}", f"I{pair_label}")
                )
                pair_rows.append(
                    {
                        "reaction": reaction,
                        "system_id": system_id,
                        "system_display": SYSTEM_DISPLAY[system_id],
                        "pair": pair_label,
                        "site_left": first_label,
                        "site_right": second_label,
                        paired_column: pair_effect,
                    }
                )
            effect, ci_low, ci_high = bootstrap_effect(
                np.asarray(paired_effects), seed=10_000 + 10 * reaction_index + system_index
            )
            summary_rows.append(
                {
                    "reaction": reaction,
                    "system_id": system_id,
                    "system_display": SYSTEM_DISPLAY[system_id],
                    effect_column: effect,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_matched_pairs": len(paired_effects),
                    "interval_method": "paired-site percentile bootstrap, 20000 resamples",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def shared_y_limits(
    table: pd.DataFrame, paired_table: pd.DataFrame, paired_column: str
) -> tuple[float, float]:
    """Return one symmetric, rounded range encompassing all effects and intervals."""
    interval_extent = np.abs(table[["ci95_low", "ci95_high"]].to_numpy(float)).max()
    paired_extent = np.abs(paired_table[paired_column].to_numpy(float)).max()
    extent = max(float(interval_extent), float(paired_extent), 2.0)
    limit = 2.0 * np.ceil((1.08 * extent) / 2.0)
    return -limit, limit


def draw_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    paired_table: pd.DataFrame,
    reaction: str,
    y_limits: tuple[float, float],
    effect_column: str,
    paired_column: str,
) -> None:
    subset = table[table["reaction"] == reaction].set_index("system_id").reindex(SYSTEM_ORDER)
    x = np.arange(len(SYSTEM_ORDER))
    effect = subset[effect_column].to_numpy(float)
    low = subset["ci95_low"].to_numpy(float)
    high = subset["ci95_high"].to_numpy(float)
    raw_pairs = paired_table[paired_table["reaction"] == reaction]
    for system_index, system_id in enumerate(SYSTEM_ORDER):
        system_pairs = raw_pairs[raw_pairs["system_id"] == system_id]
        for pair_label in PAIR_MAP:
            values = system_pairs.loc[
                system_pairs["pair"] == pair_label, paired_column
            ].to_numpy(float)
            ax.scatter(
                system_index + PAIR_X_OFFSETS[pair_label], values,
                s=16, color=PAIR_COLORS[pair_label], alpha=0.68, linewidths=0, zorder=1,
            )
    ax.errorbar(
        x, effect, yerr=np.vstack((effect - low, high - effect)), fmt="D",
        markersize=2.7, color=EFFECT_COLOR, markerfacecolor="none",
        markeredgecolor=EFFECT_COLOR, markeredgewidth=0.65, elinewidth=0.75,
        capsize=1.2, capthick=0.75, alpha=0.76, zorder=2,
    )
    ax.axhline(0, color="#8c8c8c", linewidth=0.7, zorder=0)
    ax.text(
        0.045, 0.94, REACTION_LABELS[reaction], transform=ax.transAxes,
        ha="left", va="top", fontsize=5.5,
    )
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x, [SYSTEM_DISPLAY[system] for system in SYSTEM_ORDER])
    ax.tick_params(pad=1.5)


def draw_figure(
    table: pd.DataFrame,
    paired_table: pd.DataFrame,
    output_base: Path,
    y_label: str = r"$\Delta_{\mathrm{B-I}}$ (kcal/mol)",
    effect_column: str = "effect_boundary_minus_inner",
    paired_column: str = "paired_boundary_minus_inner",
) -> None:
    y_limits = shared_y_limits(table, paired_table, paired_column)
    fig, axes = plt.subplots(
        2, 2, figsize=(3.05, 3.00), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.10, "hspace": 0.12},
    )
    axes = axes.ravel()
    for axis, reaction in zip(axes, REACTION_ORDER, strict=True):
        draw_panel(axis, table, paired_table, reaction, y_limits, effect_column, paired_column)
    for axis in (axes[0], axes[1]):
        axis.tick_params(labelbottom=False)
    fig.text(
        0.035, 0.49, y_label,
        rotation=90, va="center", fontsize=7,
    )
    fig.text(0.52, 0.045, "System", ha="center", fontsize=7)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.16, top=0.97)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def draw_pair_legend(output_base: Path) -> None:
    """Save the a/b/c/d position legend as a separately placeable figure."""
    fig, ax = plt.subplots(figsize=(2.40, 0.42))
    ax.set_axis_off()
    handles = [
        Line2D([], [], marker="o", linestyle="", color=PAIR_COLORS[pair], markersize=4)
        for pair in PAIR_MAP
    ]
    fig.legend(
        handles, list(PAIR_MAP), title="Pair", loc="center", frameon=False, ncol=4,
        fontsize=5.5, title_fontsize=5.5, handletextpad=0.25,
        columnspacing=0.65, borderaxespad=0.0,
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
        OUTPUT_DIR / "figure3d_boundary_minus_inner_effects",
        y_label=r"$\Delta_{\mathrm{a}}$ (kcal/mol)",
    )
    draw_pair_legend(OUTPUT_DIR / "figure3d_pair_legend")


if __name__ == "__main__":
    main()
