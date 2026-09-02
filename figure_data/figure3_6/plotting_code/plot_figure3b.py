"""Generate Figure 3b site-resolved energetic-shift heatmaps.

The input is the authoritative Figure2 site-energy table.  Each cell is an
EVB-minus-molecular-reference free-energy shift in kcal/mol; negative values
therefore indicate framework stabilization (a more favourable step).
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = DATA_DIR / "figure3b_site_resolved_shifts_table.csv"

SITE_ORDER = [f"{index:02d}" for index in range(1, 9)]
SITE_LABELS = {
    "01": "Ia", "02": "Ib", "03": "Ic", "04": "Id",
    "05": "Ba", "06": "Bb", "07": "Bc", "08": "Bd",
}
REACTION_ORDER = ["reaction1", "reaction2", "reaction3", "reaction4"]
REACTION_LABELS = {
    "reaction1": r"CO$_2$ adsorption",
    "reaction2": r"$^{*}$COOH formation",
    "reaction3": "C-OH cleavage",
    "reaction4": "CO desorption",
}
SYSTEM_ORDER = ["01", "02", "03", "04"]
SYSTEM_DISPLAY_LABELS = {"01": "I", "02": "II", "03": "III", "04": "IV"}
SYSTEM_LABELS = {
    "01": "water",
    "02": "electrolyte",
    "03": "electrolyte + electrode",
    "04": "electrolyte + electrode + electric field",
}
# One fixed scale for both quantities preserves direct physical comparison.
SHIFT_LIMIT = 10.0
REACTION_LABEL_WRAP_WIDTH = 12

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
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
        "legend.fontsize": 6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _system_id(series: pd.Series) -> pd.Series:
    return series.str.extract(r"sys(\d{2})", expand=False)


def load_shifts() -> pd.DataFrame:
    """Load Figure2 energy data and calculate EVB-minus-reference shifts."""
    table = pd.read_csv(ENERGY_CSV)
    table["system_id"] = _system_id(table["system"])
    table["site_id"] = table["site"].str.extract(r"site(\d{2})", expand=False)
    table["delta_activation"] = (
        table["activation_barrier"] - table["dft_reference_activation_barrier"]
    )
    table["delta_reaction"] = (
        table["reaction_energy"] - table["dft_reference_reaction_energy"]
    )
    required = len(SITE_ORDER) * len(SYSTEM_ORDER) * len(REACTION_ORDER)
    table = table[
        table["site_id"].isin(SITE_ORDER)
        & table["system_id"].isin(SYSTEM_ORDER)
        & table["reaction"].isin(REACTION_ORDER)
    ].copy()
    if len(table) != required:
        raise ValueError(f"Expected {required} site/system/reaction rows, found {len(table)}")
    if table.duplicated(["site_id", "system_id", "reaction"]).any():
        raise ValueError("Duplicate site/system/reaction records in the Figure2 input")
    return table


def make_matrix(table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [REACTION_ORDER, SYSTEM_ORDER], names=["reaction", "system"]
    )
    matrix = table.pivot(index="site_id", columns=["reaction", "system_id"], values=value_column)
    matrix = matrix.reindex(index=SITE_ORDER, columns=columns)
    if matrix.isna().any().any():
        raise ValueError(f"Missing values while building {value_column} heatmap")
    return matrix


def draw_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    quantity_label: str,
    show_system_labels: bool,
):
    """Draw one compact heatmap for Figure 3b's standalone main-figure panel."""
    cmap = plt.get_cmap("bwr").copy()
    norm = TwoSlopeNorm(vmin=-SHIFT_LIMIT, vcenter=0.0, vmax=SHIFT_LIMIT)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")

    ax.set_yticks(range(len(SITE_ORDER)), [SITE_LABELS[site] for site in SITE_ORDER])
    ax.set_xticks(
        range(len(matrix.columns)),
        [SYSTEM_DISPLAY_LABELS[system] for _, system in matrix.columns],
    )
    ax.tick_params(
        top=False, bottom=show_system_labels, labeltop=False,
        labelbottom=show_system_labels, length=0, pad=1,
    )
    ax.set_ylabel(quantity_label, fontsize=7, labelpad=15)

    ax.axhline(3.5, color="black", linewidth=0.8)
    for boundary in (3.5, 7.5, 11.5):
        ax.axvline(boundary, color="black", linewidth=0.7)
    return image


def draw_combined_heatmaps(
    activation: pd.DataFrame, reaction: pd.DataFrame, output_base: Path
) -> None:
    """Draw the standalone b panel for later placement in a two-panel figure."""
    fig, axes = plt.subplots(
        2, 1, figsize=(3.05, 3.00), sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    image = draw_panel(
        axes[0], activation, r"$\Delta\Delta G^{\ddagger}$", False
    )
    draw_panel(
        axes[1], reaction, r"$\Delta\Delta G$", True
    )
    axes[1].set_xlabel("System", labelpad=3)

    for reaction_index, reaction_name in enumerate(REACTION_ORDER):
        center = reaction_index * len(SYSTEM_ORDER) + 1.5
        axes[0].text(
            center, -0.96,
            textwrap.fill(
                REACTION_LABELS[reaction_name], width=REACTION_LABEL_WRAP_WIDTH
            ),
            ha="center", va="bottom", fontsize=6,
        )

    colorbar = fig.colorbar(image, ax=axes, pad=0.018, fraction=0.045, ticks=[-10, 0, 10])
    colorbar.ax.tick_params(labelsize=5.5, length=2, pad=1)
    #colorbar.set_label(r"$\Delta\Delta G$ (kcal mol$^{-1}$)", fontsize=6, labelpad=2)
    colorbar.set_label(r"Free-energy shift (kcal mol$^{-1}$)", fontsize=6, labelpad=2)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.86, bottom=0.12)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table = pd.read_csv(DATA_CSV, dtype={"system_id": str, "site_id": str})
    draw_combined_heatmaps(
        make_matrix(table, "delta_activation"),
        make_matrix(table, "delta_reaction"),
        OUTPUT_DIR / "figure3b_site_resolved_energetic_shifts",
    )


if __name__ == "__main__":
    main()
