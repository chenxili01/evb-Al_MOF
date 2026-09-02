"""Plot Figure 4b environmental free-energy contributions across eight sites.

The authoritative input is the site-resolved free-energy table used for
Figure 3.  For each site and reaction step, the four sequential environmental
contributions are calculated as

* MOF embedding: system I - molecular reference;
* electrolyte: system II - system I;
* electrode: system III - system II; and
* electric field: system IV - system III.

The heatmaps show the mean contribution over the eight sites.  Parenthetical
values are the corresponding sample standard deviations (ddof=1), i.e. the
site-to-site spread rather than an uncertainty on the mean.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SITE_CONTRIBUTIONS_CSV = DATA_DIR / "figure4b_site_environment_contributions.csv"
SUMMARY_CSV = DATA_DIR / "figure4b_environment_contribution_summary.csv"

SITE_ORDER = [f"{index:02d}" for index in range(1, 9)]
SYSTEM_ORDER = ["01", "02", "03", "04"]
REACTION_ORDER = ["reaction1", "reaction2", "reaction3", "reaction4"]
REACTION_LABELS = [
    "CO$_2$\nadsorption",
    "$^{*}$COOH\nformation",
    "C-OH\ncleavage",
    "CO\ndesorption",
]
CONTRIBUTIONS = [
    ("mof_embedding", "MOF embedding", "01", None, "system I - molecular reference"),
    ("electrolyte", "Electrolyte", "02", "01", "system II - system I"),
    ("electrode", "Electrode", "03", "02", "system III - system II"),
    ("electric_field", "Electric field", "04", "03", "system IV - system III"),
]
ENERGY_TYPES = [
    (
        "activation",
        r"$\Delta\Delta G^{\ddagger}$",
        "activation_barrier",
        "dft_reference_activation_barrier",
    ),
    ("reaction", r"$\Delta\Delta G$", "reaction_energy", "dft_reference_reaction_energy"),
]

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.labelsize": 10,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_energy_data() -> pd.DataFrame:
    """Read and validate the complete 8-site, 4-system, 4-step data set."""
    table = pd.read_csv(ENERGY_CSV)
    table["system_id"] = table["system"].str.extract(r"sys(\d{2})", expand=False)
    table["site_id"] = table["site"].str.extract(r"site(\d{2})", expand=False)
    table = table[
        table["site_id"].isin(SITE_ORDER)
        & table["system_id"].isin(SYSTEM_ORDER)
        & table["reaction"].isin(REACTION_ORDER)
    ].copy()
    expected_rows = len(SITE_ORDER) * len(SYSTEM_ORDER) * len(REACTION_ORDER)
    if len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} site/system/reaction records in {ENERGY_CSV}, "
            f"found {len(table)}."
        )
    if table.duplicated(["site_id", "system_id", "reaction"]).any():
        raise ValueError("Duplicate site/system/reaction records in the energy input.")
    return table.sort_values(["site_id", "reaction", "system_id"])


def calculate_site_contributions(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate all four sequential contributions for both energy types."""
    index = pd.MultiIndex.from_product(
        [SITE_ORDER, REACTION_ORDER], names=["site_id", "reaction"]
    )
    records: list[pd.DataFrame] = []

    for energy_type, _, energy_column, reference_column in ENERGY_TYPES:
        values = table.pivot(
            index=["site_id", "reaction"], columns="system_id", values=energy_column
        ).reindex(index=index, columns=SYSTEM_ORDER)
        references = table.pivot(
            index=["site_id", "reaction"],
            columns="system_id",
            values=reference_column,
        ).reindex(index=index, columns=SYSTEM_ORDER)
        if values.isna().any().any() or references.isna().any().any():
            raise ValueError(f"Missing {energy_type} values after arranging site records.")

        for contribution_key, _, to_system, from_system, definition in CONTRIBUTIONS:
            if from_system is None:
                contribution = values[to_system] - references[to_system]
            else:
                contribution = values[to_system] - values[from_system]
            records.append(
                contribution.rename("value")
                .reset_index()
                .assign(
                    energy_type=energy_type,
                    contribution=contribution_key,
                    contribution_definition=definition,
                    from_system=from_system if from_system is not None else "molecular_reference",
                    to_system=to_system,
                )
            )

    return pd.concat(records, ignore_index=True)[
        [
            "energy_type",
            "contribution",
            "contribution_definition",
            "from_system",
            "to_system",
            "site_id",
            "reaction",
            "value",
        ]
    ]


def summarise_contributions(site_contributions: pd.DataFrame) -> pd.DataFrame:
    """Return eight-site means and site-to-site sample standard deviations."""
    summary = (
        site_contributions.groupby(
            ["energy_type", "contribution", "reaction"], observed=True
        )["value"]
        .agg(mean="mean", site_to_site_sd="std", n_sites="count")
        .reset_index()
    )
    summary["contribution"] = pd.Categorical(
        summary["contribution"], [item[0] for item in CONTRIBUTIONS], ordered=True
    )
    summary["reaction"] = pd.Categorical(summary["reaction"], REACTION_ORDER, ordered=True)
    summary = summary.sort_values(["energy_type", "contribution", "reaction"])
    if not summary["n_sites"].eq(len(SITE_ORDER)).all():
        raise ValueError("Every heatmap cell must contain all eight sites.")
    return summary


def _heatmap_matrix(summary: pd.DataFrame, energy_type: str, value_column: str) -> pd.DataFrame:
    return (
        summary[summary["energy_type"] == energy_type]
        .pivot(index="contribution", columns="reaction", values=value_column)
        .reindex(
            index=[item[0] for item in CONTRIBUTIONS], columns=REACTION_ORDER
        )
    )


def _shared_limit(summary: pd.DataFrame) -> float:
    """Choose a readable symmetric limit enclosing both quantities."""
    maximum = float(np.abs(summary["mean"]).max())
    return max(1.0, np.ceil(maximum * 2.0) / 2.0)


def draw_heatmaps(summary: pd.DataFrame, output_base: Path) -> None:
    """Draw stacked annotated 4 x 4 heatmaps with one shared colour scale."""
    limit = _shared_limit(summary)
    cmap = plt.get_cmap("bwr").copy()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(
        2, 1, figsize=(6.30, 5.00), sharex=True, gridspec_kw={"hspace": 0.08}
    )
    image = None
    for axis, (energy_type, quantity_label, _, _) in zip(axes, ENERGY_TYPES):
        means = _heatmap_matrix(summary, energy_type, "mean")
        deviations = _heatmap_matrix(summary, energy_type, "site_to_site_sd")
        image = axis.imshow(
            means.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto"
        )
        axis.set_yticks(range(len(CONTRIBUTIONS)))
        axis.set_yticklabels([item[1] for item in CONTRIBUTIONS])
        axis.set_ylabel(quantity_label, labelpad=12, fontsize=11)
        axis.tick_params(axis="both", length=0, pad=3)

        for row in range(means.shape[0]):
            for column in range(means.shape[1]):
                mean = means.iloc[row, column]
                sd = deviations.iloc[row, column]
                # Keep labels legible on both dark ends of the diverging scale.
                text_color = "white" if abs(mean) > 0.62 * limit else "black"
                axis.text(
                    column,
                    row,
                    f"{mean:.1f}\n({sd:.1f})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    linespacing=1.10,
                )
        axis.set_xticks(np.arange(-0.5, len(REACTION_ORDER), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(CONTRIBUTIONS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.1)
        axis.tick_params(which="minor", bottom=False, left=False)

    axes[-1].set_xticks(range(len(REACTION_ORDER)))
    axes[-1].set_xticklabels(REACTION_LABELS)
    # A dedicated axis keeps the shared range bar centred over the full figure,
    # rather than attached to either heatmap, and gives it additional clearance.
    colorbar_axis = fig.add_axes([0.91, 0.20, 0.025, 0.70])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"Free-energy shift (kcal mol$^{-1}$)", labelpad=8, fontsize=12)
    colorbar.ax.tick_params(labelsize=10, length=3, pad=3)
    fig.subplots_adjust(left=0.24, right=0.85, bottom=0.14, top=0.98)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    site_contributions = pd.read_csv(SITE_CONTRIBUTIONS_CSV, dtype={"site_id": str})
    summary = pd.read_csv(SUMMARY_CSV)
    draw_heatmaps(summary, OUTPUT_DIR / "figure4b_environment_contribution_heatmaps")


if __name__ == "__main__":
    main()
