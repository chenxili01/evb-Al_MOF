"""Plot CO2-adsorption reaction free-energy contributions for the eight sites.

This is the Figure 4d counterpart to ``plot_figure_4c.py``.  It uses reaction1
(CO2 adsorption) rather than reaction4 (CO desorption), while retaining one
site-resolved value and one eight-site bootstrap mean/CI per effect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_figure_4c as figure4c


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONTRIBUTIONS_CSV = DATA_DIR / "figure4d_co2_adsorption_site_contributions.csv"
SUMMARY_CSV = DATA_DIR / "figure4d_co2_adsorption_bootstrap_summary.csv"
CO2_ADSORPTION = "reaction1"


def load_co2_adsorption_data() -> pd.DataFrame:
    """Load and validate all 32 site/system CO2-adsorption records."""
    table = pd.read_csv(ENERGY_CSV)
    table["system_id"] = table["system"].str.extract(r"sys(\d{2})", expand=False)
    table["site_id"] = table["site"].str.extract(r"site(\d{2})", expand=False)
    table = table[
        table["site_id"].isin(figure4c.SITE_ORDER)
        & table["system_id"].isin(figure4c.SYSTEM_ORDER)
        & table["reaction"].eq(CO2_ADSORPTION)
    ].copy()
    expected_rows = len(figure4c.SITE_ORDER) * len(figure4c.SYSTEM_ORDER)
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} CO2-adsorption records, found {len(table)}.")
    if table.duplicated(["site_id", "system_id"]).any():
        raise ValueError("Duplicate site/system CO2-adsorption records in the energy input.")
    return table.sort_values(["site_id", "system_id"])


def draw_dotplots(
    contributions: pd.DataFrame, summary: pd.DataFrame, output_base: Path
) -> None:
    """Draw four shared-y CO2-adsorption environmental-contribution dot plots."""
    y_limits = figure4c.shared_y_limits(contributions, summary)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

    for axis, (contribution, label, _, _, _) in zip(axes, figure4c.CONTRIBUTIONS):
        points = (
            contributions[contributions["contribution"].eq(contribution)]
            .set_index("site_id")
            .reindex(figure4c.SITE_ORDER)
            .reset_index()
        )
        effect_summary = summary.set_index("contribution").loc[contribution]
        x_positions = np.arange(len(figure4c.SITE_ORDER))
        for region in ("Inner", "Boundary"):
            mask = points["site_region"].eq(region)
            axis.scatter(
                x_positions[mask], points.loc[mask, "delta_delta_g"],
                s=120, color=figure4c.SITE_COLORS[region], alpha=0.9,
                edgecolor="black", linewidth=0.3, zorder=3,
            )
        mean = float(effect_summary["mean"])
        ci_low = float(effect_summary["ci95_low"])
        ci_high = float(effect_summary["ci95_high"])
        axis.errorbar(
            figure4c.SUMMARY_X, mean, yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D", color="black", markerfacecolor="black", markeredgecolor="black",
            markersize=7.0, elinewidth=1.1, capsize=2.5, capthick=1.1, zorder=4,
        )
        axis.axhline(0.0, color="#6e6e6e", linewidth=0.8, zorder=1)
        axis.axvline(3.5, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
        axis.set_title(label, pad=7)
        axis.set_xlim(-0.55, 8.78)
        axis.set_ylim(*y_limits)
        axis.set_xticks(list(x_positions) + [figure4c.SUMMARY_X])
        axis.set_xticklabels(
            [figure4c.SITE_LABELS[site] for site in figure4c.SITE_ORDER] + ["Mean"]
        )
        axis.tick_params(axis="x", pad=2)

    axes[0].set_ylabel(r"CO$_2$ adsorption $\Delta\Delta G$ (kcal mol$^{-1}$)")
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.18, top=0.87, wspace=0.14)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    contributions = pd.read_csv(CONTRIBUTIONS_CSV, dtype={"site_id": str})
    summary = pd.read_csv(SUMMARY_CSV)
    draw_dotplots(contributions, summary, OUTPUT_DIR / "figure4d_co2_adsorption_environment_contributions")


if __name__ == "__main__":
    main()
