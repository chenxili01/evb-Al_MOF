"""Plot CO-desorption reaction free-energy contributions for the eight sites.

Each point is one site-resolved reaction free-energy shift (DeltaDeltaG) for
CO desorption.  The black diamond in each panel is the mean across all eight
sites; its vertical error bar is a non-parametric 95% bootstrap confidence
interval for that one overall mean.
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
CONTRIBUTIONS_CSV = DATA_DIR / "figure4c_co_desorption_site_contributions.csv"
SUMMARY_CSV = DATA_DIR / "figure4c_co_desorption_bootstrap_summary.csv"

SITE_ORDER = [f"{index:02d}" for index in range(1, 9)]
SITE_LABELS = {
    "01": "Ia", "02": "Ib", "03": "Ic", "04": "Id",
    "05": "Ba", "06": "Bb", "07": "Bc", "08": "Bd",
}
SYSTEM_ORDER = ["01", "02", "03", "04"]
CO_DESORPTION = "reaction4"
CONTRIBUTIONS = [
    ("mof_embedding", "MOF embedding", "01", None, "system I - molecular reference"),
    ("electrolyte", "Electrolyte", "02", "01", "system II - system I"),
    ("electrode", "Electrode", "03", "02", "system III - system II"),
    ("electric_field", "Electric field", "04", "03", "system IV - system III"),
]
SITE_COLORS = {"Inner": "#173c91", "Boundary": "#ffb700"}
N_BOOTSTRAP = 10_000
SUMMARY_X = 8.35

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 10,
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


def load_co_desorption_data() -> pd.DataFrame:
    """Load and validate all 32 site/system CO-desorption records."""
    table = pd.read_csv(ENERGY_CSV)
    table["system_id"] = table["system"].str.extract(r"sys(\d{2})", expand=False)
    table["site_id"] = table["site"].str.extract(r"site(\d{2})", expand=False)
    table = table[
        table["site_id"].isin(SITE_ORDER)
        & table["system_id"].isin(SYSTEM_ORDER)
        & table["reaction"].eq(CO_DESORPTION)
    ].copy()
    expected_rows = len(SITE_ORDER) * len(SYSTEM_ORDER)
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} CO-desorption records, found {len(table)}.")
    if table.duplicated(["site_id", "system_id"]).any():
        raise ValueError("Duplicate site/system CO-desorption records in the energy input.")
    return table.sort_values(["site_id", "system_id"])


def calculate_contributions(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate the four sequential reaction-free-energy contributions."""
    values = table.pivot(index="site_id", columns="system_id", values="reaction_energy")
    references = table.pivot(
        index="site_id", columns="system_id", values="dft_reference_reaction_energy"
    )
    values = values.reindex(index=SITE_ORDER, columns=SYSTEM_ORDER)
    references = references.reindex(index=SITE_ORDER, columns=SYSTEM_ORDER)
    if values.isna().any().any() or references.isna().any().any():
        raise ValueError("Missing CO-desorption reaction-energy values after arranging sites.")

    records: list[pd.DataFrame] = []
    for contribution, _, to_system, from_system, definition in CONTRIBUTIONS:
        if from_system is None:
            result = values[to_system] - references[to_system]
        else:
            result = values[to_system] - values[from_system]
        records.append(
            result.rename("delta_delta_g")
            .rename_axis("site_id")
            .reset_index()
            .assign(
                site_label=lambda frame: frame["site_id"].map(SITE_LABELS),
                site_region=lambda frame: np.where(
                    frame["site_id"].astype(int) <= 4, "Inner", "Boundary"
                ),
                contribution=contribution,
                contribution_definition=definition,
                from_system=from_system if from_system is not None else "molecular_reference",
                to_system=to_system,
            )
        )
    return pd.concat(records, ignore_index=True)[
        [
            "contribution", "contribution_definition", "from_system", "to_system",
            "site_id", "site_label", "site_region", "delta_delta_g",
        ]
    ]


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Return the mean and percentile 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    bootstrap_means = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(values.mean()), float(lower), float(upper)


def build_summary(contributions: pd.DataFrame) -> pd.DataFrame:
    """Calculate one eight-site mean and 95% CI for each environmental effect."""
    records = []
    for index, (contribution, _, _, _, definition) in enumerate(CONTRIBUTIONS):
        values = contributions.loc[
            contributions["contribution"].eq(contribution), "delta_delta_g"
        ].to_numpy(dtype=float)
        if len(values) != len(SITE_ORDER):
            raise ValueError(f"Expected eight {contribution} site values, found {len(values)}.")
        mean, ci_low, ci_high = bootstrap_mean_ci(values, seed=41_000 + index)
        records.append(
            {
                "contribution": contribution,
                "contribution_definition": definition,
                "mean": mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_sites": len(values),
                "bootstrap_replicates": N_BOOTSTRAP,
            }
        )
    return pd.DataFrame(records)


def shared_y_limits(contributions: pd.DataFrame, summary: pd.DataFrame) -> tuple[float, float]:
    """Set one y range that includes every point, CI, and the zero reference."""
    lower = min(float(contributions["delta_delta_g"].min()), float(summary["ci95_low"].min()), 0.0)
    upper = max(float(contributions["delta_delta_g"].max()), float(summary["ci95_high"].max()), 0.0)
    span = max(upper - lower, 4.0)
    padding = 0.08 * span
    return lower - padding, upper + padding


def draw_dotplots(contributions: pd.DataFrame, summary: pd.DataFrame, output_base: Path) -> None:
    """Draw four shared-y CO-desorption environmental-contribution dot plots."""
    y_limits = shared_y_limits(contributions, summary)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

    for axis, (contribution, label, _, _, _) in zip(axes, CONTRIBUTIONS):
        points = (
            contributions[contributions["contribution"].eq(contribution)]
            .set_index("site_id")
            .reindex(SITE_ORDER)
            .reset_index()
        )
        effect_summary = summary.set_index("contribution").loc[contribution]
        x_positions = np.arange(len(SITE_ORDER))
        for region in ("Inner", "Boundary"):
            mask = points["site_region"].eq(region)
            axis.scatter(
                x_positions[mask], points.loc[mask, "delta_delta_g"],
                s=120, color=SITE_COLORS[region], alpha=0.9,
                edgecolor="black", linewidth=0.3, zorder=3,
            )
        mean = float(effect_summary["mean"])
        ci_low = float(effect_summary["ci95_low"])
        ci_high = float(effect_summary["ci95_high"])
        axis.errorbar(
            SUMMARY_X, mean, yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D", color="black", markerfacecolor="black", markeredgecolor="black",
            markersize=7.0, elinewidth=1.1, capsize=2.5, capthick=1.1, zorder=4,
        )
        axis.axhline(0.0, color="#6e6e6e", linewidth=0.8, zorder=1)
        axis.axvline(3.5, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
        axis.set_title(label, pad=7)
        axis.set_xlim(-0.55, 8.78)
        axis.set_ylim(*y_limits)
        axis.set_xticks(list(x_positions) + [SUMMARY_X])
        axis.set_xticklabels([SITE_LABELS[site] for site in SITE_ORDER] + ["Mean"])
        axis.tick_params(axis="x", pad=2)

    axes[0].set_ylabel(r"CO desorption $\Delta\Delta G$ (kcal mol$^{-1}$)")
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.18, top=0.87, wspace=0.14)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    contributions = pd.read_csv(CONTRIBUTIONS_CSV, dtype={"site_id": str})
    summary = pd.read_csv(SUMMARY_CSV)
    draw_dotplots(contributions, summary, OUTPUT_DIR / "figure4c_co_desorption_environment_contributions")


if __name__ == "__main__":
    main()
