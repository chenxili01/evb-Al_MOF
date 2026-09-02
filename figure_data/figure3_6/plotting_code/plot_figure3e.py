"""Generate Figure 3e: direct reaction-energy effects along the b axis."""

from pathlib import Path

import pandas as pd

import plot_figure3d as common


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAIR_MAP = {
    "Ic-Ia": ("03", "01"),
    "Bc-Ba": ("07", "05"),
    "Id-Ib": ("04", "02"),
    "Bd-Bb": ("08", "06"),
}
EFFECT_COLUMN = "effect_same_layer_edge_boundary"
PAIRED_COLUMN = "paired_same_layer_edge_boundary"
EFFECTS_CSV = DATA_DIR / "figure3e_same_layer_edge_boundary_effects.csv"
PAIRED_EFFECTS_CSV = DATA_DIR / "figure3e_paired_same_layer_edge_boundary_effects.csv"


def main() -> None:
    common.configure_pairs(PAIR_MAP)
    effects = pd.read_csv(EFFECTS_CSV, dtype={"system_id": str})
    paired_effects = pd.read_csv(PAIRED_EFFECTS_CSV, dtype={"system_id": str})
    common.draw_figure(
        effects,
        paired_effects,
        OUTPUT_DIR / "figure3e_same_layer_edge_boundary_effects",
        y_label=r"$\Delta_{\mathrm{b}}$ (kcal/mol)",
        effect_column=EFFECT_COLUMN,
        paired_column=PAIRED_COLUMN,
    )
    common.draw_pair_legend(OUTPUT_DIR / "figure3e_pair_legend")


if __name__ == "__main__":
    main()
