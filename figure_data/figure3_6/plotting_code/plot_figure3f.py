"""Generate Figure 3f: direct reaction-energy effects along the c axis."""

from pathlib import Path

import pandas as pd

import plot_figure3d as common


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "csv"
OUTPUT_DIR = THIS_DIR.parent / "reproduced_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAIR_MAP = {
    "Ib-Ia": ("02", "01"),
    "Bb-Ba": ("06", "05"),
    "Id-Ic": ("04", "03"),
    "Bd-Bc": ("08", "07"),
}
EFFECT_COLUMN = "effect_electrode_field_direction"
PAIRED_COLUMN = "paired_electrode_field_direction"
EFFECTS_CSV = DATA_DIR / "figure3f_electrode_field_direction_effects.csv"
PAIRED_EFFECTS_CSV = DATA_DIR / "figure3f_paired_electrode_field_direction_effects.csv"


def main() -> None:
    common.configure_pairs(PAIR_MAP)
    effects = pd.read_csv(EFFECTS_CSV, dtype={"system_id": str})
    paired_effects = pd.read_csv(PAIRED_EFFECTS_CSV, dtype={"system_id": str})
    common.draw_figure(
        effects,
        paired_effects,
        OUTPUT_DIR / "figure3f_electrode_field_direction_effects",
        y_label=r"$\Delta_{\mathrm{c}}$ (kcal/mol)",
        effect_column=EFFECT_COLUMN,
        paired_column=PAIRED_COLUMN,
    )
    common.draw_pair_legend(OUTPUT_DIR / "figure3f_pair_legend")


if __name__ == "__main__":
    main()
