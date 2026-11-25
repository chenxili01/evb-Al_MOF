"""Fig. 2 source data, derived from evb_tables/ (run scripts/build_evb_tables.py first).

    python figures/fig2/make_source_data.py

a  site-resolved shifts vs the molecular DFT reference, 8 sites x 4 steps x Systems I-IV (ΔΔG‡ and ΔΔG)
b  mean environmental contribution over the 8 sites, site-to-site SD (ddof = 1) in parentheses
c  CO2 adsorption: ΔG(boundary) − ΔG(inner) per matched pair and system
d  CO2 adsorption: per-site decomposition of the reaction-free-energy shift into the four contributions
All values kcal/mol.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TAB = HERE.parents[1] / "evb_tables"
OUT = HERE / "source_data"
OUT.mkdir(exist_ok=True)
SYS = {"01": "I", "02": "II", "03": "III", "04": "IV"}

t = pd.read_csv(TAB / "site_energies.csv", dtype={"system_id": str, "site_id": str})
label = {f"{i:02d}": l for i, l in enumerate(["Ia", "Ib", "Ic", "Id", "Ba", "Bb", "Bc", "Bd"], 1)}
a = t.assign(system=t.system_id.map(SYS), site_label=t.site_id.map(label))[
    ["reaction", "system", "site_id", "site_label", "delta_activation", "delta_reaction"]]
a.rename(columns={"delta_activation": "ddG_act_vs_reference", "delta_reaction": "ddG_rxn_vs_reference"}).to_csv(
    OUT / "fig2a_site_shifts.csv", index=False, float_format="%.4f")

s = pd.read_csv(TAB / "environment_contributions_summary.csv")
s.to_csv(OUT / "fig2b_contribution_mean_sd.csv", index=False, float_format="%.4f")

p = pd.read_csv(TAB / "paired_boundary_minus_inner.csv", dtype={"system_id": str})
p[p.reaction == "reaction1"][["system_display", "pair", "boundary_minus_inner_reaction"]].rename(
    columns={"system_display": "system", "boundary_minus_inner_reaction": "ddG_boundary_minus_inner"}).to_csv(
    OUT / "fig2c_co2_adsorption_pairs.csv", index=False, float_format="%.4f")

c = pd.read_csv(TAB / "environment_contributions_site.csv", dtype={"site_id": str})
c[(c.energy_type == "reaction") & (c.reaction == "reaction1")][
    ["contribution", "contribution_definition", "site_id", "site_label", "layer", "value"]].rename(
    columns={"value": "ddG"}).to_csv(OUT / "fig2d_co2_adsorption_contributions.csv", index=False, float_format="%.4f")
print("wrote", *sorted(p.name for p in OUT.glob("*.csv")))
