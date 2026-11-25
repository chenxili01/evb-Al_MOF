"""Molecular DFT reference (SI Tables "free-energy components" and "DFT benchmark") -> EVB fit targets.

    python dft/reaction_energies.py

PBE0/def2-SVP/SMD geometries and frequencies, PBE0/def2-TZVP/SMD single points; pH 7.3, −0.7 V vs RHE.
G = E_el + G_corr + ΔG_add (kcal/mol). Checks that the step energies equal the "dft_benchmark" targets in
evbpmf_config_portable.json that the EVB parameters were fitted to.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "dft_species_energies.csv").set_index("species")
assert (abs(d.E_el_kcal + d.G_corr_kcal + d.dG_add_kcal - d.G_kcal) < 0.006).all(), "G != sum of components"
G = d.G_kcal
steps = {  # step: (reactants, TS, products)
    "reaction1": (["CoI_TPP_minus", "CO2"], "TS_R1", ["CoTPP_CO2_minus"]),
    "reaction2": (["CoTPP_CO2_2minus_5H2O", "H2O"], "TS_R2", ["CoTPP_COOH_minus_5H2O", "OH_minus"]),
    "reaction3": (["CoTPP_COOH_minus_5H2O"], "TS_R3", ["CoTPP_CO_OH_minus_5H2O"]),
    "reaction4": (["CoTPP_CO"], "TS_R4", ["CoII_TPP", "CO"]),
}
# TS_R2 has the atoms of the 5-water complex only (the transferring water is part of it), so ΔG‡ is taken
# relative to CoTPP_CO2_2minus_5H2O, as in the SI table.
ts_ref = {"reaction2": ["CoTPP_CO2_2minus_5H2O"]}
target = json.loads((HERE.parent / "evbpmf_config_portable.json").read_text())["dft_benchmark"]
rows = []
for k, (rs, ts, ps) in steps.items():
    dg = G[ps].sum() - G[rs].sum()
    dgt = G[ts] - G[ts_ref.get(k, rs)].sum()
    rows.append((k, round(dg, 2), round(dgt, 2), target[k]["free_energy_kcal"], target[k]["barrier_kcal"]))
out = pd.DataFrame(rows, columns=["reaction", "dG_rxn", "dG_act", "evb_target_dG_rxn", "evb_target_dG_act"])
print(out.to_string(index=False))
assert ((out.dG_rxn - out.evb_target_dG_rxn).abs() < 0.006).all() and ((out.dG_act - out.evb_target_dG_act).abs() < 0.006).all()
print("DFT step energies match the EVB fit targets")

# ---------------------------------------------------------------- SI Fig. DFT profile
c = pd.read_csv(HERE / "electron_transfer.csv").set_index("quantity").value
e_shift = c.F * (c.E_abs_SHE + c.U_vs_RHE - c.nernst_slope * c.pH)
assert abs(e_shift - c.electron_free_energy_shift) < 1e-3
et = G["CoTPP_CO2_2minus"] - G["CoTPP_CO2_minus"] + e_shift           # *CO2(1-) + e- -> *CO2(2-)
s = out.set_index("reaction")
steps_prof = [  # (state order, step energy) after the CoI start state (order 2, G = 0)
    (3, s.loc["reaction1", "dG_act"]), (4, s.loc["reaction1", "dG_rxn"] - s.loc["reaction1", "dG_act"]),
    (5, et), (6, s.loc["reaction2", "dG_act"]), (7, s.loc["reaction2", "dG_rxn"] - s.loc["reaction2", "dG_act"]),
    (8, s.loc["reaction3", "dG_act"]), (9, s.loc["reaction3", "dG_rxn"] - s.loc["reaction3", "dG_act"]),
    (10, s.loc["reaction4", "dG_act"]), (11, s.loc["reaction4", "dG_rxn"] - s.loc["reaction4", "dG_act"]),
]
prof = pd.read_csv(HERE / "dft_profile.csv").set_index("order")
level = 0.0
print(f"\nelectron transfer *CO2(1-) + e- -> *CO2(2-): {et:.2f} kcal/mol")
for k, step in steps_prof:
    level += step
    ok = abs(round(level, 2) - prof.loc[k, "G_profile_kcal"]) < 0.006
    print(f"  {prof.loc[k, 'state']:<14} {level:7.2f}  plotted {prof.loc[k, 'G_profile_kcal']:7.2f}  {'ok' if ok else 'MISMATCH'}")
    assert ok
print("DFT profile levels reproduced (CoII -> CoI step −6.90 is from the experimental redox potential)")
