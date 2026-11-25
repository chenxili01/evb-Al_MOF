"""Fig. 4 source data (axial electric field).

    python figures/fig4/make_source_data.py

b  DFT response of CoTPP CO2 adsorption to a uniform field along x, y, z      data/gaussian_field_axis_response.csv
   ΔΔE vs field (field sign flipped to the Co->C49 convention, as in the paper); linear slopes per axis.
c  calibrated field prediction vs EVB ΔΔG for the 8 paper sites
     field E_ax  : md_field_screen/data/processed/anchors.csv (Co–C49 midpoint, 3 x 20 ns, t = 5–20 ns)
     ΔΔG_pred    = S (E_ax − E_ref) + offset,  S = 4.576 kcal/mol per V/nm (DFT z response), E_ref = molecule
     offset      = mean(EVB − field term) over the 8 sites = −3.225; the figure uses the rounded −3.2
     EVB ΔΔG     = System I CO2-adsorption ΔG(site) − EVB ΔG of the molecular reference (both EVB-FEP;
                   output_mapping_results ... "reference_mapping")
d  decomposition of E_ax into water + K+, linker (behind Co / beyond CO2) and other, per site and class
   from data/contrib_species.csv (per-species field, 3 replicas).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "source_data"
OUT.mkdir(exist_ok=True)
S, OFFSET_PLOTTED = 4.576, -3.2
SITES = [f"site0{i}" for i in range(1, 9)]
CLS = {s: "inner" if int(s[-1]) <= 4 else "boundary" for s in SITES} | {"reference": "reference"}

# ---------------------------------------------------------------- b
g = pd.read_csv(HERE / "data" / "gaussian_field_axis_response.csv")
g["field_v_nm_co_c49"] = -g.field_strength_v_nm          # sign convention of the paper (plot_gaussian_field_axis_response.py)
g.to_csv(OUT / "fig4b_dft_field_response.csv", index=False)
slopes = []
for ax_, d in g.groupby("axis"):
    fit = stats.linregress(d.field_v_nm_co_c49, d.drxn_kcal_mol)
    two = d[d.field_strength_v_nm.abs().round(3) == 0.514]
    two_pt = (two.drxn_kcal_mol.iloc[1] - two.drxn_kcal_mol.iloc[0]) / (two.field_v_nm_co_c49.iloc[1] - two.field_v_nm_co_c49.iloc[0])
    slopes.append({"axis": ax_, "slope_all_points": fit.slope, "slope_pm_0p514": two_pt, "n_points": len(d)})
pd.DataFrame(slopes).to_csv(OUT / "fig4b_slopes.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------- c
an = pd.read_csv(ROOT / "md_field_screen/data/processed/anchors.csv").set_index("site")
t = pd.read_csv(ROOT / "evb_tables/site_energies.csv", dtype={"system_id": str, "site_id": str})
ref_dg = next(s["reference_mapping"]["dg_kcal"] for s in json.loads(
    (ROOT / "output_mapping_results/four_reaction_mof_environment_effect_summary.json").read_text())["summaries"]
    if s["reference_mapping"] and "reaction1" in json.dumps(s["config"]))
evb = t[(t.reaction == "reaction1") & (t.system_id == "01")].set_index("site_id").reaction_energy - ref_dg
E_ref = an.loc["reference", "E_ax"]
c = pd.DataFrame({"site": SITES, "class": [CLS[s] for s in SITES]})
c["E_ax_V_nm"] = an.loc[SITES, "E_ax"].values
c["E_ax_sem_reps"] = an.loc[SITES, "E_ax_sem_reps"].values
c["field_term_kcal"] = S * (c.E_ax_V_nm - E_ref)
c["evb_ddG_kcal"] = [evb[s[-2:]] for s in SITES]
offset = (c.evb_ddG_kcal - c.field_term_kcal).mean()
c["ddG_pred_kcal_plotted"] = c.field_term_kcal + OFFSET_PLOTTED
c["ddG_pred_kcal_exact_offset"] = c.field_term_kcal + offset
c["ddG_pred_sem"] = S * c.E_ax_sem_reps
oi = c.evb_ddG_kcal - c.field_term_kcal
c["loo_abs_err_kcal"] = [abs(c.evb_ddG_kcal[i] - c.field_term_kcal[i] - oi.drop(i).mean()) for i in range(len(c))]
c.to_csv(OUT / "fig4c_paper_sites.csv", index=False, float_format="%.4f")
r = stats.pearsonr(c.field_term_kcal, c.evb_ddG_kcal)
pd.DataFrame([
    ("S_kcal_per_V_nm", S), ("E_ref_V_nm", E_ref), ("evb_reference_dG_kcal", ref_dg),
    ("offset_exact_kcal", offset), ("offset_plotted_kcal", OFFSET_PLOTTED),
    ("offset_se_kcal", oi.std(ddof=1) / np.sqrt(len(oi))),
    ("pearson_r", r.statistic), ("pearson_p", r.pvalue),
    ("spearman_rho", stats.spearmanr(c.field_term_kcal, c.evb_ddG_kcal).statistic),
    ("loo_mae_kcal", c.loo_abs_err_kcal.mean()), ("loo_max_kcal", c.loo_abs_err_kcal.max()),
    ("mean_E_ax_inner", c[c["class"] == "inner"].E_ax_V_nm.mean()),
    ("mean_E_ax_boundary", c[c["class"] == "boundary"].E_ax_V_nm.mean()),
], columns=["parameter", "value"]).to_csv(OUT / "fig4c_calibration.csv", index=False, float_format="%.5f")

# ---------------------------------------------------------------- d
sp = pd.read_csv(HERE / "data" / "contrib_species.csv")
sp["sys"] = sp.system.str[:6].replace({"refere": "reference"})
E = sp.pivot(index="sys", columns="component", values="E").loc[["reference"] + SITES]
d = pd.DataFrame({
    "class": pd.Series(CLS), "E_total": E.total, "water_K": E["solvent (SOL+K)"], "water": E.SOL, "K": E.K,
    "linker_behind_Co": E["MOL axial back"], "linker_beyond_CO2": E["MOL axial front"], "linker_net": E.MOL,
    "linker_lateral": E["MOL lateral"], "Al_OH_acetate": E.AL6 + E.MOH + E.CUT,
}).loc[["reference"] + SITES]
d.index.name = "site"
d.to_csv(OUT / "fig4d_field_components_V_nm.csv", float_format="%.4f")
cls = d.groupby("class")[["water_K", "linker_net", "linker_behind_Co", "linker_beyond_CO2", "E_total"]].agg(["mean", "std"])
cls.columns = [f"{a}_{b}" for a, b in cls.columns]
cls["bar_total"] = cls.water_K_mean + cls.linker_net_mean     # total shown in panel d (water + K+ + linker)
cls.to_csv(OUT / "fig4d_class_mean_sd_V_nm.csv", float_format="%.4f")
print("wrote", *sorted(p.name for p in OUT.glob("*.csv")))
