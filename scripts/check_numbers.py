"""Recompute every data value quoted in the manuscript / SI and compare with the printed value.

    python scripts/check_numbers.py            # exits 1 if any check fails

Each check is (where it is quoted, printed value, value recomputed from the data in this repository).
A check passes when the recomputed value rounds to the printed value at the printed precision.
Run scripts/build_evb_tables.py first (the EVB checks read evb_tables/).
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, float, bool]] = []


def check(where: str, printed: str, value: float) -> None:
    """printed is the number as it appears in the paper (e.g. '-7.7', '0.93'); compared at its precision."""
    p = Decimal(printed.replace("−", "-"))
    q = Decimal(1).scaleb(p.as_tuple().exponent)
    ok = Decimal(repr(float(value))).quantize(q) == p or abs(float(value) - float(p)) <= float(q) / 2 + 1e-9
    RESULTS.append((where, printed, float(value), ok))


# --------------------------------------------------------------------------- EVB (Fig. 2, SI S2)
TAB = ROOT / "evb_tables"
t = pd.read_csv(TAB / "site_energies.csv", dtype={"system_id": str, "site_id": str})
v = t.set_index(["reaction", "system_id", "site_id"])
s = pd.read_csv(TAB / "environment_contributions_summary.csv").set_index(["energy_type", "contribution", "reaction"])
c = pd.read_csv(TAB / "environment_contributions_site.csv", dtype={"site_id": str})
pr = pd.read_csv(TAB / "paired_boundary_minus_inner.csv")
nz = pd.read_csv(TAB / "method_noise.csv").groupby("reaction")[["abs_diff_activation", "abs_diff_reaction"]].max()

# DFT reference (main text, SI Table dft_reaction_energy) as used by the EVB fit
ref = t.groupby("reaction")[["dft_reference_activation_barrier", "dft_reference_reaction_energy"]].first()
check("Main: CO2 activation barrier (DFT)", "15.8", ref.loc["reaction1"].iloc[0])
check("Main: *COOH formation barrier (DFT)", "5.2", ref.loc["reaction2"].iloc[0])
check("Main: CO desorption barrier (DFT)", "3.6", ref.loc["reaction4"].iloc[0])

# Fig. 2b printed cell labels: mean (SD)
FIG2B = {  # (energy_type, contribution) -> printed means / SDs for reaction1..4
    ("activation", "mof_embedding"): ("−4.8 0.2 −0.8 −0.5", "1.4 0.2 0.2 0.7"),
    ("activation", "electrolyte"): ("0.3 0.0 0.1 0.0", "0.6 0.2 0.2 1.0"),
    ("activation", "electrode"): ("−0.3 0.0 0.0 −0.3", "0.7 0.3 0.3 0.8"),
    ("activation", "electric_field"): ("0.6 0.0 0.0 0.2", "0.5 0.3 0.2 0.7"),
    ("reaction", "mof_embedding"): ("−7.7 3.1 3.5 6.0", "3.0 1.4 1.3 1.2"),
    ("reaction", "electrolyte"): ("0.3 0.2 −0.1 −0.1", "0.7 0.4 0.9 1.8"),
    ("reaction", "electrode"): ("−0.5 0.0 −0.3 −0.1", "1.0 0.5 0.9 1.3"),
    ("reaction", "electric_field"): ("0.6 −0.1 0.2 0.0", "0.7 0.5 0.9 1.7"),
}
for (et, con), (means, sds) in FIG2B.items():
    for i, (m_, sd_) in enumerate(zip(means.split(), sds.split()), 1):
        r = s.loc[(et, con, f"reaction{i}")]
        check(f"Fig. 2b {et} {con} step {i} mean", m_, r["mean"])
        check(f"Fig. 2b {et} {con} step {i} SD", sd_, r.site_to_site_sd)

# Results text, Fig. 2
check("Main: CO2 ads. MOF embedding ΔΔG‡ mean", "-4.8", s.loc[("activation", "mof_embedding", "reaction1"), "mean"])
check("Main: CO2 ads. MOF embedding ΔΔG‡ SD", "1.4", s.loc[("activation", "mof_embedding", "reaction1"), "site_to_site_sd"])
check("Main: CO2 ads. MOF embedding ΔΔG mean", "-7.7", s.loc[("reaction", "mof_embedding", "reaction1"), "mean"])
check("Main: CO2 ads. MOF embedding ΔΔG SD", "3.0", s.loc[("reaction", "mof_embedding", "reaction1"), "site_to_site_sd"])
other = s.drop(index="mof_embedding", level="contribution")["mean"].abs().max()
RESULTS.append(("Main: other contributions each within ~1 kcal/mol (max |mean|)", "<1", other, other < 1.0))
for rx, lo, hi in (("reaction1", "-7.2", "-3.4"), ("reaction2", "-3.4", "-1.9"), ("reaction3", "-4.3", "0.4"),
                   ("reaction4", "-2.8", "3.1")):
    d = pr[pr.reaction == rx].boundary_minus_inner_reaction
    check(f"Main/SI S2.3: {rx} boundary−inner ΔG min", lo, d.min())
    check(f"Main/SI S2.3: {rx} boundary−inner ΔG max", hi, d.max())
check("SI S2.3: *COOH boundary−inner mean", "-2.5", pr[pr.reaction == "reaction2"].boundary_minus_inner_reaction.mean())
check("SI S2.3: CO2 boundary−inner mean", "-5.4", pr[pr.reaction == "reaction1"].boundary_minus_inner_reaction.mean())
check("Main: MOF embedding raises *COOH ΔG by", "3.1", s.loc[("reaction", "mof_embedding", "reaction2"), "mean"])
check("Main: MOF embedding raises C–OH ΔG by", "3.5", s.loc[("reaction", "mof_embedding", "reaction3"), "mean"])
check("Main: CO desorption ΔG shift mean", "6.0", s.loc[("reaction", "mof_embedding", "reaction4"), "mean"])
check("Main: CO desorption ΔG shift SD", "1.2", s.loc[("reaction", "mof_embedding", "reaction4"), "site_to_site_sd"])
check("SI S2.3: CO desorption barrier shift mean", "-0.5", s.loc[("activation", "mof_embedding", "reaction4"), "mean"])
check("SI S2.3: CO desorption barrier shift SD", "0.7", s.loc[("activation", "mof_embedding", "reaction4"), "site_to_site_sd"])
f4 = c[(c.energy_type == "reaction") & (c.reaction == "reaction4") & (c.contribution == "electric_field")].value
check("SI S2.3: CO desorption field effect min", "-1.8", f4.min())
check("SI S2.3: CO desorption field effect max", "2.3", f4.max())
check("Main: method uncertainty (CO2 ads. ΔG)", "1.5", nz.loc["reaction1", "abs_diff_reaction"])
check("SI S2.3: method uncertainty *COOH ΔG", "0.7", nz.loc["reaction2", "abs_diff_reaction"])

# SI Table noise (max columns)
for rx, a_, r_ in (("reaction1", "0.59", "1.49"), ("reaction2", "0.26", "0.65"), ("reaction3", "0.37", "0.84"),
                   ("reaction4", "0.87", "2.70")):
    check(f"SI Table noise {rx} max |ΔΔG‡|", a_, nz.loc[rx, "abs_diff_activation"])
    check(f"SI Table noise {rx} max |ΔΔG|", r_, nz.loc[rx, "abs_diff_reaction"])

# --------------------------------------------------------------------------- Fig. 3 (water reorganisation)
F3 = ROOT / "figures/fig3/source_data"
w = pd.read_csv(F3 / "fig3ce_water_bisector_c49_timeseries.csv")
m3 = w.groupby(["dataset", "state"])[["water_bisector_to_tcc_c49_deg", "water_count_6A"]].mean().unstack("state")
dth = m3["water_bisector_to_tcc_c49_deg"].PS - m3["water_bisector_to_tcc_c49_deg"].RS
dnw = m3["water_count_6A"].PS - m3["water_count_6A"].RS
inner = [d for d in dth.index if d[:6] in ("site01", "site02", "site03", "site04")]
bound = [d for d in dth.index if d[:6] in ("site05", "site06", "site07", "site08")]
RESULTS.append(("Main Fig. 3: frames per dataset and state = 501", "501", w.groupby(["dataset", "state"]).size().min(),
                (w.groupby(["dataset", "state"]).size() == 501).all()))
check("Main Fig. 3: boundary θ change, smallest decrease", "-13", dth[bound].max())
check("Main Fig. 3: boundary θ change, largest decrease", "-14", dth[bound].min())
check("Main Fig. 3: inner sites recruit ~3–6 waters (min)", "3", dnw[inner].min())
check("Main Fig. 3: inner sites recruit ~3–6 waters (max)", "6", dnw[inner].max())
RESULTS.append(("Main Fig. 3: boundary sites almost no net water change (max |ΔN| < 0.5)", "<0.5",
                dnw[bound].abs().max(), dnw[bound].abs().max() < 0.5))
dn4 = []
for d in inner:
    t3 = pd.read_csv(F3 / "dN" / f"dN_{d}.csv")
    dn4.append(t3.delta_N_water_PS_minus_RS[(t3.r_A - 3.95).abs().idxmin()])
# text says "about 1.5–2": approximate, compared to the nearest 0.5 (data 1.35–1.87)
RESULTS.append(("Main Fig. 3: inner sites gain about 1.5–2 waters within 4 Å (min, nearest 0.5)", "~1.5",
                min(dn4), round(min(dn4) * 2) / 2 == 1.5))
RESULTS.append(("Main Fig. 3: inner sites gain about 1.5–2 waters within 4 Å (max, nearest 0.5)", "~2",
                max(dn4), round(max(dn4) * 2) / 2 == 2.0))

# --------------------------------------------------------------------------- Fig. 4 (axial field)
F4 = ROOT / "figures/fig4/source_data"
cal = pd.read_csv(F4 / "fig4c_calibration.csv").set_index("parameter").value
sl = pd.read_csv(F4 / "fig4b_slopes.csv").set_index("axis")
cl = pd.read_csv(F4 / "fig4d_class_mean_sd_V_nm.csv").set_index("class")
check("Main/SI: DFT axial (z) slope, 5-point fit", "4.58", sl.loc["z", "slope_all_points"])
check("Main/SI: S = 4.576 (±0.514 V/nm two-point)", "4.576", sl.loc["z", "slope_pm_0p514"])
RESULTS.append(("SI: in-plane slopes |x|,|y| < 0.05", "<0.05", sl.loc[["x", "y"], "slope_all_points"].abs().max(),
                sl.loc[["x", "y"], "slope_all_points"].abs().max() < 0.05))
check("Main: reference field", "-5.80", cal["E_ref_V_nm"])
check("Main: mean field, inner sites", "-6.22", cal["mean_E_ax_inner"])
check("Main: mean field, boundary sites", "-7.31", cal["mean_E_ax_boundary"])
check("Main Fig. 4c: Pearson r", "0.93", cal["pearson_r"])
check("SI: Spearman ρ", "0.76", cal["spearman_rho"])
check("Main/SI: calibration offset δ", "-3.225", cal["offset_exact_kcal"])
check("SI: offset SE", "0.41", cal["offset_se_kcal"])
check("Main Fig. 4c / SI: LOO MAE", "1.13", cal["loo_mae_kcal"])
check("SI: LOO max", "1.73", cal["loo_max_kcal"])
check("Main: linker behind Co (inner)", "-2.85", cl.loc["inner", "linker_behind_Co_mean"])
check("Main: linker behind Co (boundary)", "-2.85", cl.loc["boundary", "linker_behind_Co_mean"])
check("Main: linker beyond CO2 (inner)", "4.29", cl.loc["inner", "linker_beyond_CO2_mean"])
check("Fig. 4d: water+K+ inner mean", "-7.67", cl.loc["inner", "water_K_mean"])
check("Fig. 4d: water+K+ inner SD", "0.18", cl.loc["inner", "water_K_std"])
check("Fig. 4d: water+K+ boundary mean", "-4.47", cl.loc["boundary", "water_K_mean"])
check("Fig. 4d: water+K+ boundary SD", "0.28", cl.loc["boundary", "water_K_std"])
check("Fig. 4d: total (water+K+ + linker) inner", "-6.23", cl.loc["inner", "bar_total"])
check("Fig. 4d: total (water+K+ + linker) boundary", "-7.32", cl.loc["boundary", "bar_total"])
oth = pd.read_csv(F4 / "fig4d_field_components_V_nm.csv").Al_OH_acetate.abs().max()
RESULTS.append(("Main: Al nodes, µ-OH, acetate < 0.03 V/nm", "<0.03", oth, oth < 0.03))

# --------------------------------------------------------------------------- Fig. 5 (crystal score)
sc = pd.read_csv(ROOT / "md_field_screen/data/processed/crystal_scores.csv")
fit = pd.read_csv(ROOT / "md_field_screen/data/processed/score_fit.csv").iloc[0]
an = pd.read_csv(ROOT / "md_field_screen/data/processed/anchors.csv").set_index("site")
sites = pd.read_csv(ROOT / "md_field_screen/data/processed/sites_symmetrized.csv")
check("Main: crystals", "19", len(sc))
check("Main: screened sites", "1726", sc.n_sites.sum())
check("Main: 1 N unit in kcal/mol", "-5.0", fit.span_kcal)
for k, p_ in (("a", "0.66"), ("b", "0.49"), ("c", "0.22"), ("d", "0.43")):
    check(f"Main Fig. 5a: coefficient {k}", p_, fit[k])
check("Main: LOO Q²", "0.96", fit.LOO_Q2)
check("Main: LOO MAE (N)", "0.03", fit.LOO_MAE)
check("Main Methods: LOO MAE (N)", "0.029", fit.LOO_MAE)
check("Main/SI: LOO max", "0.096", fit.LOO_max_abs_err)
check("Main: Spearman ρ", "0.90", fit.LOO_spearman)
check("Main/SI: median replica SEM", "0.025", fit.median_N_sem_reps)
check("Main: E_inner", "-6.22", fit.E_inner)
check("Main: E_boundary", "-7.31", fit.E_boundary)
check("Main: molecule N", "-0.38", an.loc["reference", "N"])
check("Main: single-layer slab N", "-0.68", sc.set_index("model").loc["slab_t1", "N"])
thick = sites.merge(sc[["model", "layers"]], on="model")
end = thick[(thick.layers >= 3)]
is_end = end.n_from_out.eq(0) | end.n_from_in.eq(0) if "n_from_in" in end else None
check("Main: end-layer sites, L ≥ 3 (pooled)", "0.85", end[is_end].N_site.mean())
check("Main: interior sites, L ≥ 3 (pooled)", "0.37", end[~is_end].N_site.mean())
check("Main: two-layer crystals (pooled sites)", "1.00", thick[thick.layers == 2].N_site.mean())

# --------------------------------------------------------------------------- SI S4.2, S4.3
dom = pd.read_csv(ROOT / "figures/si/source_data/si_domain_summary.csv").set_index("quantity").value.astype(str)
check("SI S4.2: % inside calibration range", "96.2", float(dom.pct_inside))
check("SI S4.2: % weak-field side", "2.8", float(dom.pct_weaker))
check("SI S4.2: % strong-field side", "0.9", float(dom.pct_stronger))
check("SI S4.2: range extension (V/nm)", "0.12", float(dom.extension_V_nm))
check("SI S4.2: weak-field sites (all slab_t1)", "49", float(dom.n_weaker))
cr = pd.read_csv(ROOT / "figures/si/source_data/si_cutoff_refit.csv").set_index("cutoff_kcal")
CUT_TEX = {2: "0.64 0.88 0.07 4.5 0.95 0.90", 3: "0.26 1.09 0.25 3.2 0.97 1.00", 4: "0.14 0.47 0.20 4.0 0.88 0.83"}
for k_, row in CUT_TEX.items():
    for name, p_ in zip(("a", "b", "c", "loo_mae_pp", "Q2", "rho_vs_3"), row.split()):
        check(f"SI Table cutoff {k_} kcal: {name}", p_, cr.loc[k_, name])
for k_, p_ in ((2, "0.96"), (3, "0.94"), (4, "0.77")):
    check(f"SI S4.3: ρ(N, fraction) cutoff {k_}", p_, cr.loc[k_, "rho_vs_N"])

# --------------------------------------------------------------------------- report
def main() -> int:
    width = max(len(w) for w, *_ in RESULTS)
    bad = 0
    for where, printed, value, ok in RESULTS:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {where:<{width}}  printed {printed:>7}  data {value:10.4f}")
    print(f"\n{len(RESULTS) - bad}/{len(RESULTS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
