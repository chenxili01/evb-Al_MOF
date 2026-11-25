"""Supplementary source data that are not already a main-figure source file.

    python figures/si/make_source_data.py

S2.2 Table noise, S2.3 Figs other-steps / CO-desorption decomposition -> evb_tables/ (scripts/build_evb_tables.py)
S3   Tables field response / field contributions / LOO                -> figures/fig4/source_data
S3   Top-N snapshot projection, residue decomposition tables         -> figures/fig4/data/*.tsv
S4.1 Table crystal scores                                              -> figures/fig5/source_data
Written here:
S4.2 applicability domain: axial field (lower-energy face) of all 1726 screened sites vs the calibration range
S4.3 cutoff comparison: favoured fraction per crystal at 2, 3, 4 kcal/mol below E_MID = −6.499 V/nm and the
     refit of f(L, W) = E(L)(a + b/W + c δ_{L,2}) with leave-one-out
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROC = ROOT / "md_field_screen/data/processed"
OUT = HERE / "source_data"
OUT.mkdir(exist_ok=True)
S, E_INNER, E_MID_OLD = 4.576, -6.219, -6.499

ss = pd.read_csv(PROC / "sites_symmetrized.csv")
sc = pd.read_csv(PROC / "crystal_scores.csv").set_index("model")
E_INNER = pd.read_csv(PROC / "score_fit.csv").E_inner.iloc[0]
ss["E_ax_lower_face"] = E_INNER + ss.dd_best / S

# ---------------------------------------------------------------- S4.2 applicability domain
cal = pd.read_csv(ROOT / "figures/fig4/source_data/fig4d_field_components_V_nm.csv")
lo, hi = cal.E_total.min(), cal.E_total.max()
e = ss.E_ax_lower_face
dom = pd.DataFrame([("n_sites", len(e)), ("calibration_min_V_nm", lo), ("calibration_max_V_nm", hi),
                    ("pct_inside", 100 * ((e >= lo) & (e <= hi)).mean()), ("pct_weaker", 100 * (e > hi).mean()),
                    ("n_weaker", int((e > hi).sum())), ("weaker_models", ";".join(sorted(ss[e > hi].model.unique()))),
                    ("pct_stronger", 100 * (e < lo).mean()), ("min_field_V_nm", e.min()), ("extension_V_nm", lo - e.min())],
                   columns=["quantity", "value"])
dom.to_csv(OUT / "si_domain_summary.csv", index=False)
ss[["model", "resid", "site_class", "E_ax_lower_face"]].to_csv(OUT / "si_domain_sites.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------- S4.3 cutoff comparison
dd_old = S * (ss.E_ax_lower_face - E_MID_OLD)          # shift on the earlier relative scale
shares = {c: (dd_old < -c).groupby(ss.model).mean().reindex(sc.index) for c in (2, 3, 4)}
L, W = sc.layers.values.astype(float), sc.width_ch.values.astype(float)
E = np.where(L < 1.5, 0, np.where(L < 2.5, 1, 2 / L))
X = np.stack([E, E / W, E * (np.abs(L - 2) < 0.5)], 1)
rows = []
for c, y in shares.items():
    y = y.values
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = np.array([X[i] @ np.linalg.lstsq(np.delete(X, i, 0), np.delete(y, i), rcond=None)[0] for i in range(len(y))])
    top4 = sc.layers.iloc[np.argsort(-y, kind="stable")[:4]]
    rows.append({"cutoff_kcal": c, "a": coef[0], "b": coef[1], "c": coef[2],
                 "loo_mae_pp": 100 * np.mean(np.abs(pred - y)),
                 "Q2": 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2),
                 "rho_vs_3": stats.spearmanr(y, shares[3]).statistic,
                 "rho_vs_N": stats.spearmanr(y, sc.N).statistic,
                 "top4_all_two_layer": bool((top4 == 2).all()), "slab_t1_fraction": y[sc.index.get_loc("slab_t1")]})
pd.DataFrame(rows).to_csv(OUT / "si_cutoff_refit.csv", index=False, float_format="%.4f")
pd.DataFrame({f"favoured_fraction_{c}kcal": v for c, v in shares.items()}).assign(
    layers=sc.layers, ny=sc.ny, nz=sc.nz, N=sc.N).to_csv(OUT / "si_cutoff_fractions.csv", float_format="%.4f")
print("wrote", *sorted(p.name for p in OUT.glob("*.csv")))
