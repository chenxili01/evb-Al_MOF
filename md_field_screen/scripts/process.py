#!/usr/bin/env python3
"""Raw per-run field tables -> per-site, symmetrized, class and crystal-score tables + the fitted equation.

  python scripts/process.py            # writes data/processed/*.csv
  python scripts/process.py --frames   # also re-average every site from the per-frame data and check

Inputs (data/raw/):
  <model>/sites.csv                 every Co site: position, lattice index, face/side labels
  <model>/run_XX/axial_field.csv    per product-state site: E_ax (t >= 5 ns, 3 replicas), per-replica means
  <model>/run_XX/axial_field_frames.csv.gz   the per-frame values behind those means
  paper_sites/<site>/axial_field.csv the 8 paper sites + CoTCPP molecule (anchors of N)
Outputs (data/processed/):
  sites_all.csv          one row per site, ddE_rel as simulated (CO2 on the -x face)
  sites_symmetrized.csv  + mirror partner, lower-energy-face dd_best, site score N_site
  class_summary.csv      mean ddE per model x site_class, SEM over the 3 replicas
  anchors.csv            paper sites and molecule: E_ax, ddE_rel, N
  crystal_scores.csv     per crystal: mean over ALL sites (lower-energy face) -> score (kcal/mol) and N
  score_fit.csv          fitted equation for N, leave-one-out statistics
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (BEGIN_PS, E_BOUNDARY, E_INNER, E_MID, E_MOLECULE, E_REF_PAPER, FAMILY, ORDER, PROC, RAW, REPS, S,
                    SPAN, paper_anchors)


def load():
    rows = []
    for m in ORDER:
        sites = pd.read_csv(RAW / m / "sites.csv")
        runs = sorted((RAW / m).glob("run_*"))
        f = pd.concat([pd.read_csv(r / "axial_field.csv").assign(run_dir=r.name) for r in runs], ignore_index=True)
        # every site is product-state in exactly one run
        assert f.resid.is_unique, f"{m}: site in more than one run"
        assert set(f.resid) == set(sites.resid), f"{m}: {len(set(sites.resid) - set(f.resid))} sites without a field"
        d = sites.merge(f, on="resid", how="left", validate="1:1")
        d.insert(0, "family", FAMILY[m]); d.insert(0, "model", m)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def check_frames(f):
    """Recompute the per-replica means from the frames and compare with axial_field.csv."""
    worst = 0.0
    for m in ORDER:
        for r in sorted((RAW / m).glob("run_*")):
            fr = pd.read_csv(r / "axial_field_frames.csv.gz")
            fr = fr[fr.time_ps >= BEGIN_PS]
            per = fr.groupby(["resid", "rep"]).E_ax.mean().unstack("rep")
            ref = pd.read_csv(r / "axial_field.csv").set_index("resid")
            for rep in REPS:
                worst = max(worst, float((per[rep] - ref[f"E_ax_{rep}"]).abs().max()))
            worst = max(worst, float((per.mean(axis=1) - ref.E_ax).abs().max()))
    print(f"frames -> means: max |difference| = {worst:.2e} V/nm (rounding only if < 1e-4)")
    assert worst < 1e-3


def add_dd(f):
    f["ddE_rel"] = S * (f.E_ax - E_MID)
    f["ddE_ref"] = S * (f.E_ax - E_REF_PAPER)
    for r in REPS:
        f[f"dd_{r}"] = S * (f[f"E_ax_{r}"] - E_MID)
    f["dd_sem"] = S * f.E_ax_sem_reps
    return f


def symmetrize(f):
    """Lower-energy face. CO2 can bind on either axial face of Co. In one channel column (iy, iz),
    binding on the +x face of the site with n_from_out = k is the mirror image of the simulated
    (-x) state of the site with n_from_in = k. Take the lower ddE of the two.
    The two interleaved column sets are offset by 0.36 nm along x, so the mirror is exact per
    column only; the difference between the sets is <= 0.23 V/nm (~1 kcal/mol)."""
    cols = ["dd_" + r for r in REPS]
    part = f[["model", "iy", "iz", "n_from_in", "resid", "ddE_rel", "dd_sem", *cols]]
    part = part.rename(columns={"n_from_in": "n_from_out", "resid": "mirror_resid", "ddE_rel": "dd_mirror",
                                "dd_sem": "dd_sem_mirror", **{c: c + "_mirror" for c in cols}})
    d = f.merge(part, on=["model", "iy", "iz", "n_from_out"], how="left", validate="1:1")
    assert d.dd_mirror.notna().all(), "missing mirror partner"
    flip = d.dd_mirror < d.ddE_rel
    d["orient"] = np.where(flip, "+x (mirror)", "-x (as simulated)")
    for a, b in [("dd_best", ("ddE_rel", "dd_mirror")), ("dd_best_sem", ("dd_sem", "dd_sem_mirror")),
                 *[(f"dd_best_{r}", (f"dd_{r}", f"dd_{r}_mirror")) for r in REPS]]:
        d[a] = np.where(flip, d[b[1]], d[b[0]])
    return d


def class_summary(d):
    long = []
    # class means per replicate -> SEM over the 3 replicates (sites inside one replicate are not independent)
    long = []
    for orient, pre, col in [("as_simulated", "dd_", "ddE_rel"), ("lower_energy_face", "dd_best_", "dd_best")]:
        g = d.groupby(["model", "x_face", "side", "site_class"], sort=False)
        per = g[[pre + r for r in REPS]].mean()
        c = pd.DataFrame({"n_sites": g.size(), "E_ax_mean": g.E_ax.mean(), "ddE_mean": per.mean(axis=1),
                          "ddE_sem_reps": per.std(axis=1, ddof=1) / np.sqrt(len(REPS)),
                          "ddE_sd_sites": g[col].std()})
        c["orientation"] = orient
        long.append(c.reset_index())
    return pd.concat(long, ignore_index=True)


CH_NM = 1.67    # channel spacing (y, z), nm
LAYER_NM = 0.72 # linker spacing along a channel (x), nm


def end_share(layers):
    """Share of channel-end (boundary) sites: 0 for 1 layer (no axial neighbour), 1 for 2 layers,
    2/L for L >= 3 (two end layers + inner layers)."""
    L = np.asarray(layers, float)
    return np.where(L < 2, 0.0, np.minimum(1.0, 2.0 / np.maximum(L, 1)))


def design(layers, width_ch):
    """N = E(L) * (a + b/W + c*[L = 2]) + (1 - E(L)) * d,  for L >= 2."""
    L, W = np.asarray(layers, float), np.asarray(width_ch, float)
    E = end_share(L)
    return np.stack([E, E / W, E * (np.abs(L - 2) < 0.5), 1 - E], axis=-1)


def fit_N(layers, width_ch, N):
    return np.linalg.lstsq(design(layers, width_ch), np.asarray(N, float), rcond=None)[0]


def model_N(layers, width_ch, coef):
    return design(layers, width_ch) @ np.asarray(coef)


def to_N(dd):
    """ddE relative to the inner anchor (kcal/mol) -> N (0 = paper inner site, 1 = paper boundary site)."""
    return np.asarray(dd) / SPAN


def crystal_scores(d):
    """Plain average over ALL sites of a crystal, lower-energy face. SEM over the 3 replicas (each replica's
    crystal average; sites inside one replica are not independent)."""
    rows = []
    for m, g in d.groupby("model", sort=False):
        ny, nz, L = g.iy.max() + 1, g.iz.max() + 1, int(g.col_len.iloc[0])
        end = (g.n_from_out == 0) | (g.n_from_in == 0)
        per = np.array([g[f"dd_best_{r}"].mean() for r in REPS])
        score = g.dd_best.mean()
        rows.append(dict(model=m, family=FAMILY[m], layers=L, ny=ny, nz=nz, width_ch=min(ny, nz), n_sites=len(g),
                         L_nm=L * LAYER_NM, W_nm=ny * CH_NM, H_nm=nz * CH_NM, end_share=end.mean(),
                         E_ax_mean=E_MID + score / S, score_kcal=score, score_vs_molecule_kcal=S * (E_MID + score / S - E_MOLECULE),
                         score_sem_reps=per.std(ddof=1) / np.sqrt(len(REPS)), score_sd_sites=g.dd_best.std(),
                         N=to_N(score), **{f"N_{r}": to_N(v) for r, v in zip(REPS, per)},
                         N_sem_reps=abs(per.std(ddof=1) / np.sqrt(len(REPS)) / SPAN),
                         N_end=to_N(g.dd_best[end].mean()),
                         N_interior=to_N(g.dd_best[~end].mean()) if (~end).any() else np.nan))
    r = pd.DataFrame(rows)
    ok = r.layers >= 2                     # 1 layer (no axial neighbour) is its own class, not fitted
    coef = fit_N(r.layers[ok], r.width_ch[ok], r.N[ok])
    r["N_fit"] = np.where(ok, model_N(r.layers, r.width_ch, coef), np.nan)
    loo = np.full(len(r), np.nan)
    idx = np.flatnonzero(ok)
    for i in idx:
        m = idx[idx != i]
        loo[i] = model_N(r.layers.values[i], r.width_ch.values[i], fit_N(r.layers.values[m], r.width_ch.values[m], r.N.values[m]))
    r["N_loo"] = loo
    return r, coef


def fit_stats(r, coef):
    from scipy.stats import spearmanr
    f = r[r.layers >= 2]
    y, loo, fit = f.N.values, f.N_loo.values, f.N_fit.values
    e = y - loo
    return dict(a=coef[0], b=coef[1], c=coef[2], d=coef[3], n=len(f), fit_MAE=np.abs(y - fit).mean(),
                LOO_MAE=np.abs(e).mean(), LOO_max_abs_err=np.abs(e).max(), LOO_RMSE=np.sqrt((e ** 2).mean()),
                LOO_Q2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), LOO_spearman=spearmanr(y, loo)[0],
                LOO_MAE_kcal=abs(SPAN) * np.abs(e).mean(), median_N_sem_reps=f.N_sem_reps.median(),
                E_inner=E_INNER, E_boundary=E_BOUNDARY, span_kcal=SPAN)


def anchors():
    a = paper_anchors()
    a["ddE_rel"] = S * (a.E_ax - E_MID)
    a["N"] = to_N(a.ddE_rel)
    a["N_sem"] = abs(S * a.E_ax_sem_reps / SPAN)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", action="store_true", help="re-average from per-frame data and check")
    a = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    f = load()
    if a.frames:
        check_frames(f)
    print(f"{len(f)} sites in {f.model.nunique()} models; anchors E_inner = {E_INNER:.4f}, E_boundary = {E_BOUNDARY:.4f} "
          f"V/nm (span {SPAN:.3f} kcal/mol), molecule {E_MOLECULE:.4f}")
    f = add_dd(f)
    d = symmetrize(f)
    d["N_site"] = to_N(d.dd_best)
    f.to_csv(PROC / "sites_all.csv", index=False, float_format="%.4f")
    keep = ["model", "family", "resid", "x", "y", "z", "ix", "iy", "iz", "col_len", "n_from_out", "n_from_in",
            "x_face", "side", "site_class", "E_ax", "ddE_rel", "dd_sem", *[f"dd_{r}" for r in REPS],
            "mirror_resid", "dd_mirror", "orient", "dd_best", "dd_best_sem", *[f"dd_best_{r}" for r in REPS], "N_site"]
    d[keep].to_csv(PROC / "sites_symmetrized.csv", index=False, float_format="%.4f")
    class_summary(d).to_csv(PROC / "class_summary.csv", index=False, float_format="%.4f")
    an = anchors()
    an.to_csv(PROC / "anchors.csv", index=False, float_format="%.4f")
    r, coef = crystal_scores(d)
    r.to_csv(PROC / "crystal_scores.csv", index=False, float_format="%.4f")
    st = fit_stats(r, coef)
    pd.DataFrame([st]).to_csv(PROC / "score_fit.csv", index=False, float_format="%.4f")
    print("\nanchors:\n" + an[["site", "kind", "E_ax", "ddE_rel", "N"]].round(3).to_string(index=False))
    print("\nN = E(L)(a + b/W + c[L=2]) + (1 - E(L)) d,  E = 1 | 2/L  for L = 2 | >=3  (1 layer not fitted)")
    print("  " + ", ".join(f"{k} = {v:.3f}" for k, v in st.items()))
    with pd.option_context("display.width", 200):
        print(r[["model", "layers", "width_ch", "n_sites", "E_ax_mean", "score_kcal", "N", "N_sem_reps", "N_end",
                 "N_interior", "N_loo"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
