#!/usr/bin/env python3
"""Figures from data/processed (run scripts/process.py first).

  python scripts/plot.py                 # all figures -> figures/  (PDF + PNG)
  python scripts/plot.py --only scores   # one figure: scores, classes, trend, replicates, maps, sites3d, morphology

Every figure reads only data/processed/*.csv and the constants in common.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (C_FAMILY, C_NEU, FAMILY, FIG, GREY, INK, INK2, LABEL, LIM, ORDER, PROC, REPS, SPAN, THRESH,
                    ddE_cmap)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150})
DD = r"$\Delta\Delta E_\mathrm{ads}$ vs inner paper site (kcal mol$^{-1}$)"
NLAB = "Crystal score N  (0 = paper inner site, 1 = paper boundary site)"
C_N = "#2166ac"
ORIENT = {"as_simulated": "As simulated (CO$_2$ on $-x$ face)", "lower_energy_face": "Lower-energy face"}
CMAP, NORM = ddE_cmap(), Normalize(-LIM, LIM)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def load():
    s = pd.read_csv(PROC / "sites_symmetrized.csv")
    r = pd.read_csv(PROC / "crystal_scores.csv")
    cs = pd.read_csv(PROC / "class_summary.csv")
    return s, r, cs


def anchor_lines(ax, axis="x", text=True):
    """Paper anchors: inner (N = 0), boundary (N = 1), CoTCPP molecule."""
    an = pd.read_csv(PROC / "anchors.csv")
    mol = float(an[an.kind == "molecule"].N.iloc[0])
    line = ax.axvline if axis == "x" else ax.axhline
    for v, lab, ls in [(0, "inner", "-"), (1, "boundary", "-"), (mol, "molecule", ":")]:
        line(v, color=INK2, lw=0.6, ls=ls, zorder=1)
        if text and axis == "x":
            ax.text(v, 0.0, f" {lab}", transform=ax.get_xaxis_transform(), rotation=90, va="bottom", ha="right",
                    fontsize=5.8, color=INK2)
    return mol


def ypos():
    """y positions with a gap between families (top to bottom in ORDER)."""
    y, out, prev = 0.0, {}, None
    for m in ORDER:
        if prev and FAMILY[m] != FAMILY[prev]:
            y += 0.7
        out[m] = y; y += 1; prev = m
    return out


# ---- 1. crystal score per crystal ------------------------------------------------------------
def fig_scores(s, r, cs):
    yp = ypos(); r = r.set_index("model")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    anchor_lines(ax)
    for m in ORDER:
        q = r.loc[m]
        ax.barh(yp[m], q.N, height=0.7, color=C_FAMILY[q.family], alpha=0.85, lw=0)
        ax.errorbar(q.N, yp[m], xerr=q.N_sem_reps, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.scatter([q[f"N_{k}"] for k in REPS], [yp[m]] * 3, s=4, color=INK, lw=0, zorder=4)
        if q.N >= 0:
            ax.text(q.N + q.N_sem_reps + 0.03, yp[m], f"{q.N:.2f}", va="center", fontsize=6, color=INK)
        else:
            ax.text(0.03, yp[m], f"{q.N:.2f}", va="center", ha="left", fontsize=6, color=INK)
    ax.set_yticks([yp[m] for m in ORDER], [LABEL[m] for m in ORDER]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.8, 1.25); ax.set_xlabel(NLAB)
    top = ax.secondary_xaxis("top", functions=(lambda n: n * SPAN, lambda k: k / SPAN))
    top.set_xlabel("Mean ΔΔE over all sites vs inner paper site (kcal/mol)", fontsize=7)
    fig.text(0.0, -0.03, "N = plain average over all Co sites, lower-energy face. Bar = mean of 3 replicas, error bar = SEM, "
             "dots = single replicas.", fontsize=6, color=INK2)
    fig.tight_layout(); save(fig, "fig1_crystal_scores")


# ---- 2. class means ------------------------------------------------------------------------
def fig_classes(s, r, cs):
    faces = [("out", "Out face (CO$_2$ into solvent)"), ("none", "Interior along channel"), ("in", "In face (CO$_2$ into crystal)")]
    sides = {0: ("Interior channel", "o"), 1: ("Side channel", "s"), 2: ("Edge channel", "^")}
    side_c = {0: "#4d4d4d", 1: "#2c7fb8", 2: "#e6550d"}
    xs = {m: i for i, m in enumerate(ORDER)}
    for orient in ORIENT:
        c = cs[cs.orientation == orient]
        fig, axs = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
        for ax, (face, title) in zip(axs, faces):
            ax.axhline(0, color=INK2, lw=0.5); ax.axhline(SPAN, color=INK2, lw=0.5, ls="--")
            for side, (lab, mk) in sides.items():
                g = c[(c.x_face == face) & (c.side == side)]
                ax.errorbar([xs[m] + (side - 1) * 0.2 for m in g.model], g.ddE_mean, yerr=g.ddE_sem_reps, fmt=mk,
                            ms=4, color=side_c[side], mec="white", mew=0.5, elinewidth=0.8, capsize=0, label=lab)
            if face == "out":  # slab_t1: single-linker channels are labelled "both"
                g = c[c.x_face == "both"]
                ax.errorbar([xs[m] + (sd - 1) * 0.2 for m, sd in zip(g.model, g.side)], g.ddE_mean, yerr=g.ddE_sem_reps,
                            fmt="D", ms=3.5, color="#999999", mec="white", mew=0.5, elinewidth=0.8, label="Slab 1 layer (both faces)")
            ax.set_title(title, loc="left"); ax.set_ylabel("ΔΔE (kcal/mol)")
            for b in [3.5, 8.5, 13.5]:
                ax.axvline(b, color=INK2, lw=0.4, ls=":")
        axs[0].legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.42))
        axs[-1].set_xticks(range(len(ORDER)), [LABEL[m] for m in ORDER], rotation=40, ha="right")
        fig.text(0.99, 0.005, f"Mean over sites; error bar = SEM of the 3 replicate means. Solid line: inner paper site (0), dashed: boundary paper site ({SPAN:.1f}). {ORIENT[orient]}.",
                 ha="right", fontsize=6.5, color=INK2)
        fig.tight_layout()
        save(fig, f"fig2_class_means_{orient}")


JITTER = {"slab_t4": 0.97, "part_LW0.3": 1.0, "aniso_H3": 1.05, "aniso_H5": 1.09, "aniso_H7": 1.13,
          "aniso_W9": 1.17, "aniso_H9": 1.21,  # spread the 4-layer models (log axis)
          "thin_L2_W3": 1.05, "plate_L2_3x7": 1.10, "thin_L2_W5": 1.15, "thin_L3_W3": 1.05, "thin_L3_W5": 1.10}


# ---- 3. crystal score vs layers ---------------------------------------------------------------
def fig_trend(s, r, cs):
    from process import model_N
    fit = pd.read_csv(PROC / "score_fit.csv").iloc[0]
    coef = [fit.a, fit.b, fit.c, fit.d]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    anchor_lines(ax, axis="y")
    an = pd.read_csv(PROC / "anchors.csv")
    for v, t in [(0, "inner paper site"), (1, "boundary paper site"), (float(an[an.kind == "molecule"].N.iloc[0]), "CoTCPP molecule")]:
        ax.text(29, v, t, va="bottom", ha="right", fontsize=5.8, color=INK2)
    Ls = np.geomspace(3, 24, 100)
    for W, ls in [(3, "-"), (5, "--"), (7, ":")]:
        ax.plot(Ls, model_N(Ls, np.full_like(Ls, W), coef), color=INK2, lw=0.8, ls=ls, label=f"equation, W = {W}")
        ax.plot([2], model_N([2.0], [W], coef), marker="_", ms=10, color=INK2)
    d = r.assign(xp=[q.layers * JITTER.get(q.model, 1.0) for q in r.itertuples()])
    for fam, g in d.groupby("family"):
        ax.errorbar(g.xp, g.N, yerr=g.N_sem_reps, fmt="o", ms=4.5, color=C_FAMILY[fam], mec="white", mew=0.5,
                    elinewidth=0.7, label=fam, zorder=3)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 4, 7, 14, 21], ["1", "2", "3", "4", "7", "14", "21"]); ax.set_xlim(0.85, 30)
    ax.set_xlabel("Layers along the channel (linkers per channel)"); ax.set_ylabel("Crystal score N")
    ax.legend(frameon=True, framealpha=1, edgecolor="none", fontsize=5.8, loc="center left", bbox_to_anchor=(0.02, 0.2), ncol=2)
    fig.text(0.0, -0.03, "Error bar = SEM over 3 replicas. Lines: fitted equation (≥ 2 layers) for W = 3, 5, 7 channels; "
             "4-layer points spread sideways.", fontsize=6, color=INK2)
    fig.tight_layout(); save(fig, "fig3_score_vs_layers")


# ---- 4. replicate agreement ------------------------------------------------------------------------
def fig_replicates(s, r, cs):
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True, sharey=True)
    v = s[[f"dd_{r}" for r in REPS]].to_numpy()
    lo, hi = np.floor(v.min()) - 1, np.ceil(v.max()) + 1
    for ax, (i, j) in zip(axs, pairs):
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.5)
        ax.scatter(v[:, i], v[:, j], s=3, color="#3b6ea5", alpha=0.35, lw=0)
        r = np.corrcoef(v[:, i], v[:, j])[0, 1]
        ax.text(0.04, 0.92, f"r = {r:.2f}\nn = {len(v)}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel(f"ΔΔE, {REPS[i]} (kcal/mol)"); ax.set_ylabel(f"ΔΔE, {REPS[j]}")
        ax.set_aspect("equal"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    fig.tight_layout(); save(fig, "fig4_replicate_agreement")


# ---- 5. per-model site maps ------------------------------------------------------------------------
def channel_order(g):
    """Channels sorted edge -> side -> interior, then by (iy, iz); returns list of (iy, iz)."""
    ch = g.groupby(["iy", "iz"]).side.first().reset_index()
    return [tuple(x) for x in ch.sort_values(["side", "iy", "iz"], ascending=[False, True, True])[["iy", "iz"]].to_numpy()]


def fig_maps(s, r, cs):
    out = "fig5_site_maps"
    for m in ORDER:
        g = s[s.model == m]
        L, chans = int(g.col_len.iloc[0]), channel_order(g)
        ny, nz = g.iy.max() + 1, g.iz.max() + 1
        fig = plt.figure(figsize=(7.0, 4.6))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.55, wspace=0.45)
        # top: unrolled map, rows = channels, columns = position along channel (out -> in)
        for k, (col, title) in enumerate([("ddE_rel", ORIENT["as_simulated"]), ("dd_best", ORIENT["lower_energy_face"])]):
            ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
            M = np.full((len(chans), L), np.nan)
            idx = {c: i for i, c in enumerate(chans)}
            for q in g.itertuples():
                M[idx[(q.iy, q.iz)], q.n_from_out] = getattr(q, col)
            ax.imshow(M, cmap=CMAP, norm=NORM, aspect="auto", interpolation="nearest")
            ax.set_title(title, loc="left"); ax.set_xlabel("Position along channel (0 = out face)")
            sides = [g[(g.iy == a) & (g.iz == b)].side.iloc[0] for a, b in chans]
            ax.set_ylabel("Channel"); ax.set_yticks([])
            for i in range(1, len(sides)):
                if sides[i] != sides[i - 1]:
                    ax.axhline(i - 0.5, color="white", lw=1)
            ax.set_xticks(range(0, L, max(1, L // 7)))
        # bottom: y-z maps of the out-face layer, the in-face layer, interior mean, best face of the outer layer
        panels = [("Out-face layer", g[g.n_from_out == 0], "ddE_rel"),
                  ("In-face layer", g[g.n_from_in == 0], "ddE_rel"),
                  ("Interior mean", g[(g.n_from_out > 0) & (g.n_from_in > 0)], "ddE_rel"),
                  ("Best face, outer layer", g[g.n_from_out == 0], "dd_best")]
        for k, (title, sub, col) in enumerate(panels):
            ax = fig.add_subplot(gs[1, k])
            ax.set_title(title, loc="left", fontsize=7.5)
            if sub.empty:
                ax.text(0.5, 0.5, "no interior\nsites", ha="center", va="center", transform=ax.transAxes, color=INK2)
                ax.set_axis_off(); continue
            M = sub.groupby(["iz", "iy"])[col].mean().unstack().reindex(index=range(nz), columns=range(ny)).to_numpy()
            ax.imshow(M, cmap=CMAP, norm=NORM, origin="lower", interpolation="nearest",
                      extent=[-0.5 * 1.67, (ny - 0.5) * 1.67, -0.5 * 1.67, (nz - 0.5) * 1.67])
            ax.set_xlabel("y (nm)"); ax.set_ylabel("z (nm)" if k == 0 else "")
        cax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax)
        cb.set_label(DD); cb.ax.axhline(SPAN, color=INK, lw=0.6)
        q = r.set_index("model").loc[m]
        fig.suptitle(f"{LABEL[m]}  ({len(g)} sites, {L} layers × {len(chans)} channels)   crystal score N = {q.N:.2f} ± {q.N_sem_reps:.2f}",
                     x=0.06, ha="left", y=0.99, fontsize=9)
        fig.text(0.06, 0.005, f"Grey: |ΔΔE| ≤ {GREY:g}; line on colour bar: boundary paper site ({SPAN:.1f}, N = 1). "
                 "Top rows: edge, side, interior channels (white lines).", fontsize=6, color=INK2)
        (FIG / out).mkdir(parents=True, exist_ok=True)
        save(fig, f"{out}/{m}")


# ---- 6. 3D site scatter --------------------------------------------------------------------------
def fig_sites3d(s, r, cs):
    out = "fig6_sites_3d"
    for m in ORDER:
        g = s[s.model == m]
        fig = plt.figure(figsize=(7.0, 3.4))
        for k, (col, title) in enumerate([("ddE_rel", ORIENT["as_simulated"]), ("dd_best", ORIENT["lower_energy_face"])]):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            grey = g[col].abs() <= GREY
            ax.scatter(g.x[grey], g.y[grey], g.z[grey], s=6, color=C_NEU, alpha=0.35, lw=0)
            sc = g[~grey].sort_values(col, key=np.abs)
            ax.scatter(sc.x, sc.y, sc.z, s=14, c=sc[col], cmap=CMAP, norm=NORM, alpha=0.9, lw=0, depthshade=False)
            ext = np.ptp(g[["x", "y", "z"]].to_numpy(), axis=0) + 0.8
            ax.set_box_aspect(ext); ax.view_init(elev=18, azim=-58)
            ax.set_xlabel("x (nm)", labelpad=-6); ax.set_ylabel("y", labelpad=-6); ax.set_zlabel("z", labelpad=-6)
            ax.tick_params(pad=-3, labelsize=5.5); ax.set_title(title, fontsize=7.5)
        cax = fig.add_axes([0.96, 0.2, 0.012, 0.6])
        fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax).set_label(DD)
        fig.suptitle(LABEL[m], x=0.05, ha="left", fontsize=9)
        (FIG / out).mkdir(parents=True, exist_ok=True)
        save(fig, f"{out}/{m}")


# ---- 7. morphology map (publication candidate) ----------------------------------------------------
MARKER = {"Slabs": "s", "Particles": "o", "Anisotropic plates": "D", "Thin crystals": "p"}
X12, X23 = 1.41, 2.5      # region boundaries on the log layer axis: 1 | 2 | >=3 layers


def fig_morphology(s, r, cs):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from process import model_N
    fit = pd.read_csv(PROC / "score_fit.csv").iloc[0]
    coef = [fit.a, fit.b, fit.c, fit.d]
    cmap = LinearSegmentedColormap.from_list("score", ["#f4f8fb", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
    cmap.set_under("#f4a582")
    norm = Normalize(0.4, 1.1)
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    yw = np.linspace(1.7, 8.3, 200)
    z2 = model_N(np.full_like(yw, 2.0), yw, coef)
    ax.imshow(z2[:, None], extent=[X12, X23, yw[0], yw[-1]], origin="lower", aspect="auto", cmap=cmap, norm=norm,
              alpha=0.55, interpolation="bilinear", zorder=0)
    X3, Y3 = np.meshgrid(np.geomspace(X23, 26, 300), yw)
    Z3 = model_N(X3, Y3, coef)
    ax.pcolormesh(X3, Y3, Z3, cmap=cmap, norm=norm, shading="auto", alpha=0.55, rasterized=True)
    cl = ax.contour(X3, Y3, Z3, levels=[0.5, 0.55, 0.6, 0.65, 0.7], colors=INK2, linewidths=0.5, linestyles="--")
    ax.clabel(cl, fmt="%.2f", fontsize=5.5, inline_spacing=2)
    ax.axvspan(0.85, X12, facecolor="none", edgecolor="#bbbbbb", hatch="////", lw=0)
    for xb in (X12, X23):
        ax.axvline(xb, color="#9a9a9a", lw=0.7)
    for x0, x1, t in [(0.85, X12, "1 layer\n(not fitted)"), (X12, X23, "2 layers\nall boundary"), (X23, 26, "≥3 layers: boundary + inner")]:
        ax.text(np.sqrt(x0 * x1), 7.95, t, ha="center", va="center", fontsize=5.6, color=INK2, linespacing=0.95,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
    jit = {"aniso_H3": 0.93, "aniso_H5": 0.97, "aniso_H7": 1.01, "aniso_W9": 1.05, "aniso_H9": 1.09,
           "thin_L2_W3": 0.94, "plate_L2_3x7": 1.06}
    for q in r.itertuples():
        x = q.layers * jit.get(q.model, 1.0)
        ax.scatter(x, q.width_ch, s=52 if q.family == "Thin crystals" else 46, marker=MARKER[q.family],
                   c=[q.N], cmap=cmap, norm=norm, edgecolor=INK, lw=0.6, zorder=4)
        if q.model.startswith("aniso"):
            continue
        dx = {"thin_L2_W3": -7, "plate_L2_3x7": 7}.get(q.model, 0)
        ax.annotate(f"{q.N:.2f}", (x, q.width_ch), xytext=(dx, 7), textcoords="offset points",
                    ha="center", fontsize=5.8, color=INK)
    pl = r[r.family == "Anisotropic plates"]
    ax.annotate(f"plates\n{pl.N.min():.2f}–{pl.N.max():.2f}", (4 * 1.01, 3), xytext=(0, -9),
                textcoords="offset points", ha="center", va="top", fontsize=5.8, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.85, 26); ax.set_ylim(1.7, 8.3)
    ax.set_xticks([1, 2, 3, 4, 7, 14, 21], ["1", "2", "3", "4", "7", "14", "21"])
    ax.set_yticks([3, 5, 7], [f"{w}  ({w * 1.67:.1f} nm)" for w in (3, 5, 7)])
    ax.set_xlabel("Channel length (layers; ×0.72 nm)"); ax.set_ylabel("Cross-section, narrower side (channels)")
    ax.set_title("a  Crystal score N", loc="left")
    hs = [Line2D([], [], marker=MARKER[f], ls="", mfc="#9ecae1", mec=INK, mew=0.6, ms=5, label=f) for f in MARKER]
    ax.legend(handles=hs, loc="center right", bbox_to_anchor=(1.0, 0.6), frameon=True, framealpha=0.9,
              edgecolor="none", fontsize=5.5, handletextpad=0.3, borderpad=0.4)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.015, fraction=0.04, extend="min")
    cb.set_label("N", fontsize=6.5, rotation=0); cb.ax.tick_params(labelsize=6)
    bx = fig.add_subplot(gs[1])
    f = r[r.layers >= 2]
    lo, hi = 0.4, 1.12
    bx.plot([lo, hi], [lo, hi], color=INK2, lw=0.5)
    for q in f.itertuples():
        bx.errorbar(q.N_loo, q.N, yerr=q.N_sem_reps, fmt=MARKER[q.family], ms=4.5, color="#6baed6", mec=INK, mew=0.5,
                    ecolor=INK2, elinewidth=0.6, zorder=3)
    bx.text(0.0, -0.42, f"leave-one-out, n = {int(fit.n)}\nMAE = {fit.LOO_MAE:.3f} ({fit.LOO_MAE_kcal:.2f} kcal/mol)\n"
            f"max = {fit.LOO_max_abs_err:.2f},  Q² = {fit.LOO_Q2:.2f},  ρ = {fit.LOO_spearman:.2f}",
            transform=bx.transAxes, va="top", ha="left", fontsize=5.6, linespacing=1.15)
    bx.set_xlim(lo, hi); bx.set_ylim(lo, hi); bx.set_aspect("equal")
    bx.set_xlabel("predicted N, crystal left out"); bx.set_ylabel("simulated N")
    bx.set_title("b  Leave-one-out test", loc="left")
    one = r[r.layers == 1].iloc[0]
    fig.text(0.0, -0.1, "N = (mean E$_{ax}$ over all sites, lower-energy face − E$_{inner}$) / (E$_{boundary}$ − E$_{inner}$); "
             "E$_{inner}$, E$_{boundary}$ = mean of the four inner / boundary paper sites (−6.22 / −7.31 V/nm; span "
             f"{fit.span_kcal:.2f} kcal/mol).  N = E(L)·(a + b/W + c·δ$_{{L,2}}$) + (1 − E(L))·d,  a = {fit.a:.2f}, b = {fit.b:.2f}, "
             f"c = {fit.c:.2f}, d = {fit.d:.2f};  E(L) = 1, 2/L for L = 2, ≥3 layers; W = narrower cross-section (channels). "
             f"1 layer (no axial neighbour) is its own class: N = {one.N:.2f} (orange).",
             fontsize=5.8, color=INK2, wrap=True)
    save(fig, "fig7_morphology_map")


FIGS = {"scores": fig_scores, "classes": fig_classes, "trend": fig_trend, "replicates": fig_replicates,
        "maps": fig_maps, "sites3d": fig_sites3d, "morphology": fig_morphology}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=list(FIGS), nargs="+")
    a = ap.parse_args()
    data = load()
    for name in a.only or FIGS:
        print(name); FIGS[name](*data)
