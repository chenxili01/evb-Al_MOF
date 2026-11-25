#!/usr/bin/env python3
"""Co--C49 bond-midpoint axial field for every product-state (TCC) site of a run.

Same definition as the paper analyzer (tools/analyze_co_c49_axial_field.py,
Zenodo 10.5281/zenodo.21649285, electricfield_Co_C_midpoint_field):
  * field point P = Co--C49 bond midpoint (minimum-image bond)
  * axis u = (C49 - Co)/|C49 - Co| from the same frame; E_ax = E . u
  * sources = every charged atom NOT in the site's own TCC residue whose
    minimum-image distance to C49 is <= 1.0 nm
  * E = -k sum q_j (r_j - P)/|r_j - P|^3, k = 1.439964547 V nm / e
    (positive along Co->C49; a positive charge beyond C49 gives a negative field)
  * force-field charges from the tpr, no dielectric scaling.

Usage
  # trajectories: all replicas of a run, sites from ps_sites.csv
  axial_field_traj.py run --run slab_t2/ps_runs/run_00 --tpr analysis.tpr --begin 5000
  # single snapshot, one or more TCC residues (validation against the paper)
  axial_field_traj.py snap --gro x.gro --tpr x.tpr [--resid N ...]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, minimize_vectors

K_V_NM = 1.439964547
CUTOFF_NM = 1.0
GROUPS = ["SOL", "K", "MOL", "TCC", "AL6", "MOH", "CUT"]


def _name(a):
    return str(a).strip().upper()


class Sites:
    """Index bookkeeping for a set of TCC residues in one universe."""

    def __init__(self, u, resids=None):
        if resids is None:
            res = [r for r in u.residues
                   if {"CO", "C49"} <= {_name(n) for n in r.atoms.names}]
        else:
            byid = {int(r.resid): r for r in u.residues}
            res = [byid[int(i)] for i in resids]
        self.resids = np.array([int(r.resid) for r in res])
        self.resix = np.array([r.ix for r in res])
        self.co, self.c49 = [], []
        for r in res:
            names = np.array([_name(n) for n in r.atoms.names])
            co, c = r.atoms[names == "CO"], r.atoms[names == "C49"]
            if len(co) != 1 or len(c) != 1:
                raise RuntimeError(f"resid {r.resid} ({r.resname}): need one Co and one C49")
            self.co.append(co[0].ix)
            self.c49.append(c[0].ix)
        self.co, self.c49 = np.array(self.co), np.array(self.c49)
        self.q = u.atoms.charges.astype(float)
        self.atom_resix = u.atoms.resindices
        rn = np.array([str(x) for x in u.atoms.resnames])
        self.group = np.full(len(rn), len(GROUPS), dtype=int)       # "other"
        for g, name in enumerate(GROUPS):
            self.group[rn == name] = g


def frame_fields(u, s):
    """Return (E_ax[nsite], E_ax_by_group[nsite, ngroup+1], bond_nm[nsite])."""
    pos = u.atoms.positions                      # Angstrom
    box = u.dimensions
    co, c49 = pos[s.co], pos[s.c49]
    bond = minimize_vectors(c49 - co, box)       # Angstrom
    blen = np.linalg.norm(bond, axis=1)
    axis = bond / blen[:, None]
    mid = co + 0.5 * bond
    pairs = capped_distance(c49, pos, max_cutoff=CUTOFF_NM * 10.0, box=box, return_distances=False)
    i_site, j_atom = pairs[:, 0], pairs[:, 1]
    keep = (s.atom_resix[j_atom] != s.resix[i_site]) & (s.q[j_atom] != 0.0)
    i_site, j_atom = i_site[keep], j_atom[keep]
    r = minimize_vectors(pos[j_atom] - mid[i_site], box) / 10.0   # nm, midpoint -> source
    rn = np.linalg.norm(r, axis=1)
    ok = rn > 1e-12
    i_site, j_atom, r, rn = i_site[ok], j_atom[ok], r[ok], rn[ok]
    e_ax = -K_V_NM * s.q[j_atom] * np.einsum("ij,ij->i", r, axis[i_site]) / rn**3
    n = len(s.resids)
    tot = np.bincount(i_site, weights=e_ax, minlength=n)
    by = np.zeros((n, len(GROUPS) + 1))
    np.add.at(by, (i_site, s.group[j_atom]), e_ax)
    return tot, by, blen / 10.0


def cmd_snap(a):
    u = mda.Universe(a.tpr, a.gro)
    s = Sites(u, a.resid)
    tot, by, bl = frame_fields(u, s)
    df = pd.DataFrame(by, columns=[f"E_{g}" for g in GROUPS] + ["E_other"])
    df.insert(0, "E_ax", tot)
    df.insert(0, "bond_nm", bl)
    df.insert(0, "resid", s.resids)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.10f}"))


def cmd_run(a):
    run = os.path.abspath(a.run)
    ps = pd.read_csv(os.path.join(run, "ps_sites.csv"))
    reps = a.reps or sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, "rep*"))
                            if os.path.exists(os.path.join(p, "md.xtc")))
    if not reps:
        sys.exit(f"no rep*/md.xtc in {run}")
    rows = []
    for rep in reps:
        u = mda.Universe(a.tpr, os.path.join(run, rep, "md.xtc"))
        s = Sites(u, ps.resid.values)
        bad = [int(r) for r, ix in zip(s.resids, s.resix) if u.residues[ix].resname != "TCC"]
        if bad:
            sys.exit(f"resids {bad} are not TCC in {a.tpr}: residue numbering mismatch")
        for ts in u.trajectory:
            if ts.time < a.begin or (a.end is not None and ts.time > a.end):
                continue
            if a.stride_ps and int(round(ts.time)) % int(a.stride_ps):
                continue
            tot, by, bl = frame_fields(u, s)
            for k, rid in enumerate(s.resids):
                rows.append([rep, ts.time, rid, bl[k], tot[k], *by[k]])
        print(f"{rep}: {len({r[1] for r in rows if r[0] == rep})} frames", file=sys.stderr)
    cols = ["rep", "time_ps", "resid", "bond_nm", "E_ax"] + [f"E_{g}" for g in GROUPS] + ["E_other"]
    fr = pd.DataFrame(rows, columns=cols)
    fr.to_csv(os.path.join(run, "axial_field_frames.csv.gz"), index=False, float_format="%.6g")
    ecols = ["E_ax"] + [f"E_{g}" for g in GROUPS] + ["E_other"]
    out = fr.groupby("resid")[ecols + ["bond_nm"]].mean()
    out["E_ax_std"] = fr.groupby("resid")["E_ax"].std()
    rep_means = fr.groupby(["resid", "rep"])["E_ax"].mean().unstack()
    out["E_ax_sem_reps"] = rep_means.std(axis=1, ddof=1) / np.sqrt(rep_means.notna().sum(axis=1))
    out = out.join(rep_means.add_prefix("E_ax_"))
    out["n_frames"] = fr.groupby("resid").size()
    out["bond_max_nm"] = fr.groupby("resid")["bond_nm"].max()
    out.reset_index().to_csv(os.path.join(run, "axial_field.csv"), index=False, float_format="%.5f")
    print(f"wrote {run}/axial_field.csv ({len(out)} sites, reps {reps})", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("snap"); q.add_argument("--gro", required=True); q.add_argument("--tpr", required=True)
    q.add_argument("--resid", type=int, nargs="*")
    q = sp.add_parser("run"); q.add_argument("--run", required=True)
    q.add_argument("--tpr", required=True, help="tpr readable by MDAnalysis with the run's atom order")
    q.add_argument("--reps", nargs="*"); q.add_argument("--begin", type=float, default=5000.0)
    q.add_argument("--end", type=float); q.add_argument("--stride-ps", type=float, default=0)
    a = p.parse_args()
    {"snap": cmd_snap, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
