#!/usr/bin/env python3
"""Health check of ps_runs replicas: progress, speed, box shape, crystal integrity,
rotation, gap to the periodic image, and whether CO2 stays bound.

  python -P tools/check_runs.py slab_t2/ps_runs/run_00 [more run dirs ...] > report.csv

Note: AL6/MOH/CUT/MOL/TCC are separate molecules (non-bonded cationic dummy model),
so the crystal is unwrapped by walking <3 A contacts between framework heavy atoms
(Al-O coordination included); integrity/rotation come from the Al subset, Kabsch-fitted
to the first frame. "framework split" = the contact graph falls apart.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, minimize_vectors
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

NBR_A = 3.0     # Angstrom: bonded/coordination contacts between framework heavy atoms


def unwrap(pos, box):
    """Make a connected cluster whole by BFS over minimum-image contacts."""
    pairs = capped_distance(pos, pos, NBR_A, box=box, return_distances=False)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    n = len(pos)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    ncomp, _ = connected_components(g, directed=False)
    order, pred = breadth_first_order(g, 0, directed=False)
    out = pos.copy()
    for i in order[1:]:
        p = pred[i]
        out[i] = out[p] + minimize_vectors(pos[i] - pos[p], box)
    return out, ncomp, len(order) == n


def kabsch(P, Q):
    """Rotation angle (deg) and RMSD (A) of Q fitted onto P."""
    P0, Q0 = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Q0.T @ P0)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    rmsd = np.sqrt(((Q0 @ R - P0) ** 2).sum(1).mean())
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return ang, rmsd


def perf(log):
    if not os.path.exists(log):
        return np.nan
    v = re.findall(r"^Performance:\s+([\d.]+)", open(log).read(), re.M)
    return float(v[-1]) if v else np.nan


def check_rep(run, rep, stride=10):
    d = os.path.join(run, rep)
    row = {"run": os.path.relpath(run), "rep": rep,
           "done": os.path.exists(os.path.join(d, "md.done")),
           "ns_per_day": perf(os.path.join(d, "md.log"))}
    xtc = os.path.join(d, "md.xtc")
    if not os.path.exists(xtc):
        row["status"] = "no md.xtc"
        return row
    u = mda.Universe(os.path.join(d, "md.tpr"), xtc)
    fw = u.select_atoms("resname AL6 MOH CUT MOL TCC")
    fw = fw[fw.masses > 2.5]                       # heavy atoms (+ Al dummies), no H
    is_al = (fw.resnames == "AL6") & (fw.names == "Al")
    co = u.select_atoms("resname TCC and name Co")
    c49 = u.select_atoms("resname TCC and name C49")
    frames = list(range(0, len(u.trajectory), stride)) + [len(u.trajectory) - 1]
    ref, out = None, []
    for f in sorted(set(frames)):
        ts = u.trajectory[f]
        box = ts.dimensions
        pf, ncomp, conn = unwrap(fw.positions, box)
        ext = pf.max(0) - pf.min(0)
        p = pf[is_al]
        bond = np.linalg.norm(minimize_vectors(c49.positions - co.positions, box), axis=1) / 10
        if ref is None:
            ref = p
        ang, rmsd = kabsch(ref, p)
        out.append(dict(t=ts.time, box=box[:3] / 10, gap=(box[:3] - ext) / 10, ang=ang, rmsd=rmsd / 10,
                        ncomp=ncomp, conn=conn, bmax=bond.max(), bmin=bond.min()))
    first, last = out[0], out[-1]
    row.update({
        "t_ns": last["t"] / 1000,
        "box0_nm": " ".join(f"{x:.2f}" for x in first["box"]),
        "box_nm": " ".join(f"{x:.2f}" for x in last["box"]),
        "min_image_gap_nm": min(min(o["gap"]) for o in out),      # box length minus crystal extent
        "max_rot_deg": max(o["ang"] for o in out),
        "max_Al_rmsd_nm": max(o["rmsd"] for o in out),
        "Al_graph_split": max(o["ncomp"] for o in out) > 1,
        "CoC49_min_nm": min(o["bmin"] for o in out),
        "CoC49_max_nm": max(o["bmax"] for o in out),
        "status": "ok",
    })
    flags = []
    if row["min_image_gap_nm"] < 2.0: flags.append("image gap < 2 nm")
    if row["max_rot_deg"] > 15: flags.append("rotation > 15 deg")
    if row["max_Al_rmsd_nm"] > 0.3: flags.append("Al RMSD > 0.3 nm")
    if row["Al_graph_split"]: flags.append("framework split")
    if row["CoC49_max_nm"] > 0.30: flags.append("CO2 dissociating")
    if flags:
        row["status"] = "; ".join(flags)
    return row


def main():
    runs = sys.argv[1:] or sorted(glob.glob("*/ps_runs/run_*"))
    rows = []
    for run in runs:
        for rep in sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, "rep*"))):
            try:
                rows.append(check_rep(os.path.abspath(run), rep))
            except Exception as exc:          # report and continue
                rows.append({"run": run, "rep": rep, "status": f"ERROR {exc}"})
            print(rows[-1], file=sys.stderr)
    pd.DataFrame(rows).to_csv(sys.stdout, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
