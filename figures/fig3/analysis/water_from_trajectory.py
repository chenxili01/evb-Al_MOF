"""Water orientation, water count and RDFs around the CO2 carbon (C49) from one GROMACS endpoint run.

    python water_from_trajectory.py TOPOLOGY.tpr TRAJ.xtc --state RS|PS --dataset NAME --out DIR [--t0 1000 --t1 6000]

Produces, for frames with 1000 ≤ t ≤ 6000 ps (501 frames at 10 ps; the first 1 ns is discarded):
  bisector_<state>.csv  per frame: water_count_6A = water O within 6 Å of C49;
                        water_bisector_to_tcc_c49_deg = mean over those waters of the angle between the
                        H–O–H bisector and the O→C49 vector (minimum image)
  rdf_water.csv / rdf_K.csv in <out>/<state>/: g(r) of water O / K+ around C49, 0.1 Å bins to 12 Å,
                        normalised with the last frame's box volume (same layout as figures/fig3/rdf/)
RS = FEP window 20, PS = window 00 of reaction 1 (CO2 adsorption), System IV.
C49 = atom C49 of residue TCC (the CO2 carbon bound to Co).

Inputs and the last 1 ns of the CO2-adsorption trajectories are on Zenodo
(https://doi.org/10.5281/zenodo.21649285, trajectory_last1ns_CO2adsorption.tar); the published data use
1000–6000 ps of the full 6 ns runs, so with the Zenodo files use --t0 5000 (101 frames per state). Checked against site03 (Ic) RS: counts/angles identical to the deposited time series,
max |Δg| ≤ 2e-6.
Requires MDAnalysis.
"""
import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array

T0, T1, CUT, RMAX, DR = 1000.0, 6000.0, 6.0, 12.0, 0.1


def mic(v, box):
    return v - box[:3] * np.round(v / box[:3])


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tpr"); ap.add_argument("xtc")
    ap.add_argument("--state", required=True, choices=["RS", "PS"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t0", type=float, default=T0, help="first time included, ps (default 1000)")
    ap.add_argument("--t1", type=float, default=T1, help="last time included, ps (default 6000)")
    a = ap.parse_args()
    u = mda.Universe(a.tpr, a.xtc)
    c49 = u.select_atoms("resname TCC and name C49")
    assert c49.n_atoms == 1, c49
    ow, h1, h2 = (u.select_atoms(f"resname SOL and name {n}") for n in ("OW", "HW1", "HW2"))
    k = u.select_atoms("resname K or name K")
    edges = np.arange(0, RMAX + DR / 2, DR)
    counts = {"water": np.zeros(len(edges) - 1), "K": np.zeros(len(edges) - 1)}
    rows, n = [], 0
    for ts in u.trajectory:
        if not (a.t0 - 1e-6 <= ts.time <= a.t1 + 1e-6):
            continue
        box, cp = ts.dimensions, c49.positions[0]
        d_ow = distance_array(cp[None], ow.positions, box=box)[0]
        loc = d_ow <= CUT
        b = unit(unit(mic(h1.positions[loc] - ow.positions[loc], box)) + unit(mic(h2.positions[loc] - ow.positions[loc], box)))
        to_c = unit(mic(cp[None] - ow.positions[loc], box))
        ang = np.degrees(np.arccos(np.clip((b * to_c).sum(1), -1, 1)))
        rows.append({"dataset": a.dataset, "state": a.state, "frame": n, "time_ps": ts.time,
                     "water_count_6A": int(loc.sum()), "water_bisector_to_tcc_c49_deg": ang.mean() if loc.any() else np.nan})
        counts["water"] += np.histogram(d_ow, bins=edges)[0]
        counts["K"] += np.histogram(distance_array(cp[None], k.positions, box=box)[0], bins=edges)[0]
        vol, n = float(np.prod(box[:3])), n + 1
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / f"bisector_{a.state}.csv", index=False)
    shell = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    for sp, grp in (("water", ow), ("K", k)):
        g = counts[sp] / (n * grp.n_atoms / vol * shell)
        (a.out / a.state).mkdir(exist_ok=True)
        pd.DataFrame({"r_A": edges[:-1] + DR / 2, "g_r": g, "volume_A3": vol, "target_count": grp.n_atoms,
                      "frames": n, "center": f"C49_atom{c49.indices[0] - u.select_atoms('resname TCC').indices[0] + 1}"}).to_csv(a.out / a.state / f"rdf_{sp}.csv", index=False)
    print(f"{a.dataset} {a.state}: {n} frames")


if __name__ == "__main__":
    main()
