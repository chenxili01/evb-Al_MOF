"""ΔN(r) of water O and K+ around the CO2 carbon (C49), PS − RS (Fig. 3f, h), from the RDFs in figures/fig3/rdf/.

    python figures/fig3/analysis/dN_from_rdf.py            # recompute and check against source_data/dN (|Δ| < 1e-9)
    python figures/fig3/analysis/dN_from_rdf.py --out DIR  # write the recomputed curves to DIR

ΔN(r) = Σ_{bins ≤ r} smooth5(g_PS − g_RS) · ρ_PS · V_shell
  g        RDF of the species around C49, 0.1 Å bins to 12 Å, frames 1000–6000 ps (501 frames per state)
  smooth5  5-bin (0.5 Å) centred moving average, edge-padded
  ρ_PS     number density of the species in the PS run = target_count / volume_A3 (PS file)
  V_shell  4π/3 (r_hi³ − r_lo³) of each bin
r_A in the output is the bin centre; the cumulative sum includes the whole bin.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
SPECIES = {"water": "delta_N_water_PS_minus_RS", "K": "delta_N_K_PS_minus_RS"}


def smooth(v, w=5):
    h = w // 2
    return np.convolve(np.pad(v, (h, h), mode="edge"), np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for d in sorted(p for p in (HERE / "rdf").iterdir() if p.is_dir()):
        cols = {}
        for sp, col in SPECIES.items():
            ps, rs = (pd.read_csv(d / st / f"rdf_{sp}.csv") for st in ("PS", "RS"))
            assert np.allclose(ps.r_A, rs.r_A)
            r = ps.r_A.to_numpy()
            lo, hi = r - 0.05, r + 0.05
            shell = 4 * np.pi / 3 * (hi ** 3 - np.clip(lo, 0, None) ** 3)
            rho = ps.target_count.iloc[0] / ps.volume_A3.iloc[0]
            cols["r_A"] = r
            cols[col] = np.cumsum(smooth(ps.g_r.to_numpy() - rs.g_r.to_numpy()) * rho * shell)
        new = pd.DataFrame(cols)
        if a.out:
            new.to_csv(a.out / f"dN_{d.name}.csv", index=False)
        else:
            old = pd.read_csv(HERE / "source_data" / "dN" / f"dN_{d.name}.csv")
            worst = max(worst, float(np.abs(old[new.columns].to_numpy() - new.to_numpy()).max()))
    if not a.out:
        print(f"dN recomputed from RDFs: max |difference| to source_data/dN = {worst:.1e}")
        assert worst < 1e-9


if __name__ == "__main__":
    main()
