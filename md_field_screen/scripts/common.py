"""Constants, model metadata and colours shared by process.py and plot.py.

Crystal score: N = (mean E_ax over all sites, lower-energy face - E_INNER) / (E_BOUNDARY - E_INNER),
anchored on the paper's inner (N = 0) and boundary (N = 1) sites.

Change a constant here, rerun `python scripts/process.py && python scripts/plot.py`,
and every table and figure follows.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# ---- physics / analysis constants -------------------------------------------------------
S = 4.576            # kcal/mol per V/nm: DFT uniform-field response of CoTPP (paper)
E_REF_PAPER = -4.207 # V/nm: paper's PMF-weighted molecular reference (information only, column ddE_ref)
PAPER = {"site01": "inner", "site02": "inner", "site03": "inner", "site04": "inner",
         "site05": "boundary", "site06": "boundary", "site07": "boundary", "site08": "boundary",
         "reference": "molecule"}


def paper_anchors():
    """E_ax of the 8 paper sites and the CoTCPP molecule (3 x 20 ns, same protocol as the crystals),
    from data/raw/paper_sites/<site>/axial_field.csv."""
    import pandas as pd
    rows = [dict(site=k, kind=v, **pd.read_csv(RAW / "paper_sites" / k / "axial_field.csv").iloc[0][
        ["E_ax", "E_ax_sem_reps", "E_ax_rep1", "E_ax_rep2", "E_ax_rep3"]].to_dict()) for k, v in PAPER.items()]
    return pd.DataFrame(rows)


_A = paper_anchors()
E_INNER = float(_A[_A.kind == "inner"].E_ax.mean())        # -6.219 V/nm: N = 0 and zero of ddE_rel
E_BOUNDARY = float(_A[_A.kind == "boundary"].E_ax.mean())  # -7.312 V/nm: N = 1
E_MOLECULE = float(_A[_A.kind == "molecule"].E_ax.iloc[0]) # -5.803 V/nm: CoTCPP + K+ in water
E_MID = E_INNER                                            # zero of the site-level ddE (kcal/mol)
SPAN = S * (E_BOUNDARY - E_INNER)                          # -5.00 kcal/mol: ddE of N = 1
BEGIN_PS = 5000      # frames with t >= BEGIN_PS (last 15 ns of 20 ns) enter the averages
THRESH = 3.0         # kcal/mol: colour node of the site maps only (no share/threshold in this package)
GREY = 1.0           # kcal/mol: |ddE| <= GREY drawn grey in the maps
LIM = 9.0            # kcal/mol: colour limit (99th percentile of |ddE_rel|, rounded up)
REPS = ("rep1", "rep2", "rep3")

# ---- models ------------------------------------------------------------------------------
ORDER = ["slab_t1", "slab_t2", "slab_t3", "slab_t4",
         "part_LW0.3", "part_LW0.5", "part_LW1", "part_LW2", "part_LW3",
         "aniso_H3", "aniso_H5", "aniso_H7", "aniso_W9", "aniso_H9",
         "thin_L2_W3", "thin_L3_W3", "thin_L2_W5", "thin_L3_W5", "plate_L2_3x7"]
FAMILY = {m: ("Slabs" if m.startswith("slab") else "Particles" if m.startswith("part")
              else "Anisotropic plates" if m.startswith("aniso") else "Thin crystals") for m in ORDER}
LABEL = {"slab_t1": "Slab, 1 layer", "slab_t2": "Slab, 2 layers", "slab_t3": "Slab, 3 layers",
         "slab_t4": "Slab, 4 layers", "part_LW0.3": "Particle, L/W 0.3", "part_LW0.5": "Particle, L/W 0.5",
         "part_LW1": "Particle, L/W 1", "part_LW2": "Particle, L/W 2", "part_LW3": "Particle, L/W 3",
         "aniso_H3": "Plate, 3y×3z", "aniso_H5": "Plate, 3y×5z", "aniso_H7": "Plate, 3y×7z",
         "aniso_W9": "Plate, 9y×3z", "aniso_H9": "Plate, 3y×9z",
         "thin_L2_W3": "Thin, 2 L × 3×3", "thin_L3_W3": "Thin, 3 L × 3×3", "thin_L2_W5": "Thin, 2 L × 5×5",
         "thin_L3_W5": "Thin, 3 L × 5×5", "plate_L2_3x7": "Thin, 2 L × 3×7"}

# ---- colours -----------------------------------------------------------------------------
C_FAV, C_NEU, C_UNF = "#2166ac", "#c8c8c8", "#b2182b"     # favoured / no difference / less favoured
C_FAV_LIGHT = "#92c5de"                                   # favoured in the mean, not in all 3 replicas
C_FAMILY = {"Slabs": "#1b9e77", "Particles": "#d95f02", "Anisotropic plates": "#7570b3", "Thin crystals": "#e7298a"}
INK, INK2 = "#222222", "#666666"


def ddE_cmap(lim=LIM, grey=GREY, n=361):
    """Diverging blue-grey-red map; flat grey for |v| <= grey (kcal/mol), on a scale -lim..lim."""
    from matplotlib.colors import ListedColormap, to_rgb
    v = np.linspace(-lim, lim, n)
    neg = [(-lim, "#08306b"), (-THRESH, "#4292c6"), (-grey, "#c6dbef")]
    pos = [(grey, "#fcbba1"), (THRESH, "#ef3b2c"), (lim, "#67000d")]

    def interp(x, nodes):
        xs = [p for p, _ in nodes]; cs = np.array([to_rgb(c) for _, c in nodes])
        return [np.interp(x, xs, cs[:, k]) for k in range(3)]

    rgb = [to_rgb(C_NEU) if abs(x) <= grey else interp(x, neg if x < 0 else pos) for x in v]
    return ListedColormap(rgb, name="ddE")
