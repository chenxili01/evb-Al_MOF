"""Fig. 5 source data, from md_field_screen/data/processed (run `bash md_field_screen/run_all.sh` to rebuild).

    python figures/fig5/make_source_data.py

a  equation coefficients a, b, c, d                         -> fig5a_equation_coefficients.csv
b  crystal score N per crystal (L layers, W width)          -> fig5bc_crystal_scores.csv
c  simulated N (± replica SEM) vs leave-one-out prediction   -> fig5bc_crystal_scores.csv (N, N_sem_reps, N_loo)
d  N(L, W) curves from the fitted equation                  -> fig5d_curves.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROC = HERE.parents[1] / "md_field_screen" / "data" / "processed"
OUT = HERE / "source_data"
OUT.mkdir(exist_ok=True)

sc = pd.read_csv(PROC / "crystal_scores.csv")
sc[["model", "family", "layers", "ny", "nz", "width_ch", "n_sites", "E_ax_mean", "N", "N_rep1", "N_rep2", "N_rep3",
    "N_sem_reps", "N_end", "N_interior", "N_fit", "N_loo"]].to_csv(OUT / "fig5bc_crystal_scores.csv", index=False)

fit = pd.read_csv(PROC / "score_fit.csv")
fit.to_csv(OUT / "fig5a_equation_coefficients.csv", index=False)
f = fit.iloc[0]
L = np.arange(2, 41)
rows = []
for W in (3, 5, 7):
    E = np.where(L == 2, 1.0, 2.0 / L)
    N = E * (f.a + f.b / W + f.c * (L == 2)) + (1 - E) * f.d
    rows += [{"W": W, "L": int(l), "N": n} for l, n in zip(L, N)]
pd.DataFrame(rows).to_csv(OUT / "fig5d_curves.csv", index=False, float_format="%.4f")
print("wrote", *sorted(p.name for p in OUT.glob("*.csv")))
