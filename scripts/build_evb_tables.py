"""Build every EVB-derived table used in the manuscript from the mapping results.

    python scripts/build_evb_tables.py [--mapping DIR] [--out DIR]

Inputs (defaults are the files shipped in this repository):
  output_mapping_results/four_reaction_mof_environment_effect_summary.json
      128 EVB mapping results (4 reactions x 4 systems x 8 sites): ts_kcal, dg_kcal, fitted alpha/H12
  evbpmf_config_portable.json  ("dft_benchmark")
      molecular DFT reference barrier and reaction free energy per reaction (SI Table "DFT benchmark")

Outputs (written to --out, default evb_tables/):
  site_energies.csv                   ΔG‡, ΔG per reaction/system/site, DFT reference, shift vs reference
  environment_contributions_site.csv  sequential contributions per site (MOF embedding = I − reference,
                                      electrolyte = II − I, electrode = III − II, electric field = IV − III)
  environment_contributions_summary.csv  mean and site-to-site SD (ddof = 1) over the 8 sites
  paired_boundary_minus_inner.csv     ΔG(boundary) − ΔG(inner) for the matched pairs Ba–Ia … Bd–Id
  method_noise.csv                    |difference| between symmetry-equivalent sites Bb/Bc and Ib/Ic (Systems I, II)

All energies in kcal/mol. Sites 01–04 = Ia–Id (inner layer), 05–08 = Ba–Bd (boundary layer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE_LABEL = {"01": "Ia", "02": "Ib", "03": "Ic", "04": "Id", "05": "Ba", "06": "Bb", "07": "Bc", "08": "Bd"}
SYSTEM_ROMAN = {"01": "I", "02": "II", "03": "III", "04": "IV"}
REACTIONS = ["reaction1", "reaction2", "reaction3", "reaction4"]
PAIRS = [("05", "01"), ("06", "02"), ("07", "03"), ("08", "04")]          # (boundary, inner)
SYMMETRY_PAIRS = [("06", "07", "Bb/Bc"), ("02", "03", "Ib/Ic")]           # equivalent in Systems I, II
CONTRIBUTIONS = [  # key, to_system, from_system, definition
    ("mof_embedding", "01", None, "system I - molecular reference"),
    ("electrolyte", "02", "01", "system II - system I"),
    ("electrode", "03", "02", "system III - system II"),
    ("electric_field", "04", "03", "system IV - system III"),
]
ENERGY = {"activation": "activation_barrier", "reaction": "reaction_energy"}


def site_energies(mapping: Path, config: Path) -> pd.DataFrame:
    summary = json.loads(mapping.read_text())
    ref = json.loads(config.read_text())["dft_benchmark"]
    fits = {s["config"]["name"] if "name" in s["config"] else s["reaction"]: s["fit"] for s in summary["summaries"]}
    rows = []
    for r in summary["results"]:
        fit = fits[r["reaction"]]
        rows.append({
            "system": r["system"], "site": r["site"], "reaction": r["reaction"],
            "activation_barrier": r["ts_kcal"], "reaction_energy": r["dg_kcal"],
            "dft_reference_activation_barrier": ref[r["reaction"]]["barrier_kcal"],
            "dft_reference_reaction_energy": ref[r["reaction"]]["free_energy_kcal"],
            "fitted_alpha": fit["alpha"], "fitted_H12": fit["H12"],
        })
    t = pd.DataFrame(rows)
    t["system_id"] = t.system.str.extract(r"sys(\d{2})", expand=False)
    t["site_id"] = t.site.str.extract(r"site(\d{2})", expand=False)
    t["delta_activation"] = t.activation_barrier - t.dft_reference_activation_barrier
    t["delta_reaction"] = t.reaction_energy - t.dft_reference_reaction_energy
    assert len(t) == 128 and not t.duplicated(["system_id", "site_id", "reaction"]).any()
    return t.sort_values(["reaction", "system_id", "site_id"]).reset_index(drop=True)


def contributions(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    for etype, col in ENERGY.items():
        v = t.pivot(index=["site_id", "reaction"], columns="system_id", values=col)
        ref = t.pivot(index=["site_id", "reaction"], columns="system_id", values=f"dft_reference_{col}")
        for key, to_s, from_s, definition in CONTRIBUTIONS:
            val = v[to_s] - (ref[to_s] if from_s is None else v[from_s])
            out.append(val.rename("value").reset_index().assign(
                energy_type=etype, contribution=key, contribution_definition=definition,
                from_system=from_s or "molecular_reference", to_system=to_s))
    c = pd.concat(out, ignore_index=True)
    c["site_label"] = c.site_id.map(SITE_LABEL)
    c["layer"] = np.where(c.site_id.astype(int) <= 4, "inner", "boundary")
    return c[["energy_type", "contribution", "contribution_definition", "from_system", "to_system",
              "site_id", "site_label", "layer", "reaction", "value"]]


def summary(c: pd.DataFrame) -> pd.DataFrame:
    order = {k: i for i, (k, *_) in enumerate(CONTRIBUTIONS)}
    s = (c.groupby(["energy_type", "contribution", "reaction"]).value
         .agg(mean="mean", site_to_site_sd=lambda x: x.std(ddof=1), n_sites="count").reset_index())
    s["_o"] = s.contribution.map(order)
    return s.sort_values(["energy_type", "_o", "reaction"]).drop(columns="_o").reset_index(drop=True)


def paired(t: pd.DataFrame) -> pd.DataFrame:
    v = t.set_index(["reaction", "system_id", "site_id"])
    rows = []
    for rx in REACTIONS:
        for s in SYSTEM_ROMAN:
            for b, i in PAIRS:
                rows.append({"reaction": rx, "system_id": s, "system_display": SYSTEM_ROMAN[s],
                             "pair": f"{SITE_LABEL[b]}-{SITE_LABEL[i]}",
                             "boundary_minus_inner_activation": v.loc[(rx, s, b), "activation_barrier"] - v.loc[(rx, s, i), "activation_barrier"],
                             "boundary_minus_inner_reaction": v.loc[(rx, s, b), "reaction_energy"] - v.loc[(rx, s, i), "reaction_energy"]})
    return pd.DataFrame(rows)


def noise(t: pd.DataFrame) -> pd.DataFrame:
    v = t.set_index(["reaction", "system_id", "site_id"])
    rows = []
    for rx in REACTIONS:
        for s in ("01", "02"):
            for a, b, name in SYMMETRY_PAIRS:
                rows.append({"reaction": rx, "system_display": SYSTEM_ROMAN[s], "pair": name,
                             "abs_diff_activation": abs(v.loc[(rx, s, a), "activation_barrier"] - v.loc[(rx, s, b), "activation_barrier"]),
                             "abs_diff_reaction": abs(v.loc[(rx, s, a), "reaction_energy"] - v.loc[(rx, s, b), "reaction_energy"])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", type=Path,
                    default=ROOT / "output_mapping_results/four_reaction_mof_environment_effect_summary.json")
    ap.add_argument("--config", type=Path, default=ROOT / "evbpmf_config_portable.json")
    ap.add_argument("--out", type=Path, default=ROOT / "evb_tables")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t = site_energies(a.mapping, a.config)
    c = contributions(t)
    tables = {"site_energies": t, "environment_contributions_site": c,
              "environment_contributions_summary": summary(c), "paired_boundary_minus_inner": paired(t),
              "method_noise": noise(t)}
    for name, df in tables.items():
        df.to_csv(a.out / f"{name}.csv", index=False, float_format="%.6f")
        print(f"wrote {a.out / name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
