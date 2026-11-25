# Fig. 3 — water reorganisation on CO₂ adsorption (reaction 1, System IV)

| Path | Content |
|---|---|
| `source_data/fig3ce_water_bisector_c49_timeseries.csv` | per frame, RS (FEP window 20) and PS (window 00), 8 sites + molecular reference, 501 frames each (1000–6000 ps): water O within 6 Å of C49 and their mean H–O–H bisector angle to O→C49 (panels c, e) |
| `source_data/dN/dN_*.csv` | ΔN(r) = N_PS − N_RS of water O and K⁺ around C49 (panels f, h) |
| `rdf/<site>/{RS,PS}/rdf_{water,K}.csv` | g(r) around C49, 0.1 Å bins, input of the ΔN curves |
| `analysis/water_from_trajectory.py` | trajectory → time series + RDFs (needs MDAnalysis and the trajectories) |
| `analysis/dN_from_rdf.py` | RDFs → ΔN; by default recomputes and checks against `source_data/dN` |

C49 is the carbon of the Co-bound CO₂ (atom C49 of residue TCC). Plotted state means are plain means over the
501 frames.

Checks performed when this deposit was assembled:
- `dN_from_rdf.py` reproduces every `dN_*.csv` to 1e-13.
- `water_from_trajectory.py` on site03 (Ic) RS reproduces the 501 water counts exactly, the angles to 1e-5°
  (float32) and g(r) of water and K⁺ to machine precision; checked for RS and PS.

Trajectories are not part of this repository. The simulation inputs and the last 1 ns (5000–6000 ps) of the
CO₂-adsorption endpoint trajectories are on Zenodo (https://doi.org/10.5281/zenodo.21649285,
`trajectory_last1ns_CO2adsorption.tar`). The deposited time series and RDFs use 1000–6000 ps of the full 6 ns
runs, so running the script on the Zenodo files (`--t0 5000`) gives the same quantities over the last 1 ns only,
not the plotted values.
