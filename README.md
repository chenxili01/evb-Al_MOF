# Data and code: framework-defined microenvironments of Co sites in Al₂(OH)₂TCPP-Co

Source data, analysis code and checks for every data panel and table of the manuscript
"Framework-Defined Microenvironments Differentiate Molecular CO₂ Reduction Sites" and its Supplementary
Information. Paths are relative to the repository root; nothing outside the repository is read.

```bash
python -m pip install -r requirements.txt
bash run_all.sh      # rebuild all source data from the deposited inputs and check every quoted number (~1 min)
```

`run_all.sh` ends with `scripts/check_numbers.py`, which recomputes every data value quoted in the main text,
figure labels and SI (176 values) and compares it with the printed number at the printed precision.

## Where each figure's data come from

| Item | Source data (plotted values) | Built by | From |
|---|---|---|---|
| Fig. 1 | — (schematics and renders, no data) | — | — |
| Fig. 2a | `figures/fig2/source_data/fig2a_site_shifts.csv` | `figures/fig2/make_source_data.py` | `evb_tables/site_energies.csv` |
| Fig. 2b | `.../fig2b_contribution_mean_sd.csv` (mean, site-to-site SD) | same | `evb_tables/environment_contributions_*.csv` |
| Fig. 2c | `.../fig2c_co2_adsorption_pairs.csv` | same | `evb_tables/paired_boundary_minus_inner.csv` |
| Fig. 2d | `.../fig2d_co2_adsorption_contributions.csv` | same | `evb_tables/environment_contributions_site.csv` |
| Fig. 3c, e | `figures/fig3/source_data/fig3ce_water_bisector_c49_timeseries.csv` | `figures/fig3/analysis/water_from_trajectory.py` | MD trajectories (not deposited, see `figures/fig3/README.md`) |
| Fig. 3f, h | `figures/fig3/source_data/dN/dN_*.csv` | `figures/fig3/analysis/dN_from_rdf.py` | `figures/fig3/rdf/` (RDFs from the same trajectory script) |
| Fig. 4b | `figures/fig4/source_data/fig4b_dft_field_response.csv`, `fig4b_slopes.csv` | `figures/fig4/make_source_data.py` | `figures/fig4/data/gaussian_field_axis_response.csv` (DFT) |
| Fig. 4c | `.../fig4c_paper_sites.csv`, `fig4c_calibration.csv` | same | `md_field_screen/data/processed/anchors.csv` (field) + `evb_tables` (EVB) |
| Fig. 4d | `.../fig4d_field_components_V_nm.csv`, `fig4d_class_mean_sd_V_nm.csv` | same | `figures/fig4/data/contrib_species.csv` |
| Fig. 5a–d | `figures/fig5/source_data/*.csv` | `figures/fig5/make_source_data.py` | `md_field_screen/data/processed/` |
| SI DFT tables, SI Fig. DFT profile | `dft/dft_species_energies.csv`, `dft/dft_profile.csv`, `dft/electron_transfer.csv` | `dft/reaction_energies.py` (step energies = EVB fit targets; recomputes every profile level) | — |
| SI Table noise; SI Figs other steps, CO-desorption decomposition | `evb_tables/method_noise.csv`, `paired_boundary_minus_inner.csv`, `environment_contributions_site.csv` | `scripts/build_evb_tables.py` | `output_mapping_results/` |
| SI field-response, field-contribution, LOO tables | `figures/fig4/source_data/` | `figures/fig4/make_source_data.py` | as Fig. 4 |
| SI top-N snapshot projection, residue-decomposition tables | `figures/fig4/data/site_relative_response_vs_evb.tsv`, `species_weighted_contributions.tsv`, `residue_contributions.tsv`, `snapshot_fields.tsv` | (tabulated directly) | top-10 PMF snapshots |
| SI Table crystal scores | `figures/fig5/source_data/fig5bc_crystal_scores.csv` | `md_field_screen/scripts/process.py` | `md_field_screen/data/raw/` |
| SI applicability domain, cutoff comparison | `figures/si/source_data/si_domain_*.csv`, `si_cutoff_*.csv` | `figures/si/make_source_data.py` | `md_field_screen/data/processed/sites_symmetrized.csv` |

Site labels: site01–04 = Ia–Id (inner layer), site05–08 = Ba–Bd (boundary layer).
Systems: I = sys01 (MOF in water), II = sys02 (+ CO₂/KHCO₃), III = sys03 (+ graphene), IV = sys04 (+ field).
Reactions: 1 CO₂ adsorption, 2 *COOH formation, 3 C–OH cleavage, 4 CO desorption. Energies in kcal/mol, fields in V/nm.

## Layout

```text
input_data/                 EVB-FEP PotentialA/PotentialB XVG files (4 reactions x 4 systems x 8 sites + references)
analysis_code/              EVB runner (evbpmf.py) and helpers
evbpmf_config_portable.json EVB configuration, incl. the DFT reference ("dft_benchmark") the EVB parameters are fitted to
output_mapping_results/     the 128 EVB mappings (ΔG‡, ΔG per site), PMFs, fitted α/H12, EVB reference mappings
evb_tables/                 derived EVB tables (scripts/build_evb_tables.py)
dft/                        molecular DFT free energies, electron-transfer constants, SI profile levels
md_field_screen/            classical-MD axial-field data: 8 paper sites + reference + 19 crystals (1726 sites),
                            raw per-replica fields with SHA256SUMS, processing scripts, processed tables
figures/figN/               per-figure source data and the scripts that build them; figures/si/ for the SI
scripts/                    build_evb_tables.py, check_numbers.py
run_all.sh                  everything above in order
```

The EVB mappings in `output_mapping_results/` can be regenerated from `input_data/` with
`python analysis_code/evbpmf.py evbpmf_config_portable.json --output-dir <dir>` (`--preflight-only` checks the
inputs; `--reaction/--system/--site` select a subset).

## Data definitions worth knowing

- **EVB data version.** All EVB values are from the 2026-08-23 rerun, in which *COOH formation (reaction 2) was
  re-mapped for all sites and systems; the other three steps are unchanged from the earlier run.
- **Two molecular references.** Fig. 2 and SI S2 give shifts relative to the **DFT** reference
  (e.g. CO₂ adsorption ΔG = 4.56). Fig. 4c gives EVB ΔΔG relative to the **EVB-FEP mapping of the molecule**
  (CO₂ adsorption ΔG = 4.44, `reference_mapping` in `output_mapping_results/*.json`), so both axes of the
  calibration compare simulation with simulation.
- **Calibration offset.** The fitted offset is −3.225 kcal/mol (mean over the 8 sites); the plotted Fig. 4c values
  and the SI ΔΔG^pred columns use the rounded −3.2. r, ρ and the leave-one-out errors do not depend on it.
- **Fig. 4d totals** are the sum of the components shown.
- **Fig. 3 processing.** Frames 1000–6000 ps (501 per state; first 1 ns discarded); ΔN curves use a 5-bin (0.5 Å)
  moving average of g_PS − g_RS and the PS number density; r is the bin centre and the sum includes the whole bin.

## Not included

- Simulation inputs, DFT structures, the last 1 ns of the CO₂-adsorption trajectories and the electric-field
  frame data are on Zenodo: https://doi.org/10.5281/zenodo.21649285. Full trajectories (`.xtc`) are not deposited;
  the scripts that analyse them are included and were checked against the deposited derived data.
- Renders and drawings (Fig. 1; Fig. 3a, b, d, g; Fig. 4a, e; Fig. 5a schematic).
- Figure styling/assembly scripts; the source-data files above are the plotted values.

## License and author

Copyright (c) Chenxi Li. Released under the [MIT License](LICENSE). Contact: `chenxili@kth.se`
