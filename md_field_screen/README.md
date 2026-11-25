> In this repository: rebuilt by `run_all.sh` at the root (checksums → `scripts/process.py`); `figures/` is not
> deposited (run `python scripts/plot.py` to draw it). Mentions of `../plot_package` refer to the earlier
> share-based version of this package, which is not part of the deposit.

# Crystal score N of Al2(OH)2TCPP–Co morphologies: data and figures

Successor of `../plot_package` (2026-11-22). Same raw MD data (19 crystals, Sets A–C, 1726 Co sites,
each simulated as the CO2-bound product state in 3 × 20 ns replicas, TIP3P water + K⁺, GROMACS 2026.3),
but each crystal is scored by a **continuous, normalised average** instead of a thresholded share.

```
bash run_all.sh              # checksums -> data/processed -> figures   (~1 min)
bash run_all.sh --frames     # also re-average every site from the per-frame data (check)
python scripts/plot.py --only morphology   # one group: scores classes trend replicates maps sites3d morphology
```
Needs Python ≥ 3.10 with numpy, pandas, matplotlib, scipy (`requirements.txt`).

## Crystal score

- **Site value:** E_ax at the Co–C49 bond midpoint, projected on Co→C49 (charges within 1.0 nm of C49,
  reacting TCC excluded, no dielectric scaling), mean over t = 5–20 ns of 3 replicas.
  ΔΔE = S · (E_ax − E_inner), S = 4.576 kcal mol⁻¹ (V/nm)⁻¹.
- **Lower-energy face:** CO2 may bind on either axial face; within one channel column the +x state of the
  site at `n_from_out = k` is the mirror of the simulated state of the site at `n_from_in = k`; take the lower.
- **Crystal average:** plain mean over **all** sites of the crystal (each site weight 1), lower-energy face.
- **Normalisation:**

      N = (Ē_ax − E_inner) / (E_boundary − E_inner)

  E_inner = −6.219 V/nm and E_boundary = −7.312 V/nm are the means of the four inner (site01–04) and four
  boundary (site05–08) sites of the paper, simulated with the same protocol (`data/raw/paper_sites/`).
  N = 0: the crystal's average site behaves like a paper inner site; N = 1: like a paper boundary site.
  N < 0: worse than inner (CoTCPP molecule: N = −0.38). Span E_boundary − E_inner = −5.00 kcal/mol, so
  N = mean ΔΔE / (−5.00 kcal/mol). N is independent of the slope S and of any zero point, and the anchors
  are external, so N of a crystal does not change when crystals are added.
- **Uncertainty:** `N_sem_reps` = SEM of the three per-replica crystal averages (median 0.025).

## Morphology equation

Fitted by least squares on the 18 crystals with ≥ 2 layers (`process.py`, `fit_N`):

    N(L, W) = E(L) · (a + b/W + c·δ(L=2)) + (1 − E(L)) · d
    E(L) = share of channel-end sites = 1 (L = 2), 2/L (L ≥ 3)
    a = 0.66, b = 0.49, c = 0.22 (end sites: wide crystal, 1/W cross-section term, 2-layer term),
    d = 0.43 (inner-layer sites)

L = layers along the channel, W = narrower cross-section (channels). Leave-one-out (refit without crystal i,
predict i): MAE 0.029 (0.14 kcal/mol), max 0.096 (thin_L3_W3), Q² 0.96, Spearman ρ 0.90 (`score_fit.csv`);
this is close to the replica noise. Simpler forms: without the 2-layer term Q² 0.83; without 1/W ρ 0.75.
A single layer (slab_t1; no axial neighbour) is its own class, N = −0.68, and is not fitted.

## Layout

| path | what |
|---|---|
| `data/raw/<model>/` | `sites.csv`, `run_XX/axial_field.csv`, `run_XX/axial_field_frames.csv.gz`, `run_XX/ps_sites.csv` (unchanged copy of `../plot_package/data/raw`) |
| `data/raw/paper_sites/<site>/axial_field.csv` | the 8 paper sites + `reference` (CoTCPP molecule), 3 × 20 ns; the anchors of N |
| `data/raw/SHA256SUMS`, `health.csv` | checksums; per-replica health checks |
| `scripts/common.py` | constants, anchors (read from `paper_sites`), labels, colours |
| `scripts/process.py` | raw → processed tables and the fit |
| `scripts/plot.py` | processed → figures |
| `upstream/` | scripts and settings that produced `data/raw` (need GROMACS/MDAnalysis + trajectories) |

## Processed tables (`data/processed/`)

| file | content |
|---|---|
| `sites_all.csv` | one row per site: raw columns + `ddE_rel` (vs E_inner), `dd_rep1..3`, `dd_sem` |
| `sites_symmetrized.csv` | + `mirror_resid`, `dd_mirror`, `dd_best` (lower-energy face), `dd_best_rep1..3`, `N_site` = dd_best / span |
| `class_summary.csv` | mean ΔΔE per model × site class, both orientations, replica SEM |
| `anchors.csv` | paper sites and molecule: E_ax, ΔΔE, N |
| `crystal_scores.csv` | per crystal: layers, width, `n_sites`, `E_ax_mean`, `score_kcal`, `score_vs_molecule_kcal`, `N`, `N_rep1..3`, `N_sem_reps`, `N_end`, `N_interior`, `N_fit`, `N_loo` |
| `score_fit.csv` | a, b, c, d; fit MAE; leave-one-out MAE (also in kcal/mol), max, RMSE, Q², ρ; anchors |

## Figures (`figures/`, PDF + 300 dpi PNG)

| file | content |
|---|---|
| `fig1_crystal_scores` | N per crystal (bar = mean, error = replica SEM, dots = replicas); top axis in kcal/mol; anchor lines |
| `fig2_class_means_{as_simulated,lower_energy_face}` | mean ΔΔE per face × channel position, replica SEM |
| `fig3_score_vs_layers` | N vs channel length with the fitted equation for W = 3, 5, 7 |
| `fig4_replicate_agreement` | site ΔΔE, replica vs replica |
| `fig5_site_maps/<model>` | every site (channels × position; y–z maps), title gives N ± SEM |
| `fig6_sites_3d/<model>` | 3D site scatter |
| **`fig7_morphology_map`** | **publication candidate.** a: N on layers × narrower cross-section, background = fitted equation; b: leave-one-out test |

## Differences from `../plot_package`

- No −3 kcal/mol threshold and no favoured share; no `model_summary.csv`, `morphology_scores.csv`,
  `model_fit.csv`, `end_fraction_by_class.csv`.
- The Set C blind test (share predictions frozen 2026-11-17) is not part of this package; its record stays
  in `../plot_package` (`setC_comparison.csv`), on the scale on which the predictions were made.
- Site-level ΔΔE uses E_inner = −6.219 V/nm as zero (same as `../plot_package` since 2026-11-22).

MD protocol and analysis provenance: see `../plot_package/README.md` (identical).
