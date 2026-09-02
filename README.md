# EVB PMF mapping data and code

This repository contains the input potentials, portable EVB mapping code, and
reproducible mapping results for four catalytic-framework reactions. The
package is self-contained: paths in the configuration are relative to the
repository root, and no original repository layout is required.

## What is included

```text
input_data/                    PotentialA/PotentialB XVG inputs
analysis_code/                 EVB runner and local helper modules
evbpmf_config_portable.json    Four-reaction portable configuration
output_mapping_results/        Included mapping results and PMF profiles
figure_data/figure3_6/         CSV inputs and plotting scripts for Figures 3--6
LICENSE                        MIT license
```

The configured reactions are:

1. CO2 adsorption
2. COOH formation
3. C--OH cleavage
4. CO desorption

The input inventory contains four systems, eight sites per system, eight
reference-potential directories, and 5,376 XVG files. The included mapping
result table contains 128 site mappings (32 per reaction).

## Installation

Python 3.10 or newer is recommended. Install the runtime dependencies from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

For headless environments, set `MPLCONFIGDIR` to a writable directory before
running the scripts.

## Reproduce the mapping results

Run the inventory check first:

```bash
python analysis_code/evbpmf.py \
  evbpmf_config_portable.json \
  --preflight-only
```

Run all four reactions into a new directory:

```bash
python analysis_code/evbpmf.py \
  evbpmf_config_portable.json \
  --output-dir reproduced_results
```

The runner reports completed mapping results and writes the raw and mapped
profiles. It does not generate a separate quality-check or post-processing
analysis directory. To process a subset, repeat `--reaction`, `--system`, or
`--site`, for example:

```bash
python analysis_code/evbpmf.py evbpmf_config_portable.json \
  --reaction reaction2 --site site01_res1109 \
  --output-dir reaction2_site01_result
```

## Result files

The included [output_mapping_results](output_mapping_results/) directory has:

- `four_reaction_mof_environment_effect_summary.csv`: one row per mapped
  reaction/system/site result.
- `four_reaction_mof_environment_effect_summary.json`: detailed mapping data,
  fitted parameters, state assignments, and profile references.
- `reaction1/` through `reaction4/pmf/`: per-reaction PMF CSV, PNG, and state-bin
  JSON files, including reference profiles.
- `raw_evb_profiles/`: raw intermediate profile data and diagnostic plots.
- `evbpmf_manifest.json`: portable run manifest with relative paths.

## Figure data

The CSV inputs and plotting scripts for the Figure 3--6 panel set are in
`figure_data/figure3_6/`. Each script reads only the CSV files in that folder
and writes regenerated figures to `figure_data/figure3_6/reproduced_figures/`.
For example:

```bash
python figure_data/figure3_6/plotting_code/plot_figure3b.py
```

## License and author

Copyright (c) Chenxi Li. This project is released under the [MIT License](LICENSE).

Author: Chenxi Li — `chenxili@kth.se`
