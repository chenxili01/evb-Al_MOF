# Cluster EVB Tools

Portable scripts for extracting recalculation potentials and mapping EVB PMFs.

## Contents

- `extract_recal_potential.py`: run `gmx energy` over finished `windows/*/stateA` and `windows/*/stateB` folders to write `potentialA/*potential_recalA.xvg` and `potentialB/*potential_recalB.xvg`.
- `evb_script.py`: fit the four reference reactions and map site/system PMFs from collected `potentialA`/`potentialB` XVGs.
- `mof_evb_demo_methods.py` and `evb_tools/`: local helper code used by `evb_script.py`.
- `reference_targets/`: optional DFT reference target sets.
- `run_evb.sh`: convenience wrapper.

## Python Requirements

Use a Python environment with:

```bash
python -m pip install numpy scipy matplotlib
```

On a cluster, load GROMACS before running `extract_recal_potential.py`.

## Expected EVB Data Layout

Run `evb_script.py` from a data directory with this shape:

```text
datafolder/
  cluster_evb_tools/
  reaction1/<system>/<site>/potentialA/*.xvg
  reaction1/<system>/<site>/potentialB/*.xvg
  reaction2/...
  reaction3/...
  reaction4/...
  ref_reaction1/potentialA/*.xvg
  ref_reaction1/potentialB/*.xvg
  ref_reaction2/...
  ref_reaction3/...
  ref_reaction4/...
```

The strict preflight expects four systems, eight sites, four reactions, four reference reactions, and lambda windows `0..20` for both `potentialA` and `potentialB`.

## Commands

Preflight only:

```bash
cd /path/to/datafolder
python cluster_evb_tools/evb_script.py --preflight-only
```

Run EVB with built-in default targets:

```bash
cd /path/to/datafolder
bash cluster_evb_tools/run_evb.sh
```

Run EVB with a target file:

```bash
cd /path/to/datafolder
bash cluster_evb_tools/run_evb.sh "$PWD" "$PWD/evb_result_def2tzvp" \
  "$PWD/cluster_evb_tools/reference_targets/ref_target_def2tzvp.txt"
```

Extract potentials from finished recalculation windows:

```bash
python cluster_evb_tools/extract_recal_potential.py /path/to/cases \
  --gmx gmx_mpi --start-time 0 --selection-a Potential --selection-b Potential
```

Use `--dry-run` first if you want to inspect which recalculation folders will be processed.
