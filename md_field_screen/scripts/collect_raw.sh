#!/usr/bin/env bash
# Run ON THE CLUSTER from the morphologies/ folder: copies the analysis outputs (no trajectories)
# into plot_package/data/raw and the upstream scripts/settings into plot_package/upstream.
#   bash plot_package/scripts/collect_raw.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
P=plot_package
rm -rf $P/data/raw; mkdir -p $P/data/raw $P/upstream/mdp $P/upstream/itps
for m in $(ls -d */ps_runs | cut -d/ -f1); do
  mkdir -p $P/data/raw/$m
  cp $m/ps_runs/sites.csv $P/data/raw/$m/
  for run in $m/ps_runs/run_*; do
    r=$(basename $run); mkdir -p $P/data/raw/$m/$r
    cp $run/axial_field.csv $run/axial_field_frames.csv.gz $run/ps_sites.csv $P/data/raw/$m/$r/
  done
done
cp results/full/health.csv $P/data/raw/health.csv
cp tools/axial_field_traj.py tools/check_runs.py tools/analyze_co_c49_axial_field.py jobs/run_one.sh $P/upstream/
cp mdp/*.mdp $P/upstream/mdp/
cp itps/AL6.itp $P/upstream/itps/
(cd $P/data/raw && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "raw data: $(find $P/data/raw -type f | wc -l) files, $(du -sh $P/data/raw | cut -f1)"
