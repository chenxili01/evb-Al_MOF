#!/usr/bin/env bash
# Rebuild everything from data/raw:  bash run_all.sh   (add --frames to re-average from per-frame data)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
(cd data/raw && sha256sum -c --quiet SHA256SUMS) || { echo "raw data checksum mismatch" >&2; exit 1; }
echo "raw data checksums OK"
$PY scripts/process.py "$@"
$PY scripts/plot.py
