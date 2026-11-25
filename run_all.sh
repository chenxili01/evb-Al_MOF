#!/usr/bin/env bash
# Rebuild every source-data file from the deposited data and check every quoted number.
#   bash run_all.sh          # ~1 min; starts from the deposited EVB mapping results (output_mapping_results/)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$(mktemp -d)}
sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

echo "== DFT reference";            $PY dft/reaction_energies.py
echo "== EVB tables";               $PY scripts/build_evb_tables.py
echo "== MD field screen";          (cd md_field_screen/data/raw && sha -c --quiet SHA256SUMS) && echo "raw checksums OK"
                                    (cd md_field_screen && $PY scripts/process.py > /dev/null)
echo "== source data";              $PY figures/fig2/make_source_data.py
                                    $PY figures/fig3/analysis/dN_from_rdf.py
                                    $PY figures/fig4/make_source_data.py
                                    $PY figures/fig5/make_source_data.py
                                    $PY figures/si/make_source_data.py
echo "== numbers quoted in the paper"; $PY scripts/check_numbers.py
