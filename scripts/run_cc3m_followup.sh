#!/usr/bin/env bash
# Run from any directory; completed experiments stay untouched.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${MM_SAE_PYTHON:-$repo_root/.venv/bin/python}"
config="${1:-configs/cc3m-followup-server.yaml}"
mode="${2:-}"
if [[ "$mode" == "--check" || "$mode" == "--prepare-only" ]]; then
  exec "$python_bin" -m experiments.corpus_comparison.prepare --config "$config" "$mode"
fi
if [[ -n "$mode" ]]; then
  echo "Usage: bash scripts/run_cc3m_followup.sh [config.yaml] [--check|--prepare-only]" >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
"$python_bin" -m experiments.corpus_comparison.prepare --config "$config"
sequence_path="$($python_bin -c 'import pathlib,sys,yaml; p=pathlib.Path(sys.argv[1]).resolve(); c=yaml.safe_load(p.read_text()); print((p.parent/c["output"]/"configs/sequence.yaml").resolve())' "$config")"
exec "$python_bin" scripts/run_experiment_sequence.py --config "$sequence_path"
