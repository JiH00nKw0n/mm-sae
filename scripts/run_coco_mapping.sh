#!/usr/bin/env bash
# Run the existing COCO mapping suite and its preprocessing comparisons.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: bash scripts/run_coco_mapping.sh OUTPUT_DIRECTORY [--check]

Uses existing COCO2017 activations; never trains CLIP or an SAE.
Runs Hungarian, Greedy, top-k, Sinkhorn, CCA, Procrustes, and the existing
many-to-many candidates, then compares raw/centered/standardized retrieval.
Object-removal evaluation is skipped. All original fitting settings are kept.

--check verifies required cache files and prints paths without running experiments.
MM_SAE_SOURCE_RUN overrides the default runs/elice-rq1-lexicon2 cache.
MM_SAE_PYTHON overrides the default .venv/bin/python interpreter.
USAGE
  if [[ "${1:-}" == "--help" ]]; then exit 0; else exit 2; fi
fi
if [[ $# -eq 2 && "$2" != "--check" ]]; then
  echo "The only optional second argument is --check." >&2
  exit 2
fi

mapping_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mapping_python="${MM_SAE_PYTHON:-$mapping_repo_root/.venv/bin/python}"
mapping_source="${MM_SAE_SOURCE_RUN:-$mapping_repo_root/runs/elice-rq1-lexicon2}"
export PYTHONPATH="$mapping_repo_root/src:$mapping_repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$mapping_python" - "$mapping_repo_root" "$mapping_source" "$1" "${2:-}" <<'PY'
from pathlib import Path
import subprocess
import sys
import yaml

root, source, output = (Path(p).resolve() for p in sys.argv[1:4])
required = [source / "dataset.json", source / "models/frozen.json"]
for split in ("train2017", "val2017"):
    required += [source / "activations" / split / f"{side}.npz" for side in ("image", "text")]
    required += [source / "index" / split / name for name in (
        "images.json", "concept_ids.json", "parents.npy", "presence.npy", "mentions.npy")]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit("Required COCO cache files are missing. Copy the existing cache first.\n" + "\n".join(missing))
if output == source or output in source.parents or source in output.parents:
    raise SystemExit("The output directory must be separate from the input cache.")

suite = yaml.safe_load((root / "configs/mapping-server.yaml").read_text())
ablation = yaml.safe_load((root / "configs/mapping-ablation-server.yaml").read_text())
suite.update(source_run=str(source), output=str(output / "suite"), skip_removal=True)
ablation.update(source_run=str(source), parent_run=str(output / "suite"), output=str(output / "ablation"))
jobs = (("mapping_suite", "suite.yaml", suite), ("mapping_ablation", "ablation.yaml", ablation))
print(f"Input cache: {source}", flush=True)
for _, _, cfg in jobs:
    print(f"Report: {cfg['output']}/report.html", flush=True)
    print(f"Progress: {cfg['output']}/progress.json", flush=True)
if sys.argv[4] == "--check":
    print("Required cache files exist. No inference, training, evaluation, or file writes were performed.")
    raise SystemExit(0)

output.mkdir(parents=True, exist_ok=True)
for _, filename, cfg in jobs:
    path = output / filename
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise SystemExit(f"Configuration changed. Choose a fresh output directory instead of {output}.")
for module, filename, cfg in jobs:
    path = output / filename
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run([sys.executable, "-u", "-m", f"experiments.{module}.run", "--config", str(path)],
                   cwd=root, check=True)
PY
