"""Sequential jobs, fail visibly, no silent rerun or overlapping GPU workers."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path('/mnt/working/mm-sae')
out = root / 'runs/within-modal-groups-2026-10-09'
out.mkdir(parents=True, exist_ok=True)
start = time.time()
jobs = ['coco-coco', 'cc3m-coco', 'cc3m-cc3m']
for condition in jobs:
    result = subprocess.run([sys.executable, '-u', 'experiments/within_modal_groups_20261009.py',
                             '--condition', condition], cwd=root)
    if result.returncode:
        (out / 'driver-error.json').write_text(json.dumps(dict(condition=condition,
            returncode=result.returncode, time_unix=time.time())))
        raise SystemExit(result.returncode)
(out / 'complete.json').write_text(json.dumps(dict(completed=True, conditions=jobs,
    elapsed_seconds=time.time()-start, time_unix=time.time()), indent=2))
