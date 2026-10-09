import json
import subprocess
import sys
import time
from pathlib import Path

root=Path('/mnt/working/mm-sae')
out=root/'runs/within-modal-groups-2026-10-09'
assert (out/'complete.json').exists(), 'Do not overlap with the main GPU run.'
start=time.time()
for condition in ['coco-coco','cc3m-coco','cc3m-cc3m']:
    subprocess.run([sys.executable,'-u','experiments/within_modal_groups_extended_20261009.py',
                    '--condition',condition,'--binary-only'],cwd=root,check=True)
(out/'extended-complete.json').write_text(json.dumps(dict(completed=True,
    elapsed_seconds=time.time()-start,time_unix=time.time()),indent=2))
