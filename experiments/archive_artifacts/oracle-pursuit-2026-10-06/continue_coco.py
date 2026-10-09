"""One-off conditional continuation, not a recurring notification schedule."""
import json
import os
from pathlib import Path
import subprocess
import time
root=Path('/home/elicer/jihoonkwon/mm_sae_grahtp_20261006')
status=root/'results/cc3m-coco/progress.json'
pid=int((root/'logs/cc3m.pid').read_text())
while True:
    record=json.loads(status.read_text()) if status.exists() else {}
    if record.get('state')=='completed':
        elapsed=record['seconds']
        if elapsed>7200:
            print(json.dumps(dict(state='coco_deferred',reason='CC3M exceeded two-hour early-completion budget',seconds=elapsed)),flush=True)
            break
        print(json.dumps(dict(state='starting_coco',cc3m_seconds=elapsed)),flush=True)
        config=root/'config/oracle-pursuit-coco-80g.yaml'
        with (root/'logs/coco.log').open('a') as log:
            proc=subprocess.Popen([str(root/'env/bin/python'),'-u','-m','experiments.oracle_sets.run_pursuit','--config',str(config)],stdout=log,stderr=subprocess.STDOUT,cwd=root/'code')
            (root/'logs/coco.pid').write_text(str(proc.pid))
            result=proc.wait()
        print(json.dumps(dict(state='coco_finished',exit_code=result)),flush=True)
        break
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        print(json.dumps(dict(state='coco_not_started',reason='CC3M process stopped before completion')),flush=True)
        break
    time.sleep(15)
