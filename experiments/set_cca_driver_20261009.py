import subprocess,sys,time,json
from pathlib import Path
root=Path('/mnt/working/mm-sae');out=root/'runs/set-cca-2026-10-09'
out.mkdir(parents=True,exist_ok=True);start=time.time()
for condition in ['coco-coco','cc3m-coco','cc3m-cc3m']:
    proc=subprocess.run([sys.executable,'-u','experiments/set_cca_20261009.py','--condition',condition],cwd=root)
    if proc.returncode:
        (out/'error.json').write_text(json.dumps(dict(condition=condition,returncode=proc.returncode)))
        raise SystemExit(proc.returncode)
(out/'complete.json').write_text(json.dumps(dict(completed=True,elapsed_seconds=time.time()-start),indent=2))
