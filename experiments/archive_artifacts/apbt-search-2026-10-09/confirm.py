from pathlib import Path
import subprocess,sys,shutil,json,time
R=Path('/mnt/working/mm-sae');B=R/'runs/apbt-search-2026-10-09';start=time.monotonic()
# Same seed0 subset and exact same training config: reuse alpha2 model from round2.
src=B/'round2/cc3m-coco/seed0';dst=B/'confirmation/cc3m-coco/seed0';dst.mkdir(parents=True,exist_ok=True)
for suffix in ['.npz','-fit.json','-history.json']:
 p=dst/('positive_reconstruct_2'+suffix)
 if not p.exists():shutil.copy2(src/p.name,p)
jobs=[('cc3m-coco',0),('cc3m-coco',1),('cc3m-coco',2),('coco-coco',0),('cc3m-cc3m',0)]
for cond,seed in jobs:
 args=[sys.executable,'-u','experiments/apbt_search_20261009.py','--round','confirmation','--condition',cond,'--seed',str(seed),'--epochs','60','--specs',str(B/'confirmation-specs.json')]
 subprocess.run(args,cwd=R,check=True)
 subprocess.run(args+['--evaluate','test'],cwd=R,check=True)
 print(json.dumps({'confirmed':cond,'seed':seed,'elapsed':time.monotonic()-start}),flush=True)
(B/'confirmation-complete.json').write_text(json.dumps({'completed':True,'jobs':jobs,'elapsed_seconds':time.monotonic()-start}))
