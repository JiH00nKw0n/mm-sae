import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import yaml,json
from experiments.representative_agreement.run import load_config,run
root=Path('/mnt/working/mm-sae')
inputs=[('coco-coco',root/'configs/representative-agreement-local.yaml'),('cc3m-cc3m',root/'runs/cc3m-followup-2026-10-05/configs/cc3m-fit-agreement.yaml'),('cc3m-coco',root/'runs/cc3m-followup-2026-10-05/configs/coco-fit-agreement.yaml')]
for name,path in inputs:
 cfg=load_config(path)
 cfg['models']=[m for m in cfg['models'] if m['key'] in ['hungarian','sparse_cca_16']]
 cfg['populations']=['test']
 cfg['output']=str(root/'runs/representative-agreement-2026-10-09'/name)
 cfg['protocol']={'training':name,'scope':'Independent representative identity agreement in frozen native paired coordinates; not retrieval or target AUROC. Current COCO-Stuff labels, not exact original rebuttal reproduction.'}
 Path(cfg['output']).mkdir(parents=True,exist_ok=True)
 (Path(cfg['output'])/'configuration.json').write_text(json.dumps(cfg,indent=2))
 run(cfg)
