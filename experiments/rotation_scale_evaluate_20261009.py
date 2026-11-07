from pathlib import Path
import numpy as np,json
import experiments.cca_unsupervised_rotations_20261009 as r
p=r.prior;root=p.ROOT;cond='cc3m-cc3m';out=root/'runs/rotation-logcosh-scale-2026-10-09/evaluated'/cond;out.mkdir(parents=True,exist_ok=True)
cfg=p.old.load_config(root/p.CONFIG[cond]);data,parents=p.eval_population(cfg,'test',out)
z=np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'));w={s:z[s].astype(float) for s in ['image','text']}
test={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
ix=Path(cfg['source_run'])/'index/val2017';presence=np.load(ix/'presence.npy').astype(bool);names=json.loads((root/'runs/cca-171-category-names-2026-10-09/protocol.json').read_text())['prototype_texts']
raw=np.load(root/'runs/cca-unsupervised-rotations-2026-10-09'/cond/'name-activations.npy')
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:ti=z['text_ids']
proto=raw[:,ti]@(w['text']/data['scales']['text'][:,None])-data['means']['text']@(w['text']/data['scales']['text'][:,None])
for seed in [0,1]:
 rr=np.load(root/'runs/rotation-logcosh-scale-2026-10-09'/f'scaled-scaled-{seed}-step300.npz')['rotation']
 r.evaluate(cfg,data,parents,test,proto,presence,names,rr,f'logcosh_scaled-{seed}',out,None)
 print('evaluated',seed,flush=True)
