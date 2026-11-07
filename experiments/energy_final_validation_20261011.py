"""Diagnose saved final iterates without changing the fitted models."""
from experiments.energy_alternating_20261011 import *
with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as f:base={s:a.tensor(f[s][:,:256]) for s in S}
rows=[]
for path in sorted((OUT/'fits').glob('*.npz')):
 meta=json.loads(path.with_suffix('.json').read_text());r=a.tensor(np.load(path)['rotation'])[None]
 fraction=meta.get('fraction');k=meta.get('k')
 if fraction is None and k is None:
  k=int(path.stem.split('-k')[1].split('-s')[0])
 def update(rr):
  z={s:base[s][None]@rr for s in S};sp={s:truncate(z[s],fraction,None if k is None else [k]) for s in S}
  return polar(sum(base[s].T[None]@sp[s] for s in S)),sp
 r1,sp=update(r);r2,sp1=update(r1)
 retained={s:sp[s].square().sum(-2)/(base[s][None]@r).square().sum(-2) for s in S}
 if fraction is not None:assert min(float(v.min()) for v in retained.values())>=fraction-1e-12
 else:assert all(bool(((sp[s]!=0).sum(-2)==k).all()) for s in S)
 rows.append(dict(name=path.stem,stop=meta['stop'],one_step_rotation_change=float((r1-r).norm()),two_step_return_distance=float((r2-r).norm()),support_changes={s:int(((sp[s]!=0)!=(sp1[s]!=0)).sum()) for s in S},retained_energy_min={s:float(retained[s].min()) for s in S},orthogonality_error=float((r.transpose(-2,-1)@r-torch.eye(256,device='cuda')).abs().max())))
save(OUT/'final-iterate-validation.json',rows)
e=Eval();valid=[]
for i,pair in enumerate(e.splits[:20]):
 for s,ix in zip(S,pair):
  pos=e.labels[s][ix].sum(0);neg=len(ix)-pos
  assert bool(((pos>0)&(neg>0)).all()),(i,s,pos)
  valid.append(dict(split=i,side=s,min_positive=int(pos.min()),min_negative=int(neg.min())))
save(OUT/'label-validation.json',valid)
print('validated',len(rows),'saved fits',flush=True)
