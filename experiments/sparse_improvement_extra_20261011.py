"""Existing 512-rank rotations, equal support evaluation without additional training."""
from experiments.sparse_improvement_20261011 import *
from experiments.rotation_mass_audit_20261011 import prune
if __name__=='__main__':
 try:
  e=Eval();base=e.matrices(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz',512);counts=[]
  for method in ['covariance-svd','correlation-cayley']:
   for seed in range(10):
    r=a.tensor(np.load(SRC/f'rank512/{method}-{seed}-oriented.npz')['rotation']);ws={s:base[s]@r for s in S}
    e.evaluate('rank512',method,seed,'dense',ws)
    for k in BUDGETS:e.evaluate('rank512',method,seed,f'coefficient-k{k}',{s:hard(ws[s],k) for s in S},ws)
    for frac in MASS:
     stats={s:prune(ws[s],frac)[1] for s in S};counts.append(dict(method=method,seed=seed,mass=frac,counts=stats))
  save(OUT/'rank512-energy-counts.json',counts);save(OUT/'rank512-complete.json',dict(time=time.time()))
 except Exception as ex:save(OUT/'error-rank512.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
