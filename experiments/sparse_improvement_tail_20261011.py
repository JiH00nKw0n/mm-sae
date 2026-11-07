"""Compute seeds 9,8,7 in reverse while the original worker proceeds from 3.
Each completed fit is reusable by the original worker; no new protocol or seeds.
"""
from experiments.sparse_improvement_20261011 import *
if __name__=='__main__':
 try:
  e=Eval()
  for seed in [9,8,7]:
   for k in BUDGETS:
    r,fit=fit_alternating(e.base,k,seed);r=e.oriented_rotation(r);ws={s:e.base[s]@r for s in S}
    e.evaluate('stage2','alternating',seed,f'k{k}',{s:hard(ws[s],k) for s in S},ws,fit)
    assert float((r.T@r-torch.eye(256,device='cuda')).abs().max())<1e-8
   r,fit=fit_l1(e.base,seed);r=e.oriented_rotation(r);ws={s:e.base[s]@r for s in S}
   for k in BUDGETS:e.evaluate('stage2','l1-rotation',seed,f'k{k}',{s:hard(ws[s],k) for s in S},ws,fit)
  save(OUT/'tail-complete.json',dict(time=time.time()))
 except Exception as ex:save(OUT/'error-tail.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
