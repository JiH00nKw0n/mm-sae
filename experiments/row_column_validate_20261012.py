from experiments.row_column_rotation_20261012 import *
def main():
 e=RC_Eval();eye=torch.eye(256,device='cuda',dtype=torch.float64);rows=[]
 for path in sorted((OUT/'fits').glob('*.npz')):
  r=a.tensor(np.load(path)['rotation']);q=json.loads(path.with_suffix('.json').read_text());order=q['order'];rq=q['row_q'];mode=q['column_mode']
  def step(r):
   z={s:e.base[s][None]@r[None] for s in S};t={s:target(z[s],order,rq,mode) for s in S};return polar(sum(e.base[s].T[None]@t[s] for s in S))[0]
  nxt=step(r);nxt2=step(nxt);period_error=None
  if q.get('cycle'):
   rr=r
   for _ in range(q['cycle']['period']):rr=step(rr)
   period_error=float((rr-r).norm())
   assert period_error<1e-8,(path,period_error)
  rows.append(dict(name=path.stem,stop=q['stop'],orthogonality=float((r.T@r-eye).abs().max()),next_step=float((nxt-r).norm()),two_step_return=float((nxt2-r).norm()),cycle_return_error=period_error,objective_increases=sum(x['total_increase']>1e-8 for x in q['history'])))
 assert len(rows)==240
 assert max(x['orthogonality'] for x in rows)<1e-9
 err=0.
 for q in list((OUT/'fits').glob('*.npz'))[::20]:
  r=a.tensor(np.load(q)['rotation']);sc={s:e.x[s][:100]@e.base[s] for s in S};norm=lambda v:v/v.norm(dim=1,keepdim=True)
  err=max(err,float((norm(sc['image'])@norm(sc['text']).T-norm(sc['image']@r)@norm(sc['text']@r).T).abs().max()))
 # Compare identical baseline computations against the previous report.
 old=ROOT/'runs/row-rotation-2026-10-12/eval';checks=[]
 for n,prev in [('covariance','covariance'),('fixed64','fixed64'),('row1','row1-covariance')]:
  for seed in range(10):
   x=json.loads((OUT/'eval'/f'{n}-k64-s{seed}.json').read_text());y=json.loads((old/f'{prev}-positive-k64-s{seed}.json').read_text())
   de=max(abs(x['retrieval'][s]['recall'][k]-y['retrieval'][s]['recall'][k]) for s in x['retrieval'] for k in ['1','5','10']);assert de==0
   assert x['matching']==y['matching'];checks.append(dict(method=n,seed=seed,max_retrieval_diff=de,matching_identical=True))
 save(OUT/'validation.json',dict(fits=rows,cosine_error=err,baseline_reproduction=checks,evaluations=len(list((OUT/'eval').glob('*.json')))))
if __name__=='__main__':main()
