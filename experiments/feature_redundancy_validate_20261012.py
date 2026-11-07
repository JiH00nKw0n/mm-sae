from experiments.feature_redundancy_cycle_20261012 import *

def main():
 e=RC_Eval();checks=[]
 for side in S:
  raw=full_sae(e,side);ids=e.data['feature_ids'][side];err=float(abs(raw[:,ids]-e.data['raw'][side]).max());assert err==0
  z=np.load(OUT/'analysis'/f'raw-{side}.npz')['auc'];cols=np.linspace(0,len(ids)-1,64).astype(int)
  au=a.gpu_auc(e.x[side][:,cols],e.labels[side]);au[~torch.isfinite(au)]=.5;aucerr=float(np.abs(au.cpu().numpy()-z[-1,ids[cols]]).max());assert aucerr<1e-12
  checks.append(dict(side=side,raw_alignment_error=err,standardized_auc_error=aucerr))
  diagnostics=json.loads((OUT/'analysis'/f'raw-correlation-{side}.json').read_text())
  for item in diagnostics:
   ids=np.array(item['feature_ids']);c=[v['name'] for v in json.loads((OUT/'concepts.json').read_text())].index(item['category'])
   if len(ids)<2:continue
   x=raw[:,ids].toarray()
   for tag,xx in [('all_samples',x),('category_present',x[e.labels[side][:,c].cpu().numpy()>0])]:
    if item[tag] is None:continue
    xx=xx-xx.mean(0);norm=np.linalg.norm(xx,axis=0);valid=norm>1e-12;xx/=np.maximum(norm,1e-30);corr=xx.T@xx;mask=np.triu(np.ones_like(corr,dtype=bool),1)&valid[:,None]&valid[None,:];v=corr[mask]
    item[tag]['max_correlation']=float(v.max());item[tag]['fraction_ge08']=float((v>=.8).mean())
  save(OUT/'analysis'/f'raw-correlation-{side}.json',diagnostics)
 damp=[]
 for seed in range(10):
  path=OUT/'fits'/f'damped-s{seed}.npz';r=a.tensor(np.load(path)['rotation']);q=json.loads(path.with_suffix('.json').read_text());orth=float((r.T@r-torch.eye(256,device='cuda')).abs().max());assert orth<1e-9
  rr=r;errors=[];period=None
  if q['stop']!='fixed_point':
   for step in range(1,257):
    rr=polar(((rr+update(e.base,rr))/2)[None])[0];er=float((rr-r).norm());errors.append(er)
    if step>1 and er<1e-8:period=step;break
  damp.append(dict(seed=seed,stop=q['stop'],orthogonality_error=orth,posthoc_cycle_period_up_to256=period,cycle_return_error=errors[-1] if errors else None))
 save(OUT/'validation.json',dict(raw_checks=checks,damping=damp,phase0_reproduction=json.loads((OUT/'cycle-complete.json').read_text())['phase0_reproduced']))

if __name__=='__main__':main()
