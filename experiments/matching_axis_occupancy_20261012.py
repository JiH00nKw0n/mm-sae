"""Frozen historical rotations. Reproduce retrieval/matching and add axis occupancy."""
from experiments.score_validation_20261012 import *
import hashlib
OUT=ROOT/'runs/matching-axis-occupancy-2026-10-12'
OUT.mkdir(exist_ok=True);(OUT/'eval').mkdir(exist_ok=True)
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)

def plan():
 p=[dict(key='cov-dense',name='공분산 회전',family='cov',cut='dense',k=None)]
 for k in [16,32,64,128,256]:p.append(dict(key=f'cov-k{k}',name='공분산 회전 후 한 번 자르기',family='cov',cut='top',k=k))
 for k in [16,32,64,128,256]:p.append(dict(key=f'alt-k{k}',name='축당 개수 고정 교대',family='fixed',cut='top',k=k))
 for k in [99,95,90]:p.append(dict(key=f'cov-m{k}',name='공분산 회전 후 한 번 자르기',family='cov',cut='mass',k=k))
 for k in [99,95,90]:p.append(dict(key=f'alt-m{k}',name='축당 제곱합 비율 교대',family='energy',cut='mass',k=k))
 for fam,name in [('row','특징당 한 축 목표'),('cycle','특징당 한 축 선택 후 열 90% 교대의 목적값 최저 상태'),('score','표본 점수 회전'),('oracle','50개 정답 평균 차이 회전')]:
  for cut,k in [('dense',None),('top',64)]:p.append(dict(key=f'{fam}-{cut}',name=name,family=fam,cut=cut,k=k))
 p.append(dict(key='hungarian',name='헝가리안 학습 상관 상위 256쌍',family='hungarian',cut='matched',k=1));return p

def historical(e,c,seed):
 fam,cut,k=c['family'],c['cut'],c['k'];root=ROOT/'runs';meta={};sg='saved'
 if fam=='cov':
  path=SRC/f'rank256/covariance-svd-{seed}-oriented.npz'
  if cut=='mass':old=root/f'energy-alternating-2026-10-11/eval/covariance-svd-mass{k}-s{seed}.json'
  else:old=root/'sparse-improvement-2026-10-11/stage1'/f'covariance-svd-s{seed}-{("dense" if cut=="dense" else "coefficient-k"+str(k))}.json'
 elif fam=='fixed':path=root/'sparse-improvement-2026-10-11/fits'/f'alternating-k{k}-s{seed}.npz';old=root/'sparse-improvement-2026-10-11/stage2'/f'alternating-s{seed}-k{k}.json';sg='training_skew'
 elif fam=='energy':path=root/'energy-alternating-2026-10-11/fits'/f'ind{k}-s{seed}.npz';old=root/'energy-alternating-2026-10-11/eval'/f'ind{k}-s{seed}.json';sg='training_skew'
 elif fam=='row':path=root/'row-rotation-2026-10-12/fits'/f'row1-covariance-s{seed}.npz';old=root/'row-rotation-2026-10-12/eval'/f'row1-covariance-positive-{("dense" if cut=="dense" else "k64")}-s{seed}.json';sg='positive_coefficient_energy'
 elif fam=='cycle':
  path=root/'feature-redundancy-cycle-2026-10-12/fits'/f'cycle-best-s{seed}.npz';old=root/'score-validation-2026-10-12/eval'/f'cycle-best-{("dense" if cut=="dense" else "column64")}-s{seed}.json';sg='positive_coefficient_energy'
  meta=json.loads((root/'row-column-rotation-2026-10-12/fits'/f'rowfirst-row1-mass90-covariance-s{seed}.json').read_text());meta={key:meta[key] for key in ['steps','stop','cycle'] if key in meta}
 elif fam=='score':path=ref('score500',seed);old=root/'score-validation-2026-10-12/eval'/f'score500-{("dense" if cut=="dense" else "column64")}-s{seed}.json';sg='positive_coefficient_energy'
 elif fam=='oracle':path=root/'oracle-direction-rotation-2026-10-12/fits'/f'mean-all50-s{seed}.npz';old=root/'oracle-direction-rotation-2026-10-12/eval'/f'mean-all50-{("dense" if cut=="dense" else "column64")}-s{seed}.json'
 else:
  old=root/'rotation-mass-audit-2026-10-11/rank256/hungarian-toplinks-dense.json';path=Path(next(v['path'] for v in e.cfg['models'] if v['key']=='hungarian'))
  mp=np.load(path)['mapping'];i,j=np.nonzero(mp)
  with np.load(Path(e.cfg['parent_run'])/'moments.npz') as z:
   ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
  sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:];order=np.argsort(-cv[i,ni+j],kind='stable');i=i[order];j=j[order]
  ws={'image':torch.zeros((mp.shape[0],256),device='cuda',dtype=torch.float64),'text':torch.zeros((mp.shape[1],256),device='cuda',dtype=torch.float64)};ws['image'][torch.as_tensor(i[:256],device='cuda'),torch.arange(256,device='cuda')]=1;ws['text'][torch.as_tensor(j[:256],device='cuda'),torch.arange(256,device='cuda')]=1
  return ws,old,dict(source=str(path),sign='positive original features',steps=0)
 assert path.exists() and old.exists(),(str(path),str(old))
 if path.with_suffix('.json').exists() and not meta:
  f=json.loads(path.with_suffix('.json').read_text());meta={key:f[key] for key in ['steps','stop'] if key in f}
 r=a.tensor(np.load(path)['rotation'])
 if sg=='training_skew':r=e.oriented_rotation(r)
 elif sg=='positive_coefficient_energy':r=r*positive_sign({s:e.base[s]@r for s in S})
 ws={s:e.base[s]@r for s in S}
 if cut=='top':ws={s:hard(w,k) for s,w in ws.items()}
 elif cut=='mass':ws={s:truncate(w[None],fraction=k/100)[0] for s,w in ws.items()}
 return ws,old,dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sign=sg,**meta)

def old_counts(old):
 rows=old.get('matching',old.get('rows'));res=[]
 for row in rows[:20]:
  if 'count' in row:res.append(int(row['count']))
  elif 'per_category' in row:res.append(sum(row['per_category']))
  else:
   spec=json.loads((ROOT/'runs/score-validation-2026-10-12/categories.json').read_text());res.append(sum(row['match'][i] for i in spec['groups']['old50']))
 return res

def main():
 pp=plan();save(OUT/'protocol.json',dict(rank=256,conditions=pp,categories=50,splits=20,seeds=10,new_training=False,seed_exception='Hungarian deterministic, evaluate one solution on20splits; no10 independent fits',sharing='A successfully matched category shares if >=2 successfully matched categories choose its axis. Unmatched categories excluded. Count all categories in a shared group.',unique08='Unique among ALL matched categories first, then require both AUROC>=.8; do not recalculate uniqueness after thresholding.',responsive='Per category count AUROC>=.8 axes. Axis multiplicity computed independently on each modality over the same50 categories.',ratio='100 * axes responding to >=2 categories / axes responding to >=1 category, calculated per seed/split then averaged',orientation='Preserve each historical evaluation sign exactly; original training-skew for earlier fixed/energy fits, coefficient-positive signs for later row/score fits, annotation-positive assigned axes for oracle.',energy='independent99/95/90 saved fits, not99-to95-to90 continuation',oracle='previous mean-all50 diagnostic, not LDA or25category holdout',cuts='alternating fits evaluated after same final cut; row/score/oracle dense or top64 as requested'))
 e=RC_Eval();save(OUT/'categories.json',[v for v in e.data['concepts'] if v['id'] in e.data['metadata']['eligible_category_ids']]);done=0
 for c in pp:
  for seed in range(1 if c['family']=='hungarian' else 10):
   dest=OUT/'eval'/f'{c["key"]}-s{seed}.json'
   if dest.exists():done+=1;continue
   ws,oldpath,meta=historical(e,c,seed);sc={s:e.x[s]@w for s,w in ws.items()};ret=retrieval(sc,e.parents);rows=[]
   for split,pair0 in enumerate(e.splits[:20]):
    au=[a.gpu_auc(sc[s][ix],e.labels[s][ix]) for s,ix in zip(S,pair0)];counts={}
    for side,q in zip(S,au):
     n=(q>=.8).sum(1);anyn=int((n>=1).sum());multi=int((n>=2).sum());counts[side]=dict(responsive=float((q>=.8).sum(0).double().mean()),reactive_axes=anyn,multiple_axes=multi,multiple_percent=100*multi/anyn if anyn else 0.)
    valid=torch.isfinite(au[0])&torch.isfinite(au[1]);has=valid.any(0)
    for q in au:q[~valid]=-torch.inf
    ind=[q.argmax(0) for q in au];col=torch.arange(50,device='cuda');best=[q[k,col] for q,k in zip(au,ind)];match=(ind[0]==ind[1])&has
    freq=torch.bincount(ind[0][match],minlength=256);shared=match&(freq[ind[0]]>=2);solo=match&(freq[ind[0]]==1);solo08=solo&(best[0]>=.8)&(best[1]>=.8)
    assert int(shared.sum()+solo.sum())==int(match.sum())
    rows.append(dict(split=split,match=int(match.sum()),shared=int(shared.sum()),solo=int(solo.sum()),solo08=int(solo08.sum()),image=counts['image'],text=counts['text'],image_axes=ind[0].cpu().tolist(),text_axes=ind[1].cpu().tolist(),image_auc=best[0].cpu().tolist(),text_auc=best[1].cpu().tolist(),matched=match.int().cpu().tolist(),shared_categories=shared.int().cpu().tolist(),solo_categories=solo.int().cpu().tolist()))
   old=json.loads(oldpath.read_text());expected=old_counts(old);actual=[r['match'] for r in rows];md=max(abs(x-y) for x,y in zip(expected,actual));rd=max(abs(ret[d]['recall']['5']-old['retrieval'][d]['recall']['5']) for d in ret)
   check=dict(old_file=str(oldpath),old_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),matching_max_difference=md,recall5_max_difference=rd,old_match_counts=expected,new_match_counts=actual)
   if md or rd>1e-10:
    save(OUT/'reproduction-mismatch.json',dict(condition=c,seed=seed,check=check));raise AssertionError((c['key'],seed,check))
   save(dest,dict(condition=c,seed=seed,rows=rows,retrieval=ret,features={s:float((w!=0).sum(0).double().mean()) for s,w in ws.items()},provenance=meta,reproduction=check));done+=1;log(stage='evaluation',completed=done,total=251,condition=c['key'],seed=seed)
 save(OUT/'complete.json',dict(evaluations=done,conditions=len(pp),all_matching_and_recall5_reproduced=True,time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
