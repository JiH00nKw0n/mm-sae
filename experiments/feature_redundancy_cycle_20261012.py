"""Frozen SAE redundancy diagnosis and two explicitly requested cycle treatments."""
from experiments.row_column_rotation_20261012 import *
import experiments.row_column_rotation_20261012 as rc
from scipy import sparse
from scipy.sparse.csgraph import connected_components
OUT=ROOT/'runs/feature-redundancy-cycle-2026-10-12'
PREV=ROOT/'runs/row-column-rotation-2026-10-12'
for sub in ['fits','eval','analysis','checkpoints']:(OUT/sub).mkdir(parents=True,exist_ok=True)
energy.OUT=OUT
rc.OUT=OUT
TH=[.7,.8,.9]

def log(**v):
 save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)

def update(base,r):
 z={s:base[s][None]@r[None] for s in S};t={s:target(z[s],'rowfirst',1,'mass90') for s in S}
 return polar(sum(base[s].T[None]@t[s] for s in S))[0]

def objective(base,r):
 z={s:base[s][None]@r[None] for s in S};return float(residual(z,{s:target(z[s],'rowfirst',1,'mass90') for s in S})[0])

def cycles_and_damping(e):
 cycles=[]
 for seed in range(10):
  path=PREV/'fits'/f'rowfirst-row1-mass90-covariance-s{seed}.npz';r=a.tensor(np.load(path)['rotation']);q=json.loads(path.with_suffix('.json').read_text());period=q['cycle']['period'];rs=[];objs=[]
  for phase in range(period):
   rs.append(r);objs.append(objective(e.base,r));np.savez_compressed(OUT/'fits'/f'cycle-phase{phase}-s{seed}.npz',rotation=r.cpu().numpy());r=update(e.base,r)
  assert float((r-rs[0]).norm())<1e-8
  dm=torch.stack([torch.stack([(r-t).norm() for t in rs]) for r in rs]).cpu().numpy()
  best=int(np.argmin(objs));np.savez_compressed(OUT/'fits'/f'cycle-best-s{seed}.npz',rotation=rs[best].cpu().numpy())
  cycles.append(dict(seed=seed,period=period,objective=objs,best_phase=best,rotation_distances=dm.tolist(),relative_max_rotation_distance=float(dm.max()/16),return_error=float((r-rs[0]).norm())))
 save(OUT/'cycles.json',cycles)
 # Midpoint followed by polar projection preserves the shared orthogonal constraint.
 for first in [0,5]:
  seeds=[s for s in range(first,first+5) if not (OUT/'fits'/f'damped-s{s}.npz').exists()]
  if not seeds:continue
  r=torch.stack([a.tensor(np.load(OUT/'fits'/f'cycle-phase0-s{s}.npz')['rotation']) for s in seeds]);hist=[[] for _ in seeds];recent=[]
  for j,s in enumerate(seeds):
   cp=OUT/'checkpoints'/f'damped-s{s}.npz'
   if cp.exists():r[j]=a.tensor(np.load(cp)['rotation']);hist[j]=json.loads(cp.with_suffix('.json').read_text())
  for it in range(10000):
   z={s:e.base[s][None]@r for s in S};t={s:target(z[s],'rowfirst',1,'mass90') for s in S};obj=residual(z,t)
   proposal=polar(sum(e.base[s].T[None]@t[s] for s in S));rn=polar((r+proposal)/2)
   zn={s:e.base[s][None]@rn for s in S};end=residual(zn,{s:target(zn[s],'rowfirst',1,'mass90') for s in S});delta=(rn-r).norm(dim=(1,2));vals=torch.stack([obj,end,delta],1).cpu().tolist();done=[]
   cyc=[None]*len(seeds)
   if it%100==0 and len(recent)>=48:
    for period in range(2,17):
     ds=torch.stack([(rn-recent[-k*period]).norm(dim=(1,2)) for k in [1,2,3]]).max(0).values.cpu().tolist()
     for j,d in enumerate(ds):
      if cyc[j] is None and d<1e-10 and vals[j][2]>1e-5:cyc[j]=period
   recent.append(rn.clone());recent=recent[-48:]
   for j,s in enumerate(seeds):
    hist[j].append(dict(step=len(hist[j])+1,before=vals[j][0],after=vals[j][1],rotation_change=vals[j][2]))
    conv=len(hist[j])>10 and max(h['rotation_change'] for h in hist[j][-5:])<1e-5
    if conv or cyc[j] or len(hist[j])>=10000:
     np.savez_compressed(OUT/'fits'/f'damped-s{s}.npz',rotation=rn[j].cpu().numpy());save(OUT/'fits'/f'damped-s{s}.json',dict(seed=s,stop='fixed_point' if conv else ('cycle' if cyc[j] else 'iteration_limit'),period=cyc[j],steps=len(hist[j]),history=hist[j],orthogonality_error=float((rn[j].T@rn[j]-torch.eye(256,device='cuda')).abs().max())));done.append(j)
   if it%100==0:
    for j,s in enumerate(seeds):np.savez(OUT/'checkpoints'/f'damped-s{s}.npz',rotation=rn[j].cpu().numpy());save(OUT/'checkpoints'/f'damped-s{s}.json',hist[j])
    log(stage='half_update',seeds=seeds,steps=[len(h) for h in hist],changes=[v[2] for v in vals])
   keep=[j for j in range(len(seeds)) if j not in done]
   if not keep:break
   r=rn[keep];seeds=[seeds[j] for j in keep];hist=[hist[j] for j in keep];recent=[v[keep] for v in recent]
 for q in cycles:
  seed=q['seed']
  for name in [f'cycle-phase{i}' for i in range(q['period'])]+['cycle-best','damped']:
   path=OUT/'fits'/f'{name}-s{seed}.npz';e.run_one(name,seed,a.tensor(np.load(path)['rotation']),str(path))
 # Phase zero must reproduce the already reported case, including all splits.
 for seed in range(10):
  for cut in ['k64','mass90']:
   old=json.loads((PREV/'eval'/f'rowfirst-row1-mass90-covariance-{cut}-s{seed}.json').read_text());new=json.loads((OUT/'eval'/f'cycle-phase0-{cut}-s{seed}.json').read_text())
   assert old['matching']==new['matching'] and old['retrieval']==new['retrieval']
 save(OUT/'cycle-complete.json',dict(phases=sum(v['period'] for v in cycles),phase0_reproduced=True))

def auc_chunks(x,labels,chunk=192):
 return torch.cat([a.gpu_auc(x[:,i:i+chunk],labels) for i in range(0,x.shape[1],chunk)])

def full_sae(e,side):
 src=Path(e.cfg['source_run']);idx=src/'index/val2017';records=json.loads((idx/'images.json').read_text());keep=np.isin([v['image_id'] for v in records],e.data['metadata']['half_a_image_ids']+e.data['metadata']['half_b_image_ids']);par=np.load(idx/'parents.npy');rows=np.flatnonzero(keep if side=='image' else keep[par]);z=sparse.load_npz(src/'activations/val2017'/f'{side}.npz').tocsr()[rows]
 assert len(rows)==len(e.x[side]);return z

def raw_diagnostics(e):
 eligible=np.isin([c['id'] for c in e.data['concepts']],e.data['metadata']['eligible_category_ids']);concepts=[c for c,k in zip(e.data['concepts'],eligible) if k];save(OUT/'concepts.json',concepts)
 for si,s in enumerate(S):
  path=OUT/'analysis'/f'raw-{s}.npz'
  if path.exists():continue
  raw=full_sae(e,s);active=np.flatnonzero(np.asarray(raw.getnnz(axis=0)).ravel()>0);nfeat=raw.shape[1];result=np.full((21,nfeat,50),.5,dtype=np.float64)
  for start in range(0,len(active),192):
   ids=active[start:start+192];x=a.tensor(raw[:,ids].toarray())
   for split in range(21):
    ix=e.splits[split][si] if split<20 else torch.arange(len(x),device='cuda');v=a.gpu_auc(x[ix],e.labels[s][ix]);v[~torch.isfinite(v)]=.5;result[split,ids]=v.cpu().numpy()
   if start%960==0:log(stage='raw_feature_auc',side=s,completed=min(start+192,len(active)),active_features=len(active),total_features=nfeat)
  np.savez_compressed(path,auc=result,feature_ids=e.data['feature_ids'][s]);del result
 for s in S:
  dest=OUT/'analysis'/f'raw-correlation-{s}.json'
  if dest.exists():continue
  raw=full_sae(e,s);au=np.load(OUT/'analysis'/f'raw-{s}.npz')['auc'][-1];rows=[]
  for c,concept in enumerate(concepts):
   for th in TH:
    ids=np.flatnonzero(au[:,c]>=th);x=a.tensor(raw[:,ids].toarray());pos=e.labels[s][:,c]>0
    item=dict(category=concept['name'],threshold=th,n_features=len(ids),feature_ids=ids.tolist())
    for tag,xx in [('all_samples',x),('category_present',x[pos])]:
     if len(ids)<2:item[tag]=None;continue
     xx=xx-xx.mean(0);norm=xx.norm(dim=0);valid=norm>1e-12;xx=xx/norm.clamp_min(1e-30);corr=xx.T@xx;mask=torch.triu(torch.ones_like(corr,dtype=torch.bool),diagonal=1)&valid[:,None]&valid[None,:];v=corr[mask].cpu().numpy()
     if not len(v):item[tag]=None;continue
     graph=(corr>=.9).cpu().numpy();np.fill_diagonal(graph,False);ncomp,lab=connected_components(sparse.csr_matrix(graph),directed=False);sizes=np.bincount(lab);nearest=corr.clone();nearest.fill_diagonal_(-torch.inf);nearest=nearest.max(0).values[valid].cpu().numpy()
     item[tag]=dict(n_valid_pairs=len(v),mean=float(v.mean()),median=float(np.median(v)),q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9)),fraction_ge09=float((v>=.9).mean()),fraction_ge095=float((v>=.95).mean()),nearest_median=float(np.median(nearest)),fraction_features_with_partner_ge09=float((nearest>=.9).mean()),components_at09=int(ncomp),component_sizes=sorted(sizes.tolist(),reverse=True))
    rows.append(item)
   log(stage='raw_correlation',side=s,category=concept['name'])
  save(dest,rows)

def dispersion(e):
 concepts=json.loads((OUT/'concepts.json').read_text());raw={s:np.load(OUT/'analysis'/f'raw-{s}.npz')['auc'] for s in S}
 methods=['covariance','fixed64','row1','rowfirst-row1-mass90-covariance']
 for name in methods:
  for seed in range(10):
   dest=OUT/'analysis'/f'dispersion-{name}-s{seed}.json'
   if dest.exists():continue
   source=json.loads((PREV/'eval'/f'{name}-k64-s{seed}.json').read_text())['meta']['source'];r=a.tensor(np.load(source)['rotation']);w={s:e.base[s]@r for s in S};sg=positive_sign(w);w={s:col_cut((w[s]*sg)[None],'k64')[0] for s in S};rows=[];full=[]
   for si,s in enumerate(S):
    sc=e.x[s]@w[s];support=(w[s]!=0).double();positive=(w[s]>0).double();ids=e.data['feature_ids'][s]
    for split in range(20):
     ix=e.splits[split][si];axis=a.gpu_auc(sc[ix],e.labels[s][ix]);axis08=(axis>=.8);auc=raw[s][split,ids]
     for th in TH:
      resp=a.tensor((auc>=th).astype(float));hit=support.T@resp;poshit=positive.T@resp;tot=(raw[s][split]>=th).sum(0)
      for c,concept in enumerate(concepts):
       rows.append(dict(side=s,split=split,category=concept['name'],threshold=th,total_responsive_features=int(tot[c]),eligible_responsive_features=int(resp[:,c].sum()),used_responsive_features=int(((support.sum(1)>0)&(resp[:,c]>0)).sum()),spread_axes=int((hit[:,c]>0).sum()),positive_spread_axes=int((poshit[:,c]>0).sum()),responsive_axes=int(axis08[:,c].sum()),responsive_axes_without_features=int((axis08[:,c]&(hit[:,c]==0)).sum()),responsive_axes_with_features=int((axis08[:,c]&(hit[:,c]>0)).sum())))
    # Full-population diagnostics for positive axes lacking any raw-positive feature at .8.
    axis=a.gpu_auc(sc,e.labels[s]);au=a.tensor(raw[s][-1,ids]);hit=support.T@(au>=.8).double();missing=(axis>=.8)&(hit==0)
    negsupport=(w[s]<0).double();signed=positive.T@(au>=.8).double()+negsupport.T@((1-au)>=.8).double()
    # Order each retained coefficient by its marginal standard deviation contribution.
    score=w[s].abs()*e.x[s].std(0,unbiased=False)[:,None];order=score.argsort(dim=0,descending=True,stable=True)
    prefix={}
    for k in [1,4,8,16,32]:
     ww=torch.zeros_like(w[s]).scatter_(0,order[:k],w[s].gather(0,order[:k]));prefix[k]=a.gpu_auc(e.x[s]@ww,e.labels[s])
    # Restrict to cases selected without changing their already learned coefficients.
    for ax,c in torch.nonzero(missing).cpu().tolist():
     sel=torch.nonzero(w[s][:,ax]).flatten();effect=torch.where(w[s][sel,ax,None]>=0,au[sel],1-au[sel]);best=float(effect[:,c].max())
     full.append(dict(side=s,category=concepts[c]['name'],axis=ax,auc=float(axis[ax,c]),n_raw_positive07=int((au[sel,c]>=.7).sum()),n_raw_positive08=int((au[sel,c]>=.8).sum()),n_signed_individual08=int(signed[ax,c]),max_signed_individual_auc=best,prefix_auc={str(k):float(prefix[k][ax,c]) if torch.isfinite(prefix[k][ax,c]) else .5 for k in prefix},feature_ids=e.data['feature_ids'][s][sel.cpu().numpy()].tolist(),coefficients=w[s][sel,ax].cpu().tolist(),raw_feature_auc=au[sel,c].cpu().tolist()))
   save(dest,dict(method=name,seed=seed,split_rows=rows,full_population_missing_axes=full));log(stage='dispersion',method=name,seed=seed)

def main():
 save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping, frozen PLS256',splits=20,seeds=10,raw_auc='Full original SAE dictionary plus separate mapping-eligible subset; positive activation AUROC; constant features count as 0.5',dispersion='Nonzero coefficients after top64, with positive-coefficient spread recorded separately',correlation='Full COCO validation population, and within category-present observations, feature pairs selected by full-population AUROC; descriptive, not independent validation',missing_axes='Full-population axes AUROC>=0.8 with no raw feature AUROC>=0.8; check negative coefficients and prefix ablations ordered by abs(weight)*evaluation standard deviation; descriptive only',cycle_best='Choose minimal self-consistent target residual over exact cycle, never evaluation metrics',damping='Start at saved cycling state; Rnext=polar((R+Procrustes(target(R)))/2); shared orthogonality retained; cap10000, five changes<1e-5; detect short cycles',scope='No new base space or rotation objective; only requested half-step continuation and cycle selection'))
 e=RC_Eval();log(stage='loaded',shapes={s:list(e.x[s].shape) for s in S})
 raw_diagnostics(e);dispersion(e);cycles_and_damping(e)
 save(OUT/'complete.json',dict(time=time.time(),evaluations=len(list((OUT/'eval').glob('*.json'))),dispersion_files=len(list((OUT/'analysis').glob('dispersion-*.json')))))

if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
