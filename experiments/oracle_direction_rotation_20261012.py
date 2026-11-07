"""Diagnostic label-directed shared rotation; COCO FIT labels only, frozen CC3M PLS."""
from experiments.row_column_rotation_20261012 import *
from scipy import sparse,linalg
from scipy.optimize import linear_sum_assignment
import hashlib,shutil
OUT=ROOT/'runs/oracle-direction-rotation-2026-10-12'
for sub in ['fits','eval','directions','checkpoints']:(OUT/sub).mkdir(parents=True,exist_ok=True)

def log(**v):
 save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)

def get_train(e):
 dest=OUT/'directions/train.npz'
 if dest.exists():return dict(np.load(dest))
 source=Path(e.cfg['source_run']);idx=source/'index/train2017';pop=json.loads((ROOT/'runs/mapping-suite-2026-10-04/population.json').read_text());imids=np.array([v['image_id'] for v in json.loads((idx/'images.json').read_text())]);keep=np.isin(imids,pop['fit_image_ids']);rows=np.flatnonzero(keep);parents=np.load(idx/'parents.npy')
 testids=e.data['metadata']['half_a_image_ids']+e.data['metadata']['half_b_image_ids'];assert not set(imids[keep])&set(testids)
 concepts=json.loads((ROOT/'runs/feature-redundancy-cycle-2026-10-12/concepts.json').read_text());save(OUT/'concepts.json',concepts)
 ids=np.array(json.loads((idx/'concept_ids.json').read_text()));cols=np.array([int(np.flatnonzero(ids==c['id'])[0]) for c in concepts]);result={};stats={}
 np.savez_compressed(OUT/'train-membership.npz',image_ids=imids[keep],test_image_ids=testids)
 for side,rr,labelname in [('image',rows,'presence.npy'),('text',np.flatnonzero(keep[parents]),'mentions.npy')]:
  raw=sparse.load_npz(source/'activations/train2017'/f'{side}.npz').tocsr();labels=np.load(idx/labelname,mmap_mode='r');w=e.base[side].cpu().numpy()/e.data['scales'][side][:,None];shift=e.data['means'][side]@w;total=np.zeros(256);second=np.zeros((256,256));cross=np.zeros((256,50));count=np.zeros(50)
  for start in range(0,len(rr),8192):
   ix=rr[start:start+8192];z=np.asarray(raw[ix][:,e.data['feature_ids'][side]]@w)-shift;y=np.asarray(labels[ix][:,cols],float);total+=z.sum(0);second+=z.T@z;cross+=z.T@y;count+=y.sum(0)
  n=len(rr);assert np.all((count>0)&(count<n));mu=total/n;cov=second/n-np.outer(mu,mu);delta=cross/count[None,:]-(total[:,None]-cross)/(n-count)[None,:];freq=count/n;directions=[];regularizers=[];conditions=[]
  for c in range(50):
   within=cov-freq[c]*(1-freq[c])*np.outer(delta[:,c],delta[:,c]);within=(within+within.T)/2;alpha=1e-4*np.trace(within)/256;regularizers.append(alpha);ev=np.linalg.eigvalsh(within);conditions.append(dict(min_eigenvalue=float(ev.min()),max_eigenvalue=float(ev.max()),condition_after=float((ev.max()+alpha)/(ev.min()+alpha))))
   directions.append(linalg.solve(within+alpha*np.eye(256),delta[:,c],assume_a='pos'))
  for name,d in [('mean',delta),('lda',np.stack(directions,1))]:result[side+'_'+name]=d/np.linalg.norm(d,axis=0,keepdims=True);result[side+'_'+name+'_norm']=np.linalg.norm(d,axis=0)
  result[side+'_cov']=cov;stats[side]=dict(samples=n,positive=count.tolist(),regularizer=regularizers,covariance_condition=conditions)
  log(stage='training_directions',side=side,samples=n)
 np.savez_compressed(dest,**result);save(OUT/'training.json',dict(source=str(source),split='COCO train2017 original FIT subset only',no_test_overlap=True,image_count=len(rows),label_stats=stats,text_labels='own caption mentions',lda='pooled within-class covariance + 1e-4 trace(within)/256 I; no validation tuning',normalization='each category/modality direction unit L2; equal weights'))
 return result

def fit_supervised(di,dt,selected,r0):
 # Assignment is free over all256 axes. For fixed assignment polar(D) attains
 # the global trace bound; changing assignment merely permutes axis positions.
 f=(di[:,selected]+dt[:,selected])/2;m=len(selected);bound=np.linalg.svd(f,compute_uv=False).sum();u,s,vh=linalg.svd(f,full_matrices=False);q=u@vh;n=linalg.null_space(q.T);r=r0.copy();hist=[]
 for it in range(5):
  _,assign=linear_sum_assignment(-(f.T@r));free=np.array([j for j in range(256) if j not in assign]);uu,_,vv=linalg.svd(n.T@r[:,free]);rn=np.zeros((256,256));rn[:,assign]=q;rn[:,free]=n@(uu@vv);val=float(np.sum(f*rn[:,assign]));hist.append(dict(iteration=it,assignment=assign.tolist(),score=val,global_upper_score=float(bound),gap=float(bound-val)));r=rn
  _,nextassign=linear_sum_assignment(-(f.T@r))
  if np.array_equal(assign,nextassign):break
 assert abs(bound-val)<1e-8
 # Canonical output ordering: selected category order first, residual axes after.
 return r[:,np.r_[assign,free]],dict(history=hist,global_gap=float(bound-val),singular_values=s.tolist(),direction_objective=float(2*m-2*val),stop='global_direction_optimum')

def complement_rotate(e,r,startname):
 m=startname;d=256-m;n=r[:,m:];ls=[]
 for side in S:
  l=e.cov[side]@e.base[side];l=l/l.norm(dim=1,keepdim=True).clamp_min(1e-12);ls.append(l@n)
 h=torch.eye(d,device='cuda',dtype=torch.float64);hist=[];prev=None;stale=0;stop='iteration_limit';eye=torch.eye(d,device='cuda',dtype=torch.float64)
 def obj(q):return sum(((l@q).pow(4).mean(0)-(l@q).square().mean(0).square()).sum() for l in ls)/2
 for it in range(3000):
  z=[l@h for l in ls];val=obj(h);g=sum(2/l.shape[0]*l.T@(v.pow(3)-v*v.square().mean(0)) for l,v in zip(ls,z));om=(h.T@g-g.T@h)/2;gn=om.norm();rel=float(gn)/max(abs(float(val)),1e-12)
  if it%20==0:hist.append(dict(step=it,objective=float(val),relative_gradient=rel))
  if rel<1e-6:stop='gradient';break
  trial=a.polar(g);nv=obj(trial)
  if float(nv)<float(val)-1e-14 or float((trial-h).norm())<1e-12:
   direct=om/gn.clamp_min(1e-30);alpha=1.
   for attempt in range(30):
    trial=h@torch.linalg.solve(eye-.5*alpha*direct,eye+.5*alpha*direct);nv=obj(trial)
    if float(nv)>=float(val)+1e-4*alpha*float(gn):break
    alpha*=.5
   else:stop='line_search_stalled';break
  change=abs(float(nv-val))/max(abs(float(val)),1e-12);stale=stale+1 if change<1e-11 else 0;h=trial
  if stale>=50 and rel<1e-4:stop='stable_small_gradient';break
 r=r.clone();r[:,m:]=n@h
 # Only unassigned axes may flip sign. Assigned axes retain concept-positive sign.
 ws={s:e.base[s]@r for s in S};sg=positive_sign(ws);sg[:m]=1;r=r*sg
 return r,dict(steps=it,stop=stop,relative_gradient=rel,history=hist,objective=float(obj(h)),orthogonality=float((r.T@r-torch.eye(256,device='cuda')).abs().max()))

def energy_stats(d,r):
 z=(r.T@d).square();z=z/z.sum(0,keepdim=True);v=z.sort(0,descending=True).values
 return dict(top1=v[0].cpu().tolist(),n90=((v.cumsum(0)<.9).sum(0)+1).cpu().tolist(),n95=((v.cumsum(0)<.95).sum(0)+1).cpu().tolist(),participation=(1/z.square().sum(0)).cpu().tolist())

def measure(e,name,seed,r,train,selected,heldout,meta):
 dense={s:e.base[s]@r for s in S};energy={kind:{s:energy_stats(a.tensor(train[s+'_'+kind]),r) for s in S} for kind in ['mean','lda']}
 for cut in ['dense','column64','global64']:
  dest=OUT/'eval'/f'{name}-{cut}-s{seed}.json'
  if dest.exists():continue
  ws={}
  for s in S:
   if cut=='dense':ws[s]=dense[s]
   elif cut=='column64':ws[s]=hard(dense[s],64)
   else:
    w=dense[s];ix=w.abs().flatten().argsort(descending=True,stable=True)[:256*64];ws[s]=torch.zeros_like(w).flatten().scatter_(0,ix,w.flatten()[ix]).reshape_as(w)
  sc={s:e.x[s]@ws[s] for s in S};splits=[];perauc={s:[] for s in S};axis={s:[] for s in S}
  for pair in e.splits[:20]:
   au=[a.gpu_auc(sc[s][ix],e.labels[s][ix]) for s,ix in zip(S,pair)]
   for side,q in zip(S,au):
    perauc[side].append(q.max(0).values.cpu().tolist());axis[side].append((q>=.8).sum(0).cpu().tolist())
   valid=torch.isfinite(au[0])&torch.isfinite(au[1]);has=valid.any(0)
   for q in au:q[~valid]=-torch.inf
   inds=[q.argmax(0) for q in au];col=torch.arange(50,device='cuda');vs=[q[k,col] for q,k in zip(au,inds)];match=(inds[0]==inds[1])&has
   row=dict(per_category=match.int().cpu().tolist())
   for t in [.7,.8,.9]:row[str(t)]=(match&(vs[0]>=t)&(vs[1]>=t)).int().cpu().tolist()
   splits.append(row)
  counts={s:dict(mean=float((ws[s]!=0).sum(0).double().mean()),empty=int(((ws[s]!=0).sum(0)==0).sum())) for s in S}
  ret=retrieval(sc,e.parents)
  if cut=='dense':
   base=OUT/'dense-retrieval-reference.json'
   if not base.exists():save(base,retrieval({s:e.x[s]@e.base[s] for s in S},e.parents))
   rr=json.loads(base.read_text());assert max(abs(ret[d]['recall'][str(k)]-rr[d]['recall'][str(k)]) for d in ret for k in [1,5,10])<.0005
  save(dest,dict(name=name,seed=seed,cut=cut,selected_categories=selected,heldout_categories=heldout,retrieval=ret,matching=splits,best_axis_auc=perauc,responsive_axes=axis,direction_energy=energy,counts=counts,meta=meta));log(stage='evaluation',name=name,seed=seed,cut=cut,completed=len(list((OUT/'eval').glob('*.json'))))

def main():
 save(OUT/'protocol.json',dict(space='CC3M SAE / CC3M PLS256 fixed',rotation_labels='COCO train2017 original fit subset only, val excluded',conditions='mean difference and pooled-within covariance corrected linear discriminant; all50 and 5 random25/25 category splits',seeds=10,evaluation_splits=20,objective='min 0.5 sum_c ||R.T dI_c - e_pi(c)||^2 + ||R.T dT_c - e_pi(c)||^2 with normalized d and distinct assigned axes; global solution certified by nuclear norm of (DI+DT)/2',complement='original Kaiser row-normalized covariance Varimax restricted to unassigned orthogonal complement; no unobserved-category labels used',cuts=['dense','column64','global top16384 per modality'],direction_energy='unit training direction squared projection in full256 rotated basis; BEFORE coefficient pruning',caveat='global certificate only for surrogate directional fit, NOT semantic matching maximum or proof no successful rotation exists'))
 e=RC_Eval();train=get_train(e);groups=[dict(name='all50',selected=list(range(50)),heldout=[])]
 for split in range(5):
  perm=np.random.default_rng(1281200+split).permutation(50);groups.append(dict(name=f'half{split}',selected=sorted(perm[:25].tolist()),heldout=sorted(perm[25:].tolist())))
 save(OUT/'category-splits.json',groups)
 cos={kind:dict(cosine=np.sum(train['image_'+kind]*train['text_'+kind],0).tolist(),image_gram=(train['image_'+kind].T@train['image_'+kind]).tolist(),text_gram=(train['text_'+kind].T@train['text_'+kind]).tolist()) for kind in ['mean','lda']};save(OUT/'direction-geometry.json',cos)
 # Direction classifier AUROC measures whether the frozen space contains useful directions.
 for kind in ['mean','lda']:
  q={}
  for side in S:
   zz=e.x[side]@e.base[side]@a.tensor(train[side+'_'+kind]);q[side]=[]
   for pair in e.splits[:20]:
    ix=pair[S.index(side)];q[side].append(a.gpu_auc(zz[ix],e.labels[side][ix]).diag().cpu().tolist())
  save(OUT/f'free-directions-{kind}.json',q)
 for seed in range(10):
  for reference,path in [('covariance',SRC/f'rank256/covariance-svd-{seed}-oriented.npz'),('cycle-best',ROOT/'runs/feature-redundancy-cycle-2026-10-12/fits'/f'cycle-best-s{seed}.npz')]:
   r=a.tensor(np.load(path)['rotation']);ws={s:e.base[s]@r for s in S};r=r*positive_sign(ws);measure(e,reference,seed,r,train,[],list(range(50)),dict(source=str(path)))
  r0=np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation']
  for kind in ['mean','lda']:
   for group in groups:
    name=kind+'-'+group['name'];path=OUT/'fits'/f'{name}-s{seed}.npz';jp=path.with_suffix('.json')
    if path.exists():r=a.tensor(np.load(path)['rotation']);meta=json.loads(jp.read_text())
    else:
     start=time.time();rr,supervised=fit_supervised(train['image_'+kind],train['text_'+kind],group['selected'],r0);r,unsup=complement_rotate(e,a.tensor(rr),len(group['selected']));meta=dict(supervised=supervised,complement=unsup,seconds=time.time()-start,selected_categories=group['selected'],heldout_categories=group['heldout']);np.savez_compressed(path,rotation=r.cpu().numpy());save(jp,meta);log(stage='fit',name=name,seed=seed,seconds=time.time()-start,complement_stop=unsup['stop'])
    measure(e,name,seed,r,train,group['selected'],group['heldout'],meta)
 save(OUT/'complete.json',dict(fits=120,evaluations=420,time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
