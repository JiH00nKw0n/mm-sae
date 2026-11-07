"""Evaluate nine one-shot pruning rules on frozen C / regularized CCA covariance rotations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time,traceback,hashlib,shutil
import numpy as np
from scipy import stats
from experiments import conditioned_svd_axes_cca_20261012 as ev
ROOT=ev.ROOT;AX=ROOT/'runs/conditioned-svd-axes-2026-10-12';OC=ROOT/'runs/matching-axis-occupancy-2026-10-12';OUT=ROOT/'runs/cca1-pruning-2026-10-12';S=['image','text'];CUTS=['mass99','mass95','mass90','mass80','k16','k32','k64','k128','k256']
for d in ['eval','auroc','weights']:(OUT/d).mkdir(parents=True,exist_ok=True)
def save(p,x):ev.save(p,x)
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
def cut(w,rule):
 ix=np.argsort(-abs(w),axis=0,kind='stable');v=np.take_along_axis(w,ix,0)
 if rule.startswith('mass'):
  sq=v*v;n=(np.cumsum(sq,axis=0)<float(rule[4:])/100*sq.sum(0)).sum(0)+1
 else:n=np.full(w.shape[1],int(rule[1:]))
 v=v*(np.arange(len(w))[:,None]<n);out=np.zeros_like(w);np.put_along_axis(out,ix,v,0);return out

def reuse(space,rule,seed):
 if space=='CCA1':return AX/'eval'/f'CCA-1-pullback-column64-s{seed}.json' if rule=='k64' else None
 if rule=='mass80':return None
 return OC/'eval'/f'cov-{("m"+rule[4:] if rule.startswith("mass") else rule)}-s{seed}.json'

def normalize_rows(z):
 out=[]
 for row in z['rows'][:20]:
  i=np.array(row['image_axes']);t=np.array(row['text_axes']);di=np.bincount(i,minlength=256)[i]>1;dt=np.bincount(t,minlength=256)[t]>1;match=i==t
  assert match.sum()==row['match']
  r=dict(split=row['split'],match=int(match.sum()),solo=int((match&~di&~dt).sum()),overlap_image=int(di.sum()),overlap_text=int(dt.sum()),image_axes=i.tolist(),text_axes=t.tolist(),image_auc=row['image_auc'],text_auc=row['text_auc'])
  for s in S:r['responsive_'+s]=row['responsive_'+s] if 'responsive_'+s in row else row[s]['responsive']
  out.append(r)
 return out

def main():
 start=time.time();save(OUT/'protocol.json',dict(spaces=['C','CCA lambda1 pullback'],rank=256,seeds=list(range(5)),splits=20,cuts=CUTS,new_rotation=False,training=False,pruning='stable absolute coefficient sort; minimal prefix whose squared sum reaches requested fraction, or fixed count',sign='preserve saved covariance rotation signs',constant_axes='AUROC0.5 stored; if either modality is constant in a split exclude that axis from representative selection on both sides as historical GPU evaluator',solo='matched and unique representative among all50 categories on EACH modality',F1=False))
 cfg=ev.p.old.load_config(ROOT/ev.p.CONFIG['cc3m-cc3m']);data,parents=ev.p.eval_population(cfg,'test',OUT);eligible=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert eligible.sum()==50;labels=[data['labels'][s][:,eligible].astype(float) for s in S];cats=[c for c,k in zip(data['concepts'],eligible) if k];save(OUT/'categories.json',cats);ss=np.load(ev.REF/'evaluation-splits.npz');splits=[(ss[f'image_{j}'],ss[f'text_{j}']) for j in range(20)];shutil.copy2(ev.REF/'evaluation-splits.npz',OUT/'evaluation-splits.npz')
 count=0
 for rule in CUTS:
  for space,stat in [('C','C'),('CCA1','CCA-1')]:
   for seed in range(5):
    name=f'{space}-{rule}-s{seed}';dest=OUT/'eval'/f'{name}.json'
    if dest.exists():count+=1;continue
    t0=time.time();fp=AX/'fits'/f'{stat}-pullback-s{seed}.npz';z=np.load(fp);ws=[cut(z[s],rule) for s in S];cs={s:ev.coeffstats(w) for s,w in zip(S,ws)};meta=dict(rotation_source=str(fp),rotation_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),new_rotation=False);old=reuse(space,rule,seed)
    if old is not None:
     assert old.exists(),old;q=json.loads(old.read_text());rows=normalize_rows(q);ret=q['retrieval'];meta.update(reused=True,evaluation_source=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest())
     for s in S:
      n=q['features'][s] if 'features' in q else q['coefficients'][s]['mean'];assert abs(n-cs[s]['mean'])<1e-9,(name,s,n,cs[s]['mean'])
    else:
     meta['reused']=False;sc=[np.asarray(data['raw'][s]@(w/data['scales'][s][:,None]))-data['means'][s]@(w/data['scales'][s][:,None]) for s,w in zip(S,ws)];rows=[];full=[[],[]]
     for split,pair in enumerate(splits):
      aa=[];variable=[]
      for j,ix in enumerate(pair):
       x=sc[j][ix];lab=labels[j][ix];pos=lab.sum(0);n=len(x);rank=stats.rankdata(x,axis=0,method='average');au=(rank.T@lab-pos*(pos+1)/2)/(pos*(n-pos));assert np.isfinite(au).all();variable.append(np.ptp(x,axis=0)>0);aa.append(au);full[j].append(au.T.copy())
      valid=variable[0]&variable[1];assert valid.any();meta.setdefault('constant_axes_per_split',[]).append(dict(image=int((~variable[0]).sum()),text=int((~variable[1]).sum())))
      responsive=[float((q>=.8).sum(0).mean()) for q in aa]
      for q in aa:q[~valid]=-np.inf
      ii=[q.argmax(0) for q in aa];best=[q[i,np.arange(50)] for q,i in zip(aa,ii)];dup=[np.bincount(i,minlength=256)[i]>1 for i in ii];match=ii[0]==ii[1]
      rows.append(dict(split=split,match=int(match.sum()),solo=int((match&~dup[0]&~dup[1]).sum()),overlap_image=int(dup[0].sum()),overlap_text=int(dup[1].sum()),responsive_image=responsive[0],responsive_text=responsive[1],image_axes=ii[0].tolist(),text_axes=ii[1].tolist(),image_auc=best[0].tolist(),text_auc=best[1].tolist()))
     ret=ev.retrieval(sc,parents);np.savez_compressed(OUT/'auroc'/f'{name}.npz',image=np.array(full[0]),text=np.array(full[1]),category_ids=[c['id'] for c in cats],axis_ids=np.arange(256),split_ids=np.arange(20))
    save(dest,dict(space=space,rule=rule,seed=seed,rows=rows,retrieval=ret,coefficients=cs,meta=meta,seconds=time.time()-t0));count+=1;log(stage='evaluated',completed=count,total=90,name=name,reused=meta['reused'],seconds=time.time()-t0)
 save(OUT/'complete.json',dict(time=time.time(),seconds=time.time()-start,evaluations=90,new_evaluations=45,reused=45))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
