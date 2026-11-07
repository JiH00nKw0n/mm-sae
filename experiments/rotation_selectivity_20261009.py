"""Frozen CC3M/CC3M rotations: held-out selectivity and training trajectories."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
import torch
from scipy.stats import rankdata,spearmanr
import experiments.rotation_convergence_20261009 as conv
import experiments.multiview_likelihood_20261009 as multi
p=conv.p;ROOT=p.ROOT;SRC=ROOT/'runs/rotation-convergence-2026-10-09';OUT=ROOT/'runs/rotation-selectivity-2026-10-09';OUT.mkdir(exist_ok=True)
def save(f,x):p.save(f,x)
def state(**x):save(OUT/'progress.json',x);print(json.dumps(x),flush=True)
def auc(x,y):
 n=int(y.sum());m=len(y)-n
 if min(n,m)==0:return None
 return float((rankdata(x)[y].sum()-n*(n+1)/2)/(n*m))
def trajectory(seed,ts):
 dest=OUT/f'trajectory-{seed}';dest.mkdir(exist_ok=True)
 if (dest/'complete.json').exists():return
 # Reuse exactly the published optimizer; inject checkpoint saving at each 250 steps.
 import inspect
 code=inspect.getsource(conv.fit).replace('def fit(','def trajectory_fit(',1)
 code=code.replace("if step%25==0 or step==args.max_steps:","if step%250==0 or step==args.max_steps:\n   np.savez_compressed(OUT/f'checkpoint-{seed}-{step}.npz',rotation=r.detach().cpu().numpy())\n  if step%25==0 or step==args.max_steps:")
 ns=dict(conv.__dict__);ns['OUT']=dest
 def progress(**kw):state(**{**kw,'stage':'trajectory_'+kw.get('stage','fit')})
 ns['status']=progress
 exec(code,ns);ns['trajectory_fit']('output_l1',seed,ts,type('Args',(),{'max_steps':6000})())
 save(dest/'complete.json',{'complete':True})
def run():
 state(stage='prepare');cfg,w,ts,orientation=conv.prepare()
 for seed in range(3):trajectory(seed,ts)
 del ts;torch.cuda.empty_cache()
 data,parents=p.eval_population(cfg,'test',OUT)
 base={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
 ids=np.array([c['id'] for c in data['concepts']]);eligible=np.flatnonzero(np.isin(ids,data['metadata']['eligible_category_ids']));assert len(eligible)==50
 names=[c.get('name',str(c['id'])) for c in data['concepts']]
 # Co-occurring category pairs chosen exclusively on calibration halves, before inspecting any model.
 pairs=[]
 for j,c in enumerate(eligible):
  for d in eligible[j+1:]:
   phis=[];ok=True
   for s in w:
    y=data['labels'][s][data['masks'][s+'_a']].astype(bool);a,b=y[:,c],y[:,d]
    if min(np.sum(a&~b),np.sum(b&~a),np.sum(~a&~b),np.sum(a&b))<20:ok=False;break
    phis.append(float(np.corrcoef(a,b)[0,1]))
   if ok and min(phis)>0:pairs.append((float(np.mean(phis)),int(c),int(d)))
 pairs=sorted(pairs,reverse=True)[:30];assert len(pairs)>0
 save(OUT/'pairs.json',[dict(first=names[c],second=names[d],first_index=c,second_index=d,calibration_mean_phi=phi) for phi,c,d in pairs])
 jobs=[('cca_original',np.eye(256),False),('cca_oriented',np.eye(256),True)]
 for seed in range(3):
  torch.manual_seed(1200+seed);rr=conv.old.polar(torch.randn(256,256,device='cuda')).cpu().numpy().astype(float);a,_,b=np.linalg.svd(rr);jobs.append((f'random-{seed}',a@b,True))
 for m in conv.old.METHODS:
  for seed in range(3):jobs.append((f'{m}-{seed}',np.load(SRC/f'{m}-{seed}-final-evaluated-rotation.npz')['rotation'],False))
 # Evaluate a prespecified subset of actual training snapshots, not selected by matching.
 for seed in range(3):
  dr=OUT/f'trajectory-{seed}'
  for step in [0,250,500,1000,2000,3000,4000,6000]:
   f=dr/f'checkpoint-{seed}-{step}.npz'
   if f.exists():jobs.append((f'trajectory-{seed}-{step}',np.load(f)['rotation'],True))
  jobs.append((f'trajectory-{seed}-final',np.load(dr/f'output_l1-{seed}-final.npz')['rotation'],True))
 for name,r,orient in jobs:
  target=OUT/(name+'.json')
  if target.exists():continue
  if orient:
   skew=sum(np.mean((a@r)**3,0) for a in orientation);r=r*np.where(skew<0,-1.,1.)
  scores={s:a@r for s,a in base.items()};ma,ms=multi.matching(scores,data,cfg);result=dict(name=name,matching50=ma['summary']['agree_at1_count'],signed_matching50=ms['summary']['agree_at1_count'],sides={})
  for s,z in scores.items():
   cal=data['masks'][s+'_a'];test=data['masks'][s+'_b'];y=data['labels'][s].astype(bool)
   aa=multi.auc_matrix(z[cal],y[cal],chunk_size=32);coord=aa.argmax(0)
   q=np.square(z[test]);total=q.sum(1);eff=np.divide(np.abs(z[test]).sum(1)**2,total,out=np.zeros_like(total),where=total>0)
   n95=(np.cumsum(np.sort(q,axis=1)[:,::-1],1)<.95*total[:,None]).sum(1)+1
   held=[auc(z[test,coord[c]],y[test,c]) for c in eligible]
   pp=[]
   for phi,c,d in pairs:
    for target_c,other_c in [(c,d),(d,c)]:
     k=int(coord[target_c]);yc,yo=y[test,target_c],y[test,other_c];zz=z[test,k]
     only_target=yc&~yo;only_other=yo&~yc;neither=~yc&~yo;both=yc&yo;mask=only_target|only_other
     if min(only_target.sum(),only_other.sum(),neither.sum())<10:continue
     threshold=float(np.quantile(z[cal&y[:,target_c],k],.2))
     fp_other=float(np.mean(zz[only_other]>=threshold));fp_neither=float(np.mean(zz[neither]>=threshold))
     pp.append(dict(target=names[target_c],other=names[other_c],coordinate=k,exclusive_auc=auc(zz[mask],only_target[mask]),fpr_other_only=fp_other,fpr_neither=fp_neither,excess_fpr=fp_other-fp_neither,tpr_target_only=float(np.mean(zz[only_target]>=threshold)),tpr_both=float(np.mean(zz[both]>=threshold)) if both.sum() else None,counts=[int(v.sum()) for v in [only_target,only_other,neither,both]]))
   result['sides'][s]=dict(output_effective_count_median=float(np.median(eff)),output_95_count_median=float(np.median(n95)),heldout_concept_auc_mean=float(np.mean(held)),heldout_concept_auc=held,representative_coordinates=[int(coord[c]) for c in eligible],distinct_representatives=int(len(set(coord[eligible]))),pair_count=len(pp),exclusive_auc_mean=float(np.mean([v['exclusive_auc'] for v in pp])),excess_fpr_mean=float(np.mean([v['excess_fpr'] for v in pp])),pair_details=pp)
  # Pairwise cosine invariance, including means and train-only sign convention.
  norm=lambda z:z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-15)
  err=float(np.max(abs(norm(base['image'][:128])@norm(base['text'][:256]).T-norm(scores['image'][:128])@norm(scores['text'][:256]).T)))
  assert err<1e-7
  result['sample_cosine_error']=err;result['orthogonality_error']=float(np.max(abs(r.T@r-np.eye(256))));save(target,result);state(stage='evaluated',name=name,matched=result['matching50'])
 save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',fixed50_ids=ids[eligible].tolist(),methods=conv.old.METHODS,initializations=3,labels_used_for_rotation=False,selection='Per modality representatives selected on val half_a by positive AUROC, diagnostics on disjoint half_b. Existing matching50 separately retains original image_a/text_b convention.',pairs='Up to30 highest positive mean phi pairs, require >=20 samples in each of four presence cells in both modality calibration half_a. Diagnostic test cells require >=10 exclusive positives/exclusive negatives/neither. No model-dependent selection.',fpr='Threshold calibrated for80percent sensitivity on target-positive half_a. Excess false positive = FPR on other-only minus FPR on neither, evaluated half_b. Also report sensitivity to detect threshold or discrimination confounds.',limitations='Observational mechanism audit, no isolated causal effect of sparsity. COCO validation previously explored. Three seeds are repeats, not independent data replications. No annotation-based checkpoint selection. All common rotations preserve full cosine retrieval mathematically; sample numerical invariance checked.'))
 save(OUT/'complete.json',{'complete':True,'models':len(jobs)})
if __name__=='__main__':run()
