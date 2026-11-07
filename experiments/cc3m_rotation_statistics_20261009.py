"""Exact moment-based diagnostics for previously fitted common rotations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time,traceback
import numpy as np
import torch
ROOT=Path('/mnt/working/mm-sae');SRC=ROOT/'runs/cc3m-caption-rank-sweep-2026-10-09';OUT=SRC/'statistics';OUT.mkdir(exist_ok=True)
def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def tensor(v):return torch.as_tensor(v,dtype=torch.float64,device='cuda')
def off_rms(m):return float(torch.sqrt(torch.clamp((m.square().sum()-m.diagonal().square().sum())/(m.numel()-len(m)),min=0)))
def stats(a,b,c):
 sa=a.diagonal().clamp_min(1e-20).sqrt();sb=b.diagonal().clamp_min(1e-20).sqrt();corr=c/sa[:,None]/sb[None,:]
 return dict(mean_paired_covariance=float(c.diagonal().mean()),mean_paired_correlation=float(corr.diagonal().mean()),min_paired_correlation=float(corr.diagonal().min()),max_paired_correlation=float(corr.diagonal().max()),offdiagonal_correlation_rms=off_rms(corr),offdiagonal_covariance_rms=off_rms(c),within_image_correlation_rms=off_rms(a/sa[:,None]/sa[None,:]),within_text_correlation_rms=off_rms(b/sb[:,None]/sb[None,:]),image_mean_variance=float(a.diagonal().mean()),text_mean_variance=float(b.diagonal().mean()),cross_covariance_frobenius=float(torch.linalg.norm(c)))
def main():
 start=time.time();torch.set_num_threads(2)
 parent=ROOT/'runs/mapping-suite-cc3m-2026-10-04'
 with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cca_3221.npz') as z:wi=tensor(z['image']);wt=tensor(z['text'])
 with np.load(parent/'moments.npz') as z:
  ni=len(z['image_ids']);mu=z['fit_mean'];sd=np.sqrt(np.maximum(z['fit_second'].diagonal()-mu**2,0));sd[sd==0]=1
  sigmas={}
  for split in ['fit','test']:
   mean=z[split+'_mean'];cov=z[split+'_second']-np.outer(mean,mean);cov/=sd[:,None]*sd[None,:];s=tensor(cov);del cov
   sigmas[split]=(wi.T@s[:ni,:ni]@wi,wt.T@s[ni:,ni:]@wt,wi.T@s[:ni,ni:]@wt);del s
 with np.load(SRC/'cca-label-cross.npz') as z:ct={s:tensor(z[s]) for s in ['image','text']}
 results=[]
 for r in [256,512,1024,2048,3221]:
  save('progress.json',dict(rank=r,elapsed=time.time()-start))
  with np.load(SRC/f'model-{r}.npz') as z:rot=tensor(z['rotation'])
  row=dict(rank=r,populations={})
  for pop,(af,bf,cf) in sigmas.items():
   a=af[:r,:r];b=bf[:r,:r];c=cf[:r,:r];ar=rot.T@a@rot;br=rot.T@b@rot;cr=rot.T@c@rot
   row['populations'][pop]={'before':stats(a,b,c),'after':stats(ar,br,cr)}
   # Rotation preserves trace/Frobenius, not individual Pearson correlations.
   assert torch.allclose(c.trace(),cr.trace(),atol=1e-8,rtol=1e-10)
   assert torch.allclose(torch.linalg.norm(c),torch.linalg.norm(cr),atol=1e-8,rtol=1e-10)
   if pop=='fit':
    for name,xx,yy,rr in [('before',a,b,torch.eye(r,device='cuda',dtype=torch.float64)),('after',ar,br,rot)]:
     q=row['populations'][pop][name];label={}
     for side,var in [('image',xx.diagonal()),('text',yy.diagonal())]:
      lc=rr.T@ct[side][:r,:r];diag=lc.diagonal();pc=diag/var.clamp_min(1e-20).sqrt()
      label[side]=dict(mean_assigned_label_covariance=float(diag.mean()),mean_assigned_label_correlation=float(pc.mean()),mean_standardized_label_mse=float((var+1-2*diag).mean()))
     q['labels']=label;q['rotation_objective_per_coordinate']=sum(v['mean_standardized_label_mse'] for v in label.values())/2
  assert row['populations']['fit']['after']['rotation_objective_per_coordinate']<=row['populations']['fit']['before']['rotation_objective_per_coordinate']+1e-9
  results.append(row);save('results.json',results);print(json.dumps(row),flush=True)
 save('protocol.json',dict(cca='CC3M SAE and CC3M mapping; ridge .01',fit='2324763 CC3M FIT caption-image pairs',test='COCO val2017 paired moments, image weighted by caption frequency as in existing moments.',labels='FIT-only standardized own-caption noun/noun-phrase mention matrix, identical target for both modalities. No held-out label MSE computed.',before='CCA coordinate j compared to frequency-rank concept j. CCA was not trained to follow that assignment.',mse='Per coordinate and per example average. Objective in question divided by r.',offdiagonal='Root mean square Pearson correlation over all ordered unequal coordinate pairs; signed correlations squared, not absolute mean.',invariants_verified='Crosscovariance trace and Frobenius norm, for FIT and TEST at every rank.'))
 save('complete.json',dict(complete=True,elapsed=time.time()-start))
if __name__=='__main__':
 try:main()
 except Exception as e:save('error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
