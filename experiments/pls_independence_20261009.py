"""Sparse PLS warm-start pilot with paired-group approximate HSIC.
No claim of exact TC, likelihood-based IVA, or global optimality.
"""
import os
os.environ['OMP_NUM_THREADS']='4'; os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import json,time,traceback,sys
import numpy as np
from scipy import sparse
import torch
from mm_sae.metrics.regression import Moments
from mm_sae.analysis.mapping_evaluation import paired_retrieval
import experiments.caption_matching_20261009 as cap
old=cap.old
ROOT=Path('/mnt/working/mm-sae'); OUT=ROOT/'runs/pls-independence-2026-10-09'
torch.set_num_threads(4)
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()));q.replace(p)
@torch.no_grad()
def project(w):
 ix=w.abs().topk(16,dim=0).indices;w.mul_(torch.zeros_like(w).scatter_(0,ix,1));w.div_(w.norm(dim=0,keepdim=True).clamp_min(1e-9))
def terms(x,y,a,b,omega,pairs):
 u=x@a;v=y@b;u=u-u.mean(0);v=v-v.mean(0)
 cov=(u*v).mean(0).mean()
 # Gaussian-kernel random Fourier features on each 2D paired group.
 s=torch.stack((u/(u.square().mean(0)+1e-5).sqrt(),v/(v.square().mean(0)+1e-5).sqrt()),-1)
 phase=s@omega
 f=torch.cat((phase.cos(),phase.sin()),-1)/(omega.shape[1]**.5)
 f=f-f.mean(0,keepdim=True)
 left=f[:,pairs[:,0]];right=f[:,pairs[:,1]]
 cross=torch.einsum('bpf,bpg->pfg',left,right)/(len(x)-1)
 dep=cross.square().sum((1,2)).mean()
 eye=torch.eye(a.shape[1],device=a.device)
 redundancy=((a.T@a-eye).square().sum()+(b.T@b-eye).square().sum())/(2*a.shape[1])
 return cov,dep,redundancy
CONFIGS=[('coco-coco','configs/representative-agreement-local.yaml'),('cc3m-coco','runs/cc3m-followup-2026-10-05/configs/coco-fit-agreement.yaml'),('cc3m-cc3m','runs/cc3m-followup-2026-10-05/configs/cc3m-fit-agreement.yaml')]
def condition(cond,cf):
 cfg=old.load_config(ROOT/cf);parent=Path(cfg['parent_run']);out=OUT/cond;out.mkdir(parents=True,exist_ok=True)
 source=ROOT/'runs/cc3m-sae-2026-10-04' if cond=='cc3m-cc3m' else Path(cfg['source_run'])
 pop=json.loads((parent/'population.json').read_text())
 with np.load(parent/'moments.npz') as z:
  ids={s:z[s+'_ids'].copy() for s in ['image','text']};mom=Moments(int(z['fit_n']),z['fit_mean'],z['fit_second'])
 ni=len(ids['image']);means={'image':mom.mean[:ni],'text':mom.mean[ni:]};scales={'image':mom.scale[:ni],'text':mom.scale[ni:]}
 rows=json.loads((source/'index/train2017/images.json').read_text());names=np.array([r['image_id'] for r in rows]);del rows
 rng=np.random.default_rng(911)
 fit=np.flatnonzero(np.isin(names,pop['fit_image_ids']));tune=np.flatnonzero(np.isin(names,pop['tune_image_ids']))
 fit=rng.choice(fit,min(32768,len(fit)),replace=False);tune=rng.choice(tune,min(2048,len(tune)),replace=False)
 assert not np.intersect1d(fit,tune).size
 par=np.load(source/'index/train2017/parents.npy');order=np.argsort(par,kind='stable');counts=np.bincount(par,minlength=len(names));starts=np.r_[0,counts.cumsum()[:-1]]
 ti={}
 for key,ir in [('fit',fit),('tune',tune)]:ti[key]=order[starts[ir]+(rng.random(len(ir))*counts[ir]).astype(int)]
 arrays={}
 for s in ['image','text']:
  raw=sparse.load_npz(source/'activations/train2017'/f'{s}.npz').tocsr()
  for key,ir in [('fit',fit),('tune',tune)]:
   rr=ir if s=='image' else ti[key]
   arrays[key,s]=torch.tensor(((raw[rr][:,ids[s]].toarray()-means[s])/scales[s]).astype('float32'),device='cuda')
  del raw
 np.savez_compressed(out/'split.npz',fit_rows=fit,tune_rows=tune,fit_text_rows=ti['fit'],tune_text_rows=ti['tune'])
 initial=ROOT/'runs/sparse-pls-2026-10-06'/cond/'transform.npz'
 with np.load(initial) as z:weights=[z[s].astype('float32') for s in ['image','text']]
 torch.manual_seed(0);omega=torch.randn(2,8,device='cuda')
 r=weights[0].shape[1];allpairs=torch.combinations(torch.arange(r,device='cuda'),r=2)
 fixed=allpairs[torch.randperm(len(allpairs),device='cuda')[:1024]]
 def valid(a,b):
  with torch.no_grad():return np.mean([[float(t) for t in terms(arrays['tune','image'][i:i+512],arrays['tune','text'][i:i+512],a,b,omega,fixed)] for i in range(0,len(tune),512)],axis=0)
 aa,bb=[torch.tensor(w,device='cuda') for w in weights]
 project(aa);project(bb);normal=np.maximum(np.abs(valid(aa,bb)),1e-5)
 save(out/'protocol.json',dict(condition=cond,config=cfg,fit_pairs=len(fit),tune_pairs=len(tune),epochs=12,batch=512,lr=.001,seed=0,support=16,dimensions=r,lambda_values=[0,.1,1],objective='-mean paired covariance / initial tune covariance + lambda * paired-group RFF-HSIC / initial tune HSIC + 0.1 * off-diagonal weight Gram energy',independence='Pairwise HSIC approximation, Gaussian RFF with 8 frequencies, shared fixed frequencies, 1024 randomly sampled distinct component pairs per update. Not total correlation or exact IVA.',constraints='16 largest-magnitude weights and unit L2 norm per column; signed coefficients; P=I; soft redundancy penalty shared across arms',normalizers=normal.tolist(),checkpoint='Lowest own tune objective; patience 3, max12epochs',scope='One-seed warm-start pilot on fixed 32768 paired training subset; original PLS was fitted on full training data. Annotations only in final evaluation.'))
 a=b=opt=None
 jobs=[('original_sparse_pls',initial)]
 for lam in [0,.1,1]:
  name=f'pls_hsic_{lam:g}';path=out/(name+'.npz');jobs.append((name,path))
  if path.exists():continue
  torch.manual_seed(0);rng=np.random.default_rng(0)
  a,b=[torch.nn.Parameter(torch.tensor(w,device='cuda')) for w in weights]
  project(a);project(b);opt=torch.optim.Adam([a,b],lr=.001);best=float('inf');stale=0;hist=[];beg=time.monotonic()
  for epoch in range(13):
   start=time.monotonic();train=[]
   if epoch:
    order2=rng.permutation(len(fit))
    for i in range(0,len(fit),512):
     ix=torch.tensor(order2[i:i+512],device='cuda');pairs=allpairs[torch.randperm(len(allpairs),device='cuda')[:1024]]
     c,d,g=terms(arrays['fit','image'][ix],arrays['fit','text'][ix],a,b,omega,pairs)
     loss=-c/normal[0]+lam*d/normal[1]+.1*g
     if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
     opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([a,b],5);opt.step();project(a);project(b);train.append(float(loss.detach()))
   values=valid(a,b);vl=-values[0]/normal[0]+lam*values[1]/normal[1]+.1*values[2]
   if vl<best-1e-5:
    best=float(vl);bestepoch=epoch;stale=0;ww=[w.detach().cpu().numpy().copy() for w in [a,b]]
   else:stale+=1
   row=dict(condition=cond,method=name,epoch=epoch,max_epochs=12,tune_loss=float(vl),tune_cov=float(values[0]),tune_dependence=float(values[1]),tune_redundancy=float(values[2]),train_loss=float(np.mean(train)) if train else None,epoch_seconds=time.monotonic()-start,elapsed_seconds=time.monotonic()-beg)
   hist.append(row);save(out/(name+'-history.json'),hist);save(OUT/'progress.json',dict(state='training',**row));print(json.dumps(row),flush=True)
   if epoch and stale>=3:break
  assert all(np.count_nonzero(w,axis=0).max()<=16 for w in ww)
  np.savez_compressed(path,image=ww[0],text=ww[1]);save(out/(name+'-fit.json'),dict(best_epoch=bestepoch,completed_epoch=epoch,best_tune_objective=best))
 del arrays,aa,bb,a,b,opt;torch.cuda.empty_cache()
 data=cap.load_population(cfg,'test');par=np.load(Path(cfg['source_run'])/'index/val2017/parents.npy')
 cfg['n_null']=0
 for name,path in jobs:
  resultpath=out/(name+'-evaluation.json')
  if resultpath.exists():continue
  save(OUT/'progress.json',dict(state='evaluating',condition=cond,method=name))
  with np.load(path) as z:ww={s:z[s] for s in ['image','text']}
  scores={s:np.asarray(data['raw'][s]@(ww[s]/data['scales'][s][:,None]))-data['means'][s]@(ww[s]/data['scales'][s][:,None]) for s in ww}
  ret=paired_retrieval(scores['image'],scores['text'],par,device='cuda',chunk_size=128)
  arr=old.calculate_aucs(cfg,data,ww,lambda k:None)
  res=dict(condition=cond,method=name,retrieval=ret,positive_agreement=old.evaluation(arr,cfg,signed=False),signed_agreement=old.evaluation(arr,cfg,signed=True),denominator=int(data['eligible'].sum()),eligible_category_ids=data['metadata']['eligible_category_ids'],text_labels='Own-caption dictionary mentions; image uses COCO object annotations; disjoint image halves; no labels in training')
  save(resultpath,res);np.savez_compressed(out/(name+'-auc.npz'),**arr);print(json.dumps(dict(evaluated=name,condition=cond)),flush=True)
def main():
 for cond,cf in CONFIGS:condition(cond,cf)
 save(OUT/'progress.json',dict(state='completed'))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(OUT/'progress.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc()));raise
