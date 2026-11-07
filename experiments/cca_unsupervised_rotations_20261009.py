"""Label-free shared orthogonal rotations. Fixed256 CCA, three dataset conditions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
import json,time,traceback
import numpy as np
from scipy import linalg
import torch
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.models.encoder import CLIPEncoder
from mm_sae.models.sae import TopKSAE
ROOT=prior.ROOT;OUT=ROOT/'runs/cca-unsupervised-rotations-2026-10-09';save=prior.save
METHODS=['coefficient_l1','coefficient_varimax','loading_varimax','output_l1','output_kurtosis','output_logcosh','shared_cokurtosis']
START=time.time()
def status(**kw):save(OUT/'progress.json',dict(elapsed=time.time()-START,**kw));print(json.dumps(kw),flush=True)
def polar(r):
 u,_,v=torch.linalg.svd(r,full_matrices=False);return u@v

def fit(method,seed,inputs,out):
 file=out/f'{method}-{seed}.npz'
 if file.exists():return np.load(file)['rotation']
 torch.manual_seed(1200+seed);r=torch.eye(256,device='cuda') if seed==0 else polar(torch.randn(256,256,device='cuda'))
 # All output objectives use the same fixed8192 FIT pairs; tuning uses2048 disjoint TUNE pairs.
 u,v=inputs['fit'];tu,tv=inputs['tune']
 coef=inputs['coeff'];load=inputs['load']
 def loss(rr,tune=False):
  if method.startswith('coefficient') or method=='loading_varimax':
   mats=load if method=='loading_varimax' else coef
   if method=='coefficient_l1':return sum((m@rr).abs().mean() for m in mats)/2
   return -sum(((m@rr).pow(4).mean(0)-(m@rr).square().mean(0).square()).sum() for m in mats)/2
  x,y=(tu,tv) if tune else (u,v);a=x@rr;b=y@rr
  if method=='output_l1':return (a.abs().sum(1)/a.norm(dim=1).clamp_min(1e-8)+b.abs().sum(1)/b.norm(dim=1).clamp_min(1e-8)).mean()/32
  a=a-a.mean(0);b=b-b.mean(0);a=a/a.square().mean(0).clamp_min(1e-8).sqrt();b=b/b.square().mean(0).clamp_min(1e-8).sqrt()
  if method=='output_kurtosis':return -(a.pow(4).mean()+b.pow(4).mean()-6)/2
  if method=='output_logcosh':
   logcosh=lambda z:z.abs()+torch.nn.functional.softplus(-2*z.abs())-np.log(2)
   return -((logcosh(a).mean(0)-.37456720749).square().mean()+(logcosh(b).mean(0)-.37456720749).square().mean())/2
  if method=='shared_cokurtosis':return -(a.square()*b.square()).mean() + 1 + 2*(a*b).mean(0).square().mean()
  raise ValueError(method)
 # Riemannian gradient with polar retraction and Armijo backtracking; no learning-rate grid.
 best=float('inf');best_r=r.clone();stale=0;hist=[];step_size=1.;converged=False
 for step in range(501):
  r=r.detach().requires_grad_(True);f=loss(r);g=torch.autograd.grad(f,r)[0];g=g-r@((r.T@g+g.T@r)/2);gn=float(g.norm());train=float(f)
  if step%10==0:
   with torch.no_grad():vl=float(loss(r,True))
   hist.append(dict(step=step,fit_loss=train,tune_loss=vl,gradient_norm=gn))
   if vl<best-1e-8:best=vl;best_r=r.detach().clone();best_step=step;stale=0
   else:stale+=1
   save(out/f'{method}-{seed}-history.json',hist)
   if step>=100 and stale>=8:break
  if gn<1e-7:converged=True;break
  eta=min(step_size*1.5,10.)/max(1.,gn)
  accepted=False
  with torch.no_grad():
   for _ in range(16):
    trial=polar(r-eta*g);nval=float(loss(trial))
    if nval<=train-1e-4*eta*gn*gn:accepted=True;break
    eta*=.5
   if not accepted:break
   r=trial;step_size=min(eta*max(1.,gn),10.)
 # Float64 polar correction ensures exact cosine invariance to numerical precision.
 rr=best_r.cpu().numpy().astype(float);left,_,right=linalg.svd(rr);rr=left@right
 np.savez_compressed(file,rotation=rr)
 save(out/f'{method}-{seed}-fit.json',dict(method=method,seed=seed,steps=step,best_step=best_step,best_tune_objective=best,gradient_converged=converged,global_optimum_claimed=False,selection='Own unsupervised TUNE objective only; coefficient objectives deterministic on frozen weights.',initialization='CCA identity' if seed==0 else 'random orthogonal',last_gradient_norm=gn))
 return rr

def evaluate(cfg,data,parents,test,proto,presence,names,rr,name,out,orientation):
 f=out/f'{name}-result.json'
 if f.exists():return
 if orientation:
  skew=sum(np.mean((s@rr).astype(np.float64)**3,axis=0) for s in orientation);rr=rr*np.where(skew<0,-1.,1.)[None,:]
 scores={s:test[s]@rr for s in test};pt=proto@rr;np.savez_compressed(out/f'{name}-evaluated-rotation.npz',rotation=rr)
 ret=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128)
 arr={}
 for s in scores:
  for half in ['a','b']:
   key=s+'_'+half;mask=data['masks'][key];arr[key]=auc_matrix(scores[s][mask],data['labels'][s][mask],chunk_size=32);arr[key+'_variable']=variable_coordinates(scores[s][mask])
 arr['eligible']=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert arr['eligible'].sum()==50
 arr['category_ids']=np.array([c['id'] for c in data['concepts']]);m50=prior.old.evaluation(arr,{**cfg,'n_null':0},signed=False);ms50=prior.old.evaluation(arr,{**cfg,'n_null':0},signed=True)
 auc=auc_matrix(scores['image'],presence,chunk_size=32);ki=auc.argmax(0);kt=pt.argmax(1);ok=ki==kt
 signed_i=np.abs(auc-.5).argmax(0);signed_t=np.abs(pt).argmax(1);signed_ok=(signed_i==signed_t)&(np.sign(auc[signed_i,np.arange(171)]-.5)==np.sign(pt[np.arange(171),signed_t]))
 norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-15)
 difference=float(np.max(abs(norm(test['image'][:128])@norm(test['text'][:256]).T-norm(scores['image'][:128])@norm(scores['text'][:256]).T)))
 assert difference<1e-10
 result=dict(name=name,orthogonality_error=float(np.max(abs(rr.T@rr-np.eye(256)))),sample_cosine_error=difference,retrieval={d:{k:v for k,v in val.items() if k not in ['hits','ranks']} for d,val in ret.items()},matching50=m50,signed_matching50=ms50,matching171=dict(matched=int(ok.sum()),distinct_matched=len(set(ki[ok].tolist())),distinct_image=len(set(ki.tolist())),distinct_text=len(set(kt.tolist())),signed_same_direction_count=int(signed_ok.sum()),mean_auc_at_text_coordinate=float(auc[kt,np.arange(171)].mean()),categories=[dict(name=n,image_coordinate=int(i),text_coordinate=int(t),matched=bool(i==t),image_auc=float(auc[i,j]),image_auc_at_text=float(auc[t,j])) for j,(n,i,t) in enumerate(zip(names,ki,kt))]))
 save(f,result)

 def_unused=None

def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
 save(OUT/'protocol.json',dict(rank=256,conditions=list(prior.CONFIG),methods=METHODS,seeds=[0,1,2],fit_sample=32768,objective_fit_pairs=8192,tune_pairs=2048,labels_in_training=False,rotation='Shared256x256 orthogonal matrix; no post-rotation normalization or thresholding of coefficients.',positive_polarity='Column sign fixed using sum of third raw moments of FIT CCA outputs from both modalities; same sign in both. No annotation or category names.',controls='Unmodified CCA; CCA with same sign orientation; three random oriented rotations.',coefficient_l1='Each modality weight matrix divided by its global RMS; minimize mean absolute coefficient.',varimax='Row Kaiser normalization; maximize sum of variance across input rows of squared loadings; average modalities equally.',loading='FIT Cov(standardized SAE input, CCA score), not regression coefficients.',output_l1='Mean per-example L1/L2 across modalities, /sqrt256.',kurtosis='Maximize mean marginal standardized fourth moment minus3; heavy-tail contrast, not full ICA algorithm.',logcosh='Maximize squared deviation of E logcosh(standardized output) from Gaussian expectation .37456720749; ICA contrast only.',shared_cokurtosis='Maximize mean(E[u²v²]-1-2E[uv]²), each coordinate standardized inside objective only.',evaluation='All runs COCO val2017; original fixed50 own-caption AUROC agreement and all171 image AUROC vs exact category-name text argmax. Report duplicates.',selection='Own label-free TUNE objective only. All methods and seeds reported, no annotation-based checkpoint selection.',optimization='Riemannian gradient polar retraction Armijo line search, max500steps. 80steps without TUNE improvement after100steps stop. Local solutions, no global-optimum assertion.'))
 for cond in ['cc3m-cc3m','cc3m-coco','coco-coco']:
  out=OUT/cond;out.mkdir(exist_ok=True);status(condition=cond,stage='prepare')
  cfg,arr,_=prior.prepare(cond,out,nfit=32768,seed=0)
  with np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256')) as z:w={s:z[s].astype(float) for s in ['image','text']}
  ts={key:tuple(((arr[key,s]-arr['mean',s])@torch.tensor(w[s],device='cuda',dtype=torch.float32)).detach() for s in w) for key in ['fit','tune']};del arr
  orientation=[q.cpu().numpy() for q in ts['fit']];ts['fit']=tuple(q[:8192] for q in ts['fit'])
  coeff=[];load=[]
  with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
   ni=len(z['image_ids']);mu=z['fit_mean'];cov=z['fit_second']-np.outer(mu,mu);sd=np.sqrt(np.maximum(cov.diagonal(),0));sd[sd==0]=1;cov/=sd[:,None]*sd[None,:]
  for side,block in [('image',cov[:ni,:ni]),('text',cov[ni:,ni:])]:
   ww=w[side];coeff.append(torch.tensor(ww/np.sqrt(np.mean(ww**2)),device='cuda',dtype=torch.float32));ll=block@ww;ll/=np.maximum(np.linalg.norm(ll,axis=1,keepdims=True),1e-12);load.append(torch.tensor(ll,device='cuda',dtype=torch.float32))
  # Varimax coefficient matrices need row normalization, unlike L1 global normalization.
  ts['coeff']=coeff;ts['load']=load
  data,parents=prior.eval_population(cfg,'test',out);test={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
  ix=Path(cfg['source_run'])/'index/val2017';presence=np.load(ix/'presence.npy').astype(bool);assert presence.shape==(5000,171)
  names=json.loads((ROOT/'runs/cca-171-category-names-2026-10-09/protocol.json').read_text())['prototype_texts']
  rawpath=out/'name-activations.npy'
  if not rawpath.exists():
   if cond!='coco-coco':raw=np.load(ROOT/'runs/cca-171-category-names-2026-10-09/category-name-activations.npz')['activation']
   else:
    encfg=json.loads((ROOT/'runs/cc3m-sae-2026-10-04/resolved-config.json').read_text())['encoder'];enc=CLIPEncoder(SimpleNamespace(**encfg));emb=np.concatenate([enc.texts(names[i:i+64]) for i in range(0,len(names),64)]);del enc
    sae=TopKSAE.from_pretrained(Path(cfg['source_run'])/'models/text').to('cuda').eval()
    with torch.no_grad():val,ind=sae.encode(torch.tensor(emb,device='cuda'));raw=torch.zeros((171,sae.config.latent_size),device='cuda').scatter(1,ind,val).cpu().numpy()
    del sae
   np.save(rawpath,raw)
  raw=np.load(rawpath)
  with np.load(Path(cfg['parent_run'])/'moments.npz') as z:ti=z['text_ids']
  proto=raw[:,ti]@(w['text']/data['scales']['text'][:,None])-data['means']['text']@(w['text']/data['scales']['text'][:,None])
  for name,rr,orient in [('cca_original',np.eye(256),None),('cca_oriented',np.eye(256),orientation)]:evaluate(cfg,data,parents,test,proto,presence,names,rr,name,out,orient)
  for seed in [0,1,2]:
   rng=np.random.default_rng(1200+seed);q,_=np.linalg.qr(rng.standard_normal((256,256)));evaluate(cfg,data,parents,test,proto,presence,names,q,f'random-{seed}',out,orientation)
  for method in METHODS:
   for seed in [0,1,2]:
    status(condition=cond,method=method,seed=seed,stage='fit')
    inp=dict(ts)
    if method=='coefficient_varimax':inp['coeff']=[m/m.norm(dim=1,keepdim=True).clamp_min(1e-12) for m in coeff]
    rr=fit(method,seed,inp,out);status(condition=cond,method=method,seed=seed,stage='evaluate');evaluate(cfg,data,parents,test,proto,presence,names,rr,f'{method}-{seed}',out,orientation)
  del ts,data;torch.cuda.empty_cache();save(out/'complete.json',dict(completed=True))
 save(OUT/'complete.json',dict(completed=True,elapsed=time.time()-START));status(stage='complete')
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
