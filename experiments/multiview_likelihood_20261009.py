"""Exploratory IVA/JISA likelihoods and ShIndICA specializations on frozen CCA.
No labels are used in optimization, sign alignment, or checkpoint selection.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time,argparse,traceback
import numpy as np
import torch
from scipy import linalg
import experiments.apbt_search_20261009 as p
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
from mm_sae.analysis.mapping_evaluation import paired_retrieval
ROOT=p.ROOT;OUT=ROOT/'runs/multiview-likelihood-2026-10-09';torch.set_num_threads(2)
METHODS=['iva_laplace','jisa_laplace_2','jisa_laplace_4','shindica_gmm_128','shindica_gmm_256','shindica_public_128']
def save(path,x):p.save(path,x)
def status(**kw):save(OUT/'progress.json',kw);print(json.dumps(kw),flush=True)
def whiten(x):
 mu=x.mean(0);z=x-mu;ev,evec=linalg.eigh(z.T@z/len(z));assert ev.min()>0
 w=(evec/np.sqrt(ev))@evec.T;return mu,w

def prepare(cond,out):
 cfg=p.old.load_config(ROOT/p.CONFIG[cond]);wz=np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'));weights={s:wz[s].astype(float) for s in ['image','text']}
 cache=out/'projected.npz'
 if not cache.exists():
  if cond=='cc3m-cc3m':
   z=np.load(ROOT/'runs/rotation-convergence-2026-10-09/projected.npz');a={k:z[k] for k in z.files}
  else:
   cfg,arr,_=p.prepare(cond,out,32768,0);a={}
   for kind in ['fit','tune']:
    for s in weights:a[kind+'_'+s]=((arr[kind,s]-arr['mean',s])@torch.as_tensor(weights[s],device='cuda',dtype=torch.float32)).cpu().numpy()
   del arr;torch.cuda.empty_cache()
  np.savez(cache,**a)
 with np.load(cache) as z:a={k:z[k].astype(float) for k in z.files}
 return cfg,weights,a

def loss_fn(method,z,ds,noise):
 a,b=[x@d for x,d in zip(z,ds)]
 ld=sum(torch.linalg.slogdet(d)[1] for d in ds)
 if method.startswith(('iva','jisa')):
  g=1 if method.startswith('iva') else int(method.rsplit('_',1)[1]);r=a.shape[1]
  rad=(a.reshape(-1,r//g,g).square().sum(2)+b.reshape(-1,r//g,g).square().sum(2)+1e-12).sqrt()
  return (rad.sum(1).mean()-ld)/r
 c=int(method.rsplit('_',1)[1])
 if method.startswith('shindica_public'):
  lc=lambda x:(torch.logaddexp(x,-x)-np.log(2)).sum(1).mean()
  # Exact formula of public _siica_si_main for two views, hyper=1;
  # covariance .5 input and orthogonal matrices match public whitening.
  return (lc(a[:,c:])+lc(b[:,c:])+lc(a[:,:c]+b[:,:c])-(a[:,:c]*b[:,:c]).sum(1).mean())/a.shape[1]
 # Exact ShIndICA generative specialization. Each unit-variance source has
 # equal mixture of N(0,.1),N(0,1.9); scalar isotropic Gaussian noise.
 nv=noise.exp();vs=torch.tensor([.1,1.9],device=a.device,dtype=a.dtype)
 aa=a[:,:c,None];bb=b[:,:c,None]
 det=nv*(2*vs+nv)
 quad=((vs+nv)*(aa*aa+bb*bb)-2*vs*aa*bb)/det
 lp=-np.log(2*np.pi)-.5*det.log()-.5*quad-np.log(2)
 ll=torch.logsumexp(lp,2).sum(1).mean()
 for x in [a[:,c:],b[:,c:]]:
  vv=vs+nv;lp=-.5*(np.log(2*np.pi)+vv.log()+x[:,:,None].square()/vv)-np.log(2)
  ll=ll+torch.logsumexp(lp,2).sum(1).mean()
 return (-ll-ld)/a.shape[1]

def optimize(method,fit,tune,out,max_steps):
 path=out/(method+'-model.npz')
 if path.exists():return
 dtype=torch.float64;device='cuda';d=fit[0].shape[1]
 zs=[torch.tensor(a,device=device,dtype=dtype) for a in fit];vs=[torch.tensor(a,device=device,dtype=dtype) for a in tune]
 public=method.startswith('shindica_public')
 raw=[torch.nn.Parameter(torch.eye(d,device=device,dtype=dtype)) for _ in range(2)]
 noise=torch.nn.Parameter(torch.tensor(np.log(.3),device=device,dtype=dtype))
 params=raw+([noise] if method.startswith('shindica_gmm') else [])
 # Public ShIndICA uses orthogonal transforms after whitening to covariance .5.
 if public:zs=[z/np.sqrt(2) for z in zs];vs=[v/np.sqrt(2) for v in vs]
 def ds():
  if not public:return raw
  # Cayley chart, initialized at identity; keeps public orthogonality.
  eye=torch.eye(d,device=device,dtype=dtype)
  return [torch.linalg.solve(eye+(q-q.T)/2,eye-(q-q.T)/2) for q in raw]
 opt=torch.optim.LBFGS(params,lr=1,max_iter=10,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
 hist=[];best=np.inf;bestd=None;bestnoise=None;t0=time.time();prev=None;stable=0;stop='iteration_budget';calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();f=loss_fn(method,zs,ds(),noise)
  if not torch.isfinite(f):raise FloatingPointError('nonfinite likelihood')
  f.backward();calls+=1;return f
 for outer in range(max_steps//10+1):
  with torch.no_grad():
   matrices=ds();f=float(loss_fn(method,zs,matrices,noise));v=float(loss_fn(method,vs,matrices,noise))
   if v<best:
    best=v;bestd=[q.detach().cpu().numpy().copy() for q in matrices];bestnoise=float(noise)
   state=dict(iteration=outer*10,fit=f,tune=v,seconds=time.time()-t0,closure_calls=calls)
   hist.append(state);save(out/(method+'-history.json'),hist)
   if outer%5==0:status(condition=out.name,method=method,stage='fit',**state)
   if prev is not None and abs(prev-f)<1e-7*max(1,abs(prev)):stable+=1
   else:stable=0
   if stable>=5:stop='five_checks_small_training_change';break
   prev=f
  if outer*10>=max_steps:break
  opt.step(closure)
 final=[q.detach().cpu().numpy() for q in ds()]
 np.savez_compressed(path,image=bestd[0],text=bestd[1],noise_log=bestnoise,final_image=final[0],final_text=final[1])
 save(out/(method+'-fit.json'),dict(method=method,steps=outer*10,stop=stop,seconds=time.time()-t0,selected='minimum own label-free tune objective',best_tune=best,noise_variance=float(np.exp(bestnoise)),condition_numbers=[float(np.linalg.cond(x)) for x in bestd],global_optimum_claimed=False,history=hist))

def matching(scores,data,cfg):
 arr={}
 for s in scores:
  for half in ['a','b']:
   key=s+'_'+half;mask=data['masks'][key];arr[key]=auc_matrix(scores[s][mask],data['labels'][s][mask],chunk_size=32);arr[key+'_variable']=variable_coordinates(scores[s][mask])
 arr['eligible']=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert arr['eligible'].sum()==50
 arr['category_ids']=np.array([c['id'] for c in data['concepts']])
 return p.old.evaluation(arr,{**cfg,'n_null':0},signed=False),p.old.evaluation(arr,{**cfg,'n_null':0},signed=True)
def metrics(scores,data,parents,cfg):
 ret=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128)
 ma,ms=matching(scores,data,cfg)
 return dict(retrieval={d:{k:v for k,v in val.items() if k not in ['hits','ranks']} for d,val in ret.items()},matching50=ma,signed_matching50=ms,output_coordinates=scores['image'].shape[1])

def evaluate(cond,cfg,w,a,out,methods,pre):
 data,parents=p.eval_population(cfg,'test',out)
 test={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
 tr={s:a['fit_'+s][:8192] for s in w}
 for name in ['cca_original','cca_whitened']+methods:
  path=out/(name+'-result.json')
  if path.exists():continue
  if name=='cca_original':scores=test;orient=tr;group=1;c=256
  elif name=='cca_whitened':
   scores={s:(test[s]-pre[s][0])@pre[s][1] for s in w};orient={s:(tr[s]-pre[s][0])@pre[s][1] for s in w};group=1;c=256
  else:
   zz=np.load(out/(name+'-model.npz'));group=int(name.rsplit('_',1)[1]) if name.startswith('jisa') else 1
   c=int(name.rsplit('_',1)[1]) if name.startswith('shindica') else 256
   ds={s:zz[s].copy() for s in w};scale=np.sqrt(2) if name.startswith('shindica_public') else 1
   orient={s:(tr[s]-pre[s][0])@pre[s][1]@ds[s]/scale for s in w}
   # Likelihoods have sign / within-group rotation ambiguities.
   # Align text group bases to image using only FIT paired observations.
   align=np.eye(256)
   for lo in range(0,c,group):
    sl=slice(lo,lo+group);l,_,r=linalg.svd(orient['text'][:,sl].T@orient['image'][:,sl]);align[sl,sl]=l@r
   ds['text']=ds['text']@align;orient['text']=orient['text']@align
   sign=np.where(sum((orient[s][:,:c]**3).mean(0) for s in w)<0,-1.,1.)
   for s in w:ds[s][:,:c]*=sign;orient[s]=orient[s][:,:c]*sign
   scores={s:((test[s]-pre[s][0])@pre[s][1]@ds[s]/scale)[:,:c] for s in w}
   # Export affine maps from standardized SAE activations for reproducibility.
   transformed={s:w[s]@pre[s][1]@ds[s][:,:c]/scale for s in w}
   bias={s:-pre[s][0]@pre[s][1]@ds[s][:,:c]/scale for s in w}
   np.savez_compressed(out/(name+'-transforms.npz'),**transformed,image_bias=bias['image'],text_bias=bias['text'],text_group_alignment=align)
  result=dict(method=name,condition=cond,primary=metrics(scores,data,parents,cfg))
  if group>1:
   groupscores={s:np.linalg.norm(x.reshape(len(x),-1,group),axis=2) for s,x in scores.items()}
   result['group_norm']=metrics(groupscores,data,parents,cfg)
   result['group_warning']='Coordinate matching depends on arbitrary within-group basis. Group matching has fewer candidates; do not compare counts as identical task.'
  if name!='cca_original':
   scale={s:np.maximum(np.std(orient[s],axis=0),1e-12) for s in w}
   result['fit_std_retrieval']=metrics({s:scores[s]/scale[s] for s in w},data,parents,cfg)['retrieval']
  save(path,result);status(condition=cond,method=name,stage='evaluated',matched=result['primary']['matching50']['summary']['agree_at1_count'])
 # Existing output-L1 reference from original experiment, same test protocol.
 ref=ROOT/'runs/rotation-convergence-2026-10-09/output_l1-0-final-result.json' if cond=='cc3m-cc3m' else ROOT/'runs/cca-unsupervised-rotations-2026-10-09'/cond/'output_l1-0-result.json'
 if ref.exists():save(out/'output_l1_reference.json',dict(source=str(ref),result=json.loads(ref.read_text())))

def main(args):
 OUT.mkdir(parents=True,exist_ok=True)
 save(OUT/'protocol.json',dict(conditions=args.conditions.split(','),methods=METHODS,rank=256,fit_pairs=8192,tune_pairs=2048,initialization='identity in separately FIT-whitened frozen CCA outputs; single-start exploratory',steps=args.steps,
 iva='Real multivariate radial Laplace density per cross-view scalar pair; exact likelihood, unconstrained invertible transforms.',
 jisa='Radial Laplace density per concatenated cross-view group, group size 2 or 4 per modality. Equal fixed groups; grouping emerges through transforms, group sizes not learned. FIT within-group Procrustes alignment for coordinate retrieval; group norm evaluation separately.',
 shindica_gmm='Exact noisy shared/private generative specialization: independent scalar equal mixtures of N(0,.1) and N(0,1.9), plus isotropic Gaussian noise with learned common variance. 128 or 256 shared factors; remaining private. Not original paper distribution estimator.',
 shindica_public='Public repository two-view formula (logcosh private + logcosh shared sum - shared inner product), hyper=1, whitening covariance .5, orthogonal Cayley optimization. Surrogate likelihood, NOT exact MLE.',
 comparison='COCO val2017 5000 images /25014 captions. Frozen original 50 category own-caption representative agreement, positive and sign-allowed. No labels for train, checkpoint or alignment. c128 searched using128 common coordinates and includes CCA truncation control.',
 caveats='Likelihood models still impose substantive distribution assumptions. No recovery theorem claimed for this frozen CCA preprocessing. Nonconvex, one initialization. Different per-modality transforms need not preserve original retrieval.'))
 for cond in args.conditions.split(','):
  out=OUT/cond;out.mkdir(exist_ok=True);status(condition=cond,stage='prepare')
  cfg,w,a=prepare(cond,out);pre={};fit=[];tune=[]
  for s in ['image','text']:
   mu,h=whiten(a['fit_'+s][:8192]);pre[s]=(mu,h);fit.append((a['fit_'+s][:8192]-mu)@h);tune.append((a['tune_'+s]-mu)@h)
  np.savez(out/'preprocessing.npz',**{s+'_mean':pre[s][0] for s in pre},**{s+'_whitening':pre[s][1] for s in pre})
  for method in METHODS:optimize(method,fit,tune,out,args.steps)
  evaluate(cond,cfg,w,a,out,METHODS,pre)
  save(out/'complete.json',dict(complete=True))
 save(OUT/'complete.json',dict(complete=True))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--conditions',default='cc3m-cc3m,cc3m-coco,coco-coco');ap.add_argument('--steps',type=int,default=400);args=ap.parse_args()
 try:main(args)
 except Exception as e:
  save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
