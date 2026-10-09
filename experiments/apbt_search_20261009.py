"""Exploratory APB^T search. No annotations in gradients, support selection, P, or checkpoint.
Development annotations are used for comparing variants. Never call this label-free model selection.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,json,time,torch,traceback,argparse
import torch.nn.functional as F
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from mm_sae.metrics.regression import Moments
from mm_sae.analysis.mapping_evaluation import paired_retrieval
import experiments.caption_matching_20261009 as cap
old=cap.old
ROOT=Path('/mnt/working/mm-sae');BASE=ROOT/'runs/apbt-search-2026-10-09'
CONFIG={'coco-coco':'configs/representative-agreement-local.yaml','cc3m-coco':'runs/cc3m-followup-2026-10-05/configs/coco-fit-agreement.yaml','cc3m-cc3m':'runs/cc3m-followup-2026-10-05/configs/cc3m-fit-agreement.yaml'}
torch.set_num_threads(4)
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()));q.replace(p)
@torch.no_grad()
def project(w,positive):
 if positive:w.clamp_(min=0)
 ix=w.abs().topk(16,dim=0).indices;w.mul_(torch.zeros_like(w).scatter_(0,ix,1));w.div_(w.norm(dim=0,keepdim=True).clamp_min(1e-9))
def prepare(cond,out,nfit=32768,seed=0):
 cfg=old.load_config(ROOT/CONFIG[cond]);parent=Path(cfg['parent_run']);source=ROOT/'runs/cc3m-sae-2026-10-04' if cond=='cc3m-cc3m' else Path(cfg['source_run'])
 pop=json.loads((parent/'population.json').read_text())
 with np.load(parent/'moments.npz') as z:
  ids={s:z[s+'_ids'].copy() for s in ['image','text']};mom=Moments(int(z['fit_n']),z['fit_mean'],z['fit_second'])
 ni=len(ids['image']);means={'image':mom.mean[:ni],'text':mom.mean[ni:]};scales={'image':mom.scale[:ni],'text':mom.scale[ni:]}
 records=json.loads((source/'index/train2017/images.json').read_text());names=np.array([r['image_id'] for r in records]);del records
 rng=np.random.default_rng(911+seed);fit=np.flatnonzero(np.isin(names,pop['fit_image_ids']));tune=np.flatnonzero(np.isin(names,pop['tune_image_ids']))
 fit=rng.choice(fit,min(nfit,len(fit)),replace=False);tune=rng.choice(tune,min(2048,len(tune)),replace=False)
 assert not np.intersect1d(fit,tune).size
 par=np.load(source/'index/train2017/parents.npy');order=np.argsort(par,kind='stable');counts=np.bincount(par,minlength=len(names));starts=np.r_[0,counts.cumsum()[:-1]]
 ti={key:order[starts[rr]+(rng.random(len(rr))*counts[rr]).astype(int)] for key,rr in [('fit',fit),('tune',tune)]};arrays={}
 for s in ['image','text']:
  raw=sparse.load_npz(source/'activations/train2017'/f'{s}.npz').tocsr()
  for key,rr in [('fit',fit),('tune',tune)]:
   rows=rr if s=='image' else ti[key]
   arrays[key,s]=torch.tensor((raw[rows][:,ids[s]].toarray()/scales[s]).astype('float32'),device='cuda')
  arrays['mean',s]=torch.tensor((means[s]/scales[s]).astype('float32'),device='cuda');del raw
 with np.load(next(m['path'] for m in cfg['models'] if m['key']=='sparse_cca_16')) as z:initial=[z[s].astype('float32') for s in ['image','text']]
 np.savez_compressed(out/'split.npz',fit_rows=fit,tune_rows=tune,fit_text_rows=ti['fit'],tune_text_rows=ti['tune'])
 return cfg,arrays,initial

def corr(u,v):
 u=u-u.mean(0);v=v-v.mean(0)
 return F.normalize(u,dim=0).T@F.normalize(v,dim=0)

def losses(x,y,a,b,p):
 u=x@a;rawv=y@b;v=rawv[:,p]
 logits=F.normalize(u,dim=1)@F.normalize(v,dim=1).T/.07;ii=torch.arange(len(x),device=x.device)
 pair=(F.cross_entropy(logits,ii)+F.cross_entropy(logits.T,ii))/2
 c=corr(u,v);jj=torch.arange(len(p),device=x.device)
 coord=(F.cross_entropy(c/.1,jj)+F.cross_entropy(c.T/.1,jj))/2
 rec=(((u@a.T-x).square().mean()/x.square().mean().clamp_min(1e-6))+((rawv@b.T-y).square().mean()/y.square().mean().clamp_min(1e-6)))/2
 activity=(u.abs().sum(1)/u.norm(dim=1).clamp_min(1e-6)+v.abs().sum(1)/v.norm(dim=1).clamp_min(1e-6)).mean()/2/(len(p)**.5)
 return pair,coord,rec,activity

SPEC=[dict(name='signed_pair',positive=False,coord=0,rec=0),dict(name='positive_pair',positive=True,coord=0,rec=0),dict(name='positive_coord_1',positive=True,coord=1,rec=0),dict(name='positive_coord_4',positive=True,coord=4,rec=0),dict(name='positive_reconstruct_1',positive=True,coord=0,rec=1),dict(name='positive_reconstruct_4',positive=True,coord=0,rec=4),dict(name='positive_joint',positive=True,coord=1,rec=1),dict(name='positive_learnP',positive=True,coord=1,rec=1,learnP=True)]

def fit_one(cfg,arr,initial,spec,out,args):
 name=spec['name'];path=out/(name+'.npz')
 if path.exists():return
 torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
 a,b=[torch.nn.Parameter(torch.tensor(w,device='cuda')) for w in initial]
 if spec['positive']:
  with torch.no_grad():a.abs_();b.abs_()
 project(a,spec['positive']);project(b,spec['positive']);p=torch.arange(a.shape[1],device='cuda');support0=[w.ne(0).detach().clone() for w in [a,b]]
 if spec.get('initpath'):
  with np.load(spec['initpath']) as z:
   with torch.no_grad():a.copy_(torch.tensor(z['image'],device='cuda'));b.copy_(torch.tensor(z['text'],device='cuda'))
   if 'permutation' in z:p=torch.tensor(z['permutation'],device='cuda')
 opt=torch.optim.Adam([a,b],lr=args.lr)
 centered=not spec['positive'];xx={}
 for split in ['fit','tune']:
  for s in ['image','text']:xx[split,s]=arr[split,s]-arr['mean',s] if centered else arr[split,s]
 def objective(x,y):
  vals=losses(x,y,a,b,p);return spec.get('pair',1)*vals[0]+spec['coord']*vals[1]+spec['rec']*vals[2]+spec.get('activity',0)*vals[3],vals
 def validation():
  with torch.no_grad():
   vals=np.array([[float(v) for v in objective(xx['tune','image'][j:j+512],xx['tune','text'][j:j+512])[1]] for j in range(0,len(xx['tune','image']),512)]).mean(0)
  return float(spec.get('pair',1)*vals[0]+spec['coord']*vals[1]+spec['rec']*vals[2]+spec.get('activity',0)*vals[3]),vals
 best=np.inf;stale=0;hist=[];start=time.monotonic()
 for ep in range(args.epochs+1):
  t0=time.monotonic();tl=[]
  if ep:
   seq=rng.permutation(len(xx['fit','image']))
   if spec.get('learnP'):
    with torch.no_grad():
     cross=corr(xx['fit','image'][:4096]@a,xx['fit','text'][:4096]@b).cpu().numpy();_,perm=linear_sum_assignment(-cross);p=torch.tensor(perm,device='cuda')
   for i in range(0,len(seq),512):
    ix=torch.tensor(seq[i:i+512],device='cuda');value,_=objective(xx['fit','image'][ix],xx['fit','text'][ix]);assert torch.isfinite(value)
    opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_([a,b],5);opt.step();project(a,spec['positive']);project(b,spec['positive']);tl.append(float(value.detach()))
  vl,vals=validation()
  if vl<best-1e-5:
   best=vl;bestep=ep;stale=0;weights=[w.detach().cpu().numpy().copy() for w in [a,b]];pp=p.cpu().numpy().copy()
  else:stale+=1
  row=dict(epoch=ep,tune_loss=vl,pair_loss=float(vals[0]),coord_loss=float(vals[1]),rec_loss=float(vals[2]),activity=float(vals[3]),train_loss=float(np.mean(tl)) if tl else None,elapsed=time.monotonic()-start,epoch_seconds=time.monotonic()-t0,changed_support=[int((w.ne(0)!=s).sum()) for w,s in zip([a,b],support0)])
  hist.append(row);save(out/(name+'-history.json'),hist);save(BASE/'progress.json',dict(state='training',condition=args.condition,name=name,**row));print(json.dumps(dict(name=name,**row)),flush=True)
  if ep and stale>=5:break
 np.savez_compressed(path,image=weights[0],text=weights[1],permutation=pp,centered=centered)
 save(out/(name+'-fit.json'),dict(spec=spec,best_epoch=bestep,completed_epoch=ep,best_tune_loss=best,seconds=time.monotonic()-start,final_support_changes=hist[-1]['changed_support']))

def eval_population(cfg,kind,out):
 if kind=='tune':
  annpath=Path(cfg.get('annotation_population',str(Path(cfg['parent_run'])/'population.json')));pop=json.loads(annpath.read_text());rng=np.random.default_rng(823)
  pop['tune_image_ids']=rng.choice(pop['tune_image_ids'],min(5000,len(pop['tune_image_ids'])),replace=False).tolist();save(out/'development-population.json',pop);cfg={**cfg,'annotation_population':str(out/'development-population.json')}
 data=cap.base_load(cfg,kind);split='train2017' if kind=='tune' else 'val2017';idx=Path(cfg['source_run'])/'index'/split
 ids=np.array([r['image_id'] for r in json.loads((idx/'images.json').read_text())]);par=np.load(idx/'parents.npy');keep=np.isin(ids,data['metadata']['half_a_image_ids']+data['metadata']['half_b_image_ids']);remap=np.cumsum(keep)-1;parents=remap[par[keep[par]]]
 objects=np.flatnonzero(np.array(json.loads((idx/'concept_ids.json').read_text()))<91)
 data['labels']['text']=np.load(idx/'mentions.npy')[keep[par]][:,objects].astype(bool)
 counts={k:data['labels'][k.split('_')[0]][m].sum(0) for k,m in data['masks'].items()}
 eligible=(counts['image_a']>=50)&(counts['text_b']>=50)
 for key in ['image_a','text_b']:eligible &= counts[key]<data['masks'][key].sum()
 data['eligible']=eligible
 return data,parents

def evaluate(cfg,data,parents,name,path,out,kind):
 target=out/(name+'-'+kind+'.json')
 if target.exists():return json.loads(target.read_text())
 save(BASE/'progress.json',dict(state='evaluating',name=name,population=kind))
 with np.load(path) as z:
  weights={s:z[s].copy() for s in ['image','text']};centered=bool(z['centered']) if 'centered' in z else True
  p=z['permutation'].copy() if 'permutation' in z else np.arange(weights['image'].shape[1]);weights['text']=weights['text'][:,p]
 scores={s:np.asarray(data['raw'][s]@(weights[s]/data['scales'][s][:,None])) for s in weights}
 if centered:
  for s in scores:scores[s]-=data['means'][s]@(weights[s]/data['scales'][s][:,None])
 ret=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128)
 arr=old.calculate_aucs(cfg,data,weights,lambda k:None)
 res=dict(method=name,population=kind,retrieval={d:{k:v for k,v in z.items() if k!='hits'} for d,z in ret.items()},positive=old.evaluation(arr,{**cfg,'n_null':0},signed=False),signed=old.evaluation(arr,{**cfg,'n_null':0},signed=True),denominator=int(data['eligible'].sum()),category_ids=arr['category_ids'][data['eligible']].tolist(),metric='Independent representative index agreement; text uses own-caption dictionary labels, not image labels. Not concept purity.')
 save(target,res);np.savez_compressed(out/(name+'-'+kind+'-auc.npz'),**arr)
 print(json.dumps(dict(evaluated=name,population=kind,match_positive=res['positive']['summary']['agree_at1_count'],match_signed=res['signed']['summary']['agree_at1_count'],r5=[ret[d]['recall'][5] for d in ret])),flush=True)
 return res

def main(args):
 out=BASE/args.round/args.condition/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
 if args.evaluate:
  cfg=old.load_config(ROOT/CONFIG[args.condition]);data,parents=eval_population(cfg,args.evaluate,out)
  jobs=[('sparse_cca_16',Path(next(m['path'] for m in cfg['models'] if m['key']=='sparse_cca_16')))]
  jobs += [(p.stem,p) for p in out.glob('*.npz') if p.stem not in ['split'] and not p.stem.endswith('-auc')]
  for name,path in jobs:evaluate(cfg,data,parents,name,path,out,args.evaluate)
 else:
  cfg,arr,initial=prepare(args.condition,out,args.nfit,args.seed)
  specs=SPEC if not args.specs else json.loads(Path(args.specs).read_text())
  save(out/'protocol.json',dict(args=vars(args),specs=specs,objective='Symmetric cosine InfoNCE + beta coordinate discrimination via normalized cross-correlation crossentropy + alpha tied nonnegative reconstruction + optional Hoyer activity',constraints='16 nonzeros/column and unit L2 norm; nonnegative when specified. P identity or Hungarian assignment on FIT profiles only.',model_selection='Own unlabelled tune objective checkpoint. Comparison across variants uses annotated development data; this is not annotation-free hyperparameter selection.',scope='Exploratory. No test data is evaluated during search. A P B^T numerator, cosine in paired concept coordinates.',source_cfg=cfg))
  for spec in specs:fit_one(cfg,arr,initial,spec,out,args)
  del arr;torch.cuda.empty_cache();data,parents=eval_population(cfg,'tune',out)
  jobs=[('sparse_cca_16',Path(next(m['path'] for m in cfg['models'] if m['key']=='sparse_cca_16')))] +[(sp['name'],out/(sp['name']+'.npz')) for sp in specs]
  for name,path in jobs:evaluate(cfg,data,parents,name,path,out,'tune')
 save(BASE/'progress.json',dict(state='completed',round=args.round,condition=args.condition))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--condition',default='cc3m-coco');p.add_argument('--round',default='round1');p.add_argument('--seed',type=int,default=0);p.add_argument('--nfit',type=int,default=32768);p.add_argument('--epochs',type=int,default=30);p.add_argument('--lr',type=float,default=.003);p.add_argument('--specs');p.add_argument('--evaluate',choices=['tune','test']);args=p.parse_args()
 try:main(args)
 except Exception as e:save(BASE/'progress.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc()));raise
