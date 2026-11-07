"""Category-held-out audit, train-only hybrid screening, longer shared rotations."""
import experiments.sample_score_rotation_20261012 as m
from experiments.sample_score_rotation_20261012 import *
from collections import Counter
PREV=ROOT/'runs/sample-score-screen-2026-10-12'
OUT=ROOT/'runs/score-validation-2026-10-12';m.OUT=OUT
for sub in ['fits','eval','checkpoints','snapshots']:(OUT/sub).mkdir(parents=True,exist_ok=True)
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
def ref(name,seed):
 return SRC/f'rank256/covariance-svd-{seed}-oriented.npz' if name=='covariance' else ROOT/'runs/feature-redundancy-cycle-2026-10-12/fits'/f'cycle-best-s{seed}.npz' if name=='cycle-best' else PREV/'fits'/f'confirm-cross-std0-covariance-s{seed}.npz'
def category_setup(e):
 path=OUT/'labels.npz'
 if path.exists():
  f=np.load(path);return {s:a.tensor(f[s]) for s in S},json.loads((OUT/'categories.json').read_text())
 old=set(e.data['metadata']['eligible_category_ids']);concepts=e.data['concepts'];labels={s:np.array(e.data['labels'][s],dtype=float) for s in S};groups={'old50':[],'heldout_coco':[],'caption_words':[]};audit=[]
 for j,c in enumerate(concepts):
  ok=True;cnt={}
  for side,slot in [('image',0),('text',1)]:
   nums=[labels[side][p0[slot].cpu().numpy(),j].sum() for p0 in e.splits[:20]];lens=[len(p0[slot]) for p0 in e.splits[:20]];cnt[side]=dict(min_positive=int(min(nums)),min_negative=int(min(n-v for n,v in zip(lens,nums))))
   ok &= cnt[side]['min_positive']>=20 and cnt[side]['min_negative']>=20
  if c['id'] in old:groups['old50'].append(j)
  elif ok:groups['heldout_coco'].append(j)
  audit.append(dict(**c,eligible_new=bool(ok and c['id'] not in old),counts=cnt))
 # Auxiliary word concepts. Vocabulary order fixed using training caption frequency,
 # excluding old50 aliases. Image label is any of its own captions mentioning word.
 vocab=json.loads((PAIR/'vocabulary.json').read_text());words=[v['word'] for v in vocab['ranked_words'] if not v['excluded']][:1000]
 idx=Path(e.cfg['source_run'])/'index/val2017';ids=np.array([v['image_id'] for v in json.loads((idx/'images.json').read_text())]);ids_keep=np.isin(ids,e.data['metadata']['half_a_image_ids']+e.data['metadata']['half_b_image_ids']);parents=np.load(idx/'parents.npy');rr=np.flatnonzero(ids_keep[parents]);caps=json.loads((idx/'captions.json').read_text());nlp=spacy.load('en_core_web_sm',disable=['parser','ner']);lookup={w:j for j,w in enumerate(words)};yt=np.zeros((len(rr),len(words)),bool)
 for i,doc in enumerate(nlp.pipe((caps[int(j)]['text'] for j in rr),batch_size=512)):
  for t in doc:
   j=lookup.get(t.lemma_.lower())
   if j is not None and t.pos_ in {'NOUN','PROPN','VERB','ADJ','ADV'} and t.is_alpha and not t.is_stop:yt[i,j]=True
 yi=np.zeros((len(e.x['image']),len(words)),bool);np.logical_or.at(yi,e.parents,yt);chosen=[]
 for j,w in enumerate(words):
  ok=all(20<=int(y[p0[k].cpu().numpy(),j].sum())<=len(p0[k])-20 for p0 in e.splits[:20] for k,y in enumerate([yi,yt]))
  if ok:chosen.append(j)
  if len(chosen)==100:break
 offset=len(concepts)
 for j in chosen:groups['caption_words'].append(len(concepts));concepts=concepts+[dict(id='word:'+words[j],name=words[j])]
 for side,y in [('image',yi),('text',yt)]:labels[side]=np.concatenate([labels[side],y[:,chosen]],1)
 spec=dict(concepts=concepts,groups=groups,coco_eligibility=audit,min_positive_and_negative_each_split=20,word_selection='training document frequency, excluding old50 finite alias lexicon, first100 with >=20 positives and negatives in every evaluation split',word_image_labels='weak proxy: any linked evaluation caption mentions lemma',word_text_labels='own caption lemma mentions',category_holdout_only=True)
 save(OUT/'categories.json',spec);np.savez_compressed(path,**labels);return {s:a.tensor(labels[s]) for s in S},spec

def feature_stats(w):
 v=w.square().sort(0,descending=True).values;cnt=torch.where(v.sum(0)>0,(v.cumsum(0)<.9*v.sum(0)).sum(0)+1,0);nonzero=(w!=0).sum(0)
 return dict(n90_mean=float(cnt.double().mean()),n90_median=float(cnt.double().median()),n90_min=int(cnt.min()),n90_max=int(cnt.max()),mean=float(nonzero.double().mean()),empty=int((nonzero==0).sum()))
def measure(e,labels,spec,name,seed,r):
 sg=positive_sign({s:e.base[s]@r for s in S});r=r*sg;dense={s:e.base[s]@r for s in S}
 for cut in ['dense','column64','global64']:
  dest=OUT/'eval'/f'{name}-{cut}-s{seed}.json'
  if dest.exists():continue
  ws={}
  for s,w in dense.items():
   if cut=='dense':ws[s]=w
   elif cut=='column64':ws[s]=hard(w,64)
   else:
    ix=w.abs().flatten().argsort(descending=True,stable=True)[:256*64];ws[s]=torch.zeros_like(w).flatten().scatter_(0,ix,w.flatten()[ix]).reshape_as(w)
  sc={s:e.x[s]@ws[s] for s in S};rows=[]
  for pair0 in e.splits[:20]:
   au=[a.gpu_auc(sc[s][ix],labels[s][ix]) for s,ix in zip(S,pair0)];valid=torch.isfinite(au[0])&torch.isfinite(au[1]);has=valid.any(0)
   for q in au:q[~valid]=-torch.inf
   ii=[q.argmax(0) for q in au];col=torch.arange(len(spec['concepts']),device='cuda');best=[q[k,col] for q,k in zip(au,ii)];match=(ii[0]==ii[1])&has
   row=dict(match=match.int().cpu().tolist(),best_auc={s:b.cpu().tolist() for s,b in zip(S,best)},responsive={s:{str(t):(q>=t).sum(0).cpu().tolist() for t in [.8,.9,.95]} for s,q in zip(S,au)})
   for t in [.7,.8,.9]:row['match'+str(t)]=(match&(best[0]>=t)&(best[1]>=t)).int().cpu().tolist()
   rows.append(row)
  ret=retrieval(sc,e.parents)
  save(dest,dict(name=name,seed=seed,cut=cut,rows=rows,retrieval=ret,coefficients={s:feature_stats(ws[s]) for s in S},dense_coefficients={s:feature_stats(dense[s]) for s in S}))
  log(stage='evaluation',name=name,seed=seed,cut=cut)

class Combined:
 def __init__(self,e,obj,r0,kind,weight):
  self.e=e;self.obj=obj;self.n=obj.n;self.kind=kind;self.weight=weight
  val,g=self.parts(r0);self.scales=[max(float(m.tangent(r0,gg).norm()),1e-12) for gg in g];self.offsets=list(map(float,val));self.calibration=dict(kind=kind,weight=weight,tangent_gradient_norms=self.scales,offsets=self.offsets,start='saved score-only500 iteration fit of same seed',fixed_scales=True)
 def parts(self,r):
  # Only joint fourth moment is needed, avoiding unnecessary marginal fourth gradients.
  obj=self.obj;q=[(r*(s@r)).sum(0) for s in obj.second];joint=torch.zeros(256,device='cuda',dtype=torch.float64);g=torch.zeros_like(r)
  for st in range(0,obj.n,32768):
   x,y=[v[st:st+32768] for v in obj.x];z=x@r;w=y@r;joint+=(z.square()*w.square()).sum(0);g+=2*x.T@(z*w.square())+2*y.T@(w*z.square())
  score=(joint/obj.n-q[0]*q[1]).sum();g=g/obj.n-2*(obj.second[0]@r)*q[1]-2*(obj.second[1]@r)*q[0]
  if self.kind=='loading':
   v=torch.zeros((),device='cuda',dtype=torch.float64);gg=torch.zeros_like(r)
   for l in obj.l:
    z=l@r;mu=z.square().mean(0);v+=(z.pow(4).mean(0)-mu.square()).sum()/2;gg+=2/len(l)*l.T@(z.pow(3)-z*mu)
  elif self.kind=='target':
   z={s:self.e.base[s]@r for s in S};t={s:target(z[s][None],'rowfirst',1,'mass90')[0] for s in S};v=-sum((z[s]-t[s]).square().sum() for s in S)/2;gg=-sum(self.e.base[s].T@(z[s]-t[s]) for s in S)
  else:v=score;gg=g
  return [score,v],[g,gg]
 def evaluate(self,r,kind=None,norm=None):
  val,g=self.parts(r);w=self.weight
  value=(1-w)*(val[0]-self.offsets[0])/self.scales[0]+w*(val[1]-self.offsets[1])/self.scales[1]
  gg=(1-w)*g[0]/self.scales[0]+w*g[1]/self.scales[1]
  return float(value),gg,list(map(float,val))

def fit(obj,r,name,seed,limit,callback=None):
 path=OUT/'fits'/f'{name}-s{seed}.npz'
 if path.exists():return a.tensor(np.load(path)['rotation'])
 hist=[];ss=[];ys=[];stop='iteration_limit';start=time.time();value,g,parts=obj.evaluate(r);grad=-m.tangent(r,g);last=0
 for it in range(limit+1):
  gn=float(grad.norm());hist.append(dict(step=it,objective=value,gradient=gn,components=parts))
  if it%250==0 or it==limit or gn<1e-6:
   snap=OUT/'snapshots'/f'{name}-s{seed}-i{it}.npz';np.savez_compressed(snap,rotation=r.cpu().numpy());save(snap.with_suffix('.json'),dict(history=hist,calibration=obj.calibration))
   log(stage='fit',name=name,seed=seed,step=it,gradient=gn,seconds=time.time()-start)
   if callback:callback(r,f'{name}-i{it}',seed)
  last=it
  if gn<1e-6:stop='gradient';break
  if it==limit:break
  q=grad.clone();aa=[];rho=[]
  for s,y in reversed(list(zip(ss,ys))):
   rh=1/float((s*y).sum());al=rh*float((s*q).sum());q-=al*y;aa.append(al);rho.append(rh)
  scale=float((ss[-1]*ys[-1]).sum()/ys[-1].square().sum()) if ss else 1.;direction=q*scale
  for s,y,al,rh in zip(ss,ys,reversed(aa),reversed(rho)):direction+=s*(al-rh*float((y*direction).sum()))
  direction=-m.tangent(r,direction);slope=float((grad*direction).sum())
  if slope>=-1e-20:direction=-grad;ss=[];ys=[];slope=-float(grad.square().sum())
  step=1.;accepted=False
  for trial in range(24):
   rn=a.polar(r+step*direction);nv,ng,nparts=obj.evaluate(rn)
   if nv>=value-1e-4*step*slope:accepted=True;break
   step*=.5
  if not accepted:stop='line_search_stalled';break
  newgrad=-m.tangent(rn,ng);s=m.tangent(rn,rn-r);y=newgrad-m.tangent(rn,grad);ss=[m.tangent(rn,v) for v in ss];ys=[m.tangent(rn,v) for v in ys]
  if float((s*y).sum())>1e-12*float(s.norm()*y.norm()):ss.append(s);ys.append(y);ss=ss[-10:];ys=ys[-10:]
  r=rn;value=nv;grad=newgrad;parts=nparts
 meta=dict(stop=stop,steps=last,gradient=float(grad.norm()),objective=value,components=parts,history=hist,calibration=obj.calibration,seconds=time.time()-start,orthogonality=float((r.T@r-torch.eye(256,device='cuda')).abs().max()),prior_steps=500)
 np.savez_compressed(path,rotation=r.cpu().numpy());save(path.with_suffix('.json'),meta)
 if callback:callback(r,name,seed)
 return r

def training_labels(e,indices):
 idx=Path(e.cfg['source_run'])/'index/train2017';mem=np.load(PAIR/'train-membership.npz');rr=mem['caption_rows'][indices];par=np.load(idx/'parents.npy');ids=np.array(json.loads((idx/'concept_ids.json').read_text()));old=json.loads((OLDOUT/'concepts.json').read_text());cols=[int(np.flatnonzero(ids==c['id'])[0]) for c in old]
 return {s:a.tensor(np.load(idx/f,mmap_mode='r')[rows][:,cols].astype(float)) for s,f,rows in [('image','presence.npy',par[rr]),('text','mentions.npy',rr)]}
def train_stat(e,obj,labels,r):
 sg=positive_sign({s:e.base[s]@r for s in S});r=r*sg;out={}
 for s,x in zip(S,obj.x):
  au=a.gpu_auc(x@r,labels[s]);out[s]=dict(responsive=float((au>=.8).sum(0).double().mean()),best_auc=float(au.max(0).values.mean()),features90=feature_stats(e.base[s]@r)['n90_mean'])
 return out

def validate_gradients(e,obj):
 saved=(obj.x,obj.n,obj.second);obj.x=[x[:257] for x in obj.x];obj.n=257;obj.second=[x.T@x/257 for x in obj.x];rows=[]
 try:
  r0=initial(7)
  for kind in ['loading','target','score']:
   co=Combined(e,obj,r0,kind,.5);values,grads=co.parts(r0);r=r0.detach().requires_grad_(True);z,w=[x@r for x in obj.x];score=(z.square()*w.square()).mean(0).sum()-(z.square().mean(0)*w.square().mean(0)).sum()
   if kind=='loading':sparseval=sum(((l@r).pow(4).mean(0)-(l@r).square().mean(0).square()).sum()/2 for l in obj.l)
   elif kind=='target':
    zs={s:e.base[s]@r for s in S};ts={s:target(zs[s].detach()[None],'rowfirst',1,'mass90')[0] for s in S};sparseval=-sum((zs[s]-ts[s]).square().sum() for s in S)/2
   else:sparseval=score
   for i,v in enumerate([score,sparseval]):
    g=torch.autograd.grad(v,r,retain_graph=True)[0];err=float((g-grads[i]).norm()/g.norm().clamp_min(1e-30));assert err<1e-9,(kind,i,err);rows.append(dict(kind=kind,component=i,relative_gradient_error=err,value_error=abs(float(v)-float(values[i]))))
 finally:obj.x,obj.n,obj.second=saved
 save(OUT/'gradient-validation.json',rows)

def main():
 protocol=dict(space='fixed CC3M SAE/CC3M PLS256; rotations fitted on same COCO FIT paired cache as previous experiments',stage1='no training; all remaining COCO80 categories with>=20 positives/negatives per modality per split; auxiliary100 caption lemmas',screen='six combinations x3 seeds x500 updates from saved score500,200k same paired rows',selection='training original50 only; minimize geometric mean of responsive-axis-count ratio and coefficient n90 ratio, relative to same-seed saved score500; averaged across seeds/modalities. Report best-axis AUROC separately. No independent-category or test matching selection.',gradient_scaling='orthogonal tangent gradient Frobenius norms at saved score500; fixed per-seed scales and initial objective offsets; sparse weights .25,.5,.75',confirmation='score-only and selected hybrid; 10 seeds;2500 additional updates from saved score500, total3000; saves/evaluates additional0,250,...2500; independent of screening best step',target_objective='negative half squared residual to rowtop1 then column90% target; target depends on R; piecewise analytic gradient only within fixed support. Energy threshold can cause discontinuity and line search stall.',cuts=['dense','column64','global64'],confidence='paired crossed-seed/split bootstrap, fixed evaluation population and category set',selection_supervision='training50 category annotations used for hybrid hyperparameter selection, not rotation gradient')
 save(OUT/'protocol.json',protocol);e=RC_Eval();labels,spec=category_setup(e)
 for seed in range(10):
  for name in ['covariance','cycle-best','score500']:measure(e,labels,spec,name,seed,a.tensor(np.load(ref(name,seed))['rotation']))
 save(OUT/'stage1-complete.json',dict(time=time.time(),groups={g:len(v) for g,v in spec['groups'].items()}))
 data=dict(np.load(PAIR/'directions/pairs.npz'));indices=np.load(PREV/'sample-indices.npz')['confirm'];obj=m.ScoreObjective(e,{s:data[s][indices] for s in S});del data;tl=training_labels(e,indices)
 validate_gradients(e,obj)
 basetrain=[];screen=[]
 for seed in range(3):
  r0=a.tensor(np.load(ref('score500',seed))['rotation']);bs=train_stat(e,obj,tl,r0);basetrain.append(bs)
  for kind in ['loading','target']:
   for weight in [.25,.5,.75]:
    name=f'screen-{kind}-w{int(weight*100)}';co=Combined(e,obj,r0,kind,weight);r=fit(co,r0.clone(),name,seed,500);st=train_stat(e,obj,tl,r);ratio=np.prod([sum(st[s][key] for s in S)/sum(bs[s][key] for s in S) for key in ['responsive','features90']])**.5;screen.append(dict(name=name,seed=seed,kind=kind,weight=weight,training=st,selection_score=float(ratio)));save(OUT/'screen-training.json',dict(baseline=basetrain,screen=screen));log(stage='screen_complete_fit',name=name,seed=seed,selection_score=float(ratio))
 names=sorted(set(q['name'] for q in screen));ranking=sorted([dict(name=n,score=float(np.mean([q['selection_score'] for q in screen if q['name']==n]))) for n in names],key=lambda q:q['score']);winner=next(q for q in screen if q['name']==ranking[0]['name']);save(OUT/'stage2-complete.json',dict(time=time.time(),ranking=ranking,winner=winner))
 del tl;torch.cuda.empty_cache()
 for seed in range(10):
  r0=a.tensor(np.load(ref('score500',seed))['rotation'])
  for name,kind,weight in [('long-score','score',0.),('long-combined',winner['kind'],winner['weight'])]:
   co=Combined(e,obj,r0,kind,weight);fit(co,r0.clone(),name,seed,2500,lambda r,n,s:measure(e,labels,spec,n,s,r))
 save(OUT/'complete.json',dict(time=time.time(),new_fits=38,screen_selection_uses_training50_only=True))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
