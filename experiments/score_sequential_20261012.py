"""Fixed-step shared rotations, coefficient-budget evaluation, and axis diagnostics."""
from experiments.score_validation_20261012 import *
from types import SimpleNamespace
import scipy.sparse as sp
PRIOR=ROOT/'runs/score-validation-2026-10-12'
OUT=ROOT/'runs/score-sequential-2026-10-12'
for sub in ['fits','eval','character','keywords']:(OUT/sub).mkdir(parents=True,exist_ok=True)
STEPS=[1,2,5,10,20,50]
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
def sequential(e,seed,kind):
 r=a.tensor(np.load(ref('score500',seed))['rotation']);history=[]
 for it in range(1,51):
  z={s:e.base[s]@r for s in S}
  t={s:hard(z[s],64) if kind=='column64' else target(z[s][None],'rowfirst',1,'mass90')[0] for s in S}
  before=sum((z[s]-t[s]).square().sum() for s in S)
  rn=a.polar(sum(e.base[s].T@t[s] for s in S));after=sum((e.base[s]@rn-t[s]).square().sum() for s in S)
  assert float(after-before)<1e-8,(seed,kind,it,float(after-before))
  history.append(dict(step=it,fixed_target_before=float(before),fixed_target_after=float(after),rotation_change=float((rn-r).norm())))
  r=rn
  if it in STEPS:
   path=OUT/'fits'/f'seq-{kind}-i{it}-s{seed}.npz'
   np.savez_compressed(path,rotation=r.cpu().numpy())
   save(path.with_suffix('.json'),dict(seed=seed,kind=kind,steps=it,history=history,orthogonality=float((r.T@r-torch.eye(256,device='cuda')).abs().max())))
 return history

def prune(w,cut):
 if cut=='dense':return w
 k=int(cut.removeprefix('column').removeprefix('global'))
 if cut.startswith('column'):return hard(w,k)
 ix=w.abs().flatten().argsort(descending=True,stable=True)[:256*k]
 return torch.zeros_like(w).flatten().scatter_(0,ix,w.flatten()[ix]).reshape_as(w)

def evaluate(e,labels,spec,name,seed,r,cuts):
 r=r*positive_sign({s:e.base[s]@r for s in S});dense={s:e.base[s]@r for s in S};fs={s:feature_stats(w) for s,w in dense.items()}
 for cut in cuts:
  dest=OUT/'eval'/f'{name}-{cut}-s{seed}.json'
  if dest.exists():continue
  ws={s:prune(w,cut) for s,w in dense.items()};sc={s:e.x[s]@ws[s] for s in S};rows=[]
  for pair0 in e.splits[:20]:
   au=[a.gpu_auc(sc[s][ix],labels[s][ix]) for s,ix in zip(S,pair0)];valid=torch.isfinite(au[0])&torch.isfinite(au[1]);has=valid.any(0)
   for q in au:q[~valid]=-torch.inf
   ii=[q.argmax(0) for q in au];col=torch.arange(70,device='cuda');best=[q[k,col] for q,k in zip(au,ii)];match=(ii[0]==ii[1])&has
   row=dict(match=match.int().cpu().tolist(),best_auc={s:b.cpu().tolist() for s,b in zip(S,best)},responsive={s:{str(t):(q>=t).sum(0).cpu().tolist() for t in [.8,.9,.95]} for s,q in zip(S,au)})
   for t in [.7,.8,.9]:row['match'+str(t)]=(match&(best[0]>=t)&(best[1]>=t)).int().cpu().tolist()
   rows.append(row)
  ret=retrieval(sc,e.parents)
  save(dest,dict(name=name,seed=seed,cut=cut,rows=rows,retrieval=ret,coefficients={s:feature_stats(ws[s]) for s in S},dense_coefficients=fs))
  log(stage='evaluation',name=name,seed=seed,cut=cut)

def word_cache(e):
 dest=OUT/'word-incidence.npz'
 if dest.exists():return sp.load_npz(dest),json.loads((OUT/'words.json').read_text())
 idx=Path(e.cfg['source_run'])/'index/val2017';ids=np.array([v['image_id'] for v in json.loads((idx/'images.json').read_text())]);keep=np.isin(ids,e.data['metadata']['half_a_image_ids']+e.data['metadata']['half_b_image_ids']);par=np.load(idx/'parents.npy');rr=np.flatnonzero(keep[par]);caps=json.loads((idx/'captions.json').read_text())
 vocab=json.loads((PAIR/'vocabulary.json').read_text());words=[v['word'] for v in vocab['ranked_words'] if v['captions']>=100];lookup={w:j for j,w in enumerate(words)};rows=[];cols=[]
 nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])
 for i,doc in enumerate(nlp.pipe((caps[int(j)]['text'] for j in rr),batch_size=512)):
  vals={lookup[t.lemma_.lower()] for t in doc if t.lemma_.lower() in lookup and t.pos_ in {'NOUN','PROPN','VERB','ADJ','ADV'} and t.is_alpha and not t.is_stop}
  rows.extend([i]*len(vals));cols.extend(vals)
 mat=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(rr),len(words)))
 assert len(rr)==len(e.parents)
 sp.save_npz(dest,mat);save(OUT/'words.json',words);save(OUT/'caption-row-ids.json',rr.tolist());return mat,words

def character(e,labels,spec,name,seed,r,wm,words):
 dest=OUT/'character'/f'{name}-s{seed}.json'
 if dest.exists():return
 r=r*positive_sign({s:e.base[s]@r for s in S});sc={s:e.x[s]@(e.base[s]@r) for s in S};par=torch.as_tensor(e.parents,device='cuda');rows=[]
 for split,pair0 in enumerate(e.splits[:20]):
  # Pair statistics use BOTH modalities of the same image half, not disjoint matching halves.
  mask=torch.zeros(len(sc['image']),device='cuda',dtype=torch.bool);mask[pair0[0]]=True;ix=torch.nonzero(mask[par]).flatten();z=sc['image'][par[ix]];w=sc['text'][ix];zc=z-z.mean(0);wc=w-w.mean(0)
  corr=(zc*wc).sum(0)/(zc.square().sum(0)*wc.square().sum(0)).sqrt().clamp_min(1e-30);k=int(np.ceil(.01*len(ix)))
  iz=z.argsort(dim=0,descending=True,stable=True)[:k];iw=w.argsort(dim=0,descending=True,stable=True)[:k];topz=torch.zeros_like(z,dtype=torch.bool).scatter_(0,iz,True);topw=torch.zeros_like(w,dtype=torch.bool).scatter_(0,iw,True);over=(topz&topw).sum(0).double()/k
  counts={}
  for s,j in zip(S,pair0):
   au=a.gpu_auc(sc[s][j],labels[s][j]);n=(au>=.8).sum(1);counts[s]=dict(per_axis=n.cpu().tolist(),any=int((n>=1).sum()),multiple=int((n>=2).sum()),exactly_one=int((n==1).sum()))
  rows.append(dict(split=split,pairs=len(ix),top_count=k,correlation=corr.cpu().tolist(),overlap=over.cpu().tolist(),category_counts=counts))
 save(dest,dict(name=name,seed=seed,cut='dense',rows=rows))
 # All ten initializations get a separate random20-axis table. No category-based selection.
 family={'covariance':0,'cycle-best':1,'score500':2,'sequential':3}[name];axes=np.random.default_rng(1212700+100*family+seed).choice(256,20,replace=False);allcnt=np.asarray(wm.sum(0)).ravel();n=wm.shape[0];k=int(np.ceil(.01*n));out=[]
 for axis in axes:
  item=dict(axis=int(axis),words={})
  for s in S:
   score=sc[s][par,axis] if s=='image' else sc[s][:,axis];ix=score.argsort(descending=True,stable=True)[:k].cpu().numpy();cnt=np.asarray(wm[ix].sum(0)).ravel();other=allcnt-cnt
   odds=np.log((cnt+.5)/(k-cnt+.5))-np.log((other+.5)/(n-k-other+.5));odds[cnt<3]=-np.inf;top=np.argsort(-odds,kind='stable')[:10]
   item['words'][s]=[dict(word=words[j],log_odds=float(odds[j]),top_captions=int(cnt[j]),all_captions=int(allcnt[j])) for j in top if np.isfinite(odds[j])]
  out.append(item)
 save(OUT/'keywords'/f'{name}-s{seed}.json',dict(name=name,seed=seed,top_count=k,rows=out))
 log(stage='axis_character',name=name,seed=seed)

def main():
 save(OUT/'protocol.json',dict(space='CC3M SAE and CC3M PLS256; saved score500 fitted on COCO FIT paired cache',rotation='shared orthogonal; exact fixed number of updates, no early stopping and no best-iterate replacement',steps=STEPS,selection='training old50 only, fixed200k paired rows; geometric mean of pooled I/T responsive-axis ratio and pooled I/T coefficient n90 ratio relative to same-seed score500; average10 seeds, tie lexicographic. Test retrieval and matching excluded.',groups='frozen previous old50 and independent20 COCO categories, no new eligibility selection',budget=[16,32,64,128],cuts='per-column top k and global top256*k independently in each modality',axis_character='dense coefficients; correlation and intersection/top_count of top1% on paired captions in image half of each split, stable caption-index ties; AUROC uses original disjoint modality halves',keywords='20 uniformly sampled axes per rotation per seed, seed fixed in code; top1% full paired evaluation captions per side; training document frequency>=100 content lemmas; Jeffreys-smoothed top-vs-rest log odds, >=3 top-caption occurrences',confidence='crossed initialization and split bootstrap, fixed categories and evaluation population'))
 e=RC_Eval();oldspec=json.loads((PRIOR/'categories.json').read_text());ix=oldspec['groups']['old50']+oldspec['groups']['heldout_coco'];assert len(ix)==70
 f=np.load(PRIOR/'labels.npz');labels={s:a.tensor(f[s][:,ix]) for s in S};spec=dict(concepts=[oldspec['concepts'][j] for j in ix],groups=dict(old50=list(range(50)),heldout_coco=list(range(50,70))));save(OUT/'categories.json',spec)
 for seed in range(10):
  for kind in ['column64','row90']:
   if not (OUT/'fits'/f'seq-{kind}-i50-s{seed}.npz').exists():sequential(e,seed,kind)
 log(stage='fixed_steps_saved')
 selection=OUT/'selection.json'
 if not selection.exists():
  data=np.load(PAIR/'directions/pairs.npz');indices=np.load(PREV/'sample-indices.npz')['confirm'];obj=SimpleNamespace(x=[a.tensor(data[s][indices]) for s in S]);tl=training_labels(e,indices);del data;stats=json.loads((OUT/'training.json').read_text()) if (OUT/'training.json').exists() else []
  for seed in range(10):
   existing=[q for q in stats if q['seed']==seed]
   bs=existing[0]['baseline'] if existing else train_stat(e,obj,tl,a.tensor(np.load(ref('score500',seed))['rotation']))
   for kind in ['column64','row90']:
    for it in STEPS:
     name=f'seq-{kind}-i{it}'
     if any(q['name']==name and q['seed']==seed for q in stats):continue
     path=OUT/'fits'/f'{name}-s{seed}.npz';st=train_stat(e,obj,tl,a.tensor(np.load(path)['rotation']));ratio=np.prod([sum(st[s][key] for s in S)/sum(bs[s][key] for s in S) for key in ['responsive','features90']])**.5
     stats.append(dict(name=name,seed=seed,kind=kind,step=it,training=st,baseline=bs,selection_score=float(ratio)));save(OUT/'training.json',stats)
   log(stage='training_selection',seed=seed)
  names=sorted(set(z['name'] for z in stats));ranking=sorted([dict(name=n,score=float(np.mean([z['selection_score'] for z in stats if z['name']==n]))) for n in names],key=lambda z:(z['score'],z['name']));save(selection,dict(ranking=ranking,winner=ranking[0]['name'],time=time.time()));del obj,tl;torch.cuda.empty_cache()
 selected=json.loads(selection.read_text())['winner'];log(stage='selected_without_test',name=selected)
 # Evaluate curves before evaluating budget alternatives. Saved files permit resuming.
 for seed in range(10):
  for kind in ['column64','row90']:
   for it in STEPS:
    name=f'seq-{kind}-i{it}';evaluate(e,labels,spec,name,seed,a.tensor(np.load(OUT/'fits'/f'{name}-s{seed}.npz')['rotation']),['dense','column64'])
 save(OUT/'stage1-complete.json',dict(time=time.time(),selected=selected))
 for seed in range(10):
  for name in ['covariance','cycle-best','score500','sequential']:
   path=OUT/'fits'/f'{selected}-s{seed}.npz' if name=='sequential' else ref(name,seed);r=a.tensor(np.load(path)['rotation']);evaluate(e,labels,spec,name,seed,r,['dense']+[f'{rule}{k}' for rule in ['column','global'] for k in [16,32,64,128]])
 save(OUT/'stage2-complete.json',dict(time=time.time()))
 wm,words=word_cache(e)
 for seed in range(10):
  for name in ['covariance','cycle-best','score500','sequential']:
   path=OUT/'fits'/f'{selected}-s{seed}.npz' if name=='sequential' else ref(name,seed);character(e,labels,spec,name,seed,a.tensor(np.load(path)['rotation']),wm,words)
 save(OUT/'complete.json',dict(time=time.time(),eval_files=len(list((OUT/'eval').glob('*.json'))),character_files=len(list((OUT/'character').glob('*.json'))),keyword_files=len(list((OUT/'keywords').glob('*.json'))),selected=selected))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
