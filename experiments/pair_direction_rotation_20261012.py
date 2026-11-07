"""Paired-data directions, fixed CC3M PLS256; no COCO category supervision."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import experiments.oracle_direction_rotation_20261012 as oracle
from experiments.oracle_direction_rotation_20261012 import *
import inspect,yaml,spacy
from collections import Counter
OLDOUT=OUT
OUT=ROOT/'runs/pair-direction-rotation-2026-10-12'
for sub in ['fits','eval','directions']:(OUT/sub).mkdir(parents=True,exist_ok=True)
def log(**v):
 save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)

# Reuse the EXACT evaluator, reducing only the requested pruning conditions.
exec(inspect.getsource(oracle.measure).replace("['dense','column64','global64']","['dense','column64']"),globals())

def fit_supervised(di,dt,selected,r0):
 if len(selected)<256:return oracle.fit_supervised(di,dt,selected,r0)
 f=(di[:,selected]+dt[:,selected])/2;u,s,vh=linalg.svd(f,full_matrices=False);r=u@vh;score=float(np.sum(f*r));bound=float(s.sum())
 assert abs(bound-score)<1e-8
 return r,dict(global_gap=bound-score,singular_values=s.tolist(),direction_objective=512-2*score,stop='global_direction_optimum',history=[dict(assignment=list(range(256)),score=score,global_upper_score=bound,gap=bound-score)])

def prepare(e):
 dest=OUT/'directions/pairs.npz'
 source=Path(e.cfg['source_run']);idx=source/'index/train2017'
 parents=np.load(idx/'parents.npy');imids=np.array([v['image_id'] for v in json.loads((idx/'images.json').read_text())]);pop=json.loads((ROOT/'runs/mapping-suite-2026-10-04/population.json').read_text());keep=np.isin(imids,pop['fit_image_ids']);rr=np.flatnonzero(keep[parents]);ir=np.flatnonzero(keep)
 testids=e.data['metadata']['half_a_image_ids']+e.data['metadata']['half_b_image_ids'];assert not set(imids[keep])&set(testids)
 np.savez_compressed(OUT/'train-membership.npz',image_ids=imids[keep],test_image_ids=testids,caption_rows=rr)
 if dest.exists():return dict(np.load(dest)),rr,idx
 data={}
 for side,rows in [('image',ir),('text',rr)]:
  raw=sparse.load_npz(source/'activations/train2017'/f'{side}.npz').tocsr();w=e.base[side].cpu().numpy()/e.data['scales'][side][:,None];shift=e.data['means'][side]@w
  z=np.concatenate([np.asarray(raw[ix][:,e.data['feature_ids'][side]]@w)-shift for ix in [rows[i:i+8192] for i in range(0,len(rows),8192)]])
  if side=='image':lookup=np.full(len(imids),-1);lookup[ir]=np.arange(len(ir));z=z[lookup[parents[rr]]]
  data[side]=z.astype(np.float32);log(stage='project_train',side=side,rows=len(z))
 np.savez(dest,**data);save(OUT/'training.json',dict(images=int(keep.sum()),pairs=len(rr),no_test_overlap=True,source=str(source),image_weighting='once per linked training caption',lexical_rule='spaCy noun proper-noun verb adjective adverb lemmas; alphabetic, not stop word; caption document frequency',space='frozen CC3M SAE / CC3M PLS256',kmeans='concatenate unrotated paired image/text coordinates; Euclidean Lloyd; no extra whitening or per-sample normalization'))
 return data,rr,idx

def lexical(data,rr,idx):
 if (OUT/'directions/words.npz').exists():return dict(np.load(OUT/'directions/words.npz'))
 nlp=spacy.load('en_core_web_sm',disable=['parser','ner']);caps=json.loads((idx/'captions.json').read_text());concepts=json.loads((OLDOUT/'concepts.json').read_text());ids={c['id'] for c in concepts};dictionary=yaml.safe_load((ROOT/'src/mm_sae/resources/concepts.yaml').read_text())['concepts']
 extra_aliases={'person':['human','individual','guy','gal','lady','gentleman','kid','baby','infant','toddler','adult','male','female','dude','fellow'], 'car':['automobile','auto','motorcar','taxi','cab'], 'bicycle':['cycle','pushbike'], 'motorcycle':['motorcycle','scooter'], 'bus':['motorbus','minibus','coach'], 'train':['locomotive','railcar'], 'truck':['lorry','pickup'], 'boat':['ship','vessel','canoe','yacht','sailboat'], 'traffic light':['stoplight','stop light','traffic signal'], 'bird':['avian','pigeon','duck','goose','seagull'], 'cat':['feline','kitty'], 'dog':['canine','pooch','hound'], 'horse':['equine','pony','stallion','mare'], 'cow':['cattle','bovine','calf','bull'], 'backpack':['rucksack','knapsack'], 'umbrella':['parasol'], 'handbag':['purse','pocketbook'], 'sports ball':['football','basketball','volleyball','baseball','soccer ball'], 'tennis racket':['racquet'], 'cup':['mug','teacup'], 'couch':['settee'], 'potted plant':['houseplant','pot plant'], 'toilet':['commode','lavatory','urinal'], 'tv':['telly','television set'], 'laptop':['notebook','notebook computer'], 'remote':['remote control','controller'], 'cell phone':['mobile','mobilephone','telephone','iphone'], 'sink':['washbasin','basin'], 'clock':['timepiece'], 'vase':['flowerpot']}
 excluded=set();aliasrows=[]
 for c in dictionary:
  if c['id'] not in ids:continue
  aliases=list(dict.fromkeys([c['name']]+c.get('aliases',[])+extra_aliases.get(c['name'],[])));lemmas=set()
  for doc in nlp.pipe(aliases):
   for t in doc:
    if t.is_alpha:lemmas.add(t.lemma_.lower());lemmas.add(t.text.lower())
  excluded|=lemmas;aliasrows.append(dict(category=c['name'],aliases=aliases,excluded_tokens=sorted(lemmas)))
 # Conservative exclusion also removes individual words in multiword aliases.
 # This is a declared finite lexicon, not a claim to cover every paraphrase.
 docs=[];freq=Counter()
 for j,doc in enumerate(nlp.pipe((caps[int(i)]['text'] for i in rr),batch_size=512)):
  terms=sorted({t.lemma_.lower() for t in doc if t.pos_ in {'NOUN','PROPN','VERB','ADJ','ADV'} and t.is_alpha and not t.is_stop and len(t.lemma_)>1});docs.append(terms);freq.update(terms)
  if j%25000==0:log(stage='caption_words',processed=j,total=len(rr))
 ranked=sorted(freq,key=lambda w:(-freq[w],w));full=ranked[:256];filtered=[w for w in ranked if w not in excluded][:256];union=sorted(set(full+filtered));lookup={w:i for i,w in enumerate(union)};row=[];col=[]
 for i,terms in enumerate(docs):
  for w in terms:
   if w in lookup:row.append(i);col.append(lookup[w])
 y=sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(len(rr),len(union)));cnt=np.asarray(y.sum(0)).ravel();result={}
 for side in S:
  x=data[side].astype(np.float64);sums=np.asarray(y.T@x).T;delta=sums/cnt-(x.sum(0)[:,None]-sums)/(len(x)-cnt);norm=np.linalg.norm(delta,axis=0);assert np.all(norm>1e-10)
  for name,words in [('words',full),('excluded',filtered)]:result[side+'_'+name]=(delta/norm)[:,[lookup[w] for w in words]]
 audit=dict(model=nlp.meta,ranked_words=[dict(word=w,captions=freq[w],excluded=w in excluded) for w in ranked],selected_words=full,selected_excluded=filtered,excluded_lexicon=sorted(excluded),removed_from_top256=[w for w in full if w in excluded],removed_observed_words=[w for w in ranked if w in excluded],category_aliases=aliasrows,lexicon_revision=2,warning='Conservative finite lexicon includes obvious alternate names and listed subtypes, not only strict synonyms. Finite fixed synonym dictionary; removing all component words of multiword category aliases is conservative. Related contexts remain and this does not prove semantic category information absent.')
 save(OUT/'vocabulary.json',audit);np.savez_compressed(OUT/'directions/words.npz',**result);return result

def clusters(data,k,seed):
 path=OUT/'directions'/f'cluster{k}-s{seed}.npz'
 if path.exists():return dict(np.load(path))
 # FP32 matrix products with TF32 disabled; final statistics in FP64.
 torch.backends.cuda.matmul.allow_tf32=False
 x=torch.as_tensor(np.concatenate([data[s] for s in S],1),device='cuda');rng=np.random.default_rng(7101200+seed);cent=x[torch.as_tensor(rng.choice(len(x),k,replace=False),device='cuda')].clone();history=[];prev=None;stop='iteration_limit'
 for it in range(300):
  sums=torch.zeros_like(cent);counts=torch.zeros(k,device='cuda');assigned=[];inertia=0.;far=[]
  for j in range(0,len(x),8192):
   z=x[j:j+8192];dd=z.square().sum(1)[:,None]+cent.square().sum(1)[None]-2*z@cent.T;v,ix=dd.min(1);assigned.append(ix);inertia+=float(v.sum());sums.index_add_(0,ix,z);counts.index_add_(0,ix,torch.ones_like(ix,dtype=x.dtype));vv,ii=v.topk(min(k,len(v)));far.append((vv,ii+j))
  labels=torch.cat(assigned);new=sums/counts.clamp_min(1)[:,None];empty=counts==0
  if empty.any():
   vals=torch.cat([q[0] for q in far]);inds=torch.cat([q[1] for q in far]);new[empty]=x[inds[vals.topk(int(empty.sum())).indices]]
  shift=float((new-cent).norm()/cent.norm().clamp_min(1e-12));change=None if prev is None else (prev-inertia)/max(abs(prev),1e-12);history.append(dict(iteration=it,inertia=inertia,relative_center_change=shift,relative_inertia_change=change,empty=int(empty.sum())))
  cent=new
  if it%10==0:log(stage='kmeans',k=k,seed=seed,iteration=it,inertia=inertia,shift=shift)
  if shift<1e-5 or (change is not None and 0<=change<1e-6 and it>=10 and not empty.any()):stop='converged';break
  prev=inertia
 # Final memberships must correspond to saved centers.
 lab=[]
 for j in range(0,len(x),8192):
  z=x[j:j+8192];lab.append((z.square().sum(1)[:,None]+cent.square().sum(1)[None]-2*z@cent.T).argmin(1))
 lab=torch.cat(lab).cpu().numpy();cnt=np.bincount(lab,minlength=k);assert np.all((cnt>0)&(cnt<len(x)))
 y=sparse.csr_matrix((np.ones(len(lab)),(np.arange(len(lab)),lab)),shape=(len(lab),k));result={}
 for side in S:
  z=data[side].astype(np.float64);summ=np.asarray(y.T@z).T;delta=summ/cnt-(z.sum(0)[:,None]-summ)/(len(z)-cnt);norm=np.linalg.norm(delta,axis=0);assert np.all(norm>1e-10);result[side]=delta/norm
 np.savez_compressed(path,**result);np.savez_compressed(path.with_name(path.stem+'-memberships.npz'),labels=lab,centers=cent.cpu().numpy());save(path.with_suffix('.json'),dict(seed=seed,k=k,stop=stop,history=history,counts=cnt.tolist()));del x,cent;torch.cuda.empty_cache();return result

def main():
 e=RC_Eval();train=dict(np.load(OLDOUT/'directions/train.npz'));data,rr,idx=prepare(e);lex=lexical(data,rr,idx)
 save(OUT/'protocol.json',dict(seeds=10,splits=20,conditions=['caption content lemmas','same excluding category aliases','kmeans paired concatenation'],counts=[64,128,256],cuts=['dense','column64'],references=['covariance','cycle-best','mean-all50'],labels_used_for_rotation=False,excluded_dictionary_only=True))
 for seed in range(10):
  r0=np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation']
  for kind in ['words','excluded','cluster']:
   for k in [64,128,256]:
    name=f'{kind}{k}';path=OUT/'fits'/f'{name}-s{seed}.npz'
    if path.exists():r=a.tensor(np.load(path)['rotation']);meta=json.loads(path.with_suffix('.json').read_text())
    else:
     ds=clusters(data,k,seed) if kind=='cluster' else {s:lex[s+'_'+kind][:,:k] for s in S};start=time.time();rrr,fit=fit_supervised(ds['image'],ds['text'],list(range(k)),r0)
     if k<256:r,comp=complement_rotate(e,a.tensor(rrr),k)
     else:r=a.tensor(rrr);comp=dict(stop='no_complement',steps=0,orthogonality=float((r.T@r-torch.eye(256,device='cuda')).abs().max()))
     meta=dict(fit=fit,complement=comp,kind=kind,k=k,seed=seed,seconds=time.time()-start,direction_cosine=np.sum(ds['image']*ds['text'],0).tolist());np.savez_compressed(path,rotation=r.cpu().numpy());save(path.with_suffix('.json'),meta);log(stage='fit',name=name,seed=seed,seconds=time.time()-start)
    measure(e,name,seed,r,train,[],list(range(50)),meta)
 save(OUT/'complete.json',dict(fits=90,evaluations=180,time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
