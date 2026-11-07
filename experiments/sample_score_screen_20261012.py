"""User-reduced, evaluation-selected three-stage experiment. No new space learning."""
import experiments.sample_score_rotation_20261012 as m
from experiments.sample_score_rotation_20261012 import *
PREVIOUS=m.OUT
OUT=ROOT/'runs/sample-score-screen-2026-10-12';m.OUT=OUT
for sub in ['fits','eval','directions','checkpoints']:(OUT/sub).mkdir(parents=True,exist_ok=True)
def scores(q):return np.array([v['per_category'] for v in q['matching']]).sum(1)
def getscore(name,seed):return scores(json.loads((OUT/'eval'/f'{name}-dense-s{seed}.json').read_text())).mean()
def fit_and_eval(e,train,obj,kind,norm,seed,prefix):
 name=f'{prefix}-{kind}-std{int(norm)}-covariance';r0=a.tensor(np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation'])
 r,meta=m.fit_score(obj,r0,kind,norm,name+f'-s{seed}',500);sg=positive_sign({s:e.base[s]@r for s in S});r=r*sg;meta['evaluation_signs']=sg.cpu().tolist()
 np.savez_compressed(OUT/'fits'/f'{name}-s{seed}.npz',rotation=r.cpu().numpy());save(OUT/'fits'/f'{name}-s{seed}.json',meta);m.measure(e,name,seed,r,train,[],list(range(50)),meta)
 return r,name

def main():
 e=RC_Eval();train=dict(np.load(OLDOUT/'directions/train.npz'));data=dict(np.load(PAIR/'directions/pairs.npz'))
 for name in ['train-membership.npz','training.json']:shutil.copy2(PAIR/name,OUT/name)
 rng=np.random.default_rng(121250000);idx=rng.permutation(len(data['image']));np.savez_compressed(OUT/'sample-indices.npz',screen=idx[:50000],confirm=idx[:200000])
 save(OUT/'protocol.json',dict(screen_pairs=50000,screen_seeds=3,screen_iterations=500,screen_starts='existing covariance rotation only',screen_conditions=8,cluster_counts=[16,32,48,64,96],cluster_seeds=3,cluster_rotation_seeds=5,confirmation_pairs=200000,confirmation_seeds=10,confirmation_iterations=500,selection='Choose highest mean dense matching on SAME evaluation splits, only if above paired covariance baseline. Exploratory post-selection, not independent confirmation.',evaluation_splits=20,cuts=['dense','column64'],space='fixed CC3M SAE / CC3M PLS256',cluster_pairs=len(data['image']),normalization='current rotated coordinate training SD, objective only',preserved_full_run=str(PREVIOUS)))
 for seed in range(10):
  for ref,src in [('covariance',SRC/f'rank256/covariance-svd-{seed}-oriented.npz'),('cycle-best',ROOT/'runs/feature-redundancy-cycle-2026-10-12/fits'/f'cycle-best-s{seed}.npz'),('mean-all50',OLDOUT/'fits'/f'mean-all50-s{seed}.npz')]:
   r=a.tensor(np.load(src)['rotation'])
   for cut in ['dense','column64']:
    dest=OUT/'eval'/f'{ref}-{cut}-s{seed}.json'
    if not dest.exists():shutil.copy2(OLDOUT/'eval'/dest.name,dest)
   m.measure(e,ref,seed,r,train,[],list(range(50)),dict(source=str(src)))
 subset={s:data[s][idx[:50000]] for s in S};m.check_gradient(e,subset);obj=m.ScoreObjective(e,subset)
 for kind in ['var','cross','sum','mixed']:
  for norm in [False,True]:
   for seed in range(3):r,name=fit_and_eval(e,train,obj,kind,norm,seed,'screen')
 # full-data pilot and subset: signed/permuted axis agreement and held-out train-score agreement.
 r=a.tensor(np.load(OUT/'fits/screen-var-std0-covariance-s0.npz')['rotation']);rf=a.tensor(np.load(PREVIOUS/'fits/score-var-std0-covariance-s0.npz')['rotation']);sim=(r.T@rf).cpu().numpy();ii,jj=linear_sum_assignment(-np.abs(sim))
 remaining=idx[200000:220000];agreements={}
 for s in S:
  x=a.tensor(data[s][remaining]);u=x@r;v=x@rf;u=(u-u.mean(0))/u.std(0,unbiased=False);v=(v-v.mean(0))/v.std(0,unbiased=False);c=(u.T@v/len(x)).cpu().numpy();agreements[s]=dict(same_axis_mean_abs=float(np.abs(c.diagonal()).mean()),permutation_matched_mean_abs=float(np.abs(c[ii,jj]).mean()))
 save(OUT/'subset-vs-full.json',dict(rotation_same_axis_mean_abs=float(np.abs(sim.diagonal()).mean()),rotation_permutation_matched_mean_abs=float(np.abs(sim[ii,jj]).mean()),relative_frobenius=float((r-rf).norm()/rf.norm()),heldout_training_pairs=len(remaining),score_correlations=agreements,full_iterations=1500,subset_iterations=500,caveat='Both sample size and iteration budget differ. Not a pure sample-size ablation.',subset_matching=getscore('screen-var-std0-covariance',0),full_matching=float(scores(json.loads((PREVIOUS/'eval/score-var-std0-covariance-dense-s0.json').read_text())).mean())))
 ranked=[];base=np.mean([getscore('covariance',s) for s in range(3)])
 for kind in ['var','cross','sum','mixed']:
  for norm in [False,True]:
   name=f'screen-{kind}-std{int(norm)}-covariance';value=float(np.mean([getscore(name,s) for s in range(3)]));ranked.append(dict(kind=kind,standardized=norm,matching=value,difference=value-base))
 ranked.sort(key=lambda x:-x['matching']);winner=ranked[0] if ranked[0]['difference']>0 else None
 save(OUT/'stage1-complete.json',dict(time=time.time(),baseline_matching=base,ranking=ranked,winner=winner,selection_uses_evaluation=True));m.log(stage='stage1_complete',winner=winner)
 del obj,subset;torch.cuda.empty_cache()
 pair.OUT=OUT/'cluster-data';(pair.OUT/'directions').mkdir(parents=True,exist_ok=True)
 for g in range(3):
  for suffix in ['.npz','.json','-memberships.npz']:
   src=PAIR/'directions'/f'cluster64-s{g}{suffix}'
   if src.exists():shutil.copy2(src,pair.OUT/'directions'/src.name)
 for k in [16,32,48,64,96]:
  for g in range(3):
   ds=pair.clusters(data,k,g)
   for seed in range(5):
    r0=np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation'];rr,fit=fit_supervised(ds['image'],ds['text'],list(range(k)),r0)
    for mode in ['covariance','row90']:
     name=f'cluster{k}-g{g}-{mode}';path=OUT/'fits'/f'{name}-s{seed}.npz'
     if path.exists():r=a.tensor(np.load(path)['rotation']);meta=json.loads(path.with_suffix('.json').read_text())
     elif mode=='covariance':
      r,meta=complement_rotate(e,a.tensor(rr),k);meta['direction_fit']=fit;np.savez_compressed(path,rotation=r.cpu().numpy());save(path.with_suffix('.json'),meta)
     else:r,meta=m.coefficient_complement(e,a.tensor(rr),k,name+f'-s{seed}');meta['direction_fit']=fit;save(path.with_suffix('.json'),meta)
     m.measure(e,name,seed,r,train,[],list(range(50)),meta)
 save(OUT/'stage2-complete.json',dict(time=time.time(),fits=150));m.log(stage='stage2_complete')
 if winner:
  obj=m.ScoreObjective(e,{s:data[s][idx[:200000]] for s in S})
  for seed in range(10):fit_and_eval(e,train,obj,winner['kind'],winner['standardized'],seed,'confirm')
  save(OUT/'stage3-complete.json',dict(time=time.time(),winner=winner,pairs=200000,seeds=10,selection_uses_evaluation=True))
 else:save(OUT/'stage3-complete.json',dict(time=time.time(),skipped=True,reason='No screening mean dense matching exceeded existing covariance rotation.'))
 save(OUT/'complete.json',dict(time=time.time(),fits=174+(10 if winner else 0),evaluations=408+(20 if winner else 0)))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
