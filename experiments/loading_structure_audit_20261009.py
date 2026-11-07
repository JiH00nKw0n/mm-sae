"""CC3M/CC3M loading-group semantics audit; annotations only for held-out diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
import experiments.rotation_convergence_20261009 as conv
from mm_sae.analysis.semantic_evaluation import auc_matrix
p=conv.p;ROOT=p.ROOT;OUT=ROOT/'runs/loading-structure-audit-2026-10-09';SRC=ROOT/'runs/rotation-convergence-2026-10-09';OUT.mkdir(exist_ok=True)
def save(f,x):p.save(f,x)
def state(**kw):print(json.dumps(kw),flush=True);save(OUT/'progress.json',kw)
def pair_counts(g,labels,ng=256,nc=50):
 counts=np.zeros((ng,nc),dtype=np.int64);np.add.at(counts,(g,labels),1);return counts

def grouping_stats(g,cal_labels,test_labels,test_auc):
 counts=pair_counts(g,cal_labels);test=pair_counts(g,test_labels);sizes=counts.sum(1);concept=counts.argmax(1);pred=concept[g]
 agree=float(np.mean(pred==test_labels));au=float(np.mean(test_auc[np.arange(len(g)),pred]))
 # Same-group pair agreement on completely held-out dominant category labels.
 den=np.sum(sizes*(sizes-1));num=np.sum(test*(test-1));glob=np.bincount(test_labels,minlength=50);null=np.sum(glob*(glob-1))/(len(g)*(len(g)-1)) if len(g)>1 else 0
 pooled=float(num/den) if den else None
 major=sizes[g]>=3
 return dict(features=len(g),nonempty_groups=int(np.sum(sizes>0)),groups_with_atleast3=int(np.sum(sizes>=3)),feature_coverage_in_groups_atleast3=float(np.mean(major)),group_size_median=float(np.median(sizes[sizes>0])),heldout_group_label_agreement=agree,heldout_auc_for_group_concept=au,heldout_group_label_agreement_atleast3=float(np.mean(pred[major]==test_labels[major])) if major.any() else None,within_group_pair_category_agreement=pooled,random_grouping_expected_pair_agreement=float(null),pair_agreement_lift=pooled-null if pooled is not None else None,pair_recall=float(num/np.sum(glob*(glob-1))) if np.sum(glob*(glob-1)) else None,pair_f1=float(2*num/(den+np.sum(glob*(glob-1)))) if den+np.sum(glob*(glob-1)) else None)

def geometry(l,lab,rng):
 sam=[];dif=[];co=[]
 for c in range(50):
  ix=np.flatnonzero(lab==c);other=np.flatnonzero(lab!=c)
  if len(ix)<2 or not len(other):continue
  aa=rng.choice(ix,500);bb=rng.choice(ix,500);keep=aa!=bb;aa=aa[keep];bb=bb[keep]
  sam.append(float(np.mean(np.sum(l[aa]*l[bb],1))))
  dd=rng.choice(other,len(aa));dif.append(float(np.mean(np.sum(l[aa]*l[dd],1))))
 return dict(category_count=len(sam),same_category_profile_cosine=float(np.mean(sam)),different_category_profile_cosine=float(np.mean(dif)),category_macro_difference=float(np.mean(np.array(sam)-dif)),per_category_same=sam,per_category_other=dif)

def run():
 state(stage='prepare');cfg,w,ts,orientation=conv.prepare();L={s:ts['load'][i].cpu().numpy() for i,s in enumerate(['image','text'])};del ts;torch.cuda.empty_cache()
 data,_=p.eval_population(cfg,'test',OUT);ids=np.array([c['id'] for c in data['concepts']]);eligible=np.flatnonzero(np.isin(ids,data['metadata']['eligible_category_ids']));assert len(eligible)==50
 names=[data['concepts'][i]['name'] for i in eligible];features={}
 for s in ['image','text']:
  cal=data['masks'][s+'_a'];test=data['masks'][s+'_b'];assert not np.any(cal&test)
  dest=OUT/(s+'-feature-aucs.npz')
  if not dest.exists():
   state(stage='raw_feature_auc',side=s)
   a=auc_matrix(data['raw'][s][cal],data['labels'][s][cal][:,eligible],chunk_size=64);b=auc_matrix(data['raw'][s][test],data['labels'][s][test][:,eligible],chunk_size=64)
   active=np.asarray((data['raw'][s][cal]>0).sum(0)).ravel();np.savez_compressed(dest,cal=a,test=b,active=active)
  z=np.load(dest);a=z['cal'];b=z['test'];label=np.argmax(a,1)
  features[s]=dict(cal=a,test=b,cal_label=label,test_label=b.argmax(1),active=z['active'])
 save(OUT/'geometry.json',{})
 geo={};thresholds=[.60,.65,.70]
 for threshold in thresholds:
  gg={}
  for s,v in features.items():
   keep=(v['cal'].max(1)>=threshold)&(v['active']>=20);ix=np.flatnonzero(keep);lab=v['cal_label'][ix]
   q=geometry(L[s][ix],lab,np.random.default_rng(192));q.update(features=len(ix),represented_categories=int(len(set(lab))),heldout_same_label_fraction=float(np.mean(lab==v['test_label'][ix])),heldout_auc_for_calibration_label=float(v['test'][ix,lab].mean()))
   gg[s]=q
  geo[str(threshold)]=gg
 save(OUT/'geometry.json',geo)
 # Across-view similarity of loading profiles for independently labelled raw features.
 cross=[];rng=np.random.default_rng(917)
 for c in range(50):
  ii=np.flatnonzero((features['image']['cal'].max(1)>=.65)&(features['image']['active']>=20)&(features['image']['cal_label']==c));tt=np.flatnonzero((features['text']['cal'].max(1)>=.65)&(features['text']['active']>=20)&(features['text']['cal_label']==c));oo=np.flatnonzero((features['text']['cal'].max(1)>=.65)&(features['text']['active']>=20)&(features['text']['cal_label']!=c))
  if min(len(ii),len(tt),len(oo))==0:continue
  a=rng.choice(ii,500);b=rng.choice(tt,500);d=rng.choice(oo,500)
  cross.append(dict(concept=names[c],same=float(np.mean(np.sum(L['image'][a]*L['text'][b],1))),different=float(np.mean(np.sum(L['image'][a]*L['text'][d],1)))))
 save(OUT/'cross_view_geometry.json',cross)
 jobs=[('cca_original',np.eye(256))]
 for seed in range(3):
  torch.manual_seed(1200+seed);r=conv.old.polar(torch.randn(256,256,device='cuda')).cpu().numpy().astype(float);a,_,b=np.linalg.svd(r);jobs.append((f'random-{seed}',a@b))
 for m in conv.old.METHODS:
  for seed in range(3):jobs.append((f'{m}-{seed}',np.load(SRC/f'{m}-{seed}-final-evaluated-rotation.npz')['rotation']))
 pairs=json.loads((ROOT/'runs/rotation-selectivity-2026-10-09/pairs.json').read_text())
 # Raw-feature selectivity, independent of rotation; pair set frozen in the previous audit.
 diagnostics={}
 for s,v in features.items():
  cal=data['masks'][s+'_a'];test=data['masks'][s+'_b'];y=data['labels'][s];pp=[]
  for pair in pairs:
   c,d=pair['first_index'],pair['second_index'];lc=int(np.flatnonzero(eligible==c)[0]);ld=int(np.flatnonzero(eligible==d)[0]);mask=test&(y[:,c]!=y[:,d]);yy=y[mask,c]
   ix=np.flatnonzero((v['cal'].max(1)>=.65)&(v['active']>=20)&np.isin(v['cal_label'],[lc,ld]))
   if not len(ix):continue
   a=auc_matrix(data['raw'][s][mask][:,ix],yy[:,None],chunk_size=64)[:,0];direction=np.where(v['cal_label'][ix]==lc,a,1-a)
   pp.append(dict(first=pair['first'],second=pair['second'],lc=lc,ld=ld,features=ix.tolist(),exclusive_auc_for_assigned_concept=direction.tolist(),mean_exclusive_auc=float(np.mean(direction))))
  diagnostics[s]=pp
 save(OUT/'raw_feature_exclusive.json',diagnostics)
 for name,r in jobs:
  path=OUT/(name+'-groups.json')
  if path.exists() and 'pair_recall' in json.loads(path.read_text())['thresholds']['0.65']['image']:continue
  res=dict(name=name,thresholds={},cooccurring_pairs={})
  for threshold in thresholds:
   both=[];res['thresholds'][str(threshold)]={}
   for s,v in features.items():
    ix=np.flatnonzero((v['cal'].max(1)>=threshold)&(v['active']>=20));lr=L[s][ix]@r;g=np.argmax(np.abs(lr),1)
    q=grouping_stats(g,v['cal_label'][ix],v['test_label'][ix],v['test'][ix]);q['top_coordinate_squared_fraction']=float(np.mean(np.max(lr*lr,1)/np.maximum(np.sum(lr*lr,1),1e-20)));res['thresholds'][str(threshold)][s]=q
    both.append((g,v['test_label'][ix]))
   ci=pair_counts(*both[0]);ct=pair_counts(*both[1]);den=float(np.sum(ci.sum(1)*ct.sum(1)));observed=float(np.sum(ci*ct)/den) if den else None;baseline=float(np.sum(ci.sum(0)*ct.sum(0))/(ci.sum()*ct.sum()))
   res['thresholds'][str(threshold)]['cross_modal']=dict(within_group_pair_category_agreement=observed,random_grouping_expected_pair_agreement=baseline,pair_agreement_lift=observed-baseline if observed is not None else None,pair_recall=float(np.sum(ci*ct)/np.sum(ci.sum(0)*ct.sum(0))),pair_f1=float(2*np.sum(ci*ct)/(den+np.sum(ci.sum(0)*ct.sum(0)))))
  for s,pp in diagnostics.items():
   v=features[s];rows=[]
   for pair in pp:
    ix=np.array(pair['features']);g=np.argmax(np.abs(L[s][ix]@r),1);lab=v['cal_label'][ix];a=np.bincount(g[lab==pair['lc']],minlength=256);b=np.bincount(g[lab==pair['ld']],minlength=256)
    if a.sum()==0 or b.sum()==0:continue
    crossp=float(a@b/(a.sum()*b.sum()));within_num=np.sum(a*(a-1))+np.sum(b*(b-1));within_den=a.sum()*(a.sum()-1)+b.sum()*(b.sum()-1)
    la=L[s][ix[lab==pair['lc']]];lb=L[s][ix[lab==pair['ld']]]
    cross_cos=float(la.mean(0)@lb.mean(0));same_den=len(la)*(len(la)-1)+len(lb)*(len(lb)-1);same_cos=float((np.sum(la.sum(0)**2)-np.sum(la**2)+np.sum(lb.sum(0)**2)-np.sum(lb**2))/same_den) if same_den else None
    rows.append(dict(profile_cosine_different=cross_cos,profile_cosine_same=same_cos,first=pair['first'],second=pair['second'],different_concept_group_collision=crossp,same_concept_group_collision=float(within_num/within_den) if within_den else None,raw_feature_exclusive_auc=pair['mean_exclusive_auc']))
   res['cooccurring_pairs'][s]=rows
  save(path,res);state(stage='evaluated',name=name)
 save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',rank=256,labels_used_in_rotation=False,main_threshold=.65,sensitivity_thresholds=thresholds,feature_selection='At least20 positive raw activations in calibration half_a; best positive category AUROC >=threshold. Category labels are operational best response among fixed50 objects, not ground-truth feature semantics.',feature_labels='calibration half_a and test half_b best categories are computed independently from raw SAE activations. Image object presence vs own-caption mentions. No heldout filtering.',profile='Training Cov(standardized SAE activation,CCA output), each row L2 normalized. All shapes same as prior loading_varimax.',group_assignment='Each selected activation assigned to rotation coordinate maximizing absolute loading. Signs ignored for group identity, matching the squared-loading objective. Coordinate dominance is posthoc interpretation, not sparse membership learned by the optimizer.',group_score='Calibration labels choose majority category per group; test AUC for that fixed category and agreement with independently derived test category. Also pair agreement and exact random-grouping expected pair agreement with marginal category counts fixed.',cooccurrence='Previously frozen12 category pairs. Raw features selected/labeled on half_a, exclusive category AUROC on half_b. Group collisions between differently labelled features compared to same-labelled features.',limits='Exploratory, previously used COCO val. Annotation coverage50 objects only; hard best label hides polysemantic features. Covariance-profile cosine is invariant under common orthogonal rotations; that test checks preexisting geometry, not geometry created by rotation. No MLE or recovery theorem established.'))
 save(OUT/'complete.json',dict(complete=True,models=len(jobs)))
if __name__=='__main__':run()
