import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,time,traceback,itertools
from pathlib import Path
import numpy as np,torch
from experiments.rotation_mass_audit_20261011 import Evaluator,ROOT,SRC,a,p,prune
OUT=ROOT/'runs/axis-overlap-audit-2026-10-11';OUT.mkdir(exist_ok=True)
OLD=ROOT/'runs/hungarian-rotation-diagnosis-2026-10-11'
S=['image','text'];TH=[.8,.9,.95]
def save(f,z):p.save(f,z)
def corr(x):
 x=x-x.mean(0);x=x/x.norm(dim=0).clamp_min(1e-20);return x.T@x
def run():
 e=Evaluator();concepts=e.data['concepts'];mask=np.isin([c['id'] for c in concepts],e.data['metadata']['eligible_category_ids']);cats=[c for c,b in zip(concepts,mask) if b];save(OUT/'categories.json',cats)
 with np.load(Path(e.cfg['parent_run'])/'moments.npz') as z:ids={s:z[s+'_ids'] for s in S}
 rawauc={};held={};rawcounts={};nuis={};laball={}
 for j,s in enumerate(S):
  ix=e.splits[0][j];allidx=torch.arange(len(e.x[s]),device='cuda');hi=allidx[~torch.isin(allidx,ix)];held[s]=hi
  rawauc[s]=a.gpu_auc(e.x[s][hi],e.labels[s][hi]).cpu().numpy()
  counts=[]
  for pair in e.splits[:20]:
   au=a.gpu_auc(e.x[s][pair[j]],e.labels[s][pair[j]]);counts.append({str(t):(au>=t).sum(0).cpu().tolist() for t in TH})
  rawcounts[s]=counts;laball[s]=np.asarray(e.data['labels'][s],bool)
  # Choose other-category covariates only in representative selection half.
  yl=laball[s][ix.cpu().numpy()].astype(float);std=yl.std(0);zc=(yl-yl.mean(0))/np.maximum(std,1e-8);ph=zc.T@zc/len(yl);nuis[s]={}
  for c,q in enumerate(np.flatnonzero(mask)):
   ph[q,q]=-np.inf;chosen=[int(k) for k in np.argsort(ph[q])[::-1] if ph[q,k]>0][:3];nuis[s][c]=[(k,float(ph[q,k])) for k in chosen]
 save(OUT/'raw-counts.json',rawcounts)
 threshold_summary={}
 for method in ['covariance-svd','correlation-cayley']:
  threshold_summary[method]={s:{str(t):[] for t in TH} for s in S}
  for seed in range(10):
   arr=np.load(OLD/f'{method}-{seed}-auc.npz')['auc']
   for j,s in enumerate(S):
    for t in TH:threshold_summary[method][s][str(t)].append((arr[:,j]>=t).sum(1).tolist())
 save(OUT/'threshold-counts.json',threshold_summary)
 base=e.matrices(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz',256)
 for method in ['covariance-svd','correlation-cayley']:
  rot=a.tensor(np.load(SRC/f'rank256/{method}-0-oriented.npz')['rotation']);ws={s:base[s]@rot for s in S};result={}
  for j,s in enumerate(S):
   w=ws[s];z=e.x[s]@w;cc=corr(z[held[s]]).cpu().numpy();au=np.load(OLD/f'{method}-0-auc.npz')['auc'][0,j];hau=a.gpu_auc(z[held[s]],e.labels[s][held[s]]).cpu().numpy();pw,stats=prune(w,.9);pauc=a.gpu_auc((e.x[s]@pw)[e.splits[0][j]],e.labels[s][e.splits[0][j]]).cpu().numpy()
   wn=w.cpu().numpy();keep=pw.cpu().numpy()!=0;top16=np.argsort(-abs(wn),axis=0,kind='stable')[:16];out=[]
   for c,cat in enumerate(cats):
    axes=np.flatnonzero(au[:,c]>=.8);axes=axes[np.argsort(-au[axes,c],kind='stable')];axisrows=[];pairs=[]
    for k in axes:
     top=np.argsort(-abs(wn[:,k]),kind='stable')[:5];features=[]
     for f in top:
      vals=rawauc[s][f];order=np.argsort(-vals)[:3]
      features.append(dict(sae_id=int(ids[s][f]),coefficient=float(wn[f,k]),target_auc=float(vals[c]) if np.isfinite(vals[c]) else None,associated_categories=[dict(name=cats[l]['name'],auc=float(vals[l])) for l in order if np.isfinite(vals[l])]))
     axisrows.append(dict(axis=int(k),selection_auc=float(au[k,c]),heldout_auc=float(hau[k,c]) if np.isfinite(hau[k,c]) else None,energy90_count=int(stats['counts'][k]),top_features=features))
    for k,l in itertools.combinations(axes,2):
     ks,ls=set(top16[:,k]),set(top16[:,l]);ints=np.sum(keep[:,k]&keep[:,l]);union=np.sum(keep[:,k]|keep[:,l]);bothgood=(rawauc[s][:,c]>=.8)
     pairs.append(dict(axis1=int(k),axis2=int(l),top16_intersection=len(ks&ls),top16_jaccard=len(ks&ls)/len(ks|ls),energy90_jaccard=float(ints/max(union,1)),activation_correlation=float(cc[k,l]),different_good_features_axis1=int(np.sum(keep[:,k]&~keep[:,l]&bothgood)),different_good_features_axis2=int(np.sum(keep[:,l]&~keep[:,k]&bothgood))))
    context=[]
    if len(axes):
     scores=z[held[s]][:,torch.as_tensor(axes,device='cuda')];target=e.labels[s][held[s],c:c+1]
     for nu,phi in nuis[s][c]:
      other=laball[s][held[s].cpu().numpy(),nu];strata=[]
      for val in [0,1]:
       sel=torch.as_tensor(other==val,device='cuda');pos=int(target[sel].sum());neg=int(sel.sum())-pos
       vals=a.gpu_auc(scores[sel],target[sel])[:,0].cpu().numpy() if min(pos,neg)>=20 else None
       strata.append(dict(other_present=val,positive_n=pos,negative_n=neg,axis_aucs=[float(v) if np.isfinite(v) else None for v in vals] if vals is not None else None))
      context.append(dict(other_category=concepts[nu]['name'],selection_phi=phi,strata=strata))
    out.append(dict(category=cat['name'],axis_count=len(axes),axes=axisrows,pairs=pairs,context_controls=context,pruned90_count=int((pauc[:,c]>=.8).sum()),raw_heldout_count=int((rawauc[s][:,c]>=.8).sum())))
    save(OUT/'progress.json',dict(method=method,side=s,category=cat['name']))
   gram=wn.T@wn;result[s]=dict(categories=out,coefficient_orthogonality_max_error=float(np.max(abs(gram-np.eye(256)))),output_abs_offdiag_correlation_mean=float(np.mean(abs(cc[~np.eye(256,dtype=bool)]))),output_abs_offdiag_correlation_max=float(np.max(abs(cc[~np.eye(256,dtype=bool)]))))
  save(OUT/f'{method}.json',result)
 save(OUT/'complete.json',dict(time=time.time(),seed=0,split=0,threshold_repeats='all10seeds x20splits',context='Top3 positive co-occurring annotated object categories chosen in split0 selection half; AUROC conditional on each separately in complementary half; at least20 positives/negatives; no causal guarantee or unannotated scene control'))
if __name__=='__main__':
 try:run()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
