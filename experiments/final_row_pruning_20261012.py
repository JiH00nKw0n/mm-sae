"""Frozen rotations, final row/column pruning only. No fitting."""
from experiments.row_column_rotation_20261012 import *
import hashlib
OUT=ROOT/'runs/final-row-pruning-2026-10-12'
PREV=ROOT/'runs/feature-redundancy-cycle-2026-10-12'
for sub in ['eval','rotations']:(OUT/sub).mkdir(parents=True,exist_ok=True)

def prune(w,rule):
 if rule=='a':return col_cut(w[None],'k64')[0]
 q=2 if rule=='d' else 1
 ix=torch.argsort(w.abs(),dim=1,descending=True,stable=True)[:,:q]
 z=torch.zeros_like(w).scatter_(1,ix,w.gather(1,ix))
 if rule in ['c','d']:z=col_cut(z[None],'k64')[0]
 if rule=='e':z=truncate(z[None],fraction=.9)[0]
 assert int((z!=0).sum(1).max())<=q
 if rule in ['c','d']:assert int((z!=0).sum(0).max())<=64
 return z

def main():
 import shutil
 shutil.copy2(PREV/'concepts.json',OUT/'concepts.json')
 save(OUT/'protocol.json',dict(rank=256,seeds=10,splits=20,training='CC3M/CC3M',new_training=False,rules=dict(a='column top64',b='row top1',c='row top1 then column top64',d='row top2 then column top64',e='row top1 then column squared-energy90%'),energy_denominator='AFTER row pruning',sign='shared sign from dense coefficients, fixed for all cuts',matching='own-caption labels, same50 categories and20 splits; representative axes restricted to axes finite in both modalities, as legacy; no valid axis means no match',spread='distinct axes with nonzero coefficient for at least one original positive SAE feature with AUROC >= threshold; coefficient sign ignored'))
 e=RC_Eval();raw={s:a.tensor(np.load(PREV/'analysis'/f'raw-{s}.npz')['auc'][:20,e.data['feature_ids'][s],:]) for s in S};started=time.time();completed=0
 for seed in range(10):
  refs={'covariance':SRC/f'rank256/covariance-svd-{seed}-oriented.npz','fixed64':ROOT/'runs/sparse-improvement-2026-10-11/fits'/f'alternating-k64-s{seed}.npz','row1':ROOT/'runs/row-rotation-2026-10-12/fits'/f'row1-covariance-s{seed}.npz','cycle-best':PREV/'fits'/f'cycle-best-s{seed}.npz'}
  for method,path in refs.items():
   r=a.tensor(np.load(path)['rotation']);w={s:e.base[s]@r for s in S};sg=positive_sign(w);w={s:w[s]*sg for s in S};shutil.copy2(path,OUT/'rotations'/f'{method}-s{seed}.npz')
   for rule in 'abcde':
    dest=OUT/'eval'/f'{method}-{rule}-s{seed}.json'
    if dest.exists():completed+=1;continue
    ws={s:prune(w[s],rule) for s in S};sc={s:e.x[s]@ws[s] for s in S};counts={};axisrows={s:[] for s in S};spread={s:{} for s in S};rows=[]
    for s in S:
     support=ws[s]!=0;nn=support.sum(0).cpu().numpy();nr=support.sum(1).cpu().numpy()
     counts[s]=dict(mean=float(nn.mean()),median=float(np.median(nn)),min=int(nn.min()),max=int(nn.max()),quantiles=np.quantile(nn,[.1,.25,.75,.9]).tolist(),per_axis=nn.tolist(),empty_axes=int((nn==0).sum()),unused_input_features=int((nr==0).sum()),unused_full4096_features=int(4096-(nr>0).sum()),input_features=len(nr),per_feature_axes=nr.tolist(),zero_query_vectors=int((sc[s].norm(dim=1)==0).sum()))
     for th in [.7,.8,.9]:spread[s][str(th)]=((support.T.double()[None]@(raw[s]>=th).double())>0).sum(1).cpu().tolist()
    for pair in e.splits[:20]:
     au=[a.gpu_auc(sc[s][ix],e.labels[s][ix]) for s,ix in zip(S,pair)]
     for s,q in zip(S,au):axisrows[s].append({str(t):(q>=t).sum(0).cpu().tolist() for t in [.8,.9,.95]})
     valid=torch.isfinite(au[0])&torch.isfinite(au[1]);has=valid.any(0)
     for q in au:q[~valid]=-torch.inf
     ind=[q.argmax(0) for q in au];col=torch.arange(50,device='cuda');v=[q[k,col] for q,k in zip(au,ind)];match=(ind[0]==ind[1])&has
     row=dict(count=int(match.sum()),per_category=match.int().cpu().tolist())
     for t in [.7,.8,.9]:row[str(t)]=int((match&(v[0]>=t)&(v[1]>=t)).sum())
     rows.append(row)
    ret=retrieval(sc,e.parents)
    if rule=='a':
     oldpath=(PREV/'eval'/f'cycle-best-k64-s{seed}.json') if method=='cycle-best' else ROOT/'runs/row-column-rotation-2026-10-12/eval'/f'{method}-k64-s{seed}.json'
     old=json.loads(oldpath.read_text());assert ret==old['retrieval'] and rows==old['matching'],(method,seed,'baseline mismatch')
    save(dest,dict(method=method,rule=rule,seed=seed,counts=counts,retrieval=ret,matching=rows,axis_counts=axisrows,spread=spread,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    completed+=1;save(OUT/'progress.json',dict(completed=completed,total=200,method=method,rule=rule,seed=seed,elapsed=time.time()-started));print(completed,method,rule,seed,flush=True)
 save(OUT/'complete.json',dict(evaluations=200,seconds=time.time()-started,baseline40_exact=True))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
