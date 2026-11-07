from experiments.sparse_improvement_20261011 import *
from mm_sae.analysis.sparse_cca import fit_sparse_cca
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 n=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
z=np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cca_3221.npz');start=time.time()
m=fit_sparse_cca(cv[:n,:n],cv[:n,n:],cv[n:,n:],1,k=256,ridge=.01,initial_image=z['image'][:,:1],initial_text=z['text'][:,:1])
np.savez_compressed(OUT/'sparse-cpu-validation-first.npz',image=m.image,text=m.text)
while not (OUT/'sparse-cca-k256.npz').exists():time.sleep(10)
z=np.load(OUT/'sparse-cca-k256.npz');checks={}
for s in S:
 w=getattr(m,s);v=z[s][:,:1];checks[s]=dict(same_support=bool(np.array_equal(w!=0,v!=0)),max_absolute_error=float(np.max(np.abs(w-v))))
 assert checks[s]['same_support'] and checks[s]['max_absolute_error']<1e-8,checks
save(OUT/'sparse-gpu-full-component-validation.json',dict(checks=checks,cpu_metadata=m.metadata,elapsed_including_wait=time.time()-start))
