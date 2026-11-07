"""CC3M-only score rotation; all50-category representative occupancy evaluation."""
from experiments.score_validation_20261012 import *
from scipy import sparse
import hashlib
OUT=ROOT/'runs/cc3m-score-rotation-2026-10-12';m.OUT=OUT
for sub in ['fits','eval','checkpoints']:(OUT/sub).mkdir(parents=True,exist_ok=True)
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
def prepare(e):
 path=OUT/'cc3m-pairs.npz'
 if path.exists():return dict(np.load(path))
 rows=np.load(ROOT/'runs/cc3m-caption-concepts-2026-10-09/extraction/activation-rows.npy');chosen=np.random.default_rng(1212200000).choice(len(rows),200000,replace=False);selected=rows[chosen]
 z=np.load(Path(e.cfg['parent_run'])/'moments.npz');ni=len(z['image_ids']);mu=z['fit_mean'];sd=np.sqrt(np.maximum(z['fit_second'].diagonal()-mu**2,0));sd[sd==0]=1
 assert int(z['fit_n'])==len(rows)
 data={};audit={}
 for s,sl in [('image',slice(0,ni)),('text',slice(ni,None))]:
  src=ROOT/'runs/cc3m-sae-2026-10-04/activations/train2017'/f'{s}.npz';raw=sparse.load_npz(src).tocsr();w=e.base[s].cpu().numpy()/sd[sl,None];data[s]=np.asarray(raw[selected][:,z[s+'_ids']]@w)-mu[sl]@w
  assert np.isfinite(data[s]).all();audit[s]=dict(path=str(src),shape=raw.shape,sha256=hashlib.sha256(src.read_bytes()).hexdigest());del raw
 np.savez_compressed(path,**data);np.savez_compressed(OUT/'sample-indices.npz',fit_positions=chosen,activation_rows=selected)
 save(OUT/'training.json',dict(dataset='CC3M FIT',pairs=200000,available_fit=len(rows),same_pair_rows_both_modalities=True,normalization='original CC3M FIT activation mean and SD, followed by saved PLS256; no score standardization',sources=audit));return data

class Objective(m.ScoreObjective):
 def __init__(self,e,data):super().__init__(e,data);self.cal=None
 def penalties(self,r,which):
  n=self.n;d=r.shape[1];device=r.device;dtype=r.dtype;q=[(r*(s@r)).sum(0) for s in self.second]
  ab=[torch.zeros(d,device=device,dtype=dtype) for _ in S];ags=[torch.zeros_like(r) for _ in S];cross=[torch.zeros_like(r) for _ in range(3)]
  for st in range(0,n,32768):
   xx=[x[st:st+32768] for x in self.x];z=[x@r for x in xx];h=[v.square() for v in z]
   if 1 in which:
    for j in range(2):ab[j]+=z[j].abs().sum(0)/n;ags[j]+=xx[j].T@z[j].sign()/n
   if 2 in which:
    cross[0]+=h[0].T@h[0]/n;cross[1]+=h[1].T@h[1]/n;cross[2]+=h[0].T@h[1]/n
  v1=sum((v/qq.clamp_min(1e-30).sqrt()).sum() for v,qq in zip(ab,q));g1=sum(gg/qq.clamp_min(1e-30).sqrt()-(ss@r)*v/qq.clamp_min(1e-30).pow(1.5) for gg,qq,ss,v in zip(ags,q,self.second,ab))
  v2=torch.zeros((),device=device,dtype=dtype);g2=torch.zeros_like(r)
  if 2 in which:
   cov=[cross[0]-torch.outer(q[0],q[0]),cross[1]-torch.outer(q[1],q[1]),cross[2]-torch.outer(q[0],q[1])];mask=[]
   for c in cov:c.fill_diagonal_(0);v2+=c.clamp_min(0).sum();mask.append((c>0).to(dtype))
   # Within-view ordered j!=k sums; cross-view all j!=k once.
   for st in range(0,n,32768):
    xx=[x[st:st+32768] for x in self.x];z=[x@r for x in xx];h=[v.square() for v in z]
    for j in range(2):
     dh=(h[j]-q[j])@(mask[j]+mask[j].T)
     dh+=(h[1-j]-q[1-j])@(mask[2].T if j==0 else mask[2])
     g2+=xx[j].T@(2*z[j]*dh)/n
  return [v1,v2],[g1,g2]
 def calibrate(self,r,kind,beta):
  self.kind=kind;self.beta=beta;self.which=([1] if kind=='pen1' else [2] if kind=='pen2' else [1,2] if kind=='both' else [])
  vals,gs=self.components(r,False);pv,pg=self.penalties(r,self.which) if self.which else ([0.,0.],[torch.zeros_like(r)]*2)
  base=float(m.tangent(r,gs[1]).norm());pn=[float(m.tangent(r,g).norm()) for g in pg];self.weights=[base/max(v,1e-30) if i+1 in self.which else 0 for i,v in enumerate(pn)]
  self.cal=dict(base_tangent_norm=base,penalty_tangent_norms=pn,weights=self.weights,beta=beta,combined='Each included penalty separately scaled to base tangent norm; beta multiplies each included penalty. Frozen after initialization.',within_penalty='ordered unequal axis pairs in each modality; cross-view unequal axis pairs once')
 def evaluate(self,r,kind,norm):
  if kind in ['var','cross','sum','mixed']:return super().evaluate(r,kind,False)
  vals,gs=self.components(r,False);pv,pg=self.penalties(r,self.which);scale=self.scales[('cross',False)]
  value=vals[1];g=gs[1]
  for j in range(2):value=value-self.beta*self.weights[j]*pv[j];g=g-self.beta*self.weights[j]*pg[j]
  return float(value/scale),g/scale,list(map(float,vals+pv))

def validate(e,data):
 obj=Objective(e,{s:data[s][:101] for s in S});r=initial(701).detach().requires_grad_(True);z=[x@r for x in obj.x];h=[v.square() for v in z];v1=sum((v.abs().mean(0)/v.square().mean(0).sqrt()).sum() for v in z);v2=0
 for j,k in [(0,0),(1,1),(0,1)]:
  c=h[j].T@h[k]/len(h[j])-torch.outer(h[j].mean(0),h[k].mean(0));mask=1-torch.eye(256,device='cuda');v2+=(c*mask).clamp_min(0).sum()
 pv,pg=obj.penalties(r.detach(),[1,2]);rep=[]
 for i,v in enumerate([v1,v2]):
  g=torch.autograd.grad(v,r,retain_graph=True)[0];err=float((g-pg[i]).norm()/g.norm());assert err<1e-9,err;rep.append(dict(penalty=i+1,gradient_error=err,value_difference=abs(float(v-pv[i]))))
 save(OUT/'gradient-validation.json',rep)

def measure_new(e,name,seed,r,meta):
 dense={s:e.base[s]@r for s in S}
 for cut in ['dense','column64']:
  dest=OUT/'eval'/f'{name}-{cut}-s{seed}.json'
  if dest.exists():continue
  ws={s:hard(w,64) if cut=='column64' else w for s,w in dense.items()};sc={s:e.x[s]@w for s,w in ws.items()};rows=[]
  for split,ix in enumerate(e.splits[:20]):
   au=[a.gpu_auc(sc[s][i],e.labels[s][i]) for s,i in zip(S,ix)];idx=[];best=[];dup=[]
   for q in au:
    assert bool(torch.isfinite(q).all());i=q.argmax(0);b=q[i,torch.arange(50,device='cuda')];f=torch.bincount(i,minlength=256);idx.append(i);best.append(b);dup.append(f[i]>1)
   match=idx[0]==idx[1];solo=match&~dup[0]&~dup[1]
   rows.append(dict(split=split,match=int(match.sum()),solo=int(solo.sum()),overlap_image=int(dup[0].sum()),overlap_text=int(dup[1].sum()),responsive_image=float((au[0]>=.8).sum(0).double().mean()),responsive_text=float((au[1]>=.8).sum(0).double().mean()),image_axes=idx[0].cpu().tolist(),text_axes=idx[1].cpu().tolist(),image_auc=best[0].cpu().tolist(),text_auc=best[1].cpu().tolist(),matched=match.int().cpu().tolist(),solo_categories=solo.int().cpu().tolist()))
  save(dest,dict(name=name,seed=seed,cut=cut,rows=rows,retrieval=retrieval(sc,e.parents),coefficients={s:feature_stats(w) for s,w in ws.items()},meta=meta));log(stage='evaluation',name=name,seed=seed,cut=cut)

def main():
 e=RC_Eval();data=prepare(e);validate(e,data);obj=Objective(e,data)
 conditions=[(k,0.) for k in ['var','cross','sum','mixed']]+[(k,b) for k in ['pen1','pen2','both'] for b in [.3,1.]]
 save(OUT/'protocol.json',dict(training='CC3M FIT pairs only',pairs=200000,seeds=list(range(5)),iterations=500,score_standardized=False,conditions=conditions,penalty1='sum_j Eabs(z_j)/sqrt(E z_j^2) plus same for text',penalty2='sum ordered offdiagonal positive Cov(z_j^2,z_k^2), within both modalities plus cross-modal',selection='None; report all conditions',matching='50 categories, original20 splits',solo='Matched AND no other category among all50 selects its axis in EITHER modality',overlap='all categories whose representative axis is shared by another category, per modality',orientation='same positive coefficient energy sign for new fits; historical baselines frozen as previously evaluated',confidence='paired crossed initialization/split bootstrap, conditional on fixed category/evaluation population'))
 save(OUT/'categories.json',[c for c in e.data['concepts'] if c['id'] in e.data['metadata']['eligible_category_ids']])
 for seed in range(5):
  for name in ['covariance','cycle-best','score500','oracle']:
   src=ref(name,seed) if name!='oracle' else ROOT/'runs/oracle-direction-rotation-2026-10-12/fits'/f'mean-all50-s{seed}.npz';r=a.tensor(np.load(src)['rotation'])
   if name in ['cycle-best','score500']:r=r*positive_sign({s:e.base[s]@r for s in S})
   measure_new(e,name,seed,r,dict(source=str(src),retrained=False))
 for kind,beta in conditions:
  for seed in range(5):
   name=kind if beta==0 else f'{kind}-b{beta:g}';r=a.tensor(np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation']);obj.calibrate(r,kind,beta)
   save(OUT/'fits'/f'{name}-s{seed}-calibration.json',obj.cal)
   r,meta=m.fit_score(obj,r,kind,False,f'{name}-s{seed}',500);r=r*positive_sign({s:e.base[s]@r for s in S});meta['calibration']=obj.cal
   np.savez_compressed(OUT/'fits'/f'{name}-s{seed}.npz',rotation=r.cpu().numpy());save(OUT/'fits'/f'{name}-s{seed}.json',meta);measure_new(e,name,seed,r,meta)
 save(OUT/'complete.json',dict(fits=50,evaluations=140,time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
