"""Resume existing runs; identical L1 optimization batched across independent seeds."""
from experiments.sparse_improvement_20261011 import *
import signal


def fast_polar(m):
 # Newton-Schulz computes the same polar factor as SVD for matrices near orthogonal.
 # Fall back to SVD outside its certified convergence neighborhood.
 eye=torch.eye(m.shape[-1],device=m.device,dtype=m.dtype)
 gram=m.transpose(-2,-1)@m
 if float((gram-eye).norm(dim=(-2,-1)).max())>=.9:return a.polar(m)
 q=m
 for _ in range(8):
  gram=q.transpose(-2,-1)@q
  if float((gram-eye).abs().max())<5e-15:break
  q=.5*q@(3*eye-gram)
 if float((q.transpose(-2,-1)@q-eye).abs().max())>1e-12:return a.polar(m)
 return q

def batch_l1(base,seeds):
 r=torch.stack([initial(s) for s in seeds]);n=len(seeds);hist=[[] for _ in seeds];eps=1e-6;alpha0=torch.full((n,),.1,device='cuda',dtype=torch.float64);done=torch.zeros(n,device='cuda',dtype=torch.bool);stops=['iteration_limit']*n;start=time.time();lastgrad=[None]*n;steps=[4000]*n
 def obj(rr):return sum(torch.sqrt((base[s][None]@rr).square()+eps**2).sum((1,2)) for s in S)/512
 for step in range(1,4001):
  z={s:base[s][None]@r for s in S};g=sum(base[s].T[None]@(z[s]/torch.sqrt(z[s].square()+eps**2)) for s in S)/512
  om=(r.transpose(1,2)@g-g.transpose(1,2)@r)/2;grad=r@om;gn=grad.norm(dim=(1,2));val=obj(r);alpha=torch.clamp(alpha0*1.4,max=10);accepted=done.clone();chosen=r.clone();chosenval=val.clone()
  if step==1:
   errors=[]
   for j in range(n):
    zs={s:base[s]@r[j] for s in S};gg=sum(base[s].T@(zs[s]/torch.sqrt(zs[s].square()+eps**2)) for s in S)/512
    errors.append(float((g[j]-gg).abs().max()))
   assert max(errors)<1e-12
   candidate_matrix=r-alpha[:,None,None]*grad
   fast=fast_polar(candidate_matrix);svd=a.polar(candidate_matrix);err=float((fast-svd).abs().max());assert err<1e-11
   save(OUT/'batch-l1-validation.json',dict(seeds=seeds,max_gradient_errors=errors,polar_max_difference_from_svd=err,polar_solver='Newton-Schulz within certified neighborhood, SVD fallback'))
  for ls in range(30):
   candidate=fast_polar(r-alpha[:,None,None]*grad);v=obj(candidate);ok=(v<=val-1e-4*alpha*gn.square())&~accepted
   chosen[ok]=candidate[ok];chosenval[ok]=v[ok];accepted|=ok
   if bool(accepted.all()):break
   alpha[~accepted]*=.5
  for j in range(n):
   if bool(done[j]):continue
   lastgrad[j]=float(gn[j])
   if not bool(accepted[j]):done[j]=True;stops[j]='line_search_stall';steps[j]=step;continue
   hist[j].append(float(chosenval[j]))
   if float(gn[j])<1e-6:done[j]=True;stops[j]='gradient';steps[j]=step
  r=chosen;alpha0=alpha
  if step%100==0:print(json.dumps(dict(stage='batched_l1',seeds=seeds,step=step,seconds=time.time()-start)),flush=True)
  if bool(done.all()):break
 for j,seed in enumerate(seeds):
  fit=dict(steps=steps[j],stop=stops[j],seconds=time.time()-start,batched_seeds=seeds,objective=float(obj(r[j:j+1])[0]),gradient=lastgrad[j],epsilon=eps,history=hist[j],objective_definition='Same smoothed coefficient L1 and per-seed Armijo/polar updates; independent seeds batched only for computation')
  dest=OUT/'fits'/f'l1-s{seed}.npz';save(dest.with_suffix('.json'),fit);np.savez_compressed(dest,rotation=r[j].cpu().numpy())

def stop_original_workers():
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit():continue
  try:cmd=(entry/'cmdline').read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError):continue
  if b'-m' in cmd and (b'experiments.sparse_improvement_tail_20261011' in cmd or (b'experiments.sparse_improvement_20261011' in cmd and b'2' in cmd)):
   os.kill(int(entry.name),signal.SIGTERM);print('Stopped superseded worker after saved fits',entry.name,flush=True)
if __name__=='__main__':
 try:
  # Let the two in-flight L1 fits finish; preserve every completed fit before changing orchestration.
  def ready():
   try:
    return all(np.load(OUT/'fits'/f'l1-s{i}.npz')['rotation'].shape==(256,256) for i in [3,9])
   except Exception:return False
  while not ready():time.sleep(5)
  time.sleep(2)
  stop_original_workers();e=Eval()
  for seed in range(10):
   for k in BUDGETS:
    if (OUT/'stage2'/f'alternating-s{seed}-k{k}.json').exists():continue
    r,meta=fit_alternating(e.base,k,seed);r=e.oriented_rotation(r);w={s:e.base[s]@r for s in S};e.evaluate('stage2','alternating',seed,f'k{k}',{s:hard(w[s],k) for s in S},w,meta)
  missing=[s for s in range(10) if not (OUT/'fits'/f'l1-s{s}.npz').exists()]
  if missing:batch_l1(e.base,missing)
  for seed in range(10):
   if all((OUT/'stage2'/f'l1-rotation-s{seed}-k{k}.json').exists() for k in BUDGETS):continue
   r,meta=fit_l1(e.base,seed);r=e.oriented_rotation(r);w={s:e.base[s]@r for s in S}
   for k in BUDGETS:e.evaluate('stage2','l1-rotation',seed,f'k{k}',{s:hard(w[s],k) for s in S},w,meta)
  # GPU sparse CCA uses the same update equations; wait rather than duplicate its fit on CPU.
  while not (OUT/'sparse-cca-k256.npz').exists():
   if (OUT/'error-sparse-gpu.json').exists():raise RuntimeError('GPU Sparse CCA failed; inspect its error before any retry')
   time.sleep(10)
  e.sparse_baselines();save(OUT/'stage2-complete.json',dict(time=time.time(),orchestration='independent-seed batching after preserving completed fits'))
 except Exception as ex:save(OUT/'error-batched.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
