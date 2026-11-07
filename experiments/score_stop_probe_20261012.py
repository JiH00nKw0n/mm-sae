"""Read-only directional check at stopped hybrid fits; does not update any fit."""
from experiments.score_validation_20261012 import *
def main():
 e=RC_Eval();data=dict(np.load(PAIR/'directions/pairs.npz'));ids=np.load(PREV/'sample-indices.npz')['confirm'];obj=m.ScoreObjective(e,{s:data[s][ids] for s in S});del data;rows=[]
 for seed in range(10):
  r0=a.tensor(np.load(ref('score500',seed))['rotation']);co=Combined(e,obj,r0,'target',.75);r=a.tensor(np.load(OUT/'fits'/f'long-combined-s{seed}.npz')['rotation']);value,g,parts=co.evaluate(r);d=m.tangent(r,g);slope=float(d.square().sum());z={s:e.base[s]@r for s in S};t={s:target(z[s][None],'rowfirst',1,'mass90')[0] for s in S};case=[]
  for step in [1.,.1,.01,.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]:
   rn=a.polar(r+step*d);v,_,vparts=co.evaluate(rn);zn={s:e.base[s]@rn for s in S};tn={s:target(zn[s][None],'rowfirst',1,'mass90')[0] for s in S};changed={s:int(((t[s]!=0)!=(tn[s]!=0)).sum()) for s in S};case.append(dict(step=step,objective_change=v-value,linear_prediction=step*slope,armijo_accepted=v>=value+1e-4*step*slope,changed_target_support=changed,score_change=vparts[0]-parts[0],negative_residual_change=vparts[1]-parts[1]))
  rows.append(dict(seed=seed,gradient=float(d.norm()),probes=case));save(OUT/'stopping-probe.json',rows)
if __name__=='__main__':main()
