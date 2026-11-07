from experiments.row_rotation_20261012 import *
def validate():
 e=RowEval();eye=torch.eye(256,device='cuda',dtype=torch.float64);rows=[]
 for path in sorted((OUT/'fits').glob('*.npz')):
  r=a.tensor(np.load(path)['rotation']);meta=json.loads(path.with_suffix('.json').read_text());kind=meta['kind'];h=meta['history']
  rr,z,t=signed_projection(e.base,r[None],kind);rn=polar(sum(e.base[s].T[None]@t[s] for s in S))
  rows.append(dict(name=path.stem,orthogonality=float((r.T@r-eye).abs().max()),next_step=float((rn[0]-r).norm()),objective=float(residual(z,t)[0]),stop=meta['stop'],max_objective_increase=max(v['increase'] for v in h)))
 assert len(rows)==100
 assert max(v['orthogonality'] for v in rows)<1e-9
 assert max(v['max_objective_increase'] for v in rows)<1e-8
 # Orthogonal column bases and common rotations preserve all cosine scores.
 baseorth={s:float((e.base[s].T@e.base[s]-eye).abs().max()) for s in S}
 q=a.tensor(np.load(sorted((OUT/'fits').glob('*.npz'))[0])['rotation']);scores={s:e.x[s][:100]@e.base[s] for s in S}
 norm=lambda v:v/v.norm(dim=1,keepdim=True)
 sim0=norm(scores['image'])@norm(scores['text']).T;sim1=norm(scores['image']@q)@norm(scores['text']@q).T
 save(OUT/'validation.json',dict(fits=rows,base_orthogonality=baseorth,cosine_max_error=float((sim0-sim1).abs().max()),evaluation_files=len(list((OUT/'eval').glob('*.json')))))
if __name__=='__main__':validate()
