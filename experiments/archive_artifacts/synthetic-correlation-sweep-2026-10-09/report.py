from pathlib import Path
import json,html
import numpy as np
from scipy.optimize import minimize,linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent;data=root/'results'
methods=['nce','nce_reconstruction','nce_reconstruction_activity','fastica']
labels=['Sparse nonnegative InfoNCE','+ Reconstruction','+ Reconstruction + activity sparsity','FastICA (dense)']
ko=['희소·비음수 InfoNCE','희소·비음수 InfoNCE + 복원','희소·비음수 InfoNCE + 복원 + 활성 희소성','FastICA, 입력 수 제한 없음']
rows=[];checked=0;convergence=[];bound=[]
for p in sorted(data.glob('*-seed*-rho*')):
 if not (p/'complete.json').exists():continue
 kind=p.name.split('-seed')[0];seed=int(p.name.split('-seed')[1][0]);rho=float(p.name.split('-rho')[1]);g=np.load(p/'generator.npz')
 initials=[np.load(p/f'{m}-initial.npz') for m in methods[:3]]
 for key in initials[0].files:
  for init in initials[1:]:np.testing.assert_array_equal(initials[0][key],init[key])
 # Recreate the held-out iid observations to inspect actual output sparsity.
 rng=np.random.default_rng(5000+seed);n=2048
 c1=rng.random(n)<.3;independent=rng.random(n)<.3;copy=rng.random(n)<rho
 c2=np.where(copy,c1,independent);other=(rng.random((n,6))<.35)*rng.gamma(4,.25,(n,6))
 z=np.column_stack([c1,c2,other]).astype('float32');obs={}
 for side in ['image','text']:
  noise=(rng.random((n,128))<.05)*rng.gamma(2,.025,(n,128))
  obs[side]=(np.einsum('ij,jk->ik',z,g[side])+noise).astype('float32')
 for m in methods:
  if not (p/f'{m}-evaluation.json').exists():continue
  e=json.loads((p/f'{m}-evaluation.json').read_text());w=np.load(p/f'{m}.npz');resp=[]
  for side in ['image','text']:
   a=w[side];assert np.isfinite(a).all()
   if m!='fastica':
    assert (a>=0).all();assert (np.count_nonzero(a,axis=0)<=16).all()
    np.testing.assert_allclose(np.linalg.norm(a,axis=0),1,atol=1e-5)
    dec=np.load(p/f'{m}-decoder.npz')[side];assert (dec>=0).all();assert (np.linalg.norm(dec,axis=1)<=1.00001).all()
   r=np.einsum('ij,jk->ik',g[side].astype(float)/g['scale_'+side],a)
   np.testing.assert_allclose(r,e['selectivity']['response_'+side],atol=1e-5,rtol=1e-4);resp.append(r)
  ri,rt=resp;sc=np.minimum(ri**2/np.maximum((ri**2).sum(0),1e-15),rt**2/np.maximum((rt**2).sum(0),1e-15))*(ri*rt>0)
  a,b=linear_sum_assignment(-sc[:2]);assert abs(sc[a,b].mean()-e['selectivity']['target_mean'])<1e-4
  metrics={'selectivity':e['selectivity']['target_mean'],'exclusive_auc':e['exclusive_category_auc']['recombined'],'all_factor_selectivity':e['selectivity']['all_factor_mean']}
  for split in ['iid','recombined']:
   for direction in ['image_to_text','text_to_image']:
    v=e['retrieval'][split][direction]
    for k in [1,5,10]:
     recall=v['recall'][str(k)];assert abs(np.mean(np.array(v['ranks'])<=k)-recall)<1e-10
     metrics[f'{split}_{direction}_R{k}']=100*recall
  effective=[];testrec=[]
  for side in ['image','text']:
   inp=obs[side]/g['scale_'+side]
   if bool(w['centered']):inp=(obs[side]-g['mean_'+side])/g['scale_'+side]
   u=np.einsum('ij,jk->ik',inp,w[side])
   effective.append(float(np.mean(np.abs(u).sum(1)**2/np.maximum((u*u).sum(1),1e-12))))
   if m!='fastica':
    dec=np.load(p/f'{m}-decoder.npz')[side];pred=np.einsum('ij,jk->ik',u,dec)
    testrec.append(float(np.mean((pred-inp)**2)/np.mean(inp**2)))
  metrics['effective_output_count']=float(np.mean(effective))
  if testrec:metrics['test_relative_mse']=float(np.mean(testrec))
  fit=json.loads((p/f'{m}-fit.json').read_text())
  if m!='fastica':
   best=fit['history'][fit['best_epoch']]
   metrics.update(tune_nce=best['nce'],tune_relative_mse=best['relative_mse'],tune_activity=best['mean_activity'],best_epoch=fit['best_epoch'])
   convergence.append(dict(condition=p.name,method=m,best_epoch=fit['best_epoch'],last_epoch=fit['history'][-1]['epoch']))
  else:convergence.append(dict(condition=p.name,method=m,image=fit['image'],text=fit['text']))
  rows.append(dict(kind=kind,seed=seed,rho=rho,method=m,metrics=metrics));checked+=1
 # Relaxed nonnegative selectivity upper bound, unlimited support, separately per target and modality.
 if rho==0:
  target_bounds=[]
  for j in [0,1]:
   sides=[]
   for side in ['image','text']:
    d=g[side].astype(float);ix=d[j]>0;mat=d[np.arange(8)!=j][:,ix]/d[j,ix]
    q=mat.T@mat;n=q.shape[0]
    fit=minimize(lambda a:float(a@q@a),np.ones(n)/n,jac=lambda a:2*q@a,bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(n)},method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    assert fit.success
    grad=2*q@fit.x
    lower_bound=max(0.,fit.fun+grad.min()-grad@fit.x)
    sides.append(1/(1+lower_bound))
   target_bounds.append(min(sides))
  bound.append(dict(kind=kind,seed=seed,relaxed_target_selectivity_bound=float(np.mean(target_bounds))))
summary=[]
for kind in ['anchors','noanchors']:
 for rho in [i/10 for i in range(11)]:
  for m in methods:
   group=[r for r in rows if r['kind']==kind and r['rho']==rho and r['method']==m]
   if len(group)!=3:continue
   summary.append(dict(kind=kind,rho=rho,method=m,metrics={k:dict(mean=float(np.mean([r['metrics'][k] for r in group])),std=float(np.std([r['metrics'][k] for r in group],ddof=1))) for k in group[0]['metrics']}))
(root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'raw-summary.json').write_text(json.dumps(rows,indent=2));(root/'verification.json').write_text(json.dumps(dict(evaluations=checked,convergence=convergence,relaxed_bounds=bound,verified=['identical initial encoder/decoder','support and sign constraints','analytic selectivity','saved-rank recall']),indent=2))

fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
for i,kind in enumerate(['anchors','noanchors']):
 for j,key in enumerate(['iid_image_to_text_R5','iid_text_to_image_R5','selectivity']):
  for m,lab,color in zip(methods,labels,['#658dbc','#d59257','#9478bb','#454c57']):
   rs=sorted([r for r in summary if r['kind']==kind and r['method']==m],key=lambda r:r['rho'])
   xx=[r['rho'] for r in rs];yy=np.array([r['metrics'][key]['mean'] for r in rs]);sd=np.array([r['metrics'][key]['std'] for r in rs])
   axes[i,j].plot(xx,yy,marker='o',color=color,label=lab);axes[i,j].fill_between(xx,yy-sd,yy+sd,color=color,alpha=.12)
  axes[i,j].set(title=('Dedicated inputs' if i==0 else 'All signal inputs mixed')+' / '+['Image query','Text query','Synthetic diagnostic'][j],xlabel='Target-factor correlation',ylabel='Recall@5 (%)' if j<2 else 'Selective response',xticks=np.arange(11)/10)
  axes[i,j].set_ylim((0,100) if j<2 else (0,1.02))
  axes[i,j].spines[['top','right']].set_visible(False);axes[i,j].grid(axis='y',alpha=.2)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
body='<html lang="ko"><meta charset="utf-8"><title>합성 activation의 상관계수 비교</title><style>body{max-width:1450px;margin:40px auto;font:16px/1.7 system-ui}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}b{color:#800022}u{text-underline-offset:3px}</style><h1>합성 activation에서 요인 상관계수를 0.1 간격으로 비교</h1><p>실제 이미지·캡션·임베딩·SAE를 사용하지 않았습니다. 요인 8개의 값을 서로 다른 고정 비음수 행렬로 섞어 이미지·텍스트 역할의 128차원 activation 벡터를 생성했습니다. 관심 요인 두 개는 각각 등장 확률 0.3인 이진 변수이며, 함께 등장하는 정도만 바꿨습니다. 나머지 6개 요인은 등장 확률 0.35이며 등장했을 때의 크기는 감마분포에서 뽑았습니다.</p><p>기존 0.0·0.5·0.9·1.0 결과를 재사용하고 나머지 7개 상관계수를 추가했습니다. 두 생성 구조와 세 번의 반복을 유지했습니다. 곡선은 평균이며 음영은 반복 간 표준편차입니다. 오른쪽 선택적 반응은 생성 정답을 사용하는 합성 진단으로, 실제 범주 매칭 개수가 아닙니다. FastICA에는 입력 16개 및 비음수 가중치 제약이 없습니다.</p><img src="comparison.png"><h2>양방향 검색 Recall(%)</h2>'
for kind in ['anchors','noanchors']:
 for rho in [i/10 for i in range(11)]:
  rs=[r for r in summary if r['kind']==kind and r['rho']==rho]
  if len(rs)!=4:continue
  keys=[f'iid_{d}_R{k}' for d in ['image_to_text','text_to_image'] for k in [1,5,10]]
  body+=f'<h3>{"전용 좌표 있음" if kind=="anchors" else "전용 좌표 없음"} · 요인 상관계수 {rho:.1f}</h3><table><tr><th>방법</th><th>이미지 R@1</th><th>이미지 R@5</th><th>이미지 R@10</th><th>텍스트 R@1</th><th>텍스트 R@5</th><th>텍스트 R@10</th></tr>'
  ranks={k:sorted(set(round(r['metrics'][k]['mean'],2) for r in rs),reverse=True) for k in keys}
  for r in rs:
   body+='<tr><th>'+ko[methods.index(r['method'])]+'</th>'
   for k in keys:
    val=r['metrics'][k]['mean'];text=f'{val:.2f}';rank=ranks[k].index(round(val,2))
    if rank==0:text='<b>'+text+'</b>'
    elif rank==1:text='<u>'+text+'</u>'
    body+='<td>'+text+'</td>'
   body+='</tr>'
  body+='</table>'
body+='</html>';(root/'report.html').write_text(body)
print('evaluations',checked,'summary',len(summary))
