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
 for rho in [0,.5,.9,1]:
  for m in methods:
   group=[r for r in rows if r['kind']==kind and r['rho']==rho and r['method']==m]
   if len(group)!=3:continue
   summary.append(dict(kind=kind,rho=rho,method=m,metrics={k:dict(mean=float(np.mean([r['metrics'][k] for r in group])),std=float(np.std([r['metrics'][k] for r in group],ddof=1))) for k in group[0]['metrics']}))
(root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'raw-summary.json').write_text(json.dumps(rows,indent=2));(root/'verification.json').write_text(json.dumps(dict(evaluations=checked,convergence=convergence,relaxed_bounds=bound,verified=['identical initial encoder/decoder','support and sign constraints','analytic selectivity','saved-rank recall']),indent=2))
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
colors=['#5c86bb','#d89355','#9476ba','#444a53']
for i,kind in enumerate(['anchors','noanchors']):
 for j,key in enumerate(['selectivity','iid_image_to_text_R5']):
  for m,label,color in zip(methods,labels,colors):
   rs=sorted([r for r in summary if r['kind']==kind and r['method']==m],key=lambda r:r['rho'])
   xx=[r['rho'] for r in rs];yy=np.array([r['metrics'][key]['mean'] for r in rs]);sd=np.array([r['metrics'][key]['std'] for r in rs])
   ax[i,j].plot(xx,yy,color=color,marker='o',label=label);ax[i,j].fill_between(xx,yy-sd,yy+sd,color=color,alpha=.1)
  ax[i,j].set(title=('Exclusive inputs available' if i==0 else 'No exclusive inputs')+' / '+('selectivity' if j==0 else 'retrieval'),xlabel='Correlation between target concepts',ylabel='Selective response (1 = target only)' if j==0 else 'Image-query Recall@5 (%)',ylim=(-.02,1.03) if j==0 else (0,100))
  ax[i,j].spines[['top','right']].set_visible(False);ax[i,j].grid(axis='y',alpha=.2);ax[i,j].set_xticks([0,.5,.9,1])
fig.legend(*ax[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
# Training curves, retained for convergence assessment.
fig,axes=plt.subplots(2,3,figsize=(13,6),constrained_layout=True)
for i,kind in enumerate(['anchors','noanchors']):
 for j,m in enumerate(methods[:3]):
  for seed in range(3):
   p=data/f'{kind}-seed{seed}-rho0.9'/f'{m}-fit.json'
   if not p.exists():continue
   h=json.loads(p.read_text())['history'];axes[i,j].plot([v['epoch'] for v in h],[v['total'] for v in h],label=f'Seed {seed}')
  axes[i,j].set(title=kind+' / '+labels[j],xlabel='Epoch',ylabel='Validation training objective');axes[i,j].spines[['top','right']].set_visible(False)
fig.savefig(root/'loss-curves.png',dpi=160);plt.close(fig)
table=[];full=[]
for r in summary:
 v=r['metrics'];lab=ko[methods.index(r['method'])];kind='전용 좌표 있음' if r['kind']=='anchors' else '전용 좌표 없음'
 table.append(f'<tr><td>{kind}</td><td>{r["rho"]}</td><th>{lab}</th>'+''.join(f'<td>{v[k]["mean"]:.3f} ± {v[k]["std"]:.3f}</td>' for k in ['selectivity','exclusive_auc','iid_image_to_text_R5','iid_text_to_image_R5'])+'</tr>')
 for split in ['iid','recombined']:
  full.append(f'<tr><td>{kind}</td><td>{r["rho"]}</td><th>{lab}</th><td>{split}</td>'+''.join(f'<td>{v[f"{split}_{d}_R{k}"]["mean"]:.2f} ± {v[f"{split}_{d}_R{k}"]["std"]:.2f}</td>' for d in ['image_to_text','text_to_image'] for k in [1,5,10])+'</tr>')
oldrows=json.loads((root/'previous-agreement.json').read_text());at=[]
for r in oldrows:
 at.append(f'<tr><td>{r["sae"]}</td><td>{r["mapping"]}</td><th>{html.escape(r["method"])}</th><td>{r["positive"]:.2f} /50</td><td>{r["signed"]:.2f} /50</td></tr>')
findings=(root/'findings.html').read_text() if (root/'findings.html').exists() else ''
body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>복원·희소성·FastICA로 개념 분리 비교</title><style>body{max-width:1250px;margin:45px auto;padding:0 25px;font:17px/1.8 system-ui;color:#25272b}h1{font-size:29px}h2{font-size:23px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{text-align:left}thead{background:#f1f3f5}img{max-width:100%}.scroll{overflow:auto}</style><h1>복원과 활성 희소성이 개념 분리를 개선하는가?</h1>'''+findings+'''
<p>이 보고서의 새 실험은 합성 activation을 사용한다. 실제 SAE의 객체 범주 대응 평가는 마지막에 데이터셋 조건별로 따로 제시한다. 합성 실험에 COCO의 80개 객체 범주를 적용하거나 실제 객체를 찾았다고 해석하지 않는다.</p>
<h2>같은 초기화에서 세 학습 목적을 비교했다</h2><p>희소·비음수 InfoNCE, 여기에 복원 목적을 추가한 조건, 복원과 표본별 개념 활성의 희소성을 함께 추가한 조건을 비교했다. 입력 128개, 출력 조합 8개, 조합당 최대 16개 입력, 음수 가중치 금지를 공통으로 사용했다. 각 조건의 초기 조합 행렬과 복원 행렬, 미니배치 순서가 정확히 같은지 검사했다. 개념 간 대응 P는 항등행렬로 고정했다.</p>
<p>InfoNCE는 정답 쌍이 다른 쌍보다 높은 코사인 유사도를 갖도록 학습한다. 복원 목적은 각 모달리티의 입력을 조합한 값과 비음수 복원 행렬의 곱으로 복원하는 제곱오차다. 각 모달리티의 학습 입력 평균 제곱으로 나누고 두 모달리티에서 평균했다. 활성 희소성 벌점은 조합한 값의 절댓값 평균이다. 전체 목적은 InfoNCE + α × 상대 복원 오차 + β × 평균 활성이다. 세 조건의 (α,β)는 (0,0), (1,0), (1,0.1)이다. 이 계수들을 최적이라고 주장하지 않는다.</p>
<p>조합 행렬의 각 열 길이는 1로 고정하고, 복원 행렬의 각 행 길이는 최대 1로 제한해 크기 조정만으로 희소성 벌점을 회피하지 못하게 했다. 입력은 학습 표준편차로 나누되 평균을 제거하지 않았다. 학습 자료 8,192쌍, 선택 자료 2,048쌍을 사용했고 Adam 학습률 0.003, 배치 512, 100에폭으로 학습했다. 각 조건의 전체 학습 목적이 선택 자료에서 가장 낮은 시점을 선택했다. 조건별 서로 다른 목적을 사용하는 선택 규칙이며 완전 수렴 또는 전역 최적화를 보장하지 않는다.</p>
<h2>개념별 전용 입력이 없는 경우도 평가했다</h2><p>두 관심 요인은 출현 확률 30%인 이진 값이며, 상관계수를 0, 0.5, 0.9, 1로 바꿨다. 나머지 여섯 요인은 희소하게 존재하고 양수 크기가 변한다. 전용 입력이 있는 조건에서는 각 요인만 나타내는 입력이 여덟 개 존재한다. 전용 입력이 없는 조건은 그 입력마다 다른 요인 하나의 양수 기여를 추가했다. 따라서 모든 신호 입력이 두 요인 이상을 포함한다. 3개 난수 조건을 반복했다.</p>
<p><strong>전용 입력이 없는 자료에서 비음수 선형 조합은 다른 요인의 기여를 빼서 제거할 수 없다.</strong> 따라서 선택적 반응 점수 1이 가능한지부터 달라진다. 입력 수 제한을 풀고도 달성할 수 있는 비음수 선택성의 수치적 상한을 별도로 계산했다. 이 생성 조건에서 성능이 낮다는 것만으로 추가 감독 신호가 필요하다고 결론 내리지 않는다. 표현 제약 때문에 생긴 한계일 수도 있다.</p>
<h2>FastICA는 별도의 기본 비교 방법이다</h2><p>각 모달리티의 학습 관측에 대해 상위 8차원 분산 정규화 후 tanh 비선형 함수를 사용하는 대칭 FastICA를 적용했다. 최대 2,000회, 수렴 허용오차 10⁻⁶으로 계산했다. 양쪽 성분의 학습 표본 상관 절댓값으로 헝가리안 대응을 구하고 부호를 맞췄다. 범주 정답은 사용하지 않았다. 독립된 두 FastICA를 적용한 뒤 대응시키는 방법이며 두 모달리티를 공동 최적화하는 Multi-view ICA는 아니다. 입력 수 제한과 비음수 제약은 적용하지 않았으므로 16개 희소 방법과 같은 구조 제약의 비교는 아니다. 48개 모달리티별 실행 중 1개는 2,000회 안에 정한 수렴 기준을 충족하지 않았다. 해당 결과를 포함해 보고했으며 반복별 수렴 정보는 verification.json에 남겼다. 잡음 없는 독립 Laplace 요인 8개를 혼합한 별도 검사에서 각 요인과 복원 성분의 상관이 모두 0.999 이상이었다.</p>
<h2>선택적 반응과 검색 결과</h2><p>선택적 반응 점수는 요인 하나를 바꾼 제곱 반응을 여덟 요인의 제곱 반응 합으로 나눈 값이다. 양쪽 대응 좌표에서 더 낮은 값을 사용하며, 두 관심 요인에 서로 다른 좌표를 배정했다. 1이면 관심 요인 하나에만 반응한다. 개념 이름 없이 학습한 후 생성 정답으로 평가한다. AUROC는 관심 개념 하나만 존재하는 별도 표본에서 대표 좌표가 두 개념을 얼마나 구별하는지 나타낸다. 검색은 별도 2,048쌍에서 수행했다.</p><img src="comparison.png"><div class="scroll"><table><thead><tr><th>생성 조건</th><th>개념 상관</th><th>방법</th><th>선택적 반응</th><th>개념 구별 AUROC</th><th>이미지 질의 R@5 (%)</th><th>텍스트 질의 R@5 (%)</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div><details><summary>같은 분포와 독립 재조합 분포의 양방향 Recall@1·5·10</summary><div class="scroll"><table><thead><tr><th>생성 조건</th><th>개념 상관</th><th>방법</th><th>평가 분포</th><th>이미지 R@1</th><th>R@5</th><th>R@10</th><th>텍스트 R@1</th><th>R@5</th><th>R@10</th></tr></thead><tbody>'''+''.join(full)+'''</tbody></table></div></details><h2>学習曲線</h2><p>개념 상관계수 0.9의 세 반복을 표시했다. 세 방법의 손실에는 서로 다른 항이 들어가므로 곡선 높이를 방법 사이에서 직접 비교하지 않는다.</p><img src="loss-curves.png">
<h2>앞서 측정한 실제 객체 범주 대응 결과를 조건별로 분리했다</h2><p><strong>전체 80개 중 공통 50개 범주를 평가했다.</strong> COCO val2017을 서로 겹치지 않는 두 이미지 집단으로 나누고, 한 집단의 이미지와 다른 집단의 캡션에서 각각 개념 유무 AUROC가 가장 높은 대표를 골랐다. 각각 양성 표본 50개 이상이고 음성 표본도 있는 범주만 포함했다. 두 대표가 학습된 대응에서 같은 쌍이면 일치로 계산했다. 따라서 아래 수치는 개념 순도나 검색 Recall이 아니라 범주별 대표의 일치 개수다. 텍스트에는 연결된 이미지의 COCO-Stuff 주석을 사용했다.</p><p>양의 방향은 값이 높을수록 개념이 있다고 보는 평가다. 부호 허용은 값이 낮을수록 개념이 있다고 보는 경우도 허용하되 양쪽 부호가 일치해야 한다. 두 방식은 대표를 다시 고르므로 부호 허용 결과가 반드시 더 높지는 않다. InfoNCE는 CC3M SAE와 COCO2017 대응 학습 조건에서만 실행했으며, 값은 3회 평균이다. 다른 두 데이터 조건의 InfoNCE 결과는 아직 없다.</p><table><thead><tr><th>SAE 학습 데이터</th><th>대응 학습 데이터</th><th>방법</th><th>양의 방향 일치</th><th>부호 허용 일치</th></tr></thead><tbody>'''+''.join(at)+'''</tbody></table><p>헝가리안의 대표 후보 수는 조건별 425, 2,108, 2,107개이며 Sparse CCA는 256개다. 후보 수가 동일한 비교는 아니다. 중앙 자르기 영역의 COCO-Stuff 주석과 현재 모델을 사용했으므로 원래 리부탈 평가의 정확한 재현과도 구분한다.</p></html>'''
body=body.replace('学習曲線','학습 곡선')
body=body.replace('</html>',(root/'ica-explanation.html').read_text()+'</html>')
(root/'report.html').write_text(body)
print('completed evaluations',checked,'summary rows',len(summary))
print('bounds',bound)
for r in summary:
 if r['rho'] in [0,.9,1]:print(r['kind'],r['rho'],r['method'],{k:round(r['metrics'][k]['mean'],3) for k in ['selectivity','iid_image_to_text_R5']})
