from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parent
D=ROOT/'results'
conditions=[('coco-coco','SAE와 대응을 모두 COCO2017로 학습'),('cc3m-coco','SAE는 CC3M, 대응은 COCO2017로 학습'),('cc3m-cc3m','SAE와 대응을 모두 CC3M으로 학습')]
names={'original_sparse_pls':'기존 Sparse PLS-SVD','pls_hsic_0':'공분산 추가 학습, 독립성 항 없음','pls_hsic_0.1':'공분산 추가 학습, 독립성 항 0.1','pls_hsic_1':'공분산 추가 학습, 독립성 항 1'}
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Sparse PLS-SVD와 독립성 목적 비교</title><style>body{font:16px/1.6 system-ui;margin:36px auto;max-width:1350px;padding:0 24px;color:#20242b}h1{font-size:28px}h2{font-size:22px;margin-top:35px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f0f2f5}b{color:#bc3434}u{text-underline-offset:4px}.note{color:#596271}a{color:#285dae}</style><h1>Sparse PLS-SVD에 독립성 목적을 추가한 결과</h1><p>각 이미지·텍스트 조합 쌍의 공분산을 높이면서, 서로 다른 조합 쌍 사이의 비선형 의존성을 줄였다. 각 열의 activation은 최대 16개이며, 공통 좌표는 256개다. 개념 주석은 학습에 사용하지 않았다.</p><p>기존 Sparse PLS-SVD에서 시작해 고정한 학습 쌍 32,768개로 추가 학습했다. 세 추가 학습 조건은 초기값, 표본, 최적화 방식과 중복 방지 벌점이 같고, 독립성 항의 강도만 다르다. 기존 모델은 전체 학습 자료로 구한 결과다.</p><p>독립성 항은 2차원 조합 쌍의 Gaussian kernel HSIC를 무작위 Fourier 특징으로 근사했다. 매번 서로 다른 조합 쌍 1,024개를 골라 계산한다. 정확한 전체 의존성(TC) 또는 IVA의 우도 최적화 결과는 아니다. 가중치의 부호를 허용하고, 각 열의 길이는 1로 제한했다.</p><p>검색 수치는 Recall@1·5·10의 백분율이다. 매칭은 이미지 주석과 캡션 자체의 범주 언급을 이용해 별도로 고른 대표 좌표 번호가 일치한 범주 수다. 독립된 표본 집단에서 양성이 50개 이상인 42개 범주를 평가했다. 높은 활성만 고르는 평가와 활성 반응 방향을 허용하되 양쪽 방향도 일치해야 하는 평가를 구분했다. 텍스트 주석은 사전 기반 언급이며 사람이 확정한 의미 주석은 아니다.</p><p>각 표에서 최댓값은 <b>굵게</b>, 두 번째 값은 <u>밑줄</u>로 표시했다. 1회 초기화의 탐색 결과이며, 반복 실험에 따른 불확실성은 아직 측정하지 않았다.</p>''']
summary=[]
for cond,title in conditions:
 rows=[]
 for key,label in names.items():
  p=D/cond/(key+'-evaluation.json')
  if not p.exists():continue
  v=json.loads(p.read_text());metrics=[100*v['retrieval'][s]['recall'][str(k)] for s in ['image_to_text','text_to_image'] for k in [1,5,10]]+[v[t]['summary']['agree_at1_count'] for t in ['positive_agreement','signed_agreement']]
  rows.append((key,label,metrics));summary.append(dict(condition=cond,method=key,metrics=metrics))
 parts.append('<h2>'+title+'</h2>')
 if not rows:parts.append('<p>아직 평가가 완료되지 않았다.</p>');continue
 ranks=[sorted({round(row[2][j],2) for row in rows},reverse=True) for j in range(8)]
 parts.append('<table><tr><th>방법</th>'+''.join('<th>'+h+'</th>' for h in ['이미지 질의 R@1','이미지 질의 R@5','이미지 질의 R@10','텍스트 질의 R@1','텍스트 질의 R@5','텍스트 질의 R@10','매칭 /42<br>높은 활성','매칭 /42<br>방향 허용'])+'</tr>')
 for key,label,vals in rows:
  cells=[]
  for j,v in enumerate(vals):
   s=f'{v:.2f}' if j<6 else str(v)
   if round(v,2)==ranks[j][0]:s='<b>'+s+'</b>'
   elif len(ranks[j])>1 and round(v,2)==ranks[j][1]:s='<u>'+s+'</u>'
   cells.append('<td>'+s+'</td>')
  parts.append('<tr><td>'+label+'</td>'+''.join(cells)+'</tr>')
 parts.append('</table>')
 parts.append('<details><summary>학습 목표와 선택한 회차</summary><table><tr><th>독립성 항</th><th>선택 회차</th><th>완료 회차</th><th>별도 조정 자료의 공분산</th><th>별도 조정 자료의 의존성</th></tr>')
 for key,label,_ in rows:
  p=D/cond/(key+'-fit.json')
  if not p.exists():continue
  fit=json.loads(p.read_text());hist=json.loads((D/cond/(key+'-history.json')).read_text());v=hist[fit['best_epoch']]
  parts.append(f'<tr><td>{label}</td><td>{fit["best_epoch"]}</td><td>{fit["completed_epoch"]}</td><td>{v["tune_cov"]:.5f}</td><td>{v["tune_dependence"]:.7f}</td></tr>')
 parts.append('</table><p>각 조건 자신의 목적함수가 별도 조정 자료에서 가장 낮은 회차를 선택했다. 최대 12회 학습하며, 3회 연속 개선이 없으면 중단했다. 시험 자료로 회차를 고르지 않았다.</p></details>')
parts.append('<h2>이번 비교에서 확인한 점</h2><ul><li>독립성 항을 강하게 적용하면 추가 학습 대조군보다 조합 간 의존성 측정값이 세 조건에서 약 76–80% 줄었다. 그러나 검색 성능은 세 조건 모두 낮아졌다.</li><li>높은 활성만 기준으로는 강한 독립성 항이 CC3M SAE의 범주 매칭을 늘렸다. 대응을 COCO2017로 학습한 경우 기존 모델의 17개에서 20개로, CC3M으로 학습한 경우 25개에서 27개로 늘었다. 반응 방향을 허용한 평가는 개선되지 않았다.</li><li>따라서 이번 근사 목적과 제한된 추가 학습에서는 의존성 감소가 검색과 범주 매칭의 일관된 동시 개선으로 이어지지 않았다. 정확한 ICA·IVA 또는 다른 독립성 목적 전체가 효과 없다는 결론은 아니다. 최대 회차에 도달한 조건도 있어 충분히 수렴했다고 주장하지 않는다.</li></ul>')
parts.append('</html>');(ROOT/'report.html').write_text(''.join(parts));(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print('Evaluations:',len(summary))
