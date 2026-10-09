from pathlib import Path
import json,html,statistics
ROOT=Path(__file__).resolve().parent;D=ROOT/'results'
names={'sparse_cca_16':'기존 Sparse CCA 16개','signed_pair':'부호 허용, 검색 목적만','positive_pair':'비음수, 검색 목적만','positive_coord_1':'비음수, 좌표 대응 목적 추가 1','positive_coord_4':'비음수, 좌표 대응 목적 추가 4','positive_reconstruct_1':'비음수, 복원 목적 추가 1','positive_reconstruct_4':'비음수, 복원 목적 추가 4','positive_reconstruct_2':'비음수, 검색과 복원 함께 학습 · 계수 2','positive_reconstruct_8':'비음수, 검색과 복원 함께 학습 · 계수 8','positive_reconstruct_16':'비음수, 검색과 복원 함께 학습 · 계수 16','positive_reconstruct4_activity':'복원 계수 4, 표본별 활성 희소성 추가','positive_reconstruct4_coord01':'복원 계수 4, 약한 좌표 대응 추가','positive_reconstruction_only':'비음수, 복원만 학습','positive_joint':'비음수, 좌표 대응과 복원 추가','positive_learnP':'비음수, 좌표 대응과 복원, P 학습'}
conds={'cc3m-coco':'SAE는 CC3M, 대응은 COCO2017로 학습','coco-coco':'SAE와 대응을 모두 COCO2017로 학습','cc3m-cc3m':'SAE와 대응을 모두 CC3M으로 학습'}
# Reassign path-independent group identity using relative parts.
rows=[]
for p in D.glob('*/*/seed*/*-*.json'):
 if not (p.stem.endswith('-tune') or p.stem.endswith('-test')):continue
 v=json.loads(p.read_text());rp=p.relative_to(D).parts
 vals=[100*v['retrieval'][d]['recall'][str(k)] for d in ['image_to_text','text_to_image'] for k in [1,5,10]]+[v[s]['summary']['agree_at1_count'] for s in ['positive','signed']]
 cats=[c for c in v['positive']['per_category'] if c.get('status')=='ok'];matched=[c for c in cats if c['agree_at1']]
 diag=dict(distinct_image_representatives=len({c['image_coordinate'] for c in cats}),distinct_text_representatives=len({c['text_coordinate'] for c in cats}),distinct_matched_coordinates=len({c['image_coordinate'] for c in matched}),matched_with_both_auc_at_least_07=sum(min(c['image_auc'],c['text_auc'])>=.7 for c in matched))
 rows.append(dict(round=rp[0],condition=rp[1],seed=rp[2],method=v['method'],population=v['population'],denominator=v['denominator'],values=vals,diagnostics=diag))
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><title>APBᵀ 반복 탐색</title><style>body{font:16px/1.65 system-ui;max-width:1450px;margin:36px auto;padding:0 25px;color:#242730}h1{font-size:28px}h2{font-size:21px;margin-top:34px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#f2f3f5}td:first-child,th:first-child{text-align:left}b{color:#b53232}u{text-underline-offset:4px}a{color:#265bbe}.note{color:#535d6a}</style><h1>해석과 검색을 함께 평가하는 APBᵀ 반복 탐색</h1><p>A와 B는 각 모달리티의 activation을 최대 16개씩 조합한다. P는 이미지 조합과 텍스트 조합을 대응시키는 행렬이다. 검색은 cos(xAP, yB)로 계산하며, 분자는 xAPBᵀyᵀ다.</p><p>검색 목적에 좌표별 대응 목적과 원래 activation 복원 목적을 각각 추가했다. P는 항등행렬이거나 학습 자료에서 정한 순열이다. 학습 경사, 좌표 선택, P 결정, 회차 선택에 개념 주석을 사용하지 않았다. 후보를 비교하고 후속 실험을 선택할 때는 별도 조정 자료의 범주 주석을 참고했다.</p><p>조정 자료는 COCO train2017의 별도 이미지 5,000장이고, 시험 자료는 COCO val2017이다. 표본이 다르므로 두 표의 범주 개수가 다를 수 있다. 이 시험 자료를 이전 작업에서 이미 확인했으므로 새로운 미관측 데이터 검증이라고 주장하지 않는다.</p><p>검색은 양방향 Recall@1·5·10(%)이며, 매칭은 개념별로 이미지와 텍스트에서 고른 대표 좌표의 번호가 일치한 개수다. 이미지에는 객체 주석, 텍스트에는 캡션 자체의 사전 기반 범주 언급을 사용한다. 각 표에서 <b>1등을 굵게</b>, <u>2등을 밑줄</u>로 표시했다. 여러 범주가 같은 좌표를 선택할 수도 있으므로, 매칭 수 자체가 개념 순도를 보장하지 않는다.</p>''']
parts[0]=parts[0].replace('</h1>','</h1><p><b>검색과 범주 매칭을 함께 개선하는 후보를 찾았다. 다만 CC3M SAE에서 유망했고, COCO2017 SAE에서는 범주 매칭까지 개선되지 않았다.</b> 아래는 15개 학습 조건을 두 차례 비교한 뒤 목적을 고정해 확인한 결과다.</p>',1)
summary_parts=['<section class="summary"><h2>이번 탐색에서 얻은 결과</h2><ul><li>CC3M SAE와 COCO2017 대응 조건에서 검색 전용 대조군의 높은 활성 매칭은 세 번 평균 14.67개였고, 복원을 추가하면 22.00개였다. 이미지 질의 Recall@5는 49.49%에서 48.94%, 텍스트 질의는 42.18%에서 40.93%로 바뀌었다. 이 조건에서는 작은 검색 손실과 함께 매칭이 개선되었다.</li><li>CC3M SAE와 CC3M 대응 조건에서는 복원 추가가 검색과 매칭을 모두 높였다. COCO2017 SAE에서는 검색 전용 대조군보다 검색과 매칭 모두 낮아졌다. 복원의 효과를 모든 SAE에 일반화할 근거는 없다.</li><li>기존 Sparse CCA의 낮은 활성도 허용하면, CC3M SAE와 COCO2017 대응 조건의 매칭 차이는 21개와 평균 22개다. 따라서 높은 활성만을 개념 반응으로 사용하는 해석에서 개선이 더 뚜렷하며, 부호를 허용한 의미 대응까지 크게 개선했다고 주장할 수는 없다.</li></ul></section>']
summary_parts.append('<h2>고정한 후보의 시험 결과</h2><p>세 조건 모두 목적과 복원 계수를 바꾸지 않았다. CC3M SAE와 COCO2017 대응 조건은 학습 표본 추출과 미니배치 순서를 바꾼 세 번의 평균 ± 표본 표준편차다. 기존 Sparse CCA는 재학습하지 않았으므로 한 값으로 표시한다. 다른 두 조건은 한 번의 결과다.</p>')
aggregates=[]
for cond in ['coco-coco','cc3m-coco','cc3m-cc3m']:
 ar=[]
 for method in ['sparse_cca_16','positive_pair','positive_reconstruct_2']:
  rr=[r for r in rows if r['population']=='test' and r['round']=='confirmation' and r['condition']==cond and r['method']==method]
  if not rr:continue
  means=[statistics.mean(r['values'][i] for r in rr) for i in range(8)]
  sds=[statistics.stdev(r['values'][i] for r in rr) if len(rr)>1 else 0 for i in range(8)]
  ar.append(dict(condition=cond,method=method,mean=means,sd=sds,n=len(rr),repeated=(len(rr)>1 and method!='sparse_cca_16')))
 aggregates.extend(ar)
 if not ar:continue
 ranks=[sorted({round(r['mean'][i],2) for r in ar},reverse=True) for i in range(8)]
 summary_parts.append('<h3>'+conds[cond]+'</h3><table><tr><th>학습 조건</th>'+''.join('<th>'+x+'</th>' for x in ['이미지 질의 R@1','이미지 질의 R@5','이미지 질의 R@10','텍스트 질의 R@1','텍스트 질의 R@5','텍스트 질의 R@10','높은 활성 매칭 /42','방향 허용 매칭 /42'])+'</tr>')
 for r in ar:
  summary_parts.append('<tr><td>'+names[r['method']]+'</td>')
  for i,v in enumerate(r['mean']):
   s=(f'{v:.2f} ± {r["sd"][i]:.2f}' if r['repeated'] else (f'{v:.2f}' if i<6 else str(int(v))))
   if round(v,2)==ranks[i][0]:s='<b>'+s+'</b>'
   elif len(ranks[i])>1 and round(v,2)==ranks[i][1]:s='<u>'+s+'</u>'
   summary_parts.append('<td>'+s+'</td>')
  summary_parts.append('</tr>')
 summary_parts.append('</table>')
summary_parts.append('''<h2>목적함수가 APBᵀ 구조에 주는 의미</h2><p><img src="objective.svg" alt="검색 InfoNCE와 양쪽 모달리티의 묶인 복원 오차를 더하고, A와 B의 각 열을 비음수인 최대 16개 activation 조합으로 제한한다." style="width:100%;max-width:1120px"></p><p>X와 Y는 SAE activation을 학습 자료의 좌표별 표준편차로 나눈 행렬이다. 비음수성을 유지하기 위해 평균은 빼지 않는다. A와 B의 각 열은 공통 좌표 하나를 만들고, P는 두 모달리티의 공통 좌표를 연결한다. 이번 확인 실험은 공통 좌표 256개와 P=I를 사용했다.</p><ul><li><strong>검색 목적</strong>은 같은 이미지·캡션의 유사도가 다른 쌍보다 높아지도록 학습한다. 이것만으로 각 좌표가 개념 하나를 나타내지는 않는다.</li><li><strong>묶인 복원 목적</strong>은 한 조합을 만들 때 읽은 activation을 같은 가중치의 전치로 복원한다. 따라서 조합 하나가 사용하는 activation과 설명하는 activation을 같은 집합으로 제한한다.</li><li><strong>비음수성과 16개 제한</strong>은 양수와 음수 계수의 상쇄를 막고 각 조합을 소수의 원래 SAE activation으로 추적할 수 있게 한다. 사람이 정의한 개념의 유일한 회복까지 보장하지는 않는다.</li></ul><p>이 복원 구조의 직접적인 선행 근거는 <a href="https://yangzh.folk.ntnu.no/preprints/tnn2010.pdf">Projective Nonnegative Matrix Factorization</a>이다. 검색과 희소 복원을 함께 쓰는 가까운 연구로 <a href="https://proceedings.mlr.press/v267/wen25e.html">Contrastive Sparse Representation</a>이 있다. 이 요소를 합쳤다는 사실만으로 새로운 방법의 기여가 확립되지는 않는다.</p><h2>목적을 바꾸며 확인한 내용</h2><ul><li>두 모달리티에서 같은 번호의 좌표가 다른 번호보다 상관이 높도록 하는 목적을 추가해도, 비음수 복원을 추가한 경우보다 조정 자료의 매칭이 좋지 않았다.</li><li>복원 계수를 2, 4, 8, 16으로 바꾸었다. 강하게 만들수록 좋아지지는 않았다. 조정 자료에서 정한 규칙에 따라 계수 2를 고르고 시험 결과를 확인한 후에도 유지했다.</li><li>복원만 학습하면 조정 자료의 양방향 Recall@5가 10.68%, 9.08%로 낮아졌다. 검색과 복원에 서로 다른 역할이 있다는 근거다.</li><li>복원에 표본별 활성 희소성이나 약한 좌표 대응 목적을 추가해도 선택 기준을 더 개선하지 못했다. P를 번갈아 재계산한 조건에서도 순열은 항등행렬에서 바뀌지 않았다.</li></ul><h2>남은 한계와 다음 판단 기준</h2><p>높은 활성 매칭의 개선은 긍정적이지만, 범주별 대표 번호가 같은지 평가한 결과다. 한 좌표가 다른 개념에도 반응하는지를 완전히 평가한 것이 아니며, 각 원래 SAE activation의 의미를 모두 확인한 결과도 아니다. COCO2017 SAE에서 같은 효과가 나타나지 않은 이유도 아직 분리하지 못했다.</p><p>다음 비교에서는 복원을 강하게 만드는 대신, 두 모달리티가 공유하는 조합과 한쪽에만 필요한 조합을 구분하는 부분 매칭을 검토할 가치가 있다. 현재 P는 전체 항등행렬이므로 모든 조합을 검색 유사도 계산에 포함한다. 부분 매칭이 불일치한 조합의 정렬 강요를 줄이는지는 이번 결과만으로 확인할 수 없다. 또한 원래 범주 주석을 사용한 Oracle보다 높은 매칭을 얻었다는 결과는 아니다.</p><p class="note">학습에는 범주 주석을 사용하지 않았지만, 후보 비교와 목적 계수 선택에는 조정 자료의 범주 주석을 참고했다. 시험 자료인 COCO val2017도 이전 작업에서 이미 사용했다. 이 결과는 탐색 근거이며 완전히 새로운 자료에서 검증한 최종 연구 결과가 아니다.</p><h2>모든 실험과 반복 결과</h2>''')
parts += summary_parts
(ROOT/'aggregate.json').write_text(json.dumps(aggregates,indent=2))
groups=sorted({(r['population'],r['round'],r['condition'],r['seed']) for r in rows})
for pop,round_,cond,seed in groups:
 rs=[r for r in rows if (r['population'],r['round'],r['condition'],r['seed'])==(pop,round_,cond,seed)]
 rs.sort(key=lambda r:(r['method']!='sparse_cca_16',r['method']))
 parts.append('<h2>'+('시험 평가' if pop=='test' else '조정 자료 비교')+' · '+conds.get(cond,cond)+' · '+{'round1':'첫 목적 비교','round2':'복원 목적 보강 비교','confirmation':'선택한 후보 확인'}[round_]+' · 반복 '+seed.replace('seed','')+'</h2>')
 ranks=[sorted({round(r['values'][i],2) for r in rs},reverse=True) for i in range(8)]
 parts.append('<table><tr><th>학습 조건</th>'+''.join('<th>'+s+'</th>' for s in ['이미지 질의 R@1','이미지 질의 R@5','이미지 질의 R@10','텍스트 질의 R@1','텍스트 질의 R@5','텍스트 질의 R@10',f'높은 활성 매칭 /{rs[0]["denominator"]}',f'방향 허용 매칭 /{rs[0]["denominator"]}'])+'</tr>')
 for r in rs:
  parts.append('<tr><td>'+html.escape(names.get(r['method'],r['method']))+'</td>')
  for i,v in enumerate(r['values']):
   s=f'{v:.2f}' if i<6 else str(v)
   if round(v,2)==ranks[i][0]:s='<b>'+s+'</b>'
   elif len(ranks[i])>1 and round(v,2)==ranks[i][1]:s='<u>'+s+'</u>'
   parts.append('<td>'+s+'</td>')
  parts.append('</tr>')
 parts.append('</table><details><summary>범주 매칭이 한 좌표에 집중되는지 확인</summary><table><tr><th>학습 조건</th><th>이미지 대표 좌표 종류</th><th>텍스트 대표 좌표 종류</th><th>일치한 좌표 종류</th><th>일치하고 양쪽 AUROC가 0.7 이상인 범주</th></tr>')
 for r in rs:parts.append('<tr><td>'+html.escape(names.get(r['method'],r['method']))+'</td>'+''.join('<td>'+str(v)+'</td>' for v in r['diagnostics'].values())+'</tr>')
 parts.append('</table><p>이 표는 매칭 수가 늘어난 이유를 점검하는 보조 진단이다. 학습 목적이나 새 주 평가 지표로 사용하지 않았다.</p></details>')
parts.append('<p><a href="research-notes.md">목적함수의 논리, 선행 연구와 평가 조건</a></p></html>');(ROOT/'report.html').write_text(''.join(parts));(ROOT/'summary.json').write_text(json.dumps(rows,indent=2));print(len(rows),'evaluations')
