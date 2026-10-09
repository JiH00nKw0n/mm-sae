"""Build a self-contained report from completed set-CCA evaluations."""
from pathlib import Path
import json, html
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results'
CONDITIONS={'coco-coco':('COCO2017','COCO2017'),'cc3m-coco':('CC3M','COCO2017'),'cc3m-cc3m':('CC3M','CC3M')}
NAMES={'cca_256':'일반 CCA','sparse_cca_16':'Sparse CCA · 조합당 최대 16개',
       'set_cca_correlation':'활성값 상관으로 집합 구성 후 CCA',
       'set_cca_binary_correlation':'켜짐·꺼짐 상관으로 집합 구성 후 CCA',
       'set_cca_conditional':'조건부 관계로 집합 구성 후 CCA'}
METHODS=['correlation','binary_correlation','conditional']
SHORT={'correlation':'활성값 상관','binary_correlation':'켜짐·꺼짐 상관','conditional':'조건부 관계'}
def read(path):return json.loads(path.read_text())
def get(c,m):
    d=read(DATA/c/(m+'-test.json'));g=read(DATA/c/(m+'-diagnostics.json'))
    return dict(condition=c,method=m,outputs=g['outputs'],values=[100*d['retrieval'][direction]['recall'][str(k)] for direction in ['image_to_text','text_to_image'] for k in [1,5,10]]+[d[key]['summary']['agree_at1_count'] for key in ['positive','signed']],diagnostics=g)
def table(rows):
    values=np.array([r['values'] for r in rows]);ranks=[sorted(set(values[:,j]),reverse=True) for j in range(8)]
    s='<div class="scroll"><table><thead><tr><th rowspan="2">방법</th><th rowspan="2">출력 좌표 수</th><th colspan="3">이미지 질의로 캡션 검색 (%)</th><th colspan="3">텍스트 질의로 이미지 검색 (%)</th><th colspan="2">대표 조합 매칭 /42</th></tr><tr><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>높은 활성만</th><th>반응 방향 허용</th></tr></thead><tbody>'
    for r in rows:
        name=NAMES.get(r['method'],f'일반 CCA · {r["outputs"]}개 좌표로 맞춤')
        s+=f'<tr><td>{name}</td><td>{r["outputs"]}</td>'
        for j,v in enumerate(r['values']):
            val=f'{v:.2f}' if j<6 else str(int(v))
            if v==ranks[j][0]:val=f'<b>{val}</b>'
            elif len(ranks[j])>1 and v==ranks[j][1]:val=f'<u>{val}</u>'
            s+=f'<td>{val}</td>'
        s+='</tr>'
    return s+'</tbody></table></div>'

assert read(DATA/'complete.json')['completed']
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>내부 activation 집합을 CCA로 대응한 결과</title><style>
body{margin:0;background:#f6f7f9;color:#20242c;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;line-height:1.7;font-size:16px}main{max-width:1440px;margin:45px auto;padding:0 32px 70px}h1{font-size:34px;line-height:1.35;margin:12px 0 22px}h2{font-size:24px;margin:42px 0 12px}h3{font-size:19px;margin:28px 0 10px}p{max-width:1100px}.lead{font-size:19px}.card{background:white;padding:22px 27px;border:1px solid #e1e4ea;border-radius:9px;margin:24px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin:12px 0;background:white;font-size:14px}th,td{padding:10px 11px;text-align:right;border-bottom:1px solid #e1e5eb;white-space:nowrap}th{background:#eef1f6;font-weight:600}td:first-child,th:first-child{text-align:left}b,strong{font-weight:750}u{text-underline-offset:4px;text-decoration-thickness:1.5px}.note{color:#566070;font-size:14px}a{color:#365db4}li{margin:8px 0}details{background:white;border:1px solid #dfe3ea;padding:16px 20px;margin-top:18px}summary{cursor:pointer;font-weight:650}.math{font-family:"Times New Roman",serif;font-size:22px;background:#eff2f7;padding:17px 22px}code{font-size:13px;overflow-wrap:anywhere}small{color:#566070}</style><main>
<small>2026-10-09 · 세 학습 조건의 실제 데이터 평가</small>
<h1>내부 관계로 activation 집합을 정하고,<br>CCA로 이미지·텍스트 집합을 대응했다</h1>
<p class="lead">이미지 내부와 텍스트 내부의 관계로 집합만 정했다. 집합 크기를 16개로 제한하지 않았다. 그다음 각 이미지·텍스트 집합 쌍 안에서 상관이 가장 높은 activation 조합을 CCA로 구했다.</p>
<div class="card"><strong>이번에 검증한 범위</strong><p>CCA는 두 조합이 같은 표본 쌍에서 함께 변하는 정도를 높이는 방법이다. 각 후보 집합 쌍의 첫 번째 CCA 상관을 점수로 삼아 집합을 일대일로 연결했다. 서로 다른 집합을 다시 섞는 변환이나 검색 목적의 추가 학습은 하지 않았다.</p><p>한 집합 쌍에서 숫자 하나만 남기는 방식이므로, 최종 출력 좌표 수는 대응된 집합 쌍의 수와 같다. 집합 구성에 따른 차이와 출력 차원 감소를 구분하기 위해 같은 출력 좌표 수의 일반 CCA도 함께 평가했다.</p></div>
''']
allrows=[]
for c,(sae,mapping) in CONDITIONS.items():
    rows=[get(c,m) for m in NAMES];allrows.extend(rows)
    parts.append(f'<h2>SAE 학습은 {sae}, 대응 학습은 {mapping}을 사용했다</h2>')
    parts.append(table(rows))
    parts.append('<p class="note">각 표의 열마다 최고값을 굵게, 두 번째로 높은 서로 다른 값을 밑줄로 표시했다. 검색은 COCO val2017 이미지 5,000장과 캡션 25,014개에서 평가했다. 출력 좌표 수가 다르므로 아래의 동일 차원 비교도 함께 봐야 한다.</p>')
    parts.append('<h3>집합당 activation 수를 미리 정하지 않았을 때의 결과</h3><table><tr><th>집합 구성 기준</th><th>이미지 집합 수</th><th>이미지 집합 크기<br>최소 / 중앙값 / 최대</th><th>텍스트 집합 수</th><th>텍스트 집합 크기<br>최소 / 중앙값 / 최대</th><th>대응된 집합 쌍</th></tr>')
    for method in METHODS:
        gi,gt=[read(DATA/c/f'{s}-{method}-sets.json') for s in ['image','text']]
        f=read(DATA/c/f'set_cca_{method}-fit.json')
        size=lambda d:f'{d["min_size"]} / {d["median_size"]:g} / {d["max_size"]}'
        parts.append(f'<tr><td>{SHORT[method]}</td><td>{gi["n_groups"]}</td><td>{size(gi)}</td><td>{gt["n_groups"]}</td><td>{size(gt)}</td><td>{f["outputs"]}</td></tr>')
    parts.append('</table><details><summary>출력 좌표 수를 맞춘 일반 CCA와 비교</summary>')
    for method in METHODS:
        row=get(c,'set_cca_'+method);control=get(c,'cca_dimension_'+str(row['outputs']));allrows.append(control)
        parts.append(f'<h3>{SHORT[method]}로 만든 집합과 일반 CCA를 {row["outputs"]}개 좌표에서 비교했다</h3>'+table([row,control]))
    parts.append('</details><details><summary>매칭된 범주가 몇 개의 서로 다른 조합에 모였는지 확인</summary><p>여러 범주가 같은 조합을 대표로 선택할 수 있다. 따라서 매칭된 범주 수와 매칭에 사용된 서로 다른 조합 수를 나누어 표시했다. 조합 수 자체는 높을수록 무조건 좋다는 평가 지표가 아니다.</p><table><tr><th>방법</th><th>높은 활성 매칭 /42</th><th>매칭에 사용된 서로 다른 조합 수</th><th>양쪽 AUROC가 모두 0.7 이상인 매칭 수</th></tr>')
    for row in rows:
        g=row['diagnostics'];parts.append(f'<tr><td>{NAMES[row["method"]]}</td><td>{row["values"][6]}</td><td>{g["distinct_matched_representatives"]}</td><td>{g["matched_with_both_auroc_above_07"]}</td></tr>')
    parts.append('</table><p class="note">AUROC는 개념이 있는 표본에 없는 표본보다 높은 점수를 주는 정도를 뜻한다. 0.7 기준은 결과를 읽기 위한 진단이며 모델 선택이나 학습에는 사용하지 않았다. 높은 활성만 보는 평가는 CCA 계수의 부호에 영향을 받으므로, 본표에 반응 방향을 허용한 결과도 함께 표시했다.</p></details>')

mainrows=[get(c,m) for c in CONDITIONS for m in NAMES if m.startswith('set_cca_')]
dimension_comparisons=[(row,get(row['condition'],'cca_dimension_'+str(row['outputs']))) for row in mainrows]
all_lower=all(all(row['values'][j]<control['values'][j] for j in [1,4]) for row,control in dimension_comparisons)
conclusion='<div class="card"><strong>이번 비교에서 확인한 점</strong><ul>'
if all_lower:
    conclusion+='<li><strong>집합을 먼저 나눈 CCA는 세 데이터 조건의 세 집합 구성 방식 모두에서 검색 성능이 낮았다.</strong> 출력 좌표 수를 맞춘 일반 CCA보다도 양방향 Recall@5가 낮아, 출력 좌표 수 감소만으로 차이를 설명할 수는 없다.</li>'
target=get('cc3m-coco','set_cca_conditional');td=read(DATA/'cc3m-coco/set_cca_conditional-test.json')
sr=[r for r in td['signed']['per_category'] if r['status']=='ok' and r['agree_at1']]
conclusion+=f'<li><strong>대표 조합 매칭과 검색 성능은 함께 오르지 않았다.</strong> CC3M SAE를 COCO2017에서 대응시킨 조건부 관계 방식은 반응 방향 허용 매칭이 {len(sr)}/42였지만 이미지 질의 Recall@5는 {target["values"][1]:.2f}%였다. 매칭된 {len(sr)}개 범주는 서로 다른 {len(set(r["image_coordinate"] for r in sr))}개 조합에 모였다. 매칭 개수가 늘었다는 사실만으로 각 개념을 따로 분리했다고 판단할 수는 없다.</li>'
conclusion+='<li><strong>확인한 것은 현재 집합 구성과 집합당 첫 번째 CCA 조합을 사용하는 구현의 결과다.</strong> 내부 관계에서 집합을 구한 뒤 CCA로 연결한다는 발상 전체가 불가능하다는 결과는 아니다. 이번 구현에서는 집합이 서로 겹칠 수 없고, 집합 하나에서 여러 공통 방향을 남기지 않았다.</li></ul></div>'
parts.insert(1,conclusion)
parts.append('''<h2>실험 절차</h2>
<ol><li><strong>모달리티 내부 관계로 activation 집합을 구성했다.</strong> 학습 자료 32,768개에서 이미지 좌표끼리, 텍스트 좌표끼리 관계를 따로 구했다. 활성값 상관, 양수인지 여부만 남긴 켜짐·꺼짐 상관, 나머지 좌표들의 상태를 고려하는 Ising 모델의 조건부 관계를 비교했다. 켜짐과 꺼짐이 각각 32번 이상 있는 좌표를 사용했다.</li>
<li><strong>집합 크기와 집합 수를 고정하지 않았다.</strong> 관계의 절댓값에서 자기 자신에 해당하는 대각을 제외하고, 각 좌표의 총연결 강도로 보정했다. 내부 연결이 강한 좌표를 묶는 Louvain 군집화를 적용했다. 해상도는 1, 난수 시드는 0으로 고정하고 반환된 분할 중 가장 세밀한 첫 단계를 사용했다. 서로 다른 집합의 좌표는 겹치지 않는다. 평가 성능으로 집합 수나 분할 단계를 고르지 않았다.</li>
<li><strong>고정된 집합 쌍마다 CCA를 구했다.</strong> 기존 CCA와 같은 전체 대응 학습 자료의 평균·공분산을 사용했다. COCO2017에서는 473,400쌍, CC3M에서는 2,324,763쌍이다. 공분산에 0.01배 단위행렬을 더하는 기존 안정화 조건도 유지했다. 후보 집합 쌍마다 첫 번째 CCA 조합과 상관을 구했다.</li>
<li><strong>CCA 상관의 합이 가장 큰 집합 대응을 선택했다.</strong> 헝가리안 알고리즘으로 집합을 일대일 대응시켰다. activation 하나씩을 대응시키는 기존 헝가리안과는 대응 단위가 다르다. 집합 수가 더 적은 모달리티의 집합을 모두 연결했고, 더 많은 쪽의 남는 집합은 제외했다. 낮은 상관을 이유로 연결을 거절하는 기준은 추가하지 않았다.</li>
<li><strong>대응된 조합 벡터의 코사인 유사도로 검색했다.</strong> 각 조합은 그 집합 안의 activation만 사용한다. CCA 계수에는 양수와 음수를 모두 허용했다. 검색 목적의 경사하강이나 범주 주석에 따른 가중치 조정은 하지 않았다.</li></ol>
<p class="math">W = APBᵀ, &nbsp; s(x,y) = cos(xA<sub>matched</sub>, yB<sub>matched</sub>)</p>
<p>A와 B의 각 열은 선택된 집합 안에서 구한 CCA 조합이다. P는 선택된 이미지 집합과 텍스트 집합을 연결하는 0·1 행렬이다. 집합 수가 다르면 P는 직사각형이며, 각 행과 열에 1이 최대 하나 있다. 저장된 검색용 행렬에서는 대응된 열끼리 같은 위치에 오도록 이미 순서를 맞췄다.</p>
<h2>검색과 매칭을 잰 기준</h2>
<p>Recall@K는 상위 K개 검색 후보 안에 정답이 하나 이상 포함된 질의의 비율이다. 이미지 질의에서는 해당 이미지의 캡션들 중 하나라도 포함되면 성공이며, 텍스트 질의에서는 연결된 정답 이미지를 찾으면 성공이다.</p>
<p>매칭은 같은 객체 범주를 대표하는 조합이 이미지와 텍스트에서 서로 연결되어 있는지 확인했다. COCO val2017 이미지를 두 집단으로 나누고, 첫 집단의 이미지에서 범주 존재를 가장 잘 구별하는 조합을 AUROC로 골랐다. 다른 집단의 캡션에서는 그 범주를 직접 언급했는지를 기준으로 텍스트 조합을 별도로 골랐다. 텍스트의 정답은 캡션 자체의 사전 기반 언급이며 연결된 이미지의 주석으로 대체하지 않았다.</p>
<p>80개 객체 범주 중 양쪽 평가 집단에 양성 표본이 각각 50개 이상 있는 42개를 평가했다. ‘높은 활성만’은 AUROC가 가장 높은 조합을 고른 결과다. ‘반응 방향 허용’은 낮은 활성으로 범주를 구별하는 경우도 허용하되, 대응된 조합 번호와 반응 방향이 모두 맞아야 성공으로 셌다. 이 수치는 범주별 대표 조합의 일치이며, 한 조합이 오직 한 개념만 표현한다는 검증은 아니다.</p>
<h2>결과를 해석할 때 남아 있는 조건</h2>
<p>이번 구현은 집합을 서로 겹치지 않게 나누고 집합 쌍마다 CCA 조합 하나만 남겼다. 이 선택이 검색에 필요한 정보를 얼마나 줄였는지는 따로 분리하지 않았다. 관계의 절댓값을 사용했으므로, 함께 켜지는 관계뿐 아니라 번갈아 켜지는 관계도 같은 집합 구성에 기여한다. 내부 관계로 집합을 만드는 모든 방법에 대한 결론으로 확대할 수 없다.</p>
<p>집합 구성에 사용한 좌표 필터는 세 새 조건에 동일하게 적용했다. 기존 CCA와 Sparse CCA에는 이 필터를 새로 적용하지 않았다. 같은 출력 차원의 CCA 비교로 출력 수 차이는 통제했지만, 좌표 제외와 집합 제약의 효과까지 따로 분리한 것은 아니다. 각 조건은 난수 시드 하나에서 측정한 탐색 결과이며 COCO val2017은 이전 실험에서도 사용한 평가 자료다.</p>
<p>내부 관계에서 집합을 찾는 발상은 <a href="https://arxiv.org/html/2604.28119v1">Do Sparse Autoencoders Capture Concept Manifolds?</a>를 참고했다. 집합 구성에는 <a href="https://arxiv.org/abs/0803.0476">Fast unfolding of communities in large networks</a>의 Louvain 방법을 사용했다. 해당 연구들이 여기서 측정한 이미지·텍스트 대응 성능을 보장하는 것은 아니다.</p>
<p><a href="summary.json">전체 수치 결과</a> · <a href="run.py">실험 코드</a> · <a href="verification.json">실행 결과와 파일 무결성 검증</a></p></main></html>''')
(ROOT/'report.html').write_text(''.join(parts))
(ROOT/'summary.json').write_text(json.dumps(allrows,indent=2,ensure_ascii=False))
print('Built report with',len(allrows),'rows')
