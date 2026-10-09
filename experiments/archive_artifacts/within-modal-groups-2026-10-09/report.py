"""Build a measured-results report; missing measurements remain explicitly pending."""
from pathlib import Path
import json
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results'
CONDITIONS=['coco-coco','cc3m-coco','cc3m-cc3m']
CONDNAMES={'coco-coco':'SAE와 대응을 모두 COCO2017로 학습',
           'cc3m-coco':'SAE는 CC3M, 대응은 COCO2017로 학습',
           'cc3m-cc3m':'SAE와 대응을 모두 CC3M으로 학습'}
METHODS=['cca_256','sparse_cca_16','cca_support_control','correlation_groups',
         'binary_correlation_groups','conditional_groups']
NAMES={'cca_256':'기존 CCA · 희소성 제한 없음',
       'sparse_cca_16':'기존 Sparse CCA · 최대 16개',
       'cca_support_control':'기존 Sparse CCA 좌표를 고정하고 같은 검색 학습',
       'correlation_groups':'활성 크기의 상관으로 집합 구성',
       'binary_correlation_groups':'켜짐·꺼짐의 상관으로 집합 구성',
       'conditional_groups':'다른 좌표의 상태를 고려한 관계로 집합 구성'}

def metrics(r):
    return [100*r['retrieval'][d]['recall'][str(k)]
            for d in ['image_to_text','text_to_image'] for k in [1,5,10]] + [
            r[s]['summary']['agree_at1_count'] for s in ['positive','signed']]

def read(cond,name):
    p=DATA/cond/(name+'-test.json')
    return json.loads(p.read_text()) if p.exists() else None

allrows=[]
parts=['''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>모달리티 내부 관계로 activation 집합을 만드는 비교</title>
<style>
body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;color:#242b35;background:#fff;margin:40px auto;max-width:1580px;padding:0 28px}
h1{font-size:30px;letter-spacing:-.7px}h2{font-size:22px;margin:34px 0 12px}h3{font-size:18px}p{max-width:1200px}.lead{font-size:19px;font-weight:650}.note{color:#596576;font-size:14px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}td,th{padding:10px 8px;border-bottom:1px solid #e2e5e9;text-align:right;white-space:nowrap}th{background:#f4f6f8}td:first-child,th:first-child{text-align:left;white-space:normal;min-width:245px;max-width:330px}tr.divider td{border-top:2px solid #8b97a5}b{font-weight:750;color:#ad383e}u{text-underline-offset:4px;text-decoration-thickness:1.5px}a{color:#285ba6}details{margin:20px 0;padding:15px 20px;background:#f6f7f9;border-radius:6px}summary{cursor:pointer;font-weight:650}.result{border-left:4px solid #839ab6;padding-left:16px}li{margin:6px 0}img{max-width:100%}.formula{font:20px/1.8 Georgia,"Times New Roman",serif;overflow-x:auto;padding:12px;background:#f7f8fa}
</style></head><body>
<h1>모달리티 내부 관계로 activation 집합을 만들면 대응이 개선되는가?</h1>
<p>이미지와 텍스트 각각에서 관련된 SAE activation 좌표를 묶고, 그 집합 안에서만 가중치를 학습했다. 모달리티 내부의 관계로 집합을 제한하는 것이 기존 Sparse CCA보다 검색과 개념 매칭에 도움이 되는지 비교했다.</p>
''']

complete=all(read(c,m) for c in CONDITIONS for m in METHODS)
if complete:
    outcomes=[]
    for c in CONDITIONS:
        original=metrics(read(c,'sparse_cca_16')); control=metrics(read(c,'cca_support_control'))
        for m in ['correlation_groups','binary_correlation_groups','conditional_groups']:
            v=metrics(read(c,m))
            outcomes.append((c,m,v[1]>control[1] and v[4]>control[4] and v[6]>=control[6],
                             v[1]>original[1] and v[4]>original[4] and v[6]>original[6]))
    n=sum(x[2] for x in outcomes); nb=sum(x[3] for x in outcomes)
    parts.append(f'<p class="lead">세 데이터 조건의 첫 비교를 완료했다. 집합 구성 조건 9개 중 같은 검색 학습을 적용한 대조군보다 양방향 Recall@5가 높고 높은 활성 매칭도 유지하거나 높인 경우는 {n}개였다. 기존 Sparse CCA보다 세 지표가 모두 높아진 경우는 {nb}개였다.</p>')
else:
    parts.append('<p class="lead">실험 결과를 수집하고 있다. 빈칸은 측정이 끝나지 않은 값이며 추정값을 넣지 않았다.</p>')
parts.append('''<p class="note">Recall@K는 질의별 상위 K개 안에 정답이 하나 이상 있는 비율이다. 이미지 질의는 캡션을 검색하고 텍스트 질의는 이미지를 검색한다. 매칭 수는 개념별로 이미지와 텍스트에서 독립적으로 선택한 대표 조합의 번호가 일치한 개수다. 각 표·열에서 <b>가장 높은 값은 굵게</b>, <u>두 번째로 높은 값은 밑줄</u>로 표시했다.</p>''')

for condition in CONDITIONS:
    rows=[(m,read(condition,m)) for m in METHODS]
    measured=[metrics(r) for _,r in rows if r]
    ranks=[sorted(set(round(v[i],2) for v in measured),reverse=True) for i in range(8)]
    parts.append('<h2>'+CONDNAMES[condition]+'</h2><div class="scroll"><table><thead><tr><th rowspan="2">방법</th><th colspan="3">이미지 질의 Recall (%)</th><th colspan="3">텍스트 질의 Recall (%)</th><th rowspan="2">높은 활성<br>매칭 /42</th><th rowspan="2">반응 방향 허용<br>매칭 /42</th></tr><tr>'+''.join('<th>'+x+'</th>' for x in ['@1','@5','@10']*2)+'</tr></thead><tbody>')
    for method,r in rows:
        parts.append('<tr'+(' class="divider"' if method in ['cca_support_control','correlation_groups'] else '')+'><td>'+NAMES[method]+'</td>')
        if not r:
            parts.append('<td colspan="8">평가 중</td></tr>');continue
        assert r['denominator']==42
        values=metrics(r);allrows.append(dict(condition=condition,method=method,values=values))
        for i,v in enumerate(values):
            s=f'{v:.2f}' if i<6 else str(v)
            if round(v,2)==ranks[i][0]:s='<b>'+s+'</b>'
            elif len(ranks[i])>1 and round(v,2)==ranks[i][1]:s='<u>'+s+'</u>'
            parts.append('<td>'+s+'</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    raw=read(condition,'correlation_groups'); binary=read(condition,'binary_correlation_groups'); cond=read(condition,'conditional_groups')
    if raw and cond:
        a,b=metrics(raw),metrics(cond)
        parts.append('<p class="result">활성 크기의 상관으로 만든 집합에서 조건부 관계로 만든 집합으로 바꾸면, 이미지 질의 Recall@5는 '+f'{a[1]:.2f}%에서 {b[1]:.2f}%, 텍스트 질의는 {a[4]:.2f}%에서 {b[4]:.2f}%로 바뀌었다. 높은 활성 매칭은 {a[6]}개에서 {b[6]}개로 바뀌었다.</p>')
        if binary:
            a=metrics(binary)
            parts.append('<p>켜짐·꺼짐 자료를 같게 둔 비교에서는 단순 상관의 높은 활성 매칭이 '+f'{a[6]}개, 조건부 관계의 매칭이 {b[6]}개였다. 양방향 Recall@5는 각각 {a[1]:.2f}%·{a[4]:.2f}%와 {b[1]:.2f}%·{b[4]:.2f}%였다.</p>')

parts.append('''<h2>이번 비교에서 바꾼 것</h2>
<ol><li>활성 크기의 상관 조건은 여러 학습 표본에서 두 activation 좌표가 함께 증가하거나 감소하는 정도를 사용했다.</li>
<li>켜짐·꺼짐의 상관 조건은 같은 자료에서 양수이면 1, 0이면 0으로 바꾼 뒤 상관을 구했다. 활성 크기를 버린 효과를 확인하는 대조군이다.</li>
<li>조건부 관계 조건은 같은 이진 자료로 한 좌표의 켜짐을 나머지 좌표들에서 예측하는 대칭 Ising 모델을 학습했다. 다른 좌표들의 상태를 고려한 통계적 관계를 사용했다.</li></ol>
<p>세 집합 구성 조건 모두 관계의 절댓값을 사용했다. 양의 관계와 음의 관계를 모두 집합 구성에 활용하기 위해서다. 자기 자신과의 관계인 대각 원소는 제외했다. 각 좌표의 총연결 강도로 보정한 관계 행렬에 같은 비음수 행렬 분해를 적용했다. 한 activation이 여러 집합에 포함되는 것을 허용했고, 집합 256개와 집합당 최대 16개 좌표를 사용했다.</p>
<p class="formula">G ≈ HHᵀ, &nbsp; H ≥ 0, &nbsp; 각 H 열의 0이 아닌 원소 수 ≤ 16</p>
<p>G는 좌표 간 관계의 강도이며, H의 각 열은 집합 하나의 구성 가중치다. 대각을 제외한 재구성 제곱오차를 최소화했다. 이 단계에서는 이미지·텍스트의 짝이나 개념 주석으로 집합을 바꾸지 않았다.</p>
<h2>집합을 정한 뒤에는 같은 검색 학습을 적용했다</h2>
<p>A와 B의 각 열은 해당 집합에 속한 activation만 사용한다. 계수는 비음수이고 각 열의 L2 길이는 1이다. 처음 구한 집합 조합의 학습 자료 상관을 최대화하는 헝가리안 순열을 P로 정한 뒤 고정했다. 이번에는 256개 전체 집합을 대응시켰으며 일부 집합을 대응시키지 않는 부분 매칭은 시험하지 않았다.</p>
<p class="formula">W = APBᵀ, &nbsp; s(x,y) = cos(xAP, yB)</p>
<p>정답 이미지·캡션 쌍을 같은 배치의 다른 쌍보다 높은 코사인 유사도로 구별하는 양방향 InfoNCE 목적만 학습했다. 복원이나 개념 예측 목적은 추가하지 않았다. 모든 새 조건은 같은 32,768개 학습 쌍과 별도 2,048개 조정 쌍, 같은 배치 순서를 사용했다. 학습률은 0.003, 배치 크기는 512, 온도는 0.07이며, 최대 60회 반복했다. 조정 자료의 검색 손실이 5회 동안 개선되지 않으면 중지하고 가장 낮았던 회차를 선택했다.</p>
<p>SAE activation은 학습 표준편차로 나누고 평균은 빼지 않았다. 이는 비음수 조합을 평가하기 위한 설정이다. 상관행렬을 계산할 때는 각 좌표의 평균을 뺐다. 기존 CCA와 Sparse CCA는 저장된 학습 평균·표준편차 보정과 기존 계수를 그대로 사용한 참고값이다. 같은 검색 학습을 적용한 대조군은 기존 Sparse CCA의 좌표를 고정하고 계수의 절댓값으로 시작했다.</p>
<p class="note">집합 구성에는 학습 자료에서 켜짐과 꺼짐이 각각 32번 이상 관측된 좌표만 사용했다. 기존 Sparse CCA의 좌표 대조군에는 이 빈도 필터를 새로 적용하지 않았다. 따라서 기존 Sparse CCA와의 차이를 집합 구성 방식 하나의 효과로 단정할 수는 없다. 세 집합 구성 조건 사이에는 같은 빈도 필터를 적용했다.</p>
<h2>개념 매칭을 잰 방법</h2>
<p>COCO val2017 이미지를 두 집단으로 나누었다. 첫 집단에서 각 객체 범주의 존재를 가장 잘 구별하는 이미지 조합 번호를 AUROC로 골랐다. 다른 집단의 캡션에서 같은 범주의 언급 여부를 가장 잘 구별하는 텍스트 조합 번호를 별도로 골랐다. 학습한 P에 따라 대응시킨 뒤 번호가 일치하면 매칭으로 셌다. 캡션 자체의 사전 기반 언급을 사용했으며 연결된 이미지의 주석을 텍스트 정답으로 대신하지 않았다.</p>
<p>80개 객체 범주 중 양쪽 평가 집단에 양성 표본이 각각 50개 이상인 42개를 사용했다. 높은 활성 매칭은 원래 AUROC가 가장 높은 조합을 고른 결과다. 반응 방향 허용 매칭은 낮은 활성으로 개념을 구별하는 조합도 허용하고, 번호와 반응 방향이 모두 맞는지 평가했다. 이 개수는 대표 조합의 일치이며 집합 전체의 의미가 하나로 한정된다는 뜻은 아니다.</p>
<details><summary>학습과 집합 구성 진단</summary>
''')

fig,axes=plt.subplots(1,3,figsize=(15,3.4),layout='constrained')
colors={'cca_support_control':'#6d7482','correlation_groups':'#d4777d','binary_correlation_groups':'#bda448','conditional_groups':'#678dc9'}
for ax,c in zip(axes,CONDITIONS):
    for m in colors:
        p=DATA/c/(m+'-history.json')
        if not p.exists():continue
        h=json.loads(p.read_text());ax.plot([r['epoch'] for r in h],[r['tune_loss'] for r in h],color=colors[m],label=m.replace('_groups','').replace('_control',''))
    ax.set_title(c);ax.set_xlabel('Epoch');ax.set_ylabel('Held-out InfoNCE');ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axes[-1].legend(fontsize=7,frameon=False)
fig.savefig(ROOT/'training-curves.png',dpi=160);plt.close(fig)
parts.append('<img src="training-curves.png" alt="세 데이터 조건의 조정 자료 검색 손실 곡선"><p>회차 상한에서 끝난 조건을 완전히 수렴했다고 간주하지 않는다. 희소 집합을 만드는 행렬 분해 역시 400회라는 계산 한도에서 비교했다. 초기 탐색이며 이론적 최적해의 비교가 아니다.</p>')
parts.append('<table><tr><th>자료 조건과 모달리티</th><th>좌표 수</th><th>집합 구성에 사용 가능한 좌표 수</th><th>독립 켜짐 모형의 조정 오차</th><th>조건부 모형의 조정 오차</th></tr>')
for c in CONDITIONS:
    for side in ['image','text']:
        p=DATA/c/(side+'-conditional-fit.json'); gp=DATA/c/(side+'-conditional-groups.json')
        if not p.exists() or not gp.exists():continue
        v=json.loads(p.read_text());g=json.loads(gp.read_text())
        with np.load(DATA/c/'feature-ids.npz') as z:d=len(z[side])
        parts.append('<tr><td>'+CONDNAMES[c]+' · '+('이미지' if side=='image' else '텍스트')+'</td>'+f'<td>{d}</td><td>{g["eligible_coordinates"]}</td><td>{v["zero_interaction_tune_nll"]:.5f}</td><td>{v["best_tune_nll"]:.5f}</td></tr>')
parts.append('''</table><p>조건부 모형의 오차는 각 activation 좌표의 켜짐 확률에 대한 평균 이진 교차엔트로피다. 개념 주석 예측 오차가 아니다. Ising 모델은 L2 계수 0.01을 고정한 의사우도로 학습했고, 별도 조정 자료의 이 오차로 회차를 선택했다.</p></details>
<h2>해석할 수 있는 범위</h2><p>상관이 높은 서로 다른 개념이 같은 집합으로 묶일 수 있다. 조건부 관계도 인과관계나 같은 의미를 보장하지 않는다. TopK SAE의 켜질 수 있는 좌표 수 제한도 음의 관계를 만들 수 있다. 이번 비교는 각 데이터 조건에서 한 번씩 수행했으며, 평가 자료는 이전 탐색에서도 사용했던 COCO val2017이다. 범주 주석으로 이번 집합이나 학습 회차를 선택하지 않았다.</p>
<p>참고한 연구는 <a href="https://arxiv.org/html/2604.28119v1">Do Sparse Autoencoders Capture Concept Manifolds?</a>와 <a href="https://faculty.cc.gatech.edu/~hpark/papers/DaDingParkSDM12.pdf">Symmetric Nonnegative Matrix Factorization for Graph Clustering</a>다. 전자의 좌표 관계 추정 아이디어와 후자의 집합 구성 목적을 우리 설정에 적용했다. 원 논문의 그래프 군집 알고리즘이나 규제 선택을 그대로 재현한 실험은 아니다.</p>
<p><a href="summary.json">수치 결과</a> · <a href="run.py">실험 코드</a> · <a href="verification.json">검증 내역</a></p></body></html>''')
(ROOT/'report.html').write_text(''.join(parts))
(ROOT/'summary.json').write_text(json.dumps(allrows,indent=2,ensure_ascii=False))
print('Built',len(allrows),'rows. All complete:',complete)
