from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mathtext import math_to_image

root=Path(__file__).resolve().parent
data=root/'results'
methods=['sparse_cca','nce_centered','nce_uncentered','nce_nonnegative','known_anchor_reference']
labels=['Sparse CCA','InfoNCE, centered','InfoNCE, uncentered','InfoNCE, nonnegative','Known anchor reference']
ko=['Sparse CCA','InfoNCE, 평균 제거','InfoNCE, 평균 유지','InfoNCE, 평균 유지·비음수 계수','정답 생성 구조를 알려준 참고값']
rhos=[0,.5,.9,1]
rows=[]
for rho in rhos:
    for method in methods:
        items=[json.loads((data/f'seed{s}-rho{rho:.1f}'/f'{method}-evaluation.json').read_text()) for s in range(3)]
        metrics={}
        metrics['selectivity']=[v['selectivity']['target_mean'] for v in items]
        metrics['all_factor_selectivity']=[v['selectivity']['all_factor_mean'] for v in items]
        metrics['exclusive_auc']=[v['exclusive_category_auc']['recombined'] for v in items]
        for mode in ['iid','recombined']:
            for direction in ['image_to_text','text_to_image']:
                for k in (1,5,10):
                    metrics[f'{mode}_{direction}_R{k}']=[100*v['retrieval'][mode][direction]['recall'][str(k)] for v in items]
        rows.append(dict(rho=rho,method=method,metrics={k:dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)),values=v) for k,v in metrics.items()}))
(root/'summary.json').write_text(json.dumps(rows,indent=2))

colors=['#555b66','#5385b9','#d78f50','#9580bf','#a4bda0']
fig,axes=plt.subplots(1,3,figsize=(13.5,4),constrained_layout=True)
metrics=['selectivity','iid_image_to_text_R5','recombined_image_to_text_R5']
titles=['Target-specific intervention response','Retrieval: same co-occurrence','Retrieval: independently recombined targets']
for method,label,color in zip(methods,labels,colors):
    for ax,key in zip(axes,metrics):
        vals=[next(r for r in rows if r['method']==method and r['rho']==rho)['metrics'][key] for rho in rhos]
        mean=np.array([v['mean'] for v in vals]);sd=np.array([v['std'] for v in vals])
        ax.plot(rhos,mean,color=color,marker='o',markersize=4,lw=1.8,ls='--' if method=='known_anchor_reference' else '-',label=label)
        ax.fill_between(rhos,mean-sd,mean+sd,color=color,alpha=.1)
for ax,title in zip(axes,titles):
    ax.set(title=title,xlabel='Training correlation between two concept gates')
    ax.set_xticks(rhos,['0','0.5','0.9','1.0'])
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
axes[0].set(ylim=(-.02,1.03),ylabel='Selective response score (1 = target only)')
for ax in axes[1:]:ax.set(ylim=(0,100),ylabel='Image-query Recall@5 (%)')
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
fig.savefig(root/'comparison.png',dpi=180)
plt.close(fig)
math_to_image(r'$q_{jc}=\frac{R_{jc}^{2}}{\sum_{\ell=1}^{8}R_{\ell c}^{2}}$',root/'selectivity.svg',dpi=180)

table=[]
for r in rows:
    m=r['metrics'];name=ko[methods.index(r['method'])]
    vals=[m['selectivity']['mean'],m['exclusive_auc']['mean'],m['iid_image_to_text_R5']['mean'],m['iid_text_to_image_R5']['mean'],m['recombined_image_to_text_R5']['mean'],m['recombined_text_to_image_R5']['mean']]
    table.append(f'<tr><td>{r["rho"]:.1f}</td><th>{name}</th>'+''.join(f'<td>{v:.3f}</td>' if i<2 else f'<td>{v:.2f}</td>' for i,v in enumerate(vals))+'</tr>')
full=[]
for r in rows:
    for mode,title in [('iid','학습과 같은 동시 등장 조건'),('recombined','두 관심 개념을 독립적으로 재조합한 조건')]:
        vals=[r['metrics'][f'{mode}_{direction}_R{k}'] for direction in ['image_to_text','text_to_image'] for k in [1,5,10]]
        full.append(f'<tr><td>{r["rho"]:.1f}</td><th>{ko[methods.index(r["method"])]}</th><td>{title}</td>'+''.join(f'<td>{v["mean"]:.2f} ± {v["std"]:.2f}</td>' for v in vals)+'</tr>')
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>동시 등장과 개념 분리의 합성 실험</title>
<style>body{max-width:1320px;margin:45px auto;padding:0 25px;color:#24262a;font:17px/1.8 system-ui}h1{font-size:30px}h2{font-size:23px;margin-top:35px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{text-align:left}thead{background:#f2f4f6}img{max-width:100%}.scroll{overflow:auto}.note{font-size:15px;color:#545861}</style>
<h1>함께 등장하는 두 개념을 별도로 복원할 수 있는가?</h1>
<p>이미지와 텍스트 역할의 합성 activation에서 Sparse CCA와 세 가지 희소 InfoNCE 방법을 비교했다. 실제 사진이나 캡션을 사용한 실험은 아니다. 모든 학습 방법에는 두 관측의 짝만 제공하고, 생성에 사용한 잠재 요인은 학습과 모델 선택에 제공하지 않았다.</p>
<h2>이번 실험에서 확인한 세 가지</h2>
<ul><li><strong>검색이 잘되어도 개념이 분리되었다고 판단할 수 없다.</strong> 두 관심 개념이 항상 함께 등장할 때, 평균을 제거한 InfoNCE의 이미지 질의 Recall@5는 88.92%였지만 선택적 반응 점수는 0.254였다. 같은 조건에서 Sparse CCA는 61.78%와 0.309였다. 검색 개선이 두 개념의 분리를 뜻하지 않았다.</li>
<li><strong>음수 계수를 허용하지 않는 조건은 이 생성 모형에서 개념 분리에 도움이 되었다.</strong> 평균을 유지한 InfoNCE에서 비음수 제약을 추가하면, 학습 상관계수 0.9에서 선택적 반응 점수가 0.559에서 0.975로 높아졌다. 이미지 질의 Recall@5는 74.64%와 74.54%로 비슷했다. 다만 비음수 조건은 초기화에도 절댓값을 적용하므로, 제약과 초기화의 효과를 완전히 분리한 비교는 아니다.</li>
<li><strong>두 개념이 항상 함께 등장하면 비음수 조건도 분리하지 못했다.</strong> 비음수 InfoNCE의 선택적 반응 점수는 학습 상관계수 0.9에서 0.975였으나 1에서 0.340이었다. 생성 구조를 알려준 참고값은 두 경우 모두 1이었다. 분리 가능한 조합은 존재하지만 이번 학습은 이를 찾지 못했다.</li></ul>
<p><strong>다음 검증은 비음수 생성 구조에 결과가 얼마나 의존하는지 확인하는 것이다.</strong> 현재 생성기는 각 개념만 나타내는 입력 좌표를 여덟 개씩 제공한다. 이 전용 좌표를 줄이거나 없애고, 동일한 비음수 초기화에서 음수 계수 허용 여부만 비교하면 어떤 가정이 분리를 가능하게 했는지 더 분명하게 확인할 수 있다. 이번 결과를 실제 SAE에 그대로 적용할 수 있다는 주장은 아직 하지 않는다.</p>
<h2>동시 등장 정도만 바꾼 생성 조건</h2>
<p>관심 개념 C₁과 C₂는 각각 30% 확률로 존재한다. C₂가 확률 ρ로 C₁을 복사하고 나머지 경우 독립적으로 생성되게 하여, 두 개념의 이론적 상관계수를 0, 0.5, 0.9, 1로 바꿨다. 둘 중 하나만 존재할 확률은 각각 42%, 21%, 4.2%, 0%다. 두 관심 개념의 크기는 0 또는 1로 고정했다. 따라서 상관계수 1에서는 크기 차이를 이용해 두 개념을 구별할 수도 없다.</p>
<p>별도로 여섯 개의 공통 요인을 넣었다. 각 요인은 35% 확률로 켜지고, 양수 크기는 평균 1인 Gamma(4, 0.25)를 따른다. 각 모달리티는 이 여덟 요인의 서로 다른 비음수 선형 조합과 독립적인 희소 잡음으로 생성했다. 입력 좌표는 128개이며, 각 요인은 16개 입력에 영향을 준다. 그중 8개는 해당 요인만 나타내고 8개는 다른 요인과 공유한다. 이렇게 다른 공통 요인을 넣으면 두 관심 개념을 분리하지 않고도 검색을 잘할 수 있다.</p>
<p>자료 8,192쌍으로 학습하고 별도 2,048쌍의 InfoNCE로 모델을 선택했다. 모든 방법은 출력 좌표 8개, 조합당 최대 16개 입력, P=I를 사용했다. InfoNCE는 Sparse CCA에서 시작해 최대 60에폭 학습했다. 한 에폭은 학습 자료를 한 번 모두 사용하는 과정이다. 비음수 조건은 초기 계수의 절댓값에서 시작했다. 3개 난수 조건에서 생성 행렬·표본을 바꿔 반복했다. 각 난수 조건 안에서는 같은 생성 행렬과 난수를 유지하고 두 관심 개념의 동시 등장 정도만 바꿨다.</p>
<h2>선택적 반응을 직접 측정한 방법</h2>
<p>다른 잠재 요인과 관측 잡음을 유지한 채 한 요인만 1만큼 바꿨을 때, 조합한 activation이 얼마나 변하는지 계산했다. 선형 조합이므로 생성 행렬과 학습 행렬에서 이 변화량을 정확히 계산할 수 있다. 코사인 정규화 전의 조합 값을 평가한다. 조합 c에서 요인 j를 바꾼 반응을 Rⱼ꜀라고 하면 그 요인에 대한 반응 비중은 다음과 같다.</p>
<img src="selectivity.svg" style="width:230px" alt="특정 요인에 대한 제곱 반응을 모든 여덟 요인의 제곱 반응 합으로 나눈 값">
<p>관심 요인 하나에만 반응하면 1이다. 두 요인에 같은 크기로 반응하고 나머지에는 반응하지 않으면 0.5다. 양쪽에서 같은 방향으로 반응하는 대응 좌표 중 이미지와 텍스트의 더 낮은 비중을 사용한다. 두 관심 개념에는 서로 다른 두 좌표를 배정하여 평균을 계산했다. 이 배정은 평가만을 위한 것이며 학습 행렬이나 P를 바꾸지 않는다. 이는 이번 합성 실험에서 정의한 진단 지표이며 일반적인 해석 가능성 점수는 아니다.</p>
<p>정답 생성 구조를 알려준 참고값은 각 요인만 나타내는 8개 입력을 골라 복원한 값이다. 허용한 희소성 안에서도 선택적 반응 점수 1인 조합이 존재함을 확인하기 위한 비교다. 주석 없는 학습 방법이거나 검색 성능의 수학적 상한은 아니다.</p>
<h2>검색과 선택적 반응의 변화</h2>
<img src="comparison.png" alt="상관계수에 따른 선택적 반응과 두 평가 분포의 검색 성능">
<p class="note">선은 3회 평균이며 음영은 실행 간 표준편차다. 학습과 같은 조건의 별도 2,048쌍, 그리고 두 관심 개념을 독립적으로 재조합한 별도 2,048쌍에서 각각 검색했다. 마지막 조건은 학습에 드문 조합 또는 없었던 조합에서의 평가다.</p>
<div class="scroll"><table><thead><tr><th>학습 상관계수</th><th>방법</th><th>선택적 반응</th><th>C₁만 있음·C₂만 있음 구별 AUROC</th><th>같은 분포 이미지 질의 R@5</th><th>같은 분포 텍스트 질의 R@5</th><th>재조합 이미지 질의 R@5</th><th>재조합 텍스트 질의 R@5</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div>
<p>AUROC는 독립 재조합 평가 자료 중 관심 개념 하나만 있는 표본에서 계산했다. 선택적 반응 평가에서 정한 각 대표 좌표를 그대로 사용하며, 별도 분류기를 학습하지 않았다. 두 관심 개념과 두 모달리티의 네 AUROC를 평균했다.</p>
<details><summary>양방향 Recall@1·5·10 전체 결과와 표준편차</summary><div class="scroll"><table><thead><tr><th>학습 상관계수</th><th>방법</th><th>평가 분포</th><th>이미지 질의 R@1</th><th>R@5</th><th>R@10</th><th>텍스트 질의 R@1</th><th>R@5</th><th>R@10</th></tr></thead><tbody>'''+''.join(full)+'''</tbody></table></div></details>
<h2>해석 범위</h2><p>생성 요인과 개입을 알고 있는 작은 선형 모형에서 수행한 실험이다. 이 결과만으로 실제 SAE에서 인간이 말하는 개념을 복원했다고 볼 수 없다. 상관계수 1에서는 학습 표본에 두 관심 개념이 달라지는 사례가 없지만, 평가 생성기는 이를 별도로 바꿀 수 있다. 학습 관측만으로 두 의미를 분리하려면 추가 가정이나 감독 신호가 필요하다는 문제를 확인하는 조건이다.</p>
<p>표준화 통계와 Sparse CCA 행렬은 학습 자료에서만 계산했다. 모델 선택에 잠재 요인 정답이나 시험 자료를 사용하지 않았다. InfoNCE의 희소 제약은 매 갱신 후 계수를 제거하는 근사법으로 적용했으며 전역 최적해를 보장하지 않는다. 모든 새 방법은 같은 Sparse CCA 계열 초기화에서 출발하므로 초기화의 영향을 별도로 분리하지 않았다.</p></html>'''
(root/'report.html').write_text(doc)
for rho in rhos:
    print('rho',rho)
    for r in rows:
        if r['rho']==rho:
            print(r['method'],{k:round(r['metrics'][k]['mean'],4) for k in ['selectivity','exclusive_auc','iid_image_to_text_R5','recombined_image_to_text_R5']})
