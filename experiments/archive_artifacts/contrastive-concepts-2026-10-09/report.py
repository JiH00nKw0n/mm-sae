"""Build a self-contained numerical report from completed exploratory runs."""
from pathlib import Path
import json, csv, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
folders=[root/'results']+[root/f'results-seed{s}' for s in (1,2)]
names={
 'sparse_cca_16':'기존 Sparse CCA',
 'sparse_cca_unitnorm':'Sparse CCA의 열 길이만 보정',
 'nce_centered':'InfoNCE, 평균 제거',
 'nce_uncentered':'InfoNCE, 평균 유지',
 'nce_nonnegative':'InfoNCE, 평균 유지, 비음수 계수',
}
all_rows=[]
retrieval_rows=[]
semantic_rows=[]
for key,label in names.items():
    ds=[json.loads((p/f'{key}-evaluation.json').read_text()) for p in folders if (p/f'{key}-evaluation.json').exists()]
    if key.startswith('sparse_cca'):ds=ds[:1]
    vals=[]
    for d in ds:
        for direction,v in d['retrieval'].items():
            ranks=np.asarray(v['ranks'])
            for k,value in v['recall'].items():
                assert abs(np.mean(ranks<=int(k))-value)<1e-12
        vals.append([100*d['retrieval'][direction]['recall'][str(k)] for direction in ('image_to_text','text_to_image') for k in (1,5,10)])
    ar=np.asarray(vals)
    mean,std=ar.mean(0),ar.std(0,ddof=1) if len(ar)>1 else np.zeros(6)
    sem=np.array([[d['positive_agreement']['summary']['agree_at1_count'],d['signed_agreement']['summary']['agree_at1_count'],
         d['positive_representative_reuse']['image_a']['categories_in_shared_representatives'],
         d['positive_representative_reuse']['text_b']['categories_in_shared_representatives']] for d in ds])
    def cells(v):return ''.join(f'<td>{x:.2f}</td>' for x in v)
    retrieval_rows.append(f'<tr><th>{label}</th>'+''.join(f'<td>{x:.2f}'+(f' ± {s:.2f}' if len(ds)>1 else '')+'</td>' for x,s in zip(mean,std))+'</tr>')
    semantic_rows.append(f'<tr><th>{label}</th>'+cells(sem.mean(0))+'</tr>')
    all_rows.append(dict(method=key,label=label,repeats=len(ds),recall_mean=mean.tolist(),recall_std=std.tolist(),
        semantic_mean=sem.mean(0).tolist(),semantic_per_seed=sem.tolist()))
save=root/'summary.json';save.write_text(json.dumps(all_rows,ensure_ascii=False,indent=2))

fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
colors=['#5a82c2','#d99255','#8673b7']
for key,label,color in zip(list(names)[2:],['Centered InfoNCE','Uncentered InfoNCE','Nonnegative InfoNCE'],colors):
    for si,p in enumerate(folders):
        if not (p/f'{key}-history.json').exists():continue
        records=json.loads((p/f'{key}-history.json').read_text())
        for ax,metric in zip(axes,['train_loss','tune_loss']):
            rr=[r for r in records if r[metric] is not None]
            ax.plot([r['epoch'] for r in rr],[r[metric] for r in rr],color=color,alpha=.9 if si==0 else .35,label=label if si==0 else None)
for ax,title in zip(axes,['Training paired loss','Held-out paired loss']):
    ax.set(title=title,xlabel='Epoch',ylabel='Symmetric InfoNCE')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
axes[1].legend(frameon=False,fontsize=8)
fig.savefig(root/'loss-curves.png',dpi=170)
plt.close(fig)

protocol=json.loads((root/'results/protocol.json').read_text())
dist=json.loads((root/'results/distribution.json').read_text())
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>희소 대조학습과 개념 대응 탐색</title>
<style>body{max-width:1150px;margin:50px auto;padding:0 28px;color:#222;font:17px/1.75 system-ui}h1{font-size:30px}h2{margin-top:40px;font-size:23px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}thead{background:#f3f4f6}img{max-width:100%}.note{color:#555;font-size:15px}strong{color:#9c3535}</style>
<h1>희소 대조학습과 개념 대응을 비교한 초기 실험</h1>
<p>CC3M으로 학습한 SAE를 고정하고 COCO의 이미지·캡션 짝으로 조합 행렬을 학습했다. 범주 주석은 학습에 사용하지 않았다. 이 실험은 검색 개선이 개념별 대응 개선으로 이어지는지 살펴본 탐색이다.</p>
<h2>실험 조건</h2><p>모든 방법은 공통 좌표 256개와 각 열당 최대 16개 activation을 사용한다. 개념 후보 사이의 대응은 항등행렬로 고정했다. 부분 순열은 학습하지 않았다. InfoNCE 조건은 기존 Sparse CCA 계수에서 시작하며, 비음수 조건은 그 절댓값에서 시작했다. 조합 행렬의 각 열 길이는 1로 맞췄다.</p>
<p>COCO train2017의 학습 이미지 94,630장을 사용했다. 매 반복에서 이미지마다 연결된 캡션 하나를 무작위로 골라 같은 이미지가 한 배치에 중복되지 않게 했다. 배치 크기는 512, 온도는 0.07, Adam 학습률은 0.003이며 최대 20회 반복했다. 매 갱신 후 열마다 절댓값이 큰 계수 최대 16개만 남겼다. 이 방식은 희소 최적화의 근사법이며 전역 최적해를 보장하지 않는다.</p>
<p>학습에 사용하지 않은 train2017 이미지 2,048장과 고정 캡션에서 InfoNCE가 가장 낮은 모델을 선택했다. 모델 선택에 COCO val2017 또는 범주 주석을 사용하지 않았다. 반복 실행은 학습 표본 순서와 캡션 추출 난수만 바꿨으며 초기 Sparse CCA 행렬은 같다.</p>
<h2>양방향 검색 성능</h2><p>COCO val2017 이미지 5,000장과 캡션 25,014개에서 평가했다. 각 질의에서 정답이 상위 K개 안에 하나 이상 있는 비율을 Recall@K로 계산했다. 표의 값은 백분율이며 반복 실행의 평균이다. 기준 Sparse CCA는 재학습하지 않았다.</p>
<table><thead><tr><th>방법</th><th>이미지 질의 R@1</th><th>R@5</th><th>R@10</th><th>텍스트 질의 R@1</th><th>R@5</th><th>R@10</th></tr></thead><tbody>'''+''.join(retrieval_rows)+'''</tbody></table>
<h2>범주별 대표 조합의 대응</h2><p>80개 객체 범주 중 독립된 이미지·텍스트 평가 집단에 충분한 양성 표본이 있는 50개 범주를 평가했다. 각 범주에 대해 이미지와 텍스트에서 개념 유무 AUROC가 가장 높은 조합을 독립적으로 골랐다. 두 대표가 같은 좌표이면 일치로 계산했다. 텍스트의 정답은 캡션 자체의 언급 여부가 아니라 연결된 이미지의 주석이다.</p>
<p>검색 표는 평균과 실행 간 표준편차를, 아래 개수 표는 실행 간 평균을 표시한다. 양의 방향 평가는 activation이 클수록 개념이 있다고 보는 경우다. 부호 허용 평가는 낮은 값이 개념을 나타내는 경우도 허용하되 양쪽 부호까지 일치해야 한다. 공유 범주 수는 다른 범주와 같은 대표 조합을 고른 범주의 개수다. 공유가 줄었다는 사실만으로 의미가 잘 분리됐다고 판단할 수는 없다.</p>
<table><thead><tr><th>방법</th><th>양의 방향 일치 /50</th><th>부호 허용 일치 /50</th><th>이미지에서 대표를 공유한 범주 /50</th><th>텍스트에서 대표를 공유한 범주 /50</th></tr></thead><tbody>'''+''.join(semantic_rows)+'''</tbody></table>
<h2>학습 곡선</h2><img src="loss-curves.png" alt="학습 및 별도 짝 자료에서의 InfoNCE 곡선"><p class="note">색이 같은 선은 같은 방법의 반복 실행이다. 20회에서 여전히 손실이 감소하면 수렴한 결과라고 해석하지 않는다. 이번 비교는 동일한 제한된 학습 예산에서의 탐색이다.</p>
<h2>Bernoulli–Exponential 가정 검사</h2><p>학습 표본에서 양수 값이 100개 이상인 좌표를 검사했다. 양수 값의 표준편차를 평균으로 나눈 값의 중앙값은 이미지 '''+f"{dist['image']['median_positive_cv']:.3f}"+'''이고 텍스트 '''+f"{dist['text']['median_positive_cv']:.3f}"+'''이다. 지수분포의 모집단 값은 1이다. 단순한 지수분포를 양수 activation에 바로 가정할 근거는 약하다. 이 검사는 양수 주변분포를 확인한 것이며 모달리티 사이의 결합분포를 검증하지 않는다.</p>
<h2>개념의 정의와 해석 범위</h2><p><a href="https://proceedings.mlr.press/v80/kim18d.html">TCAV</a>는 사람이 제공한 예시 집합으로 개념을 정의하고, 그 예시를 구별하는 방향을 추정한다. <a href="https://arxiv.org/abs/2311.03658">The Linear Representation Hypothesis and the Geometry of Large Language Models</a>는 다른 요인과 구분하여 바꿀 수 있는 변동 요인으로 개념을 다룬다. <a href="https://arxiv.org/abs/2406.01506">The Geometry of Categorical and Hierarchical Concepts in Large Language Models</a>는 이를 범주와 계층 관계로 확장한다. 이 언어모델 이론이 SAE 집합에 그대로 적용되는 것은 아니다.</p>
<p>이번 탐색에서 측정한 것은 검색과 범주 대표의 일치 및 반복 사용이다. 다른 속성을 유지하며 특정 개념만 바꾸었을 때 반응하는지는 측정하지 않았다. 따라서 이번 결과를 인과적인 개념 분리의 증거로 볼 수 없다.</p>
<p class="note">전체 수치와 반복별 결과는 같은 폴더의 summary.json 및 results 폴더에 보존했다. 원본 결과의 순위에서 Recall@1·5·10을 다시 계산해 일치 여부를 검사했다.</p></html>'''
(root/'report.html').write_text(doc)
print(json.dumps(all_rows,ensure_ascii=False,indent=2))
