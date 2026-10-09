from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'results'
CS={'coco-coco':('COCO2017','COCO2017'),'cc3m-coco':('CC3M','COCO2017'),'cc3m-cc3m':('CC3M','CC3M')}
KS=[8,16,32,64,128,256];MASSES=[90,95,99]
read=lambda p:json.loads(p.read_text())
get=lambda c,m:read(DATA/c/(m+'-test.json'))
diag=lambda c,m:read(DATA/c/(m+'-diagnostics.json'))
assert read(DATA/'complete.json')['completed']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def label(c):
    a,b=CS[c];return f'SAE: {a} / mapping: {b}'
def finish(fig,name):
    fig.savefig(ROOT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(ROOT/(name+'.svg'),bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for ax,c in zip(axes,CS):
    with np.load(DATA/c/'cca_full.npz') as z:
        for side,color,ls in [('image','#ce7070','-'),('text','#527fac','--')]:
            a=np.sort(z[side]**2,axis=0)[::-1];cum=a.cumsum(0)/a.sum(0);x=np.arange(1,len(a)+1)
            ax.plot(x,np.median(cum,axis=1),color=color,ls=ls,label=side.title())
            lo,hi=np.quantile(cum,[.25,.75],axis=1);ax.fill_between(x,lo,hi,color=color,alpha=.12)
    ax.set_xscale('log');ax.set_ylim(0,1.02);ax.set_title(label(c),fontsize=11);ax.set_xlabel('Number of largest coefficients retained')
    ax.axhline(.9,color='#888',ls=':',lw=1);ax.set_ylabel('Retained squared coefficient mass');ax.grid(alpha=.13);ax.legend(frameon=False)
finish(fig,'coefficient-concentration')

fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for ax,c in zip(axes,CS):
    with np.load(DATA/c/'cca_full-correlations.npz') as z:
        for k,col,ls,lab in [('fit_regularized','#999','-.','Training regularized score'),('fit_pearson','#ce7070','-','Training Pearson'),('test_pearson','#527fac','--','COCO val2017 Pearson')]:ax.plot(np.arange(1,257),z[k],color=col,ls=ls,label=lab)
    ax.set_ylim(0,1.02);ax.set_xlim(1,256);ax.set_title(label(c),fontsize=11);ax.set_xlabel('CCA component index');ax.set_ylabel('Correlation / regularized score');ax.grid(alpha=.13);ax.legend(frameon=False,fontsize=8)
finish(fig,'component-correlations')

fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for ax,c in zip(axes,CS):
    for direction,col,ls,lab in [('image_to_text','#ce7070','-','Image query'),('text_to_image','#527fac','--','Text query')]:
        ys=[100*get(c,f'keep_{k}')['retrieval'][direction]['recall']['5'] for k in KS]
        ax.plot(KS,ys,color=col,ls=ls,marker='o',label=lab)
        full=100*get(c,'cca_full')['retrieval'][direction]['recall']['5'];ax.axhline(full,color=col,ls=':',alpha=.7)
    ax.set_xscale('log',base=2);ax.set_xticks(KS,[str(x) for x in KS]);ax.set_ylim(0,55);ax.set_title(label(c),fontsize=11);ax.set_xlabel('Retained inputs per CCA component');ax.set_ylabel('Recall@5 (%)');ax.grid(alpha=.13);ax.legend(frameon=False)
finish(fig,'pruning-retrieval')

styles=(ROOT.parent/'set-cca-2026-10-09/report.html').read_text().split('<style>')[1].split('</style>')[0]
parts=[f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CCA의 큰 계수 개수와 작은 계수 제거 결과</title><style>{styles}img{{width:100%;height:auto}}</style><main>',
'''<small>2026-10-09 · 기존 CCA 256개 조합을 고정한 분석</small><h1>CCA의 작은 계수는 얼마나 제거할 수 있을까?</h1>
<p class="lead"><strong>큰 계수 약 13–24개가 계수 제곱합의 절반을 차지했지만, 90%를 남기려면 중앙값 기준 약 98–311개가 필요했다.</strong> 16개만 남기면 세 조건 모두 검색 성능이 낮아졌다. 99%를 유지하면 원래 CCA와 가까운 검색 성능을 보였다.</p>
<p>각 조합은 표준화한 SAE activation 좌표의 가중합이다. 이미지 행렬 A와 텍스트 행렬 B의 열 하나가 각각 u와 v 조합 하나를 만든다. 아래 개수는 표본 개수가 아니라, 그 열에서 남기는 activation 좌표 수다. 기존 계수의 절댓값 순서로 제거했고, 남은 계수의 값과 부호는 그대로 뒀다. 가중치를 다시 학습하거나 조합 크기를 재조정하지 않았다.</p>
<h2>큰 계수는 몇 개에 집중돼 있었는가?</h2><p>아래는 256개 조합에서 필요한 좌표 수의 중앙값이다. 14.5처럼 소수점이 있는 것은 조합별 개수가 정수여도 중앙값이 두 개수의 평균일 수 있기 때문이다. ‘90%’는 계수 제곱합 중 남긴 비율이며, 설명한 분산이나 정보량의 90%라는 뜻은 아니다.</p>
<div class="scroll"><table><tr><th>SAE 학습 자료</th><th>대응 학습 자료</th><th>모달리티</th><th>사용 가능한 좌표 수</th><th>제곱합 50%에 필요한 개수</th><th>제곱합 90%에 필요한 개수</th><th>제곱합 95%에 필요한 개수</th><th>제곱합 99%에 필요한 개수</th></tr>''']
for c,(sae,mapping) in CS.items():
    for side,z in read(DATA/c/'concentration.json').items():
        nums=[z['counts_for_squared_coefficient_mass'][str(p)]['median'] for p in [.5,.9,.95,.99]]
        parts.append(f'<tr><td>{sae}</td><td>{mapping}</td><td>{"이미지" if side=="image" else "텍스트"}</td><td>{z["input_coordinates"]}</td>'+''.join(f'<td>{n:g}</td>' for n in nums)+'</tr>')
parts.append('''</table></div><img src="coefficient-concentration.png" alt="계수가 큰 순서대로 남겼을 때 계수 제곱합의 누적 비율"><p class="note">곡선은 256개 조합의 중앙값이며 음영은 25–75백분위다. 이미지 조합은 붉은 실선, 텍스트 조합은 파란 점선이다. activation끼리 상관이 있어 작은 계수가 중복 성분을 상쇄할 수도 있으므로, 계수 크기만으로 제거해도 되는지를 판단할 수는 없다.</p>
<h2>각 u와 v의 상관은 1에 가까웠는가?</h2><p><strong>일부 조합의 상관은 0.95를 넘었지만 모든 조합이 1에 가까운 것은 아니었다.</strong> 아래 Pearson 상관은 고정된 조합을 각각의 자료에서 다시 중심화해서 계산했다. 학습 자료와 COCO val2017을 구분했다. 이미지·텍스트가 같은 쌍이 되도록 이미지 표현을 캡션 수만큼 반복해 계산했다.</p><table><tr><th>SAE 학습 자료</th><th>대응 학습 자료</th><th>학습 Pearson 평균</th><th>학습 Pearson 최소–최대</th><th>평가 Pearson 평균</th><th>평가 Pearson 최소–최대</th><th>평가 상관 ≥0.9 /256</th></tr>''')
for c,(sae,mapping) in CS.items():
    d=diag(c,'cca_full');a=d['fit_pearson'];b=d['test_pearson']
    parts.append(f'<tr><td>{sae}</td><td>{mapping}</td><td>{a["mean"]:.3f}</td><td>{a["minimum"]:.3f}–{a["maximum"]:.3f}</td><td>{b["mean"]:.3f}</td><td>{b["minimum"]:.3f}–{b["maximum"]:.3f}</td><td>{d["test_above_09"]}</td></tr>')
parts.append('''</table><img src="component-correlations.png" alt="256개 CCA 조합의 학습 상관과 평가 상관"><p class="note">가로축은 원래 학습한 CCA 조합의 순서다. 평가 상관 순서로 재정렬하지 않았다. 실선은 학습 Pearson 상관, 파란 점선은 COCO val2017 Pearson 상관, 회색 선은 안정화 항이 포함된 학습 점수다.</p>
<h2>실제로 최적화한 목적함수 값은 얼마였는가?</h2><p>기본 CCA는 같은 번호의 조합 사이 상관을 더한 값을 높인다. 조합 하나의 상관은 최대 1이고 256개의 합은 최대 256이다. 현재 구현은 공분산에 0.01배 단위행렬을 더한 제약을 썼으므로, 실제 학습 목적은 순수 Pearson 상관의 합과 조금 다르다.</p>
<p class="math">maximize tr(AᵀC<sub>XY</sub>B), &nbsp; Aᵀ(C<sub>XX</sub> + 0.01I)A = I, &nbsp; Bᵀ(C<sub>YY</sub> + 0.01I)B = I</p>
<p>C<sub>XX</sub>와 C<sub>YY</sub>는 학습 자료에서 표준화한 activation의 모달리티 내부 공분산이며, C<sub>XY</sub>는 두 모달리티 사이 공분산이다. tr은 대각 원소의 합이다. 표준화했으므로 C<sub>XY</sub>의 각 원소는 원래 activation 좌표 사이의 coactivation correlation과 같다.</p>
<table><tr><th>SAE 학습 자료</th><th>대응 학습 자료</th><th>실제 학습 목적값의 합 /256</th><th>조합당 평균 점수</th><th>안정화 분모를 뺀 Pearson 합</th></tr>''')
for c,(sae,mapping) in CS.items():
    d=diag(c,'cca_full');parts.append(f'<tr><td>{sae}</td><td>{mapping}</td><td>{d["fit_objective_trace"]:.3f}</td><td>{d["fit_regularized_pair_correlation"]["mean"]:.3f}</td><td>{d["fit_pearson_sum"]:.3f}</td></tr>')
parts.append('''</table><p>계수를 제거한 행렬은 위의 직교·정규화 제약을 그대로 만족하지 않는다. 제거 후 Pearson 상관과 검색 성능은 계산할 수 있지만, 그 값을 다시 최적화된 CCA 목적값이라고 부르지는 않는다. 기본 CCA 정의는 <a href="https://people.eecs.berkeley.edu/~jordan/papers/688.pdf">A Probabilistic Interpretation of Canonical Correlation Analysis의 3.1절</a>을 참고할 수 있다.</p>
<h2>작은 계수를 제거하면 검색과 매칭이 어떻게 달라졌는가?</h2><img src="pruning-retrieval.png" alt="CCA 각 조합에서 남긴 activation 개수에 따른 양방향 Recall@5"><p class="note">붉은 실선은 이미지 질의, 파란 점선은 텍스트 질의다. 가로로 그은 점선은 계수를 전부 사용한 원래 CCA의 성능이다. 모든 조건에서 출력 조합은 256개로 유지했다.</p>''')
summary=[]
for c,(sae,mapping) in CS.items():
    methods=['cca_full']+[f'keep_{k}' for k in KS]+[f'mass_{p}' for p in MASSES]+['sparse_cca_16'];rows=[]
    for m in methods:
        z=get(c,m);vals=[100*z['retrieval'][d]['recall'][str(k)] for d in ['image_to_text','text_to_image'] for k in [1,5,10]]+[z[a]['summary']['agree_at1_count'] for a in ['positive','signed']]
        rows.append((m,vals));summary.append(dict(condition=c,method=m,values=vals))
    ranks=[sorted(set(v[j] for _,v in rows),reverse=True) for j in range(8)]
    parts.append(f'<h3>SAE 학습은 {sae}, 대응 학습은 {mapping}을 사용했다</h3><div class="scroll"><table><tr><th rowspan="2">계수 처리 방법</th><th rowspan="2">남긴 좌표 수 중앙값<br>이미지 / 텍스트</th><th colspan="3">이미지 질의 검색 (%)</th><th colspan="3">텍스트 질의 검색 (%)</th><th colspan="2">대표 조합 매칭 /42</th></tr><tr><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th><th>높은 활성만</th><th>반응 방향 허용</th></tr>')
    for m,vals in rows:
        if m=='cca_full':name='원래 CCA · 계수 전부 유지'
        elif m.startswith('keep_'):name=f'CCA · 큰 계수 {m.split("_")[1]}개만 유지'
        elif m.startswith('mass_'):name=f'CCA · 계수 제곱합 {m.split("_")[1]}% 유지'
        else:name='참고 · 좌표와 계수를 다시 학습한 Sparse CCA 16개'
        counts='16 / 16' if m=='sparse_cca_16' else ' / '.join(f'{diag(c,m)["nonzero_counts"][s]["median"]:g}' for s in ['image','text'])
        parts.append(f'<tr><td>{name}</td><td>{counts}</td>')
        for j,v in enumerate(vals):
            t=f'{v:.2f}' if j<6 else str(v)
            if v==ranks[j][0]:t=f'<b>{t}</b>'
            elif len(ranks[j])>1 and v==ranks[j][1]:t=f'<u>{t}</u>'
            parts.append(f'<td>{t}</td>')
        parts.append('</tr>')
    parts.append('</table></div>')
parts.append('''<p>각 표의 열마다 최고값을 굵게, 두 번째로 높은 서로 다른 값을 밑줄로 표시했다. ‘계수 제곱합 유지’는 조합마다 남기는 개수가 다르다. 표의 개수는 그 중앙값이다. Sparse CCA는 계수 제거 후 고정한 모델이 아니라, 좌표 선택과 가중치를 다시 학습했던 기존 비교 모델이다.</p>
<h2>평가 기준과 이번 분석의 범위</h2><p>검색은 COCO val2017 이미지 5,000장과 캡션 25,014개로 평가했다. Recall@K는 상위 K개 안에 정답이 하나 이상 포함된 질의의 비율이다. 집합의 의미는 범주별 대표 조합 번호의 일치로 평가했다. 첫 이미지 집단의 객체 주석과 다른 집단의 캡션 자체에 나타난 범주 언급으로 조합을 각각 골랐다. 80개 객체 중 양성 표본이 각 집단에 50개 이상인 42개가 평가 대상이다.</p>
<p>‘높은 활성만’은 높은 조합 값으로 개념 존재를 구별한 경우다. ‘반응 방향 허용’은 낮은 조합 값으로 존재를 구별하는 경우도 허용하되, 대표 번호와 반응 방향이 양쪽에서 모두 같아야 성공으로 셌다. 이 개수는 의미가 하나로 분리됐음을 보장하지 않는다. 주석을 사용해 제거할 계수나 임계값을 고르지 않았다.</p>
<p><strong>제거한 계수를 전부 불필요한 잡음이라고 해석할 수는 없다.</strong> 각 입력은 학습 평균과 표준편차로 표준화했지만 입력끼리의 상관은 남아 있다. 실제로 상위 16개만 남긴 조건은 같은 16개 제한으로 다시 학습한 Sparse CCA보다 검색 성능이 낮았다. 작은 계수를 제거할 수 있는지와, 소수 좌표로 새 조합을 학습할 수 있는지는 다른 질문이다.</p>
<p><a href="summary.json">검색·매칭 수치</a> · <a href="run.py">분석 코드</a> · <a href="verification.json">검증 내역</a></p></main></html>''')
(ROOT/'report.html').write_text(''.join(parts));(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

checks=[]
for c in CS:
    with np.load(DATA/c/'cca_full.npz') as z:full={s:z[s].copy() for s in ['image','text']}
    previous=read(ROOT.parent/'set-cca-2026-10-09/results'/c/'cca_256-test.json')
    baseline=get(c,'cca_full')
    for direction in ['image_to_text','text_to_image']:assert previous['retrieval'][direction]['recall']==baseline['retrieval'][direction]['recall']
    for m in ['keep_'+str(k) for k in KS]+['mass_'+str(p) for p in MASSES]:
        with np.load(DATA/c/(m+'.npz')) as z:
            for s in full:
                w=z[s];assert w.shape==full[s].shape and w.shape[1]==256
                mask=w!=0;assert np.array_equal(w[mask],full[s][mask])
                if m.startswith('keep_'):assert np.all(mask.sum(0)==min(int(m.split('_')[1]),len(w)))
                else:assert np.all((w*w).sum(0)/(full[s]*full[s]).sum(0)>=int(m.split('_')[1])/100-1e-12)
        checks.append(dict(condition=c,method=m,passed=True))
for row in summary:
    assert 0<=row['values'][0]<=row['values'][1]<=row['values'][2]<=100
    assert 0<=row['values'][3]<=row['values'][4]<=row['values'][5]<=100
(ROOT/'verification.json').write_text(json.dumps(dict(passed=True,pruned_models=checks,baseline_recall_exactly_matches_previous_evaluation=True,output_components=256),indent=2))
print('Built report and verified',len(checks),'pruned models')
