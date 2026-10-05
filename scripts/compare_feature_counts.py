"""Aggregate the same subset-size probes across independently fixed rankings."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import yaml

from experiments.representation_diagnosis.feature_count_report import PANELS
from experiments.representation_diagnosis.report import save_figure, write_rows
from mm_sae.analysis.data import save_json
from mm_sae.analysis.evaluation import distribution
from mm_sae.io import sha256

NAMES = {'pooled_auroc':'원본·가림 AUROC', 'paired_mean_drop':'평균 활성값 감소량',
         'single_logistic':'단일 좌표 분류 손실', 'probe_attribution':'개념 분류기 기여도'}
COLORS = ['#aa5555','#a77c39','#4c8c64','#437f9f']


def compare(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    roots = {k:(config_path.parent/v).resolve() for k,v in config['runs'].items()}
    out = (config_path.parent/config['output']).resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows, hashes = [], {}
    for method,root in roots.items():
        state = json.loads((root/'progress.json').read_text())['state']
        if state != 'completed':
            raise ValueError(f'Run is not completed: {method}, {state}')
        source = root/'report/category_metrics.csv'
        hashes[str(source)] = sha256(source)
        for r in csv.DictReader(source.open()):
            r['category_id'] = int(r['category_id'])
            for key in ['auroc','paired_decrease','mean_paired_change']:
                r[key] = float(r[key]) if r.get(key) else None
            rows.append(dict(method=method,**r))
    lookup = {(r['method'],r['category_id'],r['representation'],r['trained_on'],r['evaluated_on']):r for r in rows}
    counts = config['feature_counts']
    reps = ['selected_five' if n==5 else f'selected_{n}' for n in counts]
    summaries = []
    excluded = set(config.get('exclude_categories',[]))
    all_ids = sorted({r['category_id'] for r in rows if r['category_id'] not in excluded})
    for task,metric,_ in PANELS:
        common = [c for c in all_ids if all(lookup.get((m,c,p,task,metric),{}).get('auroc') is not None
                  for m in roots for p in reps+['all_sae','embedding'])]
        # All-feature and embedding references must be identical across selection criteria.
        for cid in common:
            for p in ['all_sae','embedding']:
                z=[lookup[m,cid,p,task,metric]['auroc'] for m in roots]
                np.testing.assert_allclose(z,z[0],rtol=0,atol=1e-12)
        for kind in ['object','background']:
            ids=[c for c in common if (c<91)==(kind=='object')]
            for method in roots:
                for rep in reps+['all_sae','embedding']:
                    rr=[lookup[method,c,rep,task,metric] for c in ids]
                    summaries.append(dict(method=method,kind=kind,trained_on=task,evaluated_on=metric,
                                          representation=rep,category_ids=ids,
                                          **distribution([r['auroc'] for r in rr])))
    write_rows(out/'category_metrics.csv',rows)
    write_rows(out/'summary.csv',summaries)
    save_json(out/'provenance.json',dict(config=config,inputs=hashes))
    font=Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family=font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family='NanumGothic'
    plt.rcParams.update({'font.family':family,'axes.unicode_minus':False,'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(17,8.5))
    for row,kind in enumerate(['object','background']):
        for col,(task,metric,title) in enumerate(PANELS):
            ax=axes[row,col]
            ss=[r for r in summaries if r['kind']==kind and r['trained_on']==task and r['evaluated_on']==metric]
            for j,method in enumerate(roots):
                z=[next(r['mean'] for r in ss if r['method']==method and r['representation']==p) for p in reps]
                ax.plot(np.arange(len(counts)),z,marker=['o','s','^','D'][j],color=COLORS[j],label=NAMES[method],lw=2)
            full=next(r['mean'] for r in ss if r['representation']=='all_sae')
            emb=next(r['mean'] for r in ss if r['representation']=='embedding')
            ax.axhline(full,color='#475569',ls='--',lw=1.2,label=f'SAE 전체 평균 {full:.3f}')
            ax.axhline(emb,color='#94a3b8',ls=':',lw=1.5,label=f'원본 임베딩 평균 {emb:.3f}')
            ax.set(xticks=np.arange(len(counts)),xticklabels=[str(n) for n in counts],ylim=(.48,1.015),
                   xlabel='선택한 고정 특징 수',ylabel='범주별 AUROC 평균')
            ax.set_title(title+'\n'+('물체' if kind=='object' else '배경')+f" {ss[0]['n']}개",fontsize=11)
            ax.grid(axis='y',alpha=.15)
            ax.legend(handles=ax.get_legend_handles_labels()[0][-2:],fontsize=8,loc='lower right',frameon=False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles[:4],labels[:4],loc='lower center',ncol=4,frameon=False)
    fig.suptitle('네 가지 특징 선택 기준과 특징 수의 비교',fontsize=20,fontweight='bold')
    fig.tight_layout(rect=(0,.065,1,.93))
    save_figure(fig,out,'selection_criteria_and_counts')
    fig,axes=plt.subplots(2,2,figsize=(12.5,9))
    for ax,method in zip(axes.flat,roots):
        ss=[r for r in summaries if r['method']==method and r['kind']=='background'
            and r['trained_on']=='presence' and r['evaluated_on']=='presence']
        ids=ss[0]['category_ids']
        data=[[lookup[method,c,p,'presence','presence']['auroc'] for c in ids]
                                        for p in reps+['all_sae']]
        boxes=ax.boxplot(data,tick_labels=[str(n) for n in counts]+['전체\n4,096개'],patch_artist=True,
                         whis=(0,100),showfliers=False)
        for patch,color in zip(boxes['boxes'],['#FFADAD','#FFD6A5','#FDFFB6','#CAFFBF','#9BF6FF']):
            patch.set_facecolor(color)
        ax.plot(np.arange(1,len(data)+1),[np.mean(x) for x in data],'o',color='#334155',ms=4,label='평균')
        ax.axhline(.5,color='#94a3b8',ls=':',lw=.8)
        ax.set(title=NAMES[method],ylim=(.3,1.02),ylabel='원본의 배경 유무 분류 AUROC',xlabel='선택한 고정 특징 수')
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('배경 90개에서 특징 수에 따른 성능 분포',fontsize=20,fontweight='bold')
    fig.tight_layout(rect=(0,.03,1,.94))
    fig.text(.04,.013,'상자 = 1·3사분위수, 중앙선 = 중앙값, 수염 = 최솟값·최댓값, 검은 점 = 평균',fontsize=11)
    save_figure(fig,out,'background_presence_distributions')
    lines=['# 네 가지 특징 선택 기준에서 선택 개수를 늘린 결과','',
           '각 기준의 기존 범주별 특징 순위를 고정하고 상위 5·8·16·64개를 비교했다. 이미지 인코더와 SAE를 재학습하지 않았다. 각 분류기의 학습·조정·검증 분할과 규제 후보를 동일하게 유지했다. 이미지 한 장에서 활성화되는 좌표 수는 계속 최대 8개다. 아래 평균은 물체 80개와 배경 90개에 각각 동일 비중을 준 값이며, 모든 조건에 같은 범주를 사용했다.','']
    for task,metric,title in PANELS:
        lines += ['## '+title,'']
        for kind,label in [('object','물체 80개'),('background','배경 90개')]:
            lines += [label+'의 범주별 평균 AUROC를 비교했다.','',
                      '| 특징 선택 기준 | '+' | '.join(f'{n}개' for n in counts)+' |',
                      '| --- | '+' | '.join(['---:']*len(counts))+' |']
            for m in roots:
                z=[next(r['mean'] for r in summaries if r['method']==m and r['kind']==kind and
                        r['trained_on']==task and r['evaluated_on']==metric and r['representation']==p) for p in reps]
                lines.append('| '+NAMES[m]+' | '+' | '.join(f'{v:.3f}' for v in z)+' |')
            ref={p:next(r['mean'] for r in summaries if r['kind']==kind and r['trained_on']==task and
                       r['evaluated_on']==metric and r['representation']==p) for p in ['all_sae','embedding']}
            lines += ['',f"전체 SAE 특징의 평균 AUROC는 {ref['all_sae']:.3f}, 원본 임베딩은 {ref['embedding']:.3f}다.",'']
    lines += ['원본 범주 유무의 높은 AUROC는 함께 등장한 객체로 예측했을 가능성을 배제하지 않는다. 가림 구별을 직접 학습한 분류기는 편집 흔적에 반응할 수 있다. 원본에서 학습한 존재 분류기의 원본·가림 구별 AUROC와 이미지별 점수 감소 비율도 구분해야 한다. 이 비교는 기존에 관찰한 검증 자료에서 수행한 탐색 분석이다.','']
    (out/'README.md').write_text('\n'.join(lines))
    print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    compare(parser.parse_args().config)
