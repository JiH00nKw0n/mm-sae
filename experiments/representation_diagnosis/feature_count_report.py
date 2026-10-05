"""Compare subset sizes without mixing presence and removal classification."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from experiments.representation_diagnosis.report import write_rows, save_figure
from mm_sae.analysis.evaluation import distribution

PANELS = [
    ('presence', 'presence', '원본에서 범주 유무 분류'),
    ('removal', 'removal', '가림 구별을 학습한 원본·가림 분류'),
    ('presence', 'removal', '원본에서 학습한 분류기의 원본·가림 구별'),
]


def make_report(out):
    out = Path(out)
    options = json.loads((out / 'config.json').read_text())
    method_name = {
        'paired_mean_drop': '평균 활성값 감소량',
        'single_logistic': '단일 좌표 분류 손실',
        'probe_attribution': '개념 분류기 기여도',
    }.get(Path(options['feature_run']).name, '원본·가림 AUROC')
    rows = [r for p in sorted((out / 'jobs').glob('*.json'))
            for r in json.loads(p.read_text())['records']]
    root = out / 'report'
    root.mkdir(exist_ok=True)
    write_rows(root / 'category_metrics.csv', rows)
    counts = sorted([5, *options['feature_counts']])
    reps = ['selected_five' if n == 5 else f'selected_{n}' for n in counts] + ['all_sae']
    summaries = []
    excluded = set(options.get('report_exclude_categories', []))
    for task, metric, _ in PANELS:
        if task not in options['objectives']:
            continue
        for kind in ['object', 'background', 'all']:
            for cohort in ['all_available', 'common_with_rq1']:
                available = [r for r in rows if r['trained_on'] == task
                             and r['evaluated_on'] == metric and r['representation'] in reps + ['embedding']
                             and (kind == 'all' or r['kind'] == kind)
                             and (cohort == 'all_available' or r['category_id'] not in excluded)]
                valid_sets = [{r['category_id'] for r in available
                               if r['representation'] == rep and r['auroc'] is not None}
                              for rep in reps + ['embedding']]
                common = set.intersection(*valid_sets)
                for rep in reps + ['embedding']:
                    rr = [r for r in available if r['representation'] == rep and r['category_id'] in common]
                    summaries.append(dict(trained_on=task, evaluated_on=metric, kind=kind,
                                          cohort=cohort, representation=rep,
                                          category_ids=sorted(common),
                                          **distribution([r['auroc'] for r in rr])))
    write_rows(root / 'summary.csv', summaries)
    examples = [r for r in rows if r['category_id'] in [105,181,124,153,154,158,118,17]
                and r['representation'] in reps + ['embedding']]
    write_rows(root / 'case_examples.csv', examples)
    font = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = 'NanumGothic'
    plt.rcParams.update({'font.family':family, 'axes.unicode_minus':False, 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    panels = [p for p in PANELS if p[0] in options['objectives']]
    fig, axes = plt.subplots(2, len(panels), figsize=(6*len(panels), 8.8), squeeze=False)
    colors = ['#FFADAD', '#FFD6A5', '#FDFFB6', '#CAFFBF', '#9BF6FF']
    for row, kind in enumerate(['object', 'background']):
        for col, (task, metric, title) in enumerate(panels):
            ax = axes[row, col]
            summary = [r for r in summaries if r['trained_on']==task and r['evaluated_on']==metric
                       and r['kind']==kind and r['cohort']=='common_with_rq1']
            common = set(summary[0]['category_ids'])
            values = [[r['auroc'] for r in rows if r['category_id'] in common
                       and r['representation']==rep and r['trained_on']==task
                       and r['evaluated_on']==metric] for rep in reps]
            bp = ax.boxplot(values, tick_labels=[str(n) for n in counts]+['전체\n4,096개'],
                            patch_artist=True, whis=(0,100), showfliers=False, widths=.55)
            for patch,color in zip(bp['boxes'],colors):
                patch.set_facecolor(color)
            means = [np.mean(v) for v in values]
            ax.plot(np.arange(1,len(reps)+1),means,'o-',color='#475569',lw=1,ms=4,label='범주별 AUROC 평균')
            embedding = next(r['mean'] for r in summary if r['representation']=='embedding')
            ax.axhline(embedding,color='#64748b',ls='--',lw=1.2,label=f'원본 임베딩 평균 {embedding:.3f}')
            ax.axhline(.5,color='#94a3b8',ls=':',lw=.8)
            ax.set(ylim=(.25,1.025),xlabel='분류기에 사용할 수 있는 고정 특징 수',ylabel='검증 AUROC')
            ax.set_title(title+'\n'+('물체' if kind=='object' else '배경')+f' {len(common)}개',fontsize=12)
            ax.grid(axis='y',alpha=.15)
            ax.legend(fontsize=9,loc='lower right',frameon=False)
    fig.suptitle(method_name + ' 기준의 특징 수별 분류 성능',fontsize=21,fontweight='bold')
    fig.tight_layout(rect=(0,.035,1,.94))
    fig.text(.02,.014,'상자 = 1·3사분위수, 중앙선 = 중앙값, 수염 = 최솟값·최댓값 · 범주별 특징 순위와 데이터 분할 고정',fontsize=11)
    save_figure(fig,root,'feature_count_distributions')
    lines=[f'# {method_name} 기준으로 특징 수를 늘린 분류기 비교', '',
           '이미지 인코더와 SAE를 고정하고, 학습용 자료에서 해당 선택 기준으로 만든 기존 순위의 상위 특징을 사용했다. 각 범주에서 상위 5개는 상위 8개에, 상위 8개는 상위 16개에 포함된다. 원본 이미지 94,630장에서 분류기를 학습하고 23,657장에서 규제 강도를 조정했으며 val 이미지 5,000장에서 평가했다. 기존 5개와 전체 특징 분류기는 입력 파일의 해시와 통계 설정을 확인해 재사용했다.', '',
           '이미지 한 장에서 활성화되는 좌표는 여전히 최대 8개다. 선택하는 16개·64개는 서로 다른 이미지에서 사용할 수 있는 고정 좌표의 목록을 뜻하며 SAE의 활성화 개수를 바꾸지 않았다.', '',
           '표와 그림은 기존 상관행렬과 동일한 물체 80개와 배경 90개를 사용한다. 학습은 배경 ceiling-tile을 포함한 171개 전체 범주에서 수행했으며 전체 범주 집계도 CSV에 함께 저장했다.', '']
    for task, metric, title in panels:
        lines += [f'## {title}', '', '| 범주 | 통계 | '+ ' | '.join([f'특징 {n}개' for n in counts]+['전체 특징','원본 임베딩'])+' |',
                  '| --- | --- | '+ ' | '.join(['---:']*(len(reps)+1))+' |']
        for kind in ['object','background']:
            for stat,label in [('mean','평균'),('median','중앙값')]:
                ss=[next(r for r in summaries if r['trained_on']==task and r['evaluated_on']==metric
                         and r['kind']==kind and r['cohort']=='common_with_rq1' and r['representation']==rep)
                    for rep in reps+['embedding']]
                lines.append('| '+('물체 80개' if kind=='object' else '배경 90개')+' | '+label+' | '+
                             ' | '.join(f'{r[stat]:.3f}' for r in ss)+' |')
        lines.append('')
    lines += ['원본의 범주 유무 분류 성능이 높아도 배경 자체 대신 함께 등장한 객체를 사용했을 가능성은 남는다. 원본·가림 분류를 직접 학습한 조건은 흰색 편집 흔적도 활용할 수 있다. 원본에서 학습한 분류기의 원본·가림 AUROC는 같은 이미지에서 점수가 감소한 비율과 다르며, 이미지별 감소 비율은 범주별 CSV에 별도로 저장했다.', '']
    (root / 'README.md').write_text('\n'.join(lines))
