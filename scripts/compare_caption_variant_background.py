"""Reproduce meeting slide 12 and compare caption variants on the same category set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mm_sae.io import atomic_json, sha256, write_csv


def panel(root):
    representatives = json.loads((root / 'representatives.json').read_text())
    raw = dict(np.load(root / 'panel.npz'))
    ids = sorted(map(int, representatives))
    matrix = np.full((len(ids), len(ids)), np.nan)
    for row, cid in enumerate(ids):
        for column, other in enumerate(ids):
            image, text = representatives[str(cid)]['image'], representatives[str(other)]['text']
            if image is not None and text is not None and raw['valid_image'][image] and raw['valid_text'][text]:
                matrix[row, column] = raw['C'][image, text]
    return ids, matrix


def analyze(label, ids, matrix):
    diagonal = np.diag(matrix)
    assert np.isfinite(matrix).all()
    background = np.array(ids) >= 91
    ranks, pairs, same, categories = [], [], [], []
    for cid, value in zip(ids, diagonal):
        categories.append(dict(condition=label, category_id=cid,
                               kind='background' if cid >= 91 else 'object', same_score=float(value)))
    for kind, keep in [('object', ~background), ('background', background)]:
        values = diagonal[keep]
        same.append(dict(condition=label, kind=kind, categories=int(keep.sum()),
            mean=float(values.mean()), median=float(np.median(values)), std=float(values.std()),
            min=float(values.min()), q1=float(np.quantile(values, .25)),
            q3=float(np.quantile(values, .75)), max=float(values.max())))
    for direction, directed in [('image', matrix), ('text', matrix.T)]:
        higher = directed > diagonal[:, None]
        np.fill_diagonal(higher, False)
        for kind, keep in [('all', np.ones(len(ids), bool)), ('object', ~background),
                           ('background', background)]:
            count = int(higher[keep].any(axis=1).sum())
            ranks.append(dict(condition=label, direction=direction, anchor_kind=kind,
                categories=int(keep.sum()), categories_with_higher=count,
                percent=100 * count / int(keep.sum()), rivals_per_category=len(ids)-1))
        for name, anchor, rival in [('object_object', ~background, ~background),
                                    ('object_background', ~background, background),
                                    ('background_object', background, ~background),
                                    ('background_background', background, background)]:
            population = anchor[:, None] & rival[None, :]
            np.fill_diagonal(population, False)
            count = int(higher[population].sum())
            pairs.append(dict(condition=label, direction=direction, kind=name,
                pairs=int(population.sum()), higher_pairs=count, percent=100*count/population.sum()))
    return ranks, pairs, same, categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--variant', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--font', default='AppleGothic')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = {r['id']: r['name'] for r in json.loads((args.variant / 'dataset.json').read_text())['concepts']}
    inputs = {label: panel(root) for label, root in [('original', args.baseline), ('detailed', args.variant)]}
    common = sorted(set.intersection(*(
        {cid for cid, value in zip(ids, np.diag(matrix)) if np.isfinite(value)}
        for ids, matrix in inputs.values())))
    excluded = sorted(set.union(*(set(ids) for ids, _ in inputs.values())) - set(common))
    ranking, pair_rows, same_scores, category_rows = [], [], [], []
    historical = []
    for label, (ids, matrix) in inputs.items():
        if np.isfinite(matrix).all():
            historical += analyze(label, ids, matrix)[0]
        rows = [ids.index(cid) for cid in common]
        a, b, c, d = analyze(label, common, matrix[np.ix_(rows, rows)])
        ranking += a
        pair_rows += b
        same_scores += c
        category_rows += d
    for filename, rows in [('ranking.csv', ranking), ('pair_rates.csv', pair_rows),
                           ('same_scores.csv', same_scores), ('category_scores.csv', category_rows)]:
        write_csv(args.out / filename, rows)
    atomic_json(args.out / 'summary.json', dict(
        common_category_ids=common, excluded_categories=[dict(id=i, name=names[i]) for i in excluded],
        definition='One representative per category selected by original-vs-masked pooled AUROC on all train images; each condition uses its own train captions for correlation.',
        ranking=ranking, pair_rates=pair_rows, same_scores=same_scores,
        original_full_population_ranking=historical,
        sources={str(root / name): sha256(root / name) for root in [args.baseline, args.variant]
                 for name in ['panel.npz', 'representatives.json', 'run.json']},
        limitations=['Caption content and caption count both change.',
                     'These are descriptive train-set statistics, not independent validation.',
                     'Different category labels are not human-verified semantic errors.']))
    plt.rcParams.update({'font.family': args.font, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'path', 'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), layout='constrained')
    for axis, direction, title in zip(axes, ['image', 'text'],
                                      ['이미지 범주를 기준으로 비교', '텍스트 범주를 기준으로 비교']):
        for offset, condition, label, color in [(-.19, 'original', '기존 캡션', '#FFADAD'),
                                                 (.19, 'detailed', '긴 캡션', '#9BF6FF')]:
            rows = [next(r for r in ranking if r['direction']==direction and r['condition']==condition
                         and r['anchor_kind']==kind) for kind in ['all', 'object', 'background']]
            bars = axis.bar(np.arange(3)+offset, [r['percent'] for r in rows], width=.35,
                            color=color, edgecolor='#5b6770', linewidth=.6, label=label)
            axis.bar_label(bars, labels=[f"{r['percent']:.1f}%\n{r['categories_with_higher']}/{r['categories']}"
                                        for r in rows], fontsize=10, padding=4)
        axis.set(ylim=(0, 104), xticks=range(3), xticklabels=['전체 범주', '물체 범주', '배경 범주'],
                 ylabel='같은 범주보다 높은 상대가 하나라도 있는 범주의 비율 (%)', title=title)
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle(f'범주당 특징 1개 · 두 조건에서 동일한 {len(common)}개 범주 비교', fontsize=17)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(args.out / f'slide12_caption_comparison.{suffix}', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7.6, 5), layout='constrained')
    values = [[r['same_score'] for r in category_rows if r['condition']==label and r['kind']=='background']
              for label in ['original','detailed']]
    bp = axis.boxplot(values, tick_labels=['기존 캡션', '긴 캡션'], patch_artist=True,
                      whis=(0,100), showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='#750014', markeredgecolor='#750014'))
    for box, color in zip(bp['boxes'], ['#FFADAD','#9BF6FF']):
        box.set_facecolor(color)
    axis.set(title=f'동일한 배경 범주 {len(values[0])}개의 이미지·텍스트 상관점수', ylabel='같은 범주의 coactivation correlation')
    axis.spines[['top','right']].set_visible(False)
    axis.grid(axis='y',alpha=.2)
    axis.set_axisbelow(True)
    axis.text(.5,-.16,'상자는 사분위수, 수염은 최솟값·최댓값, 마름모는 평균',transform=axis.transAxes,ha='center',fontsize=10)
    for suffix in ['png','pdf','svg']:
        fig.savefig(args.out / f'background_same_score_boxplot.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(args.out),common_categories=len(common),excluded=excluded)))


if __name__ == '__main__':
    main()
