"""Evaluate fixed concept readouts on a chosen cached split, without refitting them."""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.feature_sets.config import validate
from experiments.feature_sets.context import Context
from experiments.feature_sets.correspondence import analyze
from experiments.feature_sets.report import boxes, finish
from mm_sae.analysis.data import verify_source
from mm_sae.config import load_config
from mm_sae.io import write_csv
from mm_sae.progress import ProgressReporter, stage_progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[1, 5])
    args = parser.parse_args()
    config = load_config(args.config)
    ctx = Context(config.output, validate(config))
    splits = {s.name: s for s in [ctx.data.train, ctx.data.test]}
    if args.split not in splits:
        raise ValueError('Requested split is not present in the configured study')
    if not set(args.counts) <= set(ctx.counts):
        raise ValueError('Feature counts must have fitted readouts in the source study')
    root = ctx.out / f'correspondence_{args.split}'
    root.mkdir(parents=True, exist_ok=True)
    ctx.o['multi_feature_rq1']['legacy_n1_reproduction'] = False
    with ProgressReporter(root, ['verify', 'analyze', 'plot'], [], 10):
        with stage_progress('verify'):
            verify_source(ctx.data.source, ctx.out, [ctx.data.train.name, ctx.data.test.name], ctx.ids)
        with stage_progress('analyze'):
            analyze(ctx, split=splits[args.split], folder=root.name,
                    counts=args.counts, conditions=['learned'])
        with stage_progress('plot'):
            plt.rcParams.update({'font.family': ctx.o['report']['font'], 'axes.unicode_minus': False,
                                 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
            distributions, summaries = [], []
            for n in args.counts:
                result = json.loads((root / f'learned_{n}.json').read_text())
                distributions += [dict(feature_count=n, **r) for r in result['distributions']]
                summaries += [dict(feature_count=n, **r) for r in result['summary']]
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                for axis, direction, title in zip(axes, ['image_to_text', 'text_to_image'],
                                                  ['이미지를 기준으로 비교', '텍스트를 기준으로 비교']):
                    rows = [r for r in result['distributions'] if r['direction'] == direction]
                    boxes(axis, rows, [(lambda r: r['condition'] == 'original', '원본'),
                                       (lambda r: r['condition'] == 'controlled', '동시 등장 관계 통제')],
                          ['#FFADAD', '#9BF6FF'])
                    axis.set_title(title)
                fig.suptitle(f'{args.split}, 특징 {n}개, 고정한 분류기 점수의 상관')
                finish(fig, root, f'controlled_scores_n{n}')
            write_csv(root / 'distributions.csv', distributions)
            write_csv(root / 'summary.csv', summaries)


if __name__ == '__main__':
    main()
