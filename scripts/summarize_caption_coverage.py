"""Report visible dictionary mentions, including an equal-one-caption baseline diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np

from mm_sae.data.index import Index
from mm_sae.io import atomic_json, write_csv


def summarize(root, split, label, first_only=False):
    index = Index(root, split)
    maskable = np.zeros_like(index.mentions)
    full = np.zeros_like(index.mentions)
    for row, caption in enumerate(index.captions):
        for cid in caption['mask_token_positions']:
            maskable[row, index.columns[int(cid)]] = True
        for cid in caption.get('full_concept_ids', caption['concept_ids']):
            full[row, index.columns[int(cid)]] = True
    keep = np.arange(len(index.captions))
    if first_only:
        # The index orders captions by image ID then caption ID.
        _, keep = np.unique(index.parents, return_index=True)
    parents, maskable, full = index.parents[keep], maskable[keep], full[keep]
    counts = np.bincount(parents, minlength=len(index.images))
    per_image_weight = 1 / counts[parents]
    rows = []
    for j, cid in enumerate(index.concept_ids):
        eligible = index.presence[:, j]
        present = eligible[parents]
        den = float(np.sum(per_image_weight[present]))
        mentioned_images = np.unique(parents[maskable[:, j]])
        rows.append(dict(condition=label, category_id=cid, kind='object' if cid < 91 else 'background',
            images=len(index.images), captions=len(parents), category_present_images=int(eligible.sum()),
            full_caption_mentions=int(full[:, j].sum()), visible_caption_mentions=int(maskable[:, j].sum()),
            image_weighted_visible_mention_rate=(
                float(per_image_weight[present & maskable[:, j]].sum()/den) if den else None),
            present_images_with_any_visible_mention=int(eligible[mentioned_images].sum())))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--split', default='train2017')
    args = parser.parse_args()
    rows = summarize(args.source, args.split, 'replacement')
    rows += summarize(args.baseline, args.split, 'original_all_captions')
    rows += summarize(args.baseline, args.split, 'original_lowest_id_caption', first_only=True)
    out = args.source / 'caption_coverage'
    out.mkdir(exist_ok=True)
    write_csv(out / 'categories.csv', rows)
    summaries = []
    for label in sorted({r['condition'] for r in rows}):
        for kind in ['object', 'background']:
            selected = [r for r in rows if r['condition'] == label and r['kind'] == kind]
            rates = [r['image_weighted_visible_mention_rate'] for r in selected
                     if r['image_weighted_visible_mention_rate'] is not None]
            summaries.append(dict(condition=label, kind=kind,
                visible_mentions=sum(r['visible_caption_mentions'] for r in selected),
                full_mentions=sum(r['full_caption_mentions'] for r in selected),
                mean_category_visible_mention_rate=float(np.mean(rates)) if rates else None))
    atomic_json(out / 'summary.json', dict(summary=summaries,
        definition='Dictionary mentions with valid mask token positions in the actual CLIP input',
        baseline_note='Lowest-ID caption is a descriptive count control, not an SAE training control',
        source=str(args.source.resolve()), baseline=str(args.baseline.resolve())))
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
