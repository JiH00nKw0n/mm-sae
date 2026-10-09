"""Frozen-CLIP localization audit. Pixel annotations are evaluation-only.

All choices are recorded before inference. No model training or adaptive tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

from mm_sae.data.text import CaptionEditor, load_concepts
from mm_sae.models.text_masking import clip_pool_positions, project_at_positions


METHODS = ['patch_token', 'patch_phrase', 'local_token', 'local_phrase']
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def transform_mask(processor, value):
    p = processor.image_processor
    value = value[None]
    value = p.resize(value, size=p.size, resample=Image.Resampling.NEAREST,
                     input_data_format='channels_first', data_format='channels_first')
    value = p.center_crop(value, size=p.crop_size,
                          input_data_format='channels_first', data_format='channels_first')
    return value[0]


def expand_patch(values):
    return np.repeat(np.repeat(values.reshape(7, 7), 32, 0), 32, 1)


def scores_to_mask(scores):
    # Stable tie breaking by patch index, never by ground-truth area or mask.
    selected = np.argsort(-scores, kind='stable')[:5]
    mask = np.zeros(49, bool)
    mask[selected] = True
    return mask, expand_patch(mask)


def localization(scores, target):
    patches, pred = scores_to_mask(scores)
    intersection = int((pred & target).sum())
    centers = target[16::32, 16::32].reshape(-1)
    return {
        'point_hit': float(centers[np.argmax(scores)]),
        'precision': intersection / int(pred.sum()),
        'recall': intersection / int(target.sum()),
        'iou': intersection / int((pred | target).sum()),
        'selected_patches': np.flatnonzero(patches).tolist(),
    }


@torch.inference_mode()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases').mkdir(exist_ok=True)
    cfg = {
        'model': 'openai/clip-vit-base-patch32', 'revision': REVISION,
        'split': 'COCO val2017', 'sample_images': args.images, 'seed': 20261009,
        'sampling': 'uniform image sample; one uniformly sampled caption per image; before masks are read',
        'target_phrases': 'existing frozen caption dictionary; 80 object categories; source caption words only',
        'image_annotations': 'COCO-Stuff semantic masks, category union of instances, evaluation only',
        'mask_budget_patches': 5, 'patch_grid': [7, 7], 'input_pixels': [224, 224],
        'phrase_prompt': 'a photo of {literal caption span}.',
        'token_embedding': 'mean projected final hidden states of tokens overlapping source character spans',
        'local_variant': 'MaskCLIP-style last-block value projection plus residual and MLP; earlier blocks unchanged',
        'image_mask_fill': 'CLIP channel mean (zero in normalized input)',
        'text_mask': 'delete matched caption character spans; re-encode changed text; no new mask token',
        'random_control': '5 independently sampled 5-patch subsets per image-phrase; same controls for all methods',
        'matching_control': 'compare image-mask change with correct text deletion versus another object deletion in same caption',
        'training': 'none', 'new_retrieval_or_sae_matching_results': False,
        'caveats': ['Caption dictionary is not human-reviewed.', 'No exact single-concept intervention is assumed.',
                    'Global CLIP change cosine is a diagnostic, not independent semantic ground truth.',
                    'Changing text deletion can change grammar and positions.',
                    'Semantic masks combine all same-category instances.', 'Validation data have been used in prior exploration.'],
        'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save_json(out / 'protocol.json', cfg)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    model = CLIPModel.from_pretrained(cfg['model'], revision=REVISION,
                                     local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    proc = CLIPProcessor.from_pretrained(cfg['model'], revision=REVISION,
                                         local_files_only=True, use_fast=False)
    tok = CLIPTokenizerFast.from_pretrained(cfg['model'], revision=REVISION, local_files_only=True)
    index = Path(args.index)
    images = json.loads((index / 'images.json').read_text())
    caps = json.loads((index / 'captions.json').read_text())
    by_image = defaultdict(list)
    for c in caps:
        by_image[c['image_row']].append(c)
    concepts = [c for c in load_concepts(None) if c.id <= 89]
    names = {c.id: c.name for c in concepts}
    editor = CaptionEditor(concepts)
    rng = np.random.default_rng(cfg['seed'])
    chosen = rng.choice(len(images), args.images, replace=False)
    population = [{'image_row': int(i), 'image': images[i],
                   'caption': by_image[int(i)][int(rng.integers(len(by_image[int(i)])))]} for i in chosen]
    save_json(out / 'population.json', population)
    records, skipped = [], Counter()
    begin = time.monotonic()
    checks = {'global_image_max_abs_error': 0., 'global_text_max_abs_error': 0.,
              'all_token_similarities_finite': True, 'annotations_used_to_rank_patches': False}

    def text_features(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
        return F.normalize(model.get_text_features(input_ids=enc.input_ids,
                                                   attention_mask=enc.attention_mask), dim=-1)

    for image_no, entry in enumerate(population):
        info, caption = entry['image'], entry['caption']
        text = caption['text']
        _, spans, _, _ = editor.analyze(caption['caption_id'], text)
        if not spans:
            skipped['caption_without_object_dictionary_match'] += 1
            continue
        photo = Image.open(info['image']).convert('RGB')
        pixels = proc(images=photo, return_tensors='pt').pixel_values.cuda()
        vis = model.vision_model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
        normal = F.normalize(model.visual_projection(model.vision_model.post_layernorm(
            vis.last_hidden_state[:, 1:])), dim=-1)[0]
        pre = vis.hidden_states[-2]
        block = model.vision_model.encoder.layers[-1]
        local = pre + block.self_attn.out_proj(block.self_attn.v_proj(block.layer_norm1(pre)))
        local = local + block.mlp(block.layer_norm2(local))
        local = F.normalize(model.visual_projection(model.vision_model.post_layernorm(local[:, 1:])), dim=-1)[0]
        global_image = F.normalize(model.visual_projection(vis.pooler_output), dim=-1)[0]
        enc = tok(text, return_offsets_mapping=True, return_tensors='pt', truncation=True, max_length=77)
        offsets = enc.pop('offset_mapping')[0].numpy()
        enc = enc.to('cuda')
        txt = model.text_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, return_dict=True)
        token_proj = model.text_projection(txt.last_hidden_state)[0]
        global_text = F.normalize(model.text_projection(txt.pooler_output), dim=-1)[0]
        valid_positions = list(range(1, int(clip_pool_positions(enc.input_ids, model.config.text_config.eos_token_id)[0])))
        all_token = F.normalize(token_proj[valid_positions], dim=-1)
        all_scores = (normal @ all_token.T).cpu().numpy()
        checks['all_token_similarities_finite'] &= bool(np.isfinite(all_scores).all())
        if image_no < 3:
            a = F.normalize(model.get_image_features(pixel_values=pixels), dim=-1)[0]
            b = text_features([text])[0]
            checks['global_image_max_abs_error'] = max(checks['global_image_max_abs_error'], float((a-global_image).abs().max()))
            checks['global_text_max_abs_error'] = max(checks['global_text_max_abs_error'], float((b-global_text).abs().max()))

        phrase_items = []
        for category, intervals in spans.items():
            positions = [j for j, (s,e) in enumerate(offsets)
                         if j in valid_positions and any(s < b and a < e for a,b in intervals)]
            if not positions or any(b > max(offsets[:, 1]) for a,b in intervals):
                skipped['truncated_target'] += 1
                continue
            phrase = text[intervals[0][0]:intervals[0][1]]
            # Group repeated mentions for text deletion and spatial category-union evaluation.
            deleted = text
            for s,e in sorted(intervals, reverse=True):
                deleted = deleted[:s] + deleted[e:]
            deleted = ' '.join(deleted.split())
            phrase_items.append((category, intervals, positions, phrase, deleted))
        if not phrase_items:
            continue
        prompt_features = text_features([f'a photo of {p[3]}.' for p in phrase_items])
        deleted_text_features = text_features([p[4] for p in phrase_items])
        text_deltas = global_text[None] - deleted_text_features
        # Make and save all prediction maps BEFORE loading any segmentation labels.
        pred_maps = []
        for pno, (_, _, positions, _, _) in enumerate(phrase_items):
            word = F.normalize(token_proj[positions].mean(0), dim=0)
            phrase = prompt_features[pno]
            pred_maps.append({
                'patch_token': (normal @ word).cpu().numpy(),
                'patch_phrase': (normal @ phrase).cpu().numpy(),
                'local_token': (local @ word).cpu().numpy(),
                'local_phrase': (local @ phrase).cpu().numpy(),
            })
        label = transform_mask(proc, np.asarray(Image.open(info['mask'])))
        assert label.shape == (224, 224)
        rgb = np.asarray(proc.image_processor(images=photo, do_normalize=False, do_rescale=False,
                                              return_tensors='np').pixel_values[0]).transpose(1,2,0).astype(np.uint8)
        case_dir = out / 'cases' / str(info['image_id'])
        case_dir.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(case_dir / 'input.png')
        np.savez_compressed(case_dir / 'all_tokens.npz', scores=all_scores,
                            tokens=np.array(tok.convert_ids_to_tokens(enc.input_ids[0, valid_positions].tolist())))

        for pno, (category, intervals, positions, phrase, deleted) in enumerate(phrase_items):
            target = label == category
            if not target.any():
                skipped['mentioned_object_absent_in_visible_annotation'] += 1
                continue
            maps = pred_maps[pno]
            masks = [scores_to_mask(maps[m])[1] for m in METHODS]
            crng = np.random.default_rng(cfg['seed'] + info['image_id'] * 191 + category)
            random_masks = []
            for repeat in range(5):
                selected = crng.choice(49, 5, replace=False)
                patch_mask = np.zeros(49, bool); patch_mask[selected] = True
                random_masks.append(expand_patch(patch_mask))
            all_masks = masks + random_masks
            masked_pixels = pixels.repeat(len(all_masks), 1, 1, 1)
            for row, mask in enumerate(all_masks):
                masked_pixels[row, :, torch.as_tensor(mask, device='cuda')] = 0
            masked_image_features = F.normalize(model.get_image_features(pixel_values=masked_pixels), dim=-1)
            image_deltas = global_image[None] - masked_image_features
            delta_cos = F.normalize(image_deltas, dim=-1) @ F.normalize(text_deltas, dim=-1).T
            target_drop = (global_image @ prompt_features[pno] - masked_image_features @ prompt_features[pno]).cpu().numpy()
            row = {'image_id': info['image_id'], 'caption_id': caption['caption_id'], 'caption': text,
                   'category_id': category, 'category_name': names[category], 'phrase': phrase,
                   'intervals': intervals, 'token_positions': positions, 'deleted_caption': deleted,
                   'area_fraction': float(target.mean()), 'methods': {}, 'random': {},
                   'other_caption_targets': [p[3] for j,p in enumerate(phrase_items) if j != pno]}
            for mi, name in enumerate(METHODS):
                values = localization(maps[name], target)
                values['target_similarity_drop'] = float(target_drop[mi])
                values['delta_cosine'] = float(delta_cos[mi, pno])
                alternatives = [j for j in range(len(phrase_items)) if j != pno]
                values['delta_correct_minus_other'] = (float(delta_cos[mi,pno] - delta_cos[mi,alternatives].mean())
                                                         if alternatives else None)
                values['delta_target_rank1'] = (float(delta_cos[mi].argmax() == pno) if alternatives else None)
                row['methods'][name] = values
            centers = target[16::32,16::32].reshape(-1)
            row['random'] = {'point_hit': float(centers.mean()), 'precision': float(target.mean()),
                             'recall': 5/49,
                             'iou': float(np.mean([(m & target).sum() / (m | target).sum() for m in random_masks])),
                             'target_similarity_drop': float(target_drop[len(METHODS):].mean()),
                             'delta_cosine': float(delta_cos[len(METHODS):, pno].mean()),
                             'delta_correct_minus_other': (float((delta_cos[len(METHODS):,pno] - delta_cos[len(METHODS):, alternatives].mean(-1)).mean()) if alternatives else None),
                             'delta_target_rank1': (float((delta_cos[len(METHODS):].argmax(-1)==pno).float().mean()) if alternatives else None)}
            np.savez_compressed(case_dir / f'{category}.npz', target=target,
                                **{name: v for name,v in maps.items()}, random_mask=random_masks[0])
            records.append(row)
        if (image_no + 1) % 25 == 0:
            progress = {'images_processed': image_no+1, 'images_total': args.images,
                        'evaluated_mentions': len(records), 'elapsed_seconds': time.monotonic()-begin}
            save_json(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)

    summary = {'sampled_images': args.images, 'evaluated_images': len({r['image_id'] for r in records}),
               'evaluated_mentions': len(records), 'categories': len({r['category_id'] for r in records}),
               'skipped': dict(skipped), 'elapsed_seconds': time.monotonic()-begin,
               'checks': checks, 'methods': {}, 'by_category': {}, 'by_area': {}}

    def summarize(rows):
        result = {}
        for method in METHODS + ['random']:
            values = [r['random'] if method == 'random' else r['methods'][method] for r in rows]
            result[method] = {}
            for metric in values[0]:
                if metric == 'selected_patches':
                    continue
                available = [v[metric] for v in values if v[metric] is not None]
                result[method][metric] = {'mean': float(np.mean(available)) if available else None,
                                         'n': len(available)}
        return result

    summary['methods'] = summarize(records)
    for cat in sorted({r['category_id'] for r in records}):
        rows = [r for r in records if r['category_id'] == cat]
        summary['by_category'][names[cat]] = {'n': len(rows), 'methods': summarize(rows)}
    for label, low, high in [('under_5pct',0,.05), ('5_to_20pct',.05,.2), ('over_20pct',.2,1.01)]:
        rows = [r for r in records if low <= r['area_fraction'] < high]
        summary['by_area'][label] = {'n': len(rows), 'methods': summarize(rows)} if rows else {'n': 0}
    save_json(out / 'records.json', records)
    save_json(out / 'summary.json', summary)
    save_json(out / 'complete.json', {'complete': True, 'elapsed_seconds': summary['elapsed_seconds']})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['by_category','by_area']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', default='/mnt/working/mm-sae/runs/elice-rq1-lexicon2/index/val2017')
    parser.add_argument('--images', type=int, default=500)
    run(parser.parse_args())
