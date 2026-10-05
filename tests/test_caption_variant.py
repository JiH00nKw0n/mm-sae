"""Caption replacement must retrain text while retaining identical image computations."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from safetensors.torch import load_file

from mm_sae.config import load_config
from mm_sae.data.index import replacement_captions
from mm_sae.io import RunStore
from mm_sae.models.text_masking import plan_masks
from experiments.rq1.run import run


def test_replacement_requires_coverage_and_allocates_disjoint_ids(tmp_path):
    path = tmp_path / 'captions.json'
    rows = [{'image_id': 2, 'caption': 'a dog'}, {'image_id': 3, 'caption': 'snow'}]
    path.write_text(json.dumps({'annotations': rows}))
    a = replacement_captions(path, {2, 3}, 0)
    b = replacement_captions(path, {2, 3}, 1)
    assert {r['id'] for r in a}.isdisjoint(r['id'] for r in b)
    assert [r['caption'] for r in a] == ['a dog', 'snow']
    with pytest.raises(ValueError, match='missing'):
        replacement_captions(path, {2, 3, 4}, 0)
    with pytest.raises(ValueError, match='unknown image'):
        replacement_captions(path, {2}, 0)
    path.write_text(json.dumps({'annotations': rows + [rows[0]]}))
    with pytest.raises(ValueError, match='Duplicate'):
        replacement_captions(path, {2, 3}, 0)


def test_visible_repetition_survives_truncation_but_partial_word_does_not():
    text = 'a dog and dogs'
    offsets = [(0, 0), (0, 1), (2, 5), (6, 9), (0, 0)]
    special = [1, 0, 0, 0, 1]
    positions, unavailable = plan_masks(text, offsets, special,
                                        {17: [(2, 5), (10, 14)], 1: [(10, 14)]}, True)
    assert positions == {17: [2]}
    assert unavailable == {1: 'target_outside_model_input'}
    positions, unavailable = plan_masks(text, [(0, 0), (2, 4), (0, 0)], [1, 0, 1],
                                        {17: [(2, 5)]}, True)
    assert not positions and unavailable[17] == 'target_truncated_or_not_aligned'


def test_caption_variant_trains_only_text_and_invalidates_text_caches(tmp_path):
    raw = yaml.safe_load((Path(__file__).parents[1] / 'configs/smoke.yaml').read_text())
    raw.update(output=str(tmp_path / 'original'), cache=str(tmp_path / 'cache'))
    raw['data'].update(root=str(tmp_path / 'data'), fixture_images=8)
    raw['training']['arguments'].update(max_steps=2, warmup_ratio=0)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    with RunStore(cfg).lock():
        store = RunStore(cfg)
        for stage in ['prepare', 'embed', 'train', 'select']:
            run(cfg, store, stage)
    source = cfg.output
    caps = [{'image_id': i, 'caption': 'A grass scene.' if i % 2 else 'A person beside a bicycle.'}
            for i in range(8)]
    override = tmp_path / 'replacement.json'
    override.write_text(json.dumps({'annotations': caps}))
    raw['output'] = str(tmp_path / 'variant')
    raw['data'].update(caption_overrides={'train2017': str(override)}, text_label_scope='maskable_input')
    raw['training']['image_checkpoint'] = str(source / 'models/image')
    raw['reuse'] = dict(source_run=str(source), sae_weights=False)
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    store = RunStore(cfg)
    with store.lock():
        for stage in ['prepare', 'embed', 'train', 'select']:
            run(cfg, store, stage)
    history = json.loads((cfg.output / 'training.json').read_text())
    assert history['histories']['image']['optimizer_steps'] == 0
    assert history['histories']['text']['optimizer_steps'] == 2
    assert history['captions'] == 8
    original = load_file(str(source / 'models/image/model.safetensors'))
    new = load_file(str(cfg.output / 'models/image/model.safetensors'))
    assert all(np.array_equal(original[k].numpy(), new[k].numpy()) for k in original)
    reuse = {r['artifact']: r for r in json.loads((cfg.output / 'reuse.json').read_text())['items']}
    assert reuse['embeddings/train2017/image']['status'] == 'reused'
    assert reuse['embeddings/train2017/text']['status'] == 'recomputed'
    assert reuse['activations/train2017/image']['status'] == 'reused'
    assert reuse['activations/val2017/text']['status'] == 'recomputed'
    assert reuse['counterfactual/train2017/0/image']['checks']['sae_weights_identical']
    assert not (cfg.output / 'checkpoints/image').exists()
    # The external caption content is part of the resume signature.
    override.write_text(json.dumps({'annotations': caps + [caps[0]]}))
    with pytest.raises(ValueError, match='different code/config'):
        with RunStore(cfg).lock():
            pass
