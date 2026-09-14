import json

import pytest
import torch
from transformers import CLIPConfig, CLIPModel

from mm_sae.data.text import CaptionEditor, Concept
from mm_sae.models.text_masking import (
    clip_pool_positions,
    plan_masks,
    positions_for_spans,
    project_at_positions,
    replace_with_unknown,
)


def test_repeated_and_split_words_mask_every_target_token_without_the_food():
    text = "A hot dog beside a dachshund dog."
    offsets = [
        (0, 0),
        (0, 1),
        (2, 5),
        (6, 9),
        (10, 16),
        (17, 18),
        (19, 23),
        (23, 28),
        (29, 32),
        (32, 33),
        (0, 0),
    ]
    selected = positions_for_spans(text, offsets, [1] + [0] * 9 + [1], [(19, 28), (29, 32)])
    assert selected == [6, 7, 8]
    ids = torch.tensor([[20, 1, 2, 3, 4, 5, 6, 7, 3, 8, 21]])
    attention = torch.ones_like(ids)
    masked = replace_with_unknown(ids, attention, [selected], 21, torch.tensor([10]))
    assert masked.tolist() == [[20, 1, 2, 3, 4, 5, 21, 21, 21, 8, 21]]
    assert ids[0, 6:9].tolist() == [6, 7, 3]
    assert attention.eq(1).all()


def test_truncated_or_partial_token_targets_are_explicitly_unavailable():
    text = "a dog and dogs"
    offsets = [(0, 0), (0, 1), (2, 5), (6, 9), (0, 0)]
    accepted, unavailable = plan_masks(text, offsets, [1, 0, 0, 0, 1], {17: [(2, 5), (10, 14)]})
    assert accepted == {} and unavailable == {17: "target_truncated_or_not_aligned"}
    with pytest.raises(ValueError, match="non_target"):
        positions_for_spans("hotdog", [(0, 6)], [0], [(3, 6)])
    with pytest.raises(ValueError, match="BOS, EOS"):
        replace_with_unknown(torch.tensor([[9, 3, 10, 10]]), torch.tensor([[1, 1, 1, 0]]), [[2]], 10, [2])
    with pytest.raises(ValueError, match="no unknown token"):
        replace_with_unknown(torch.tensor([[9, 3, 10]]), torch.ones(1, 3), [[1]], None, [2])


@pytest.mark.parametrize("configured_eos", [2, 10])
@torch.inference_mode()
def test_unk_equal_to_eos_preserves_original_pooling_and_suffix_context(configured_eos):
    torch.manual_seed(0)
    model = CLIPModel(
        CLIPConfig(
            text_config={
                "vocab_size": 11,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 8,
                "bos_token_id": 9,
                "eos_token_id": configured_eos,
                "pad_token_id": 10,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
            },
            projection_dim=8,
        )
    ).eval()
    ids = torch.tensor([[9, 3, 4, 5, 10], [9, 6, 10, 10, 10]])
    attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    pooled_at = clip_pool_positions(ids, configured_eos)
    assert pooled_at.tolist() == [4, 2]
    torch.testing.assert_close(
        project_at_positions(model, ids, attention, pooled_at),
        model.get_text_features(input_ids=ids, attention_mask=attention),
    )
    masked = replace_with_unknown(ids, attention, [[1], [1]], 10, pooled_at)
    correct = project_at_positions(model, masked, attention, pooled_at)
    naive = model.get_text_features(input_ids=masked, attention_mask=attention)
    assert not torch.allclose(correct, naive)
    # A change after the masked position must still reach the sentence representation.
    changed_suffix = masked.clone()
    changed_suffix[0, 3] = 7
    assert not torch.allclose(
        correct[0], project_at_positions(model, changed_suffix, attention, pooled_at)[0]
    )
    torch.testing.assert_close(
        naive, model.get_text_features(input_ids=changed_suffix, attention_mask=attention)
    )


def test_reviewed_annotations_require_exact_spans_instead_of_rewritten_sentences(tmp_path):
    path = tmp_path / "captions.jsonl"
    record = {
        "caption_id": 1,
        "original": "A dog.",
        "concept_ids": [17],
        "spans": {"17": [{"text": "dog", "start": 2, "end": 5}]},
    }
    path.write_text(json.dumps(record) + "\n")
    concepts = [Concept(17, "dog", ("dog",))]
    assert CaptionEditor(concepts, path).analyze(1, "A dog.")[1] == {17: [(2, 5)]}
    record["edits"] = {"17": "A ."}
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Rewritten captions"):
        CaptionEditor(concepts, path).analyze(1, "A dog.")
