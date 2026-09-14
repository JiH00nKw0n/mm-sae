"""Map source character spans to token positions without changing sequence geometry."""

from __future__ import annotations

import torch


def positions_for_spans(text, offsets, special, spans):
    """Reject truncated spans or tokens that also contain non-target characters."""
    target = set()
    for start, end in spans:
        if not (0 <= start < end <= len(text)):
            raise ValueError("invalid_character_span")
        target.update(i for i in range(start, end) if not text[i].isspace())
    if not target:
        raise ValueError("empty_target_span")
    positions, covered = [], set()
    for position, ((start, end), is_special) in enumerate(zip(offsets, special)):
        characters = {i for i in range(start, end) if not text[i].isspace()}
        if is_special or not (characters & target):
            continue
        if characters - target:
            raise ValueError("token_also_contains_non_target_characters")
        positions.append(position)
        covered.update(characters)
    if covered != target:
        raise ValueError("target_truncated_or_not_aligned")
    return positions


def plan_masks(text, offsets, special, spans_by_concept):
    positions, unavailable = {}, {}
    for concept, spans in spans_by_concept.items():
        try:
            positions[concept] = positions_for_spans(text, offsets, special, spans)
        except ValueError as exc:
            unavailable[concept] = str(exc)
    return positions, unavailable


def replace_with_unknown(input_ids, attention_mask, positions, unk_token_id, pool_positions):
    """Replace each selected token ID once, preserving attention and the original EOS."""
    if unk_token_id is None:
        raise ValueError("Tokenizer has no unknown token; do not add an untrained token")
    if len(positions) != len(input_ids):
        raise ValueError("One token-position list is required per caption")
    masked = input_ids.clone()
    for row, selected in enumerate(positions):
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate mask token position")
        for position in selected:
            if not (0 < position < int(pool_positions[row])) or not attention_mask[row, position]:
                raise ValueError("Cannot mask BOS, EOS, padding, or a position outside the model input")
            masked[row, position] = unk_token_id
    return masked


def clip_pool_positions(input_ids, eos_token_id):
    """Match HF CLIP's pool selection on ORIGINAL IDs, including its legacy config."""
    if eos_token_id == 2:
        return input_ids.to(torch.int).argmax(dim=-1)
    is_eos = input_ids == eos_token_id
    if not is_eos.any(dim=-1).all():
        raise ValueError("Caption has no EOS token")
    return is_eos.int().argmax(dim=-1)


def project_at_positions(model, input_ids, attention_mask, pool_positions):
    # HF performs the entire transformer computation. Ignore its pooler, whose first-EOS
    # selection would move to an inserted UNK when UNK and EOS share an ID.
    hidden = model.text_model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    ).last_hidden_state
    pooled = hidden[torch.arange(len(hidden), device=hidden.device), pool_positions.to(hidden.device)]
    return model.text_projection(pooled)
