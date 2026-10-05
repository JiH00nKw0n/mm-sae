"""Agreement of independently annotated representatives under a fixed pairing.

Rows of both AUROC matrices must already denote paired native coordinates.
For CCA these are the two learned weighted combinations with the same index;
for a permutation they are its actual matched SAE coordinates. Annotation
AUROC chooses each modality's representative independently. The reported
outcome is agreement of those choices, not the target coordinate's AUROC and
not a claim that each coordinate represents one exclusive semantic concept.

The caller owns the observation split, annotation definition, and cross-method
category eligibility. In particular, corresponding images and captions must
not cross the two independent annotation-selection groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def variable_coordinates(scores, *, chunk_size: int = 256) -> np.ndarray:
    """Return finite, nonconstant columns of an observation-by-coordinate matrix.

An AUROC of 0.5 cannot distinguish a constant coordinate from a varying but
uninformative coordinate. Callers therefore supply these validity flags to
``evaluate_representative_agreement`` separately from their AUROC matrices.
Sparse inputs are inspected in bounded dense column blocks.
"""
    values = scores.tocsc() if sparse.issparse(scores) else np.asarray(scores)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("scores must be a real observation-by-coordinate matrix")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    valid = np.zeros(values.shape[1], dtype=bool)
    if values.shape[0] < 2:
        return valid
    for start in range(0, values.shape[1], int(chunk_size)):
        block: Any = values[:, start:start + int(chunk_size)]
        block = np.asarray(block.toarray() if sparse.issparse(block) else block)
        valid[start:start + block.shape[1]] = (
            np.all(np.isfinite(block), axis=0) & np.any(block != block[0], axis=0)
        )
    return valid


def _auc(value, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real coordinate-by-category matrix")
    values = np.asarray(raw, dtype=np.float64)
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    finite = np.isfinite(values)
    if np.any(finite & ((values < 0) | (values > 1))):
        raise ValueError(f"finite {name} entries must be AUROC values in [0, 1]")
    return values


def _validity(value, shape: tuple[int, int], name: str) -> np.ndarray:
    valid = np.asarray(value)
    if valid.dtype != bool:
        raise ValueError(f"{name} must contain boolean validity flags")
    if valid.shape == (shape[0],):
        return np.broadcast_to(valid[:, None], shape)
    if valid.shape != shape:
        raise ValueError(f"{name} must have one flag per coordinate or coordinate/category pair")
    return valid


def _order(values: np.ndarray, valid: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Sort scores descending and tied coordinates ascending, without rounding.

Each tie group is anchored at its largest score. This avoids a nontransitive
approximate comparator while absorbing floating-point differences such as
``1 - 0.8`` versus ``0.2``. The first tie group is returned separately.
    """
    coordinates = np.flatnonzero(valid)
    coordinates = coordinates[np.argsort(-values[coordinates], kind="stable")]
    groups = []
    start = 0
    while start < len(coordinates):
        stop = start + 1
        while stop < len(coordinates) and values[coordinates[start]] - values[coordinates[stop]] <= tolerance:
            stop += 1
        groups.append(np.sort(coordinates[start:stop]))
        start = stop
    if not groups:
        empty = np.empty(0, dtype=int)
        return empty, empty
    return np.concatenate(groups), groups[0]


def _selection(values: np.ndarray, valid: np.ndarray, signed: bool, tolerance: float) -> dict[str, Any]:
    allowed = valid & np.isfinite(values)
    score = np.maximum(values, 1 - values) if signed else values
    signs = np.where(signed & (values < 0.5), -1, 1)
    order, ties = _order(score, allowed, tolerance)
    coordinate = int(order[0]) if len(order) else None
    return {
        "coordinate": coordinate,
        "sign": None if coordinate is None else int(signs[coordinate]),
        "auc": None if coordinate is None else float(score[coordinate]),
        "raw_auc": None if coordinate is None else float(values[coordinate]),
        "order": order,
        "ties": ties,
        "signs": signs,
    }


def _counterpart_rank(source: dict[str, Any], target: dict[str, Any]) -> tuple[int | None, bool, bool]:
    coordinate = source["coordinate"]
    if coordinate is None or target["coordinate"] is None:
        return None, False, False
    positions = np.flatnonzero(target["order"] == coordinate)
    same_polarity = bool(source["sign"] == target["signs"][coordinate])
    if not len(positions) or not same_polarity:
        return None, same_polarity, False
    rank = int(positions[0]) + 1
    return rank, True, bool(coordinate in target["ties"])


def _fraction(count: int, total: int) -> float | None:
    return float(count / total) if total else None


def _direction_summary(rows: list[dict[str, Any]], rank_key: str, tie_key: str) -> dict[str, Any]:
    result: dict[str, Any] = {"n_categories": len(rows)}
    for k in (1, 5, 10):
        count = sum(row[rank_key] is not None and row[rank_key] <= k for row in rows)
        result[f"top{k}_count"] = int(count)
        result[f"top{k}"] = _fraction(count, len(rows))
    count = sum(row[tie_key] for row in rows)
    result["top1_tie_aware_count"] = int(count)
    result["top1_tie_aware"] = _fraction(count, len(rows))
    result["opposite_polarity_count"] = sum(
        not row[rank_key.replace("rank_", "polarity_match_")] for row in rows
    )
    return result


def _null_summary(draws: np.ndarray, observed: float | None) -> dict[str, Any]:
    if not len(draws):
        return {"n_permutations": 0, "mean": None, "std": None, "p95": None,
                "p_ge_observed": None}
    return {
        "n_permutations": len(draws),
        "mean": float(draws.mean()),
        "std": float(draws.std()),
        "p95": float(np.quantile(draws, 0.95)),
        "p_ge_observed": None if observed is None else float((1 + (draws >= observed).sum()) / (len(draws) + 1)),
    }


def _controls(rows: list[dict[str, Any]], coordinate_pool: np.ndarray, n_null: int, seed: int):
    image = np.asarray([row["image_coordinate"] for row in rows], dtype=int)
    text = np.asarray([row["text_coordinate"] for row in rows], dtype=int)
    image_sign = np.asarray([row["image_sign"] for row in rows], dtype=int)
    text_sign = np.asarray([row["text_sign"] for row in rows], dtype=int)
    same_sign = image_sign == text_sign
    observed = _fraction(int(((image == text) & same_sign).sum()), len(rows))
    random_pairs = np.empty(n_null if rows else 0)
    shuffled_categories = np.empty_like(random_pairs)
    rng = np.random.default_rng(seed)
    image_positions = np.searchsorted(coordinate_pool, image)
    for trial in range(len(random_pairs)):
        # Rewire the native coordinate pairing; retain each representative's
        # independently selected polarity. No new target representative is fit.
        permuted = rng.permutation(coordinate_pool)
        random_pairs[trial] = np.mean((permuted[image_positions] == text) & same_sign)
        permutation = rng.permutation(len(rows))
        shuffled_categories[trial] = np.mean(
            (image == text[permutation]) & (image_sign == text_sign[permutation])
        )
    return {
        "random_pairing": _null_summary(random_pairs, observed),
        "category_label_shuffle": _null_summary(shuffled_categories, observed),
        "random_pairing_expected_agreement": (
            float(same_sign.mean() / len(coordinate_pool)) if rows and len(coordinate_pool) else None
        ),
        "random_pairing_coordinate_only_chance": _fraction(1, len(coordinate_pool)),
        "null_polarity_policy": "Keep the independently chosen signs fixed while shuffling links or category labels.",
    }


def _stability(
    repeat_auc, repeat_variable, baseline: list[dict[str, Any]], joint_valid: np.ndarray,
    signed: bool, tolerance: float, modality: str,
) -> dict[str, Any] | None:
    if repeat_auc is None:
        if repeat_variable is not None:
            raise ValueError(f"{modality}_repeat_variable requires {modality}_repeat_auc")
        return None
    if repeat_variable is None:
        raise ValueError(f"{modality}_repeat_auc requires {modality}_repeat_variable to exclude constants")
    values = _auc(repeat_auc, f"{modality}_repeat_auc", joint_valid.shape)
    valid = _validity(repeat_variable, values.shape, f"{modality}_repeat_variable") & joint_valid
    compared = []
    matches = []
    for row in baseline:
        category = row["category_index"]
        selected = _selection(values[:, category], valid[:, category], signed, tolerance)
        if selected["coordinate"] is None:
            continue
        compared.append(row["category_id"])
        matches.append(selected["coordinate"] == row[f"{modality}_coordinate"]
                       and selected["sign"] == row[f"{modality}_sign"])
    return {
        "n_categories": len(compared), "agree_at1_count": sum(matches),
        "agree_at1": _fraction(sum(matches), len(compared)),
        "category_ids": compared, "n_missing_repeat_categories": len(baseline) - len(compared),
        "interpretation": "Within-modality representative selection stability; not a mathematical performance ceiling.",
    }


def evaluate_representative_agreement(
    image_auc, text_auc, *, image_variable, text_variable, category_ids=None, eligible_categories=None,
    signed: bool = True, image_repeat_auc=None, text_repeat_auc=None,
    image_repeat_variable=None, text_repeat_variable=None,
    n_null: int = 1000, seed: int = 0, tie_tolerance: float = float(4 * np.finfo(float).eps),
) -> dict[str, Any]:
    """Evaluate independently selected representatives under a frozen pairing.

    AUROC inputs have shape ``(paired_coordinates, categories)``. Required
    boolean ``*_variable`` flags have shape ``(paired_coordinates,)`` or the
    full AUROC shape; the latter can additionally encode per-category support.
    Only the intersection of the two validity masks and finite AUROC entries
    is eligible. Eligibility is settled before ranking, never by the observed
    match outcome. Every category retains a result row, including exclusions.

    With ``signed=True``, each modality independently chooses the coordinate
    maximizing ``max(AUC, 1-AUC)`` and its preferred polarity. An opposite
    polarity is not a valid counterpart, even if the coordinate indices match.
    With ``signed=False``, raw AUROC ranks positive-polarity coordinates only.
    Near ties are grouped within ``tie_tolerance`` of a group's highest score
    and resolved by the lowest coordinate index. Strict top-1 therefore equals
    exact representative agreement. Tie-aware top-1 is a separate diagnostic.

    Optional repeat matrices use independent observations from the same
    modality and require their own nonconstant-coordinate flags. The function
    cannot verify the caller's data independence. All outputs are strict JSON
    serializable: undefined values and opposite-polarity ranks use ``None``.
    """
    image = _auc(image_auc, "image_auc")
    text = _auc(text_auc, "text_auc", image.shape)
    if not isinstance(signed, bool):
        raise ValueError("signed must be a boolean")
    if isinstance(n_null, bool) or not isinstance(n_null, (int, np.integer)) or n_null < 0:
        raise ValueError("n_null must be a nonnegative integer")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    image_valid = _validity(image_variable, image.shape, "image_variable")
    text_valid = _validity(text_variable, text.shape, "text_variable")
    paired_valid = image_valid & text_valid
    joint_valid = paired_valid & np.isfinite(image) & np.isfinite(text)
    coordinate_pool = np.flatnonzero(np.any(paired_valid, axis=1))
    n_categories = image.shape[1]
    eligible = np.ones(n_categories, dtype=bool) if eligible_categories is None else np.asarray(eligible_categories)
    if eligible.shape != (n_categories,) or eligible.dtype != bool:
        raise ValueError("eligible_categories must be one boolean flag per category")
    ids = list(range(n_categories)) if category_ids is None else list(category_ids)
    if len(ids) != n_categories or any(not isinstance(x, (str, int, np.integer)) for x in ids):
        raise ValueError("category_ids must contain one string or integer identifier per category")
    ids = [int(x) if isinstance(x, np.integer) else x for x in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("category_ids must be unique")

    rows: list[dict[str, Any]] = []
    for category, category_id in enumerate(ids):
        row: dict[str, Any] = {
            "category_index": category, "category_id": category_id,
            "status": "excluded_by_caller" if not eligible[category] else "no_valid_paired_coordinates",
            "n_candidate_coordinates": int(joint_valid[:, category].sum()),
            "image_coordinate": None, "text_coordinate": None, "image_sign": None, "text_sign": None,
            "image_auc": None, "text_auc": None, "image_raw_auc": None, "text_raw_auc": None,
            "rank_in_text": None, "rank_in_image": None,
            "polarity_match_in_text": None, "polarity_match_in_image": None,
            "agree_at1": None, "image_tie_count": 0, "text_tie_count": 0,
            "image_pick_in_text_top_tie": None, "text_pick_in_image_top_tie": None,
            "image_pick_in_text_auc_with_image_sign": None, "text_pick_in_image_auc_with_text_sign": None,
        }
        if not eligible[category] or not np.any(joint_valid[:, category]):
            rows.append(row)
            continue
        selected_i = _selection(image[:, category], joint_valid[:, category], signed, tie_tolerance)
        selected_t = _selection(text[:, category], joint_valid[:, category], signed, tie_tolerance)
        rank_t, polarity_t, tie_t = _counterpart_rank(selected_i, selected_t)
        rank_i, polarity_i, tie_i = _counterpart_rank(selected_t, selected_i)
        ci, ct = selected_i["coordinate"], selected_t["coordinate"]
        for modality, selected in (("image", selected_i), ("text", selected_t)):
            row.update({f"{modality}_{key}": selected[key] for key in ("coordinate", "sign", "auc", "raw_auc")})
            row[f"{modality}_tie_count"] = len(selected["ties"])
        row.update(
            status="ok", rank_in_text=rank_t, rank_in_image=rank_i,
            polarity_match_in_text=polarity_t, polarity_match_in_image=polarity_i,
            agree_at1=bool(ci == ct and selected_i["sign"] == selected_t["sign"]),
            image_pick_in_text_top_tie=tie_t, text_pick_in_image_top_tie=tie_i,
            image_pick_in_text_auc_with_image_sign=float(
                text[ci, category] if selected_i["sign"] == 1 else 1 - text[ci, category]),
            text_pick_in_image_auc_with_text_sign=float(
                image[ct, category] if selected_t["sign"] == 1 else 1 - image[ct, category]),
        )
        rows.append(row)

    scored = [row for row in rows if row["status"] == "ok"]
    count = sum(row["agree_at1"] for row in scored)
    controls = _controls(scored, coordinate_pool, int(n_null), seed)
    controls["image_split_stability"] = _stability(
        image_repeat_auc, image_repeat_variable, scored, joint_valid, signed, tie_tolerance, "image")
    controls["text_split_stability"] = _stability(
        text_repeat_auc, text_repeat_variable, scored, joint_valid, signed, tie_tolerance, "text")
    return {
        "metric": "independently_selected_representative_agreement",
        "n_coordinates": image.shape[0], "n_candidate_coordinates": len(coordinate_pool),
        "n_categories": n_categories, "n_eligible_categories": int(eligible.sum()),
        "n_evaluated_categories": len(scored),
        "protocol": {
            "signed": signed, "tie_tolerance": float(tie_tolerance),
            "tie_policy": "Descending AUROC; tied coordinates use increasing native paired-coordinate index.",
            "polarity_policy": "Independent max(AUC,1-AUC) with positive polarity at 0.5; signs must agree."
            if signed else "Positive polarity only; rank raw AUROC.",
            "top_k_definition": "The source representative's fixed counterpart is among the target annotation ranking's top k, with the same independently selected polarity.",
            "scope": "Representative agreement, not category-pure semantic matching accuracy.",
        },
        "summary": {
            "agree_at1_count": int(count), "agree_at1": _fraction(count, len(scored)),
            "image_to_text": _direction_summary(scored, "rank_in_text", "image_pick_in_text_top_tie"),
            "text_to_image": _direction_summary(scored, "rank_in_image", "text_pick_in_image_top_tie"),
        },
        "selection_statistics": {
            **{f"distinct_{side}_coordinates": len({row[f"{side}_coordinate"] for row in scored})
               for side in ("image", "text")},
            **{f"distinct_{side}_signed_representatives": len({(row[f"{side}_coordinate"], row[f"{side}_sign"])
                                                              for row in scored}) for side in ("image", "text")},
            **{f"{side}_categories_with_top_ties": sum(row[f"{side}_tie_count"] > 1 for row in scored)
               for side in ("image", "text")},
        },
        "controls": controls, "per_category": rows,
    }
