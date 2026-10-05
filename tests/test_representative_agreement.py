import json

import numpy as np
import pytest
from scipy import sparse

from mm_sae.analysis.representative_agreement import (
    evaluate_representative_agreement,
    variable_coordinates,
)


def evaluate(image, text, **kwargs):
    image = np.asarray(image, dtype=float)
    text = np.asarray(text, dtype=float)
    kwargs.setdefault("image_variable", np.ones(image.shape[0], dtype=bool))
    kwargs.setdefault("text_variable", np.ones(text.shape[0], dtype=bool))
    kwargs.setdefault("n_null", 0)
    return evaluate_representative_agreement(image, text, **kwargs)


def test_independent_representatives_not_high_transfer_auc_determine_agreement():
    result = evaluate([[0.95], [0.85]], [[0.90], [0.97]])
    row = result["per_category"][0]
    assert row["image_coordinate"] == 0
    assert row["text_coordinate"] == 1
    assert row["image_pick_in_text_auc_with_image_sign"] == 0.90
    assert row["rank_in_text"] == row["rank_in_image"] == 2
    assert result["summary"]["agree_at1"] == 0
    assert result["summary"]["image_to_text"]["top5"] == 1


def test_native_pairing_is_supplied_by_row_order_not_relearned_from_annotations():
    image = np.array([[0.9, 0.6], [0.6, 0.95]])
    text = image[::-1]
    unpaired = evaluate(image, text)
    fixed_pairing = evaluate(image, text[[1, 0]])
    assert unpaired["summary"]["agree_at1"] == 0
    assert fixed_pairing["summary"]["agree_at1"] == 1
    assert [row["text_coordinate"] for row in fixed_pairing["per_category"]] == [0, 1]


def test_opposite_independent_polarities_never_count_as_top_k_match():
    result = evaluate([[0.95], [0.6]], [[0.05], [0.6]])
    row = result["per_category"][0]
    assert row["image_coordinate"] == row["text_coordinate"] == 0
    assert row["image_sign"] == 1 and row["text_sign"] == -1
    assert row["rank_in_text"] is None and row["rank_in_image"] is None
    for direction in ("image_to_text", "text_to_image"):
        assert result["summary"][direction]["top10"] == 0
        assert result["summary"][direction]["opposite_polarity_count"] == 1


def test_common_sign_flip_preserves_selection_ranks_and_agreement():
    image = np.array([[0.91, 0.2], [0.3, 0.95], [0.65, 0.7]])
    text = np.array([[0.8, 0.1], [0.2, 0.97], [0.7, 0.8]])
    original = evaluate(image, text)
    # A simultaneous sign reversal of a CCA component is representationally arbitrary.
    image[0], text[0] = 1 - image[0], 1 - text[0]
    flipped = evaluate(image, text)
    assert original["summary"] == flipped["summary"]
    for before, after in zip(original["per_category"], flipped["per_category"], strict=True):
        assert before["image_coordinate"] == after["image_coordinate"]
        assert before["text_coordinate"] == after["text_coordinate"]
        assert before["rank_in_text"] == after["rank_in_text"]


def test_positive_only_protocol_does_not_flip_low_auc_coordinates():
    signed = evaluate([[0.05], [0.8]], [[0.05], [0.9]])
    positive = evaluate([[0.05], [0.8]], [[0.05], [0.9]], signed=False)
    assert signed["per_category"][0]["image_coordinate"] == 0
    assert signed["per_category"][0]["image_sign"] == -1
    assert positive["per_category"][0]["image_coordinate"] == 1
    assert positive["per_category"][0]["image_sign"] == 1


def test_deterministic_ties_and_tie_aware_diagnostic_are_distinct():
    result = evaluate([[0.8], [0.8]], [[0.7], [0.9]])
    row = result["per_category"][0]
    assert row["image_coordinate"] == 0
    assert row["image_tie_count"] == 2
    assert row["text_tie_count"] == 1
    assert result["summary"]["agree_at1"] == 0
    assert result["summary"]["text_to_image"]["top1"] == 0
    assert result["summary"]["text_to_image"]["top1_tie_aware"] == 1
    assert result["selection_statistics"]["image_categories_with_top_ties"] == 1


def test_roundoff_ties_choose_lowest_native_coordinate_with_positive_sign_at_half():
    result = evaluate([[0.2, 0.5], [0.8, 0.5]], [[0.2, 0.5], [0.8, 0.5]])
    assert [row["image_coordinate"] for row in result["per_category"]] == [0, 0]
    assert [row["image_sign"] for row in result["per_category"]] == [-1, 1]
    assert [row["image_tie_count"] for row in result["per_category"]] == [2, 2]
    assert result["summary"]["agree_at1"] == 1


def test_top_k_denominator_and_directions_use_each_source_pick():
    image = np.linspace(0.99, 0.61, 12)[:, None]
    # Image pick 0 has text rank 12, while text pick 3 has image rank 4.
    text = np.linspace(0.61, 0.99, 12)[:, None]
    text[[3, 11]] = text[[11, 3]]
    result = evaluate(image, text)
    row = result["per_category"][0]
    assert row["rank_in_text"] == 12
    assert row["rank_in_image"] == 4
    assert result["summary"]["image_to_text"]["top10"] == 0
    assert result["summary"]["text_to_image"]["top5"] == 1
    assert result["summary"]["text_to_image"]["n_categories"] == 1


@pytest.mark.parametrize("as_sparse", [False, True])
def test_constant_and_nonfinite_score_columns_are_invalid(as_sparse):
    scores = np.array([[0, 2, 1, 1, 1], [0, 2, 3, np.nan, np.inf], [0, 2, 2, 1, 1]])
    flags = variable_coordinates(sparse.csr_matrix(scores) if as_sparse else scores, chunk_size=2)
    np.testing.assert_array_equal(flags, [False, False, True, False, False])
    result = evaluate(np.full((5, 1), 0.9), np.full((5, 1), 0.9),
                      image_variable=flags, text_variable=flags)
    assert result["n_candidate_coordinates"] == 1
    assert result["per_category"][0]["image_coordinate"] == 2


def test_nonfinite_auc_and_caller_exclusions_remain_explicit():
    result = evaluate([[np.nan, 0.8, np.nan], [np.inf, 0.9, 0.7]],
                      [[0.7, 0.8, 0.6], [0.8, 0.9, 0.7]],
                      eligible_categories=np.array([True, False, True]), category_ids=["missing", "excluded", "dog"])
    assert result["n_eligible_categories"] == 2
    assert result["n_evaluated_categories"] == 1
    assert [row["status"] for row in result["per_category"]] == [
        "no_valid_paired_coordinates", "excluded_by_caller", "ok"]
    assert result["per_category"][2]["image_coordinate"] == 1
    assert result["per_category"][2]["n_candidate_coordinates"] == 1
    json.dumps(result, allow_nan=False)


def test_validity_intersection_can_encode_per_category_support():
    image = np.array([[0.95, 0.8], [0.8, 0.95]])
    text = image.copy()
    result = evaluate(image, text, image_variable=np.array([[False, True], [True, False]]))
    assert [row["image_coordinate"] for row in result["per_category"]] == [1, 0]
    assert result["summary"]["agree_at1"] == 1


def test_degenerate_shared_coordinate_is_exposed_by_category_shuffle_null():
    image = np.array([[0.99, 0.98, 0.97, 0.96], [0.6, 0.6, 0.6, 0.6]])
    result = evaluate(image, image, n_null=1000, seed=3)
    assert result["summary"]["agree_at1"] == 1
    assert result["selection_statistics"]["distinct_image_coordinates"] == 1
    assert result["controls"]["category_label_shuffle"]["mean"] == 1
    assert result["controls"]["category_label_shuffle"]["p_ge_observed"] == 1
    assert result["controls"]["random_pairing"]["mean"] == pytest.approx(0.5, abs=0.05)
    assert result["controls"]["random_pairing_expected_agreement"] == 0.5


def test_nulls_are_reproducible_and_keep_polarities_fixed():
    image = np.array([[0.99, 0.6], [0.6, 0.99]])
    text = 1 - image
    first = evaluate(image, text, n_null=40, seed=4)
    second = evaluate(image, text, n_null=40, seed=4)
    assert first["controls"] == second["controls"]
    assert first["controls"]["random_pairing"]["mean"] == 0
    assert first["controls"]["random_pairing_expected_agreement"] == 0
    assert first["controls"]["category_label_shuffle"]["mean"] == 0


def test_repeat_stability_checks_both_coordinate_and_polarity():
    auc = np.array([[0.95, 0.6], [0.6, 0.95]])
    repeat = np.array([[0.05, 0.6], [0.6, 0.95]])
    result = evaluate(auc, auc, image_repeat_auc=repeat, image_repeat_variable=np.ones(2, dtype=bool))
    stability = result["controls"]["image_split_stability"]
    assert stability["n_categories"] == 2
    assert stability["agree_at1_count"] == 1
    assert stability["agree_at1"] == 0.5
    assert result["controls"]["text_split_stability"] is None


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_empty_axes_and_undefined_results_are_strict_json_serializable(shape):
    result = evaluate(np.empty(shape), np.empty(shape), n_null=10)
    assert result["n_evaluated_categories"] == 0
    assert result["summary"]["agree_at1"] is None
    assert result["controls"]["random_pairing"]["mean"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"image_variable": [1, 1]},
    {"eligible_categories": [True, False]},
    {"n_null": -1},
    {"tie_tolerance": float("inf")},
    {"signed": "true"},
    {"category_ids": ["a", "b"]},
    {"image_repeat_auc": np.array([[0.8], [0.9]])},
])
def test_invalid_protocol_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate([[0.8], [0.9]], [[0.8], [0.9]], **kwargs)


def test_out_of_range_auc_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="AUROC"):
        evaluate([[1.01]], [[0.9]])
