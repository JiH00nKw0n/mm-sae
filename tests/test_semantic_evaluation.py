import numpy as np
import pytest
from scipy import sparse

from mm_sae.analysis.evaluation import auroc
from mm_sae.analysis.semantic_evaluation import (
    auc_matrix,
    calibrate_target_signs,
    evaluate_selected_coordinates,
    select_from_auc,
    select_source_coordinates,
)


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 4, 32])
def test_auc_matrix_matches_scalar_pair_counting_with_ties_and_signed_scores(as_sparse, chunk_size):
    rng = np.random.default_rng(20)
    scores = rng.integers(-3, 4, size=(91, 13)).astype(float)
    scores[:, 2] = 0
    labels = rng.integers(0, 2, size=(91, 7))
    labels[:, -2] = 0
    labels[:, -1] = 1
    matrix = sparse.csr_matrix(scores) if as_sparse else scores
    result = auc_matrix(matrix, labels, chunk_size=chunk_size)
    expected = np.array([[auroc(labels[:, c], scores[:, k]) for c in range(7)] for k in range(13)])
    np.testing.assert_allclose(result, expected, atol=1e-14, equal_nan=True)
    np.testing.assert_allclose(result[2, :-2], 0.5)
    assert np.isnan(result[:, -2:]).all()


def test_auc_matrix_matches_sklearn_for_continuous_and_tied_scores():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(31)
    scores = rng.normal(size=(83, 6))
    scores[:, 3:] = np.round(scores[:, 3:])
    labels = rng.integers(0, 2, size=(83, 5))
    expected = np.array([[sklearn_metrics.roc_auc_score(labels[:, c], scores[:, k])
                          for c in range(5)] for k in range(6)])
    np.testing.assert_allclose(auc_matrix(scores, labels), expected, atol=1e-14)


def test_source_selection_prefers_lower_coordinate_then_positive_sign():
    values = np.array([[0.2, 0.5, np.nan], [0.8, 0.5, np.nan], [0.5, 0.5, np.nan]])
    selected = select_from_auc(values)
    assert selected[0]["coordinate"] == 0
    assert selected[0]["sign"] == -1
    assert selected[0]["source_tune_auc"] == 0.8
    assert selected[1]["coordinate"] == 0
    assert selected[1]["sign"] == 1
    assert selected[2]["coordinate"] is None
    assert selected[2]["sign"] is None
    assert selected[2]["status"] == "missing_source_tune_class"


def test_held_out_transfer_preserves_source_sign_without_target_label_selection():
    labels = np.array([[0], [0], [1], [1]])
    source_tune = np.array([[3, 0], [2, 0], [1, 0], [0, 0]])
    selection = select_source_coordinates(source_tune, labels)
    source_val = np.array([[9, 0], [8, 0], [-1, 0], [-3, 0]])
    target_val = -source_val
    result = evaluate_selected_coordinates(selection, source_val, labels, target_val, labels)
    flipped = evaluate_selected_coordinates(selection, source_val, labels, target_val, 1 - labels)
    assert result[0]["coordinate"] == flipped[0]["coordinate"] == 0
    assert result[0]["sign"] == flipped[0]["sign"] == -1
    assert result[0]["source_auc"] == result[0]["source_tune_auc"] == 1
    assert result[0]["target_transfer_auc"] == result[0]["paired_minimum_auc"] == 0
    assert flipped[0]["target_transfer_auc"] == 1
    # Neither modality is allowed to repair the source tuning polarity on validation.
    reversed_source = evaluate_selected_coordinates(selection, -source_val, labels, target_val, labels)
    assert reversed_source[0]["source_auc"] == 0
    assert selection[0]["sign"] == -1


def test_source_selects_coordinate_even_if_another_coordinate_transfers_better():
    labels = np.array([[0], [0], [1], [1]])
    source = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    target = np.array([[3, 0], [2, 1], [1, 2], [0, 3]])
    selection = select_source_coordinates(source, labels)
    result = evaluate_selected_coordinates(selection, source, labels, target, labels)
    assert result[0]["coordinate"] == 0
    assert result[0]["target_transfer_auc"] == 0


def test_target_calibration_is_separate_and_never_changes_transfer_sign():
    labels = np.array([[0], [0], [1], [1]])
    source = np.arange(4)[:, None]
    selection = select_source_coordinates(source, labels)
    calibration = calibrate_target_signs(selection, -source, labels)
    result = evaluate_selected_coordinates(selection, source, labels, -source, labels,
                                           target_calibration=calibration)
    assert result[0]["sign"] == 1
    assert result[0]["target_tune_sign"] == -1
    assert result[0]["target_transfer_auc"] == 0
    assert result[0]["target_calibrated_auc"] == 1
    # Even the diagnostic target polarity is frozen before validation.
    changed = evaluate_selected_coordinates(selection, source, labels, source, labels,
                                            target_calibration=calibration)
    assert changed[0]["target_calibrated_auc"] == 0


def test_missing_classes_remain_in_results_with_different_modality_sample_counts():
    tune_labels = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])
    scores = np.arange(8).reshape(4, 2)
    selection = select_source_coordinates(scores, tune_labels)
    source_labels = tune_labels.copy()
    source_labels[:, 2] = 1
    target_labels = np.array([[1, 0, 0], [1, 1, 1], [1, 0, 1]])
    result = evaluate_selected_coordinates(selection, scores, source_labels, scores[:3], target_labels)
    assert len(result) == 3
    assert result[0]["source_auc"] == 1
    assert result[0]["target_val_status"] == "missing_class"
    assert result[0]["target_val_positive_count"] == 3
    assert np.isnan(result[0]["target_transfer_auc"])
    assert result[1]["status"] == "missing_source_tune_class"
    assert result[1]["source_val_status"] == result[1]["target_val_status"] == "not_selected"
    assert result[2]["source_val_status"] == "missing_class"
    assert all(np.isnan(row["paired_minimum_auc"]) for row in result)


def test_empty_score_or_observation_axes_keep_concept_denominator():
    labels = np.array([[0, 1], [1, 0]])
    selection = select_source_coordinates(np.empty((2, 0)), labels)
    assert len(selection) == 2
    assert all(row["status"] == "no_coordinates" for row in selection)
    result = evaluate_selected_coordinates(selection, np.empty((2, 0)), labels, np.empty((0, 0)),
                                           np.empty((0, 2)))
    assert len(result) == 2
    assert all(np.isnan(row["source_auc"]) for row in result)
    assert np.isnan(auc_matrix(np.empty((0, 4)), np.empty((0, 2)))).all()


@pytest.mark.parametrize("scores,labels,kwargs", [
    (np.array([[0], [np.nan]]), np.array([[0], [1]]), {}),
    (np.ones((2, 1)), np.array([[0], [2]]), {}),
    (np.ones((2, 1)), np.array([[0]]), {}),
    (np.ones((2, 1)), np.array([[0], [1]]), {"chunk_size": 0}),
    (np.ones((2, 1)), np.array([[0], [1]]), {"chunk_size": 1.5}),
])
def test_auc_matrix_rejects_invalid_inputs(scores, labels, kwargs):
    with pytest.raises(ValueError):
        auc_matrix(scores, labels, **kwargs)


def test_evaluation_rejects_unpaired_coordinates_and_changed_calibration_coordinate():
    labels = np.array([[0], [1]])
    source = np.array([[0, 0], [1, 0]])
    selection = select_source_coordinates(source, labels)
    with pytest.raises(ValueError, match="equal paired-coordinate"):
        evaluate_selected_coordinates(selection, source, labels, source[:, :1], labels)
    calibration = calibrate_target_signs(selection, source, labels)
    calibration[0]["coordinate"] = 1
    with pytest.raises(ValueError, match="preserve source-selected"):
        evaluate_selected_coordinates(selection, source, labels, source, labels, target_calibration=calibration)
