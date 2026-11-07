import numpy as np
import pytest
from scipy.special import expit
from mm_sae.evaluation.concepts import evaluate, entropy_scores
from mm_sae.evaluation import retrieval, edits


def test_validation_selects_axis_even_when_test_reverses_it():
    labels = ["car"] * 8 + ["road"] * 8
    rows = [
        dict(group=f"{split}/{i}", split=split, labels=[label])
        for split in ("train", "val", "test")
        for i, label in enumerate(labels)
    ]
    truth = np.array([1.0] * 8 + [-1.0] * 8)
    # Coordinates agree on train; validation favors axis 1 and test favors axis 0.
    scores = np.vstack(
        (np.column_stack((truth, truth)), np.column_stack((-truth, truth)), np.column_stack((truth, -truth)))
    )
    result, matrix = evaluate(scores, rows, 2, True)
    assert result["concepts"]["car"]["axis"] == 1
    assert result["macro_test_loss"] > 1
    assert matrix["importance"][0, 0] == 0
    rows[-1]["group"] = rows[0]["group"]
    with pytest.raises(ValueError, match="crosses"):
        evaluate(scores, rows, 2, True)


def test_diagnostics_directions_and_empty_mass():
    assert entropy_scores(np.eye(3))["mixing"] == 0.0
    assert entropy_scores(np.ones((3, 4)))["dispersion"] == pytest.approx(1.0)
    assert entropy_scores(np.array([[1.0, 0.0], [1.0, 0.0]]))["zero_importance_concepts"] == 1
    assert entropy_scores(np.zeros((4, 3)))["mixing"] is None


def test_exact_bidirectional_retrieval_handles_multiple_captions_and_ties():
    image = np.array([[1.0, 0.0], [0.0, 1.0]])
    text = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    images = [dict(group=k) for k in "ab"]
    captions = [dict(group=k, truncated=i == 0) for i, k in enumerate("aab")]
    out, ranks = retrieval.evaluate(image, text, images, captions)
    assert out["image_to_text"]["R@1"] == 0.5
    assert ranks["image_ranks"] == [1, 2]
    assert out["text_to_image"]["R@1"] == pytest.approx(2 / 3)
    assert out["truncated_fraction"] == pytest.approx(1 / 3)


def test_edits_score_the_frozen_original_data_probe():
    changes = dict(removed=["car"], added=["truck"], retained=["road", "tree"])
    rows = [dict(group="pair", edit=changes | {"role": role}) for role in ("original", "edited")]
    probes = {c: dict(axis=j, weight=1, bias=0) for j, c in enumerate(("car", "road", "truck"))}
    out, details = edits.evaluate(np.array([[1, 2, -1], [-1, 2, 1]]), rows, probes)
    assert out["removed"]["mean_delta"] == pytest.approx(expit(-1) - expit(1))
    assert out["added"]["mean_delta"] > 0
    assert out["retained"]["mean_absolute_delta"] == 0
    assert out["retained"]["missing_probe"] == 1 and len(details) == 3
