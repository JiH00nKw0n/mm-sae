"""Independent integration checks for annotation-guided retrieval leakage."""

import copy
import json

import numpy as np
from scipy import sparse

from experiments.oracle_sets import run
from mm_sae.analysis.data import save_json
from mm_sae.metrics.regression import sparse_moments


def test_training_moments_ignore_tune_images_and_do_not_open_evaluation(tmp_path, monkeypatch):
    source, parent = tmp_path / "source", tmp_path / "parent"
    parent.mkdir()
    cfg = {"source_run": str(source), "parent_run": str(parent),
           "label_targets": ["propagated_presence", "caption_mentions"]}
    save_json(parent / "population.json", {
        "fit_image_ids": [101, 103], "tune_image_ids": [102], "test_image_ids": [201]})
    save_json(source / "dataset.json", {"concepts": [{"id": 18, "name": "dog"}]})
    # The held-aside middle image deliberately has extreme values and labels.
    image = np.array([[1., 0.], [10000., 50000.], [4., 3.]])
    text = np.array([[2., 0.], [10000., 40000.], [0., 2.], [1., 3.], [3., 1.]])
    parents = np.array([0, 1, 2, 2, 2])
    fit_rows = np.array([0, 2, 3, 4])
    fit = sparse_moments(sparse.csr_matrix(image[parents[fit_rows]]),
                         sparse.csr_matrix(text[fit_rows]))
    np.savez(parent / "moments.npz", image_ids=[0, 1], text_ids=[0, 1],
             fit_n=fit.n, fit_mean=fit.mean, fit_second=fit.second)
    opened = []

    class TrainingCache:
        def __init__(self, _source, split):
            opened.append(split)
            assert split == "train2017", "Fitting must not open the evaluation split"
            self.image_ids = np.array([101, 102, 103])
            self.parents = parents
            self.activations = {"image": sparse.csr_matrix(image), "text": sparse.csr_matrix(text)}
            self.presence = np.array([[0], [1], [1]])
            self.mentions = np.array([[0], [1], [0], [1], [1]])
            self.ids = [18]

    monkeypatch.setattr(run, "CachedSplit", TrainingCache)
    data = run.load_fit(cfg)
    assert opened == ["train2017"]
    assert data["fit_image_count"] == 2
    assert data["fit_caption_count"] == 4
    np.testing.assert_allclose(data["moments"]["image"]["label_mean"], [.75])
    np.testing.assert_allclose(data["moments"]["propagated_presence"]["label_mean"], [.75])
    np.testing.assert_allclose(data["moments"]["caption_mentions"]["label_mean"], [.5])
    for side, labels, block in (("image", [0, 1, 1, 1], slice(0, 2)),
                                ("propagated_presence", [0, 1, 1, 1], slice(2, 4)),
                                ("caption_mentions", [0, 0, 1, 1], slice(2, 4))):
        raw = image[parents[fit_rows]] if side == "image" else text[fit_rows]
        z = (raw - fit.mean[block]) / fit.scale[block]
        expected = np.mean(z * (np.array(labels) - np.mean(labels))[:, None], axis=0)
        np.testing.assert_allclose(data["moments"][side]["cross"][:, 0], expected, atol=1e-14)


def test_changing_all_evaluation_annotations_cannot_change_retrieval_or_predictions(tmp_path):
    rng = np.random.default_rng(819)
    image, text = rng.normal(size=(4, 3)), rng.normal(size=(8, 2))
    labels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
    parents = np.repeat(np.arange(4), 2)
    test = {"raw": {"image": image, "text": text}, "parents": parents,
            "image_labels": labels, "text_presence": labels[parents], "text_mentions": labels[parents]}
    data = {"ids": {"image": np.array([2, 7, 9]), "text": np.array([5, 11])},
            "concepts": [{"id": 18, "name": "dog", "group": "object"},
                         {"id": 19, "name": "horse", "group": "object"}]}
    coefficients = {"image": np.array([[.8, 0.], [0., 1.2], [.4, 0.]]),
                    "text": np.array([[.6, .2], [.5, -.4]])}
    results, predictions = [], []
    for trial in range(2):
        output = tmp_path / f"trial_{trial}"
        for folder in ("fits", "results", "cases", "predictions"):
            (output / folder).mkdir(parents=True)
        cfg = {"output": str(output), "budgets": [2], "include_dense": False,
               "label_targets": ["propagated_presence"], "device": "cpu", "retrieval_chunk": 2,
               "case_concepts": ["dog"], "conditional_pairs": [["dog", "horse"]]}
        for side, key in (("image", "image"), ("text", "propagated_presence")):
            b = coefficients[side]
            np.savez(output / "fits" / f"{key}_2.npz", coefficients=b,
                     feature_mean=np.zeros(b.shape[0]), feature_scale=np.ones(b.shape[0]),
                     active=np.ones(2, bool))
        changed = copy.deepcopy(test)
        if trial:
            for key in ("image_labels", "text_presence", "text_mentions"):
                changed[key] = ~changed[key]
        run.evaluate_sets(cfg, data, changed)
        results.append(json.loads((output / "results/propagated_presence_2.json").read_text()))
        with np.load(output / "predictions/propagated_presence_2.npz") as saved:
            predictions.append({side: saved[side].copy() for side in ("image", "text")})
    assert results[0]["retrieval"] == results[1]["retrieval"]
    for side in ("image", "text"):
        np.testing.assert_array_equal(predictions[0][side], predictions[1][side])
        # An independent scalar projection verifies original feature ordering.
        raw = test["raw"][side]
        expected = np.array([[sum(float(row[j]) * float(coefficients[side][j, c])
                                  for j in range(len(row))) for c in range(2)] for row in raw])
        np.testing.assert_allclose(predictions[0][side], expected, atol=1e-14)
    for before, after in zip(results[0]["semantic"], results[1]["semantic"], strict=True):
        np.testing.assert_allclose(before["image_auc"] + after["image_auc"], 1.)
        np.testing.assert_allclose(before["text_auc"] + after["text_auc"], 1.)
