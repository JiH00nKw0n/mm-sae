import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from experiments.representative_agreement import run as runner
from mm_sae.analysis.semantic_evaluation import auc_matrix


def test_hungarian_native_pairs_reorder_text_coordinates_and_retain_original_ids(tmp_path):
    mapping = np.zeros((3, 4))
    mapping[0, 2] = mapping[2, 1] = 1
    path = tmp_path / "hungarian.npz"
    np.savez(path, mapping=mapping)
    data = {"feature_ids": {"image": np.array([20, 21, 22]), "text": np.array([30, 31, 32, 33])}}
    weights, identities = runner.load_model({"path": str(path), "kind": "permutation"}, data)
    assert identities == [
        {"coordinate": 0, "image_feature": 20, "text_feature": 32},
        {"coordinate": 1, "image_feature": 22, "text_feature": 31},
    ]
    image = np.array([[2.0, 100.0, 7.0], [3.0, 200.0, 8.0]])
    text = np.array([[100.0, 7.0, 2.0, 200.0], [200.0, 8.0, 3.0, 300.0]])
    np.testing.assert_array_equal(image @ weights["image"], image[:, [0, 2]])
    np.testing.assert_array_equal(text @ weights["text"], text[:, [2, 1]])
    np.testing.assert_array_equal(image @ weights["image"], text @ weights["text"])


def test_hungarian_rejects_multiple_partners_instead_of_treating_them_as_native_pairs(tmp_path):
    path = tmp_path / "not_a_permutation.npz"
    np.savez(path, mapping=np.array([[1, 1], [0, 0]]))
    data = {"feature_ids": {"image": np.arange(2), "text": np.arange(2)}}
    with pytest.raises(ValueError, match="one-to-one"):
        runner.load_model({"path": str(path), "kind": "permutation"}, data)


def test_cca_native_projection_is_preserved_without_annotation_rematching(tmp_path):
    wi = np.array([[2.0, -3.0], [4.0, 5.0], [-6.0, 7.0]])
    wt = np.array([[8.0, 9.0], [-10.0, 11.0]])
    path = tmp_path / "cca.npz"
    np.savez(path, image=wi, text=wt)
    data = {"feature_ids": {"image": np.arange(3), "text": np.arange(2)}}
    weights, identities = runner.load_model({"path": str(path), "kind": "projection"}, data)
    np.testing.assert_array_equal(weights["image"], wi)
    np.testing.assert_array_equal(weights["text"], wt)
    assert identities == [{"coordinate": 0}, {"coordinate": 1}]


def test_sparse_projection_matches_centered_standardized_dense_coordinates(monkeypatch):
    raw = {
        "image": np.array([[0., 1.], [3., 0.], [1., 4.], [2., 2.], [0., 3.], [4., 0.]]),
        "text": np.array([[1., 0.], [0., 2.], [5., 0.], [0., 4.], [1., 3.], [2., 0.]]),
    }
    weights = {
        "image": np.array([[1., -2., 0.], [3., 4., 0.]]),
        "text": np.array([[2., -1., 0.], [-3., 4., 0.]]),
    }
    # Binary-exact means/scales preserve exact score ties in both equivalent
    # algebraic evaluations; this test checks normalization, not roundoff order.
    means = {"image": np.array([0.5, 1.5]), "text": np.array([1.5, 0.5])}
    scales = {"image": np.array([2., 4.]), "text": np.array([4., 0.5])}
    labels = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0], [0, 1]], dtype=bool)
    first = np.array([True, True, True, False, False, False])
    masks = {"image_a": first, "image_b": ~first, "text_a": first, "text_b": ~first}
    data = dict(raw={side: sparse.csr_matrix(value) for side, value in raw.items()},
                means=means, scales=scales, labels={"image": labels, "text": labels}, masks=masks,
                eligible=np.ones(2, dtype=bool), concepts=[{"id": 1}, {"id": 2}])
    captured = []

    def capture_scores(scores, y, **kwargs):
        captured.append((scores.copy(), y.copy()))
        return auc_matrix(scores, y, **kwargs)

    monkeypatch.setattr(runner, "auc_matrix", capture_scores)
    stages = []
    arrays = runner.calculate_aucs({"auc_chunk": 1}, data, weights, stages.append)
    assert stages == ["image_a", "image_b", "text_a", "text_b"]
    for key, (scores, y) in zip(stages, captured, strict=True):
        side = key.split("_")[0]
        expected = ((raw[side] - means[side]) / scales[side]) @ weights[side]
        np.testing.assert_allclose(scores, expected[masks[key]], atol=1e-14)
        np.testing.assert_array_equal(y, labels[masks[key]])
        np.testing.assert_allclose(arrays[key], auc_matrix(expected[masks[key]], labels[masks[key]]))
        np.testing.assert_array_equal(arrays[key + "_variable"], [True, True, False])


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def population_fixture(tmp_path):
    source, parent = tmp_path / "source", tmp_path / "parent"
    index = source / "index" / "train2017"
    activations = source / "activations" / "train2017"
    index.mkdir(parents=True)
    activations.mkdir(parents=True)
    parent.mkdir()
    image_ids = np.arange(101, 107)
    parents = np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
    concept_ids = [*range(1, 81), 92]
    presence = ((np.arange(6)[:, None] + np.arange(81)[None, :]) % 3 == 0)
    _write_json(index / "images.json", [{"image_id": int(i)} for i in image_ids])
    _write_json(index / "concept_ids.json", concept_ids)
    _write_json(source / "dataset.json", {"concepts": [{"id": i, "name": f"category{i}"} for i in concept_ids]})
    np.save(index / "parents.npy", parents)
    np.save(index / "presence.npy", presence)
    image_activations = np.arange(18).reshape(6, 3).astype(float)
    text_activations = np.arange(40).reshape(10, 4).astype(float)
    sparse.save_npz(activations / "image.npz", sparse.csr_matrix(image_activations))
    sparse.save_npz(activations / "text.npz", sparse.csr_matrix(text_activations))
    population = {"tune_image_ids": [106, 101, 104, 103], "fit_image_ids": [102], "test_image_ids": [201]}
    _write_json(parent / "population.json", population)
    means = np.array([2., 3., 4., 5.])
    np.savez(parent / "moments.npz", image_ids=np.array([0, 2]), text_ids=np.array([1, 3]),
             fit_n=100, fit_mean=means, fit_second=np.outer(means, means) + np.diag([1., 4., 9., 16.]))
    cfg = {"source_run": str(source), "parent_run": str(parent), "seed": 3, "min_positive": 1}
    return cfg, image_ids, parents, presence, image_activations, text_activations


def test_population_split_keeps_every_caption_with_parent_and_inherits_its_labels(population_fixture):
    cfg, image_ids, all_parents, presence, raw_image, raw_text = population_fixture
    data = runner.load_population(cfg, "tune")
    image_rows = np.array([0, 2, 3, 5])
    text_rows = np.array([0, 1, 3, 4, 5, 6, 8, 9])
    parents = all_parents[text_rows]
    np.testing.assert_array_equal(data["raw"]["image"].toarray(), raw_image[image_rows][:, [0, 2]])
    np.testing.assert_array_equal(data["raw"]["text"].toarray(), raw_text[text_rows][:, [1, 3]])
    np.testing.assert_array_equal(data["labels"]["image"], presence[image_rows, :80])
    np.testing.assert_array_equal(data["labels"]["text"], presence[parents, :80])
    np.testing.assert_array_equal(data["scales"]["image"], [1, 2])
    np.testing.assert_array_equal(data["scales"]["text"], [3, 4])
    image_a_ids = set(image_ids[image_rows][data["masks"]["image_a"]])
    image_b_ids = set(image_ids[image_rows][data["masks"]["image_b"]])
    text_a_parent_ids = set(image_ids[parents][data["masks"]["text_a"]])
    text_b_parent_ids = set(image_ids[parents][data["masks"]["text_b"]])
    assert image_a_ids == text_a_parent_ids
    assert image_b_ids == text_b_parent_ids
    assert not image_a_ids & text_b_parent_ids
    assert not image_b_ids & text_a_parent_ids
    assert image_a_ids | image_b_ids == {101, 103, 104, 106}
    assert len(data["concepts"]) == 80
    assert all(concept["id"] != 92 for concept in data["concepts"])
    repeated = runner.load_population(cfg, "tune")
    for key in data["masks"]:
        np.testing.assert_array_equal(repeated["masks"][key], data["masks"][key])


def test_population_rejects_images_used_to_fit_correspondence(population_fixture):
    cfg, *_ = population_fixture
    path = Path(cfg["parent_run"]) / "population.json"
    population = json.loads(path.read_text())
    population["fit_image_ids"] = [101]
    _write_json(path, population)
    with pytest.raises(ValueError, match="overlaps correspondence fitting"):
        runner.load_population(cfg, "tune")


def test_runner_evaluates_image_a_against_text_b_not_the_same_image_group():
    right = np.array([[0.9], [0.6]])
    wrong = np.array([[0.6], [0.9]])
    arrays = {"image_a": right, "image_b": wrong, "text_a": right, "text_b": wrong,
              "eligible": np.array([True]), "category_ids": np.array([1])}
    for key in ("image_a", "image_b", "text_a", "text_b"):
        arrays[key + "_variable"] = np.ones(2, dtype=bool)
    result = runner.evaluation(arrays, {"n_null": 0, "seed": 0}, signed=True)
    assert result["summary"]["agree_at1"] == 0
    assert result["per_category"][0]["image_coordinate"] == 0
    assert result["per_category"][0]["text_coordinate"] == 1
    assert result["controls"]["image_split_stability"]["agree_at1"] == 0
    assert result["controls"]["text_split_stability"]["agree_at1"] == 0
