"""Integration checks for split membership, native coordinates, and saved retrieval."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from experiments.mapping_semantics import run
from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping_evaluation import mapping_retrieval, paired_retrieval


def _output_folders(output):
    for folder in ("results", "cases", "calibration", "sign-fit", "sparse-fit"):
        (output / folder).mkdir(parents=True, exist_ok=True)


def _corpus(tmp_path):
    source, parent, output = (tmp_path / name for name in ("source", "parent", "output"))
    parent.mkdir()
    _output_folders(output)
    cfg = {"source_run": str(source), "parent_run": str(parent), "output": str(output),
           "projection_run": str(tmp_path / "projection"), "pruning_run": str(tmp_path / "pruning"),
           "auc_chunk": 2, "retrieval_chunk": 2, "device": "cpu", "case_concepts": [],
           "case_examples": 2, "baseline_k": [2], "sign_k": [2], "sparse_k": [2],
           "sinkhorn_epsilon": 0.05}
    population = {"fit_image_ids": [101, 103], "tune_image_ids": [102, 104],
                  "test_image_ids": [201, 202, 203, 204], "fit_images": 2, "tune_images": 2}
    save_json(parent / "population.json", population)
    mean, scale = np.array([1., 2., 3., 4., 5.]), np.array([2., 3., 4., 5., 6.])
    np.savez(parent / "moments.npz", image_ids=[4, 1, 3], text_ids=[2, 0], fit_n=20,
             fit_mean=mean, fit_second=np.diag(scale**2) + np.outer(mean, mean))
    save_json(source / "dataset.json", {"concepts": [{"id": 18, "name": "dog"}]})
    raw = {}
    rng = np.random.default_rng(651)
    for split, image_ids, parents in (
        ("train2017", [101, 102, 103, 104], np.array([0, 1, 1, 2, 3, 3, 3, 0])),
        ("val2017", [201, 202, 203, 204], np.array([0, 0, 1, 2, 3, 3])),
    ):
        index = source / "index" / split
        index.mkdir(parents=True)
        activations = source / "activations" / split
        activations.mkdir(parents=True)
        image = rng.normal(size=(4, 5)) * 3 + 4
        text = rng.normal(size=(len(parents), 3)) * 2 + 5
        raw[split] = {"image": image, "text": text, "parents": parents}
        sparse.save_npz(activations / "image.npz", sparse.csr_matrix(image))
        sparse.save_npz(activations / "text.npz", sparse.csr_matrix(text))
        save_json(index / "images.json", [{"image_id": value} for value in image_ids])
        save_json(index / "captions.json", [{"text": f"caption {row}"} for row in range(len(parents))])
        save_json(index / "concept_ids.json", [18])
        np.save(index / "parents.npy", parents)
        presence = np.array([[0], [0], [1], [1]])
        np.save(index / "presence.npy", presence)
        np.save(index / "mentions.npy", presence[parents])
    return cfg, raw, mean, scale


def test_loading_splits_by_parent_image_preserves_latent_ids_and_rejects_leakage(tmp_path):
    cfg, raw, mean, scale = _corpus(tmp_path)
    data = run.load_data(cfg)
    expected_images = (raw["train2017"]["image"][[1, 3]][:, [4, 1, 3]] - mean[:3]) / scale[:3]
    expected_texts = (raw["train2017"]["text"][[1, 2, 4, 5, 6]][:, [2, 0]] - mean[3:]) / scale[3:]
    np.testing.assert_allclose(data["tune"]["x"]["image"], expected_images)
    np.testing.assert_allclose(data["tune"]["x"]["text"], expected_texts)
    np.testing.assert_array_equal(data["tune"]["labels"]["text"].ravel(), [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(data["parents"], raw["val2017"]["parents"])
    repeated = run.load_data(cfg)
    for partition in ("tune", "test"):
        for side in ("image", "text"):
            np.testing.assert_array_equal(data[partition]["x"][side], repeated[partition]["x"][side])
    population_path = Path(cfg["parent_run"]) / "population.json"
    original = json.loads(population_path.read_text())
    for partition, replacement in (("fit_image_ids", [101, 102]), ("tune_image_ids", [102, 201]),
                                    ("tune_image_ids", [102, 999])):
        changed = {**original, partition: replacement}
        save_json(population_path, changed)
        with pytest.raises((AssertionError, ValueError)):
            run.load_data(cfg)
    save_json(population_path, original)


def test_evaluation_freezes_source_coordinate_and_sign_and_reports_original_feature_ids(tmp_path):
    cfg = {"output": str(tmp_path / "first"), "auc_chunk": 2, "retrieval_chunk": 2, "device": "cpu",
           "case_concepts": ["dog"], "case_examples": 2}
    _output_folders(Path(cfg["output"]))
    labels = np.array([[0], [0], [1], [1]])
    image = np.column_stack([np.ones(4), [3., 2., 1., 0.], np.zeros(4)])
    text = np.column_stack([[0., 1., 2., 3.], [3., 2., 1., 0.]])
    data = {"ids": {"image": np.array([900, 701, 123]), "text": np.array([42, 11])},
            "tune": {"x": {"image": image, "text": text}, "labels": {"image": labels, "text": labels}},
            "test": {"x": {"image": image, "text": text}, "labels": {"image": labels, "text": labels}},
            "parents": np.arange(4), "concepts": [{"id": 18, "name": "dog", "group": "object"}],
            "images": [{"image_id": 201 + row} for row in range(4)],
            "captions": [{"text": f"caption {row}"} for row in range(4)]}
    model = run.Model("paired", "baseline", "cca", "fixture", "common",
                      np.array([[0., 0.], [1., 0.], [0., 1.]]), np.eye(2), [])
    run.evaluate_model(cfg, data, model)
    first = json.loads((Path(cfg["output"]) / "results/paired.json").read_text())
    selected = next(row for row in first["semantic"] if row["direction"] == "image_to_text")
    assert selected["coordinate"] == 0
    assert selected["sign"] == -1
    assert selected["source_tune_auc"] == selected["source_auc"] == 1
    assert selected["target_transfer_auc"] == selected["paired_minimum_auc"] == 0
    cases = json.loads((Path(cfg["output"]) / "cases/paired.json").read_text())
    image_case = next(case for case in cases if case["source"] == "image")
    assert image_case["image_weights"] == [{"feature": 701, "weight": -1.0}]
    assert image_case["text_weights"] == [{"feature": 42, "weight": -1.0}]

    changed = copy.deepcopy(data)
    changed["tune"]["labels"]["text"] = 1 - labels
    changed["test"]["labels"]["text"] = 1 - labels
    changed["test"]["x"]["image"] = image.copy()
    changed["test"]["x"]["image"][:, 1] *= -1
    cfg["output"] = str(tmp_path / "changed_target_labels")
    _output_folders(Path(cfg["output"]))
    run.evaluate_model(cfg, changed, model)
    second = json.loads((Path(cfg["output"]) / "results/paired.json").read_text())
    checked = next(row for row in second["semantic"] if row["direction"] == "image_to_text")
    assert checked["coordinate"] == selected["coordinate"]
    assert checked["sign"] == selected["sign"]
    assert checked["source_tune_auc"] == 1
    assert checked["source_auc"] == 0
    assert checked["target_transfer_auc"] == 1


def test_model_coordinates_and_reused_retrieval_match_the_actual_projection_matrices(tmp_path):
    cfg, _, _, _ = _corpus(tmp_path)
    data = run.load_data(cfg)
    parent, pruning, projection = (Path(cfg[name]) for name in ("parent_run", "pruning_run", "projection_run"))
    for folder in (parent / "candidates", pruning / "transforms", pruning / "sinkhorn/transforms"):
        folder.mkdir(parents=True)
    save_json(parent / "selected.json", [{"family": "hungarian", "key": "hungarian_000"}])
    mapping = np.array([[0., 1.], [1., 0.], [0., 0.]])
    np.savez(parent / "candidates/hungarian_000.npz", mapping=mapping)
    image, text, parents = data["test"]["x"]["image"], data["test"]["x"]["text"], data["parents"]
    expected = {}

    def save_mapping_reference(prefix, key, matrix):
        retrieval = mapping_retrieval(image, text, matrix, parents)
        for suffix, space in (("text_projected_to_image", "image"), ("image_projected_to_text", "text")):
            save_json(Path(str(prefix) + "__" + suffix + ".json"), {"retrieval": retrieval[suffix]})
            expected[key + "__" + space] = retrieval[suffix]

    save_mapping_reference(projection / "results/hungarian__standardized", "hungarian", mapping)
    wi, wt = np.array([[1., -2.], [2., 0.], [-1., 1.]]), np.array([[-1., 1.], [2., 0.5]])
    for tag, factor in (("full", 1.), ("2", -0.6)):
        ai, at = wi.copy(), wt.copy()
        ai[1, 0] *= factor
        at[0, 1] *= factor
        np.savez(pruning / "transforms" / f"cca_{tag}.npz", image=ai, text=at)
        cca = paired_retrieval(image @ ai, text @ at, parents)
        save_json(pruning / "results" / f"cca_{tag}.json", {"retrieval": cca})
        expected[f"cca_{tag}"] = cca
        direct = np.array([[0.8, 0.1 * factor], [-0.6, 0.2], [0.3 * factor, 1.]])
        np.savez(pruning / "transforms" / f"procrustes_{tag}.npz", image_by_text=direct)
        proc = paired_retrieval(image, text @ direct.T, parents)
        save_json(pruning / "results" / f"procrustes_{tag}.json", {"retrieval": proc})
        expected[f"procrustes_{tag}__image"] = proc
        expected[f"procrustes_{tag}__text"] = paired_retrieval(image @ direct, text, parents)
        transport = np.array([[0.2, 0.4], [0.7, 0.3], [0.1, 0.9]])
        if tag == "2":
            transport[0, 0] = transport[1, 1] = transport[2, 0] = 0
        key = f"sinkhorn_e0.05_k{tag}"
        np.savez(pruning / "sinkhorn/transforms" / (key + ".npz"), mapping=transport)
        save_mapping_reference(pruning / "sinkhorn/results" / key, key, transport)
    t2i = np.array([[1., -1., 0.2], [-0.5, 2., 0.]])
    i2t = np.array([[0.2, 0.7], [-1., 0.3], [0.8, -0.2]])
    for constraint in ("signed", "nonnegative"):
        a, b = ((t2i, i2t) if constraint == "signed" else (np.maximum(t2i, 0), np.maximum(i2t, 0)))
        path = Path(cfg["output"]) / "sign-fit" / f"procrustes_support_2_{constraint}.npz"
        np.savez(path, text_to_image=a, image_to_text=b)
        save_json(path.with_suffix(".json"), {})
        expected[f"procrustes_support_2_{constraint}__image"] = paired_retrieval(image, text @ a, parents)
        expected[f"procrustes_support_2_{constraint}__text"] = paired_retrieval(image @ b, text, parents)
    sparse_path = Path(cfg["output"]) / "sparse-fit/sparse_cca_k2.npz"
    np.savez(sparse_path, image=wi, text=wt)
    save_json(sparse_path.with_suffix(".json"), {
        "converged_components": 2,
        "components": [{"converged": True}, {"converged": True}],
    })
    expected["sparse_cca_2"] = paired_retrieval(image @ wi, text @ wt, parents)
    # This corpus has identity standardized fit covariance. Unit variance with
    # ridge 0.01 therefore has a closed form independent of the runner helper.
    with np.load(pruning / "transforms/cca_2.npz") as saved:
        control_image, control_text = saved["image"], saved["text"]
    control_image /= np.sqrt(1.01) * np.linalg.norm(control_image, axis=0)
    control_text /= np.sqrt(1.01) * np.linalg.norm(control_text, axis=0)
    expected["cca_pruned_renormalized_2"] = paired_retrieval(image @ control_image, text @ control_text, parents)

    built = run.models(cfg, data, "all")
    assert len(built) == 18
    for model in built:
        assert model.image.shape[1] == model.text.shape[1]
        if model.space == "image":
            np.testing.assert_array_equal(model.image, np.eye(3))
        elif model.space == "text":
            np.testing.assert_array_equal(model.text, np.eye(2))
        if model.key == "sparse_cca_2":
            np.testing.assert_array_equal(model.image, wi)
            np.testing.assert_array_equal(model.text, wt)
        elif model.key == "cca_pruned_renormalized_2":
            np.testing.assert_allclose(model.image, control_image)
            np.testing.assert_allclose(model.text, control_text)
        run.evaluate_model(cfg, data, model)
        result = json.loads((Path(cfg["output"]) / "results" / (model.key + ".json")).read_text())
        actual = paired_retrieval(run.project(image, model.image), run.project(text, model.text), parents)
        target = expected[model.key]
        for direction in ("image_to_text", "text_to_image"):
            np.testing.assert_array_equal(actual[direction]["ranks"], target[direction]["ranks"])
            np.testing.assert_array_equal(result["retrieval"][direction]["ranks"], target[direction]["ranks"])
        if model.reference is not None:
            assert result["retrieval_source"] == str(model.reference)
