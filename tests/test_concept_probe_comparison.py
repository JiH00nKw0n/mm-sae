"""Runner-level population, normalization, and experiment separation checks."""

from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit
import yaml

from experiments.concept_probe_comparison import run
from mm_sae.analysis.concept_logistic import fit_fixed_support_logistic
from mm_sae.analysis.concept_supervision import project_concepts
from mm_sae.analysis.mapping_evaluation import paired_retrieval


def folders(path):
    for name in ("fits", "diagnostics", "predictions", "results"):
        (path / name).mkdir()


def weighted_stats(x, weights, *, shift=0.0):
    actual_mean = np.average(x, axis=0, weights=weights)
    centered = x - actual_mean
    scale = np.sqrt(np.average(centered**2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1
    covariance = np.einsum("ni,nj,n->ij", centered / scale, centered / scale, weights / weights.sum())
    return dict(mean=actual_mean + shift, scale=scale, cov=covariance)


def read_projection(path, raw):
    with np.load(path) as fit:
        return (project_concepts(raw, fit["feature_mean"], fit["feature_scale"], fit["coefficients"])
                - fit["projection_offset"])


def test_saved_logistic_projection_matches_direct_weighted_standardized_fit_logits(tmp_path):
    rng = np.random.default_rng(391)
    x = rng.normal(size=(85, 5))
    weights = np.arange(len(x)) % 5 + 1
    # Exercise the nonzero projection offset, not only the usual exact-fit mean.
    stats = weighted_stats(x, weights, shift=0.17)
    labels = np.column_stack((x[:, 1] > 0, x[:, 2] + x[:, 3] > 0, np.ones(len(x))))
    fit = fit_fixed_support_logistic(x, labels, stats["mean"], stats["scale"], [[1], [2, 3], [0]],
                                     sample_weight=weights)
    dest = tmp_path / "fit.npz"
    run.save_fit(dest, fit.coefficients, stats, np.arange(5), fit.bias,
                 fit_prediction_mean=fit.fit_prediction_mean)
    logits = np.einsum("ni,ic->nc", (x - stats["mean"]) / stats["scale"], fit.coefficients) + fit.bias
    center = np.average(logits, axis=0, weights=weights)
    centered = logits - center
    std = np.sqrt(np.average(centered**2, axis=0, weights=weights))
    active = std > 1e-10
    expected = np.zeros_like(logits)
    expected[:, active] = centered[:, active] / std[active]
    np.testing.assert_allclose(read_projection(dest, x), expected, atol=1e-12)
    np.testing.assert_allclose(np.average(read_projection(dest, x), axis=0, weights=weights), 0, atol=1e-12)
    np.testing.assert_allclose(np.average(read_projection(dest, x)**2, axis=0, weights=weights), [1, 1, 0],
                               atol=1e-12)
    unseen = rng.normal(size=(9, 5)) + 2.5
    unseen_logits = np.einsum("ni,ic->nc", (unseen - stats["mean"]) / stats["scale"], fit.coefficients) + fit.bias
    expected_unseen = np.zeros_like(unseen_logits)
    expected_unseen[:, active] = (unseen_logits[:, active] - center[active]) / std[active]
    np.testing.assert_allclose(read_projection(dest, unseen), expected_unseen, atol=1e-12)
    # Search uses centered logits, not sigmoid probabilities or test centering.
    assert not np.allclose(read_projection(dest, unseen)[:, :2], expit(unseen_logits)[:, :2])
    assert np.max(np.abs(read_projection(dest, unseen).mean(axis=0))) > 0.1


def test_logistic_runner_population_weights_equal_caption_repetition(monkeypatch, tmp_path):
    image = np.arange(20).reshape(4, 5).astype(float)
    parents = np.array([2, 0, 0, 1, 2, 3, 3, 3])
    text = np.arange(32).reshape(8, 4).astype(float)
    presence = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]], bool)
    cached = SimpleNamespace(image_ids=np.array([10, 20, 30, 40]), parents=parents, presence=presence,
                             mentions=np.zeros((8, 3), bool),
                             activations={"image": sparse.csr_matrix(image), "text": sparse.csr_matrix(text)})
    opened = []

    def cache(path, split):
        opened.append(split)
        return cached

    monkeypatch.setattr(run, "CachedSplit", cache)
    data = dict(population={"fit_image_ids": [40, 20]}, concepts=[{}, {}], fit_caption_count=4,
                ids={"image": np.array([4, 1]), "text": np.array([2, 0])})
    arrays = run.logistic_training_arrays({"source_run": str(tmp_path)}, data)
    assert opened == ["train2017"]
    xi, yi, mass = arrays["image"]
    xt, yt, text_mass = arrays["text"]
    np.testing.assert_array_equal(xi.toarray(), image[[1, 3]][:, [4, 1]])
    np.testing.assert_array_equal(mass, [1, 3])
    np.testing.assert_array_equal(yi, presence[[1, 3], :2])
    np.testing.assert_array_equal(xt.toarray(), text[[3, 5, 6, 7]][:, [2, 0]])
    np.testing.assert_array_equal(yt, presence[parents[[3, 5, 6, 7]], :2])
    np.testing.assert_array_equal(yt, np.repeat(yi, mass.astype(int), axis=0))
    np.testing.assert_allclose(np.average(yi, axis=0, weights=mass), yt.mean(axis=0))
    assert text_mass is None
    with pytest.raises(ValueError, match="population changed"):
        run.logistic_training_arrays({"source_run": str(tmp_path)}, {**data, "fit_caption_count": 5})


def test_logistic_runner_freezes_forward_support_not_refined_support(monkeypatch, tmp_path):
    folders(tmp_path)
    rng = np.random.default_rng(70)
    raw = rng.normal(size=(40, 4))
    y = np.column_stack([raw[:, 0] > 0, raw[:, 2] > 0])
    weights = np.arange(40) % 4 + 1
    stats = weighted_stats(raw, weights)
    data: dict[str, Any] = dict(concepts=[{}, {}], stats={s: stats for s in run.SIDES},
                               ids={s: np.array([10, 20, 30, 40]) for s in run.SIDES})
    forward = np.array([[1., 0], [0, 0], [0, 1], [0, 0]])
    swapped = np.array([[0., 0], [1, 0], [0, 0], [0, 1]])
    for side in run.SIDES:
        run.save_fit(tmp_path / "fits" / f"forward_ridge_{side}_1.npz", forward, stats,
                     data["ids"][side], y.mean(axis=0))
        run.save_fit(tmp_path / "fits" / f"swap_ridge_{side}_1.npz", swapped, stats,
                     data["ids"][side], y.mean(axis=0))
    arrays = {"image": (sparse.csr_matrix(raw), y, weights),
              "text": (sparse.csr_matrix(np.repeat(raw, weights, axis=0)), np.repeat(y, weights, axis=0), None)}
    monkeypatch.setattr(run, "logistic_training_arrays", lambda *_: arrays)
    cfg = dict(output=str(tmp_path), budgets=[1], device="cpu", logistic_device="cpu",
               logistic_penalty=0.01, logistic_maxiter=1000)
    run.fit_logistic(cfg, data)
    saved = {}
    for side in run.SIDES:
        with np.load(tmp_path / "fits" / f"fixed_support_logistic_{side}_1.npz") as fit:
            saved[side] = {key: fit[key].copy() for key in ("raw_coefficients", "raw_bias", "fit_prediction_mean")}
            np.testing.assert_array_equal(fit["raw_coefficients"] != 0, forward != 0)
            assert not np.array_equal(fit["raw_coefficients"] != 0, swapped != 0)
    for key in saved["image"]:
        np.testing.assert_allclose(saved["image"][key], saved["text"][key], atol=1e-9)
    # All-complete resumption must not reopen training data or refit models.
    monkeypatch.setattr(run, "logistic_training_arrays", lambda *_: pytest.fail("completed fit reopened training"))
    run.fit_logistic(cfg, data)


def test_evaluate_uses_saved_centering_and_reports_both_search_directions(monkeypatch, tmp_path):
    folders(tmp_path)
    rng = np.random.default_rng(132)
    parents = np.repeat(np.arange(6), 2)
    raw = {"image": rng.normal(size=(6, 4)), "text": rng.normal(size=(12, 3))}
    labels = rng.random((6, 2)) > 0.5
    concepts = [dict(id=1, name="first", group="object"), dict(id=92, name="second", group="background")]
    test = dict(raw=raw, parents=parents, image_labels=labels, text_presence=labels[parents],
                text_mentions=np.zeros((12, 2), bool))
    calls = []

    def load_test(*_):
        calls.append("test")
        return test

    monkeypatch.setattr(run, "load_test", load_test)
    monkeypatch.setattr(run, "_all_concepts", lambda _: concepts)
    expected = {}
    for side in run.SIDES:
        d = raw[side].shape[1]
        coef = rng.normal(size=(d, 2))
        stats = dict(mean=rng.normal(size=d), scale=rng.uniform(0.5, 2, size=d), cov=np.eye(d))
        bias, center = np.array([2., -3]), np.array([2.5, -4])
        std = np.sqrt(np.sum(coef**2, axis=0))
        expected[side] = (np.einsum("ni,ic->nc", (raw[side] - stats["mean"]) / stats["scale"], coef)
                          + bias - center) / std
        for method in (*run.METHODS, "sparse_cca"):
            run.save_fit(tmp_path / "fits" / f"{method}_{side}_4.npz", coef, stats, np.arange(d), bias,
                         fit_prediction_mean=center)
    cfg = dict(output=str(tmp_path), budgets=[4], device="cpu", retrieval_chunk=3)
    run.evaluate(cfg, {"concepts": concepts})
    assert calls == ["test"]
    direct = paired_retrieval(expected["image"], expected["text"], parents, chunk_size=3)
    for method in (*run.METHODS, "sparse_cca"):
        with np.load(tmp_path / "predictions" / f"{method}_4.npz") as prediction:
            for side in run.SIDES:
                np.testing.assert_allclose(prediction[side], expected[side], atol=1e-12)
        result = json.loads((tmp_path / "results" / f"{method}_4.json").read_text())
        for direction in ("image_to_text", "text_to_image"):
            assert result["retrieval"][direction]["recall"] == {
                str(k): value for k, value in direct[direction]["recall"].items()
            }
            assert set(result["retrieval"][direction]["recall"]) == {"1", "5", "10"}
        assert result["dimensions"] == 2
        assert len(result["semantic"]) == (0 if method == "sparse_cca" else 2)
        if method != "sparse_cca":
            # Captions are evaluated against inherited image labels, not mentions.
            assert [row["text_test_positive"] for row in result["semantic"]] == labels[parents].sum(axis=0).tolist()


def test_main_finishes_all_fitting_before_opening_evaluation(monkeypatch, tmp_path):
    order = []
    cfg = dict(output=str(tmp_path), budgets=[8, 16, 32], ridge=0.01, logistic_penalty=0.01,
                progress_interval_seconds=10)
    data = dict(concepts=[dict(id=1)], moments={}, fit_image_count=5, fit_caption_count=12,
                population={"test_image_ids": [101, 102]})
    monkeypatch.setattr(sys, "argv", ["run", "--config", "unused.yaml"])
    monkeypatch.setattr(run, "config_from", lambda *_a, **_k: cfg)
    monkeypatch.setattr(run, "ProgressReporter", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(run, "stage_progress", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(run, "freeze_inputs", lambda _: order.append("freeze"))

    def load_fit(_):
        order.append("load_fit")
        return data

    monkeypatch.setattr(run, "load_fit", load_fit)
    for name in ("fit_forward", "fit_refined", "fit_logistic", "fit_controls", "evaluate"):
        monkeypatch.setattr(run, name, lambda *_args, name=name: order.append(name))
    report = ModuleType("experiments.concept_probe_comparison.report")
    setattr(report, "build_report", lambda _: order.append("report"))
    monkeypatch.setitem(sys.modules, "experiments.concept_probe_comparison.report", report)
    run.main()
    assert order == ["freeze", "load_fit", "fit_forward", "fit_refined", "fit_logistic", "fit_controls",
                     "evaluate", "report"]
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert protocol["budgets"] == [8, 16, 32]
    assert protocol["fit_caption_pairs"] == 12
    assert "no sigmoid" in protocol["search_representation"]


@pytest.mark.parametrize("override", [
    {"label_targets": ["caption_mentions"]}, {"budgets": [0]}, {"budgets": [1.5]}, {"budgets": [True]},
])
def test_config_rejects_changed_labels_or_invalid_budgets(tmp_path, override):
    cfg: dict[str, Any] = {key: key for key in
                           ("source_run", "parent_run", "reference_run", "semantics_run", "output")}
    cfg.update(label_targets=["propagated_presence"], budgets=[8, 16, 32])
    cfg.update(override)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        run.config_from(path)
