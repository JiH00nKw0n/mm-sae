import json
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from mm_sae.cache import embedding_directory
from mm_sae.config import Config
from mm_sae.features import encode_file
from mm_sae.io import atomic_json
from mm_sae.metrics.statistics import correlation
from mm_sae.metrics.matching import match
from mm_sae.metrics.sparse_ops import take_rows
from experiments.rq1 import interventions
from experiments.rq1.config import RQ1Config
from experiments.rq1.correspondence import assess


def test_embedding_cache_reuses_content_across_runs_but_invalidates_changed_input(tmp_path):
    config = Config(output=tmp_path / "run-a", cache=tmp_path / "cache")
    index = config.output / "index" / "train2017"
    atomic_json(index / "images.json", [{"image_id": 1, "image": "/first/mount.jpg", "image_sha256": "abc"}])
    atomic_json(index / "captions.json", [{"caption_id": 7, "text": "A dog.", "concept_ids": [1]}])
    first = embedding_directory(config, "train2017")
    config.output = tmp_path / "run-b"
    index = config.output / "index" / "train2017"
    atomic_json(index / "images.json", [{"image_id": 1, "image": "/second/mount.jpg", "image_sha256": "abc"}])
    atomic_json(index / "captions.json", [{"caption_id": 7, "text": "A dog.", "concept_ids": [2]}])
    assert embedding_directory(config, "train2017") == first
    atomic_json(index / "images.json", [{"image_id": 1, "image_sha256": "different-image"}])
    assert embedding_directory(config, "train2017") != first


def test_interrupted_embedding_extraction_resumes_at_completed_batch(tmp_path):
    path = tmp_path / "image.npy"
    calls = []

    def interrupted(a, b):
        calls.append(a)
        if a == 2:
            raise RuntimeError("interrupted")
        return np.full((b - a, 3), a, dtype=np.float32)

    import pytest

    with pytest.raises(RuntimeError, match="interrupted"):
        encode_file(path, 4, 3, 2, interrupted)
    result = encode_file(path, 4, 3, 2, lambda a, b: np.full((b - a, 3), 9, dtype=np.float32))
    np.testing.assert_array_equal(result[:2], 0)
    np.testing.assert_array_equal(result[2:], 9)


def test_conditional_intervention_recomputes_both_matchings_and_resumes(tmp_path, monkeypatch):
    # Analytic activation fixture, not a COCO result. The wrong partner is correlated
    # because two labels co-occur; only the targeted feature changes on removal.
    n = 100
    a = np.arange(n) < 50
    b = (np.arange(n) < 40) | ((np.arange(n) >= 50) & (np.arange(n) < 60))
    image = sparse.csr_matrix(a[:, None].astype(float))
    text = sparse.csr_matrix(np.column_stack([1 + a + np.random.default_rng(42).uniform(0, 5, n), b]))
    index = SimpleNamespace(
        images=[{"image_id": i} for i in range(n)],
        parents=np.arange(n),
        columns={0: 0, 1: 1},
        presence=np.column_stack([a, b]),
        mentions=np.column_stack([a, b]),
        areas=np.column_stack([a * 0.2, b * 0.2]),
        captions=[
            {
                "image_row": i,
                "concept_ids": [1] if b[i] else [],
                "mask_token_positions": {"1": [1]} if b[i] else {},
            }
            for i in range(n)
        ],
    )
    reps = {"0": {"image": 0, "text": 0}, "1": {"image": None, "text": 1}}
    base = correlation(image, text)
    np.savez_compressed(tmp_path / "panel.npz", **base)
    atomic_json(tmp_path / "representatives.json", reps)
    assignments = match(
        base["C"], base["alive_image"], base["alive_text"], base["valid_image"], base["valid_text"]
    )
    baseline = [
        {**r, "method": m} for m in ["greedy", "hungarian"] for r in assess(base, assignments[m], reps)
    ]
    assert all(r["status"] == "different" for r in baseline)
    atomic_json(tmp_path / "rq1" / "experiment2" / "assessed_rows.json", baseline)
    edited_rows = np.flatnonzero(b)
    edited_text = take_rows(text, edited_rows).toarray()
    edited_text[:, 1] = 0
    frozen_checks = []
    monkeypatch.setattr(interventions, "Index", lambda *_: index)
    monkeypatch.setattr(interventions, "original_latents", lambda *_: (image, text))
    monkeypatch.setattr(
        interventions,
        "read_counterfactual",
        lambda *_: (edited_rows, edited_rows, image[edited_rows], sparse.csr_matrix(edited_text)),
    )
    monkeypatch.setattr(interventions, "assert_frozen", lambda root: frozen_checks.append(root))
    config = Config(output=tmp_path)
    options = RQ1Config(intervention_repeats=1, area_strata=2)
    interventions.experiment3(config, options)
    path = tmp_path / "rq1" / "experiment3" / "summary.json"
    summary = json.loads(path.read_text())
    assert summary["eligible_pairs"] == [[0, 1]]
    rows = summary["outcomes"]
    assert len(rows) == 6
    for row in rows:
        if row["condition"] == "cooccurrence_removal":
            assert row["recovered"]
            assert abs(row["label_correlation"]) < 1e-12
            assert row["images_removed"] == row["captions_edited"] == 30
    assert len(frozen_checks) == 2
    interventions.experiment3(config, options)
    assert json.loads(path.read_text()) == summary
