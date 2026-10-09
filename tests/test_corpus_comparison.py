"""Guard sample alignment and supervision boundaries in cross-corpus follow-ups."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
import yaml

from experiments.corpus_comparison import cache, prepare
from experiments.corpus_comparison.report import paired_interval
from experiments.corpus_comparison.diagnose import projection_diagnostics
from mm_sae.analysis.annotation_population import load_population
from mm_sae.cache import embedding_directory
from mm_sae.config import Config, EncoderConfig
from mm_sae.io import atomic_json, sha256
from mm_sae.models.sae import SAEConfig, TopKSAE
from mm_sae.training import frozen_fingerprint
from mm_sae.metrics.regression import sparse_moments


def test_frozen_reencoding_preserves_caption_order_and_annotations(tmp_path):
    source, trained, out = [tmp_path / x for x in ("coco", "cc3m", "encoded")]
    encoder = EncoderConfig(backend="synthetic", synthetic_dim=6, device="cpu")
    config = Config.model_validate(dict(output=source, cache=tmp_path / "cache",
                                       data=dict(source="synthetic"), encoder=encoder))
    index = source / "index/train2017"
    index.mkdir(parents=True)
    atomic_json(source / "run.json", dict(config=config.model_dump(mode="json")))
    atomic_json(source / "dataset.json", dict(concepts=[dict(id=1, name="object")]))
    atomic_json(index / "images.json", [dict(image_id=i, image_sha256=str(i)) for i in (10, 20, 30)])
    atomic_json(index / "captions.json", [dict(caption_id=i, text=str(i)) for i in range(5)])
    atomic_json(index / "concept_ids.json", [1])
    np.save(index / "parents.npy", [2, 0, 1, 0, 2])
    np.save(index / "presence.npy", [[1], [0], [1]])
    np.save(index / "mentions.npy", [[0], [1], [0], [1], [1]])
    embeddings = embedding_directory(config, "train2017")
    embeddings.mkdir(parents=True)
    np.save(embeddings / "image.npy", np.eye(6, dtype=np.float32)[:3])
    np.save(embeddings / "text.npy", np.eye(6, dtype=np.float32)[[0, 1, 2, 3, 0]])
    for side in ("image", "text"):
        TopKSAE(SAEConfig(hidden_size=6, latent_size=8, k=2)).save_pretrained(trained / "models" / side)
    atomic_json(trained / "models/frozen.json", frozen_fingerprint(trained))
    atomic_json(trained / "resolved-config.json", dict(encoder=encoder.model_dump(), source=dict(dataset="test")))
    cfg = dict(model_run=str(trained), annotation_run=str(source), output=str(out),
               device="cpu", batch_size=2, splits=["train2017"], max_images=2)
    before = sha256(trained / "models/image/model.safetensors")
    cache.run(cfg)
    np.testing.assert_array_equal(np.load(out / "index/train2017/parents.npy"), [0, 1, 0])
    captions = json.loads((out / "index/train2017/captions.json").read_text())
    assert [c["caption_id"] for c in captions] == [1, 2, 3]
    np.testing.assert_array_equal(np.load(out / "index/train2017/mentions.npy"), [[1], [0], [1]])
    assert sparse.load_npz(out / "activations/train2017/text.npz").shape == (3, 8)
    assert before == sha256(trained / "models/image/model.safetensors")
    cache.run(cfg)
    assert json.loads((out / "completed.json").read_text())["sae_training_steps"] == 0
    np.save(index / "mentions.npy", [[1], [1], [0], [1], [1]])
    with pytest.raises(ValueError, match="annotation changed"):
        cache.run(cfg)


def test_annotation_population_retains_cc3m_fit_and_rejects_overlap(tmp_path):
    parent = tmp_path / "mapping"
    atomic_json(parent / "population.json", dict(fit_image_ids=["cc3m:a"], tune_image_ids=["cc3m:b"],
                                                 test_image_ids=[30], tune_images=1, fit_images=1))
    file = tmp_path / "annotations.json"
    atomic_json(file, dict(fit_image_ids=[10], tune_image_ids=[20], test_image_ids=[30]))
    result = load_population(parent, str(file))
    assert result["fit_image_ids"] == ["cc3m:a"]
    assert result["tune_image_ids"] == [20]
    atomic_json(file, dict(tune_image_ids=[30], test_image_ids=[30]))
    with pytest.raises(ValueError, match="overlaps"):
        load_population(parent, str(file))
    atomic_json(file, dict(tune_image_ids=[20], test_image_ids=[31]))
    with pytest.raises(ValueError, match="evaluation image IDs differ"):
        load_population(parent, str(file))


def test_followup_supervision_uses_coco_rows_not_cc3m_rows(tmp_path):
    cfg = dict(model_run=str(tmp_path / "model"), annotation_run=str(tmp_path / "coco"),
               annotation_population=str(tmp_path / "coco-pop.json"), native_parent=str(tmp_path / "native"),
               native_projection=str(tmp_path / "projection"), output=str(tmp_path / "followup"),
               device="cpu", retrieval_chunk=16, sparse_supports=[4, 8, 16, 32], sparse_max_iter=1000)
    sequence, jobs = prepare.prepare(cfg)
    configurations = Path(sequence).parent
    native = yaml.safe_load((configurations / "cc3m-fit-semantics.yaml").read_text())
    oracle = yaml.safe_load((configurations / "annotation-sets.yaml").read_text())
    probes = yaml.safe_load((configurations / "probe-comparison.yaml").read_text())
    assert native["parent_run"] == cfg["native_parent"]
    assert native["annotation_population"] == cfg["annotation_population"]
    assert native["source_run"] != cfg["model_run"]
    assert oracle["source_run"] == probes["source_run"] == native["source_run"]
    assert oracle["parent_run"] == probes["parent_run"] != cfg["native_parent"]
    assert jobs[0]["name"] == "annotated_cache"
    assert not any("paired_corpus.run" in job["argv"] for job in jobs)
    assert all(job["argv"][0] == "-m" for job in jobs)
    assert prepare.prepare(cfg)[1] == jobs


def test_paired_recall_difference_uses_image_groups_and_reports_signed_difference():
    result = paired_interval([1, 1, 2, 2], [2, 2, 1, 1], np.array([0, 0, 1, 1]), "text_to_image", 1)
    assert result["difference"] == 0
    assert result["low"] == -1 and result["high"] == 1


def test_missing_report_module_is_rejected_before_sequence_launch(tmp_path, monkeypatch):
    actual = prepare.find_spec
    monkeypatch.setattr(prepare, "find_spec", lambda name: None if name == "experiments.oracle_sets.report" else actual(name))
    cfg = dict(model_run=str(tmp_path / "model"), annotation_run=str(tmp_path / "coco"),
               annotation_population=str(tmp_path / "coco-pop.json"), native_parent=str(tmp_path / "native"),
               native_projection=str(tmp_path / "projection"), output=str(tmp_path / "followup"),
               device="cpu", retrieval_chunk=16, sparse_supports=[4], sparse_max_iter=1000)
    with pytest.raises(ModuleNotFoundError, match="experiments.oracle_sets.report"):
        prepare.prepare(cfg)
    assert not (tmp_path / "followup/configs/sequence.yaml").exists()


def test_projection_diagnostics_matches_direct_correlation_without_refitting():
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(100, 3)), rng.normal(size=(100, 2))
    train = sparse_moments(sparse.csr_matrix(x), sparse.csr_matrix(y))
    x = x + [4, 8, 2]
    y = y + [10, 3]
    test = sparse_moments(sparse.csr_matrix(x), sparse.csr_matrix(y))
    wi, wt = rng.normal(size=(3, 2)), rng.normal(size=(2, 2))
    rows = projection_diagnostics(test, train, wi, wt)
    a = (x - train.mean[:3]) / train.scale[:3] @ wi
    b = (y - train.mean[3:]) / train.scale[3:] @ wt
    for j, row in enumerate(rows):
        assert row["correlation"] == pytest.approx(np.corrcoef(a[:, j], b[:, j])[0, 1])
        assert row["image_mean"] == pytest.approx(a[:, j].mean())
