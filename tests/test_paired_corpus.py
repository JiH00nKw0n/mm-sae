"""Exercise corpus orchestration with local arrays and a mocked shard inventory."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.paired_corpus import run as corpus
from mm_sae.cache import embedding_directory
from mm_sae.config import Config, EncoderConfig
from mm_sae.io import sha256


@pytest.fixture
def evaluation_cache(tmp_path):
    source = tmp_path / "evaluation"
    index = source / "index/val2017"
    index.mkdir(parents=True)
    encoder = EncoderConfig(backend="synthetic", synthetic_dim=6, device="cpu", batch_size=2)
    cfg = Config.model_validate(
        {"output": source, "cache": tmp_path / "cache", "data": {"source": "synthetic"}, "encoder": encoder}
    )
    (source / "run.json").write_text(json.dumps({"config": cfg.model_dump(mode="json")}))
    images = [{"image_id": 11, "image_sha256": "image-a"}, {"image_id": 22, "image_sha256": "image-b"}]
    captions = [{"caption_id": i, "text": f"caption {i}"} for i in range(4)]
    (index / "images.json").write_text(json.dumps(images))
    (index / "captions.json").write_text(json.dumps(captions))
    parents = np.array([1, 0, 1, 0], dtype=np.int64)
    np.save(index / "parents.npy", parents)
    directory = embedding_directory(cfg, "val2017")
    directory.mkdir(parents=True)
    image = np.eye(6, dtype=np.float32)[:2]
    text = np.eye(6, dtype=np.float32)[[1, 2, 3, 4]]
    np.save(directory / "image.npy", image)
    np.save(directory / "text.npy", text)
    return {"evaluation": {"source_run": str(source), "split": "val2017", "name": "test"}}, encoder, directory


def test_evaluation_preserves_all_rows_and_uses_source_manifest_cache(
    evaluation_cache, monkeypatch, tmp_path
):
    cfg, encoder, directory = evaluation_cache
    monkeypatch.chdir(tmp_path)
    records, parents, matrices, identity = corpus.evaluation_inputs(cfg, encoder)
    assert [r["image_id"] for r in records] == [11, 22]
    np.testing.assert_array_equal(parents, [1, 0, 1, 0])
    assert isinstance(matrices["image"], np.memmap)
    assert isinstance(matrices["text"], np.memmap)
    assert matrices["image"].shape == (2, 6)
    assert matrices["text"].shape == (4, 6)
    assert identity[str(directory / "image.npy")] == sha256(directory / "image.npy")
    assert identity[str(directory / "text.npy")] == sha256(directory / "text.npy")


def test_evaluation_smoke_limit_retains_matching_captions_in_order(evaluation_cache):
    cfg, encoder, directory = evaluation_cache
    cfg["evaluation"]["max_images"] = 1
    records, parents, matrices, identity = corpus.evaluation_inputs(cfg, encoder)
    assert [r["image_id"] for r in records] == [11]
    np.testing.assert_array_equal(parents, [0, 0])
    np.testing.assert_array_equal(matrices["text"], np.load(directory / "text.npy")[[1, 3]])
    assert len(identity) == 4


def test_evaluation_ignores_execution_device_and_batch_size(evaluation_cache):
    cfg, encoder, _ = evaluation_cache
    changed = encoder.model_copy(update={"device": "cuda", "batch_size": 99})
    records, _, _, _ = corpus.evaluation_inputs(cfg, changed)
    assert len(records) == 2


def test_evaluation_rejects_changed_encoder_before_reading_embeddings(evaluation_cache):
    cfg, encoder, directory = evaluation_cache
    (directory / "image.npy").unlink()
    changed = encoder.model_copy(update={"text_max_length": 20})
    with pytest.raises(ValueError, match="different frozen encoder"):
        corpus.evaluation_inputs(cfg, changed)


def test_evaluation_rejects_misaligned_rows(evaluation_cache):
    cfg, encoder, directory = evaluation_cache
    np.save(directory / "image.npy", np.eye(6, dtype=np.float32)[:1])
    with pytest.raises(ValueError, match="row counts differ"):
        corpus.evaluation_inputs(cfg, encoder)


@pytest.mark.parametrize(
    "parents",
    [np.array([1.0, 0.0, 1.0, 0.0]), np.array([1, -1, 1, 0]), np.array([2, 0, 1, 0]), np.zeros(4, dtype=int)],
)
def test_evaluation_preflight_rejects_invalid_parents(evaluation_cache, parents):
    cfg, encoder, _ = evaluation_cache
    np.save(Path(cfg["evaluation"]["source_run"]) / "index/val2017/parents.npy", parents)
    with pytest.raises(ValueError):
        corpus.evaluation_inputs(cfg, encoder)


@pytest.mark.parametrize("kind", ["nonfinite", "wrong_dtype", "unnormalized", "wrong_width"])
def test_evaluation_preflight_rejects_invalid_embedding_values(evaluation_cache, kind):
    cfg, encoder, directory = evaluation_cache
    values = np.load(directory / "text.npy")
    if kind == "nonfinite":
        values[0, 0] = np.nan
    elif kind == "wrong_dtype":
        values = values.astype(np.float64)
    elif kind == "unnormalized":
        values *= 2
    else:
        values = values[:, :5]
    np.save(directory / "text.npy", values)
    with pytest.raises(ValueError):
        corpus.evaluation_inputs(cfg, encoder)


def test_resolve_config_paths_use_config_directory(tmp_path, monkeypatch):
    folder = tmp_path / "configs"
    folder.mkdir()
    path = folder / "paired.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "output": "../runs/new",
                "embedding_cache": "../cache/new",
                "evaluation": {"source_run": "../runs/old"},
                "followups": [{"module": "experiments.mapping_suite.run", "config": "mapping.yaml"}],
            }
        )
    )
    monkeypatch.chdir(tmp_path.parent)
    cfg = corpus.resolve_config(path)
    assert cfg["output"] == str(tmp_path / "runs/new")
    assert cfg["embedding_cache"] == str(tmp_path / "cache/new")
    assert cfg["evaluation"]["source_run"] == str(tmp_path / "runs/old")
    assert cfg["followups"][0]["config"] == str(folder / "mapping.yaml")


def test_shard_inventory_is_pinned_sorted_and_reused(tmp_path, monkeypatch):
    revision = "a" * 40
    calls = []

    class FakeApi:
        def dataset_info(self, dataset_id, revision, files_metadata):
            calls.append((dataset_id, revision, files_metadata))
            return SimpleNamespace(
                sha=revision,
                siblings=[
                    SimpleNamespace(rfilename="train-002.tar", size=20),
                    SimpleNamespace(rfilename="train-001.tar", size=10),
                    SimpleNamespace(rfilename="validation.tar", size=99),
                ],
            )

    monkeypatch.setattr(corpus, "HfApi", FakeApi)
    cfg = {"source": {"dataset_id": "fixture/paired", "revision": revision, "shard_pattern": "train-*.tar"}}
    extraction = corpus.shard_configuration(cfg, tmp_path)
    assert extraction["shard_paths"] == ["train-001.tar", "train-002.tar"]
    assert extraction["estimated_source_bytes"] == 30
    assert corpus.shard_configuration(cfg, tmp_path) == extraction
    assert len(calls) == 1
    cfg["source"]["revision"] = "b" * 40
    with pytest.raises(ValueError, match="source request changed"):
        corpus.shard_configuration(cfg, tmp_path)


def test_training_index_preserves_part_order_and_namespaces_keys(tmp_path):
    embedding_root = tmp_path / "embeddings"
    embedding_root.mkdir()
    (embedding_root / "first.json").write_text('[{"key":"2"},{"key":"1"}]')
    (embedding_root / "second.json").write_text('[{"key":"2"}]')
    manifest = {
        "config": {"dataset_id": "cc3m"},
        "rows": 3,
        "parts": [
            {"keys_path": "first.json", "rows": 2, "shard": "first.tar"},
            {"keys_path": "second.json", "rows": 1, "shard": "second.tar"},
        ],
    }
    output = tmp_path / "index"
    corpus.write_training_index(manifest, embedding_root, output)
    assert [r["image_id"] for r in json.loads((output / "images.json").read_text())] == [
        "cc3m:first.tar:2",
        "cc3m:first.tar:1",
        "cc3m:second.tar:2",
    ]
    np.testing.assert_array_equal(np.load(output / "parents.npy"), [0, 1, 2])
    original = (output / "images.json").read_bytes()
    manifest["rows"] = 4
    with pytest.raises(ValueError, match="does not cover every embedding"):
        corpus.write_training_index(manifest, embedding_root, output)
    assert (output / "images.json").read_bytes() == original


def test_evaluation_preflight_failure_prevents_corpus_extraction(tmp_path, monkeypatch):
    def invalid_evaluation(*args):
        raise ValueError("evaluation preflight failed")

    def forbidden_extraction(*args, **kwargs):
        pytest.fail("Extraction must not begin after evaluation preflight failure")

    monkeypatch.setattr(corpus, "evaluation_inputs", invalid_evaluation)
    monkeypatch.setattr(corpus, "extract", forbidden_extraction)
    with pytest.raises(ValueError, match="evaluation preflight failed"):
        corpus.run({"output": str(tmp_path / "run"), "encoder": {"device": "cpu"}})
