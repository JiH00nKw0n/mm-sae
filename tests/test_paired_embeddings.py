"""Streaming extraction tests use tiny paired images, never the remote corpus."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mm_sae.data import paired_embeddings as pe


REVISION = "a" * 40
ENCODER = {"backend": "synthetic", "synthetic_dim": 6, "device": "cpu"}


class FakeEncoder:
    dim = 6

    def __init__(self):
        self.image_batches: list[int] = []
        self.text_batches: list[int] = []
        self.fail_text_batch: int | None = None

    @staticmethod
    def vectors(values):
        result = np.asarray([[value, 1, 2, 3, 4, 5] for value in values], dtype=np.float32)
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    def images(self, images):
        self.image_batches.append(len(images))
        return self.vectors([np.asarray(image).mean() for image in images])

    def texts(self, texts):
        self.text_batches.append(len(texts))
        if self.fail_text_batch == len(self.text_batches):
            raise RuntimeError("simulated encoder interruption")
        return self.vectors([int(text) for text in texts])


def config(**changes):
    return {
        "dataset_id": "fixture/paired",
        "revision": REVISION,
        "shard_paths": ["train/000.tar", "train/001.tar"],
        "batch_size": 2,
    } | changes


def row(number):
    return {"__key__": str(number), "jpg": Image.new("RGB", (3, 3), (number,) * 3), "txt": str(number)}


def install_rows(monkeypatch, shards):
    opened = []

    def stream(settings, shard, download_dir):
        opened.append(shard)
        for value in shards[shard]:
            yield row(value) if isinstance(value, int) else value

    monkeypatch.setattr(pe, "_stream_rows", stream)
    return opened


def test_sharded_extraction_preserves_pairs_and_resumes(monkeypatch, tmp_path):
    opened = install_rows(monkeypatch, {"train/000.tar": [1, 2, 3], "train/001.tar": [4, 5]})
    encoder = FakeEncoder()
    progress = []
    manifest = pe.extract(config(), tmp_path, encoder_config=ENCODER, encoder=encoder, on_progress=progress.append)
    assert manifest["rows"] == 5
    assert manifest["all_selected_sources_exhausted"]
    assert manifest["source_rows_read"] == 5
    assert manifest["excluded_rows"] == 0
    assert encoder.image_batches == [2, 1, 2]
    assert len(progress) == 3
    for part, expected in zip(manifest["parts"], ([1, 2, 3], [4, 5])):
        images = np.load(tmp_path / part["image_path"], mmap_mode="r")
        texts = np.load(tmp_path / part["text_path"], mmap_mode="r")
        np.testing.assert_allclose(images, FakeEncoder.vectors(expected))
        np.testing.assert_array_equal(images, texts)
        assert [r["key"] for r in json.loads((tmp_path / part["keys_path"]).read_text())] == list(map(str, expected))
    resumed = pe.extract(config(batch_size=64), tmp_path, encoder_config=ENCODER)
    assert resumed["rows"] == 5
    assert opened == ["train/000.tar", "train/001.tar"]
    assert not list(tmp_path.rglob("*.raw"))
    assert json.loads((tmp_path / "embedding_progress.json").read_text())["state"] == "completed"


def test_interrupted_shard_restarts_without_reencoding_completed_parts(monkeypatch, tmp_path):
    opened = install_rows(monkeypatch, {"train/000.tar": [1, 2], "train/001.tar": [3, 4, 5]})
    encoder = FakeEncoder()
    encoder.fail_text_batch = 2
    with pytest.raises(RuntimeError, match="interruption"):
        pe.extract(config(), tmp_path, encoder_config=ENCODER, encoder=encoder)
    assert (tmp_path / "parts/00000/completion.json").exists()
    assert not (tmp_path / "parts/00001/completion.json").exists()
    assert (tmp_path / "parts/00001.pending").is_dir()
    assert json.loads((tmp_path / "manifest.json").read_text())["state"] == "failed"
    resumed_encoder = FakeEncoder()
    manifest = pe.extract(config(), tmp_path, encoder_config=ENCODER, encoder=resumed_encoder)
    assert manifest["rows"] == 5
    assert opened == ["train/000.tar", "train/001.tar", "train/001.tar"]
    assert resumed_encoder.image_batches == [2, 1]
    assert not (tmp_path / "parts/00001.pending").exists()


def test_limits_are_encoded_pair_limits_and_a_different_cache_identity(monkeypatch, tmp_path):
    opened = install_rows(monkeypatch, {"train/000.tar": [1, {"txt": ""}, 2, 3], "train/001.tar": [4]})
    manifest = pe.extract(config(max_samples=2), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())
    assert manifest["rows"] == 2
    assert manifest["source_rows_read"] == 3
    assert manifest["excluded_rows"] == 1
    assert not manifest["all_selected_sources_exhausted"]
    assert opened == ["train/000.tar"]
    with pytest.raises(ValueError, match="different source, encoder, code, or sample limit"):
        pe.extract(config(), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())


@pytest.mark.parametrize("change", [{"revision": "b" * 40}, {"max_shards": 1}, {"caption_column": "caption"}])
def test_changed_source_definition_rejected(monkeypatch, tmp_path, change):
    install_rows(monkeypatch, {"train/000.tar": [1], "train/001.tar": [2]})
    pe.extract(config(), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())
    with pytest.raises(ValueError, match="different source"):
        pe.extract(config(**change), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())


def test_completed_array_corruption_is_not_silently_reused(monkeypatch, tmp_path):
    install_rows(monkeypatch, {"train/000.tar": [1], "train/001.tar": [2]})
    pe.extract(config(), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())
    path = tmp_path / "parts/00000/image.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum changed"):
        pe.extract(config(), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())


def test_only_known_row_errors_are_excluded(monkeypatch, tmp_path):
    install_rows(monkeypatch, {"train/000.tar": [
        1, {"__key__": "bad-image", "jpg": {"bytes": b"not an image", "path": None}, "txt": "2"},
        {"__key__": "empty-caption", "jpg": None, "txt": ""}, {"__key__": "no-image", "txt": "3"}, 4,
    ]})
    manifest = pe.extract(config(max_shards=1), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())
    assert manifest["rows"] == 2
    assert manifest["excluded_rows"] == 3
    excluded = [json.loads(line) for line in (tmp_path / "parts/00000/excluded.jsonl").read_text().splitlines()]
    assert [entry["key"] for entry in excluded] == ["bad-image", "empty-caption", "no-image"]
    assert excluded[0]["reason"].startswith("image_decode_error")


def test_empty_valid_shard_has_well_shaped_arrays(monkeypatch, tmp_path):
    install_rows(monkeypatch, {"train/000.tar": [{"txt": ""}]})
    manifest = pe.extract(config(max_shards=1), tmp_path, encoder_config=ENCODER, encoder=FakeEncoder())
    assert manifest["rows"] == 0
    assert np.load(tmp_path / "parts/00000/image.npy").shape == (0, 6)
    assert json.loads((tmp_path / "parts/00000/keys.json").read_text()) == []


def test_local_shard_uses_hf_reader_and_removes_owned_download(monkeypatch, tmp_path):
    import huggingface_hub

    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as archive:
        for number in (1, 2, 3):
            encoded = io.BytesIO()
            Image.new("RGB", (3, 3), (number,) * 3).save(encoded, format="PNG")
            image_bytes = encoded.getvalue() if number != 2 else b"corrupted PNG"
            for extension, value in (("jpg", image_bytes), ("txt", str(number).encode())):
                info = tarfile.TarInfo(f"{number}.{extension}")
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
    downloads = []

    def download(*, repo_id, repo_type, revision, filename, local_dir):
        assert repo_id == "fixture/paired" and repo_type == "dataset" and revision == REVISION
        downloads.append(Path(local_dir))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    output = tmp_path / "output"
    manifest = pe.extract(
        config(max_shards=1, staging="local_shard"), output, encoder_config=ENCODER, encoder=FakeEncoder()
    )
    assert manifest["rows"] == 2 and manifest["excluded_rows"] == 1
    assert len(downloads) == 1 and not downloads[0].exists()
    assert source.exists()
    assert manifest["identity"]["source"]["revision"] == REVISION


@pytest.mark.parametrize("changes", [
    {"revision": "main"}, {"shard_paths": ["../outside.tar"]},
    {"shard_paths": ["a.tar", "a.tar"]}, {"max_samples": 0}, {"unexpected": 1},
])
def test_invalid_config_fails_before_creating_data(changes):
    with pytest.raises(ValueError):
        pe.ShardEmbeddingConfig.model_validate(config(**changes))
