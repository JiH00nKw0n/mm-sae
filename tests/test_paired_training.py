"""Chunked indexing and immutable training reuse on a real tiny CPU model."""

import json
import pickle

import numpy as np
import pytest

from mm_sae.io import atomic_json, sha256
from mm_sae.paired_training import ChunkedMatrix, train_pairs
import mm_sae.paired_training as training


def cache(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    rng = np.random.default_rng(1)
    all_rows = rng.normal(size=(17, 6)).astype(np.float32)
    all_rows /= np.linalg.norm(all_rows, axis=1, keepdims=True)
    parts = []
    offset = 0
    for i, count in enumerate((4, 0, 5, 8)):
        folder = root / "parts" / str(i)
        folder.mkdir(parents=True)
        for side in ("image", "text"):
            np.save(folder / f"{side}.npy", all_rows[offset:offset + count])
        files = {f"{s}.npy": {"bytes": (folder / f"{s}.npy").stat().st_size,
                               "sha256": sha256(folder / f"{s}.npy")} for s in ("image", "text")}
        receipt = {"fingerprint": "fixture", "rows": count, "dimension": 6, "files": files}
        atomic_json(folder / "completion.json", receipt)
        parts.append(receipt | {f"{s}_path": f"parts/{i}/{s}.npy" for s in ("image", "text")})
        offset += count
    atomic_json(root / "manifest.json", {
        "state": "completed", "fingerprint": "fixture", "identity": {"test": True},
        "rows": 17, "parts": parts,
    })
    return root, all_rows


def test_chunk_matrix_indexing_and_lru(tmp_path):
    root, expected = cache(tmp_path)
    matrix = ChunkedMatrix(root, "image", max_open_files=1)
    assert matrix.shape == expected.shape and len(matrix) == len(expected)
    for index in (0, -1, slice(None), slice(None, None, -2), [16, 0, 5, 5, -2], [],
                  np.array([[0, 9], [3, 15]]), np.arange(17) % 2 == 0):
        np.testing.assert_array_equal(matrix[index], expected[index])
        assert len(matrix._maps) <= 1
    retained = matrix[:3]
    matrix[-1]
    np.testing.assert_array_equal(retained, expected[:3])
    restored = pickle.loads(pickle.dumps(matrix))
    assert not restored._maps
    np.testing.assert_array_equal(restored[[12, 1]], expected[[12, 1]])
    matrix.close()
    restored.close()


@pytest.mark.parametrize("index", [17, -18, [1.5], np.array([True, False])])
def test_chunk_invalid_indices(tmp_path, index):
    root, _ = cache(tmp_path)
    with pytest.raises(IndexError):
        ChunkedMatrix(root, "text")[index]


def test_rejects_incomplete_and_corrupt_shapes(tmp_path):
    root, _ = cache(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["state"] = "running"
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="finish"):
        ChunkedMatrix(root, "image")
    manifest["state"] = "completed"
    atomic_json(path, manifest)
    np.save(root / "parts/0/image.npy", np.zeros((4, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        ChunkedMatrix(root, "image")[0]


def test_train_tiny_cpu_and_validate_completed_reuse(tmp_path):
    root, _ = cache(tmp_path)
    output = tmp_path / "training"
    cfg = {"latent_size": 12, "top_k": 3, "seed": 11, "max_open_files": 2, "arguments": {
        "max_steps": 2, "per_device_train_batch_size": 8,
        "disable_tqdm": True, "logging_steps": 1,
    }}
    result = train_pairs(root, output, cfg, "cpu")
    assert result["state"] == "completed" and result["optimizer_steps"] == 2
    assert result == train_pairs(root, output, cfg, "cpu")
    assert (output / "models/image/model.safetensors").exists()
    with pytest.raises(ValueError, match="configuration changed"):
        train_pairs(root, output, cfg | {"top_k": 2}, "cpu")
    weights = output / "models/text/model.safetensors"
    content = weights.read_bytes()
    weights.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="weights"):
        train_pairs(root, output, cfg, "cpu")


def test_rejects_modified_embedding_before_reuse(tmp_path):
    root, _ = cache(tmp_path)
    path = root / "parts/0/text.npy"
    values = np.load(path)
    values[0] *= 2
    np.save(path, values)
    with pytest.raises(ValueError, match="content changed"):
        train_pairs(root, tmp_path / "train", {}, "cpu")


def test_resume_real_trainer_checkpoint(tmp_path, monkeypatch):
    root, _ = cache(tmp_path)
    output = tmp_path / "training"
    cfg = {"latent_size": 12, "top_k": 3, "arguments": {
        "max_steps": 4, "per_device_train_batch_size": 8, "disable_tqdm": True,
        "logging_steps": 1, "save_strategy": "steps", "save_steps": 2,
    }}
    original = training.TrainingProgressCallback

    class InterruptAfterSave(original):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == 2:
                raise RuntimeError("test interruption after a complete checkpoint")

    monkeypatch.setattr(training, "TrainingProgressCallback", InterruptAfterSave)
    with pytest.raises(RuntimeError, match="test interruption"):
        train_pairs(root, output, cfg, "cpu")
    assert (output / "checkpoints/paired/checkpoint-2/trainer_state.json").exists()
    monkeypatch.setattr(training, "TrainingProgressCallback", original)
    result = train_pairs(root, output, cfg, "cpu")
    assert result["state"] == "completed" and result["optimizer_steps"] == 4
