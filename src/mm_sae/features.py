"""Cache CLIP embeddings once and retain sparse SAE activations for all requested edits."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from tqdm import tqdm

from .data.index import Index
from .cache import embedding_directory
from .io import atomic_json, write_csv, file_lock
from .training import load_saes
from .metrics.sparse_ops import activation_changes


def read_image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def mask_image(record, concept, rgb):
    image = np.array(read_image(record["image"]))
    with Image.open(record["mask"]) as im:
        mask = np.array(im)
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError("Image/annotation geometry mismatch")
    image[mask == concept] = rgb
    return Image.fromarray(image)


def encode_file(path: Path, n, dim, batch_size, encode_batch):
    # Different experiments may request the same shared embeddings concurrently.
    with file_lock(path.with_suffix(".lock")):
        return _encode_file(path, n, dim, batch_size, encode_batch)


def _encode_file(path: Path, n, dim, batch_size, encode_batch):
    if path.exists():
        a = np.load(path, mmap_mode="r")
        if a.shape != (n, dim):
            raise ValueError(f"Cached embedding shape mismatch: {path}")
        return a
    path.parent.mkdir(parents=True, exist_ok=True)
    partial, progress = path.with_suffix(".partial.npy"), path.with_suffix(".progress.json")
    start = json.loads(progress.read_text())["rows"] if progress.exists() else 0
    if start and not partial.exists():
        raise ValueError("Progress file exists without its embedding data")
    if n == 0:
        np.save(path, np.empty((0, dim), np.float32))
        return np.load(path, mmap_mode="r")
    if partial.exists():
        out = np.load(partial, mmap_mode="r+")
        if out.shape != (n, dim):
            raise ValueError("Partial cache has the wrong shape")
    else:
        out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(n, dim))
    for begin in tqdm(range(start, n, batch_size), desc=path.parent.name + "/" + path.stem, leave=False):
        end = min(n, begin + batch_size)
        result = encode_batch(begin, end)
        if result.shape != (end - begin, dim) or not np.isfinite(result).all():
            raise ValueError("Invalid embedding batch")
        out[begin:end] = result
        out.flush()
        atomic_json(progress, {"rows": end})
    del out
    os.replace(partial, path)
    progress.unlink(missing_ok=True)
    return np.load(path, mmap_mode="r")


def embed(config, encoder):
    for split in config.data.splits:
        index = Index(config.output, split)
        out = embedding_directory(config, split)
        encode_file(
            out / "image.npy",
            len(index.images),
            encoder.dim,
            config.encoder.batch_size,
            lambda a, b: encoder.images([read_image(r["image"]) for r in index.images[a:b]]),
        )
        encode_file(
            out / "text.npy",
            len(index.captions),
            encoder.dim,
            config.encoder.batch_size,
            lambda a, b: encoder.texts([r["text"] for r in index.captions[a:b]]),
        )


@torch.inference_mode()
def sparse_encode(model, embedding, path, batch_size):
    if path.exists():
        return sparse.load_npz(path).tocsr()
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, len(embedding), batch_size):
        x = torch.from_numpy(np.array(embedding[start : start + batch_size], dtype=np.float32)).to(
            model.device
        )
        values, indices = model.encode(x)
        v, j = values.float().cpu().numpy(), indices.cpu().numpy()
        i = np.repeat(np.arange(len(x)), v.shape[1])
        part = sparse.csr_matrix((v.ravel(), (i, j.ravel())), shape=(len(x), model.config.latent_size))
        part.eliminate_zeros()
        parts.append(part)
    result = (
        sparse.vstack(parts, format="csr")
        if parts
        else sparse.csr_matrix((0, model.config.latent_size), dtype=np.float32)
    )
    tmp = path.with_suffix(".partial.npz")
    sparse.save_npz(tmp, result)
    os.replace(tmp, path)
    return result


def original_latents(config, split, models=None):
    models = models or load_saes(config)
    root = config.output
    return tuple(
        sparse_encode(
            m,
            np.load(embedding_directory(config, split) / f"{side}.npy", mmap_mode="r"),
            root / "activations" / split / f"{side}.npz",
            config.features.batch_size,
        )
        for side, m in zip(["image", "text"], models)
    )


def counterfactual(config, encoder, index, split, concept, models):
    out = config.output / "counterfactual" / split / str(concept)
    out.mkdir(parents=True, exist_ok=True)
    image_rows = np.flatnonzero(index.presence[:, index.columns[concept]])
    text_rows = np.array(
        [i for i, r in enumerate(index.captions) if str(concept) in r["edits"]], dtype=np.int64
    )
    np.save(out / "image_rows.npy", image_rows)
    np.save(out / "text_rows.npy", text_rows)
    ie = encode_file(
        out / "image.npy",
        len(image_rows),
        encoder.dim,
        config.encoder.batch_size,
        lambda a, b: encoder.images(
            [mask_image(index.images[int(i)], concept, config.features.mask_rgb) for i in image_rows[a:b]]
        ),
    )
    te = encode_file(
        out / "text.npy",
        len(text_rows),
        encoder.dim,
        config.encoder.batch_size,
        lambda a, b: encoder.texts([index.captions[int(i)]["edits"][str(concept)] for i in text_rows[a:b]]),
    )
    xi = sparse_encode(models[0], ie, out / "image_activations.npz", config.features.batch_size)
    yt = sparse_encode(models[1], te, out / "text_activations.npz", config.features.batch_size)
    return image_rows, text_rows, xi, yt


def read_counterfactual(root, split, concept):
    out = root / "counterfactual" / split / str(concept)
    return (
        np.load(out / "image_rows.npy"),
        np.load(out / "text_rows.npy"),
        sparse.load_npz(out / "image_activations.npz"),
        sparse.load_npz(out / "text_activations.npz"),
    )


def save_changes(path, original, changed, source_rows, ids):
    changes = activation_changes(original[source_rows], changed)
    write_csv(
        path,
        [
            {
                "source_id": int(ids[int(row)]),
                "turned_off": int(a),
                "turned_on": int(b),
                "sum_abs_activation_change": float(d),
            }
            for row, a, b, d in zip(
                source_rows, changes["turned_off"], changes["turned_on"], changes["sum_abs_activation_change"]
            )
        ],
        ["source_id", "turned_off", "turned_on", "sum_abs_activation_change"],
    )
    sparse.save_npz(path.with_suffix(".delta.npz"), changed - original[source_rows])
