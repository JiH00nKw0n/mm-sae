"""Read cached representations and split every view by its parent image."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from mm_sae.io import atomic_json, sha256

SIDES = ("image", "text")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    atomic_json(path, clean(value))


class CachedSplit:
    def __init__(self, source: Path, name: str):
        self.source, self.name = source, name
        index = source / "index" / name
        self.ids = json.loads((index / "concept_ids.json").read_text())
        self.columns = {int(c): j for j, c in enumerate(self.ids)}
        self.image_ids = np.array([r["image_id"] for r in json.loads((index / "images.json").read_text())])
        self.parents = np.load(index / "parents.npy")
        self.presence = np.load(index / "presence.npy", mmap_mode="r")
        self.mentions = np.load(index / "mentions.npy", mmap_mode="r")
        self.activations = {s: sparse.load_npz(source / "activations" / name / f"{s}.npz").tocsr()
                            for s in SIDES}
        self.groups = {"image": np.arange(len(self.image_ids)), "text": self.parents}

    def labels(self, side, cid):
        return np.asarray((self.presence if side == "image" else self.mentions)[:, self.columns[cid]])

    def removed(self, side, cid):
        root = self.source / "counterfactual" / self.name / str(cid)
        rows = np.load(root / f"{side}_rows.npy")
        values = sparse.load_npz(root / f"{side}_activations.npz").tocsr()
        if len(rows) != values.shape[0]:
            raise ValueError(f"Counterfactual row mismatch: {root} {side}")
        return rows, values


class StudyData:
    def __init__(self, source, split_options):
        self.source = Path(source)
        self.train = CachedSplit(self.source, split_options["fit_and_tune"])
        self.test = CachedSplit(self.source, split_options["evaluation"])
        assert self.train.ids == self.test.ids
        assert not np.intersect1d(self.train.image_ids, self.test.image_ids).size
        rng = np.random.default_rng(split_options["seed"])
        order = rng.permutation(len(self.train.image_ids))
        nt = round(len(order) * split_options["tune_image_fraction"])
        self.tune_images = np.zeros(len(order), bool)
        self.tune_images[order[:nt]] = True
        self.fit = {s: ~self.tune_images[self.train.groups[s]] for s in SIDES}
        self.tune = {s: ~self.fit[s] for s in SIDES}
        self.names = {int(c["id"]): c["name"] for c in json.loads(
            (self.source / "dataset.json").read_text())["concepts"]}

    def samples(self, side):
        return [(self.train, self.fit[side]), (self.train, self.tune[side]),
                (self.test, np.ones(self.test.activations[side].shape[0], bool))]


def verify_source(source, output, split_names, ids):
    paths = [source / "dataset.json", source / "run.json", source / "representatives.json",
             source / "panel.npz", source / "models/frozen.json"]
    for side in SIDES:
        paths += [source / "models" / side / "model.safetensors", source / "models" / side / "config.json"]
    for split in split_names:
        paths += [source / "index" / split / name for name in
                  ["images.json", "concept_ids.json", "parents.npy", "presence.npy", "mentions.npy"]]
        paths += [source / "activations" / split / f"{s}.npz" for s in SIDES]
        for cid in ids:
            paths += [source / "counterfactual" / split / str(cid) / f"{s}_{suffix}"
                      for s in SIDES for suffix in ["rows.npy", "activations.npz"]]
    from mm_sae.progress import iter_progress
    records = {}
    for path in iter_progress(paths, "Verify source cache hashes", unit="files"):
        if not path.exists():
            raise FileNotFoundError(f"Required cache missing; inference will not be started: {path}")
        records[str(path.relative_to(source))] = sha256(path)
    dest = output / "source_hashes.json"
    if dest.exists() and json.loads(dest.read_text()) != records:
        raise ValueError("Source cache changed. Use a new output directory.")
    atomic_json(dest, records)
