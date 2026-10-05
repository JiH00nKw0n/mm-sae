"""Read unannotated paired activations for correspondence and retrieval studies.

The cache uses train2017/val2017 directory names for compatibility with the
mapping ablation runner. Those names do not establish the dataset identity.
Image IDs must be unique across the two splits, including across datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from mm_sae.analysis.data import SIDES


class PairedCachedSplit:
    def __init__(self, source: Path, name: str):
        self.source, self.name = source, name
        index = source / "index" / name
        records = json.loads((index / "images.json").read_text())
        if not isinstance(records, list) or not records:
            raise ValueError(f"Paired image index must be a nonempty list: {index}")
        ids = [record["image_id"] for record in records]
        if any(isinstance(value, bool) or not isinstance(value, (int, str)) for value in ids):
            raise ValueError(f"Image IDs must be strings or integers: {index}")
        if len(set(map(str, ids))) != len(ids):
            raise ValueError(f"Image IDs must be unique within a split: {index}")
        self.image_ids = np.asarray(ids)
        self.parents = np.load(index / "parents.npy", allow_pickle=False)
        if self.parents.ndim != 1 or not np.issubdtype(self.parents.dtype, np.integer):
            raise ValueError(f"Caption parents must be a one-dimensional integer array: {index}")
        if not len(self.parents) or np.any(self.parents < 0) or np.any(self.parents >= len(ids)):
            raise ValueError(f"Caption parent index outside image rows: {index}")
        if np.unique(self.parents).size != len(ids):
            raise ValueError(f"Every indexed image must have at least one caption: {index}")
        self.activations = {
            side: sparse.load_npz(source / "activations" / name / f"{side}.npz").tocsr() for side in SIDES
        }
        for side, expected in (("image", len(ids)), ("text", len(self.parents))):
            matrix = self.activations[side]
            if matrix.shape[0] != expected or matrix.shape[1] == 0:
                raise ValueError(f"Paired activation row count or feature width mismatch: {name}/{side}")
            # Validate bounded chunks rather than allocate a boolean per stored activation.
            for start in range(0, len(matrix.data), 1_000_000):
                if not np.isfinite(matrix.data[start : start + 1_000_000]).all():
                    raise ValueError(f"Paired activations contain non-finite values: {name}/{side}")
        self.groups = {"image": np.arange(len(ids)), "text": self.parents}


class PairedStudyData:
    def __init__(self, source, split_options):
        self.source = Path(source)
        self.train = PairedCachedSplit(self.source, split_options["fit_and_tune"])
        self.test = PairedCachedSplit(self.source, split_options["evaluation"])
        if np.intersect1d(self.train.image_ids.astype(str), self.test.image_ids.astype(str)).size:
            raise ValueError(
                "Fit/tune and evaluation image IDs overlap; namespace IDs from distinct datasets"
            )
        for side in SIDES:
            if self.train.activations[side].shape[1] != self.test.activations[side].shape[1]:
                raise ValueError(f"Training and evaluation feature widths differ: {side}")
        fraction = split_options["tune_image_fraction"]
        if not 0 < fraction < 1:
            raise ValueError("tune_image_fraction must be strictly between zero and one")
        rng = np.random.default_rng(split_options["seed"])
        order = rng.permutation(len(self.train.image_ids))
        nt = round(len(order) * fraction)
        if nt == 0 or nt == len(order):
            raise ValueError("The image split must leave nonempty fit and tune partitions")
        self.tune_images = np.zeros(len(order), bool)
        self.tune_images[order[:nt]] = True
        self.fit = {side: ~self.tune_images[self.train.groups[side]] for side in SIDES}
        self.tune = {side: ~self.fit[side] for side in SIDES}
        for partition in (self.fit, self.tune):
            if partition["text"].sum() < 2:
                raise ValueError("Each fit/tune partition must contain at least two caption pairs")
        if len(self.test.parents) < 2:
            raise ValueError("Evaluation must contain at least two caption pairs")

    def samples(self, side):
        return [
            (self.train, self.fit[side]),
            (self.train, self.tune[side]),
            (self.test, np.ones(self.test.activations[side].shape[0], bool)),
        ]
