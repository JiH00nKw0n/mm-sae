from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np


from mm_sae.io import sha256


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Embedding part paths must remain inside their cache")
    return path


def _manifest(root: Path) -> dict[str, Any]:
    value = json.loads((root / "manifest.json").read_text())
    if value.get("state") != "completed":
        raise ValueError("Paired embedding extraction must finish before training")
    parts = value["parts"]
    if not parts or value["rows"] <= 0 or sum(p["rows"] for p in parts) != value["rows"]:
        raise ValueError("Paired embedding manifest has inconsistent row counts")
    dims = {p["dimension"] for p in parts}
    if len(dims) != 1 or next(iter(dims)) <= 0 or any(p["rows"] < 0 for p in parts):
        raise ValueError("Paired embedding parts must have a common positive dimension")
    return value


class ChunkedMatrix:
    """Array-like rows across cache parts, with a bounded LRU of open memmaps.

    Returned arrays own their storage, so evicting a map never invalidates a
    previous result. Random index order, duplicates and negative indices follow
    NumPy semantics. No full embedding table is assembled in memory.
    """

    def __init__(
        self,
        root: Path,
        modality: Literal["image", "text"],
        max_open_files: int = 16,
    ):
        if modality not in ("image", "text") or max_open_files < 1:
            raise ValueError("Choose image/text and a positive open-file limit")
        self.root, self.modality, self.max_open_files = Path(root), modality, max_open_files
        self.manifest = _manifest(self.root)
        self.parts = self.manifest["parts"]
        self.shape = (int(self.manifest["rows"]), int(self.parts[0]["dimension"]))
        self.dtype = np.dtype(np.float32)
        self.offsets = np.concatenate(([0], np.cumsum([p["rows"] for p in self.parts])))
        self._maps: OrderedDict[int, np.memmap] = OrderedDict()

    def __len__(self) -> int:
        return self.shape[0]

    def _open(self, part_index: int) -> np.memmap:
        if part_index in self._maps:
            self._maps.move_to_end(part_index)
            return self._maps[part_index]
        while len(self._maps) >= self.max_open_files:
            _, old = self._maps.popitem(last=False)
            old._mmap.close()  # type: ignore[union-attr]
        part = self.parts[part_index]
        value = np.load(_path(self.root, part[f"{self.modality}_path"]), mmap_mode="r", allow_pickle=False)
        if value.shape != (part["rows"], self.shape[1]) or value.dtype != self.dtype:
            value._mmap.close()
            raise ValueError("Embedding part shape or dtype disagrees with its manifest")
        self._maps[part_index] = value
        return value

    def __getitem__(self, index):
        if isinstance(index, slice):
            rows = np.arange(*index.indices(len(self)), dtype=np.int64)
        else:
            rows = np.asarray(index)
            if rows.dtype == bool:
                if rows.shape != (len(self),):
                    raise IndexError("Boolean embedding index must have one entry per row")
                rows = np.flatnonzero(rows)
            elif rows.size == 0:
                rows = rows.astype(np.int64)
            elif not np.issubdtype(rows.dtype, np.integer):
                raise IndexError("Embedding row indices must be integers")
        result_shape = rows.shape + (self.shape[1],)
        rows = rows.astype(np.int64, copy=True).reshape(-1)
        rows[rows < 0] += len(self)
        if np.any((rows < 0) | (rows >= len(self))):
            raise IndexError("Embedding row index out of bounds")
        output = np.empty((rows.size, self.shape[1]), dtype=np.float32)
        part_indices = np.searchsorted(self.offsets[1:], rows, side="right")
        for part_index in np.unique(part_indices):
            positions = np.flatnonzero(part_indices == part_index)
            output[positions] = self._open(int(part_index))[rows[positions] - self.offsets[part_index]]
        return output.reshape(result_shape)

    def close(self):
        for value in self._maps.values():
            value._mmap.close()  # type: ignore[union-attr]
        self._maps.clear()

    def __getstate__(self):
        return self.__dict__ | {"_maps": OrderedDict()}

    def __del__(self):
        if hasattr(self, "_maps"):
            self.close()


def _verified_identity(root: Path, manifest: dict) -> dict:
    """Verify receipts and content, including on completed-training reuse."""
    for part in manifest["parts"]:
        directory = _path(root, part["image_path"]).parent
        if any(_path(root, part[f"{side}_path"]) != directory / f"{side}.npy" for side in ("image", "text")):
            raise ValueError("Paired arrays must refer to files covered by their part receipt")
        receipt = json.loads((directory / "completion.json").read_text())
        if any(part.get(key) != value for key, value in receipt.items()):
            raise ValueError("Manifest and embedding part receipt disagree")
        if receipt["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Embedding receipt has a different source fingerprint")
        for filename, expected in receipt["files"].items():
            path = _path(root, str(directory.relative_to(root.resolve()) / filename))
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Embedding content changed: {path}")
    return {key: manifest[key] for key in ("fingerprint", "identity", "rows", "parts")}
