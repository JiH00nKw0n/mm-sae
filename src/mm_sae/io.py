"""Atomic artifacts and explicit resume signatures; incomplete stages are never marked complete."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager

import numpy as np


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default))
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def file_lock(path: Path, blocking=True):
    """Serialize writers on Linux/macOS; release the lock if a worker exits."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle, flags)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def code_digest():
    h = hashlib.sha256()
    from importlib.util import find_spec

    for name in ["mm_sae", "experiments"]:
        root = Path(find_spec(name).origin).parent
        for path in sorted(p for p in root.rglob("*") if p.suffix in {".py", ".yaml"}):
            h.update((name + "/" + path.relative_to(root).as_posix()).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


class RunStore:
    def __init__(self, config):
        self.root = config.output
        self.root.mkdir(parents=True, exist_ok=True)
        self.signature = {
            "config": config.digest(),
            "code": code_digest(),
            "input_definitions": {
                str(p): sha256(p)
                for p in [config.data.concepts_file, config.data.reviewed_captions]
                if p is not None
            },
        }
        self.config = config
        self.marks = self.root / "completed"

    def initialize(self):
        manifest = self.root / "run.json"
        if manifest.exists() and json.loads(manifest.read_text())["signature"] != self.signature:
            raise ValueError("Output belongs to different code/config. Choose a new output directory.")
        if not manifest.exists():
            atomic_json(
                manifest,
                {
                    "signature": self.signature,
                    "config": self.config.model_dump(mode="json"),
                    "synthetic_test_only": self.config.data.source == "synthetic",
                },
            )
        self.marks.mkdir(exist_ok=True)

    def done(self, stage):
        return (self.marks / f"{stage}.json").exists()

    def complete(self, stage):
        atomic_json(self.marks / f"{stage}.json", self.signature)

    def require(self, *stages):
        for stage in stages:
            if not self.done(stage):
                raise ValueError(f"Run stage {stage!r} first")

    @contextmanager
    def lock(self):
        with file_lock(self.root / ".lock", blocking=False):
            self.initialize()
            yield
