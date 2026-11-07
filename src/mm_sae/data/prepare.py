"""Turn native annotations into ordered embedding tables shared by all methods."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import numpy as np
from PIL import Image
from mm_sae.io import atomic_json, sha256, file_lock
from .concepts import coco, cub, visual_genome
from .retrieval import retrieval
from .edits import edited_pairs
from .paired_embeddings import _write_npy


def records(kind, files, seed):
    if kind in ("coco", "stuff"):
        return coco(files, seed, stuff=kind == "stuff")
    if kind == "cub":
        return cub(files, seed)
    if kind == "visual_genome":
        return visual_genome(files, seed)
    if kind in ("syncoco", "syncub"):
        return edited_pairs(kind, files)
    return retrieval(kind, files)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def prepare(spec, root, dest, encoder, seed):
    files = {k: (root / v).resolve() for k, v in spec.files.items()}
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    sources = {}
    for k, path in files.items():
        if path.is_file():
            sources[k] = sha256(path)
        else:
            h = hashlib.sha256()
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    stat = f.stat()
                    h.update(f"{f.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
            sources[k] = h.hexdigest()
    contract = {
        "dataset": spec.dataset,
        "files": {k: str(v) for k, v in files.items()},
        "split_seed": seed,
        "encoder": encoder.config.model_dump(exclude={"device", "batch_size"}),
    }
    readers = sorted(Path(__file__).parent.glob("*.py"))
    signature = digest([contract, sources, [sha256(p) for p in readers]])
    with file_lock(dest.parent / (dest.name + ".lock")):
        if dest.exists():
            manifest = json.loads((dest / "manifest.json").read_text())
            if manifest["signature"] != signature:
                raise ValueError(f"Dataset or encoder changed; choose a new benchmark cache: {dest}")
            return
        with TemporaryDirectory(dir=dest.parent) as tmp:
            build = Path(tmp) / "table"
            build.mkdir()
            batches = {"image": [], "text": []}
            counts = {"image": 0, "text": 0}
            seen = set()

            def flush(side):
                batch = batches[side]
                if not batch:
                    return
                if side == "image":
                    images = []
                    for r in batch:
                        with Image.open(r.value) as im:
                            images.append(im.convert("RGB"))
                    vectors = encoder.images(images)
                    lengths = [None] * len(batch)
                else:
                    texts = [r.value for r in batch]
                    vectors = encoder.texts(texts)
                    lengths = encoder.text_lengths(texts)
                if vectors.shape != (len(batch), encoder.dim) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid frozen encoder output")
                with (build / (side + ".raw")).open("ab") as f:
                    np.asarray(vectors, np.float32).tofile(f)
                with (build / (side + ".jsonl")).open("a") as f:
                    for r, n in zip(batch, lengths):
                        meta = asdict(r)
                        del meta["value"]
                        if n is not None:
                            meta["truncated"] = n > encoder.config.text_max_length
                        f.write(json.dumps(meta) + "\n")
                counts[side] += len(batch)
                batch.clear()

            for row in records(spec.dataset, files, seed):
                key = (row.side, row.id)
                if key in seen:
                    raise ValueError(f"Duplicate sample {key}")
                seen.add(key)
                batch = batches[row.side]
                batch.append(row)
                if len(batch) >= encoder.config.batch_size:
                    flush(row.side)
            for side in batches:
                flush(side)
                if counts[side]:
                    _write_npy(
                        build / (side + ".raw"), build / (side + ".npy"), counts[side], encoder.dim, 1024
                    )
            if not any(counts.values()):
                raise ValueError(f"Empty benchmark: {spec.dataset}")
            atomic_json(
                build / "manifest.json",
                {"signature": signature, "rows": counts, "source": sources, "contract": contract},
            )
            os.rename(build, dest)
