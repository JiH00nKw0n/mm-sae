"""Bounded-memory, resumable frozen embeddings from paired WebDataset shards.

Only encoded arrays are retained. Hugging Face owns archive reading and image
decoding; the optional local-shard mode keeps at most one downloaded source TAR.
An interrupted shard is rebuilt, while completed shards are never re-encoded.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from collections.abc import Callable, Generator, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import quote

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mm_sae.io import atomic_json, file_lock, sha256
from mm_sae.progress import progress_task

if TYPE_CHECKING:
    from mm_sae.config import EncoderConfig


class PairedEncoder(Protocol):
    dim: int

    def images(self, images: list[Image.Image]) -> np.ndarray: ...
    def texts(self, texts: list[str]) -> np.ndarray: ...


class ShardEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    shard_paths: list[str] = Field(min_length=1)
    image_column: str = "jpg"
    caption_column: str = "txt"
    batch_size: int = Field(default=128, ge=1)
    staging: Literal["stream", "local_shard"] = "stream"
    max_shards: int | None = Field(default=None, ge=1)
    max_samples: int | None = Field(default=None, ge=1)
    estimated_rows: int | None = Field(default=None, ge=1)
    estimated_source_bytes: int | None = Field(default=None, ge=1)
    verify_cached_hashes: bool = True

    @model_validator(mode="after")
    def valid_paths(self):
        if len(set(self.shard_paths)) != len(self.shard_paths):
            raise ValueError("Source shard paths must be unique")
        for path in self.shard_paths:
            if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                raise ValueError("Source shard paths must be relative dataset paths")
        return self


def _stream_rows(config: ShardEmbeddingConfig, shard: str, download_dir: Path) -> Generator[dict, None, None]:
    # streaming=True never creates a materialized Hugging Face dataset cache.
    from datasets import Image as HFImage
    from datasets import load_dataset

    if config.staging == "local_shard":
        from huggingface_hub import hf_hub_download

        source = hf_hub_download(
            repo_id=config.dataset_id,
            repo_type="dataset",
            revision=config.revision,
            filename=shard,
            local_dir=download_dir,
        )
    else:
        source = (
            f"https://huggingface.co/datasets/{config.dataset_id}/resolve/"
            f"{config.revision}/{quote(shard, safe='/')}"
        )
    dataset = load_dataset("webdataset", data_files={"train": [source]}, split="train", streaming=True)
    # Decode inside our loop so one malformed JPEG can be recorded and excluded
    # without terminating the Hugging Face streaming iterator for the whole TAR.
    dataset = dataset.cast_column(config.image_column, HFImage(decode=False))
    for row in dataset:
        if not isinstance(row, dict):
            raise TypeError("Paired streaming datasets must yield individual row dictionaries")
        yield row


def _decode_image(raw: Any) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    from datasets import Image as HFImage

    decoded = HFImage().decode_example(raw)
    try:
        return decoded.convert("RGB")
    finally:
        decoded.close()


def _write_npy(raw: Path, output: Path, rows: int, dim: int, batch_size: int):
    target = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(rows, dim))
    if rows:
        source = np.memmap(raw, mode="r", dtype=np.float32, shape=(rows, dim))
        for start in range(0, rows, batch_size):
            target[start : start + batch_size] = source[start : start + batch_size]
        del source
    target.flush()
    del target
    raw.unlink()


def _validate_embeddings(value: np.ndarray, rows: int, dim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (rows, dim) or not np.isfinite(result).all():
        raise ValueError("Encoder returned incorrect embedding shape or non-finite values")
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("The paired embedding cache requires L2-normalized encoder outputs")
    return result


def _extract_part(
    config: ShardEmbeddingConfig,
    shard: str,
    part: Path,
    download_dir: Path,
    encoder: PairedEncoder,
    fingerprint: str,
    limit: int | None,
    notify: Callable[[int, int, int], None],
) -> dict:
    pending = part.with_name(part.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    images: list[Image.Image] = []
    captions: list[str] = []
    keys: list[dict] = []
    encoded, read, excluded = 0, 0, 0
    started = time.monotonic()
    exhausted = False
    with (
        (pending / "image.raw").open("wb") as image_file,
        (pending / "text.raw").open("wb") as text_file,
        (pending / "keys.json").open("w") as key_file,
        (pending / "excluded.jsonl").open("w") as excluded_file,
    ):
        key_file.write("[")

        def flush():
            nonlocal encoded
            if not images:
                return
            try:
                image_values = _validate_embeddings(encoder.images(images), len(images), encoder.dim)
                text_values = _validate_embeddings(encoder.texts(captions), len(images), encoder.dim)
                image_values.tofile(image_file)
                text_values.tofile(text_file)
                for record in keys:
                    if encoded:
                        key_file.write(",\n")
                    key_file.write(json.dumps(record, ensure_ascii=False))
                    encoded += 1
            finally:
                for image in images:
                    image.close()
                images.clear()
                captions.clear()
                keys.clear()
            notify(encoded, read, excluded)

        stream = _stream_rows(config, shard, download_dir)
        try:
            for row_number, row in enumerate(stream):
                read += 1
                key = str(row.get("__key__", f"row-{row_number}"))
                reason = None
                caption = row.get(config.caption_column)
                raw_image = row.get(config.image_column)
                if not isinstance(caption, str) or not caption.strip():
                    reason = "missing_or_empty_caption"
                elif raw_image is None:
                    reason = "missing_image"
                else:
                    try:
                        image = _decode_image(raw_image)
                    except (
                        OSError, ValueError, TypeError, KeyError,
                        UnidentifiedImageError, Image.DecompressionBombError,
                    ) as exc:
                        reason = f"image_decode_error:{type(exc).__name__}"
                    else:
                        images.append(image)
                        captions.append(caption)
                        keys.append({"key": key, "source_row": row_number})
                if reason:
                    excluded += 1
                    excluded_file.write(
                        json.dumps({"key": key, "source_row": row_number, "reason": reason}) + "\n"
                    )
                    if excluded % config.batch_size == 0:
                        notify(encoded, read, excluded)
                if len(images) >= config.batch_size:
                    flush()
                if limit is not None and encoded + len(images) >= limit:
                    flush()
                    break
            else:
                exhausted = True
            flush()
            key_file.write("]\n")
        finally:
            stream.close()
            for image in images:
                image.close()
        for handle in (image_file, text_file, key_file, excluded_file):
            handle.flush()
            os.fsync(handle.fileno())
    for modality in ("image", "text"):
        _write_npy(pending / f"{modality}.raw", pending / f"{modality}.npy", encoded, encoder.dim, 4096)
    files = {
        name: {"bytes": (pending / name).stat().st_size, "sha256": sha256(pending / name)}
        for name in ("image.npy", "text.npy", "keys.json", "excluded.jsonl")
    }
    receipt = {
        "fingerprint": fingerprint,
        "shard": shard,
        "rows": encoded,
        "source_rows_read": read,
        "excluded_rows": excluded,
        "source_exhausted": exhausted,
        "dimension": encoder.dim,
        "dtype": "float32",
        "elapsed_seconds": time.monotonic() - started,
        "files": files,
    }
    atomic_json(pending / "completion.json", receipt)
    os.replace(pending, part)
    if download_dir.exists():
        shutil.rmtree(download_dir)
    return receipt


def _read_part(part: Path, fingerprint: str, shard: str, verify_hashes: bool) -> dict:
    receipt = json.loads((part / "completion.json").read_text())
    if receipt["fingerprint"] != fingerprint or receipt["shard"] != shard:
        raise ValueError(f"Completed shard does not match this extraction: {part}")
    for name, properties in receipt["files"].items():
        path = part / name
        if path.stat().st_size != properties["bytes"]:
            raise ValueError(f"Completed shard file size changed: {path}")
        if verify_hashes and sha256(path) != properties["sha256"]:
            raise ValueError(f"Completed shard file checksum changed: {path}")
    for modality in ("image", "text"):
        value = np.load(part / f"{modality}.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != (receipt["rows"], receipt["dimension"]) or value.dtype != np.float32:
            raise ValueError(f"Completed shard array shape or dtype changed: {part}")
    return receipt


def extract(
    config: Mapping[str, Any] | ShardEmbeddingConfig,
    output: Path,
    *,
    encoder_config: EncoderConfig | Mapping[str, Any],
    encoder: PairedEncoder | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Extract paired arrays, returning manifest with mmap-ready relative paths.

    ``max_samples`` caps successfully encoded pairs, across the selected shards.
    Limits are part of the cache identity, so a smoke cache cannot silently be
    used as a complete dataset. Source/decode failures outside individual images
    abort the current shard and are never treated as successful completion.
    """
    from mm_sae.config import EncoderConfig

    settings = ShardEmbeddingConfig.model_validate(config)
    encoder_settings = EncoderConfig.model_validate(encoder_config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shard_paths = settings.shard_paths[: settings.max_shards]
    identity = {
        "format_version": 1,
        "source": settings.model_dump(
            exclude={"batch_size", "staging", "verify_cached_hashes", "estimated_rows", "estimated_source_bytes"}
        ),
        "encoder": encoder_settings.model_dump(mode="json", exclude={"device", "batch_size"}),
        "implementation": {
            "paired_embeddings.py": sha256(Path(__file__)),
            "encoder.py": sha256(Path(__file__).parents[1] / "models" / "encoder.py"),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("datasets", "transformers", "Pillow", "numpy", "torch")
        },
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    with file_lock(output / ".lock", blocking=False):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing["fingerprint"] != fingerprint:
                raise ValueError("Embedding cache belongs to a different source, encoder, code, or sample limit")
        manifest = {
            "fingerprint": fingerprint,
            "identity": identity,
            "config": settings.model_dump(),
            "rows": 0,
            "parts": [],
            "state": "running",
        }
        started = time.monotonic()
        initial = 0
        verified: set[int] = set()
        try:
            for number, shard in enumerate(shard_paths):
                part = output / "parts" / f"{number:05d}"
                if not part.exists():
                    break
                initial += _read_part(part, fingerprint, shard, settings.verify_cached_hashes)["rows"]
                verified.add(number)
        except Exception as exc:
            manifest["state"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(manifest_path, manifest)
            atomic_json(output / "embedding_progress.json", {"state": "failed", "error": manifest["error"]})
            raise
        atomic_json(manifest_path, manifest)
        total = settings.max_samples or settings.estimated_rows
        with progress_task("Encode paired image-caption shards", total, "pairs", initial=initial) as progress:
            try:
                for number, shard in enumerate(shard_paths):
                    completed = int(manifest["rows"])
                    if settings.max_samples is not None and completed >= settings.max_samples:
                        break
                    part = output / "parts" / f"{number:05d}"
                    download_dir = output / ".downloads" / part.name

                    def notify(rows: int, source_rows: int, excluded: int):
                        done = completed + rows
                        progress.update(
                            done, shard=shard, shard_index=number, shards_total=len(shard_paths),
                            source_rows_in_shard=source_rows, excluded_in_shard=excluded,
                            total_is_estimate=settings.max_samples is None,
                        )
                        snapshot = progress.snapshot() | {"shard": shard, "state": "running"}
                        atomic_json(output / "embedding_progress.json", snapshot)
                        if on_progress is not None:
                            on_progress(snapshot)

                    if part.exists():
                        receipt = _read_part(
                            part, fingerprint, shard, settings.verify_cached_hashes and number not in verified
                        )
                        # A crash after the atomic commit can leave a downloaded TAR.
                        if download_dir.exists():
                            shutil.rmtree(download_dir)
                    else:
                        if encoder is None:
                            from mm_sae.models.encoder import CLIPEncoder

                            if encoder_settings.backend != "clip":
                                raise ValueError("Provide an explicit encoder for non-CLIP extraction tests")
                            encoder = CLIPEncoder(encoder_settings)
                        limit = None if settings.max_samples is None else settings.max_samples - completed
                        receipt = _extract_part(
                            settings, shard, part, download_dir, encoder, fingerprint, limit, notify
                        )
                    entry = receipt | {
                        "image_path": f"parts/{part.name}/image.npy",
                        "text_path": f"parts/{part.name}/text.npy",
                        "keys_path": f"parts/{part.name}/keys.json",
                        "excluded_path": f"parts/{part.name}/excluded.jsonl",
                    }
                    manifest["parts"].append(entry)
                    manifest["rows"] = completed + receipt["rows"]
                    atomic_json(manifest_path, manifest)
                    progress.update(max(initial, manifest["rows"]))
                manifest["state"] = "completed"
                manifest["invocation_elapsed_seconds"] = time.monotonic() - started
                manifest["source_rows_read"] = sum(p["source_rows_read"] for p in manifest["parts"])
                manifest["excluded_rows"] = sum(p["excluded_rows"] for p in manifest["parts"])
                manifest["all_selected_sources_exhausted"] = (
                    len(manifest["parts"]) == len(shard_paths)
                    and all(p["source_exhausted"] for p in manifest["parts"])
                )
                progress.total = manifest["rows"]
                atomic_json(manifest_path, manifest)
                atomic_json(output / "embedding_progress.json", progress.snapshot() | {"state": "completed"})
            except BaseException as exc:
                manifest["state"] = "failed"
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                atomic_json(manifest_path, manifest)
                atomic_json(
                    output / "embedding_progress.json",
                    progress.snapshot() | {"state": "failed", "error": manifest["error"]},
                )
                raise
    return manifest
