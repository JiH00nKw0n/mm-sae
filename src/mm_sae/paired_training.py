"""Train independent, jointly stepped SAEs from bounded-memory paired caches."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from transformers import Trainer, TrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .io import atomic_json, file_lock, sha256
from .models.encoder import device_for
from .models.sae import PairedSAEs, SAEConfig, TopKSAE
from .progress import progress_task
from .training import DecoderNormCallback, TrainingProgressCallback, cached_dataset, frozen_fingerprint


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
        self, root: Path, modality: Literal["image", "text"], max_open_files: int = 16,
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
        if any(_path(root, part[f"{side}_path"]) != directory / f"{side}.npy"
               for side in ("image", "text")):
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


def train_pairs(embedding_root: Path, output: Path, cfg: dict, device: str) -> dict:
    """Train a paired SAE with HF Trainer; return saved training metadata.

    ``cfg`` holds seed, latent_size (per modality), top_k, and HF ``arguments``.
    Mean/std activation preprocessing belongs to later alignment, not training.
    Inputs here are the extractor's L2-normalized frozen encoder embeddings.
    """
    embedding_root, output = Path(embedding_root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".training.lock", blocking=False):
        return _train_pairs(embedding_root, output, cfg, device)


def _train_pairs(embedding_root: Path, output: Path, cfg: dict, device: str) -> dict:
    manifest = _manifest(embedding_root)
    identity = _verified_identity(embedding_root, manifest)
    arguments = {
        "num_train_epochs": 10, "per_device_train_batch_size": 1024,
        "learning_rate": 5e-4, "weight_decay": 1e-5, "adam_beta1": .9,
        "adam_beta2": .999, "adam_epsilon": 1e-8, "optim": "adamw_torch",
        "lr_scheduler_type": "cosine", "warmup_ratio": .05, "max_grad_norm": 1.,
        "gradient_accumulation_steps": 1, "fp16": False, "bf16": False,
        "dataloader_num_workers": 0, "dataloader_drop_last": False,
        "save_strategy": "epoch", "save_total_limit": 2, "logging_steps": 50,
        "eval_strategy": "no", "seed": int(cfg.get("seed", 0)),
        "data_seed": int(cfg.get("seed", 0)),
    } | cfg.get("arguments", {})
    reserved = {"output_dir", "remove_unused_columns", "report_to", "use_cpu", "dataloader_pin_memory"}
    if reserved & arguments.keys():
        raise ValueError("Training arguments contain runner-managed settings")
    if arguments["gradient_accumulation_steps"] != 1 or arguments["dataloader_drop_last"]:
        raise ValueError("Paired reference training uses accumulation 1 and retains the final batch")
    if device not in ("cpu", "cuda"):
        raise ValueError("Paired training currently supports cpu or cuda")
    device_for(device)
    batches = math.ceil(manifest["rows"] / arguments["per_device_train_batch_size"])
    steps = int(arguments.get("max_steps", -1))
    if steps <= 0:
        steps = math.ceil(batches * arguments["num_train_epochs"])
    if not arguments.get("warmup_steps", 0):
        arguments["warmup_steps"] = max(1, int(arguments["warmup_ratio"] * steps))
    model_settings = {
        "hidden_size": manifest["parts"][0]["dimension"],
        "latent_size": int(cfg.get("latent_size", 4096)), "k": int(cfg.get("top_k", 32)),
    }
    signature_payload = {
        "cache": identity, "model": model_settings, "arguments": arguments,
        "initialization_seed": int(cfg.get("seed", 0)),
        "implementation": {
            "paired_training": sha256(Path(__file__)),
            "sae": sha256(Path(__file__).parent / "models" / "sae.py"),
            "training": sha256(Path(__file__).parent / "training.py"),
        },
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    path = output / "train.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["signature"] != signature:
            raise ValueError("Training inputs or configuration changed; use a separate output directory")
        if previous["state"] == "completed":
            actual = frozen_fingerprint(output)
            frozen = json.loads((output / "models" / "frozen.json").read_text())
            if actual != frozen or actual != previous["models"]:
                raise ValueError("Completed SAE weights or configuration changed")
            return previous
    elif (output / "checkpoints").exists() or (output / "models").exists():
        raise ValueError("Existing training artifacts lack their input signature")
    metadata = {
        "signature": signature, "signature_payload": signature_payload,
        "state": "running", "rows_per_epoch": manifest["rows"],
        "planned_optimizer_steps": steps, "sampling": "paired_repeat_image",
        "trainer": "transformers.Trainer",
    }
    atomic_json(path, metadata)
    max_open = int(cfg.get("max_open_files", 16))
    image = ChunkedMatrix(embedding_root, "image", max_open)
    text = ChunkedMatrix(embedding_root, "text", max_open)
    try:
        set_seed(int(cfg.get("seed", 0)))
        model_config = SAEConfig(**model_settings)
        pair = PairedSAEs(TopKSAE(model_config), TopKSAE(model_config))
        checkpoint_dir = output / "checkpoints" / "paired"
        args = TrainingArguments(
            **arguments, output_dir=str(checkpoint_dir), report_to=[], remove_unused_columns=False,
            use_cpu=device == "cpu", dataloader_pin_memory=device == "cuda",
        )
        trainer = Trainer(
            model=pair, args=args, train_dataset=cached_dataset(image, text, np.arange(len(image))),
            data_collator=default_data_collator, callbacks=[DecoderNormCallback()],
        )
        last = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
        with progress_task("Train paired image/text SAEs", steps, "optimizer steps") as meter:
            trainer.add_callback(TrainingProgressCallback(meter))
            trainer.train(resume_from_checkpoint=last)
        trainer.save_state()
        for side, model in (("image", pair.image_sae), ("text", pair.text_sae)):
            model.save_pretrained(output / "models" / side, safe_serialization=True)
        fingerprints = frozen_fingerprint(output)
        atomic_json(output / "models" / "frozen.json", fingerprints)
        metadata.update(
            state="completed", optimizer_steps=trainer.state.global_step,
            logs=trainer.state.log_history, models=fingerprints,
        )
        atomic_json(path, metadata)
        return metadata
    except BaseException as exc:
        metadata.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(path, metadata)
        raise
    finally:
        image.close()
        text.close()
