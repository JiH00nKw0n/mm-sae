"""Prepare unannotated paired data with HF encoders and train SAEs with HF Trainer.

Example: python -m experiments.paired_corpus.run --config configs/cc3m-server.yaml
Dataset identity and limits are recorded. Training and evaluation use separate
corpora; evaluation embeddings can be reused only with the identical encoder.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi

from mm_sae.cache import embedding_directory
from mm_sae.config import Config, EncoderConfig
from mm_sae.data.paired_embeddings import extract
from mm_sae.features import sparse_encode
from mm_sae.io import atomic_json, file_lock, sha256
from mm_sae.models.sae import TopKSAE
from mm_sae.progress import ProgressReporter, stage_progress
from mm_sae.training import assert_frozen


def resolve_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    for field in ("output", "embedding_cache"):
        cfg[field] = str((path.parent / cfg[field]).resolve())
    cfg["evaluation"]["source_run"] = str((path.parent / cfg["evaluation"]["source_run"]).resolve())
    for followup in cfg.get("followups", []):
        followup["config"] = str((path.parent / followup["config"]).resolve())
    return cfg


def shard_configuration(cfg: dict, root: Path) -> dict:
    """Pin the archive inventory once rather than depending on a mutable listing."""
    source = dict(cfg["source"])
    pattern = source.pop("shard_pattern", "*train*.tar")
    path = root / "sources.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["request"] != cfg["source"]:
            raise ValueError("The source request changed; use a new run directory")
        return saved["extraction"]
    from fnmatch import fnmatch

    info = HfApi().dataset_info(source["dataset_id"], revision=source["revision"], files_metadata=True)
    if info.sha != source["revision"]:
        raise ValueError("The requested dataset revision did not resolve to the pinned commit")
    files = sorted((f for f in info.siblings or [] if fnmatch(f.rfilename, pattern)),
                   key=lambda f: f.rfilename)
    if not files:
        raise ValueError("No source archives matched the configured pattern")
    source["shard_paths"] = [f.rfilename for f in files]
    source["estimated_source_bytes"] = sum(f.size or 0 for f in files)
    atomic_json(path, {"request": cfg["source"], "extraction": source})
    return source


def evaluation_inputs(cfg: dict, encoder: EncoderConfig):
    source = Path(cfg["evaluation"]["source_run"])
    source_config = Config.model_validate(json.loads((source / "run.json").read_text())["config"])
    source_config.output = source
    exclude = {"device", "batch_size"}
    if source_config.encoder.model_dump(exclude=exclude) != encoder.model_dump(exclude=exclude):
        raise ValueError("Evaluation embeddings use a different frozen encoder")
    split = cfg["evaluation"].get("split", "val2017")
    embeddings = embedding_directory(source_config, split)
    records = json.loads((source / "index" / split / "images.json").read_text())
    parents = np.load(source / "index" / split / "parents.npy")
    matrices = {side: np.load(embeddings / f"{side}.npy", mmap_mode="r") for side in ("image", "text")}
    if len(matrices["image"]) != len(records) or len(matrices["text"]) != len(parents):
        raise ValueError("Evaluation embedding and index row counts differ")
    if not records or not len(parents):
        raise ValueError("Evaluation data must contain images and captions")
    if parents.ndim != 1 or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("Evaluation parents must be a one-dimensional integer array")
    if np.any(parents < 0) or np.any(parents >= len(records)):
        raise ValueError("Evaluation caption parent is outside the image index")
    if len(np.unique(parents)) != len(records):
        raise ValueError("Every evaluation image must have at least one caption")
    for values in matrices.values():
        if values.ndim != 2 or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Evaluation embeddings must be finite float32 matrices")
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4, rtol=1e-4):
            raise ValueError("Evaluation embeddings must be L2 normalized")
    if matrices["image"].shape[1] != matrices["text"].shape[1]:
        raise ValueError("Evaluation encoder dimensions differ between modalities")
    count = cfg["evaluation"].get("max_images")
    if count:
        records = records[:count]
        keep = parents < len(records)
        parents = parents[keep]
        matrices = {"image": matrices["image"][:len(records)], "text": matrices["text"][keep]}
    identity = {str(p): sha256(p) for p in [
        embeddings / "image.npy", embeddings / "text.npy",
        source / "index" / split / "images.json", source / "index" / split / "parents.npy",
    ]}
    return records, parents, matrices, identity


def write_training_index(manifest: dict, embedding_root: Path, output: Path):
    """Write IDs in exact embedding row order without retaining millions of dicts."""
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "images.pending.json"
    count = 0
    with pending.open("w") as stream:
        stream.write("[")
        for part in manifest["parts"]:
            keys = json.loads((embedding_root / part["keys_path"]).read_text())
            if len(keys) != part["rows"]:
                raise ValueError("Source keys and embedding rows differ")
            for key in keys:
                if count:
                    stream.write(",\n")
                identity = f'{manifest["config"]["dataset_id"]}:{part["shard"]}:{key["key"]}'
                stream.write(json.dumps({"image_id": identity}))
                count += 1
        stream.write("]\n")
    if count != manifest["rows"]:
        raise ValueError("Training index does not cover every embedding")
    pending.replace(output / "images.json")
    np.save(output / "parents.npy", np.arange(count, dtype=np.int64))


def activation_cache(cfg: dict, manifest: dict):
    from mm_sae.paired_training import ChunkedMatrix

    root, embeddings = Path(cfg["output"]), Path(cfg["embedding_cache"])
    assert_frozen(root)
    encoder = EncoderConfig.model_validate(cfg["encoder"])
    records, parents, test, provenance = evaluation_inputs(cfg, encoder)
    identity = {"train_embeddings": manifest["fingerprint"], "evaluation": provenance,
                "evaluation_limit": cfg["evaluation"].get("max_images"),
                "models": json.loads((root / "models/frozen.json").read_text())}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    receipt = root / "activation-source.json"
    if receipt.exists() and json.loads(receipt.read_text())["signature"] != signature:
        raise ValueError("Frozen activation inputs changed; choose a new output directory")
    atomic_json(receipt, {"signature": signature, **identity})
    # These split directory names are the shared analysis storage convention.
    # dataset.json records their actual dataset identities, not COCO for both.
    write_training_index(manifest, embeddings, root / "index/train2017")
    dest = root / "index/val2017"
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "images.json", records)
    np.save(dest / "parents.npy", parents)
    atomic_json(root / "dataset.json", {
        "train2017": {"dataset": manifest["config"]["dataset_id"], "rows": manifest["rows"],
                      "revision": manifest["config"]["revision"], "captions_per_image": 1},
        "val2017": {"dataset": cfg["evaluation"]["name"], "images": len(records), "captions": len(parents)},
        "annotation_usage": "No concept labels or object masking in correspondence/retrieval evaluation",
    })
    for side in ("image", "text"):
        model = TopKSAE.from_pretrained(root / "models" / side)
        cast(torch.nn.Module, model).to(torch.device(encoder.device)).eval().requires_grad_(False)
        train = ChunkedMatrix(embeddings, side)
        for name, values in (("train2017", train), ("val2017", test[side])):
            result = sparse_encode(model, values, root / "activations" / name / f"{side}.npz",
                                   cfg.get("activation_batch_size", 2048))
            del result
        del train, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(cfg: dict):
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    stages = ["embedding_extraction", "sae_training", "activation_cache"]
    stages += [f'followup_{job["module"]}' for job in cfg.get("followups", [])]
    with file_lock(root / ".run.lock", blocking=False):
        config_path = root / "resolved-config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError("Run configuration changed; use a new output directory")
        atomic_json(config_path, cfg)
        # Check the small evaluation cache before committing to expensive extraction.
        evaluation_inputs(cfg, EncoderConfig.model_validate(cfg["encoder"]))
        from transformers import TrainingArguments
        TrainingArguments(**cfg["training"].get("arguments", {}), output_dir=str(root / "checkpoints"),
                          report_to=[], use_cpu=True)
        for job in cfg.get("followups", []):
            followup_path = Path(job["config"])
            followup = yaml.safe_load(followup_path.read_text())
            if (followup_path.parent / followup["source_run"]).resolve() != root:
                raise ValueError("A follow-up points to a different source run")
        with ProgressReporter(root, stages, [], cfg.get("progress_interval_seconds", 10)):
            with stage_progress("embedding_extraction"):
                extraction = shard_configuration(cfg, root)
                manifest = extract(extraction, Path(cfg["embedding_cache"]), encoder_config=cfg["encoder"])
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            with stage_progress("sae_training"):
                from mm_sae.paired_training import train_pairs
                train_pairs(Path(cfg["embedding_cache"]), root, cfg["training"], cfg["encoder"]["device"])
            with stage_progress("activation_cache"):
                activation_cache(cfg, manifest)
            for job in cfg.get("followups", []):
                with stage_progress(f'followup_{job["module"]}'):
                    subprocess.run([sys.executable, "-m", job["module"], "--config", job["config"]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(resolve_config(args.config.resolve()))


if __name__ == "__main__":
    main()
