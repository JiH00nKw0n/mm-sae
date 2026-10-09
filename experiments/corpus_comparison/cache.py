"""Apply frozen SAEs to an existing annotated embedding cache without retraining."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
from typing import cast

import numpy as np
import torch
import yaml

from experiments.paired_corpus.run import evaluation_inputs
from mm_sae.config import EncoderConfig
from mm_sae.features import sparse_encode
from mm_sae.io import atomic_json, file_lock, sha256
from mm_sae.models.sae import TopKSAE
from mm_sae.progress import ProgressReporter, stage_progress
from mm_sae.training import assert_frozen


def load_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    for name in ("model_run", "annotation_run", "output"):
        cfg[name] = str((path.parent / cfg[name]).resolve())
    return cfg


def write_index(source: Path, output: Path, split, records, parents):
    """Retain annotation columns and the exact image/caption embedding row order."""
    origin, dest = source / "index" / split, output / "index" / split
    original_images = json.loads((origin / "images.json").read_text())
    if records != original_images[:len(records)]:
        raise ValueError("Image rows must be an unchanged prefix of the annotated cache")
    original_parents = np.load(origin / "parents.npy")
    keep_captions = original_parents < len(records)
    np.testing.assert_array_equal(parents, original_parents[keep_captions])
    captions = json.loads((origin / "captions.json").read_text())
    if len(captions) != len(original_parents):
        raise ValueError("Caption records and parent indices have different lengths")
    presence, mentions = (np.load(origin / f"{n}.npy") for n in ("presence", "mentions"))
    cids = json.loads((origin / "concept_ids.json").read_text())
    if presence.shape != (len(original_images), len(cids)) or mentions.shape != (len(captions), len(cids)):
        raise ValueError("Annotation dimensions do not match image/caption rows and concept columns")
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "images.json", records)
    atomic_json(dest / "captions.json", [c for c, keep in zip(captions, keep_captions, strict=True) if keep])
    atomic_json(dest / "concept_ids.json", cids)
    np.save(dest / "parents.npy", parents)
    np.save(dest / "presence.npy", presence[:len(records)])
    np.save(dest / "mentions.npy", mentions[keep_captions])


def run(cfg):
    model_root, source, out = (Path(cfg[n]) for n in ("model_run", "annotation_run", "output"))
    if out in (model_root, source) or out.is_relative_to(model_root) or out.is_relative_to(source):
        raise ValueError("Use a separate output directory; source runs are immutable")
    assert_frozen(model_root)
    model_cfg = json.loads((model_root / "resolved-config.json").read_text())
    encoder = EncoderConfig.model_validate(model_cfg["encoder"])
    encoder.device = cfg.get("device", "cuda")
    out.mkdir(parents=True, exist_ok=True)
    splits = cfg.get("splits", ["train2017", "val2017"])
    with file_lock(out / ".run.lock", blocking=False), ProgressReporter(out, splits, [], 10):
        paths = [model_root / "models/frozen.json", model_root / "resolved-config.json", source / "dataset.json"]
        inputs = {}
        # Validate both splits before writing any reusable activation file.
        for split in splits:
            request = {"evaluation": {"source_run": str(source), "split": split,
                                       "max_images": cfg.get("max_images")}}
            records, parents, matrices, provenance = evaluation_inputs(request, encoder)
            inputs[split] = records, parents, matrices
            paths.extend(Path(p) for p in provenance)
            paths.extend(source / "index" / split / n for n in
                         ("captions.json", "concept_ids.json", "presence.npy", "mentions.npy"))
        identity = dict(config=cfg, sources={str(p): sha256(p) for p in paths},
                        code_sha256=sha256(Path(__file__)))
        receipt = out / "activation-source.json"
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise ValueError("Frozen model, embedding cache or annotation changed; use a new output")
        atomic_json(receipt, identity)
        for side in ("image", "text"):
            for name in ("config.json", "model.safetensors"):
                dest = out / "models" / side / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(model_root / "models" / side / name, dest)
        shutil.copyfile(model_root / "models/frozen.json", out / "models/frozen.json")
        assert_frozen(out)
        dataset = json.loads((source / "dataset.json").read_text())
        dataset["frozen_sae_source"] = str(model_root)
        dataset["annotation_source"] = str(source)
        dataset["sae_training"] = model_cfg.get("source")
        dataset["protocol"] = "COCO annotated embeddings re-encoded with frozen CC3M SAEs; no SAE training"
        atomic_json(out / "dataset.json", dataset)
        for split, (records, parents, matrices) in inputs.items():
            with stage_progress(split):
                write_index(source, out, split, records, parents)
                for side in ("image", "text"):
                    model = TopKSAE.from_pretrained(out / "models" / side)
                    cast(torch.nn.Module, model).to(torch.device(encoder.device)).eval().requires_grad_(False)
                    values = sparse_encode(model, matrices[side], out / "activations" / split / f"{side}.npz",
                                           cfg.get("batch_size", 2048))
                    if values.shape != (len(matrices[side]), model.config.latent_size):
                        raise ValueError("Frozen activation cache shape is incorrect")
                    del values, model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        assert_frozen(out)
        atomic_json(out / "completed.json", dict(model_run=str(model_root), annotation_run=str(source),
                                                  sae_training_steps=0, splits=splits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config.resolve()))


if __name__ == "__main__":
    main()
