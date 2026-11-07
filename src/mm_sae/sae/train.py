"""HF Trainer for two independent Top-K autoencoders on paired embeddings."""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint
from mm_sae.data.paired import ChunkedMatrix, _manifest, _verified_identity
from mm_sae.encoders.clip import device_for
from mm_sae.io import atomic_json, file_lock, sha256
from mm_sae.progress import progress_task, TaskProgress
from .model import TopKSAE, SAEConfig, PairedSAEs


class DecoderNormCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.normalize_decoder()


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, meter: TaskProgress):
        self.meter = meter

    def on_train_begin(self, args, state, control, **kwargs):
        self.meter.total = state.max_steps
        self.meter.initial = state.global_step
        self.meter.update(state.global_step, epoch=state.epoch)

    def on_step_end(self, args, state, control, **kwargs):
        self.meter.update(state.global_step, epoch=state.epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.meter.update(state.global_step, **(logs or {}))


def cached_dataset(image, text=None, parents=None):
    n = len(image) if text is None else len(text)
    ds = Dataset.from_dict({"row": np.arange(n, dtype=np.int64)})

    def transform(batch):
        rows = np.asarray(batch["row"])
        if text is None:
            return {"inputs": torch.from_numpy(np.array(image[rows], dtype=np.float32))}
        if parents is None:
            raise ValueError("Paired training requires caption-to-image row indices")
        return {
            "image": torch.from_numpy(np.array(image[parents[rows]], dtype=np.float32)),
            "text": torch.from_numpy(np.array(text[rows], dtype=np.float32)),
        }

    return ds.with_transform(transform)


def frozen_fingerprint(root: Path):
    return {
        side: {name: sha256(root / "models" / side / name) for name in ["model.safetensors", "config.json"]}
        for side in ["image", "text"]
    }


def assert_frozen(root: Path):
    expected = json.loads((root / "models" / "frozen.json").read_text())
    if expected != frozen_fingerprint(root):
        raise RuntimeError("SAE weights changed after training; analyses must use the original frozen models")


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
        "num_train_epochs": 10,
        "per_device_train_batch_size": 1024,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "fp16": False,
        "bf16": False,
        "dataloader_num_workers": 0,
        "dataloader_drop_last": False,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 50,
        "eval_strategy": "no",
        "seed": int(cfg.get("seed", 0)),
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
        "latent_size": int(cfg.get("latent_size", 4096)),
        "k": int(cfg.get("top_k", 32)),
    }
    signature_payload = {
        "cache": identity,
        "model": model_settings,
        "arguments": arguments,
        "initialization_seed": int(cfg.get("seed", 0)),
        "implementation": {
            "paired_training": sha256(Path(__file__)),
            "sae": sha256(Path(__file__).with_name("model.py")),
            "training": sha256(Path(__file__).parents[1] / "data" / "paired.py"),
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
        "signature": signature,
        "signature_payload": signature_payload,
        "state": "running",
        "rows_per_epoch": manifest["rows"],
        "planned_optimizer_steps": steps,
        "sampling": "paired_repeat_image",
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
            **arguments,
            output_dir=str(checkpoint_dir),
            report_to=[],
            remove_unused_columns=False,
            use_cpu=device == "cpu",
            dataloader_pin_memory=device == "cuda",
        )
        trainer = Trainer(
            model=pair,
            args=args,
            train_dataset=cached_dataset(image, text, np.arange(len(image))),
            data_collator=default_data_collator,
            callbacks=[DecoderNormCallback()],
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
            state="completed",
            optimizer_steps=trainer.state.global_step,
            logs=trainer.state.log_history,
            models=fingerprints,
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
