"""Training and data loading are delegated to HF Trainer and HF Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .io import atomic_json, sha256
from .cache import embedding_directory
from .models.encoder import device_for
from .models.sae import SAEConfig, TopKSAE, PairedSAEs
from .progress import progress_task, TaskProgress


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


def validate_training_arguments(config):
    """Validate HF options without loading a model or requiring a GPU."""
    values = dict(config.training.arguments)
    for reserved in ["output_dir", "remove_unused_columns", "report_to", "use_cpu", "dataloader_pin_memory"]:
        if reserved in values:
            raise ValueError(f"training.arguments.{reserved} is managed by the runner")
    TrainingArguments(**values, output_dir=str(config.output / "checkpoints"), use_cpu=True, report_to=[])


def train(config):
    cfg, root = config.training, config.output
    device_for(config.encoder.device)
    cache = embedding_directory(config, cfg.split)
    image, text = [np.load(cache / f"{side}.npy", mmap_mode="r") for side in ["image", "text"]]
    parents = np.load(root / "index" / cfg.split / "parents.npy")
    set_seed(cfg.seed)
    model_config = SAEConfig(hidden_size=image.shape[1], latent_size=cfg.latent_size, k=cfg.top_k)
    image_sae, text_sae = TopKSAE(model_config), TopKSAE(model_config)
    history = {}
    if cfg.image_checkpoint:
        image_sae, text_sae = [
            TopKSAE.from_pretrained(p) for p in [cfg.image_checkpoint, cfg.text_checkpoint]
        ]
        for model in [image_sae, text_sae]:
            if (model.config.hidden_size, model.config.latent_size, model.config.k) != (
                image.shape[1],
                cfg.latent_size,
                cfg.top_k,
            ):
                raise ValueError("Loaded SAE checkpoint dimensions/TopK disagree with the run config")
    else:
        jobs = [("image", image_sae, cached_dataset(image)), ("text", text_sae, cached_dataset(text))]
        if cfg.sampling == "paired_repeat_image":
            jobs = [("paired", PairedSAEs(image_sae, text_sae), cached_dataset(image, text, parents))]
        for name, model, dataset in jobs:
            output = root / "checkpoints" / name
            arguments = dict(cfg.arguments)
            arguments.update(
                output_dir=str(output),
                report_to=[],
                remove_unused_columns=False,
                use_cpu=config.encoder.device == "cpu",
                dataloader_pin_memory=config.encoder.device == "cuda",
            )
            args = TrainingArguments(**arguments)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=default_data_collator,
                callbacks=[DecoderNormCallback()],
            )
            last = get_last_checkpoint(str(output)) if output.exists() else None
            with progress_task(f"Train {name} SAE", unit="optimizer steps") as meter:
                trainer.add_callback(TrainingProgressCallback(meter))
                trainer.train(resume_from_checkpoint=last)
            trainer.save_state()
            history[name] = {
                "rows_per_epoch": len(dataset),
                "optimizer_steps": trainer.state.global_step,
                "logs": trainer.state.log_history,
            }
    for side, model in [("image", image_sae), ("text", text_sae)]:
        model.save_pretrained(root / "models" / side, safe_serialization=True)
    atomic_json(
        root / "training.json",
        {
            "trainer": "transformers.Trainer",
            "sampling": cfg.sampling,
            "unique_images": len(image),
            "captions": len(text),
            "text_to_image_ratio": len(text) / len(image),
            "histories": history,
            "frozen_after_this_stage": True,
        },
    )
    atomic_json(root / "models" / "frozen.json", frozen_fingerprint(root))


def frozen_fingerprint(root: Path):
    return {
        side: {name: sha256(root / "models" / side / name) for name in ["model.safetensors", "config.json"]}
        for side in ["image", "text"]
    }


def assert_frozen(root: Path):
    expected = json.loads((root / "models" / "frozen.json").read_text())
    if expected != frozen_fingerprint(root):
        raise RuntimeError("SAE weights changed after training; analyses must use the original frozen models")


def load_saes(config):
    assert_frozen(config.output)
    device = device_for(config.encoder.device)
    models = []
    for side in ["image", "text"]:
        model = TopKSAE.from_pretrained(config.output / "models" / side)
        cast(torch.nn.Module, model).to(device)
        models.append(model.eval().requires_grad_(False))
    return tuple(models)
