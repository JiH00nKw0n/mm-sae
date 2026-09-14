"""One validated configuration contract shared by every stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def default_training_arguments():
    return {
        "num_train_epochs": 30,
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
        "seed": 0,
        "data_seed": 0,
        "eval_strategy": "no",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
        "dataloader_drop_last": False,
    }


class DataConfig(StrictModel):
    source: Literal["coco", "synthetic"] = "coco"
    root: Path = Path("data/coco")
    download: bool = False
    splits: list[str] = ["train2017", "val2017"]
    expected_images: dict[str, int] = {"train2017": 118287, "val2017": 5000}
    images_pattern: str = "images/{split}/{image_id:012d}.jpg"
    masks_pattern: str = "annotations/{split}/{image_id:012d}.png"
    captions_pattern: str = "annotations/captions_{split}.json"
    image_limits: dict[str, int] = {}
    concepts_file: Path | None = None
    reviewed_captions: Path | None = None
    images_url_pattern: str = "http://images.cocodataset.org/{split}/{image_id:012d}.jpg"
    label_scope: Literal["model_input", "full_image"] = "model_input"
    fixture_images: int = Field(default=48, ge=8)
    fixture_captions: int = Field(default=5, ge=1)
    fixture_size: int = Field(default=64, ge=16)
    downloads: dict[str, str] = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "stuffthingmaps_trainval2017.zip": "https://huggingface.co/datasets/JPShi/COCO-Stuff/resolve/b7af13b9d74c9ab4a6e0ffa787a40f0ab8ec1a40/stuffthingmaps_trainval2017.zip",
    }


class EncoderConfig(StrictModel):
    backend: Literal["clip", "synthetic"] = "clip"
    model_id: str = "openai/clip-vit-base-patch32"
    revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    batch_size: int = Field(default=128, ge=1)
    text_max_length: int = Field(default=77, ge=1)
    synthetic_dim: int = Field(default=16, ge=6)


class TrainingConfig(StrictModel):
    split: str = "train2017"
    sampling: Literal["unique_image_all_text", "paired_repeat_image"] = "unique_image_all_text"
    latent_size: int = Field(default=4096, ge=1)
    top_k: int = Field(default=8, ge=1)
    arguments: dict = Field(default_factory=default_training_arguments)

    @model_validator(mode="after")
    def merge_arguments(self):
        self.arguments = default_training_arguments() | self.arguments
        return self

    @property
    def max_steps(self):
        value = self.arguments.get("max_steps", -1)
        return value if value > 0 else None

    @property
    def seed(self):
        return self.arguments.get("seed", 0)

    image_checkpoint: Path | None = None
    text_checkpoint: Path | None = None


class FeatureConfig(StrictModel):
    batch_size: int = Field(default=2048, ge=1)
    mask_rgb: tuple[int, int, int] = (255, 255, 255)


class ExperimentConfig(StrictModel):
    name: str = "rq1"
    options: dict = {}


class ExecutionConfig(StrictModel):
    progress_interval_seconds: float = Field(default=10, gt=0)
    require_approval: bool = False
    approval_file: Path | None = None
    review_document: Path | None = None


class Config(StrictModel):
    output: Path = Path("runs/coco-rq1")
    cache: Path = Path("cache")
    data: DataConfig = Field(default_factory=DataConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @model_validator(mode="after")
    def coherent(self):
        if self.execution.require_approval and (
            self.execution.approval_file is None or self.execution.review_document is None
        ):
            raise ValueError("Approval requires both an approval_file and a review_document")
        if self.training.top_k > self.training.latent_size:
            raise ValueError("top_k cannot exceed latent_size")
        if bool(self.training.image_checkpoint) != bool(self.training.text_checkpoint):
            raise ValueError("Provide both pretrained SAE checkpoints, or neither")
        if (self.data.source == "synthetic") != (self.encoder.backend == "synthetic"):
            raise ValueError("Synthetic data and encoder must be selected together; no mock encoder on COCO")
        if self.training.split not in self.data.splits:
            raise ValueError("training.split must be present in data.splits")
        if any(not 0 <= x <= 255 for x in self.features.mask_rgb):
            raise ValueError("mask_rgb must contain byte values")
        if self.data.image_limits and set(self.data.image_limits) != set(self.data.splits):
            raise ValueError("A smoke run must explicitly limit every split")
        return self

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    config = Config.model_validate(raw)
    # Every relative path is relative to the configuration file, never the caller's working directory.
    for model, fields in [
        (config, ["output", "cache"]),
        (config.data, ["root", "concepts_file", "reviewed_captions"]),
        (config.training, ["image_checkpoint", "text_checkpoint"]),
        (config.execution, ["approval_file", "review_document"]),
    ]:
        for field in fields:
            value = getattr(model, field)
            if value is not None:
                setattr(
                    model, field, value.resolve() if value.is_absolute() else (path.parent / value).resolve()
                )
    return config
