"""Use Hugging Face CLIPModel/CLIPProcessor, with no reimplementation of the backbone."""

from __future__ import annotations
from typing import Protocol, cast, Mapping

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor


class Encoder(Protocol):
    """Shared contract for frozen image/text encoders and their spatial preprocessing."""

    dim: int

    def images(self, images: list[Image.Image]) -> np.ndarray: ...
    def texts(self, texts: list[str]) -> np.ndarray: ...
    def visible_mask(self, mask: np.ndarray) -> np.ndarray: ...


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use the CPU smoke config or an NVIDIA server.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))
    if np.any(norm == 0) or not np.isfinite(x).all():
        raise ValueError("Encoder emitted a zero or non-finite embedding")
    return x / norm


class CLIPEncoder:
    def __init__(self, config):
        self.config = config
        self.device = device_for(config.device)
        self.processor = cast(
            CLIPProcessor,
            CLIPProcessor.from_pretrained(config.model_id, revision=config.revision, use_fast=False),
        )
        self.image_processor = cast(CLIPImageProcessor, getattr(self.processor, "image_processor"))
        self.model = CLIPModel.from_pretrained(config.model_id, revision=config.revision)
        cast(torch.nn.Module, self.model).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.dim = self.model.config.projection_dim

    @torch.inference_mode()
    def images(self, images):
        inputs = cast(Mapping[str, torch.Tensor], self.processor(images=images, return_tensors="pt"))
        pixels = inputs["pixel_values"].to(self.device)
        out = self.model.get_image_features(pixel_values=cast(torch.FloatTensor, pixels))
        return normalize(out.float().cpu().numpy())

    @torch.inference_mode()
    def texts(self, texts):
        inputs = cast(
            Mapping[str, torch.Tensor],
            self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.text_max_length,
            ),
        )
        out = self.model.get_text_features(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
        )
        return normalize(out.float().cpu().numpy())

    def visible_mask(self, mask: np.ndarray):
        # Apply the actual HF processor's geometry to integer IDs with nearest-neighbor interpolation.
        p = self.image_processor
        value = mask[None, :, :]
        if p.do_resize:
            value = p.resize(
                value,
                size=p.size,
                resample=Image.Resampling.NEAREST,
                input_data_format="channels_first",
                data_format="channels_first",
            )
        if p.do_center_crop:
            value = p.center_crop(
                value, size=p.crop_size, input_data_format="channels_first", data_format="channels_first"
            )
        return value[0]


class SyntheticEncoder:
    """Explicitly artificial embedding backend for deterministic unit/integration tests only."""

    def __init__(self, config):
        self.dim = config.synthetic_dim
        self.projection = np.random.default_rng(37).normal(size=(6, self.dim)).astype(np.float32)

    def images(self, images):
        rows = []
        for image in images:
            a = np.asarray(image.convert("RGB"))
            signals = [
                np.mean((a == color).all(-1)) for color in [(210, 50, 60), (40, 100, 210), (60, 160, 70)]
            ]
            rows.append(signals + [1.0, float(a.mean()) / 255, float(a.std()) / 255])
        return normalize(np.asarray(rows, dtype=np.float32) @ self.projection)

    def texts(self, texts):
        rows = [
            [float(word in text.lower()) for word in ["person", "bicycle", "grass"]] + [1.0, 0.3, 0.2]
            for text in texts
        ]
        return normalize(np.asarray(rows, dtype=np.float32) @ self.projection)

    def visible_mask(self, mask):
        return mask


def make_encoder(config) -> Encoder:
    return CLIPEncoder(config) if config.backend == "clip" else SyntheticEncoder(config)
