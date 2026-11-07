"""Frozen CLIP inference with one shared token limit for all benchmarks."""

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


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
        self.processor = CLIPProcessor.from_pretrained(
            config.model_id, revision=config.revision, use_fast=False
        )
        self.model = (
            CLIPModel.from_pretrained(config.model_id, revision=config.revision).eval().to(self.device)
        )
        self.model.requires_grad_(False)
        self.dim = self.model.config.projection_dim
        if config.text_max_length > self.model.config.text_config.max_position_embeddings:
            raise ValueError("Text limit exceeds the encoder context")

    @torch.inference_mode()
    def images(self, images):
        pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        return normalize(self.model.get_image_features(pixel_values=pixels).float().cpu().numpy())

    def text_lengths(self, texts):
        return [len(x) for x in self.processor.tokenizer(texts, truncation=False, verbose=False).input_ids]

    @torch.inference_mode()
    def texts(self, texts):
        batch = self.processor.tokenizer(
            texts, padding=True, truncation=True, max_length=self.config.text_max_length, return_tensors="pt"
        )
        return normalize(self.model.get_text_features(**batch.to(self.device)).float().cpu().numpy())
