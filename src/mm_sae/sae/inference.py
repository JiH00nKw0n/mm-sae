"""Load original Hugging Face model directories without mutating weights."""

import json
import numpy as np
import torch
from scipy import sparse
from mm_sae.io import sha256
from .model import TopKSAE


def identity(root):
    if not (root / "image").is_dir():
        root = root / "models"
    hashes = {
        s: {f: sha256(root / s / f) for f in ("config.json", "model.safetensors")} for s in ("image", "text")
    }
    if (root / "frozen.json").exists() and json.loads((root / "frozen.json").read_text()) != hashes:
        raise ValueError("Saved SAE fingerprint does not match weights")
    return root, hashes


class FrozenSAEs:
    def __init__(self, root, device):
        root, self.hashes = identity(root)
        self.models = {}
        for side in ("image", "text"):
            model, info = TopKSAE.from_pretrained(
                root / side, local_files_only=True, output_loading_info=True
            )
            if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Incompatible {side} SAE weights: {info}")
            self.models[side] = model.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.inference_mode()
    def activations(self, side, embeddings):
        model = self.models[side]
        if embeddings.ndim != 2 or embeddings.shape[1] != model.config.hidden_size:
            raise ValueError(f"{side} embedding width differs from saved SAE")
        x = torch.tensor(embeddings, device=self.device, dtype=torch.float32)
        values, indices = model.encode(x)
        k = model.config.k
        matrix = sparse.csr_matrix(
            (values.cpu().numpy().ravel(), indices.cpu().numpy().ravel(), np.arange(0, (len(x) + 1) * k, k)),
            shape=(len(x), model.config.latent_size),
        )
        matrix.eliminate_zeros()
        return matrix
