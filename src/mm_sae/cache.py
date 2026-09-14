"""Content-addressed original embeddings shared by independent research questions."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def embedding_directory(config, split):
    index = config.output / "index" / split
    images = json.loads((index / "images.json").read_text())
    captions = json.loads((index / "captions.json").read_text())
    encoder = config.encoder.model_dump(mode="json", exclude={"device", "batch_size"})
    # Contents and row order matter. Mount paths and research-specific labels do not.
    identity = {
        "encoder": encoder,
        "images": [(r["image_id"], r["image_sha256"]) for r in images],
        "captions": [(r["caption_id"], r["text"]) for r in captions],
        "libraries": {p: version(p).split("+")[0] for p in ["torch", "transformers", "numpy", "Pillow"]},
        "normalization": "float32_l2_no_epsilon_v1",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Path(config.cache) / "embeddings" / key / split
