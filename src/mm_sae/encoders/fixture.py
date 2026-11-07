"""Deterministic test encoder: visible RGB stripes and corresponding words."""

import numpy as np
from .clip import normalize


class FixtureEncoder:
    def __init__(self, config):
        self.dim = config.synthetic_dim
        self.config = config
        self.projection = np.random.default_rng(41).normal(size=(4, self.dim))

    def images(self, images):
        vectors = [[*np.asarray(im.convert("RGB")).mean(axis=(0, 1)) / 255, 1.0] for im in images]
        return normalize(np.asarray(vectors) @ self.projection)

    def texts(self, texts):
        vectors = [
            [
                *(
                    word in text.lower().split()[: self.config.text_max_length - 2]
                    for word in ("red", "green", "blue")
                ),
                1.0,
            ]
            for text in texts
        ]
        return normalize(np.asarray(vectors) @ self.projection)

    def text_lengths(self, texts):
        return [len(text.split()) + 2 for text in texts]
