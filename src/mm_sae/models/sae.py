"""Top-K SAE adapted from lvlm_hallucination/src/models/modeling_sae.py (TopKSAE).

Preserves subtractive decoder bias, ReLU+TopK, untied decoder initialized from the encoder,
variance-normalized reconstruction, and unit decoder atoms. AuxK is absent because the reference
COCO reconstruction-only run does not activate its optional dead-feature auxiliary loss.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class SAEConfig(PretrainedConfig):
    model_type = "mm_sae_topk"

    def __init__(self, hidden_size=512, latent_size=4096, k=8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.latent_size, self.k = hidden_size, latent_size, k


class TopKSAE(PreTrainedModel):
    config_class = SAEConfig
    base_model_prefix = "sae"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = nn.Linear(config.hidden_size, config.latent_size)
        nn.init.zeros_(self.encoder.bias)
        self.W_dec = nn.Parameter(self.encoder.weight.detach().clone())
        self.b_dec = nn.Parameter(torch.zeros(config.hidden_size))
        self.normalize_decoder()
        self.post_init()

    def _init_weights(self, module):
        # Keep the original nn.Linear initialization and copied decoder, as in the source SAE.
        pass

    def encode(self, x):
        pre = torch.relu(self.encoder(x - self.b_dec))
        return pre.topk(self.config.k, dim=-1, sorted=False)

    def forward(self, inputs):
        values, indices = self.encode(inputs)
        dense = inputs.new_zeros((len(inputs), self.config.latent_size)).scatter(1, indices, values)
        reconstruction = dense @ self.W_dec + self.b_dec
        variance = (inputs - inputs.mean(0)).square().sum()
        # A singleton/constant batch must not produce NaNs. Normal COCO batches use the original loss.
        loss = (reconstruction - inputs).square().sum() / variance.clamp_min(torch.finfo(inputs.dtype).eps)
        return {"loss": loss}

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + torch.finfo(self.W_dec.dtype).eps)


class PairedSAEs(nn.Module):
    """Optional reference-style paired training; the two sides share no parameters."""

    def __init__(self, image, text):
        super().__init__()
        self.image_sae, self.text_sae = image, text

    def forward(self, image, text):
        return {"loss": (self.image_sae(image)["loss"] + self.text_sae(text)["loss"]) / 2}

    def normalize_decoder(self):
        self.image_sae.normalize_decoder()
        self.text_sae.normalize_decoder()
