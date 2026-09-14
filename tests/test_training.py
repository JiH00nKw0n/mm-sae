import numpy as np
import torch
from transformers import Trainer
from mm_sae.training import cached_dataset
from mm_sae.models.sae import TopKSAE, SAEConfig


def test_hf_dataset_distinguishes_unique_image_and_caption_rows():
    images = np.eye(3, dtype=np.float32)
    captions = np.tile(images, (5, 1))
    parents = np.tile(np.arange(3), 5)
    assert len(cached_dataset(images)) == 3
    assert len(cached_dataset(captions)) == 15
    ds = cached_dataset(images, captions, parents)
    assert len(ds) == 15
    np.testing.assert_array_equal(ds[4]["image"], images[1])
    assert Trainer.__module__.startswith("transformers")


def test_sae_sparsity_save_load_and_decoder_geometry(tmp_path):
    torch.manual_seed(3)
    model = TopKSAE(SAEConfig(hidden_size=8, latent_size=12, k=3))
    x = torch.randn(5, 8)
    values, indices = model.encode(x)
    assert values.shape == indices.shape == (5, 3)
    assert (values >= 0).all()
    assert torch.isfinite(model(x)["loss"])
    assert torch.isfinite(model(x[:1])["loss"])
    model.save_pretrained(tmp_path)
    restored = TopKSAE.from_pretrained(tmp_path)
    torch.testing.assert_close(restored.encode(x)[0], values)
    torch.testing.assert_close(restored.W_dec.norm(dim=1), torch.ones(12), atol=1e-6, rtol=1e-6)
