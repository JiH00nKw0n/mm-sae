import numpy as np
import pytest
import torch
from scipy import sparse

from mm_sae.sae.model import SAEConfig, SharedSAE, TopKSAE
from mm_sae.evaluation import concepts, concepts_cuda, retrieval


@pytest.mark.parametrize("kind", ["group_sparse", "iso_align"])
def test_shared_objectives_match_the_paired_topk_definition(kind):
    torch.manual_seed(0)
    model = TopKSAE(SAEConfig(hidden_size=5, latent_size=12, k=3))
    pair = SharedSAE(model, kind, 0.2)
    image, text = torch.randn(11, 5), torch.randn(11, 5)
    li, zi = model.reconstruct(image)
    lt, zt = model.reconstruct(text)
    if kind == "group_sparse":
        loss = (zi.square() + zt.square() + 1e-12).sqrt().sum(-1).mean()
    else:
        mask = torch.ones_like(zi)
        for codes in (zi, zt):
            mask.scatter_(1, codes.topk(1).indices, 0)
        loss = -torch.nn.functional.cosine_similarity(mask * zi, mask * zt).mean()
    actual = pair(image, text)["loss"]
    torch.testing.assert_close(actual, (li + lt) / 2 + 0.2 * loss)
    actual.backward()
    assert model.encoder.weight.grad.isfinite().all()


def test_sparse_scores_preserve_probe_losses_and_retrieval_ties(monkeypatch):
    gen = np.random.default_rng(5)
    x = gen.normal(size=(300, 8)).clip(0, None)
    x[gen.random(x.shape) < 0.87] = 0
    x[:, 0] = 0
    y = gen.random((300, 3)) < 0.1
    ids = dict(train=np.arange(170), val=np.arange(170, 230), test=np.arange(230, 300))
    rows = [
        dict(group=str(i), split=split, labels=[str(j) for j in np.flatnonzero(y[i])])
        for split, indices in ids.items()
        for i in indices
    ]
    reference, original = concepts.evaluate(x, rows, 2, True)
    csr = sparse.csr_matrix(x)
    result, _ = concepts.evaluate(csr, rows, 2, True)
    assert result["macro_test_loss"] == pytest.approx(reference["macro_test_loss"])
    monkeypatch.setattr(concepts_cuda, "as_gpu", lambda value: torch.as_tensor(value, dtype=torch.float64))
    _, details = concepts_cuda.evaluate(csr, y, ["0", "1", "2"], ids, 2, True)
    np.testing.assert_allclose(details["validation_loss"], original["validation_loss"], atol=1e-6)
    labels = [str(i) for i in range(300)]
    assert retrieval.direction(csr, csr, labels, labels) == retrieval.direction(x, x, labels, labels)
