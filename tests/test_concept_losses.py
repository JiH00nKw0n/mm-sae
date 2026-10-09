import numpy as np
import pytest
from scipy import sparse
import torch
from mm_sae.analysis.concept_losses import supervised_loss, fit_fixed_support


def test_soft_ce_uniform_mass_on_all_present_labels():
    logits = torch.tensor([[1., 3., -2.], [-1., 2., 1.]], dtype=torch.float64)
    labels = torch.tensor([[1., 1., 0.], [0., 0., 0.]], dtype=torch.float64)
    loss = supervised_loss(logits, labels, torch.tensor([2., 1.]), 'ce')
    expected = -torch.log_softmax(logits[0], dim=0)[:2].mean()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize('kind', ['binary_ce', 'ce'])
def test_sample_weights_match_repeated_pairs(kind):
    logits = torch.tensor([[1., -2.], [3., 1.]])
    labels = torch.tensor([[1., 0.], [1., 1.]])
    a = supervised_loss(logits, labels, torch.tensor([2., 1.]), kind)
    b = supervised_loss(logits[[0, 0, 1]], labels[[0, 0, 1]], torch.ones(3), kind)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize('kind', ['binary_ce', 'ce'])
def test_fixed_support_probe_decreases_loss_and_preserves_support(kind):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(80, 4))
    labels = np.column_stack([x[:, 0] > 0, x[:, 1] > 0, x[:, 0] <= 0]).astype(float)
    support = np.zeros((4, 3))
    support[0, 0] = support[1, 1] = support[0, 2] = 1
    coef, audit = fit_fixed_support(sparse.csr_matrix(x), labels, np.ones(80),
                                   x.mean(0), x.std(0), support, kind=kind, device='cpu', max_iter=100)
    assert (coef[support == 0] == 0).all()
    assert audit['final_loss'] < audit['initial_loss']
    assert audit['gradient_inf'] < 1e-3
