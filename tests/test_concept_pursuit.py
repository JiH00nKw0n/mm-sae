import numpy as np
import pytest
import torch
from mm_sae.analysis.concept_pursuit import ProbeLoss, fit_pursuit


def data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 6))
    x = (x - x.mean(0)) / x.std(0)
    y = np.column_stack([x[:, 0] + 0.3 * x[:, 1] > 0, x[:, 3] - 0.4 * x[:, 4] > 0]).astype(float)
    return torch.tensor(x), torch.tensor(y), torch.ones(100, dtype=torch.float64)


@pytest.mark.parametrize("kind", ["mse", "binary_ce", "ce"])
def test_full_gradient_matches_autograd(kind):
    x, y, q = data()
    q[0] = 2
    if kind == "mse":
        x = x - (x * q[:, None]).sum(0) / q.sum()
    p = ProbeLoss(x, y, q, kind, 0.01)
    w = torch.randn(6, 2, dtype=x.dtype, requires_grad=True) * 0.1
    b = torch.zeros(2, dtype=x.dtype, requires_grad=True)
    score = x @ w + b
    if kind == "mse":
        loss = 0.5 * ((score - y) ** 2 * q[:, None]).sum() / q.sum()
    elif kind == "binary_ce":
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(score, y, reduction="none") * q[:, None]
        ).sum() / q.sum()
    else:
        used = q * (y.sum(1) > 0)
        loss = (
            -(y / y.sum(1, keepdim=True).clamp_min(1) * score.log_softmax(1) * used[:, None]).sum()
            / used.sum()
        )
    loss += 0.005 * w.square().sum()
    expected = torch.autograd.grad(loss, (w, b))
    val, gw, gb = p.full(w.detach(), b.detach())
    assert val == pytest.approx(float(loss), abs=1e-10)
    torch.testing.assert_close(gw, expected[0])
    torch.testing.assert_close(gb, expected[1])


@pytest.mark.parametrize("kind", ["mse", "binary_ce", "ce"])
@pytest.mark.parametrize("penalty", [0.0, 0.01])
def test_budget_objective_and_reselection(kind, penalty):
    x, y, q = data()
    w, b, a = fit_pursuit(x, y, q, kind=kind, penalty=penalty, budget=2, max_outer=6, inner_iter=100)
    assert (np.count_nonzero(w, axis=0) <= 2).all()
    values = [r["objective"] for r in a["history"] if r.get("state") == "outer_completed"]
    assert len(values) > 0
    assert all(v <= u + 1e-9 for u, v in zip(values, values[1:]))
    assert a["global_optimum_certified"] is False
    if kind == "mse":
        assert w[0, 0] != 0 and w[3, 1] != 0


def test_unselected_coordinates_can_enter():
    x, y, q = data()
    w = np.zeros((6, 2))
    w[5, :] = 0.1
    fitted, _, _ = fit_pursuit(
        x, y, q, kind="mse", penalty=0.01, budget=1, max_outer=8, initial=(w, y.mean(0).numpy())
    )
    assert fitted[0, 0] != 0 and fitted[3, 1] != 0
    assert np.all(fitted[5] == 0)
