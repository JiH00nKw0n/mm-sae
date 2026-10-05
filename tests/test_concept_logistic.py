from typing import Any

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import check_grad, minimize
from scipy.special import expit

from mm_sae.analysis.concept_logistic import _objective_cpu, fit_fixed_support_logistic


def sample():
    rng = np.random.default_rng(433)
    x = rng.normal(size=(140, 7))
    x[rng.random(x.shape) < 0.8] = 0
    y = np.column_stack((x[:, 1] - x[:, 4] + rng.normal(size=len(x)) > 0,
                         x[:, 3] + rng.normal(size=len(x)) > 0))
    return x, y, x.mean(axis=0), np.maximum(x.std(axis=0), 0.01), [[1, 4], [0, 3, 6]]


def test_weighted_compressed_fit_matches_uncompressed_optimum_and_has_fixed_support():
    x, y, mean, scale, supports = sample()
    # Normalization is intentionally not the sample mean, checking that the
    # returned retrieval mean is computed rather than assumed to equal bias.
    mean = mean + 0.07
    mass = np.linspace(0, 3, len(x))
    mass /= mass.sum()
    penalty = 0.03
    inputs = [array.copy() for array in (x, y, mean, scale, mass)]
    seen = []
    fit = fit_fixed_support_logistic(x, y, mean, scale, supports, sample_weight=mass,
                                     penalty=penalty, callback=lambda c, info: seen.append((c, info)))
    assert [c for c, _ in seen] == [0, 1]
    for c, features in enumerate(supports):
        z = (x[:, features] - mean[features]) / scale[features]

        def direct(theta):
            logits = z @ theta[:-1] + theta[-1]
            return np.dot(mass, np.logaddexp(0, logits) - y[:, c] * logits) + (
                penalty * np.dot(theta[:-1], theta[:-1]) / 2
            )

        optimum = minimize(direct, np.zeros(len(features) + 1), method="BFGS", tol=1e-9)
        theta = np.r_[fit.coefficients[features, c], fit.bias[c]]
        np.testing.assert_allclose(theta, optimum.x, atol=5e-5)
        residual = mass * (expit(z @ theta[:-1] + theta[-1]) - y[:, c])
        np.testing.assert_allclose(z.T @ residual + penalty * theta[:-1], 0, atol=2e-6)
        assert abs(residual.sum()) < 2e-6
        np.testing.assert_array_equal(fit.coefficients[np.setdiff1d(np.arange(7), features), c], 0)
        logits = z @ theta[:-1] + theta[-1]
        assert fit.fit_prediction_mean[c] == pytest.approx(mass @ logits)
        assert mass @ (logits - fit.fit_prediction_mean[c]) == pytest.approx(0, abs=1e-12)
        assert fit.diagnostics[c]["objective"] == pytest.approx(direct(theta))
        assert fit.diagnostics[c]["optimized_rows"] < len(x)
    for original, after in zip(inputs, (x, y, mean, scale, mass)):
        np.testing.assert_array_equal(original, after)


def test_integer_sample_weights_equal_explicit_row_duplication():
    x, y, mean, scale, supports = sample()
    repeats = np.arange(len(x)) % 6
    weighted = fit_fixed_support_logistic(x, y, mean, scale, supports, sample_weight=repeats)
    duplicated = fit_fixed_support_logistic(np.repeat(x, repeats, axis=0), np.repeat(y, repeats, axis=0),
                                            mean, scale, supports)
    np.testing.assert_allclose(weighted.coefficients, duplicated.coefficients, atol=1e-9)
    np.testing.assert_allclose(weighted.bias, duplicated.bias, atol=1e-9)
    np.testing.assert_allclose(weighted.fit_prediction_mean, duplicated.fit_prediction_mean, atol=1e-9)
    rescaled = fit_fixed_support_logistic(x, y, mean, scale, supports, sample_weight=repeats * 37)
    np.testing.assert_allclose(weighted.coefficients, rescaled.coefficients, atol=1e-9)


def test_sparse_and_dense_inputs_agree():
    x, y, mean, scale, supports = sample()
    dense = fit_fixed_support_logistic(x, y, mean, scale, supports)
    sparse_fit = fit_fixed_support_logistic(sparse.csr_matrix(x), y, mean, scale, supports)
    np.testing.assert_allclose(dense.coefficients, sparse_fit.coefficients, atol=1e-12)
    np.testing.assert_allclose(dense.bias, sparse_fit.bias, atol=1e-12)
    np.testing.assert_allclose(dense.fit_prediction_mean, sparse_fit.fit_prediction_mean, atol=1e-12)


def test_objective_gradient_matches_finite_differences():
    rng = np.random.default_rng(654)
    z = rng.normal(size=(31, 4))
    y = rng.random(31) > 0.3
    mass = rng.random(31)
    mass /= mass.sum()
    objective = _objective_cpu(z, y, mass, 0.04)
    error = check_grad(lambda theta: objective(theta)[0], lambda theta: objective(theta)[1],
                       rng.normal(size=5))
    assert error < 1e-6


def test_constant_labels_empty_support_and_zero_mass_rows_are_explicit():
    x = np.array([[5., -1.], [1., 3.], [8., 2.]])
    y = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1]])
    fit = fit_fixed_support_logistic(x, y, np.zeros(2), np.ones(2), [[0, 1], [1], []],
                                     sample_weight=np.array([1, 3, 0]))
    assert [row["status"] for row in fit.diagnostics] == ["constant_label", "constant_label", "intercept_only"]
    np.testing.assert_array_equal(fit.coefficients, 0)
    np.testing.assert_allclose(expit(fit.bias), [1 - 1e-8, 1e-8, 0.25], atol=1e-12)
    np.testing.assert_array_equal(fit.fit_prediction_mean, fit.bias)


def test_optimization_failure_is_not_returned_as_a_valid_fit():
    x, y, mean, scale, supports = sample()
    with pytest.raises(RuntimeError, match="Concept 0 logistic optimizer failed"):
        fit_fixed_support_logistic(x, y, mean, scale, supports, maxiter=1)


@pytest.mark.parametrize("change, match", [
    ({"y": np.zeros((5, 2))}, "dimensions"),
    ({"mean": np.ones(1)}, "dimensions"),
    ({"scale": np.zeros(7)}, "positive scales"),
    ({"sample_weight": np.zeros(140)}, "positive total mass"),
    ({"sample_weight": np.full(140, -1)}, "nonnegative"),
    ({"sample_weight": np.ones(139)}, "Sample weights"),
    ({"sample_weight": np.full(140, np.nan)}, "finite"),
    ({"y": np.full((140, 2), 0.5)}, "binary"),
    ({"supports": [[1, 1], [0, 3]]}, "unique valid integer"),
    ({"supports": [[1.2], [0, 3]]}, "unique valid integer"),
    ({"supports": [[-1], [0, 3]]}, "unique valid integer"),
    ({"supports": [[7], [0, 3]]}, "unique valid integer"),
    ({"supports": [[True], [0, 3]]}, "unique valid integer"),
    ({"penalty": 0}, "positive"),
    ({"penalty": np.nan}, "positive"),
    ({"maxiter": 0}, "positive integer"),
    ({"maxiter": 2.5}, "positive integer"),
    ({"device": "mps"}, "cpu or a CUDA"),
])
def test_invalid_inputs_are_rejected(change, match):
    x, y, mean, scale, supports = sample()
    args: dict[str, Any] = dict(x=x, y=y, mean=mean, scale=scale, supports=supports)
    args.update(change)
    with pytest.raises(ValueError, match=match):
        fit_fixed_support_logistic(**args)


def test_cuda_matches_cpu_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x, y, mean, scale, supports = sample()
    cpu = fit_fixed_support_logistic(x, y, mean, scale, supports)
    cuda = fit_fixed_support_logistic(x, y, mean, scale, supports, device="cuda")
    np.testing.assert_allclose(cpu.coefficients, cuda.coefficients, atol=1e-7)
    np.testing.assert_allclose(cpu.bias, cuda.bias, atol=1e-7)
    np.testing.assert_allclose(cpu.fit_prediction_mean, cuda.fit_prediction_mean, atol=1e-7)
