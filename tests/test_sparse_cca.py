"""Behavioral checks for original-coordinate sparse regularized CCA."""

import numpy as np
import pytest
from scipy import linalg

from mm_sae.analysis.mapping_ablation import fit_common_projection
from mm_sae.analysis.sparse_cca import fit_sparse_cca


_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


def _moments():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(300, 4))
    x = _gemm(1., latent, rng.normal(size=(4, 7))) + rng.normal(size=(300, 7))
    y = _gemm(1., latent, rng.normal(size=(4, 6))) + rng.normal(size=(300, 6))
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    return _gemm(1., x.T, x) / len(x), _gemm(1., x.T, y) / len(x), _gemm(1., y.T, y) / len(x)


def test_full_support_recovers_dense_ridge_cca():
    xx, cross, yy = _moments()
    dense = fit_common_projection(xx, cross, yy, 4, whiten=True, ridge=.01)
    actual = fit_sparse_cca(xx, cross, yy, 4, k=7, ridge=.01)
    for j in range(4):
        sign = np.sign(dense.image[:, j] @ actual.image[:, j])
        np.testing.assert_allclose(actual.image[:, j] * sign, dense.image[:, j], atol=1e-8)
        np.testing.assert_allclose(actual.text[:, j] * sign, dense.text[:, j], atol=1e-8)
    np.testing.assert_allclose(actual.singular_values, dense.singular_values, atol=1e-9)


def test_support_norm_objective_and_reproducibility():
    xx, cross, yy = _moments()
    result = fit_sparse_cca(xx, cross, yy, 3, k=2, max_iter=20)
    repeat = fit_sparse_cca(xx, cross, yy, 3, k=2, max_iter=20)
    np.testing.assert_array_equal(result.image, repeat.image)
    np.testing.assert_array_equal(result.text, repeat.text)
    assert np.all(np.count_nonzero(result.image, axis=0) <= 2)
    assert np.all(np.count_nonzero(result.text, axis=0) <= 2)
    for component in result.metadata["components"]:
        assert np.min(np.diff(component["objective_history"])) >= -1e-10
        assert component["image_regularized_variance"] == pytest.approx(1.)
        assert component["text_regularized_variance"] == pytest.approx(1.)
    # Selected coordinates are original features despite off-diagonal covariance.
    assert np.max(np.abs(xx - np.diag(np.diag(xx)))) > .1
    assert result.metadata["fit_moments_only"]


def test_support_is_optimized_and_original_objective_is_recorded():
    xx, cross, yy = _moments()
    result = fit_sparse_cca(xx, cross, yy, 3, k=2, max_iter=30)
    assert any(c["accepted_support_swaps"] for c in result.metadata["components"])
    for j, component in enumerate(result.metadata["components"]):
        assert component["output_original_regularized_correlation"] == pytest.approx(
            result.image[:, j] @ cross @ result.text[:, j])


def test_iteration_limit_is_honest():
    result = fit_sparse_cca(*_moments(), 3, k=2, max_iter=1)
    assert any(not c["converged"] for c in result.metadata["components"])
    assert all(c["stop_reason"] == "max_iter" for c in result.metadata["components"] if not c["converged"])


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"ridge": -1}, {"dimensions": 0},
                                   {"max_iter": 0}, {"candidate_pool": 0}, {"tolerance": 0}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        fit_sparse_cca(*_moments(), **kwargs)


def test_output_orientation_uses_original_fit_covariance_and_preserves_deflation_history():
    rng = np.random.default_rng(1)
    loading = rng.normal(size=(8, 8))
    covariance = _gemm(1., loading, loading.T)
    xx, cross, yy = covariance[:4, :4], covariance[:4, 4:], covariance[4:, 4:]
    model = fit_sparse_cca(xx, cross, yy, 4, k=1)
    assert model.metadata["output_text_flipped_components"] == [2]
    residual = cross.copy()
    a, b = xx + .01 * np.eye(4), yy + .01 * np.eye(4)
    for j, record in enumerate(model.metadata["components"]):
        u = model.image[:, j]
        raw_v = model.text[:, j] * record["text_output_orientation"]
        rho = float(u @ residual @ raw_v)
        assert rho == pytest.approx(record["residual_objective"])
        assert float(u @ cross @ model.text[:, j]) >= 0
        assert record["output_original_regularized_correlation"] == pytest.approx(
            abs(record["original_regularized_correlation"]))
        residual -= rho * np.outer(a @ u, b @ raw_v)
