"""Numerical controls for fit-only preprocessing and common-space comparisons."""

import numpy as np
import pytest
from scipy import linalg, sparse

from mm_sae.analysis.mapping_ablation import fit_common_projection, group_projection, preprocess


def matmul(a, b):
    gemm = linalg.blas.get_blas_funcs(["gemm"], dtype=np.float64)[0]
    return gemm(1.0, np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def moments():
    rng = np.random.default_rng(19)
    a = rng.normal(size=(9, 9))
    covariance = matmul(a, a.T) + np.eye(9)
    scale = np.sqrt(np.diag(covariance))
    covariance = covariance / scale[:, None] / scale[None, :]
    return covariance[:5, :5], covariance[:5, 5:], covariance[5:, 5:]


def test_preprocessing_uses_supplied_fit_stats_and_preserves_inputs():
    # Evaluation data have a shifted mean and deliberately different variance.
    held = np.array([[100.0, 20.0], [120.0, 60.0]])
    original = held.copy()
    mean, scale = np.array([10.0, 4.0]), np.array([2.0, 8.0])
    np.testing.assert_array_equal(preprocess(held, mean, scale), held)
    np.testing.assert_array_equal(preprocess(held, mean, scale, "centered"), held - mean)
    np.testing.assert_array_equal(preprocess(sparse.csr_matrix(held), mean, scale, "standardized"),
                                  [[45.0, 2.0], [55.0, 7.0]])
    np.testing.assert_array_equal(held, original)


@pytest.mark.parametrize("mean,scale,mode", [
    ([1], [1, 1], "raw"), ([1, 1], [1], "raw"),
    ([1, np.nan], [1, 1], "centered"), ([1, 1], [1, np.inf], "raw"),
    ([1, 1], [1, 0], "standardized"), ([1, 1], [1, -1], "standardized"),
    ([1, 1], [1, 1], "unknown"),
])
def test_preprocessing_rejects_invalid_statistics(mean, scale, mode):
    with pytest.raises(ValueError):
        preprocess([[1, 2]], mean, scale, mode)


def test_cca_matches_existing_cholesky_reference_and_normalizes_ridge_covariance():
    xx, c, yy = moments()
    ridge = 0.01
    result = fit_common_projection(xx, c, yy, 3, whiten=True, ridge=ridge)
    left = linalg.cholesky(xx + ridge * np.eye(5), lower=True)
    right = linalg.cholesky(yy + ridge * np.eye(4), lower=True)
    cross = linalg.solve_triangular(left, c, lower=True)
    cross = linalg.solve_triangular(right, cross.T, lower=True).T
    u, s, vt = linalg.svd(cross, full_matrices=False)
    expected_i = linalg.solve_triangular(left.T, u[:, :3], lower=False)
    expected_t = linalg.solve_triangular(right.T, vt.T[:, :3], lower=False)
    np.testing.assert_allclose(result.image, expected_i, atol=1e-12)
    np.testing.assert_allclose(result.text, expected_t, atol=1e-12)
    np.testing.assert_allclose(result.singular_values, s[:3], atol=1e-12)
    np.testing.assert_allclose(matmul(result.image.T, matmul(xx + ridge * np.eye(5), result.image)),
                               np.eye(3), atol=1e-12)
    np.testing.assert_allclose(matmul(result.text.T, matmul(yy + ridge * np.eye(4), result.text)),
                               np.eye(3), atol=1e-12)
    np.testing.assert_allclose(matmul(result.image.T, matmul(c, result.text)),
                               np.diag(s[:3]), atol=1e-12)


def test_full_cross_svd_retains_only_common_rank_and_correct_cross_orientation():
    xx, c, yy = moments()
    result = fit_common_projection(xx, c, yy)
    assert result.image.shape == (5, 4)
    assert result.text.shape == (4, 4)
    np.testing.assert_allclose(matmul(result.image.T, matmul(c, result.text)),
                               np.diag(result.singular_values), atol=1e-12)
    np.testing.assert_allclose(matmul(result.image.T, result.image), np.eye(4), atol=1e-12)
    # The larger modality's fifth, orthogonal direction must not survive.
    full_u, _, _ = linalg.svd(c, full_matrices=True)
    null = full_u[:, 4:]
    np.testing.assert_allclose(matmul(null.T, result.image), np.zeros((1, 4)), atol=1e-12)
    u, _, vt = linalg.svd(c, full_matrices=False)
    rng = np.random.default_rng(20)
    x, y = rng.normal(size=(7, 5)), rng.normal(size=(8, 4))
    actual = matmul(matmul(x, result.image), matmul(y, result.text).T)
    expected = matmul(matmul(x, matmul(u, vt)), y.T)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    # No accidental singular-value weighting is applied.
    weighted = matmul(matmul(x, c), y.T)
    assert not np.allclose(actual, weighted)


@pytest.mark.parametrize("dimensions", [0, -1, 1.5, True])
def test_invalid_projection_dimensions_are_rejected(dimensions):
    xx, c, yy = moments()
    with pytest.raises(ValueError):
        fit_common_projection(xx, c, yy, dimensions)


def test_projection_dimension_clips_without_using_heldout_data():
    xx, c, yy = moments()
    result = fit_common_projection(xx, c, yy, 256)
    assert result.image.shape == (5, 4)
    assert result.text.shape == (4, 4)
    assert result.metadata["dimensions_requested"] == 256


def test_group_l1_normalization_removes_factor_scaling_ambiguity():
    xx, _, yy = moments()
    rng = np.random.default_rng(7)
    u, v = rng.uniform(size=(5, 3)), rng.uniform(size=(4, 3))
    scaling = np.array([0.02, 2.0, 100.0])
    for normalize in [False, True]:
        result = group_projection(u, v, xx, yy, normalize_variance=normalize)
        scaled = group_projection(u * scaling, v / scaling, xx, yy, normalize_variance=normalize)
        np.testing.assert_allclose(result.image, scaled.image, atol=1e-12)
        np.testing.assert_allclose(result.text, scaled.text, atol=1e-12)
        if not normalize:
            np.testing.assert_allclose(result.image.sum(axis=0), 1)
            np.testing.assert_allclose(result.text.sum(axis=0), 1)
        else:
            np.testing.assert_allclose(np.diag(matmul(result.image.T, matmul(xx, result.image))), 1)
            np.testing.assert_allclose(np.diag(matmul(result.text.T, matmul(yy, result.text))), 1)


def test_groups_missing_one_side_and_zero_variance_are_removed_in_pairs():
    # Columns 0 and 1 are absent on one side. Column 2 is present but cancels
    # under this valid covariance. Column 3 remains, with unequal side variance.
    u = np.array([[1, 0, 1, 3], [0, 0, 1, 0]], float)
    v = np.array([[0, 1, 1, 2], [0, 0, 0, 1]], float)
    xx = np.array([[1, -1], [-1, 1]], float)
    yy = np.eye(2)
    result = group_projection(u, v, xx, yy)
    assert result.metadata["retained_group_indices"] == [3]
    assert result.metadata["removed_missing_side_indices"] == [0, 1]
    assert result.metadata["removed_zero_fit_variance_indices"] == [2]
    np.testing.assert_allclose(result.image, [[1], [0]])
    np.testing.assert_allclose(result.text, [[2 / 3], [1 / 3]])
    normalized = group_projection(u, v, xx, yy, normalize_variance=True)
    np.testing.assert_allclose(normalized.text, [[2 / np.sqrt(5)], [1 / np.sqrt(5)]])


def test_all_dead_groups_return_explicit_empty_projection():
    result = group_projection(np.zeros((3, 2)), np.ones((4, 2)), np.eye(3), np.eye(4))
    assert result.image.shape == (3, 0)
    assert result.text.shape == (4, 0)
    assert result.metadata["dimensions"] == 0


def test_negative_factors_and_inconsistent_dimensions_are_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        group_projection([[-1]], [[1]], [[1]], [[1]])
    with pytest.raises(ValueError, match="same number"):
        group_projection([[1, 1]], [[1]], [[1]], [[1]])
    with pytest.raises(ValueError, match="negative group variance"):
        group_projection([[1]], [[1]], [[-1]], [[1]])
    with pytest.raises(ValueError, match="symmetric"):
        fit_common_projection([[1, 1], [0, 1]], [[1], [1]], [[1]])
