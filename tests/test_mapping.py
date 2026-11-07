import numpy as np
from scipy.sparse import csr_matrix
from mm_sae.mapping.rotation import rotate, row_normalize
from mm_sae.mapping.model import project, prune


def test_rotation_keeps_cross_scores_and_norms_and_optimizes_the_requested_criterion():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(7, 4)), rng.normal(size=(8, 4))
    x, y = rng.normal(size=(45, 7)), rng.normal(size=(45, 8))
    for kind, left, right in (
        ("covariance", row_normalize(a), row_normalize(b)),
        ("squared_score_covariance", x @ a, y @ b),
    ):
        r, diagnostic = rotate(left, right, kind, 40, 1e-8)
        np.testing.assert_allclose(r.T @ r, np.eye(4), atol=1e-12)
        np.testing.assert_allclose((a @ r) @ (b @ r).T, a @ b.T, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(x @ a @ r, axis=1), np.linalg.norm(x @ a, axis=1))
        assert diagnostic["final"] > diagnostic["initial"]


def test_sparse_projection_matches_standardized_dense_reference():
    x = np.array([[0, 2, 0, 17], [1, 0, 0, 17]], float)
    mean = np.array([0.3, 0.6, 0, 17])
    inv = np.array([1, 2, 0, 0])
    w = np.arange(8).reshape(4, 2)
    np.testing.assert_allclose(project(csr_matrix(x), w, mean, inv), ((x - mean) * inv) @ w, atol=1e-6)
    assert ((prune(w, 2) != 0).sum(axis=0) == 2).all()
