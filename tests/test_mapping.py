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


def test_concept_budget_selects_largest_coefficients_without_changing_retrieval_weights():
    from mm_sae.mapping.model import scores

    class IdentityEncoder:
        def activations(self, side, block):
            return csr_matrix(block)

    x = np.eye(3, dtype=np.float32)
    weights = np.array([[-3.0, 0.2], [2.0, -5.0], [1.0, 1.0]])
    mapping = {"image": weights.copy(), "image_inv": np.array([1.0, 2.0, 3.0]), "image_mean": np.zeros(3)}
    capped = scores(IdentityEncoder(), mapping, "image", x, 2, feature_budget=1)
    original = scores(IdentityEncoder(), mapping, "image", x, 2)
    np.testing.assert_array_equal(capped, [[-3.0, 0.0], [0.0, -10.0], [0.0, 0.0]])
    np.testing.assert_allclose(original, weights * np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_array_equal(mapping["image"], weights)


def test_direct_maps_in_matched_image_coordinates():
    from mm_sae.config import Mapping
    from mm_sae.mapping.direct import fit_in_image_coordinates

    cross = np.array([[0.2, 0.9, 0], [0.8, 0.3, 0.1], [0.1, 0.0, 0.7], [0, 0.1, 0]])
    for method in ("sinkhorn", "procrustes"):
        a, b, info = fit_in_image_coordinates(cross, Mapping(method=method, rank=2, tolerance=1e-9, transport_epsilon=0.2))
        assert a.shape == (4, 2) and b.shape == (3, 2)
        np.testing.assert_array_equal(a.argmax(0), [0, 1])
        t = b[info["text_indices_active"]].T
        if method == "sinkhorn":
            assert info["converged"]
            np.testing.assert_allclose(t.sum(0), 1.0, atol=1e-8)
            np.testing.assert_allclose(t.sum(1), 1.0, atol=1e-8)
            assert np.all(t >= 0)
        else:
            np.testing.assert_allclose(t @ t.T, np.eye(2), atol=1e-12)
            u, _, vt = np.linalg.svd(cross[np.ix_([0, 1], [1, 0])])
            np.testing.assert_allclose(t, u @ vt, atol=1e-12)
