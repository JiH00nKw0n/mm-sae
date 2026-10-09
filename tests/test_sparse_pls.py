import numpy as np

from mm_sae.analysis.sparse_pls import fit_sparse_pls


def test_unrestricted_recovers_svd_and_sparse_respects_support():
    cross = np.random.default_rng(7).normal(size=(12, 9))
    u, s, vt = np.linalg.svd(cross, full_matrices=False)
    dense = fit_sparse_pls(cross, u[:, :4], vt.T[:, :4], k=12)
    np.testing.assert_allclose(dense.singular_values, s[:4], atol=1e-9)
    np.testing.assert_allclose(dense.image @ np.diag(dense.singular_values) @ dense.text.T,
                               u[:, :4] @ np.diag(s[:4]) @ vt[:4], atol=1e-9)
    sparse = fit_sparse_pls(cross, u[:, :4], vt.T[:, :4], k=2)
    assert ((sparse.image != 0).sum(0) <= 2).all()
    assert ((sparse.text != 0).sum(0) <= 2).all()
    np.testing.assert_allclose(np.linalg.norm(sparse.image, axis=0), 1)
    np.testing.assert_allclose(np.linalg.norm(sparse.text, axis=0), 1)
    for component in sparse.metadata["components"]:
        assert np.all(np.diff(component["objective_history"]) >= -1e-9)
