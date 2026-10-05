import numpy as np
from scipy import sparse

from mm_sae.metrics.regression import (
    RidgePath, prediction_mse, score_matrices, sparse_moments, varying_columns,
)
from scripts.analyze_regression_correspondence import ranking_flags


def test_ridge_matches_direct_centered_fit_and_heldout_predictions():
    rng = np.random.default_rng(37)
    x, y = rng.normal(size=(80, 5)), rng.normal(size=(80, 3))
    fit = sparse_moments(sparse.csr_matrix(x), sparse.csr_matrix(y))
    g = fit.standardized(fit)
    b = RidgePath.from_moments(g[:5, :5], g[:5, 5:]).coefficients(.2)
    z = (np.column_stack([x, y]) - fit.mean) / fit.scale
    expected = np.linalg.solve(z[:, :5].T @ z[:, :5] / 80 + .2 * np.eye(5),
                               z[:, :5].T @ z[:, 5:] / 80)
    np.testing.assert_allclose(b, expected, atol=1e-12)
    new = rng.normal(size=(24, 8)) + 2  # Evaluation shifts must not be recentered away.
    held = sparse_moments(sparse.csr_matrix(new[:, :5]), sparse.csr_matrix(new[:, 5:]))
    h = held.standardized(fit)
    prediction_error = prediction_mse(b, h[:5, :5], h[:5, 5:], h[5:, 5:])
    zn = (new - fit.mean) / fit.scale
    np.testing.assert_allclose(prediction_error, ((zn[:, 5:] - zn[:, :5] @ b)**2).mean(0))


def test_shared_context_removed_when_both_concepts_have_distinct_predictors():
    within = np.array([[1, .85], [.85, 1]])
    scores = score_matrices(within, within, within, 1e-8, 1e-8)
    assert scores["pearson"][0, 1] == .85
    np.testing.assert_allclose(scores["ridge_bidirectional_mean"], np.eye(2), atol=1e-6)


def test_identity_within_modality_recovers_pearson_and_scalar_shrink_preserves_ranks():
    it = np.array([[.7, .2], [.3, .5]])
    scores = score_matrices(np.eye(2), it, np.eye(2), 0, 0)
    np.testing.assert_allclose(scores["ridge_bidirectional_mean"], it)
    penalized = score_matrices(np.eye(2), it, np.eye(2), .1, 2)
    np.testing.assert_allclose(penalized["ridge_bidirectional_mean"], it * (1 / 1.1 + 1 / 3) / 2)


def test_constant_features_excluded_and_input_sparse_arrays_unchanged():
    x = sparse.csr_matrix(np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32))
    old = x.copy()
    assert varying_columns(x).tolist() == [0]
    sparse_moments(x, x)
    np.testing.assert_array_equal(x.toarray(), old.toarray())


def test_bidirectional_ranks_use_the_anchor_diagonal_and_keep_zero_ties():
    score = np.array([[.6, .7, .6], [.1, .8, .2], [.0, -.1, .0]])
    valid, higher = ranking_flags(score)
    assert valid.sum() == 6
    assert higher.tolist() == [[False, True, False], [False, False, False], [False, False, False]]
    _, reverse = ranking_flags(score.T)
    assert reverse.tolist() == [[False, False, False], [False, False, False], [True, True, False]]
    # A missing diagonal is unscorable, never an implicit zero.
    score[0, 0] = np.nan
    valid, higher = ranking_flags(score)
    assert not valid[0].any() and not higher[0].any()
