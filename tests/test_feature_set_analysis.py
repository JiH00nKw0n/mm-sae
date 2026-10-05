import itertools

import numpy as np
from scipy import sparse
from scipy.stats import mannwhitneyu

from mm_sae.analysis.evaluation import auroc, auc_interval, sparse_binary_auroc
from mm_sae.analysis.linear import compress_zero_rows, fit_logistic, identity_score, logloss
from mm_sae.analysis.regression import CrossMoments, connection_models, forward_path, predict
from mm_sae.analysis.selection import removal_ranking
from experiments.feature_sets.intervention import shuffle_image_groups


def test_auc_ties_and_sparse_zeros_match_reference():
    rng = np.random.default_rng(91)
    x = rng.choice([0., 0., 0., 1., 2.], size=(50, 7))
    y = rng.random(50) < .4
    expected = np.array([mannwhitneyu(x[y, j], x[~y, j]).statistic/(y.sum()*(~y).sum())
                         for j in range(7)])
    np.testing.assert_allclose(sparse_binary_auroc(sparse.csr_matrix(x), y), expected)
    np.testing.assert_allclose([auroc(y, x[:, j]) for j in range(7)], expected)
    weights = rng.integers(1, 5, 50)
    np.testing.assert_allclose(auroc(y, x[:, 0], weights),
                               auroc(np.repeat(y, weights), np.repeat(x[:, 0], weights)))


def test_zero_compression_preserves_exact_logistic_objective():
    raw = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 2]], float)
    labels = np.array([0, 0, 1, 1, 0])
    x, y, w = compress_zero_rows(raw, labels)
    mean, scale, coeff = np.array([.2, .5]), np.array([.5, .7]), np.array([.9, -.2])
    actual = logloss(y, ((x-mean)/scale) @ coeff+.3, w)
    expected = logloss(labels, ((raw-mean)/scale) @ coeff+.3)
    np.testing.assert_allclose(actual, expected)


def test_two_complementary_features_improve_held_out_removal_auc():
    before = sparse.csr_matrix(np.tile([[1., 0.], [0., 1.]], (60, 1)))
    after = sparse.csr_matrix(before.shape)
    x = sparse.vstack([before, after]).tocsr()
    y = np.r_[np.ones(120), np.zeros(120)]
    rank, values = removal_ranking(before, after, np.array([0, 1]))
    assert list(rank) == [0, 1]
    np.testing.assert_allclose(values, .75)
    model = fit_logistic(x, y, x, y, np.array([0, 1]), np.zeros(2), np.ones(2), [.01, .1], True)
    assert model is not None and np.all(model.weight >= 0)
    evaluation = sparse.csr_matrix([[3., 0.], [0., 2.], [0., 0.], [0., 0.]])
    assert auroc([1, 1, 0, 0], model.predict(evaluation)) == 1
    one = identity_score(0, np.zeros(2), np.ones(2))
    assert auroc([1, 1, 0, 0], one.predict(evaluation)) == .75


def test_cluster_bootstrap_keeps_paired_views_and_handles_missing_class():
    result = auc_interval(np.array([1, 1, 0, 0]), np.array([1., 2., 0., 0.]),
                          np.array([0, 1, 0, 1]), 50, 17)
    assert result["auroc"] == result["ci_low"] == result["ci_high"] == 1
    assert np.isnan(auroc([1, 1], [1, 2]))


def test_connections_reuse_and_multiple_inputs_preserve_targets():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1500, 2))
    y = np.column_stack([x[:, 0], x[:, 0]+x[:, 1], 2*x[:, 0]])
    fit = CrossMoments.compute(x[:500], y[:500])
    tune = CrossMoments.compute(x[500:1000], y[500:1000])
    models = connection_models(fit, tune, np.ones(2), np.ones(3), np.arange(2), [1, 2], [.0001, .01])
    errors = {k: np.mean((predict(x[1000:], b, fit, np.ones(2), np.ones(3))-y[1000:])**2)
              for k, b in models.items()}
    assert all(b.shape == (2, 3) for b in models.values())
    assert np.all(np.sum(models["one_to_one"] != 0, axis=0) <= 1)
    assert np.all(np.sum(models["one_to_one"] != 0, axis=1) <= 1)
    assert errors["reusable_one"] < errors["one_to_one"]
    assert errors["reusable_2"] < errors["reusable_one"]


def test_forward_selection_matches_brute_force_extension_fit_error():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(200, 5)), rng.normal(size=200)
    g, h, v = x.T @ x/200, x.T @ y/200, np.mean(y*y)
    penalty = .1
    path = forward_path(g, h, v, np.arange(5), penalty, [1, 2, 3])
    first = path[1][0][0]
    errors = []
    for other in range(5):
        if other == first:
            continue
        idx = np.array([first, other])
        b = np.linalg.solve(g[np.ix_(idx, idx)]+penalty*np.eye(2), h[idx])
        errors.append((v-2*b @ h[idx]+b @ g[np.ix_(idx, idx)] @ b, other))
    assert path[2][0][1] == min(errors)[1]
    for n, (idx, b) in path.items():
        assert len(idx) == n
        np.testing.assert_allclose(b, np.linalg.solve(g[np.ix_(idx, idx)]+penalty*np.eye(n), h[idx]))


def test_image_shuffle_never_splits_caption_groups_or_crosses_categories():
    groups = np.array([0, 0, 1, 2, 2, 3])
    cats = np.array([10, 10, 10, 11, 11, 11])
    indices = shuffle_image_groups(groups, cats, 8)
    np.testing.assert_array_equal(cats, cats[indices])
    for category, image in itertools.product(np.unique(cats), np.unique(groups)):
        rows = np.flatnonzero((cats == category) & (groups == image))
        if len(rows):
            assert len(np.unique(groups[indices[rows]])) == 1
            assert groups[indices[rows[0]]] != image
