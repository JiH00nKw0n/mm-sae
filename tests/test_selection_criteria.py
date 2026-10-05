import numpy as np
import torch
from scipy import sparse

from mm_sae.analysis.feature_selection import (
    SparseUnivariateLogistic, paired_mean_drop, probe_attribution, ranked,
)
from mm_sae.analysis.linear import fit_logistic, logloss, scale_statistics


def test_paired_difference_retains_magnitude_and_sign():
    before = sparse.csr_matrix([[2., 0., 1.], [0., 6., 1.]])
    after = sparse.csr_matrix([[1., 0., 2.], [0., 0., 2.]])
    scores = paired_mean_drop(before, after)
    np.testing.assert_allclose(scores, [.5, 3., -1.])
    assert list(ranked(scores, [0, 1, 2])) == [1, 0, 2]


def test_parallel_single_probes_equal_independent_scipy_fits():
    torch.set_num_threads(1)
    rng = np.random.default_rng(43)
    x = rng.exponential(size=(900, 7)) * (rng.random((900, 7)) < .12)
    x[:, 5] = 0
    x[0, 5] = 4
    x[:, 6] = 1
    y = (x[:, 0]-2*x[:, 1]+rng.normal(0, .3, 900)) > .1
    sx = sparse.csr_matrix(x)
    mean, scale = scale_statistics(sx[:600])
    train = SparseUnivariateLogistic(sx[:600], scale, np.arange(7))
    tune = SparseUnivariateLogistic(sx[600:], scale, np.arange(7))
    result = train.fit_rank(y[:600], tune, y[600:], [.001, .1, 10.])
    actual = dict(zip(result["ranking"], result["tune_logloss"]))
    for j in range(7):
        model = fit_logistic(sx[:600], y[:600], sx[600:], y[600:],
                             np.array([j]), mean, scale, [.001, .1, 10.])
        assert model is not None
        np.testing.assert_allclose(actual[j], logloss(y[600:], model.predict(sx[600:])), atol=3e-6)
    assert set(result["ranking"][:2]) == {0, 1}
    coefficients = dict(zip(result["ranking"], result["coefficient"]))
    assert coefficients[0] > 0 and coefficients[1] < 0


def test_attribution_matches_linear_score_change_and_is_scale_invariant():
    x = sparse.csr_matrix([[1., 2.], [3., 1.], [0., 4.], [0., 2.]])
    labels = np.array([True, True, False, False])
    decoder = np.array([[1., 2.], [3., -1.]])
    direction = np.array([.2, -.3])
    score = probe_attribution(x, labels, decoder, direction)
    raw = (x @ decoder) @ direction
    np.testing.assert_allclose(score.sum(), raw[labels].mean()-raw[~labels].mean())
    scale = np.array([10., .5])
    np.testing.assert_allclose(score, probe_attribution(x.multiply(scale).tocsr(), labels,
                                                       decoder/scale[:, None], direction))
