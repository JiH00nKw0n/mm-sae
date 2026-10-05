import numpy as np
import pytest
from scipy import sparse

from mm_sae.metrics.reweighting import (
    balanced_binary_masses, binary_correlation, grouped_moments,
    independent_binary_masses, sparse_columns,
)


def test_sparse_moments_match_direct_weighted_pearson_including_zero_rows():
    rng = np.random.default_rng(91)
    values = rng.uniform(size=(101, 4))
    values[rng.uniform(size=values.shape) < .7] = 0
    groups = rng.integers(-1, 4, size=len(values))
    counts = np.bincount(groups[groups >= 0], minlength=4)
    moments = grouped_moments(sparse_columns(sparse.csc_matrix(values)), groups, counts)
    for rho in [0, .2, .8]:
        masses = balanced_binary_masses(rho)
        keep = groups >= 0
        weights = masses[groups[keep]] / counts[groups[keep]]
        selected = values[keep]
        centered = selected - np.sum(selected * weights[:, None], axis=0)
        covariance = centered.T @ (weights[:, None] * centered)
        expected = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        np.testing.assert_allclose(moments.correlations(masses), expected, atol=1e-13)


def test_binary_marginals_and_correlation_are_exact():
    groups = np.arange(4)
    labels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    moments = grouped_moments(sparse_columns(labels), groups, np.ones(4, dtype=int))
    for rho in [0, .2, .4, .6, .8]:
        masses = balanced_binary_masses(rho)
        np.testing.assert_allclose(masses @ labels, [.5, .5])
        assert moments.correlations(masses)[0, 1] == pytest.approx(rho)


def test_empty_groups_constants_and_shared_columns_are_not_false_reversals():
    values = np.array([[0, 3, 1, 1], [2, 3, 4, 4], [0, 3, 2, 2], [1, 3, 0, 0]])
    groups = np.arange(4)
    moments = grouped_moments(sparse_columns(values), groups, np.ones(4, dtype=int))
    result = moments.correlations(balanced_binary_masses(.8))
    assert np.isnan(result[1]).all()
    assert result[0, 2] == result[0, 3]
    moments.counts[0] = 0
    with pytest.raises(ValueError, match="empty stratum"):
        moments.correlations(balanced_binary_masses(0))


def test_independence_keeps_unequal_natural_prevalences_and_eliminates_dependence():
    counts = np.array([565645, 827, 11205, 14076])
    natural = counts / counts.sum()
    target = independent_binary_masses(counts)
    labels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_allclose(target @ labels, natural @ labels, atol=1e-15)
    assert binary_correlation(natural) == pytest.approx(.7167419409813958)
    assert binary_correlation(target) == pytest.approx(0, abs=1e-14)
    # Perfect concept features have self-correlation 1, and cross-correlation 0 after reweighting.
    moments = grouped_moments(sparse_columns(labels), np.arange(4), np.ones(4, dtype=int))
    np.testing.assert_allclose(moments.correlations(target), np.eye(2), atol=1e-14)


def test_independence_does_not_invent_missing_observations():
    counts = np.array([50, 0, 20, 30])
    target = independent_binary_masses(counts)
    assert target[1] > 0  # This distribution cannot be represented by reweighting the available rows.
    moments = grouped_moments(sparse_columns(np.zeros((100, 1))), np.repeat(np.arange(4), counts), counts)
    with pytest.raises(ValueError, match="empty stratum"):
        moments.correlations(target)
