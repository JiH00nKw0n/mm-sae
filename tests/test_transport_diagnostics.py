"""Support and mass concentration must remain different measurements."""

import numpy as np

from mm_sae.analysis.mapping import fit_mapping
from mm_sae.analysis.mapping_pruning import prune_coefficients
from mm_sae.analysis.transport_diagnostics import transport_summary


def test_uniform_concentrated_and_empty_rows_have_known_effective_counts():
    values = np.array([[.25, .25, .25, .25], [1., 0, 0, 0], [0, 0, 0, 0]])
    stats = transport_summary(values)
    assert stats["edge_count"] == 5
    np.testing.assert_allclose(stats["image_effective_degree_mean"], 5 / 3)
    np.testing.assert_allclose(stats["image_mass95_degree_mean"], 5 / 3)
    np.testing.assert_allclose(stats["image_coverage"], 2 / 3)
    scaled = transport_summary(values * .01)
    for key in ("image_effective_degree_mean", "image_mass95_degree_mean", "text_effective_degree_mean"):
        np.testing.assert_allclose(stats[key], scaled[key])


def test_small_epsilon_concentrates_but_does_not_remove_edges():
    correlation = np.array([[.8, .2], [.2, .8]])
    low = fit_mapping(correlation, "sinkhorn", {"epsilon": .03, "tol": 1e-12}).weights
    high = fit_mapping(correlation, "sinkhorn", {"epsilon": .3, "tol": 1e-12}).weights
    for weights in (low, high):
        np.testing.assert_allclose(weights.sum(0), [.5, .5], atol=1e-12)
        np.testing.assert_allclose(weights.sum(1), [.5, .5], atol=1e-12)
        assert transport_summary(weights)["edge_count"] == 4
    assert (transport_summary(low)["image_effective_degree_mean"]
            < transport_summary(high)["image_effective_degree_mean"])
    pruned = prune_coefficients(low, 1, rule="mutual")
    assert transport_summary(pruned, low)["edge_count"] == 2
    assert transport_summary(pruned, low)["retained_mass"] < 1
