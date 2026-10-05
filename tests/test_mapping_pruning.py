"""Check signed pruning constraints and the unpruned retrieval geometry."""

import numpy as np
from scipy import linalg

from mm_sae.analysis.mapping_pruning import procrustes_matrix, prune_coefficients


def test_signed_pruning_preserves_values_and_limits_both_degrees():
    values = np.array([[1., -8., 3.], [-8., 2., -4.], [5., 6., 7.], [0., -9., 1.]])
    for k in (1, 2, 5):
        pruned = prune_coefficients(values, k, rule="mutual")
        support = pruned != 0
        assert (support.sum(0) <= k).all() and (support.sum(1) <= k).all()
        np.testing.assert_array_equal(pruned[support], values[support])
    assert prune_coefficients(values, 1, rule="mutual")[3, 1] == -9
    np.testing.assert_array_equal(prune_coefficients(values, 5, rule="mutual"), values)


def test_column_pruning_limits_component_inputs_and_breaks_ties_stably():
    values = np.array([[3., -2.], [-3., 2.], [1., 4.]])
    np.testing.assert_array_equal(prune_coefficients(values, 1, rule="column"), [[3, 0], [0, 0], [0, 4]])


def test_procrustes_larger_space_keeps_original_common_cosine():
    rng = np.random.default_rng(44)
    for ni, nt in ((7, 4), (4, 7)):
        cross = rng.normal(size=(ni, nt))
        x, y = rng.normal(size=(9, ni)), rng.normal(size=(8, nt))
        u, _, vt = linalg.svd(cross, full_matrices=True)
        original_x = np.pad(x @ u, ((0, 0), (0, max(ni, nt) - ni)))
        original_y = np.pad(y @ vt.T, ((0, 0), (0, max(ni, nt) - nt)))
        direct = procrustes_matrix(cross)
        a, b = (x, y @ direct.T) if ni >= nt else (x @ direct, y)
        original = original_x @ original_y.T / np.outer(np.linalg.norm(original_x, axis=1),
                                                       np.linalg.norm(original_y, axis=1))
        actual = a @ b.T / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
        np.testing.assert_allclose(actual, original, atol=1e-12)
