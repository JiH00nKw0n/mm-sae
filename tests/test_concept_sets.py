from typing import Any

import numpy as np
import pytest
from scipy import linalg

from mm_sae.analysis.concept_sets import fit_concept_sets


def moments(x, y):
    centered_x, centered_y = x - x.mean(axis=0), y - y.mean(axis=0)
    xx = np.einsum("ni,nj->ij", centered_x, centered_x) / len(x)
    xy = np.einsum("ni,nj->ij", centered_x, centered_y) / len(x)
    return xx, xy, y.mean(axis=0), y.var(axis=0)


def objective(xx, xy, variance, support, penalty):
    if not support:
        return variance / 2
    local = xx[np.ix_(support, support)] + penalty * np.eye(len(support))
    beta = linalg.solve(local, xy[support], assume_a="pos")
    return float(0.5 * (variance - np.einsum("i,i->", beta, xy[support])))


def test_forward_choice_matches_brute_force_refit_and_dense_solution():
    rng = np.random.default_rng(162)
    x = rng.normal(size=(90, 7))
    x[:, 2] += x[:, 0]
    y = np.column_stack([2 * x[:, 0] - x[:, 4] + rng.normal(size=90), x[:, 3] + rng.normal(size=90)])
    xx, xy, mean, variance = moments(x, y)
    penalty = 0.07
    fit = fit_concept_sets(xx, xy, mean, variance, budgets=(1, 2, 4, 7), penalty=penalty)
    for concept, sequence in enumerate(fit.selected_features):
        selected = []
        for step, winner in enumerate(sequence):
            candidates = [j for j in range(len(xx)) if j not in selected]
            losses = [objective(xx, xy[:, concept], variance[concept], selected + [j], penalty)
                      for j in candidates]
            assert winner == candidates[int(np.argmin(losses))]
            selected.append(winner)
            np.testing.assert_allclose(fit.objective_history[concept, step + 1], min(losses), atol=1e-13)
    expected_dense = linalg.solve(xx + penalty * np.eye(len(xx)), xy, assume_a="pos")
    np.testing.assert_allclose(fit.dense_coefficients, expected_dense, atol=1e-12)
    np.testing.assert_allclose(fit.coefficients[7], expected_dense, atol=1e-12)
    assert np.min(fit.gain_history) >= -1e-12
    np.testing.assert_allclose(-np.diff(fit.objective_history, axis=1), fit.gain_history, atol=1e-13)


def test_budget_prefixes_and_fit_coefficients_have_exact_supported_objective():
    rng = np.random.default_rng(744)
    x = rng.normal(size=(75, 9))
    y = (rng.random((75, 3)) > 0.4).astype(float)
    inputs = moments(x, y)
    fit = fit_concept_sets(*inputs, budgets=(4, 1, 2, 4))
    short = fit_concept_sets(*inputs, budgets=(2,))
    np.testing.assert_allclose(short.coefficients[2], fit.coefficients[2])
    assert all(a == b[:2] for a, b in zip(short.selected_features, fit.selected_features))
    xx, xy, mean, variance = inputs
    for budget, coef in fit.coefficients.items():
        assert np.all(np.count_nonzero(coef, axis=0) <= budget)
        predictions = np.einsum("ni,ic->nc", x - x.mean(axis=0), coef) + mean
        sample_loss = 0.5 * (np.mean((predictions - y)**2, axis=0) + 0.01 * np.sum(coef**2, axis=0))
        np.testing.assert_allclose(sample_loss, fit.objective_history[:, budget], atol=1e-12)
        gradient = np.einsum("ij,jc->ic", xx + 0.01 * np.eye(len(xx)), coef) - xy
        np.testing.assert_allclose(gradient[coef != 0], 0, atol=1e-12)
    assert np.any(fit.coefficients[4] < 0)


def test_constant_labels_zero_features_and_larger_budgets():
    x = np.array([[1., 0.], [0., 0.], [-1., 0.]])
    y = np.array([[1., 1.], [1., 0.], [1., 0.]])
    fit = fit_concept_sets(*moments(x, y), budgets=(1, 4))
    assert fit.selected_features == [[], [0]]
    assert not fit.dense_coefficients[:, 0].any()
    assert not fit.coefficients[4][:, 0].any()
    np.testing.assert_array_equal(fit.coefficients[1], fit.coefficients[4])
    np.testing.assert_array_equal(fit.gain_history[:, 1:], 0)
    empty = fit_concept_sets(np.empty((0, 0)), np.empty((0, 2)), np.array([0.2, 1.]),
                             np.array([0.16, 0.]), budgets=(1, 4))
    assert empty.selected_features == [[], []]
    assert empty.dense_coefficients.shape == (0, 2)
    np.testing.assert_array_equal(empty.objective_history[:, 0], [0.08, 0])


def test_zero_crosscovariance_early_stop_and_exact_ties_choose_index():
    zero = fit_concept_sets(np.eye(3), np.zeros((3, 1)), np.array([0.5]), np.array([0.25]))
    assert zero.selected_features == [[]]
    np.testing.assert_array_equal(zero.objective_history, 0.125)
    tie = fit_concept_sets(np.eye(3), np.array([[0.1], [0.1], [0.1]]),
                           np.array([0.5]), np.array([0.25]), budgets=(1, 2, 3))
    assert tie.selected_features == [[0, 1, 2]]
    stopped = fit_concept_sets(np.eye(3), np.array([[0.1], [0.1], [0.1]]),
                               np.array([0.5]), np.array([0.25]), min_gain=0.1)
    assert stopped.selected_features == [[]]


@pytest.mark.parametrize("overrides, message", [
    ({"xx": np.array([[1., 2.], [2., 1.]])}, "positive semidefinite"),
    ({"xx": np.array([[1., 1.], [0., 1.]])}, "symmetric"),
    ({"xy": np.array([[np.nan], [0.]])}, "finite"),
    ({"xy": np.zeros((3, 1))}, "dimensions"),
    ({"label_variance": np.array([-1.])}, "nonnegative"),
    ({"label_variance": np.array([0.]), "xy": np.array([[1e-5], [0.]])}, "constant label"),
    ({"xy": np.array([[2.], [0.]])}, "covariance bound"),
    ({"xy": np.array([[0.4], [0.4]])}, "negative ridge objective"),
    ({"penalty": 0.}, "positive"),
    ({"min_gain": -1.}, "nonnegative"),
    ({"budgets": ()}, "positive integer"),
    ({"budgets": (1, 2.5)}, "positive integer"),
    ({"budgets": (True,)}, "positive integer"),
])
def test_invalid_inputs(overrides, message):
    params: dict[str, Any] = dict(xx=np.eye(2), xy=np.array([[0.1], [0.]]), label_mean=np.array([0.5]),
                                  label_variance=np.array([0.25]))
    params.update(overrides)
    with pytest.raises(ValueError, match=message):
        fit_concept_sets(**params)
