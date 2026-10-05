from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
from scipy import linalg

from mm_sae.analysis.concept_refinement import refine_concept_supports
from mm_sae.analysis.concept_sets import fit_concept_sets


def moments(x, y):
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    return (np.einsum("ni,nj->ij", xc, xc) / len(x),
            np.einsum("ni,nj->ij", xc, yc) / len(x), y.mean(axis=0), y.var(axis=0))


def fitted_objective(xx, response, variance, support, penalty):
    if not support:
        return variance / 2
    beta = linalg.solve(xx[np.ix_(support, support)] + penalty * np.eye(len(support)),
                        response[support], assume_a="pos")
    return float(0.5 * (variance - np.einsum("i,i->", beta, response[support])))


def test_exhaustive_best_swap_gains_match_brute_force_refits():
    rng = np.random.default_rng(4167)
    x = rng.normal(size=(160, 9))
    x[:, 2] += 1.5 * x[:, 0]
    x[:, 5] -= 0.8 * x[:, 4]
    y = np.column_stack([2 * x[:, 0] - x[:, 4] + rng.normal(size=len(x)),
                         3 * x[:, 3] + 1.5 * x[:, 8] + rng.normal(size=len(x))])
    xx, xy, mean, variance = moments(x, y)
    initial = np.zeros_like(xy)
    initial[[1, 2, 5], 0] = [1, -1, 2]
    initial[[2, 6, 7], 1] = [-1, 1, 1]
    penalty = 0.07
    fit = refine_concept_supports(xx, xy, mean, variance, initial, penalty=penalty, max_passes=20)
    for concept, trace in enumerate(fit.traces):
        support = list(trace.initial_support)
        for step, swap in enumerate(trace.swaps):
            objective = fitted_objective(xx, xy[:, concept], variance[concept], support, penalty)
            moves = [(removed, added) for removed in support for added in range(len(xx))
                     if added not in support]
            losses = [fitted_objective(xx, xy[:, concept], variance[concept],
                                       sorted([j for j in support if j != removed] + [added]), penalty)
                      for removed, added in moves]
            best = int(np.argmin(losses))
            assert (swap.removed_feature, swap.added_feature) == moves[best]
            np.testing.assert_allclose(swap.predicted_gain, objective - losses[best], atol=1e-12)
            np.testing.assert_allclose(swap.gain, swap.predicted_gain, atol=1e-12)
            np.testing.assert_allclose(trace.objective_history[step + 1], losses[best], atol=1e-12)
            support.remove(swap.removed_feature)
            support = sorted(support + [swap.added_feature])
        assert trace.converged and not trace.hit_pass_limit
        assert trace.best_remaining_gain <= fit.tolerance
        assert tuple(support) == trace.final_support
        assert np.all(np.diff(trace.objective_history) <= 0)
        assert np.count_nonzero(fit.coefficients[:, concept]) <= len(trace.initial_support)
        final_loss = fitted_objective(xx, xy[:, concept], variance[concept], support, penalty)
        np.testing.assert_allclose(trace.objective_history[-1], final_loss, atol=1e-12)
    assert any(fit.coefficients.ravel() < 0)
    # Traces can be saved as structured run diagnostics without custom serializers.
    assert asdict(fit.traces[0])["concept"] == 0


def test_bad_support_improves_and_initial_fixed_support_refit_is_separate():
    xx = np.eye(4)
    xy = np.array([[0.4], [-0.3], [0.1], [0.01]])
    mean, variance = np.array([0.5]), np.array([0.5])
    initial = np.array([[0.], [0.], [1.], [2.]])
    fit = refine_concept_supports(xx, xy, mean, variance, initial, max_passes=10)
    trace = fit.traces[0]
    assert trace.initial_support == (2, 3)
    assert trace.final_support == (0, 1)
    assert len(trace.swaps) == 2
    assert trace.refit_gain > 0
    assert trace.objective_history[-1] < trace.objective_history[0]
    np.testing.assert_allclose(fit.coefficients[:2, 0], xy[:2, 0] / 1.01)
    assert fit.coefficients[1, 0] < 0


def test_optimal_support_unchanged_and_deterministic_index_ties():
    xx = np.eye(4)
    xy = np.array([[0.2], [0.2], [0.01], [0.01]])
    mean, variance = np.array([0.5]), np.array([0.25])
    initial = np.array([[0.], [0.], [0.01 / 1.01], [0.01 / 1.01]])
    fit = refine_concept_supports(xx, xy, mean, variance, initial)
    assert (fit.traces[0].swaps[0].removed_feature, fit.traces[0].swaps[0].added_feature) == (2, 0)
    optimal = refine_concept_supports(xx, xy, mean, variance, fit.coefficients)
    np.testing.assert_allclose(optimal.coefficients, fit.coefficients, atol=1e-14)
    assert optimal.traces[0].converged
    assert optimal.traces[0].swaps == ()
    assert optimal.traces[0].refit_gain == pytest.approx(0, abs=1e-14)


def test_final_exhaustive_scan_distinguishes_convergence_from_pass_limit():
    xx = np.eye(4)
    xy = np.array([[0.4], [-0.3], [0.1], [0.01]])
    args = (xx, xy, np.array([0.5]), np.array([0.5]), np.array([[0.], [0.], [1.], [2.]]))
    stopped = refine_concept_supports(*args, max_passes=1)
    assert stopped.traces[0].hit_pass_limit and not stopped.traces[0].converged
    assert stopped.traces[0].best_remaining_gain > 0
    completed = refine_concept_supports(*args, max_passes=2)
    assert completed.traces[0].converged and not completed.traces[0].hit_pass_limit
    zero = refine_concept_supports(*args, max_passes=0)
    assert zero.traces[0].hit_pass_limit
    assert not zero.traces[0].swaps


def test_forward_initialized_refinement_never_worsens_fit_at_multiple_budgets():
    rng = np.random.default_rng(515)
    x = rng.normal(size=(80, 13))
    x[:, 3] += x[:, 7]
    y = (rng.random((len(x), 3)) > 0.5).astype(float)
    inputs = moments(x, y)
    forward = fit_concept_sets(*inputs, budgets=(1, 4, 8, 13))
    events = []
    for budget, coefficients in forward.coefficients.items():
        fit = refine_concept_supports(*inputs, coefficients,
                                      callback=lambda done, total, trace: events.append((done, total, trace)))
        assert np.isfinite(fit.coefficients).all()
        assert np.all(np.count_nonzero(fit.coefficients, axis=0) <= budget)
        for concept, trace in enumerate(fit.traces):
            assert trace.objective_history[-1] <= forward.objective_history[concept, budget] + 1e-13
            prediction = np.einsum("ni,i->n", x - x.mean(axis=0), fit.coefficients[:, concept]) + y[:, concept].mean()
            actual = 0.5 * (np.mean((prediction - y[:, concept])**2)
                            + fit.penalty * np.sum(fit.coefficients[:, concept]**2))
            np.testing.assert_allclose(actual, trace.objective_history[-1], atol=1e-12)
    assert len(events) == 12
    assert [(done, total) for done, total, _ in events[:3]] == [(1, 3), (2, 3), (3, 3)]


def test_empty_constant_and_full_supports():
    empty = refine_concept_supports(np.empty((0, 0)), np.empty((0, 2)), np.array([0.5, 1.]),
                                    np.array([0.25, 0.]), np.empty((0, 2)))
    assert all(trace.converged and not trace.swaps for trace in empty.traces)
    np.testing.assert_array_equal(empty.coefficients, np.empty((0, 2)))
    fit = refine_concept_supports(np.eye(2), np.zeros((2, 2)), np.array([0.5, 1.]),
                                  np.array([0.25, 0.]), np.array([[0., 1.], [0., 1.]]))
    np.testing.assert_array_equal(fit.coefficients, 0)
    assert all(trace.converged for trace in fit.traces)
    single = refine_concept_supports(np.eye(2), np.array([[0.1], [-0.3]]), np.array([0.5]),
                                     np.array([0.25]), np.array([[1.], [0.]]))
    assert single.traces[0].final_support == (1,)


@pytest.mark.parametrize("overrides, message", [
    ({"initial_coefficients": np.ones((3, 1))}, "dimensions"),
    ({"initial_coefficients": np.array([[np.nan], [1.]])}, "finite"),
    ({"max_passes": -1}, "nonnegative integer"),
    ({"max_passes": True}, "nonnegative integer"),
    ({"max_passes": 1.5}, "nonnegative integer"),
    ({"penalty": 0.}, "positive"),
    ({"tolerance": -1.}, "nonnegative"),
    ({"xx": np.array([[1., 2.], [2., 1.]])}, "positive semidefinite"),
    ({"xx": np.array([[1., 0.1], [0., 1.]])}, "symmetric"),
    ({"xy": np.array([[2.], [0.]])}, "covariance bound"),
    ({"xy": np.array([[0.4], [0.4]])}, "negative ridge objective"),
])
def test_invalid_inputs(overrides, message):
    params: dict[str, Any] = dict(xx=np.eye(2), xy=np.array([[0.1], [0.]]), label_mean=np.array([0.5]),
                                  label_variance=np.array([0.25]), initial_coefficients=np.ones((2, 1)))
    params.update(overrides)
    with pytest.raises(ValueError, match=message):
        refine_concept_supports(**params)
