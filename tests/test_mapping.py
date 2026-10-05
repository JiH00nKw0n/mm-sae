"""Small reference problems establish the fitted objectives and constraints."""

from itertools import product

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

from mm_sae.analysis.mapping import fit_mapping


def _brute_matching(c, tau, image_cap, text_cap, budget):
    best = 0.0
    for bits in product((0, 1), repeat=c.size):
        weights = np.array(bits).reshape(c.shape)
        if (np.any(weights.sum(1) > image_cap) or np.any(weights.sum(0) > text_cap)
                or weights.sum() > budget or np.any(weights[c <= tau])):
            continue
        best = max(best, float(np.sum((c - tau) * weights)))
    return best


@pytest.mark.parametrize("budget", [None, 0, 1, 3, 5])
def test_b_matching_equals_binary_enumeration_without_duplicate_pairs(budget):
    for seed in range(4):
        c = np.random.default_rng(seed).uniform(-.4, .9, size=(2, 3))
        image_cap, text_cap, tau = np.array([2, 1]), np.array([1, 2, 1]), .1
        options = dict(k_image=image_cap, k_text=text_cap, tau=tau, edge_budget=budget)
        result = fit_mapping(c, "partial_b_matching", options)
        expected = _brute_matching(c, tau, image_cap, text_cap, c.size if budget is None else budget)
        np.testing.assert_allclose(result.metadata["objective"], expected, atol=1e-10)
        assert set(np.unique(result.weights)) <= {0, 1}
        assert np.all(result.weights.sum(1) <= image_cap)
        assert np.all(result.weights.sum(0) <= text_cap)
        assert np.all(result.weights[c <= tau] == 0)
        assert result.metadata["integrality_residual"] < 1e-6


def test_signed_baselines_keep_negative_edges_and_topk_ties_are_stable():
    c = np.array([[-.8, -.1, -.1], [-.5, -.4, -.9]])
    greedy = fit_mapping(c, "greedy").weights
    np.testing.assert_array_equal(greedy, [[0, 1, 0], [0, 1, 0]])
    topk = fit_mapping(c, "topk", {"k": 2}).weights
    np.testing.assert_array_equal(topk, [[0, 1, 1], [1, 1, 0]])
    hungarian = fit_mapping(c, "hungarian").weights
    assert hungarian.sum() == 2
    np.testing.assert_allclose(np.sum(hungarian * c), -.5)
    partial = fit_mapping(c, "partial_b_matching", {"tau": 0})
    assert partial.weights.sum() == 0


def test_b_matching_rejects_noninteger_degree_caps():
    with pytest.raises(ValueError, match="integer"):
        fit_mapping(np.ones((2, 3)), "partial_b_matching", {"k_image": 1.5})


def _reference_qp(c, tau, penalty, image_cap, text_cap):
    m, n = c.shape
    constraints = np.zeros((m + n, m * n))
    for row in range(m):
        constraints[row, row * n:(row + 1) * n] = 1
    for column in range(n):
        constraints[m + column, column::n] = 1
    score = (c - tau).ravel()

    def objective(x):
        return .5 * penalty * x @ x - score @ x, penalty * x - score

    solved = minimize(objective, np.zeros(c.size), jac=True, method="SLSQP",
                      bounds=Bounds(0, 1),
                      constraints=LinearConstraint(constraints, 0, np.r_[image_cap, text_cap]),
                      options={"ftol": 1e-12, "maxiter": 2000})
    assert solved.success, solved.message
    return solved.x.reshape(c.shape), -float(solved.fun)


@pytest.mark.parametrize("seed", [0, 9, 23])
def test_sparse_transport_matches_independent_constrained_qp(seed):
    c = np.random.default_rng(seed).uniform(-.3, 1.3, (4, 3))
    image_cap, text_cap = np.array([.6, 1.1, .8, 1.7]), np.array([1.4, 1.1, .9])
    tau, penalty = .12, .4
    expected, expected_objective = _reference_qp(c, tau, penalty, image_cap, text_cap)
    events = []
    result = fit_mapping(c, "sparse_transport", dict(k_image=image_cap, k_text=text_cap,
                                                   tau=tau, lambda_=penalty, tol=1e-11),
                         progress=events.append)
    assert np.min(result.weights) >= 0 and np.max(result.weights) <= 1
    assert np.all(result.weights.sum(1) <= image_cap + 1e-12)
    assert np.all(result.weights.sum(0) <= text_cap + 1e-12)
    np.testing.assert_allclose(result.weights, expected, atol=2e-6)
    np.testing.assert_allclose(result.metadata["objective"], expected_objective, atol=2e-7)
    assert result.metadata["duality_gap"] <= 2e-7
    assert result.metadata["dual_objective"] >= expected_objective - 1e-10
    assert expected_objective - result.metadata["objective"] <= result.metadata["duality_gap"] + 1e-10
    assert result.metadata["converged"]
    completed = [event.get("iterations", event.get("iteration", 0)) for event in events]
    assert completed == sorted(completed)


def test_sparse_transport_zero_capacity_and_iteration_limit_are_explicit():
    c = np.array([[1., 1., .5], [.8, .6, 1.]])
    result = fit_mapping(c, "sparse_transport", {"k_image": [0, .7], "k_text": .4,
                                               "lambda_": .15, "max_iter": 1, "tol": 1e-12})
    assert not result.metadata["converged"]
    assert result.metadata["iterations"] == 1
    assert np.all(result.weights[0] == 0)
    assert result.weights[1].sum() <= .7 + 1e-12
    assert np.all(result.weights.sum(0) <= .4 + 1e-12)
    assert result.metadata["duality_gap"] > 0


def test_rectangular_sinkhorn_preserves_balanced_raw_mass_and_signed_scores():
    c = np.array([[.9, -.4, .2], [-.7, .8, -.3]])
    result = fit_mapping(c, "sinkhorn", {"epsilon": .2, "tol": 1e-10})
    assert result.metadata["converged"]
    np.testing.assert_allclose(result.weights.sum(1), np.full(2, 1 / 2), atol=1e-10)
    np.testing.assert_allclose(result.weights.sum(0), np.full(3, 1 / 3), atol=1e-10)
    np.testing.assert_allclose(result.weights.sum(), 1, atol=1e-12)
    shifted = fit_mapping(c - 1000, "sinkhorn", {"epsilon": .2, "tol": 1e-10})
    np.testing.assert_allclose(shifted.weights, result.weights, atol=1e-12)
    # Signed preferences survive even when every input entry is negative.
    assert shifted.weights[0, 0] > shifted.weights[1, 0]


def test_sinkhorn_supports_zero_target_masses_and_reports_nonconvergence():
    c = np.array([[2., -.1, .5], [.1, 1., -.4]])
    result = fit_mapping(c, "sinkhorn", {"row_mass": [2, 1], "col_mass": [0, 3, 1],
                                       "epsilon": .3, "tol": 1e-10})
    np.testing.assert_allclose(result.weights.sum(1), [2 / 3, 1 / 3], atol=1e-10)
    np.testing.assert_allclose(result.weights.sum(0), [0, .75, .25], atol=1e-10)
    incomplete = fit_mapping(c, "sinkhorn", {"epsilon": .001, "max_iter": 1, "tol": 1e-12})
    assert not incomplete.metadata["converged"]
    assert incomplete.metadata["iterations"] == 1


def test_factorization_obeys_support_caps_and_decreases_the_signed_objective():
    rng = np.random.default_rng(13)
    c = rng.uniform(-.5, 1., size=(8, 7))
    options = {"rank": 3, "k_image": 3, "k_text": 2, "lambda_": .04,
               "n_init": 3, "seed": 71, "max_iter": 100, "tol": 1e-9}
    events = []
    result = fit_mapping(c, "sparse_factorization", options, progress=events.append)
    assert result.u is not None and result.v is not None
    assert result.u.shape == (8, 3) and result.v.shape == (7, 3)
    assert np.all(result.u >= 0) and np.all(result.v >= 0)
    assert np.all(np.count_nonzero(result.u, axis=0) <= 3)
    assert np.all(np.count_nonzero(result.v, axis=0) <= 2)
    np.testing.assert_allclose(result.u.sum(0), result.v.sum(0), atol=1e-12)
    np.testing.assert_allclose(result.weights, result.u @ result.v.T)
    expected = .5 * np.sum((c - result.weights)**2) + .04 * (result.u.sum() + result.v.sum())
    np.testing.assert_allclose(result.metadata["objective"], expected, atol=1e-10)
    assert result.metadata["objective"] < result.metadata["initial_objective"]
    assert np.max(np.diff(result.metadata["objective_history"])) <= 1e-9
    assert result.metadata["objective"] == min(result.metadata["restart_objectives"])
    assert "no global guarantee" in result.metadata["optimality"]
    completed = [event.get("iterations", event.get("iteration", 0)) for event in events]
    assert completed == sorted(completed)
    assert completed[-1] == sum(result.metadata["restart_iterations"])
    repeated = fit_mapping(c, "sparse_factorization", options)
    assert repeated.u is not None
    np.testing.assert_array_equal(repeated.weights, result.weights)
    np.testing.assert_array_equal(repeated.u, result.u)


def test_factorization_all_negative_matrix_has_zero_optimum():
    c = -np.ones((3, 4))
    result = fit_mapping(c, "sparse_factorization", {"rank": 2, "lambda_": .1})
    np.testing.assert_array_equal(result.weights, np.zeros_like(c))
    assert result.metadata["objective"] == 6


def test_progress_empty_shapes_and_invalid_input():
    events = []
    result = fit_mapping(np.ones((2, 3)), "greedy", progress=events.append)
    assert result.weights.shape == (2, 3)
    assert [event["status"] for event in events] == ["started", "finished"]
    empty = fit_mapping(np.empty((0, 3)), "sinkhorn")
    assert empty.weights.shape == (0, 3) and empty.metadata["empty"]
    for bad in [np.ones(3), np.array([[np.nan]]), np.array([[np.inf]])]:
        with pytest.raises(ValueError, match="two-dimensional"):
            fit_mapping(bad, "greedy")
    with pytest.raises(ValueError, match="Unknown"):
        fit_mapping(np.ones((2, 2)), "absent")
