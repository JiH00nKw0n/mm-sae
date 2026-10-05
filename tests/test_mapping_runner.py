import numpy as np

from experiments.mapping_suite.run import calibrated_coefficients, candidate_settings, r2_and_error
from mm_sae.metrics.regression import Moments


def test_calibration_retains_ratios_and_unmatched_target():
    p = np.array([[2., 0.], [1., 0.]])
    xx = np.eye(2)
    c = np.array([[.8, .1], [.4, .2]])
    b = calibrated_coefficients(p, xx, c, penalty=0)
    np.testing.assert_allclose(b[:, 0], [.8, .4])
    np.testing.assert_array_equal(b[:, 1], 0)


def test_prediction_error_keeps_test_mean_shift():
    fit = Moments(10, np.zeros(2), np.eye(2))
    # Test target has mean 2 and variance 1. Predicting fit mean 0 gives MSE 5.
    test = Moments(10, np.array([0., 2.]), np.diag([1., 5.]))
    r2, mse = r2_and_error(np.zeros((1, 1)), test, fit, 1)
    np.testing.assert_allclose(mse, [5.])
    np.testing.assert_allclose(r2, [-4.])


def test_grid_contains_each_setting_and_preserves_fixed_options():
    result = candidate_settings([{"name": "bounded", "method": "partial_b_matching",
        "options": {"k_text": 3}, "grid": {"k_image": [1, 2], "tau": [0., .1]}}])
    assert len(result) == 4
    assert len({r["key"] for r in result}) == 4
    assert all(r["options"]["k_text"] == 3 for r in result)
