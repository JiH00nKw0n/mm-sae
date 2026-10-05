import json

import numpy as np
import pytest
from scipy.optimize import lsq_linear

from experiments.mapping_semantics.fit_sign import PENALTY, fit_sign_experiment
from mm_sae.analysis.signed_mapping import fit_objective, fit_signed_ridge, fixed_support_sign_fits
from mm_sae.metrics.regression import Moments


def test_signed_ridge_matches_sample_level_ridge_with_fixed_zeros():
    rng = np.random.default_rng(61)
    x = rng.normal(size=(70, 6))
    y = x @ rng.normal(size=(6, 4)) + rng.normal(size=(70, 4))
    support = rng.random((6, 4)) > 0.3
    support[:, 1] = False
    penalty = 0.01
    result = fit_signed_ridge(x.T @ x / len(x), x.T @ y / len(x), support, penalty)
    assert result.converged
    assert result.kkt_residual < 1e-12
    assert np.all(result.coefficients[~support] == 0)
    for target in range(y.shape[1]):
        selected = np.flatnonzero(support[:, target])
        if not len(selected):
            continue
        design = np.vstack([x[:, selected] / np.sqrt(len(x)), np.sqrt(penalty) * np.eye(len(selected))])
        response = np.r_[y[:, target] / np.sqrt(len(x)), np.zeros(len(selected))]
        expected = np.linalg.lstsq(design, response, rcond=None)[0]
        np.testing.assert_allclose(result.coefficients[selected, target], expected, atol=1e-12)


def test_sign_comparison_uses_same_support_penalty_and_fit_standardization():
    rng = np.random.default_rng(43)
    image = rng.normal(size=(100, 3)) * [2, 1, 3] + [4, -3, 2]
    text = image @ np.array([[-1., 0.7], [0.4, 0.1], [0.2, -0.8]]) + rng.normal(size=(100, 2))
    joint = np.column_stack([image, text])
    fit = Moments(len(joint), joint.mean(axis=0), joint.T @ joint / len(joint))
    support = np.array([[True, True], [False, True], [True, True]])
    fits = fixed_support_sign_fits(fit, 3, support)
    zi, zt = ((image - fit.mean[:3]) / fit.scale[:3], (text - fit.mean[3:]) / fit.scale[3:])
    for direction, x, y, allowed in (("image_to_text", zi, zt, support),
                                    ("text_to_image", zt, zi, support.T)):
        signed = fits["signed"][direction].coefficients
        nonnegative = fits["nonnegative"][direction].coefficients
        assert signed.shape == nonnegative.shape == allowed.shape
        assert np.all(signed[~allowed] == 0)
        assert np.all(nonnegative[~allowed] == 0)
        assert np.any(signed < 0)
        assert np.all(nonnegative >= 0)
        xx, xy, yy = x.T @ x / len(x), x.T @ y / len(x), y.T @ y / len(x)
        signed_objective = fit_objective(signed, xx, xy, yy, PENALTY)["objective_sum_over_targets"]
        nonnegative_objective = fit_objective(nonnegative, xx, xy, yy, PENALTY)["objective_sum_over_targets"]
        assert signed_objective < nonnegative_objective
        expected_objective = 0.5 * (((y - x @ signed)**2).sum() / len(x) + PENALTY * (signed**2).sum())
        np.testing.assert_allclose(signed_objective, expected_objective, atol=1e-12)
        for target in range(y.shape[1]):
            selected = np.flatnonzero(allowed[:, target])
            design = np.vstack([x[:, selected] / np.sqrt(len(x)), np.sqrt(PENALTY) * np.eye(len(selected))])
            response = np.r_[y[:, target] / np.sqrt(len(x)), np.zeros(len(selected))]
            expected = lsq_linear(design, response, bounds=(0, np.inf), tol=1e-12).x
            np.testing.assert_allclose(nonnegative[selected, target], expected, atol=1e-7)
    assert not np.allclose(fits["signed"]["text_to_image"].coefficients,
                           fits["signed"]["image_to_text"].coefficients.T)


@pytest.mark.parametrize("options", [{}, {"penalty": 0.08, "tolerance": 1e-9}])
def test_sign_experiment_reads_no_heldout_moments_and_saves_both_orientations(tmp_path, options):
    rng = np.random.default_rng(21)
    joint = rng.normal(size=(50, 5))
    joint[:, 3] = -joint[:, 0] + joint[:, 1]
    moments = tmp_path / "moments.npz"
    # Object arrays fail if accessed with allow_pickle=False. Their presence
    # checks that the fit-only stage never reads validation or test moments.
    forbidden = np.array([None], dtype=object)
    np.savez(moments, fit_n=len(joint), fit_mean=joint.mean(axis=0),
             fit_second=joint.T @ joint / len(joint), image_ids=[7, 8, 9], text_ids=[4, 6],
             tune_n=forbidden, tune_mean=forbidden, tune_second=forbidden,
             test_n=forbidden, test_mean=forbidden, test_second=forbidden)
    support_path = tmp_path / "procrustes_2.npz"
    direct = np.array([[-0.8, 0.0], [0.4, 0.1], [0.0, -0.5]])
    np.savez(support_path, image_by_text=direct)
    output = tmp_path / "fitted"
    result = fit_sign_experiment(moments, {2: support_path}, output, **options)
    penalty, tolerance = options.get("penalty", 0.01), options.get("tolerance", 1e-7)
    assert result["penalty"] == result["signature"]["penalty"] == penalty
    assert result["tolerance"] == result["signature"]["tolerance"] == tolerance
    assert result["model_class"] == "sparse linear regression"
    records = result["conditions"]
    assert len(records) == 2
    assert records[0]["allowed_support_sha256"] == records[1]["allowed_support_sha256"]
    for record in records:
        assert record["penalty"] == penalty
        assert record["tolerance"] == tolerance
        assert record["source_hashes"][str(moments.resolve())]
        assert record["allowed_support_count"] == 4
        with np.load(record["coefficients_path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["allowed_support_image_by_text"], direct != 0)
            assert saved["image_to_text"].shape == (3, 2)
            assert saved["text_to_image"].shape == (2, 3)
            for direction in ["image_to_text", "text_to_image"]:
                diagnostic = record["directions"][direction]
                assert diagnostic["actual_nonzero_count"] == np.count_nonzero(saved[direction])
                assert diagnostic["negative_count"] == np.count_nonzero(saved[direction] < 0)
                assert diagnostic["converged"]
                if record["sign_constraint"] == "signed":
                    standardized = (joint - joint.mean(axis=0)) / joint.std(axis=0)
                    x, y, allowed = ((standardized[:, :3], standardized[:, 3:], direct != 0)
                                     if direction == "image_to_text" else
                                     (standardized[:, 3:], standardized[:, :3], direct.T != 0))
                    gradient = x.T @ (x @ saved[direction] - y) / len(x) + penalty * saved[direction]
                    np.testing.assert_allclose(gradient[allowed], 0, atol=1e-12)
        assert "Procrustes" not in record["label"]
    assert json.loads((output / "manifest.json").read_text())["directions_fitted_independently"]
    np.savez(support_path, image_by_text=-direct)
    with pytest.raises(ValueError, match="changed"):
        fit_sign_experiment(moments, {2: support_path}, output, **options)


def test_signed_fit_empty_support_and_invalid_moments():
    result = fit_signed_ridge(np.eye(2), np.ones((2, 3)), np.zeros((2, 3), bool), 0.01)
    assert result.converged
    assert result.kkt_residual == 0
    assert not result.coefficients.any()
    with pytest.raises(ValueError, match="dimensions"):
        fit_signed_ridge(np.eye(2), np.ones((2, 3)), np.ones((3, 2), bool), 0.01)
    with pytest.raises(ValueError, match="symmetric"):
        fit_signed_ridge(np.array([[1., 1.], [0., 1.]]), np.ones((2, 3)), np.ones((2, 3), bool), 0.01)
    with pytest.raises(ValueError, match="positive"):
        fit_signed_ridge(np.eye(2), np.ones((2, 3)), np.ones((2, 3), bool), 0)
