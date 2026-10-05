import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear

from mm_sae.analysis.mapping_evaluation import (
    fit_nonnegative_ridge,
    mapping_retrieval,
    mapping_structure,
    paired_retrieval,
    prediction_r2,
)


@pytest.mark.parametrize("sparse_support_limit", [0, 64])
def test_nonnegative_ridge_matches_sample_level_constrained_solution(sparse_support_limit):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(80, 7))
    y = x @ rng.normal(size=(7, 4)) + rng.normal(size=(80, 4))
    support = rng.random((7, 4)) > 0.35
    support[:, 2] = False
    penalty = 0.08
    result = fit_nonnegative_ridge(x.T @ x / len(x), x.T @ y / len(x), support, penalty,
                                  tolerance=1e-9, sparse_support_limit=sparse_support_limit)
    assert result.converged
    assert result.kkt_residual < 1e-8
    assert np.all(result.coefficients >= 0)
    assert np.all(result.coefficients[~support] == 0)
    for target in range(y.shape[1]):
        selected = np.flatnonzero(support[:, target])
        if not len(selected):
            continue
        design = np.vstack([x[:, selected] / np.sqrt(len(x)), np.sqrt(penalty) * np.eye(len(selected))])
        response = np.r_[y[:, target] / np.sqrt(len(x)), np.zeros(len(selected))]
        expected = lsq_linear(design, response, bounds=(0, np.inf), tol=1e-12).x
        np.testing.assert_allclose(result.coefficients[selected, target], expected, atol=1e-7)


def test_nonnegative_ridge_handles_singular_unregularized_moments():
    xx = np.array([[1., 1.], [1., 1.]])
    xy = np.array([[2.], [2.]])
    result = fit_nonnegative_ridge(xx, xy, np.ones((2, 1), bool), 0)
    assert result.converged
    np.testing.assert_allclose(xx @ result.coefficients, xy)


def test_nonnegative_ridge_reports_failure_to_converge():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(30, 8))
    xx = x.T @ x / len(x)
    result = fit_nonnegative_ridge(xx, np.ones((8, 2)), np.ones((8, 2), bool), 0.001,
                                  max_iter=1, tolerance=1e-12, sparse_support_limit=0)
    assert not result.converged
    assert result.iterations == 1


def test_r2_retains_heldout_mean_shift_and_unmatched_targets():
    rng = np.random.default_rng(22)
    train_x, train_y = rng.normal(size=(50, 3)), rng.normal(size=(50, 4))
    x, y = rng.normal(size=(70, 3)) + 1, rng.normal(size=(70, 4)) + 2
    coefficients = rng.random((3, 4))
    coefficients[:, 1] = 0
    zx = (x - train_x.mean(0)) / train_x.std(0)
    zy = (y - train_y.mean(0)) / train_y.std(0)
    result = prediction_r2(coefficients, zx.T @ zx / len(x), zx.T @ zy / len(x),
                           zy.T @ zy / len(x), held_mean_y=zy.mean(0))
    expected = 1 - ((zy - zx @ coefficients)**2).sum(0) / ((zy - zy.mean(0))**2).sum(0)
    np.testing.assert_allclose(result, expected)
    assert result[1] < 0


def test_r2_constant_heldout_target_is_explicitly_undefined():
    result = prediction_r2(np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)),
                           np.array([4.]), held_mean_y=np.array([2.]))
    assert np.isnan(result[0])


def test_retrieval_uses_every_positive_caption_and_fixed_denominators():
    images = np.array([[1., 0.], [0., 1.], [0., 0.]])
    texts = np.array([[0., 1.], [1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    parents = np.array([0, 0, 1, 2, 2])
    result = paired_retrieval(images, texts, parents, ks=(1, 2, 10), chunk_size=1)
    image_to_text, text_to_image = result["image_to_text"], result["text_to_image"]
    assert image_to_text["query_count"] == 3
    assert text_to_image["query_count"] == 5
    # Image zero succeeds using its second caption, while equal-score captions
    # break ties by index and image one therefore needs rank two.
    assert image_to_text["hits"][1].tolist() == [True, False, False]
    assert image_to_text["hits"][2].tolist() == [True, True, False]
    assert image_to_text["recall"][10] == 2 / 3
    assert text_to_image["hits"][1].tolist() == [False, True, True, False, False]
    assert text_to_image["zero_norm_query_fraction"] == 2 / 5
    assert image_to_text["zero_norm_query_fraction"] == 1 / 3
    other = paired_retrieval(images, texts, parents, ks=(1, 2, 10), chunk_size=3)
    for direction in result:
        np.testing.assert_array_equal(result[direction]["ranks"], other[direction]["ranks"])


def test_retrieval_supports_external_parent_ids_and_signed_representations():
    images = np.array([[1., -1.], [-1., 1.]])
    result = paired_retrieval(images, images[::-1], np.array([58, 91]), image_ids=np.array([91, 58]))
    assert result["image_to_text"]["recall"][1] == 1
    assert result["text_to_image"]["recall"][1] == 1


def test_mapping_projection_normalizes_each_direction_separately():
    images = np.array([[1., 2.], [4., 1.], [0., 3.]])
    texts = np.array([[1., 3., 1.], [3., 1., 4.], [0., 2., 0.]])
    mapping = np.array([[1., 2., 0.], [0., 2., 3.]])
    parents = np.arange(3)
    result = mapping_retrieval(sparse.csr_matrix(images), texts, mapping, parents)
    expected_forward = paired_retrieval(images, texts @ (mapping / mapping.sum(1)[:, None]).T, parents)
    expected_reverse = paired_retrieval(images @ (mapping / mapping.sum(0)[None, :]), texts, parents)
    for name, expected in [("text_projected_to_image", expected_forward),
                           ("image_projected_to_text", expected_reverse)]:
        for direction in expected:
            np.testing.assert_array_equal(result[name][direction]["ranks"], expected[direction]["ranks"])


def test_empty_mapping_retains_all_retrieval_queries_as_misses():
    result = mapping_retrieval(np.eye(3), np.eye(3), np.zeros((3, 3)), np.arange(3))
    for space in result.values():
        for direction in space.values():
            assert direction["query_count"] == 3
            assert direction["recall"][10] == 0


def test_mapping_structure_uses_explicit_support_and_weighted_effective_degree():
    mapping = np.array([[1., 3., 0., 0.], [0., 2., 0., 0.], [0., 0., 5., 0.], [0., 0., 0., 1e-15]])
    result = mapping_structure(mapping, threshold=1e-10)
    assert result["edge_count"] == 4
    assert result["density"] == 0.25
    assert result["image_coverage"] == result["text_coverage"] == 0.75
    np.testing.assert_array_equal(result["image_degree"], [2, 1, 1, 0])
    np.testing.assert_allclose(result["image_effective_degree"], [1.6, 1, 1, 0])
    assert result["group_count"] == 2
    assert sorted((group["image_features"], group["text_features"]) for group in result["groups"]) == [(1, 1), (2, 2)]
    assert result["image_group"][-1] == -1
    assert mapping_structure(mapping)["edge_count"] == 5


@pytest.mark.parametrize("mapping", [np.zeros((2, 3)), np.empty((0, 3)), np.empty((3, 0))])
def test_mapping_structure_accepts_no_edges(mapping):
    result = mapping_structure(mapping)
    assert result["group_count"] == result["edge_count"] == 0
    assert result["image_coverage"] == result["text_coverage"] == 0


def test_retrieval_rejects_unknown_caption_parent():
    with pytest.raises(ValueError, match="outside"):
        paired_retrieval(np.eye(2), np.eye(2), np.array([0, 2]))
