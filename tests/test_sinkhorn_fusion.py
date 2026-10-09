import numpy as np
import pytest

from experiments.mapping_semantics.run_sinkhorn_fusion import mean_cosine_vectors, unit_rows


def test_equal_weight_mean_cosine_not_mean_recall():
    rng = np.random.default_rng(42)
    x, xp, y, yp = [rng.normal(size=shape) for shape in ((5, 4), (5, 3), (12, 3), (12, 4))]
    left, right = mean_cosine_vectors(xp, y, x, yp)
    actual = unit_rows(left) @ unit_rows(right).T
    expected = .5 * (unit_rows(xp) @ unit_rows(y).T + unit_rows(x) @ unit_rows(yp).T)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_zero_vector_does_not_silently_change_score_weights():
    with pytest.raises(ValueError, match='nonzero'):
        mean_cosine_vectors(np.zeros((2, 3)), np.ones((3, 3)), np.ones((2, 4)), np.ones((3, 4)))


def test_bidirectional_recall_matches_explicit_mean_scores():
    from mm_sae.analysis.mapping_evaluation import paired_retrieval

    rng = np.random.default_rng(17)
    x, xp, y, yp = [rng.normal(size=shape) for shape in ((4, 7), (4, 5), (12, 5), (12, 7))]
    parents = np.repeat(np.arange(4), 3)
    left, right = mean_cosine_vectors(xp, y, x, yp)
    result = paired_retrieval(left, right, parents, ks=(1, 5, 10))
    scores = .5 * (unit_rows(xp) @ unit_rows(y).T + unit_rows(x) @ unit_rows(yp).T)
    image_ranks = []
    for i in range(4):
        best = scores[i, parents == i].max()
        image_ranks.append(1 + np.sum(scores[i] > best))
    text_ranks = [1 + np.sum(scores[:, j] > scores[parents[j], j]) for j in range(12)]
    np.testing.assert_array_equal(result['image_to_text']['ranks'], image_ranks)
    np.testing.assert_array_equal(result['text_to_image']['ranks'], text_ranks)
