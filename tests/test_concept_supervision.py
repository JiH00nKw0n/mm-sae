import numpy as np
import pytest
from scipy import sparse

from mm_sae.analysis.concept_supervision import (
    annotation_moments, project_concepts, standardize_concept_weights,
)


def test_weighted_annotations_equal_explicit_caption_repetition():
    x = np.array([[1., 0.], [3., 1.], [0., 2.]])
    y = np.array([[1, 0], [0, 1], [1, 1]])
    weights = np.array([2, 3, 1])
    rows = np.repeat(np.arange(3), weights)
    mean, scale = x[rows].mean(0), x[rows].std(0)
    moments = annotation_moments(sparse.csr_matrix(x), y, mean, scale, weights=weights)
    z = (x[rows]-mean)/scale
    target = y[rows]-y[rows].mean(0)
    expected = np.einsum('ij,ik->jk', z, target)/len(rows)
    np.testing.assert_allclose(moments['cross'], expected, atol=1e-14)
    np.testing.assert_allclose(moments['standardized_mean'], 0, atol=1e-14)
    np.testing.assert_allclose(moments['label_mean'], y[rows].mean(0))


def test_projection_has_train_unit_variance_without_any_inference_labels():
    rng = np.random.default_rng(51)
    x = rng.normal(size=(100, 7))
    mean, scale = x.mean(0), x.std(0)
    z = (x-mean)/scale
    covariance = np.einsum('ij,ik->jk', z, z)/len(x)
    b = rng.normal(size=(7, 3))
    b[:, 2] = 0
    normalized, _, active = standardize_concept_weights(b, covariance)
    predicted = project_concepts(x, mean, scale, normalized)
    np.testing.assert_allclose(predicted.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(predicted.std(0), [1, 1, 0], atol=1e-14)
    assert active.tolist() == [True, True, False]


def test_nonbinary_annotations_are_rejected():
    with pytest.raises(ValueError, match='binary'):
        annotation_moments(np.ones((3, 2)), np.full((3, 1), .5), np.zeros(2), np.ones(2))
