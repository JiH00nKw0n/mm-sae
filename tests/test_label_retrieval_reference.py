import numpy as np

from mm_sae.analysis.label_retrieval_reference import label_only_retrieval


def test_uniform_tie_expectation_counts_captions_and_images_separately():
    labels = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = label_only_retrieval(labels, np.array([0, 0, 1, 2, 3]), ks=(1, 2, 10))
    image = result['retrieval']['image_to_text']['recall']
    text = result['retrieval']['text_to_image']['recall']
    np.testing.assert_allclose([image[1], image[2], image[10]], [.75, 11/12, 1])
    np.testing.assert_allclose([text[1], text[2], text[10]], [.7, 1, 1])
    assert result['distinct_label_sets'] == 3
    assert result['images_with_unique_label_set'] == 2


def test_empty_vectors_remain_failed_queries_and_are_not_silently_removed():
    result = label_only_retrieval(np.array([[0, 0], [1, 0]]), np.array([0, 1]))
    assert result['empty_label_images'] == 1
    for direction in result['retrieval'].values():
        assert direction['recall'] == {1: .5, 5: .5, 10: .5}
