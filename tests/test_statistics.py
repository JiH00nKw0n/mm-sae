import numpy as np
from scipy import sparse
from mm_sae.metrics.statistics import correlation, paired_auroc, bin_summary
from mm_sae.metrics.matching import match
from experiments.rq1.correspondence import assess


def test_correlation_does_not_mutate_unsorted_float32_sparse_inputs():
    x = sparse.csr_matrix((np.array([1, 3, 4, 2, 5, 6], dtype=np.float32),
                           np.array([1, 0, 1, 0, 1, 0]), np.array([0, 2, 4, 6])), shape=(3, 2))
    y = x.copy()
    before_x, before_y = x.toarray(), y.toarray()
    result = correlation(x, y)
    np.testing.assert_array_equal(x.toarray(), before_x)
    np.testing.assert_array_equal(y.toarray(), before_y)
    np.testing.assert_allclose(result["C"], np.corrcoef(before_x.T), atol=1e-12)


def test_signed_pearson_and_constant_status():
    x = np.array([[0, 7], [1, 7], [2, 7], [3, 7]])
    y = np.array([[3, 0], [2, 1], [1, 2], [0, 3]])
    p = correlation(x, y)
    np.testing.assert_allclose(p["C"][0], [-1, 1])
    assert not p["valid_image"][1]
    assert p["alive_image"][1]  # A constant positive feature is alive, but its correlation is undefined.


def test_exact_auc_including_sparse_zero_ties():
    p = np.array([[0, 0.1, 1], [0, 3, 1], [2, 0, 0]], dtype=float)
    q = np.array([[0, 2, 1], [1, 0, 1], [2, 0, 0]], dtype=float)
    expected = np.array(
        [
            np.mean((p[:, f, None] > q[:, f]).astype(float) + 0.5 * (p[:, f, None] == q[:, f]))
            for f in range(3)
        ]
    )
    np.testing.assert_allclose(paired_auroc(sparse.csr_matrix(p), sparse.csr_matrix(q)), expected)


def test_full_signed_bins_include_boundaries_and_empty_bins():
    rows = [
        {"label_correlation": v, "coactivation_correlation": v / 2, "valid": True}
        for v in [-1, -0.8, 0, 0.2, 1]
    ]
    result = bin_summary(rows, 0.2)
    assert len(result) == 10 and sum(r["n_pairs"] for r in result) == 5
    assert (
        result[0]["n_pairs"]
        == result[1]["n_pairs"]
        == result[5]["n_pairs"]
        == result[6]["n_pairs"]
        == result[9]["n_pairs"]
        == 1
    )
    assert result[2]["mean"] is None


def test_greedy_reuses_text_hungarian_is_one_to_one_and_signed():
    c = np.array([[0.9, 0.8], [0.85, -0.99]])
    result = match(c, *[np.ones(2, bool) for _ in range(4)])
    assert result["greedy"].tolist() == [0, 0]
    assert result["hungarian"].tolist() == [1, 0]


def test_undefined_correlations_cannot_displace_valid_negative_edges():
    c = np.array([[-0.4, 0.0], [0.0, 0.0]])
    result = match(c, np.ones(2, bool), np.ones(2, bool), np.array([True, False]), np.array([True, False]))
    for method in ["greedy", "hungarian"]:
        assert result[method].tolist() == [0, -1]
        panel = {"C": c, "valid_image": np.array([True, False]), "valid_text": np.array([True, False])}
        reps = {"0": {"image": 0, "text": 0}, "1": {"image": 1, "text": 1}}
        assert [r["status"] for r in assess(panel, result[method], reps)] == ["same", "undefined_correlation"]


def test_undefined_and_multi_concept_matches_never_count_as_recovery():
    panel = {"C": np.eye(2), "valid_image": np.array([True, True]), "valid_text": np.array([False, True])}
    reps = {"0": {"image": 0, "text": 0}, "1": {"image": 1, "text": 1}, "123": {"image": 1, "text": None}}
    assert [r["status"] for r in assess(panel, np.array([0, 1]), reps)] == [
        "undefined_correlation",
        "ambiguous_or_unlabeled",
    ]
