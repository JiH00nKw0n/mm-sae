import numpy as np
from scipy import sparse
from mm_sae.metrics.statistics import correlation, paired_auroc, bin_summary
from mm_sae.metrics.matching import match
from experiments.rq1.correspondence import assess


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
    result = match(c, np.ones(2, bool), np.ones(2, bool))
    assert result["greedy"].tolist() == [0, 0]
    assert result["hungarian"].tolist() == [1, 0]


def test_undefined_and_multi_concept_matches_never_count_as_recovery():
    panel = {"C": np.eye(2), "valid_image": np.array([True, True]), "valid_text": np.array([False, True])}
    reps = {"0": {"image": 0, "text": 0}, "1": {"image": 1, "text": 1}, "123": {"image": 1, "text": None}}
    assert [r["status"] for r in assess(panel, np.array([0, 1]), reps)] == [
        "undefined_correlation",
        "ambiguous_or_unlabeled",
    ]
