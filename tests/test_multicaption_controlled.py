import importlib.util
from pathlib import Path

import numpy as np
from scipy import sparse


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_multicaption_controlled.py"
SPEC = importlib.util.spec_from_file_location("multicaption", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_caption_controls_are_deterministic_disjoint_and_duplicate_invariant():
    groups = [[i * 5 + j for j in range(5)] for i in range(8)]
    first = MODULE.caption_sets(groups, 2)
    second = MODULE.caption_sets(groups, 2)
    for key in first:
        assert np.array_equal(first[key], second[key])
    same = first[(0, 5, "same_image")]
    duplicate = first[(0, 5, "duplicate")]
    cross = first[(0, 5, "cross_image")]
    assert np.all(duplicate == duplicate[:, :1])
    assert np.all(same[:, 0] == cross[:, 0])
    for row in range(len(groups)):
        assert all(value // 5 != row for value in cross[row, 1:])


def test_pre_mean_normalizes_and_post_mean_averages_sparse_rows():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    indices = np.asarray([[0, 1], [2, 3]])
    expected = np.asarray([[2**-0.5, 2**-0.5], [2**-0.5, 2**-0.5]], dtype=np.float32)
    assert np.allclose(MODULE.mean_embeddings(embeddings, indices), expected)
    codes = sparse.csr_matrix(embeddings)
    assert np.allclose(MODULE.mean_sparse_rows(codes, indices).toarray(), [[0.5, 0.5], [1.0, 1.0]])
