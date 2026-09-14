import numpy as np
from scipy import sparse
from experiments.rq1.interventions import deletion_budget, choose_images
from mm_sae.metrics.sparse_ops import replace_rows
from mm_sae.metrics.statistics import correlation
from mm_sae.data.text import CaptionEditor, Concept


def test_attenuation_and_control_keep_exact_deletion_budgets():
    a = np.r_[np.ones(50, bool), np.zeros(50, bool)]
    b = np.r_[np.ones(40, bool), np.zeros(10, bool), np.ones(10, bool), np.zeros(40, bool)]
    plan = deletion_budget(a, b)
    assert plan["n_remove"] == 30 and np.isclose(plan["q"], 0.75)
    caps, mentions = np.full(100, 5), np.full(100, 2)
    token_counts = 2 + np.arange(100) % 3
    conditional, random, _ = choose_images(
        a, b, np.ones(100, bool), np.ones(100), caps, mentions, 30, 5, np.random.default_rng(0), token_counts
    )
    changed = b.copy()
    changed[conditional] = False
    assert np.isclose(correlation(a[:, None], changed[:, None])["C"][0, 0], 0)
    assert len(random) == len(conditional) == 30
    assert mentions[random].sum() == mentions[conditional].sum()
    assert token_counts[random].sum() == token_counts[conditional].sum()
    assert b[random].all()


def test_no_b_without_a_is_explicitly_unassessable():
    a = np.array([1, 1, 0, 0], bool)
    assert deletion_budget(a, a)["status"] == "no_target_without_anchor"


def test_sparse_replacement_does_not_perturb_unedited_rows():
    original = sparse.csr_matrix(np.array([[1.0, 0], [0, 2.0], [0.3, 0.1]], np.float32))
    new = replace_rows(original, [1], sparse.csr_matrix([[9.0, 1.0]]))
    expected = np.asarray([[1.0, 0], [9.0, 1.0], [np.float32(0.3), np.float32(0.1)]])
    np.testing.assert_array_equal(new.toarray(), expected)


def test_caption_labels_are_not_inherited_from_images_and_spans_do_not_overlap():
    editor = CaptionEditor(
        [
            Concept(0, "person", ("person",)),
            Concept(17, "dog", ("dog",)),
            Concept(57, "hot dog", ("hot dog",)),
        ]
    )
    text = "A person eats a hot dog."
    present, spans, _, _ = editor.analyze(1, text)
    assert present == [0, 57] and 17 not in spans
    assert [text[a:b] for a, b in spans[57]] == ["hot dog"]
    assert [text[a:b] for a, b in spans[0]] == ["person"]
