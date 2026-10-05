import pytest

from scripts.analyze_bidirectional_rq1 import build_directions, ranking_summary, verify_pooling


def example_rows():
    # An asymmetric matrix and unequal diagonals distinguish row and column ranking.
    diagonal = [.2, .8, .6]
    matrix = [[.2, .5, .2], [.1, .8, .7], [.4, .3, .6]]
    return [{
        "image_category": str(a), "other_category": str(b),
        "image_name": str(a), "other_name": str(b),
        "category_types": "_".join("object" if c < 2 else "background" for c in [a, b]),
        "original_annotation_correlation": ".3" if {a, b} == {0, 1} else "-.1",
        "original_same": str(diagonal[a]), "original_other": str(matrix[a][b]),
        "independent_same": str(diagonal[a]-.1), "independent_other": str(matrix[a][b]-.05),
    } for a in range(3) for b in range(3) if a != b]


def test_text_anchor_uses_its_own_diagonal_and_preserves_actual_pair_score():
    directions = build_directions(example_rows())
    reverse = next(r for r in directions["text"] if r.anchor_id == 1 and r.candidate_id == 0)
    assert reverse.original_other == .5
    assert reverse.original_same == .8
    assert reverse.controlled_same == pytest.approx(.7)
    assert reverse.controlled_other == pytest.approx(.45)
    assert ranking_summary(directions["image"])["categories_with_higher_alternative"] == 1
    assert ranking_summary(directions["text"])["categories_with_higher_alternative"] == 2
    assert ranking_summary(directions["image"])["ties"] == 1
    assert ranking_summary(directions["text"])["ties"] == 0


def test_pooled_distributions_equal_and_cross_type_roles_swap():
    directions = build_directions(example_rows())
    verify_pooling(directions)
    image = next(r for r in directions["image"] if r.anchor_id == 0 and r.candidate_id == 2)
    text = next(r for r in directions["text"] if r.anchor_id == 2 and r.candidate_id == 0)
    assert (image.anchor_type, image.candidate_type) == ("object", "background")
    assert (text.anchor_type, text.candidate_type) == ("background", "object")
    assert image.original_other == text.original_other


def test_missing_reverse_pair_is_rejected():
    with pytest.raises(ValueError, match="all off-diagonal"):
        build_directions(example_rows()[:-1])
