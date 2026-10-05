import pytest

from scripts.compare_natural_independence import bin_all_category_comparisons, compare_all_categories


def example_rows():
    rows = []
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            positive = (a + b) == 1
            rows.append({
                "image_category": a, "other_category": b, "image_name": str(a), "other_name": str(b),
                "category_types": "object_object", "status": "included" if positive else "missing_group",
                "original_annotation_correlation": .2 if positive else -.1,
                "independent_annotation_correlation": 0.,
                "original_same": .5, "original_other": .6 if (a, b) in {(0, 1), (2, 0)} else .1,
                "independent_same": .5 if positive else None, "independent_other": .1 if positive else None,
            })
    return rows


def test_all_rivals_remain_and_nonpositive_pairs_keep_original_scores():
    comparisons, anchors, pair_summary, category_summary = compare_all_categories(example_rows(), [0, 1, 2])
    assert pair_summary[0]["paired_ordered_pairs"] == 6
    assert pair_summary[0]["original_exceeds"] == 2
    assert pair_summary[0]["cooccurrence_removed_exceeds"] == 1
    assert category_summary[0]["original_exceeds"] == 2
    assert category_summary[0]["cooccurrence_removed_exceeds"] == 1
    assert category_summary[0]["cooccurrence_removed_percent"] == pytest.approx(100 / 3)
    assert all(r["other_categories"] == 2 for r in anchors)
    for row in comparisons:
        if row["original_annotation_correlation"] < 0:
            assert row["independent_other"] == row["original_other"]
            assert row["independent_same"] == row["original_same"]
            assert row["margin_change"] == 0


def test_bins_stay_at_original_annotation_and_distinguish_pairs_from_categories():
    comparisons, *_ = compare_all_categories(example_rows(), [0, 1, 2])
    bins = bin_all_category_comparisons(comparisons, [0, 1, 2])
    positive = next(r for r in bins if r["left"] == .2)
    negative = next(r for r in bins if r["left"] == -.2)
    assert positive["paired_ordered_pairs"] == 2
    assert positive["categories_with_computable_rivals_in_bin"] == 2
    assert positive["original_percent_of_eligible_categories"] == 50
    assert positive["cooccurrence_removed_percent_of_eligible_categories"] == 0
    assert negative["paired_ordered_pairs"] == 4
    assert negative["categories_with_computable_rivals_in_bin"] == 3
    assert negative["original_percent"] == negative["cooccurrence_removed_percent"] == 25
    assert sum(r["paired_ordered_pairs"] for r in bins) == 6
