import pytest

from experiments.rq1.plotting import CONDITIONS, paired_score_summary


def test_score_comparison_keeps_one_cohort_and_retains_undefined_counts():
    rows = []
    for anchor in [0, 1]:
        for condition in CONDITIONS:
            for method in ["greedy", "hungarian"]:
                rows.append(
                    {
                        "anchor": anchor,
                        "remove": 2,
                        "repeat": 0,
                        "condition": condition,
                        "method": method,
                        "same_concept_score": 0.5 if anchor == 0 else 0.9,
                        "wrong_score": None if anchor == 1 and condition == "cooccurrence_removal" else 0.2,
                    }
                )
    result = paired_score_summary(rows)
    assert result["total_case_repetitions"] == 2
    assert result["complete_case_repetitions"] == 1
    assert list(result["means"]["same_concept_score"].values()) == pytest.approx([0.5] * 3)
    assert result["undefined_counts"]["cooccurrence_removal"]["wrong_score"] == 1
    assert result["excluded"][0]["anchor"] == 1


def test_score_comparison_reports_no_complete_units_without_filling_zero():
    result = paired_score_summary(
        [
            {
                "anchor": 0,
                "remove": 1,
                "repeat": 0,
                "condition": "original",
                "same_concept_score": 0.5,
                "wrong_score": 0.6,
            }
        ]
    )
    assert result["complete_case_repetitions"] == 0
    assert all(value is None for value in result["means"]["wrong_score"].values())
