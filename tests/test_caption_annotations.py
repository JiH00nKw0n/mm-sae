import pytest
from pydantic import ValidationError

from mm_sae.data.caption_annotations import CaptionAnnotation, validate_and_locate


def source():
    return {
        "split": "illustrative",
        "caption_id": 1,
        "image_id": 2,
        "original": "A dog sits beside a dog.",
        "objects": [{"concept_id": 17, "name": "dog"}, {"concept_id": 57, "name": "hot dog"}],
    }


def test_every_exact_match_is_located_once_in_source_order():
    parsed = CaptionAnnotation.model_validate(
        {
            "objects": [
                {"concept_id": 17, "spans": ["dog", "dog"]},
                {"concept_id": 57, "spans": []},
            ]
        }
    )
    result = validate_and_locate(source(), parsed)
    assert result["objects"][0]["spans"] == [
        {"text": "dog", "start": 2, "end": 5},
        {"text": "dog", "start": 20, "end": 23},
    ]


def test_all_matches_intentionally_include_different_meanings():
    row = source()
    row["original"] = "People train beside a train."
    row["objects"] = [{"concept_id": 6, "name": "train"}]
    parsed = CaptionAnnotation.model_validate({"objects": [{"concept_id": 6, "spans": ["train"]}]})
    result = validate_and_locate(row, parsed)
    assert [(s["start"], s["end"]) for s in result["objects"][0]["spans"]] == [(7, 12), (22, 27)]


def test_exact_matches_keep_case_and_word_boundaries():
    row = source()
    row["original"] = "Dog dog dogs doghouse dog."
    parsed = CaptionAnnotation.model_validate(
        {"objects": [{"concept_id": 17, "spans": ["dog"]}, {"concept_id": 57, "spans": []}]}
    )
    result = validate_and_locate(row, parsed)
    assert [(s["start"], s["end"]) for s in result["objects"][0]["spans"]] == [(4, 7), (22, 25)]


def test_unknown_and_absent_are_distinct_and_fields_are_required():
    parsed = CaptionAnnotation.model_validate(
        {
            "objects": [
                {"concept_id": 17, "spans": []},
                {"concept_id": 57, "spans": None},
            ]
        }
    )
    result = validate_and_locate(source(), parsed)
    assert result["objects"][0]["spans"] == []
    assert result["objects"][1]["spans"] is None
    with pytest.raises(ValidationError):
        CaptionAnnotation.model_validate({"objects": [{"concept_id": 17}]})
    with pytest.raises(ValidationError):
        CaptionAnnotation.model_validate({"objects": [], "extra": True})
    with pytest.raises(ValidationError):
        CaptionAnnotation.model_validate(
            {"objects": [{"concept_id": 17, "spans": [{"text": "dog", "occurrence": 1}]}]}
        )


def test_valid_json_is_not_enough_to_accept_wrong_source_alignment():
    parsed = CaptionAnnotation.model_validate(
        {
            "objects": [
                {"concept_id": 17, "spans": ["puppy"]},
                {"concept_id": 57, "spans": []},
            ]
        }
    )
    with pytest.raises(ValueError, match="does not exist"):
        validate_and_locate(source(), parsed)
    parsed.objects[0].concept_id = 999
    with pytest.raises(ValueError, match="supplied categories"):
        validate_and_locate(source(), parsed)
