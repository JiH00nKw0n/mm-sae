import pytest
from pydantic import ValidationError

from mm_sae.data.caption_annotations import CaptionAnnotation, validate_and_locate


def source():
    return {
        "split": "illustrative",
        "caption_id": 1,
        "image_id": 2,
        "original": "A hot dog beside a dog.",
        "objects": [{"concept_id": 17, "name": "dog"}, {"concept_id": 57, "name": "hot dog"}],
    }


def test_repeated_expression_locates_animal_without_removing_food():
    parsed = CaptionAnnotation.model_validate(
        {
            "objects": [
                {"concept_id": 17, "spans": [{"text": "dog", "occurrence": 2}]},
                {"concept_id": 57, "spans": [{"text": "hot dog", "occurrence": 1}]},
            ]
        }
    )
    result = validate_and_locate(source(), parsed)
    animal = result["objects"][0]["spans"][0]
    text = result["original"]
    assert text[: animal["start"]] + text[animal["end"] :] == "A hot dog beside a ."


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


def test_valid_json_is_not_enough_to_accept_wrong_source_alignment():
    parsed = CaptionAnnotation.model_validate(
        {
            "objects": [
                {"concept_id": 17, "spans": [{"text": "puppy", "occurrence": 1}]},
                {"concept_id": 57, "spans": []},
            ]
        }
    )
    with pytest.raises(ValueError, match="does not exist"):
        validate_and_locate(source(), parsed)
    parsed.objects[0].concept_id = 999
    with pytest.raises(ValueError, match="supplied categories"):
        validate_and_locate(source(), parsed)
