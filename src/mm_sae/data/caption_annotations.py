"""Typed caption annotations and exact validation against source text and image categories."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict


class ObjectSpans(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    concept_id: int
    spans: list[str] | None


class CaptionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    objects: list[ObjectSpans]


def response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": CaptionAnnotation.__name__,
            "strict": True,
            "schema": CaptionAnnotation.model_json_schema(),
        },
    }


def validate_and_locate(source: dict, annotation: CaptionAnnotation) -> dict:
    expected = [obj["concept_id"] for obj in source["objects"]]
    actual = [obj.concept_id for obj in annotation.objects]
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("Returned categories must equal the supplied categories in input order")
    text = source["original"]
    objects = []
    for obj in annotation.objects:
        located = None if obj.spans is None else []
        intervals: set[tuple[int, int]] = set()
        for expression in obj.spans or []:
            if not expression.strip():
                raise ValueError("Expression must be nonempty")
            matches = list(re.finditer(r"(?<!\w)" + re.escape(expression) + r"(?!\w)", text))
            if not matches:
                raise ValueError("The returned expression does not exist in the source caption")
            # All exact matches are intentional, even when their meanings differ.
            intervals.update((match.start(), match.end()) for match in matches)
        for start, end in sorted(intervals):
            assert located is not None
            located.append({"text": text[start:end], "start": start, "end": end})
        objects.append({"concept_id": obj.concept_id, "spans": located})
    return {
        "caption_id": source["caption_id"],
        "image_id": source["image_id"],
        "split": source["split"],
        "original": text,
        "image_concept_ids": expected,
        "objects": objects,
    }
