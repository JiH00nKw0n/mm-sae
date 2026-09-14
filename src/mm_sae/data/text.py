"""Caption labels come from explicit text spans, never inherited image labels.

Automatic edits only delete matched spans; they do not claim grammatical or human semantic validation.
A reviewed JSONL can replace labels and edits without changing the rest of the experiment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Concept:
    id: int
    name: str
    aliases: tuple[str, ...]


def load_concepts(path: Path | None, synthetic=False) -> list[Concept]:
    value = yaml.safe_load((path or files("mm_sae").joinpath("resources/concepts.yaml")).read_text())
    concepts = [Concept(int(c["id"]), c["name"], tuple(c["aliases"])) for c in value["concepts"]]
    if len({c.id for c in concepts}) != len(concepts):
        raise ValueError("Duplicate concept IDs")
    if any(not 0 <= c.id <= 181 for c in concepts):
        raise ValueError("Concept IDs must be raw PNG indices; void 255 is not a concept")
    return [c for c in concepts if c.id in {0, 1, 123}] if synthetic else concepts


class CaptionEditor:
    def __init__(self, concepts: list[Concept], reviewed: Path | None = None):
        self.patterns = {c.id: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in sorted(c.aliases, key=len, reverse=True)) + r")(?!\w)", re.I)
                         for c in concepts if c.aliases}
        self.reviewed = {}
        if reviewed:
            for line in reviewed.read_text().splitlines():
                row = json.loads(line)
                key = int(row["caption_id"])
                if key in self.reviewed:
                    raise ValueError(f"Duplicate reviewed caption {key}")
                self.reviewed[key] = row

    def analyze(self, caption_id: int, text: str):
        if caption_id in self.reviewed:
            row = self.reviewed[caption_id]
            if row["original"] != text:
                raise ValueError(f"Reviewed caption {caption_id} does not match COCO text")
            present = sorted(set(map(int, row["concept_ids"])))
            if any(c not in self.patterns for c in present):
                raise ValueError("Reviewed caption uses an unknown concept ID")
            edits = {int(k): v for k, v in row["edits"].items()}
            if any(c not in present for c in edits) or any(v == text for v in edits.values()):
                raise ValueError("Reviewed edits must remove a present concept and change the text")
            return present, edits, "human_reviewed", [c for c in present if c not in edits]
        spans = {c: [(m.start(), m.end()) for m in regex.finditer(text)] for c, regex in self.patterns.items()}
        spans = {c: s for c, s in spans.items() if s}
        # Overlapping labels (e.g. hot dog/dog) go to the longest phrase, never both.
        keep = {c: [(a, b) for a, b in ss if not any(x <= a and b <= y and y-x > b-a
                for other, os in spans.items() if other != c for x, y in os)] for c, ss in spans.items()}
        keep = {c: s for c, s in keep.items() if s}
        edits, unavailable = {}, []
        for c, ss in keep.items():
            if any(max(a, x) < min(b, y) for a, b in ss for other, os in keep.items() if other != c for x, y in os):
                unavailable.append(c)
                continue
            new = text
            for a, b in sorted(ss, reverse=True):
                new = new[:a] + new[b:]
            new = re.sub(r"\s+", " ", new).strip()
            if self.patterns[c].search(new):
                unavailable.append(c)
                continue
            edits[c] = new
        return sorted(keep), edits, "automatic_span_deletion", unavailable
