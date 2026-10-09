"""Separate annotated calibration populations from unannotated mapping-fit data."""
import json
from pathlib import Path


def load_population(parent: Path, annotation_population: str | None = None):
    population = json.loads((parent / "population.json").read_text())
    if annotation_population is None:
        return population
    annotated = json.loads(Path(annotation_population).read_text())
    if population["test_image_ids"] != annotated["test_image_ids"]:
        raise ValueError("Annotation and mapping evaluation image IDs differ")
    ids = annotated["tune_image_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Annotation calibration requires unique, nonempty image IDs")
    if set(ids) & (set(population["fit_image_ids"]) | set(population["test_image_ids"])):
        raise ValueError("Annotation calibration overlaps mapping fit or test images")
    return {**population, "tune_image_ids": ids, "tune_images": len(ids),
            "annotation_population": annotation_population,
            "mapping_tune_images": population["tune_images"]}
