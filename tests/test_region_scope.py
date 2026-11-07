import json

from mm_sae.config import RegionSelection
from mm_sae.data.concepts import visual_genome


def test_visual_genome_keeps_only_selected_annotations_and_coco_images(tmp_path):
    ids = [12, 23, 99, None]
    metadata = [dict(image_id=i, coco_id=value) for i, value in enumerate(ids)]
    regions = [
        dict(region_id=0, phrase="a red car", objects=[dict(names=["car"], attributes=["red"])]),
        dict(region_id=1, phrase="on", relationships=[dict(predicate="on")]),
        dict(region_id=2, phrase="surface"),
    ]
    inputs = {
        "metadata": metadata,
        "graphs": [dict(image_id=i, regions=regions) for i in range(len(ids))],
        "coco_train_instances": {"images": [{"id": 12}]},
        "coco_test_instances": {"images": [{"id": 23}]},
    }
    paths = {}
    for key, value in inputs.items():
        paths[key] = tmp_path / (key + ".json")
        paths[key].write_text(json.dumps(value))
    selection = RegionSelection(
        images="coco_trainval", label_types=("object", "relation"), annotated_only=True
    )
    rows = list(visual_genome(paths, 0, selection))
    assert len(rows) == 4
    assert {r.group for r in rows} == {"coco/12", "coco/23"}
    assert {tuple(r.labels) for r in rows} == {("object/car",), ("relation/on",)}
    assert {r.group for r in rows if r.split == "test"} == {"coco/23"}
