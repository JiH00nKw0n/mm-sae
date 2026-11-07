"""Native visual and region-level annotations. Each held-out group is an image."""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .records import Record, fold, json_array


def coco(files, seed, *, stuff=False):
    # The official val2017 images are the held-out test set. Train/val split train2017 only.
    for partition in ("train", "test"):
        data = json.loads(files[partition + "_instances"].read_text())
        categories = {item["id"]: item["name"] for item in data["categories"]}
        labels = defaultdict(set)
        for obj in data["annotations"]:
            labels[obj["image_id"]].add(categories[obj["category_id"]])
        if stuff:
            names = dict(
                line.rstrip().split(": ", 1)
                for line in files["stuff_names"].read_text().splitlines()
                if line.strip()
            )
        for row in data["images"]:
            key = f"coco/{row['id']}"
            split = partition if partition == "test" else ("val" if fold(key, seed) == "val" else "train")
            terms = labels[row["id"]]
            if stuff:
                mask = files[partition + "_masks"] / Path(row["file_name"]).with_suffix(".png")
                with Image.open(mask) as im:
                    indices = np.unique(np.asarray(im))
                # Official 164k PNG IDs are zero-based; labels.txt IDs are one-based.
                terms = {names[str(int(i) + 1)] for i in indices if i != 255}
            yield Record(
                key,
                key,
                "image",
                str(files[partition + "_images"] / row["file_name"]),
                split,
                tuple(sorted(terms)),
            )


def visual_genome(files, seed):
    meta = {int(x["image_id"]): x.get("coco_id") for x in json_array(files["metadata"])}
    test_ids = {x["id"] for x in json.loads(files["coco_test_instances"].read_text())["images"]}
    for image in json_array(files["graphs"]):
        idx = image["image_id"]
        coco_id = meta[idx]
        key = f"coco/{coco_id}" if coco_id is not None else f"vg/{idx}"
        split = (
            ("test" if coco_id in test_ids else ("val" if fold(key, seed) == "val" else "train"))
            if coco_id is not None
            else fold(key, seed)
        )
        for r in image["regions"]:
            terms = set()
            for obj in r.get("objects", []):
                names = obj.get("names") or ([obj["name"]] if obj.get("name") else [])
                terms.update("object/" + x.lower() for x in names)
                terms.update("attribute/" + x.lower() for x in obj.get("attributes", []))
            for att in r.get("attributes", []):
                values = att.get("attributes", []) if isinstance(att, dict) else [att]
                if isinstance(att, dict) and "attribute" in att:
                    values = [att["attribute"]]
                terms.update("attribute/" + x.lower() for x in values)
            terms.update(
                "relation/" + x["predicate"].lower() for x in r.get("relationships", []) if x.get("predicate")
            )
            yield Record(f"vg/{idx}/{r['region_id']}", key, "text", r["phrase"], split, tuple(sorted(terms)))


def numbered(path):
    return {
        int(k): v
        for k, v in (line.strip().split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())
    }


def cub(files, seed):
    names = numbered(files["attribute_names"])
    test = numbered(files["partition"])
    labels = defaultdict(list)
    with files["attributes"].open() as stream:
        for line in stream:
            image, attr, present, *_ = line.split()
            if int(present):
                labels[int(image)].append(names[int(attr)])
    for idx, path in numbered(files["images_list"]).items():
        key = f"cub/{idx}"
        split = "test" if test[idx] == "0" else ("val" if fold(key, seed) == "val" else "train")
        yield Record(key, key, "image", str(files["images"] / path), split, tuple(labels[idx]))
