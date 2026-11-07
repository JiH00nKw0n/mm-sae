"""Read benchmark v1.0 pair labels in the same vocabulary as original probes."""

import csv
import json
from .concepts import numbered
from .records import Record


def edited_pairs(kind, files):
    if kind == "syncoco":
        cats = json.loads(files["instances"].read_text())["categories"]
        names = [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
        fields = ("orig_labels", "syn_labels")
    else:
        names = [name for _, name in sorted(numbered(files["attribute_names"]).items())]
        fields = ("orig_attr_idx", "syn_attr_idx")
    with files["metadata"].open() as stream:
        for row in csv.DictReader(stream):
            pair = f"{kind}/{row['pair_id']}"
            before, after = [{names[i] for i in json.loads(row[k])} for k in fields]
            targets = (
                ({row["removed_class"]}, set())
                if kind == "syncoco"
                else ({row["old_attr"]}, {row["new_attr"]})
            )
            if (before - after, after - before) != targets:
                raise ValueError(f"Pair metadata and vocabulary disagree for {pair}")
            edit = {
                "removed": sorted(before - after),
                "added": sorted(after - before),
                "retained": sorted(before & after),
            }
            for role, key in [("original", "orig_image"), ("edited", "syn_image")]:
                yield Record(
                    pair + "/" + role,
                    pair,
                    "image",
                    str(files["images_root"] / row[key]),
                    "test",
                    edit=edit | {"role": role},
                )
