"""Small generated files in native benchmark formats, for offline integration tests."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from mm_sae.io import atomic_json, file_lock

NAMES = ("red", "green", "blue")


def scene(i):
    active = [j for j in range(3) if (i >> j) & 1]
    im = Image.new("RGB", (12, 12), tuple(240 if j in active else 12 for j in range(3)))
    return im, " ".join(NAMES[j] for j in active) or "empty", active


def training_rows(*_):
    for i in range(128):
        im, caption, _ = scene(i)
        yield {"__key__": f"fit-{i}", "jpg": im, "txt": caption}


def generate(dest):
    with file_lock(dest.parent / ".fixture.lock"):
        if (dest / "fixture-v1.json").exists():
            return
        if dest.exists():
            raise ValueError(f"Fixture directory already exists: {dest}")
        with TemporaryDirectory(dir=dest.parent) as tmp:
            root = Path(tmp) / "fixture"
            root.mkdir()
            _write_fixture(root)
            atomic_json(root / "fixture-v1.json", {"synthetic": True, "version": 1})
            root.rename(dest)


def _write_fixture(root):
    for d in ("images", "masks", "edits"):
        (root / d).mkdir()
    rows = []
    attrs = []
    captions = []
    graphs = []
    meta = []
    flickr = []
    docci = []
    iiw = []
    (root / "names.txt").write_text("1: red\n2: green\n3: blue\n")
    (root / "attributes.txt").write_text("1 red\n2 green\n3 blue\n")
    cub_labels = []
    cub_names = []
    partition = []
    for i in range(120):
        im, text, active = scene(i)
        filename = f"{i}.jpg"
        im.save(root / "images" / filename)
        rows.append(dict(id=i, file_name=filename))
        attrs.extend(dict(image_id=i, category_id=j + 1) for j in active)
        mask = np.full((12, 12), 255, np.uint8)
        for j in active:
            mask[j] = j
        Image.fromarray(mask).save(root / "masks" / (str(i) + ".png"))
        captions.extend(dict(id=i * 2 + j, image_id=i, caption=text + " picture" * j) for j in range(2))
        graphs.append(
            dict(
                image_id=i,
                regions=[
                    dict(
                        region_id=i,
                        phrase=text,
                        objects=[dict(name=NAMES[j], attributes=["bright"]) for j in active],
                        relationships=[],
                    )
                ],
            )
        )
        meta.append(dict(image_id=i, coco_id=i))
        cub_names.append(f"{i + 1} {filename}")
        partition.append(f"{i + 1} {int(i < 88)}")
        cub_labels.extend(f"{i + 1} {j + 1} {int(j in active)} 4 0" for j in range(3))
        if i >= 88:
            flickr.append(
                dict(filename=filename, split="test", sentences=[dict(raw=text), dict(raw=text + " bird")])
            )
            docci.append(dict(example_id=str(i), split="test", description=text + " details" * 12))
            iiw.append({"image/key": str(i), "IIW": text + " objects" * 14})
    cats = [dict(id=j + 1, name=n) for j, n in enumerate(NAMES)]
    for part, sel in (("train", lambda i: i < 88), ("test", lambda i: i >= 88)):
        atomic_json(
            root / (part + ".json"),
            dict(
                images=[x for x in rows if sel(x["id"])],
                categories=cats,
                annotations=[a for a in attrs if sel(a["image_id"])],
            ),
        )
    atomic_json(
        root / "captions.json",
        dict(images=rows[88:], annotations=[c for c in captions if c["image_id"] >= 88]),
    )
    atomic_json(root / "graphs.json", graphs)
    atomic_json(root / "meta.json", meta)
    atomic_json(root / "flickr.json", dict(images=flickr))
    for name, items in (("docci", docci), ("iiw", iiw)):
        (root / (name + ".jsonl")).write_text("".join(json.dumps(x) + "\n" for x in items))
    for name, items in (("images", cub_names), ("partition", partition), ("image_attributes", cub_labels)):
        (root / (name + ".txt")).write_text("\n".join(items) + "\n")
    for kind in ("syncoco", "syncub"):
        examples = []
        for i in range(3):
            old = i
            new = (i + 1) % 3
            keep = (i + 2) % 3
            before = [old, keep]
            after = [keep] if kind == "syncoco" else [new, keep]
            ex = dict(pair_id=i)
            for role, labels in (("orig_image", before), ("syn_image", after)):
                path = f"edits/{kind}-{i}-{role}.png"
                scene(sum(1 << j for j in labels))[0].save(root / path)
                ex[role] = path
            if kind == "syncoco":
                ex.update(
                    orig_labels=json.dumps(before), syn_labels=json.dumps(after), removed_class=NAMES[old]
                )
            else:
                ex.update(
                    orig_attr_idx=json.dumps(before),
                    syn_attr_idx=json.dumps(after),
                    old_attr=NAMES[old],
                    new_attr=NAMES[new],
                )
            examples.append(ex)
        with (root / (kind + ".csv")).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(examples[0]))
            writer.writeheader()
            writer.writerows(examples)
