"""Each unique image and each human caption retains its own retrieval row."""

import json
from .records import Record, jsonlines


def retrieval(kind, files):
    if kind == "coco_retrieval":
        data = json.loads(files["captions"].read_text())
        for item in data["images"]:
            key = f"coco/{item['id']}"
            yield Record(key, key, "image", str(files["images"] / item["file_name"]), "test")
        for item in data["annotations"]:
            yield Record(f"caption/{item['id']}", f"coco/{item['image_id']}", "text", item["caption"], "test")
    elif kind == "flickr30k":
        for item in json.loads(files["karpathy"].read_text())["images"]:
            if item["split"] != "test":
                continue
            key = "flickr/" + item["filename"]
            yield Record(key, key, "image", str(files["images"] / item["filename"]), "test")
            for j, s in enumerate(item["sentences"]):
                yield Record(f"{key}/{j}", key, "text", s["raw"], "test")
    else:
        for item in jsonlines(files["captions"]):
            if kind == "docci" and item["split"] != "test":
                continue
            key = item["example_id"] if kind == "docci" else item["image/key"]
            caption = item["description"] if kind == "docci" else item["IIW"]
            yield Record(key, key, "image", str(files["images"] / (key + ".jpg")), "test")
            yield Record(key + "/caption", key, "text", caption, "test")
