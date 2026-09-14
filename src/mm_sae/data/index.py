"""Index each image once and retain every caption with an explicit image-row mapping."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..io import atomic_json, write_csv, sha256
from .download import annotation_archive, extract_member, fetch
from .text import CaptionEditor, load_concepts
from ..progress import iter_progress

LOG = logging.getLogger(__name__)


class Index:
    def __init__(self, root: Path, split: str):
        self.root = root / "index" / split
        self.images = json.loads((self.root / "images.json").read_text())
        self.captions = json.loads((self.root / "captions.json").read_text())
        self.presence = np.load(self.root / "presence.npy")
        self.full_presence = np.load(self.root / "full_presence.npy")
        self.areas = np.load(self.root / "areas.npy")
        self.mentions = np.load(self.root / "mentions.npy")
        self.parents = np.load(self.root / "parents.npy")
        self.concept_ids = json.loads((self.root / "concept_ids.json").read_text())
        self.columns = {int(c): i for i, c in enumerate(self.concept_ids)}


def fixture(config):
    root = config.data.root
    for split_no, split in enumerate(config.data.splits):
        records, captions = [], []
        rng = np.random.default_rng(91 + split_no)
        n = config.data.fixture_images if split_no == 0 else max(8, config.data.fixture_images // 3)
        for i in range(n):
            image_id = split_no * 10000 + i
            size = config.data.fixture_size
            rgb = np.full((size, size, 3), 230, np.uint8)
            mask = np.full((size, size), 255, np.uint8)
            a = rng.random() < 0.5
            b = rng.random() < (0.8 if a else 0.2)
            flags = [
                (0, a, (210, 50, 60), (slice(8, size // 2), slice(6, size // 2))),
                (1, b, (40, 100, 210), (slice(8, size // 2), slice(size // 2 + 2, size - 3))),
                (
                    123,
                    rng.random() < 0.7,
                    (60, 160, 70),
                    (slice(size // 2 + 3, size - 2), slice(2, size - 2)),
                ),
            ]
            names = []
            for c, on, color, region in flags:
                if on:
                    rgb[region] = color
                    mask[region] = c
                    names.append({0: "person", 1: "bicycle", 123: "grass"}[c])
            image_path = root / config.data.images_pattern.format(split=split, image_id=image_id)
            mask_path = root / config.data.masks_pattern.format(split=split, image_id=image_id)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            # Lossless fixture RGB even though the compatibility filename ends in .jpg.
            Image.fromarray(rgb).save(image_path, format="PNG")
            Image.fromarray(mask).save(mask_path)
            records.append({"id": image_id, "file_name": f"{image_id:012d}.jpg"})
            for j in range(config.data.fixture_captions):
                text = "A scene with " + (" and ".join(names) if names else "nothing") + f" in view {j}."
                captions.append({"id": image_id * 10 + j, "image_id": image_id, "caption": text})
        dest = root / config.data.captions_pattern.format(split=split)
        atomic_json(dest, {"images": records, "annotations": captions})


def source_records(config, split):
    path = config.data.root / config.data.captions_pattern.format(split=split)
    if not path.exists():
        if not config.data.download:
            raise FileNotFoundError(f"Missing captions {path}. Set data.download=true or mount the dataset.")
        with annotation_archive(config, "annotations_trainval2017.zip") as archive:
            extract_member(archive, f"annotations/captions_{split}.json", path)
    value = json.loads(path.read_text())
    images = sorted(value["images"], key=lambda r: r["id"])
    expected = config.data.expected_images.get(split)
    if expected and len(images) != expected:
        raise ValueError(f"{split}: expected {expected} source images, found {len(images)}")
    limit = config.data.image_limits.get(split)
    if limit is not None:
        if limit < 2:
            raise ValueError("Smoke subsets need at least two images")
        images = images[:limit]
    image_ids = {r["id"] for r in images}
    captions = [c for c in value["annotations"] if c["image_id"] in image_ids]
    return (
        images,
        captions,
        {"source_images": len(value["images"]), "source_captions": len(value["annotations"])},
    )


def acquire_pixels(config, split, images):
    missing_masks, missing_images = [], []
    for row in images:
        image_id = row["id"]
        ip = config.data.root / config.data.images_pattern.format(split=split, image_id=image_id)
        mp = config.data.root / config.data.masks_pattern.format(split=split, image_id=image_id)
        if not ip.exists():
            missing_images.append((image_id, ip))
        if not mp.exists():
            missing_masks.append((image_id, mp))
    if (missing_masks or missing_images) and not config.data.download:
        raise FileNotFoundError("Missing image/mask files. Enable download or mount the full COCO directory.")
    if missing_masks:
        with annotation_archive(config, "stuffthingmaps_trainval2017.zip") as archive:
            for image_id, path in iter_progress(
                tqdm(missing_masks, desc=f"Download masks {split}"),
                f"Mask files {split}",
                total=len(missing_masks),
                unit="files",
            ):
                extract_member(archive, f"{split}/{image_id:012d}.png", path)
    if missing_images:
        if config.data.image_limits:
            for image_id, path in iter_progress(
                tqdm(missing_images, desc=f"Download images {split}"),
                f"Image files {split}",
                total=len(missing_images),
                unit="files",
            ):
                fetch(config.data.images_url_pattern.format(split=split, image_id=image_id), path)
        else:
            with annotation_archive(config, f"{split}.zip") as archive:
                for image_id, path in iter_progress(
                    tqdm(missing_images, desc=f"Extract images {split}"),
                    f"Extract images {split}",
                    total=len(missing_images),
                    unit="files",
                ):
                    extract_member(archive, f"{split}/{image_id:012d}.jpg", path)


def prepare(config, encoder):
    if config.data.source == "synthetic":
        fixture(config)
    concepts = load_concepts(config.data.concepts_file, config.data.source == "synthetic")
    editor = CaptionEditor(concepts, config.data.reviewed_captions)
    columns = {c.id: i for i, c in enumerate(concepts)}
    seen_images, seen_captions, summaries = set(), set(), {}
    for split in config.data.splits:
        images, caps, source_counts = source_records(config, split)
        ids = {r["id"] for r in images}
        if seen_images & ids:
            raise ValueError("An image occurs in multiple splits")
        seen_images |= ids
        acquire_pixels(config, split, images)
        out = config.output / "index" / split
        out.mkdir(parents=True, exist_ok=True)
        presence = np.zeros((len(images), len(concepts)), dtype=bool)
        full = presence.copy()
        areas = np.zeros(presence.shape, np.float32)
        indexed_images = []
        for i, row in enumerate(
            iter_progress(
                tqdm(images, desc=f"Read annotations {split}"),
                f"Read annotations {split}",
                total=len(images),
                unit="images",
            )
        ):
            image_id = row["id"]
            ip = config.data.root / config.data.images_pattern.format(split=split, image_id=image_id)
            mp = config.data.root / config.data.masks_pattern.format(split=split, image_id=image_id)
            with Image.open(mp) as label_image:
                if label_image.format != "PNG" or label_image.mode not in {"L", "P"}:
                    raise ValueError(f"Expected lossless integer-label PNG: {mp}")
                mask = np.array(label_image)
            with Image.open(ip) as photo:
                if mask.shape != (photo.height, photo.width):
                    raise ValueError(f"Image and mask dimensions differ: {image_id}")
            if set(np.unique(mask)) - (set(columns) | {255}) and config.data.concepts_file is None:
                raise ValueError(f"Unexpected raw mask IDs in {mp}; possible remapping or JPEG corruption")
            visible = encoder.visible_mask(mask)
            scoped = visible if config.data.label_scope == "model_input" else mask
            for c in concepts:
                full[i, columns[c.id]] = np.any(mask == c.id)
                presence[i, columns[c.id]] = np.any(scoped == c.id)
                areas[i, columns[c.id]] = np.mean(scoped == c.id)
            indexed_images.append(
                {
                    "image_id": image_id,
                    "image": str(ip),
                    "mask": str(mp),
                    "image_sha256": sha256(ip),
                    "mask_sha256": sha256(mp),
                }
            )
        image_rows = {r["image_id"]: i for i, r in enumerate(indexed_images)}
        indexed_caps, reviews = [], []
        mentions = np.zeros((len(caps), len(concepts)), bool)
        for i, cap in enumerate(
            iter_progress(
                sorted(caps, key=lambda c: (c["image_id"], c["id"])),
                f"Index captions {split}",
                unit="captions",
            )
        ):
            if cap["id"] in seen_captions:
                raise ValueError("Duplicate caption ID")
            seen_captions.add(cap["id"])
            present, edits, status, unavailable = editor.analyze(cap["id"], cap["caption"])
            for c in present:
                mentions[i, columns[c]] = True
            record = {
                "caption_id": cap["id"],
                "image_row": image_rows[cap["image_id"]],
                "text": cap["caption"],
                "concept_ids": present,
                "edits": edits,
                "annotation_status": status,
                "unavailable": unavailable,
            }
            indexed_caps.append(record)
            for c in present:
                reviews.append(
                    {
                        "caption_id": cap["id"],
                        "image_id": cap["image_id"],
                        "concept_id": c,
                        "original": cap["caption"],
                        "edited": edits.get(c),
                        "status": status,
                    }
                )
        parents = np.array([r["image_row"] for r in indexed_caps], np.int64)
        counts = np.bincount(parents, minlength=len(images))
        if np.any(counts == 0):
            raise ValueError("Every indexed image must have at least one caption")
        for name, value in [
            ("images", indexed_images),
            ("captions", indexed_caps),
            ("concept_ids", list(columns)),
        ]:
            atomic_json(out / f"{name}.json", value)
        for name, value in [
            ("parents", parents),
            ("presence", presence),
            ("full_presence", full),
            ("areas", areas),
            ("mentions", mentions),
        ]:
            np.save(out / f"{name}.npy", value)
        write_csv(out / "caption_edit_review.csv", reviews)
        unique_counts, frequencies = np.unique(counts, return_counts=True)
        summaries[split] = {
            **source_counts,
            "images": len(images),
            "captions": len(caps),
            "caption_count_histogram": dict(zip(map(str, unique_counts), map(int, frequencies))),
            "label_scope": config.data.label_scope,
            "automatic_caption_annotations": sum(
                c["annotation_status"] != "human_reviewed" for c in indexed_caps
            ),
            "captions_with_available_edits": sum(bool(c["edits"]) for c in indexed_caps),
        }
        LOG.info("Indexed %s", summaries[split])
    atomic_json(
        config.output / "dataset.json",
        {
            "splits": summaries,
            "concepts": [c.__dict__ for c in concepts],
            "small_subset": bool(config.data.image_limits),
            "test_only": config.data.source == "synthetic",
        },
    )
