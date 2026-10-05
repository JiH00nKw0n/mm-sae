"""Compare caption dictionaries against saved COCO captions without model inference."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np

from mm_sae.data.text import CaptionEditor, load_concepts
from mm_sae.io import atomic_json, sha256, write_csv


def audit(baseline: Path, proposal: Path, reference: Path, captions_dir: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    old_concepts, new_concepts = load_concepts(baseline), load_concepts(proposal)
    old_by_id = {c.id: c for c in old_concepts}
    new_by_id = {c.id: c for c in new_concepts}
    if set(old_by_id) != set(new_by_id):
        raise ValueError("An alias review must preserve every concept ID")
    old_editor = CaptionEditor(old_concepts)
    editor = CaptionEditor(new_concepts)
    counts, examples, split_summaries = [], [], []
    start = time.monotonic()
    for split in ("train2017", "val2017"):
        index = reference / "index" / split
        ids = json.loads((index / "concept_ids.json").read_text())
        columns = {c: k for k, c in enumerate(ids)}
        presence = np.load(index / "presence.npy", mmap_mode="r")
        mentions = np.load(index / "mentions.npy", mmap_mode="r")
        parents = np.load(index / "parents.npy", mmap_mode="r")
        cached_captions = json.loads((index / "captions.json").read_text())
        source = json.loads((captions_dir / f"captions_{split}.json").read_text())
        captions = sorted(source["annotations"], key=lambda r: (r["image_id"], r["id"]))
        image_rows = {r["id"]: k for k, r in enumerate(sorted(source["images"], key=lambda r: r["id"]))}
        if (
            len(captions) != len(mentions)
            or len(cached_captions) != len(captions)
            or len(image_rows) != len(presence)
            or set(ids) != set(old_by_id)
        ):
            raise ValueError("Caption source does not match the full cached dataset")
        new_labels = np.zeros(mentions.shape, dtype=bool)
        added, removed, matched_terms = Counter(), Counter(), Counter()
        example_counts: Counter = Counter()
        no_spans = Counter()
        for k, cap in enumerate(captions):
            if image_rows[cap["image_id"]] != int(parents[k]):
                raise ValueError("Caption ordering differs from the cache")
            cached_caption = cached_captions[k]
            if (
                cached_caption["caption_id"] != cap["id"]
                or cached_caption["text"] != cap["caption"]
            ):
                raise ValueError("Caption identity or text differs from the cache")
            present, spans, _, unavailable = editor.analyze(cap["id"], cap["caption"])
            old_present = set(old_editor.analyze(cap["id"], cap["caption"])[0])
            if old_present != {ids[j] for j in np.flatnonzero(mentions[k])}:
                raise ValueError(
                    f"Baseline dictionary does not reproduce cached labels for caption {cap['id']}"
                )
            new_present = set(present)
            for c in present:
                new_labels[k, columns[c]] = True
                for a, b in spans.get(c, []):
                    matched_terms[(c, cap["caption"][a:b].lower())] += 1
            no_spans.update(unavailable)
            for change, concepts in (
                ("added", new_present - old_present),
                ("removed", old_present - new_present),
            ):
                for c in concepts:
                    (added if change == "added" else removed)[c] += 1
                    visible = bool(presence[int(parents[k]), columns[c]])
                    key = (c, change, visible)
                    if example_counts[key] < 6:
                        examples.append(
                            {
                                "split": split,
                                "concept_id": c,
                                "name": old_by_id[c].name,
                                "change": change,
                                "visible_in_model_crop": visible,
                                "caption_id": cap["id"],
                                "image_id": cap["image_id"],
                                "caption": cap["caption"],
                                "matched_spans": [
                                    {"start": a, "end": b, "text": cap["caption"][a:b]}
                                    for a, b in spans.get(c, [])
                                ],
                            }
                        )
                        example_counts[key] += 1
            if (k + 1) % 50000 == 0:
                elapsed = time.monotonic() - start
                print(f"{split}: {k + 1:,}/{len(captions):,} captions; elapsed {elapsed:.1f}s", flush=True)
        np.save(output / f"{split}_proposed_mentions.npy", new_labels)
        for c in ids:
            j = columns[c]
            visible = presence[parents, j].astype(bool)
            original_count = int(np.count_nonzero(mentions[:, j]))
            proposed_count = int(np.count_nonzero(new_labels[:, j]))
            counts.append(
                {
                    "split": split,
                    "concept_id": c,
                    "name": old_by_id[c].name,
                    "old_caption_count": original_count,
                    "proposed_caption_count": proposed_count,
                    "newly_labeled_captions": added[c],
                    "no_longer_labeled_captions": removed[c],
                    "visible_caption_rows": int(visible.sum()),
                    "old_detected_with_visible_label": int(np.count_nonzero(mentions[visible, j])),
                    "proposed_detected_with_visible_label": int(np.count_nonzero(new_labels[visible, j])),
                    "proposed_unavailable_spans": no_spans[c],
                    "added_aliases": json.dumps(
                        sorted(set(new_by_id[c].aliases) - set(old_by_id[c].aliases))
                    ),
                    "removed_aliases": json.dumps(
                        sorted(set(old_by_id[c].aliases) - set(new_by_id[c].aliases))
                    ),
                }
            )
        write_csv(
            output / f"{split}_matched_expressions.csv",
            [
                {"concept_id": c, "name": old_by_id[c].name, "expression": term, "occurrences": n}
                for (c, term), n in sorted(matched_terms.items())
            ],
        )
        split_summaries.append(
            {
                "split": split,
                "captions_checked": len(captions),
                "changed_caption_label_sets": int(np.any(new_labels != mentions, axis=1).sum()),
                "changed_concepts": sum(bool(added[c] or removed[c]) for c in ids),
                "zero_match_concepts_before": [c for c in ids if not mentions[:, columns[c]].any()],
                "zero_match_concepts_after": [c for c in ids if not new_labels[:, columns[c]].any()],
            }
        )
    write_csv(output / "all_concept_counts.csv", counts)
    with (output / "changed_caption_examples.jsonl").open("w") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "baseline_dictionary": str(baseline.resolve()),
        "baseline_sha256": sha256(baseline),
        "baseline_cache_validation": (
            "Recomputed baseline labels for every caption and verified exact equality with the "
            "reference mentions, caption IDs, and caption text before comparing the proposal."
        ),
        "proposed_dictionary": str(proposal.resolve()),
        "proposal_sha256": sha256(proposal),
        "reference_run": str(reference.resolve()),
        "concepts_checked": len(ids),
        "splits": split_summaries,
        "elapsed_seconds": time.monotonic() - start,
        "scope": "Dictionary label census, not human semantic precision/recall. Image visibility is diagnostic only and never decides text labels. No model calls, tokenization, training, or experimental matching reruns.",
        "cache_warning": "Counts concern label changes; an unchanged label can still have changed character/token spans. Reuse masked-text caches only after exact caption-row and mask-token-position equality checks.",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--captions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.baseline, args.proposal, args.reference_run, args.captions_dir, args.output)


if __name__ == "__main__":
    main()
