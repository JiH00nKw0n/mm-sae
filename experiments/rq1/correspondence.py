"""The first two RQ1 analyses. All concept pairs and both full-feature assignments are retained."""

import json
import numpy as np

from mm_sae.data.index import Index
from mm_sae.features import original_latents
from mm_sae.io import atomic_json, write_csv
from mm_sae.metrics.statistics import correlation, bin_summary
from mm_sae.metrics.matching import match


def build_panel(config, options):
    index = Index(config.output, options.correlation_split)
    image, text = original_latents(config, options.correlation_split)
    panel = correlation(image[index.parents], text)
    np.savez_compressed(config.output / "panel.npz", **panel)
    return panel


def feature_labels(representatives, side):
    labels = {}
    for c, pair in representatives.items():
        f = pair.get(side)
        if f is not None:
            labels.setdefault(int(f), set()).add(int(c))
    return labels


def assess(panel, assignment, representatives):
    images, texts = [feature_labels(representatives, side) for side in ["image", "text"]]
    rows = []
    for i, concepts in sorted(images.items()):
        j = int(assignment[i])
        partner = texts.get(j, set())
        if j < 0:
            status = "unmatched"
        elif not panel["valid_image"][i] or not panel["valid_text"][j]:
            status = "undefined_correlation"
        elif len(concepts) != 1 or len(partner) != 1:
            status = "ambiguous_or_unlabeled"
        else:
            status = "same" if concepts == partner else "different"
        rows.append(
            {
                "image_feature": i,
                "text_feature": j,
                "image_concepts": sorted(concepts),
                "text_concepts": sorted(partner),
                "status": status,
                "score": float(panel["C"][i, j])
                if j >= 0 and panel["valid_image"][i] and panel["valid_text"][j]
                else None,
            }
        )
    return rows


def experiment1(config, options):
    root = config.output
    panel = dict(np.load(root / "panel.npz"))
    reps = json.loads((root / "representatives.json").read_text())
    index = Index(root, options.correlation_split)
    image_labels = correlation(index.presence, index.presence)
    cross_labels = correlation(index.presence[index.parents], index.mentions)
    selected_labels = image_labels if options.label_correlation == "image_image" else cross_labels
    rows = []
    for a, av in reps.items():
        for b, bv in reps.items():
            if a == b or av["image"] is None or bv["text"] is None:
                continue
            i, j = av["image"], bv["text"]
            ac, bc = index.columns[int(a)], index.columns[int(b)]
            lv = bool(selected_labels["valid_image"][ac] and selected_labels["valid_text"][bc])
            fv = bool(panel["valid_image"][i] and panel["valid_text"][j])
            row = {
                "image_concept": int(a),
                "text_concept": int(b),
                "image_feature": i,
                "text_feature": j,
                "label_correlation": float(selected_labels["C"][ac, bc]) if lv else None,
                "coactivation_correlation": float(panel["C"][i, j]) if fv else None,
                "valid": lv and fv,
                "label_source": options.label_correlation,
                "undefined_reason": "" if lv and fv else "constant_label" if not lv else "constant_feature",
            }
            for name, source in [("image_image", image_labels), ("image_text", cross_labels)]:
                row[name + "_label_correlation"] = (
                    float(source["C"][ac, bc])
                    if source["valid_image"][ac] and source["valid_text"][bc]
                    else None
                )
            rows.append(row)
    out = root / "rq1" / "experiment1"
    write_csv(out / "all_ordered_concept_pairs.csv", rows)
    bins = bin_summary(rows, options.bin_width)
    write_csv(out / "bins.csv", bins)
    atomic_json(
        out / "summary.json",
        {
            "label_source": options.label_correlation,
            "pairs": len(rows),
            "valid_pairs": sum(r["valid"] for r in rows),
            "bins": bins,
            "std_definition": "population standard deviation over every eligible ordered concept pair, ddof=0",
            "self_pairs": "excluded",
            "sampling": "none",
            "claim": "association with co-occurrence, not semantic classification accuracy",
        },
    )


def experiment2(config, options):
    root = config.output
    panel = dict(np.load(root / "panel.npz"))
    reps = json.loads((root / "representatives.json").read_text())
    index = Index(root, options.correlation_split)
    labels = correlation(index.presence, index.presence)
    assignments = match(panel["C"], panel["alive_image"], panel["alive_text"])
    out = root / "rq1" / "experiment2"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "all_feature_assignments.npz", **assignments)
    all_rows, summary = [], {}
    for method in ["greedy", "hungarian"]:
        rows = assess(panel, assignments[method], reps)
        for r in rows:
            r["method"] = method
            if len(r["image_concepts"]) == 1:
                a = r["image_concepts"][0]
                i, j = r["image_feature"], reps[str(a)]["text"]
                r["same_concept_text_feature"] = j
                r["same_concept_score"] = (
                    float(panel["C"][i, j])
                    if j is not None and panel["valid_image"][i] and panel["valid_text"][j]
                    else None
                )
            if method == "greedy":
                r["top_score_ties"] = int(assignments["greedy_ties"][r["image_feature"]])
            if len(r["image_concepts"]) == len(r["text_concepts"]) == 1:
                a, b = r["image_concepts"][0], r["text_concepts"][0]
                ac, bc = index.columns[a], index.columns[b]
                r["image_label_correlation"] = (
                    float(labels["C"][ac, bc])
                    if labels["valid_image"][ac] and labels["valid_text"][bc]
                    else None
                )
        summary[method] = {
            "denominator": len(rows),
            "counts": {
                s: sum(r["status"] == s for r in rows)
                for s in ["same", "different", "ambiguous_or_unlabeled", "undefined_correlation", "unmatched"]
            },
        }
        all_rows.extend(rows)
    atomic_json(out / "assessed_rows.json", all_rows)
    write_csv(out / "assessed_rows.csv", all_rows)
    atomic_json(out / "summary.json", summary)
