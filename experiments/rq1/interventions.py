"""Case-specific co-occurrence attenuation with a deletion-budget-matched random control."""

from __future__ import annotations

import json
import numpy as np

from mm_sae.data.index import Index
from mm_sae.features import original_latents, read_counterfactual
from mm_sae.io import atomic_json, write_csv
from mm_sae.metrics.statistics import correlation
from mm_sae.metrics.matching import match
from mm_sae.metrics.sparse_ops import replace_rows, activation_changes, take_rows
from mm_sae.training import assert_frozen
from mm_sae.progress import iter_progress, progress_task
from .correspondence import assess


def deletion_budget(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or a.all():
        return {"status": "constant_anchor_label"}
    p1, p0 = float(b[a].mean()), float(b[~a].mean())
    if p0 <= 0:
        return {"status": "no_target_without_anchor", "p1": p1, "p0": p0}
    if p1 <= p0:
        return {"status": "not_positive_cooccurrence", "p1": p1, "p0": p0}
    q = 1 - p0 / p1
    count = int(np.rint(q * np.count_nonzero(a & b)))
    return {
        "status": "ready" if count else "zero_integer_budget",
        "p1": p1,
        "p0": p0,
        "q": q,
        "n_remove": count,
    }


def choose_images(a, b, eligible, areas, caption_counts, mention_counts, budget, strata, rng):
    conditional_pool = np.flatnonzero(a & b & eligible)
    if len(conditional_pool) < budget:
        raise ValueError("insufficient_editable_cooccurrences")
    conditional = np.sort(rng.choice(conditional_pool, budget, replace=False))
    control_pool = np.flatnonzero(b & eligible)
    boundaries = np.unique(np.quantile(areas[control_pool], np.linspace(0, 1, strata + 1)))
    bins = np.searchsorted(boundaries[1:-1], areas, side="right")
    keys = [(int(caption_counts[i]), int(mention_counts[i]), int(bins[i])) for i in range(len(a))]
    control = []
    for key in sorted({keys[i] for i in conditional}):
        n = sum(keys[i] == key for i in conditional)
        pool = [i for i in control_pool if keys[i] == key]
        if len(pool) < n:
            raise ValueError("insufficient_matched_control")
        control.extend(rng.choice(pool, n, replace=False).tolist())
    control = np.array(sorted(control), np.int64)
    if mention_counts[conditional].sum() != mention_counts[control].sum():
        raise AssertionError("Caption deletion budgets differ")
    return conditional, control, boundaries


def altered_activations(index, chosen, original_i, original_t, cf):
    image_rows, text_rows, ci, ct = cf
    image_lookup = {int(row): k for k, row in enumerate(image_rows)}
    replace_i = take_rows(ci, [image_lookup[int(i)] for i in chosen])
    chosen_captions = np.isin(index.parents[text_rows], chosen)
    return (
        replace_rows(original_i, chosen, replace_i),
        replace_rows(original_t, text_rows[chosen_captions], take_rows(ct, np.flatnonzero(chosen_captions))),
    )


def safe_score(panel, i, j):
    if j is None or not panel["valid_image"][i] or not panel["valid_text"][j]:
        return None
    return float(panel["C"][i, j])


def experiment3(config, options):
    root = config.output
    assert_frozen(root)
    index = Index(root, options.correlation_split)
    original_i, original_t = original_latents(config, options.correlation_split)
    base = dict(np.load(root / "panel.npz"))
    reps = json.loads((root / "representatives.json").read_text())
    baseline_rows = json.loads((root / "rq1" / "experiment2" / "assessed_rows.json").read_text())
    baselines = {
        m: {r["image_feature"]: r for r in baseline_rows if r["method"] == m} for m in ["greedy", "hungarian"]
    }
    pairs = sorted(
        {(r["image_concepts"][0], r["text_concepts"][0]) for r in baseline_rows if r["status"] == "different"}
    )
    out = root / "rq1" / "experiment3"
    out.mkdir(parents=True, exist_ok=True)
    caption_counts = np.bincount(index.parents, minlength=len(index.images))
    outcomes, exclusions, eligible_pairs = [], [], []
    for a, b in iter_progress(pairs, "All original error concept pairs", unit="pairs"):
        ac, bc = index.columns[a], index.columns[b]
        has_a, has_b = index.presence[:, ac], index.presence[:, bc]
        plan = deletion_budget(has_a, has_b)
        if plan["status"] != "ready":
            exclusions.append({"anchor": a, "remove": b, **plan})
            continue
        image_feature = reps[str(a)]["image"]
        right_feature, wrong_feature = reps[str(a)]["text"], reps[str(b)]["text"]
        if right_feature is None:
            exclusions.append({"anchor": a, "remove": b, "status": "no_same_concept_text_representative"})
            continue
        eligible = np.ones(len(index.images), bool)
        for cap in index.captions:
            if b in cap["concept_ids"] and str(b) not in cap["edits"]:
                eligible[cap["image_row"]] = False
        mention_counts = np.bincount(
            index.parents, weights=index.mentions[:, bc], minlength=len(index.images)
        ).astype(int)
        if np.count_nonzero(has_a & has_b & eligible) < int(plan["n_remove"]):
            exclusions.append(
                {
                    "anchor": a,
                    "remove": b,
                    "status": "insufficient_editable_cooccurrences",
                    **{k: v for k, v in plan.items() if k != "status"},
                }
            )
            continue
        eligible_pairs.append([a, b])
        cf = read_counterfactual(root, options.correlation_split, b)
        for repeat in iter_progress(
            range(options.intervention_repeats), f"Removal repetitions for {a}, {b}", unit="repetitions"
        ):
            case = out / f"{a}-{b}" / str(repeat)
            if (case / "outcomes.json").exists():
                outcomes.extend(json.loads((case / "outcomes.json").read_text()))
                continue
            rng = np.random.default_rng(np.random.SeedSequence([options.intervention_seed, a, b, repeat]))
            conditional, random, boundaries = choose_images(
                has_a,
                has_b,
                eligible,
                index.areas[:, bc],
                caption_counts,
                mention_counts,
                plan["n_remove"],
                options.area_strata,
                rng,
            )
            case.mkdir(parents=True, exist_ok=True)
            selections = {
                "original": np.array([], np.int64),
                "random_removal": random,
                "cooccurrence_removal": conditional,
            }
            atomic_json(
                case / "selection.json",
                {
                    "budget": plan,
                    "area_boundaries": boundaries.tolist(),
                    "overlap_images": int(np.intersect1d(random, conditional).size),
                    "conditions": {
                        name: [index.images[i]["image_id"] for i in selected]
                        for name, selected in selections.items()
                    },
                },
            )
            for condition, selected in selections.items():
                mean_off = mean_on = 0.0
                if condition == "original":
                    panel = base
                else:
                    xi, yt = altered_activations(index, selected, original_i, original_t, cf)
                    changes = activation_changes(take_rows(original_i, selected), take_rows(xi, selected))
                    mean_off = float(changes["turned_off"].mean())
                    mean_on = float(changes["turned_on"].mean())
                    with progress_task(f"Recompute full Pearson matrix for {a}, {b}, {condition}"):
                        panel = correlation(take_rows(xi, index.parents), yt)
                    # Original candidate universe is fixed, including features that become constant after masking.
                    if options.save_intervention_panels:
                        np.savez_compressed(case / f"{condition}_panel.npz", **panel)
                with progress_task(f"Greedy and Hungarian matching for {a}, {b}, {condition}"):
                    assignment = match(
                        panel["C"],
                        base["alive_image"],
                        base["alive_text"],
                        panel["valid_image"],
                        panel["valid_text"],
                    )
                np.savez_compressed(case / f"{condition}_assignments.npz", **assignment)
                after_b = has_b.copy()
                after_b[selected] = False
                label_panel = correlation(has_a[:, None], after_b[:, None])
                after_text_b = index.mentions[:, bc].copy()
                after_text_b[np.isin(index.parents, selected)] = False
                cross_label = correlation(has_a[index.parents, None], after_text_b[:, None])
                for method in ["greedy", "hungarian"]:
                    assessed = assess(panel, assignment[method], reps)
                    atomic_json(case / f"{condition}_{method}_assessed.json", assessed)
                    now = {r["image_feature"]: r for r in assessed}
                    target = now[image_feature]
                    old = baselines[method][image_feature]
                    correct_rows = [i for i, row in baselines[method].items() if row["status"] == "same"]
                    invalidated = sum(
                        now[i]["status"] in {"undefined_correlation", "ambiguous_or_unlabeled", "unmatched"}
                        for i in correct_rows
                    )
                    newly_wrong = sum(now[i]["status"] == "different" for i in correct_rows)
                    record = {
                        "anchor": a,
                        "remove": b,
                        "repeat": repeat,
                        "condition": condition,
                        "method": method,
                        "original_status": old["status"],
                        "new_status": target["status"],
                        "recovered": old["status"] == "different" and target["status"] == "same",
                        "wrong_score": safe_score(panel, image_feature, wrong_feature),
                        "same_concept_score": safe_score(panel, image_feature, right_feature),
                        "baseline_correct_rows": len(correct_rows),
                        "newly_wrong_rows": newly_wrong,
                        "newly_unassessable_rows": invalidated,
                        "label_correlation": safe_score(label_panel, 0, 0),
                        "image_text_label_correlation": safe_score(cross_label, 0, 0),
                        "p_b_given_a_after": float(after_b[has_a].mean()),
                        "p_b_given_not_a_after": float(after_b[~has_a].mean()),
                        "images_removed": len(selected),
                        "captions_edited": int(mention_counts[selected].sum()),
                        "mean_image_features_turned_off": mean_off,
                        "mean_image_features_turned_on": mean_on,
                        "mean_mask_area": float(index.areas[selected, bc].mean()) if len(selected) else 0.0,
                        "random_conditional_overlap": int(np.intersect1d(random, conditional).size),
                    }
                    if record["same_concept_score"] is not None and record["wrong_score"] is not None:
                        record["correct_minus_wrong"] = record["same_concept_score"] - record["wrong_score"]
                    outcomes.append(record)
            atomic_json(case / "outcomes.json", outcomes[-6:])
    write_csv(
        out / "outcomes.csv",
        outcomes,
        None
        if outcomes
        else [
            "anchor",
            "remove",
            "repeat",
            "condition",
            "method",
            "original_status",
            "new_status",
            "recovered",
        ],
    )
    write_csv(out / "exclusions.csv", exclusions, None if exclusions else ["anchor", "remove", "status"])
    atomic_json(
        out / "summary.json",
        {
            "original_error_concept_pairs": len(pairs),
            "eligible_pairs": eligible_pairs,
            "excluded": exclusions,
            "outcomes": outcomes,
            "interpretation": "Separate interventions per original error pair; not one global improved matching system",
        },
    )
    assert_frozen(root)
