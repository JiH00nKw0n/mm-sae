"""AUROC chooses each concept's representative; correlations never enter this decision."""

from __future__ import annotations

import numpy as np

from mm_sae.data.index import Index
from mm_sae.features import counterfactual, original_latents, save_changes
from mm_sae.io import atomic_json, write_csv
from mm_sae.training import load_saes
from mm_sae.metrics.statistics import paired_auroc, select_representative


def select(config, options, encoder):
    models = load_saes(config)
    train_i, train_t = original_latents(config, options.correlation_split, models)
    alive = [train_i.getnnz(axis=0) > 0, train_t.getnnz(axis=0) > 0]
    selection = options.selection_split
    splits = list(
        dict.fromkeys(s for s in [selection, options.correlation_split, options.validation_split] if s)
    )
    representatives = {}
    rows = []
    for split in splits:
        index = Index(config.output, split)
        originals = original_latents(config, split, models)
        for concept in index.concept_ids:
            concept = int(concept)
            ir, tr, im, tx = counterfactual(config, encoder, index, split, concept, models)
            for side_no, (side, source_rows, edited) in enumerate([("image", ir, im), ("text", tr, tx)]):
                orig = originals[side_no]
                out = config.output / "counterfactual" / split / str(concept)
                source_ids = (
                    [r["image_id"] for r in index.images]
                    if side == "image"
                    else [r["caption_id"] for r in index.captions]
                )
                save_changes(out / f"{side}_changes.csv", orig, edited, source_rows, source_ids)
                row = {"split": split, "concept_id": concept, "side": side, "n_pairs": len(source_rows)}
                aucs = paired_auroc(orig[source_rows], edited) if len(source_rows) else None
                if aucs is not None:
                    np.save(out / f"{side}_auroc.npy", aucs)
                if split == selection:
                    feature, ties = (
                        select_representative(aucs, alive[side_no]) if aucs is not None else (None, 0)
                    )
                    representatives.setdefault(str(concept), {})[side] = feature
                    row["max_auc_tie_count"] = ties
                else:
                    feature = representatives[str(concept)][side]
                row["feature"] = feature
                row["auroc"] = float(aucs[feature]) if aucs is not None and feature is not None else None
                row["status"] = (
                    "selected_here"
                    if split == selection and feature is not None
                    else "fixed_feature_check"
                    if feature is not None
                    else "no_candidate"
                )
                if feature is not None and len(source_rows):
                    delta = (
                        orig[source_rows, feature].toarray().ravel() - edited[:, feature].toarray().ravel()
                    )
                    row["paired_decrease_fraction"] = float(np.mean(delta > 0))
                    row["mean_paired_decrease"] = float(delta.mean())
                    row["original_firing_fraction"] = float(orig[:, feature].getnnz() / orig.shape[0])
                rows.append(row)
    atomic_json(config.output / "representatives.json", representatives)
    write_csv(config.output / "representatives.csv", rows)
