"""Evaluate frozen original-data probes on the two members of each edit pair."""

from collections import defaultdict
import numpy as np
from scipy.special import expit


def evaluate(scores, rows, probes):
    pairs = defaultdict(dict)
    for i, row in enumerate(rows):
        pairs[row["group"]][row["edit"]["role"]] = i
    results = []
    skipped = defaultdict(int)
    for pair, indices in pairs.items():
        if set(indices) != {"original", "edited"}:
            raise ValueError(f"Incomplete pair {pair}")
        a, b = indices["original"], indices["edited"]
        for role in ("removed", "added", "retained"):
            for concept in rows[a]["edit"][role]:
                if concept not in probes:
                    skipped[role] += 1
                    continue
                p = probes[concept]
                before, after = expit(scores[[a, b], p["axis"]].astype(float) * p["weight"] + p["bias"])
                results.append(
                    dict(
                        pair=pair,
                        role=role,
                        concept=concept,
                        before=before,
                        after=after,
                        delta=after - before,
                    )
                )
    summary = {}
    for role in ("removed", "added", "retained"):
        values = np.array([r["delta"] for r in results if r["role"] == role])
        summary[role] = {
            "count": len(values),
            "missing_probe": skipped[role],
            "mean_delta": float(values.mean()) if len(values) else None,
            "mean_absolute_delta": float(np.abs(values).mean()) if len(values) else None,
        }
    return summary, results
