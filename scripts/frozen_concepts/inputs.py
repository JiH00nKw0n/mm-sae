"""Load frozen representations and the annotated populations used by the 11-method report."""

import hashlib
import json
import numpy as np
from scipy import sparse


def source(work, ds):
    if ds != "visual_genome":
        pack = work / "packs" / ds
        t = np.load(pack / "truth.npz")
        return sparse.load_npz(pack / "image.npz"), sparse.csr_matrix(t["labels"]), t["concepts"], t["split"]
    scope = work / "vg-coco2017"
    m = np.load(scope / "population.npz")
    sel = m["rows"]
    blocks = []
    start = 0
    for p in sorted((work / "packs/visual_genome").glob("part-*/text.npz")):
        X = sparse.load_npz(p)
        stop = start + X.shape[0]
        lo, hi = np.searchsorted(sel, [start, stop])
        if hi > lo:
            blocks.append(X[sel[lo:hi] - start])
        start = stop
    return (
        sparse.vstack(blocks, format="csr"),
        sparse.load_npz(scope / "labels.npz"),
        m["concepts"],
        np.array(["train", "val", "test"])[m["splits"]],
    )


def coefficients(work, method, side, expected_digest):
    if method in ("sinkhorn", "procrustes"):
        weights = np.load(work / "full32" / (method + ".npy"))
        if side == "image":
            weights = weights.T
    else:
        matches = []
        for p in list((work / "results/mappings").glob("*/fit.json")) + list(
            (work / "extra-fits").glob("*/fit.json")
        ):
            meta = json.loads(p.read_text())
            if meta.get("condition", meta["config"]["method"]) == method:
                if method == "cca" and meta["config"]["rotation"] != "none":
                    continue
                matches.append(p)
        (p,) = matches
        weights = np.load(p.parent / "mapping.npz")[side]
    digest = hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"Unexpected frozen coefficients: {method}/{side}")
    with np.load(work / "full32/scales.npz") as scales:
        return weights * scales[side + "_inv"][:, None]
