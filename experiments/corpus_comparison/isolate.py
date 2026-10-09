"""Separate feature support, fitting corpus, scale and retrieval centering.

CCA is ALWAYS fitted to covariance centered at its own fitting-corpus mean.
Changing the retrieval center is a frozen-model intervention, not a new CCA fit.
All normalization references come from training partitions, never test data.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
import yaml

from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping_ablation import fit_common_projection
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.io import sha256, write_csv
from mm_sae.metrics.regression import Moments
from mm_sae.progress import ProgressReporter, progress_task, stage_progress


def subset_moments(moments, ids, chosen):
    positions = []
    offset = 0
    for side in ("image", "text"):
        lookup = {int(value): i + offset for i, value in enumerate(ids[side])}
        positions.extend(lookup[int(value)] for value in chosen[side])
        offset += len(ids[side])
    index = np.asarray(positions)
    return Moments(moments.n, moments.mean[index], moments.second[np.ix_(index, index)])


def fit_projection(moments, scale, ni, method, dimensions, ridge):
    covariance = (moments.second - np.outer(moments.mean, moments.mean)) / np.outer(scale, scale)
    covariance = (covariance + covariance.T) / 2
    cross = covariance[:ni, ni:]
    if method == "procrustes":
        u, _, vt = linalg.svd(cross, full_matrices=True)
        width = max(cross.shape)
        return (np.pad(u, ((0, 0), (0, width - len(u)))),
                np.pad(vt.T, ((0, 0), (0, width - len(vt)))))
    projection = fit_common_projection(covariance[:ni, :ni], cross, covariance[ni:, ni:],
                                       dimensions, whiten=method == "cca", ridge=ridge)
    return projection.image, projection.text


def score_diagnostics(image, text, parents, chunk=256, device="cpu"):
    """Cosine margins against all wrong candidates, plus coordinate contributions.

    Image-to-text uses the highest-scoring positive caption. All captions of
    the same image are excluded from negatives. Zero vectors keep zero scores.
    These are descriptions of frozen test outputs, never fitting criteria.
    """
    image = np.asarray(image, dtype=np.float64)
    text = np.asarray(text, dtype=np.float64)
    parents = np.asarray(parents)
    image = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    result, components = {}, []
    for direction, queries, candidates in (("image_to_text", image, text),
                                           ("text_to_image", text, image)):
        candidate_tensor = None
        if device != "cpu":
            import torch
            candidate_tensor = torch.as_tensor(candidates, dtype=torch.float64, device=device)
        positive, negative, average_negative = [], [], []
        correct_sum = np.zeros(image.shape[1])
        wrong_sum = np.zeros_like(correct_sum)
        for start in range(0, len(queries), chunk):
            q = queries[start:start + chunk]
            if candidate_tensor is None:
                scores = q @ candidates.T
            else:
                import torch
                scores = (torch.as_tensor(q, dtype=torch.float64, device=device)
                          @ candidate_tensor.T).cpu().numpy()
            query_ids = np.arange(start, start + len(q))
            correct = (query_ids[:, None] == parents[None, :] if direction == "image_to_text"
                       else parents[query_ids, None] == np.arange(len(image))[None, :])
            if np.any(~correct.any(1)) or np.any(correct.all(1)):
                raise ValueError("Every query needs a positive and a negative candidate")
            pi = np.where(correct, scores, -np.inf).argmax(1)
            wi = np.where(correct, -np.inf, scores).argmax(1)
            rows = np.arange(len(q))
            positive.extend(scores[rows, pi])
            negative.extend(scores[rows, wi])
            average_negative.extend(np.where(correct, 0, scores).sum(1) / (~correct).sum(1))
            correct_sum += (q * candidates[pi]).sum(0)
            wrong_sum += (q * candidates[wi]).sum(0)
        p, n = np.asarray(positive), np.asarray(negative)
        margin = p - n
        result[direction] = dict(
            query_count=len(queries), best_positive_mean=float(p.mean()),
            hardest_negative_mean=float(n.mean()), average_negative_mean=float(np.mean(average_negative)),
            margin_mean=float(margin.mean()), margin_std=float(margin.std()),
            margin_quantiles=np.quantile(margin, [0, .25, .5, .75, 1]),
            strictly_beats_all_negatives=float(np.mean(margin > 0)),
            tied_best_positive_and_negative=float(np.mean(margin == 0)),
        )
        components.extend(dict(direction=direction, coordinate=i,
                               positive_contribution=float(a / len(queries)),
                               negative_contribution=float(b / len(queries)),
                               margin_contribution=float((a - b) / len(queries)))
                          for i, (a, b) in enumerate(zip(correct_sum, wrong_sum, strict=True)))
    return result, components


def run(cfg):
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    source = Path(cfg["source_run"])
    paths = [Path(p) / "moments.npz" for p in cfg["mapping_runs"].values()]
    paths += [source / "activations/val2017" / f"{side}.npz" for side in ("image", "text")]
    paths += [source / "index/val2017/parents.npy", source / "index/val2017/images.json"]
    paths += [Path(p) / "population.json" for p in cfg["mapping_runs"].values()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing server inputs: " + json.dumps(missing))
    manifest = dict(config=cfg, sources={str(p): sha256(p) for p in paths},
                    code_sha256=sha256(Path(__file__)))
    receipt = out / "manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != manifest:
        raise ValueError("Inputs/config/code changed; choose a new output directory")
    save_json(receipt, manifest)
    names = list(cfg["mapping_runs"])
    if len(names) != 2:
        raise ValueError("Exactly two fitting corpora are required")
    with ProgressReporter(out, ["load", "compare"], [], 10):
        with stage_progress("load"):
            moments, ids = {}, {}
            records = json.loads((source / "index/val2017/images.json").read_text())
            test_ids = [r["image_id"] for r in records]
            for name, path in cfg["mapping_runs"].items():
                with np.load(Path(path) / "moments.npz") as z:
                    moments[name] = Moments(int(z["fit_n"]), z["fit_mean"], z["fit_second"])
                    ids[name] = {side: z[side + "_ids"] for side in ("image", "text")}
                population = json.loads((Path(path) / "population.json").read_text())
                np.testing.assert_array_equal(population["test_image_ids"], test_ids)
            common = {side: np.intersect1d(*(ids[n][side] for n in names)) for side in ("image", "text")}
            all_values = {side: sparse.load_npz(source / "activations/val2017" / f"{side}.npz")
                          for side in ("image", "text")}
            parents = np.load(source / "index/val2017/parents.npy")
            specifications = []
            for corpus in names:
                # Reproduce native feature support with native training statistics.
                specifications.append(("native", corpus, corpus, corpus))
            specifications += [("common", *choice) for choice in itertools.product(names, repeat=3)]
            save_json(out / "protocol.json", dict(
                common_ids=common, fitting_center="Own fitting-corpus mean in EVERY model",
                factors=["feature support", "fitting corpus", "fit-and-test scale source", "test center source"],
                test_statistics_used_for_fitting=False,
                limitation="Corpus effect includes sample count and captions per image; not isolated further",
                conditions=len(specifications) * len(cfg["methods"])))
        with stage_progress("compare"):
            for support, corpus, scale_source, center_source in specifications:
                chosen = common if support == "common" else ids[corpus]
                refs = {n: subset_moments(moments[n], ids[n], chosen)
                        for n in (names if support == "common" else [corpus])}
                ni = len(chosen["image"])
                values = {side: all_values[side][:, chosen[side]].toarray() for side in ("image", "text")}
                scale, center = refs[scale_source].scale, refs[center_source].mean
                for method in cfg["methods"]:
                    fit_key = f"{support}__fit_{corpus}__scale_{scale_source}__{method}"
                    key = f"{fit_key}__center_{center_source}"
                    dest = out / f"{key}.json"
                    if dest.exists():
                        continue
                    with progress_task(key, unit="conditions"):
                        transform = out / f"{fit_key}.npz"
                        if transform.exists():
                            with np.load(transform) as z:
                                wi, wt = z["image"], z["text"]
                        else:
                            wi, wt = fit_projection(refs[corpus], scale, ni, method,
                                                    cfg["dimensions"], cfg["ridge"])
                            np.savez_compressed(transform, image=wi, text=wt)
                        image = ((values["image"] - center[:ni]) / scale[:ni]) @ wi
                        text = ((values["text"] - center[ni:]) / scale[ni:]) @ wt
                        retrieval = paired_retrieval(image, text, parents, device=cfg["device"],
                                                     chunk_size=cfg["chunk_size"])
                        diagnostics, contributions = score_diagnostics(
                            image, text, parents, cfg["chunk_size"], cfg["device"])
                        write_csv(out / f"{key}-contributions.csv", contributions)
                        save_json(dest, dict(support=support, fit_corpus=corpus, scale_source=scale_source,
                                             center_source=center_source, method=method, retrieval=retrieval,
                                             diagnostics=diagnostics))
                    print(json.dumps(dict(completed=key)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ("source_run", "output"):
        cfg[key] = str((args.config.parent / cfg[key]).resolve())
    cfg["mapping_runs"] = {n: str((args.config.parent / p).resolve()) for n, p in cfg["mapping_runs"].items()}
    run(cfg)


if __name__ == "__main__":
    main()
