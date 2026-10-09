"""GraHTP concept probes: all selection and fitting use training labels only."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from mm_sae.analysis.concept_pursuit import ProbeLoss, fit_pursuit
from mm_sae.analysis.concept_sets import fit_concept_sets
from mm_sae.analysis.data import CachedSplit, save_json
from mm_sae.analysis.mapping_evaluation import paired_retrieval


def within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside personal experiment folder: {path}")
    return path


def on_gpu(raw, mean, scale, device):
    x = torch.empty(raw.shape, dtype=torch.float64, device=device)
    for start in range(0, len(raw.indptr) - 1, 8192):
        a = raw[start : start + 8192].toarray().astype(np.float64)
        a = (a - mean) / scale
        if not np.isfinite(a).all():
            raise ValueError("Invalid standardized activations")
        x[start : start + len(a)].copy_(torch.as_tensor(a, device=device))
    return x


def fit_forward_ridge(x, y, weights, *, kind, penalty, budget, **kwargs):
    if kind != "mse":
        raise ValueError("Forward ridge requires MSE")
    started = time.monotonic()
    problem = ProbeLoss(x, y, weights, kind, penalty)
    mean = problem.mean.cpu().numpy()
    fitted = fit_concept_sets(
        problem.cov.cpu().numpy(), problem.cross.cpu().numpy(),
        mean, mean * (1 - mean), budgets=[16, budget], penalty=penalty,
    )
    raw = fitted.coefficients[budget]
    w = torch.as_tensor(raw, device=x.device)
    value, gw, _ = problem.full(w, problem.mean)
    restricted = float(gw[w != 0].abs().max())
    return raw, mean, dict(
        kind=kind, penalty=penalty, objective=value,
        objective_at_16=float(fitted.objective_history[:, 16].sum()),
        max_nonzeros=int(np.count_nonzero(raw, axis=0).max()),
        restricted_gradient=restricted, stop_reason="forward_budget_reached",
        seconds=time.monotonic()-started, global_optimum_certified=False,
        selection_method="forward greedy ridge", budget=budget,
        objective_history=fitted.objective_history.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg["root"]).resolve()
    if "/jihoonkwon/" not in str(root):
        raise ValueError("Expected personal experiment root")
    source = within(root / cfg["source"], root)
    parent = within(root / cfg["parent"], root)
    out = within(root / cfg["output"], root)
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for the server experiment")
    torch.set_num_threads(cfg.get("cpu_threads", 8))
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda:0"
    history_path = out / "events.jsonl"

    def status(state, **kw):
        record = dict(state=state, wall_time=time.time(), **kw)
        save_json(out / "progress.json", record)
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    status(
        "loading_data",
        gpu=torch.cuda.get_device_name(),
        gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 2**30,
    )
    train = CachedSplit(source, "train2017")
    population = json.loads((parent / "population.json").read_text())
    partitions = [set(population[k + "_image_ids"]) for k in ["fit", "tune", "test"]]
    assert all(not partitions[i] & partitions[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    keep = np.isin(train.image_ids, population["fit_image_ids"])
    assert set(train.image_ids[keep]) == partitions[0]
    captions = keep[train.parents]
    with np.load(parent / "feature_statistics.npz") as z:
        ids = {s: z[s + "_ids"] for s in ["image", "text"]}
        mean = z["fit_mean"]
        variance = z["fit_variance"]
        n = int(z["fit_n"])
    scale = np.where(variance > 0, np.sqrt(variance), 1)
    assert n == int(captions.sum())
    split = len(ids["image"])
    blocks = {"image": slice(0, split), "text": slice(split, None)}
    save_json(
        out / "protocol.json",
        dict(
            config=cfg,
            fit_images=int(keep.sum()),
            fit_captions=n,
            target="COCO image presence propagated to captions",
            support_selection=cfg.get("selection", "loss-specific column-budget GraHTP"),
            loss_normalization="MSE: half sum of per-concept squared errors; BCE: sum of per-concept BCE; CE: soft-label class sum. Then weighted sample mean.",
            regularization="0.5 * lambda * sum(W**2); intercept not penalized",
            retrieval="centered linear concept scores, no intercept/sigmoid/softmax, divided by training output standard deviation",
            zero_label_ce="excluded from soft-label CE objective, included in reported multi-label squared error",
            seed=cfg["seed"],
            dtype="float64",
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            algorithm_sha256=hashlib.sha256(
                Path(__file__).parents[2].joinpath("src/mm_sae/analysis/concept_pursuit.py").read_bytes()
            ).hexdigest(),
            global_optimum_claim=False,
            test_labels_for_training=False,
        ),
    )
    started = time.monotonic()
    for side in ["image", "text"]:
        rows = keep if side == "image" else captions
        raw = train.activations[side][rows][:, ids[side]]
        labels = np.asarray(
            train.presence[keep] if side == "image" else train.presence[train.parents[captions]],
            dtype=np.float64,
        )
        weights = (
            np.bincount(train.parents[captions], minlength=len(keep))[keep].astype(float)
            if side == "image"
            else np.ones(len(labels))
        )
        section = blocks[side]
        status("loading_gpu", side=side, shape=list(raw.shape))
        x = on_gpu(raw, mean[section], scale[section], device)
        y = torch.as_tensor(labels, device=device)
        weight = torch.as_tensor(weights, device=device)
        observed = (x * weight[:, None]).sum(0) / weight.sum()
        assert float(observed.abs().max()) < 1e-6
        for method in cfg["methods"]:
            kind, penalty = method["loss"], method["penalty"]
            name = f"{kind}_l2_{penalty:g}"
            fit_path = out / f"{name}_{side}.npz"
            if fit_path.exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            fit_start = time.monotonic()
            status("fitting", side=side, method=name)

            def progress(record):
                state = record.pop("state")
                status(
                    state,
                    side=side,
                    method=name,
                    fit_elapsed=time.monotonic() - fit_start,
                    peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30,
                    **record,
                )

            initial = None
            if cfg.get("initial_run"):
                initial_path = within(root / cfg["initial_run"] / f"ridge_0.01_{side}.npz", root)
                with np.load(initial_path) as saved:
                    initial_w = saved["raw"].copy()
                assert initial_w.shape == (x.shape[1], y.shape[1])
                assert np.isfinite(initial_w).all()
                assert (np.count_nonzero(initial_w, axis=0) <= cfg["budget"]).all()
                prevalence = np.average(labels, axis=0, weights=weights)
                clipped = np.clip(prevalence, 1e-8, 1 - 1e-8)
                intercept = (prevalence if kind == "mse" else
                             np.log(clipped / (1 - clipped)) if kind == "binary_ce" else
                             np.log(clipped))
                initial = (initial_w, intercept)
            fitter = fit_forward_ridge if cfg.get("selection") == "forward_ridge" else fit_pursuit
            w, b, audit = fitter(
                x,
                y,
                weight,
                kind=kind,
                penalty=penalty,
                budget=cfg["budget"],
                max_outer=2 if args.benchmark_only else cfg["max_outer"],
                inner_iter=cfg["inner_iter"],
                tolerance=cfg["gradient_tolerance"],
                progress=progress,
                initial=initial,
            )
            if initial is not None:
                audit["initial_file_sha256"] = hashlib.sha256(initial_path.read_bytes()).hexdigest()
                audit["support_entries_changed"] = int(np.count_nonzero((initial[0] != 0) != (w != 0)))
                audit["concepts_with_changed_support"] = int(np.any((initial[0] != 0) != (w != 0), axis=0).sum())
                assert audit["objective"] <= audit["initial_objective"] + 1e-9
            audit["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if args.benchmark_only:
                save_json(out / "benchmark.json", dict(side=side, method=name, audit=audit))
                status("benchmark_completed", seconds=time.monotonic() - started)
                return
            # Use raw coefficients for prediction; normalize only retrieval coordinates.
            wg = torch.as_tensor(w, device=device)
            total_error = torch.zeros((), device=device, dtype=x.dtype)
            total_score = torch.zeros(y.shape[1], device=device, dtype=x.dtype)
            total_square = torch.zeros_like(total_score)
            for st in range(0, len(x), 32768):
                score = x[st : st + 32768] @ wg
                wt = weight[st : st + 32768, None]
                logits = score + torch.as_tensor(b, device=device)
                pred = (
                    logits
                    if kind == "mse"
                    else logits.sigmoid()
                    if kind == "binary_ce"
                    else logits.softmax(1)
                )
                total_error += (wt * (pred - y[st : st + 32768]).square()).sum()
                total_score += (wt * score).sum(0)
                total_square += (wt * score.square()).sum(0)
            mass = weight.sum()
            pred_var = (total_square / mass - (total_score / mass).square()).clamp_min(0)
            output_scale = pred_var.sqrt().cpu().numpy()
            coefficients = np.divide(
                w, output_scale[None, :], out=np.zeros_like(w), where=output_scale[None, :] > 1e-12
            )
            audit["train_squared_error_sum"] = float(total_error / mass)
            temp = fit_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp,
                raw=w,
                intercept=b,
                coefficients=coefficients,
                output_scale=output_scale,
                feature_ids=ids[side],
                mean=mean[section],
                scale=scale[section],
            )
            os.replace(temp, fit_path)
            save_json(out / f"{name}_{side}.json", audit)
            status(
                "fit_completed",
                side=side,
                method=name,
                seconds=time.monotonic() - fit_start,
                stop_reason=audit["stop_reason"],
                restricted_gradient=audit["restricted_gradient"],
                peak_gpu_gb=audit["peak_gpu_gb"],
            )
            del wg
        del x, y, weight, raw, labels
        gc.collect()
        torch.cuda.empty_cache()
    del train
    gc.collect()
    test = CachedSplit(source, "val2017")
    np.testing.assert_array_equal(test.image_ids, population["test_image_ids"])
    for method in cfg["methods"]:
        name = f"{method['loss']}_l2_{method['penalty']:g}"
        dest = out / f"{name}_retrieval.json"
        if dest.exists():
            continue
        scores = {}
        for side in ["image", "text"]:
            with np.load(out / f"{name}_{side}.npz") as z:
                scaled = z["coefficients"] / z["scale"][:, None]
                scores[side] = (
                    np.asarray(test.activations[side][:, z["feature_ids"]] @ scaled) - z["mean"] @ scaled
                )
        status("evaluating_retrieval", method=name)
        metrics = paired_retrieval(
            scores["image"], scores["text"], test.parents, device=device, chunk_size=256
        )
        # Keep exact ranks and full metrics for subsequent reporting.
        save_json(dest, metrics)
    status("completed", seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
