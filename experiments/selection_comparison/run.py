"""Prepare auditable rankings, then reuse the existing experiment implementations."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from safetensors.numpy import load_file
import torch
import yaml

from experiments.feature_sets.context import Context
from mm_sae.analysis.data import save_json, verify_source
from mm_sae.analysis.feature_selection import (
    SparseUnivariateLogistic, paired_mean_drop, probe_attribution, ranked,
)
from mm_sae.analysis.linear import LinearScore
from mm_sae.analysis.probes import fit_probe, representation_statistics
from mm_sae.analysis.selection import removal_ranking
from mm_sae.config import load_config
from mm_sae.io import sha256
from mm_sae.progress import ProgressReporter, progress_task, stage_progress

METHODS = ("pooled_auroc", "paired_mean_drop", "single_logistic", "probe_attribution")


def load_options(path):
    path = Path(path).resolve()
    o = yaml.safe_load(path.read_text())
    for key in ["output", "base_config", "baseline_run", "diagnosis_config", "diagnosis_run"]:
        o[key] = str((path.parent / o[key]).resolve())
    o["embedding_directories"] = {s: str((path.parent / p).resolve())
                                  for s, p in o["embedding_directories"].items()}
    return o


def context(options, metric):
    config = load_config(options["base_config"])
    config.output = Path(options["output"]) / metric
    o = copy.deepcopy(config.experiment.options)
    o["selection"]["metric"] = metric
    if metric != "pooled_auroc":
        o["selection"]["prepared_directory"] = str(Path(options["output"]) / "rankings" / metric)
    o["selection"]["task"] = ("original_vs_category_removed" if metric in METHODS[:2]
                               else "original_category_presence")
    if options.get("categories"):
        o["smoke_categories"] = options["categories"]
    if options.get("bootstrap_resamples") is not None:
        o["uncertainty"]["resamples"] = options["bootstrap_resamples"]
    config.experiment.options = o
    config.output.mkdir(parents=True, exist_ok=True)
    return config, Context(config.output, o)


def check_embeddings(ctx, options, side):
    data = {}
    weights = load_file(str(ctx.data.source / "models" / side / "model.safetensors"))
    k = json.loads((ctx.data.source / "models" / side / "config.json").read_text())["k"]
    checks = []
    for split in [ctx.data.train, ctx.data.test]:
        path = Path(options["embedding_directories"][split.name]) / f"{side}.npy"
        x = np.load(path, mmap_mode="r")
        if x.shape[0] != split.activations[side].shape[0]:
            raise ValueError("Embedding rows differ from the frozen SAE cache")
        rows = np.unique(np.linspace(0, len(x)-1, 32, dtype=int))
        raw = np.maximum((x[rows]-weights["b_dec"]) @ weights["encoder.weight"].T
                         + weights["encoder.bias"], 0)
        ix = np.argpartition(raw, -k, axis=1)[:, -k:]
        codes = np.zeros_like(raw)
        np.put_along_axis(codes, ix, np.take_along_axis(raw, ix, axis=1), axis=1)
        error = float(np.max(np.abs(codes-split.activations[side][rows].toarray())))
        if error > 2e-5:
            raise ValueError(f"Embedding order/checkpoint mismatch: {side}, {split.name}, {error}")
        data[split.name] = x
        checks.append(dict(path=str(path), sha256=sha256(path), sampled_rows=len(rows), max_error=error))
    save_json(Path(options["output"]) / f"{side}_embedding_validation.json", checks)
    return data, weights


def prepare(options, metric):
    _, ctx = context(options, metric)
    root = Path(options["output"]) / "rankings" / metric
    root.mkdir(parents=True, exist_ok=True)
    with ProgressReporter(root, ["verify", "selection"], [], 10):
        with stage_progress("verify"):
            verify_source(ctx.data.source, root, [ctx.data.train.name, ctx.data.test.name], ctx.ids)
            expected = json.loads((Path(options["baseline_run"])/"splits.json").read_text())
            np.testing.assert_array_equal(expected["fit_image_ids"],
                                          ctx.data.train.image_ids[ctx.data.fit["image"]])
            save_json(root / "splits.json", expected)
        with stage_progress("selection"), progress_task("Select concept coordinates", 2*len(ctx.ids), "categories") as p:
            done = 0
            for side in ["image", "text"]:
                fit, tune = ctx.data.fit[side], ctx.data.tune[side]
                x, tx = ctx.data.train.activations[side][fit], ctx.data.train.activations[side][tune]
                single_fit = single_tune = None
                embedding = weights = None
                ex = et = em = es = None
                if metric == "single_logistic":
                    single_fit = SparseUnivariateLogistic(x, ctx.scale[side], ctx.alive[side], options["device"])
                    single_tune = SparseUnivariateLogistic(tx, ctx.scale[side], ctx.alive[side], options["device"])
                if metric == "probe_attribution":
                    embedding, weights = check_embeddings(ctx, options, side)
                    ex, et = embedding[ctx.data.train.name][fit], embedding[ctx.data.train.name][tune]
                    em, es = representation_statistics(ex)
                for cid in ctx.ids:
                    path = root / f"{side}_{cid}.json"
                    p.update(done, side=side, category_id=cid)
                    if path.exists():
                        done += 1
                        p.update(done)
                        continue
                    start = time.time()
                    labels = ctx.data.train.labels(side, cid)
                    y, ty = labels[fit], labels[tune]
                    rows, changed = ctx.data.train.removed(side, cid)
                    keep = fit[rows]
                    original, removed = ctx.data.train.activations[side][rows[keep]], changed[keep]
                    _, auc = removal_ranking(original, removed, ctx.alive[side])
                    extra = {}
                    if metric == "paired_mean_drop":
                        scores = paired_mean_drop(original, removed)
                        ranking = ranked(scores, ctx.alive[side])
                        extra = dict(selection_score=scores[ranking])
                    elif metric == "single_logistic":
                        assert single_fit is not None and single_tune is not None
                        extra = single_fit.fit_rank(y, single_tune, ty, ctx.penalties,
                            callback=lambda v: p.update(done, penalty=v))
                        ranking = np.asarray(extra.pop("ranking"), int)
                    elif metric == "probe_attribution":
                        assert weights is not None and em is not None and es is not None
                        cached = Path(options["diagnosis_run"])/"jobs"/f"{cid}_embedding_presence.json"
                        if side == "image" and cached.exists():
                            saved = json.loads(cached.read_text())
                            model = LinearScore.from_record(saved["model"])
                            np.testing.assert_allclose(model.mean, em[model.features], atol=1e-9)
                            np.testing.assert_allclose(model.scale, es[model.features], atol=1e-9)
                            trace = dict(reused=str(cached), sha256=sha256(cached))
                        else:
                            model, trace = fit_probe(ex, y, et, ty, em, es, ctx.penalties,
                                device=options["device"], maxiter=10000,
                                callback=lambda v: p.update(done, penalty=v))
                        if model is None:
                            scores = np.full(x.shape[1], np.nan)
                        else:
                            direction = np.zeros(weights["W_dec"].shape[1])
                            direction[model.features] = model.weight/model.scale
                            scores = probe_attribution(x, y, weights["W_dec"], direction)
                        ranking = ranked(scores, ctx.alive[side])
                        extra = dict(selection_score=scores[ranking], probe=model.record() if model else None,
                                     optimization=trace)
                    else:
                        raise ValueError(metric)
                    record = dict(**ctx.metadata(side, cid), metric=metric, ranking=ranking,
                        auroc=auc[ranking], fit_pairs=int(keep.sum()), fit_positives=int(y.sum()),
                        tune_positives=int(ty.sum()), seconds=time.time()-start, **extra)
                    save_json(path, record)
                    done += 1
                    p.update(done)
                    print(json.dumps(dict(metric=metric, side=side, cid=cid,
                                          seconds=round(time.time()-start, 2), top=ranking[:5].tolist())), flush=True)
                del single_fit, single_tune, ex, et, embedding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def suite(options, metric):
    config, _ = context(options, metric)
    destination = Path(options["output"]) / f"{metric}.yaml"
    destination.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    subprocess.run([sys.executable, "-m", "mm_sae", "--config", str(destination), "all"], check=True)
    subprocess.run([sys.executable, "scripts/summarize_feature_sets.py", str(config.output)], check=True)


def train_rq1(options, metric):
    _, ctx = context(options, metric)
    if metric == "pooled_auroc":
        ctx.out = Path(options["baseline_run"])
    from experiments.feature_sets.correspondence import analyze
    output = Path(options["output"]) / metric
    # Read immutable fitted models from their original directory and write only new analyses.
    original_out = ctx.out
    original_readout, original_file = ctx.readout, ctx.file
    ctx.readout = lambda side, cid, n, condition="learned": original_readout(side, cid, n, condition)
    # Context.file resolves self.out dynamically; keep it bound to the source for model reads.
    ctx.file = lambda group, key: original_out / group / f"{key}.json"
    ctx.out = output
    output.mkdir(parents=True, exist_ok=True)
    ctx.o["multi_feature_rq1"]["legacy_n1_reproduction"] = False
    (output / "train_rq1_progress").mkdir(exist_ok=True)
    with ProgressReporter(output / "train_rq1_progress", ["train_rq1"], [], 10):
        with stage_progress("train_rq1"):
            analyze(ctx, split=ctx.data.train, folder="correspondence_train", counts=[1], conditions=["learned"])
    ctx.file = original_file


def diagnosis(options, metric):
    from experiments.representation_diagnosis.run import Diagnosis, load_config as load_diagnosis
    from experiments.representation_diagnosis.report import make_report

    o = load_diagnosis(options["diagnosis_config"])
    out = Path(options["output"]) / metric / "diagnosis"
    o.update(output=str(out), feature_run=str(Path(options["output"]) / metric), device=options["device"])
    if options.get("categories"):
        o["categories"] = options["categories"]
    if options.get("bootstrap_resamples") is not None:
        o["bootstrap_resamples"] = options["bootstrap_resamples"]
    for group in ["jobs", "scores", "baselines", "report"]:
        (out / group).mkdir(parents=True, exist_ok=True)
    with ProgressReporter(out, ["verify", "selected_probes", "report"], [], 10):
        with stage_progress("verify"):
            ctx = Diagnosis(o)
            ctx.verify()
            reuse = {}
            for cid in ctx.ids:
                for rep in ["embedding", "reconstruction", "all_sae"]:
                    for task in ["presence", "removal"]:
                        key = f"{cid}_{rep}_{task}.json"
                        source = Path(options["diagnosis_run"])/"jobs"/key
                        shutil.copyfile(source, out/"jobs"/key)
                        reuse[key] = sha256(source)
                        score = Path(options["diagnosis_run"])/"scores"/key.replace(".json", ".npz")
                        if score.exists():
                            shutil.copyfile(score, out/"scores"/score.name)
                            reuse[f"scores/{score.name}"] = sha256(score)
            save_json(out / "reused_unchanged_probes.json", reuse)
            save_json(out / "config.json", o)
        with stage_progress("selected_probes"), progress_task("Evaluate selected image coordinates", 2*len(ctx.ids), "jobs") as p:
            for i, cid in enumerate(ctx.ids):
                for j, task in enumerate(["presence", "removal"]):
                    p.update(2*i+j, category_id=cid, objective=task)
                    ctx.job(cid, "selected_five", task, p)
                    p.update(2*i+j+1)
                ctx.baselines(cid)
        with stage_progress("report"):
            make_report(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metric", required=True, choices=METHODS)
    parser.add_argument("--stage", choices=["prepare", "suite", "train_rq1", "diagnosis"], required=True)
    args = parser.parse_args()
    options = load_options(args.config)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = "1"
    torch.set_num_threads(1)
    globals()[args.stage](options, args.metric)


if __name__ == "__main__":
    main()
