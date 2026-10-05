"""Run a resumable, cache-only comparison of frozen image representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import sparse
from safetensors.numpy import load_file
import yaml

from mm_sae.analysis.data import StudyData, save_json
from mm_sae.analysis.evaluation import auroc, auc_interval
from mm_sae.analysis.linear import LinearScore
from mm_sae.analysis.probes import fit_probe, representation_statistics
from mm_sae.io import sha256
from mm_sae.progress import ProgressReporter, iter_progress, progress_task, stage_progress

REPRESENTATIONS = ("embedding", "reconstruction", "all_sae", "selected_five")


def selected_count(representation, legacy_count=5):
    """Resolve a fixed subset size while retaining names used by saved runs."""
    if representation == "selected_five":
        return legacy_count
    if representation in {"selected_five_positive", "single_feature"}:
        return 5 if representation == "selected_five_positive" else 1
    if representation.startswith("selected_"):
        suffix = representation.removeprefix("selected_")
        if not suffix.isdecimal() or int(suffix) < 1:
            raise ValueError(f"Invalid selected-feature representation: {representation}")
        return int(suffix)
    return None


def source_representation(representation):
    return "all_sae" if selected_count(representation) is not None else representation


def load_config(path):
    path = Path(path).resolve()
    o = yaml.safe_load(path.read_text())
    for key in ["output", "source_run", "feature_run"]:
        o[key] = str((path.parent / o[key]).resolve())
    o["original_embeddings"] = {
        s: str((path.parent / p).resolve()) for s, p in o["original_embeddings"].items()
    }
    if o["output"] in [o["source_run"], o["feature_run"]]:
        raise ValueError("Output must not overwrite a source run")
    if o["feature_count"] != 5 or set(o["representations"]) != set(REPRESENTATIONS):
        raise ValueError("This diagnostic compares all four representations and the existing top five")
    if set(o["objectives"]) != {"presence", "removal"}:
        raise ValueError("Both original presence and removal objectives are required")
    if min(o["penalties"]) <= 0 or o["bootstrap_resamples"] < 0:
        raise ValueError("Invalid regularization or bootstrap configuration")
    return o


class Diagnosis:
    def __init__(self, options):
        self.o = options
        self.out = Path(options["output"])
        self.source = Path(options["source_run"])
        self.feature_run = Path(options["feature_run"])
        self.data = StudyData(self.source, options["splits"])
        self.ids = options.get("categories") or self.data.train.ids
        self.weights = load_file(str(self.source / "models/image/model.safetensors"))
        self.sae_config = json.loads((self.source / "models/image/config.json").read_text())
        self.original = {}
        self.statistics = {}
        for split in [self.data.train, self.data.test]:
            embedding = np.load(options["original_embeddings"][split.name], mmap_mode="r")
            activations = split.activations["image"]
            if embedding.shape != (len(split.image_ids), self.weights["b_dec"].size):
                raise ValueError(f"Original embedding rows or dimension differ for {split.name}")
            self.original[split.name] = dict(
                embedding=embedding,
                all_sae=activations,
                reconstruction=np.asarray(activations @ self.weights["W_dec"] + self.weights["b_dec"]),
            )
        fit = self.data.fit["image"]
        for rep in ["embedding", "all_sae", "reconstruction"]:
            self.statistics[rep] = representation_statistics(self.original[self.data.train.name][rep][fit])

    def selected(self, cid, count=None):
        selection = json.loads((self.feature_run / "selection" / f"image_{cid}.json").read_text())
        count = self.o["feature_count"] if count is None else count
        ranking = np.asarray(selection["ranking"], int)
        if len(ranking) < count or len(np.unique(ranking)) != len(ranking):
            raise ValueError(f"Insufficient or duplicate ranked coordinates for category {cid}")
        return ranking[:count]

    def removed(self, split, cid, rep):
        root = self.source / "counterfactual" / split.name / str(cid)
        rows = np.load(root / "image_rows.npy")
        if rep == "embedding":
            values = np.load(root / "image.npy", mmap_mode="r")
        else:
            values = sparse.load_npz(root / "image_activations.npz").tocsr()
            if rep == "reconstruction":
                values = np.asarray(values @ self.weights["W_dec"] + self.weights["b_dec"])
        if len(rows) != values.shape[0]:
            raise ValueError("Masked representation row mismatch")
        return rows, values

    def samples(self, cid, representation, objective):
        rep = source_representation(representation)
        samples = []
        for split, keep in self.data.samples("image"):
            original = self.original[split.name][rep]
            if objective == "presence":
                samples.append((original[keep], split.labels("image", cid)[keep], split.image_ids[keep]))
            else:
                rows, changed = self.removed(split, cid, rep)
                selected = keep[rows]
                before, after = original[rows[selected]], changed[selected]
                x = (
                    sparse.vstack([before, after], format="csr")
                    if sparse.issparse(before)
                    else np.concatenate([before, after])
                )
                y = np.r_[np.ones(selected.sum(), bool), np.zeros(selected.sum(), bool)]
                samples.append((x, y, np.tile(split.image_ids[rows[selected]], 2)))
        return samples

    def verify(self):
        expected = json.loads((self.feature_run / "splits.json").read_text())
        actual = dict(
            fit_image_ids=self.data.train.image_ids[self.data.fit["image"]],
            tune_image_ids=self.data.train.image_ids[self.data.tune["image"]],
            evaluation_image_ids=self.data.test.image_ids,
        )
        for key, value in actual.items():
            np.testing.assert_array_equal(value, expected[key])
        save_json(self.out / "splits.json", actual)
        paths = [
            self.source / "dataset.json",
            self.source / "models/image/model.safetensors",
            self.source / "models/image/config.json",
            self.feature_run / "splits.json",
        ]
        for split in [self.data.train, self.data.test]:
            paths += [
                Path(self.o["original_embeddings"][split.name]),
                self.source / "activations" / split.name / "image.npz",
            ]
            paths += [
                self.source / "index" / split.name / f
                for f in ["images.json", "concept_ids.json", "presence.npy"]
            ]
            for cid in self.ids:
                root = self.source / "counterfactual" / split.name / str(cid)
                paths += [root / f for f in ["image.npy", "image_rows.npy", "image_activations.npz"]]
                np.testing.assert_array_equal(
                    np.load(root / "image_rows.npy"), np.flatnonzero(split.labels("image", cid))
                )
            for cid in self.ids:
                paths += [
                    self.feature_run / folder / f"image_{cid}.json" for folder in ["selection", "concepts"]
                ]
        manifest = {str(p): sha256(p) for p in iter_progress(sorted(set(paths)), "Verify frozen input files")}
        manifest_path = self.out / "input_hashes.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Inputs changed; create a new output directory")
        save_json(manifest_path, manifest)
        # Check row identity by regenerating sparse codes for deterministic sampled rows.
        checks = []
        for split in [self.data.train, self.data.test]:
            checks.append(
                (split.name + "/original", self.original[split.name]["embedding"], split.activations["image"])
            )
            for cid in self.ids:
                root = self.source / "counterfactual" / split.name / str(cid)
                checks.append(
                    (
                        f"{split.name}/{cid}",
                        np.load(root / "image.npy", mmap_mode="r"),
                        sparse.load_npz(root / "image_activations.npz"),
                    )
                )
        validation = []
        for name, embedding, codes in iter_progress(checks, "Check embedding/code correspondence"):
            if embedding.shape[0] != codes.shape[0] or embedding.shape[1] != self.weights["b_dec"].size:
                raise ValueError(f"Embedding shape mismatch at {name}")
            rows = np.unique(np.linspace(0, len(embedding) - 1, min(8, len(embedding)), dtype=int))
            if not len(rows):
                continue
            pre = np.maximum(
                (embedding[rows] - self.weights["b_dec"]) @ self.weights["encoder.weight"].T
                + self.weights["encoder.bias"],
                0,
            )
            k = self.sae_config["k"]
            indices = np.argpartition(pre, -k, axis=1)[:, -k:]
            recomputed = np.zeros_like(pre)
            np.put_along_axis(recomputed, indices, np.take_along_axis(pre, indices, axis=1), axis=1)
            error = float(np.max(np.abs(recomputed - codes[rows].toarray())))
            if error > 2e-5:
                raise ValueError(f"Embeddings do not reproduce cached codes at {name}: {error}")
            validation.append(dict(source=name, rows=len(rows), max_error=error))
        save_json(self.out / "cache_alignment.json", validation)

    def evaluate(self, cid, representation, objective, model):
        records, saved_scores = [], {}
        rep = source_representation(representation)
        original = self.original[self.data.test.name][rep]
        rows, changed = self.removed(self.data.test, cid, rep)
        original_score, removed_score = model.predict(original), model.predict(changed)
        groups = self.data.test.image_ids
        for metric in ["presence", "removal"]:
            if metric == "presence":
                y, score, group = self.data.test.labels("image", cid), original_score, groups
            else:
                y = np.r_[np.ones(len(rows), bool), np.zeros(len(rows), bool)]
                score, group = np.r_[original_score[rows], removed_score], np.tile(groups[rows], 2)
            stats = auc_interval(
                y, score, group, self.o["bootstrap_resamples"], self.o["bootstrap_seed"], return_samples=True
            )
            saved_scores[f"{metric}_bootstrap"] = np.asarray(stats.pop("_bootstrap", []))
            records.append(
                dict(
                    category_id=cid,
                    name=self.data.names[cid],
                    kind="object" if cid < 91 else "background",
                    representation=representation,
                    trained_on=objective,
                    evaluated_on=metric,
                    observations=len(y),
                    positives=int(y.sum()),
                    images=len(np.unique(group)),
                    **stats,
                )
            )
            saved_scores[f"{metric}_y"] = y
            saved_scores[f"{metric}_score"] = score
            saved_scores[f"{metric}_groups"] = group
        delta = original_score[rows] - removed_score
        records[1].update(
            paired_decrease=float(np.mean(delta > 0)) if len(delta) else None,
            paired_equal=float(np.mean(delta == 0)) if len(delta) else None,
            mean_paired_change=float(np.mean(delta)) if len(delta) else None,
        )
        if representation == "embedding" and objective == "presence":
            # Preserve the exact embedding-trained direction across SAE reconstruction.
            reconstruction = self.original[self.data.test.name]["reconstruction"]
            _, masked_reconstruction = self.removed(self.data.test, cid, "reconstruction")
            score_original, score_removed = (
                model.predict(reconstruction),
                model.predict(masked_reconstruction),
            )
            for metric in ["presence", "removal"]:
                y = saved_scores[f"{metric}_y"]
                group = saved_scores[f"{metric}_groups"]
                score = score_original if metric == "presence" else np.r_[score_original[rows], score_removed]
                stats = auc_interval(y, score, group, self.o["bootstrap_resamples"], self.o["bootstrap_seed"])
                records.append(
                    dict(
                        category_id=cid,
                        name=self.data.names[cid],
                        kind="object" if cid < 91 else "background",
                        representation="reconstruction_fixed_probe",
                        trained_on=objective,
                        evaluated_on=metric,
                        observations=len(y),
                        positives=int(y.sum()),
                        images=len(np.unique(group)),
                        **stats,
                    )
                )
                saved_scores[f"reconstruction_fixed_{metric}_score"] = score
        return records, saved_scores

    def job(self, cid, representation, objective, progress):
        key = f"{cid}_{representation}_{objective}"
        path = self.out / "jobs" / f"{key}.json"
        if path.exists():
            return
        start = time.time()
        samples = self.samples(cid, representation, objective)
        x, y, _ = samples[0]
        tx, ty, _ = samples[1]
        rep = source_representation(representation)
        count = selected_count(representation, self.o["feature_count"])
        mean, scale = self.statistics[rep]
        model, trace = fit_probe(
            x,
            y,
            tx,
            ty,
            mean,
            scale,
            self.o["penalties"],
            features=self.selected(cid, count) if count is not None else None,
            device=self.o["device"],
            maxiter=self.o["maxiter"],
            callback=lambda penalty: progress.update(progress.completed, penalty=penalty),
        )
        record: dict[str, Any] = dict(
            category_id=cid,
            representation=representation,
            objective=objective,
            fit_rows=len(y),
            tune_rows=len(ty),
            fit_positives=int(y.sum()),
            tune_positives=int(ty.sum()),
            optimization=trace,
            records=[],
        )
        if model is not None:
            records, scores = self.evaluate(cid, representation, objective, model)
            record.update(
                model=model.record(),
                records=records,
                fit_auroc=auroc(y, model.predict(x)),
                tune_auroc=auroc(ty, model.predict(tx)),
            )
            np.savez_compressed(self.out / "scores" / f"{key}.npz", **scores)
        record["seconds"] = time.time() - start
        save_json(path, record)
        print(
            json.dumps(
                dict(
                    job=key,
                    seconds=round(record["seconds"], 2),
                    auroc=[r["auroc"] for r in record["records"]],
                )
            ),
            flush=True,
        )

    def baselines(self, cid):
        record = json.loads((self.feature_run / "concepts" / f"image_{cid}.json").read_text())
        output = []
        for name, n in [("selected_five_positive", 5), ("single_feature", 1)]:
            model = LinearScore.from_record(record["models"][f"learned_{n}"])
            records, scores = self.evaluate(cid, name, "removal", model)
            # Same val cohort and scoring code must reproduce the previous experiment.
            reference = next(
                r
                for r in record["records"]
                if r["task"] == "removal" and r["condition"] == "learned" and r["n"] == n
            )
            np.testing.assert_allclose(records[1]["auroc"], reference["auroc"], atol=1e-12)
            output.extend(records)
            np.savez_compressed(self.out / "scores" / f"{cid}_{name}_removal.npz", **scores)
        save_json(self.out / "baselines" / f"{cid}.json", dict(records=output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    options = load_config(args.config)
    out = Path(options["output"])
    for folder in ["jobs", "scores", "baselines", "report"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("report.py"),
        Path(__file__).parents[2] / "src/mm_sae/analysis/probes.py",
    ]
    signature = dict(options=options, code={str(p.name): sha256(p) for p in code_paths})
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    lock = out / "signature.json"
    if lock.exists() and json.loads(lock.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Configuration or analysis code changed; choose a new output directory")
    save_json(lock, dict(fingerprint=fingerprint, **signature))
    save_json(out / "config.json", options)
    from .report import make_report

    if args.report_only:
        make_report(out)
        return
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[key] = "1"
    stages = ["verify_inputs", "linear_probes", "legacy_baselines", "report"]
    with ProgressReporter(out, stages, [], interval=10):
        with stage_progress("verify_inputs"):
            ctx = Diagnosis(options)
            ctx.verify()
        jobs = [
            (cid, rep, objective)
            for cid in ctx.ids
            for rep in options["representations"]
            for objective in options["objectives"]
        ]
        finished = sum((out / "jobs" / f"{c}_{r}_{t}.json").exists() for c, r, t in jobs)
        with (
            stage_progress("linear_probes"),
            progress_task(
                "Fit frozen-representation probes", len(jobs), "jobs", initial=finished
            ) as progress,
        ):
            for cid, rep, objective in jobs:
                if (out / "jobs" / f"{cid}_{rep}_{objective}.json").exists():
                    continue
                progress.update(
                    progress.completed,
                    category_id=cid,
                    name=ctx.data.names[cid],
                    representation=rep,
                    objective=objective,
                    penalty=None,
                )
                ctx.job(cid, rep, objective, progress)
                progress.update(progress.completed + 1)
        with stage_progress("legacy_baselines"):
            for cid in iter_progress(ctx.ids, "Reproduce previous one/five-feature results"):
                if not (out / "baselines" / f"{cid}.json").exists():
                    ctx.baselines(cid)
        with stage_progress("report"):
            make_report(out)


if __name__ == "__main__":
    main()
