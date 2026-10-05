"""Existing CLI adapter with reusable cached stages and resumable category jobs."""

import concurrent.futures
import itertools
import multiprocessing
import os

import numpy as np

from mm_sae.analysis.data import save_json, verify_source
from mm_sae.progress import progress_task, stage_progress
from .config import validate
from .context import Context, initialize_worker
from .concepts import concept_job, pair_job, select_job
from .prediction import raw_job

STAGES = ["verify_cache", "selection", "concept_readouts", "original_rq2", "shared_concepts",
          "intervention_connections", "multi_feature_rq1", "report"]


def parallel_jobs(ctx, jobs, function, folder, label):
    pending = []
    for job in jobs:
        key = "_".join(map(str, job))
        if not ctx.file(folder, key).exists():
            pending.append(job)
    workers = min(ctx.o["runtime"]["workers"], os.cpu_count() or 1)
    with progress_task(label, len(jobs), unit="jobs", initial=len(jobs)-len(pending)) as progress:
        progress.details.update(workers=workers, pending=len(pending))
        if workers == 1:
            initialize_worker(str(ctx.out), ctx.o)
            for job in pending:
                function(job)
                progress.update(progress.completed+1, last_job=list(job))
        elif pending:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker, initargs=(str(ctx.out), ctx.o)) as pool:
                futures = {pool.submit(function, job): job for job in pending}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        progress.update(progress.completed, failed_job=list(futures[future]),
                                        error=str(error))
                        for queued in futures:
                            queued.cancel()
                        raise
                    progress.update(progress.completed+1, last_job=list(futures[future]))


def prepare(ctx):
    verify_source(ctx.data.source, ctx.out, [ctx.data.train.name, ctx.data.test.name], ctx.ids)
    save_json(ctx.out / "splits.json", dict(
        fit_image_ids=ctx.data.train.image_ids[~ctx.data.tune_images],
        tune_image_ids=ctx.data.train.image_ids[ctx.data.tune_images],
        evaluation_image_ids=ctx.data.test.image_ids))
    np.savez_compressed(ctx.out / "scaling.npz",
                        **{f"{s}_mean": ctx.mean[s] for s in ctx.mean},
                        **{f"{s}_scale": ctx.scale[s] for s in ctx.scale})


def run(config, store, stage="all"):
    options = validate(config)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[key] = str(options["runtime"]["blas_threads_per_worker"])
    ctx = Context(config.output, options)
    jobs = list(itertools.product(["image", "text"], ctx.ids))
    suites = options["suites"]
    enabled = {"verify_cache", "selection", "concept_readouts", "report"}
    if "original_rq2" in suites:
        enabled.add("original_rq2")
    if "refined_rq2" in suites:
        enabled.update(["shared_concepts", "intervention_connections"])
    if "multi_feature_rq1" in suites:
        enabled.add("multi_feature_rq1")
    if stage != "all" and stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    # Revalidate hashes even when resuming, before trusting per-job files.
    if store.done("verify_cache"):
        prepare(ctx)
    for name in STAGES if stage == "all" else [stage]:
        if store.done(name):
            continue
        if name not in enabled:
            save_json(ctx.out / f"{name}_skipped.json", dict(reason="suite disabled"))
            store.complete(name)
            continue
        if name != "verify_cache":
            store.require("verify_cache")
        if name not in {"verify_cache", "selection"}:
            store.require("selection")
        if name in {"multi_feature_rq1", "report"}:
            store.require("concept_readouts")
        with stage_progress(name):
            if name == "verify_cache":
                prepare(ctx)
            elif name == "selection":
                parallel_jobs(ctx, jobs, select_job, "selection", "Select features using fit images")
            elif name == "concept_readouts":
                parallel_jobs(ctx, jobs, concept_job, "concepts", "Fit concept probes and removal readouts")
            elif name == "original_rq2":
                parallel_jobs(ctx, jobs, raw_job, "raw_prediction", "Original RQ2 activation prediction")
            elif name == "shared_concepts":
                pairs = []
                for side in ["image", "text"]:
                    for a, b in itertools.combinations(ctx.ids, 2):
                        fa, fb = ctx.features(side, a, 1), ctx.features(side, b, 1)
                        if len(fa) and len(fb) and fa[0] == fb[0]:
                            pairs.append((side, a, b))
                save_json(ctx.out / "shared_pair_inventory.json", dict(pairs=pairs))
                parallel_jobs(ctx, pairs, pair_job, "shared_pairs", "Discriminate shared-feature concepts")
            elif name == "intervention_connections":
                from .intervention import intervention_analysis
                intervention_analysis(ctx)
            elif name == "multi_feature_rq1":
                from .correspondence import analyze
                analyze(ctx)
            elif name == "report":
                from .report import report
                report(ctx)
            store.complete(name)
    save_json(ctx.out / "execution_status.json", dict(
        approved_in_conversation=True, stages={s: store.done(s) for s in STAGES},
        finished=all(store.done(s) for s in STAGES), configuration=options))
