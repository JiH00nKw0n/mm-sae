"""Run the approved comparison with persistent logs and failure-aware scheduling."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from experiments.selection_comparison.run import load_options
from mm_sae.analysis.data import save_json
from mm_sae.io import code_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    options = load_options(args.config)
    root = Path(options["output"])
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    methods = ["paired_mean_drop", "single_logistic", "probe_attribution"]
    jobs = [(m, s) for m in methods for s in ["prepare", "suite", "train_rq1", "diagnosis"]]
    jobs.append(("pooled_auroc", "train_rq1"))
    signature = dict(options=options, code=code_digest())
    signature_path = root / "scheduler_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Comparison code/config changed; choose a new output directory")
    save_json(signature_path, signature)
    state: dict[str, Any] = dict(started_at=time.time(), pid=os.getpid(), state="running", jobs={
        f"{m}/{s}": dict(state="pending") for m, s in jobs})
    lock = threading.Lock()
    failed = []

    def update(key, **values):
        with lock:
            state["jobs"][key].update(values)
            state["updated_at"] = time.time()
            save_json(root / "scheduler.json", state)

    def run(method, stage):
        key = f"{method}/{stage}"
        receipt = logs / f"{method}-{stage}.done.json"
        if receipt.exists():
            update(key, state="completed", resumed=True)
            return
        start = time.time()
        update(key, state="running", started_at=start)
        command = [sys.executable, "-m", "experiments.selection_comparison.run", "--config",
                   args.config, "--metric", method, "--stage", stage]
        with (logs / f"{method}-{stage}.log").open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        update(key, state="completed" if result.returncode == 0 else "failed",
               seconds=time.time()-start, returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(f"{key} failed, see {logs}")
        save_json(receipt, dict(command=command, seconds=time.time()-start))

    def first_suite():
        try:
            for stage in ["suite", "train_rq1", "diagnosis"]:
                run(methods[0], stage)
        except Exception as error:
            failed.append(str(error))

    try:
        run(methods[0], "prepare")
        worker = threading.Thread(target=first_suite)
        worker.start()
        # The existing suite uses CPUs. New selectors use the GPU without retraining an SAE.
        for method in methods[1:]:
            run(method, "prepare")
        worker.join()
        if failed:
            raise RuntimeError(failed[0])
        for method in methods[1:]:
            for stage in ["suite", "train_rq1", "diagnosis"]:
                run(method, stage)
        run("pooled_auroc", "train_rq1")
        state["state"] = "completed"
    except Exception as error:
        state.update(state="failed", error=str(error))
        raise
    finally:
        state["updated_at"] = time.time()
        save_json(root / "scheduler.json", state)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
