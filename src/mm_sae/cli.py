"""Small experiment dispatcher. Shared modules contain no RQ-specific branches."""

import argparse
import importlib
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import json
from pathlib import Path

from .config import load_config
from .io import RunStore, atomic_json
from .progress import ProgressReporter, read_status, format_status


def main():
    parser = argparse.ArgumentParser(description="Run a configured multimodal SAE experiment")
    parser.add_argument(
        "--config", default=os.environ.get("CONFIG"), help="YAML config path (or CONFIG environment variable)"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        help="all, validate, review, status, or an experiment-defined stage",
    )
    parser.add_argument(
        "--run-dir", type=Path, help="Read status from an output directory without importing models"
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable status")
    args = parser.parse_args()
    if args.stage == "status":
        root = args.run_dir or (load_config(args.config).output if args.config else None)
        if root is None:
            parser.error("status requires --run-dir or --config")
        status = read_status(root.resolve())
        print(json.dumps(status, indent=2, ensure_ascii=False) if args.json else format_status(status))
        return
    if not args.config:
        parser.error("--config or CONFIG is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", config.experiment.name):
        raise ValueError("Experiment name must be a Python identifier")
    os.environ.setdefault("HF_HOME", str(config.cache / "huggingface"))
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    experiment = importlib.import_module(f"experiments.{config.experiment.name}.run")
    if hasattr(experiment, "validate"):
        experiment.validate(config)
    if args.stage == "validate":
        print(config.model_dump_json(indent=2))
        return
    store = RunStore(config)
    if args.stage == "review":
        destination = config.execution.approval_file
        if destination is None:
            parser.error("review requires execution.approval_file")
        request = destination.with_suffix(".request.json")
        atomic_json(
            request,
            {"decision": "pending", "signature": store.signature, "config": config.model_dump(mode="json")},
        )
        print(f"Review request saved to {request}. No approval has been recorded.")
        return
    store.require_approval()
    with store.lock():
        git = (
            subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            if shutil.which("git")
            else None
        )
        atomic_json(
            config.output / "environment.json",
            {
                "git_commit": git.stdout.strip() if git and git.returncode == 0 else None,
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in ["torch", "transformers", "datasets", "accelerate", "numpy", "scipy"]
                },
            },
        )
        stages = getattr(experiment, "STAGES", [])
        completed = [stage for stage in stages if store.done(stage)]
        with ProgressReporter(config.output, stages, completed, config.execution.progress_interval_seconds):
            experiment.run(config, store, args.stage)


if __name__ == "__main__":
    main()
