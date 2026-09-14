"""Small experiment dispatcher. Shared modules contain no RQ-specific branches."""

import argparse
import importlib
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess

from .config import load_config
from .io import RunStore, atomic_json


def main():
    parser = argparse.ArgumentParser(description="Run a configured multimodal SAE experiment")
    parser.add_argument(
        "--config", default=os.environ.get("CONFIG"), help="YAML config path (or CONFIG environment variable)"
    )
    parser.add_argument(
        "stage", nargs="?", default="all", help="all, validate, or an experiment-defined stage"
    )
    args = parser.parse_args()
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
        experiment.run(config, store, args.stage)


if __name__ == "__main__":
    main()
