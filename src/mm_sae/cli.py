"""The same stages are used by Docker, local runs, and batch jobs."""

import argparse
from pathlib import Path
from .config import load_recipe


def main():
    parser = argparse.ArgumentParser(prog="mm-sae")
    parser.add_argument("stage", choices=["prepare", "train-sae", "fit-mapping", "evaluate", "run"])
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--sae",
        type=Path,
        help="Existing directory with image/ and text/ save_pretrained checkpoints, or their run root",
    )
    parser.add_argument("--mapping", type=Path, help="Replace the mapping section with this YAML recipe")
    parser.add_argument("--output", type=Path, help="Override the run directory")
    parser.add_argument("--fit", type=Path, help="For evaluate: mapping directory containing fit.json")
    args = parser.parse_args()
    if (args.stage == "evaluate") != (args.fit is not None):
        parser.error("--fit is required exclusively for evaluate")
    cfg = load_recipe(args.config, args.mapping)
    if args.output:
        cfg.output = args.output.resolve()
    from .runner import run

    print(run(cfg, args.stage, args.sae, args.fit), flush=True)
