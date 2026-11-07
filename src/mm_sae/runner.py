"""Stage orchestration; numerical methods never dispatch on dataset names."""

import fnmatch
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from mm_sae.data import fixture
from mm_sae.data.paired import _manifest, _verified_identity
from mm_sae.data.paired_embeddings import extract
from mm_sae.data.prepare import prepare, digest
from mm_sae.data.records import jsonlines
from mm_sae.encoders.clip import CLIPEncoder
from mm_sae.encoders.fixture import FixtureEncoder
from mm_sae.evaluation import concepts, edits, retrieval
from mm_sae.io import atomic_json, file_lock, write_csv, sha256
from mm_sae.mapping.model import fit, scores
from mm_sae.mapping.statistics import fit_moments
from mm_sae.sae.inference import FrozenSAEs
from mm_sae.sae.train import train_pairs


def implementation():
    root = Path(__file__).parent
    return digest([(str(p.relative_to(root)), sha256(p)) for p in sorted(root.rglob("*.py"))])


def ensure_pairs(cfg, encoder):
    if cfg.source is None:
        manifest = _manifest(cfg.paired_cache)
        _verified_identity(cfg.paired_cache, manifest)
        recorded = manifest["identity"]["encoder"]
        for key in ("backend", "model_id", "revision", "text_max_length"):
            if recorded.get(key) != getattr(cfg.encoder, key):
                raise ValueError(f"Paired cache encoder mismatch: {key}")
        return manifest
    source = dict(cfg.source)
    pattern = source.pop("shard_pattern", None)
    if pattern:
        from huggingface_hub import list_repo_files

        source["shard_paths"] = sorted(
            fnmatch.filter(
                list_repo_files(source["dataset_id"], revision=source["revision"], repo_type="dataset"),
                pattern,
            )
        )
    return extract(
        source,
        cfg.paired_cache,
        encoder_config=cfg.encoder,
        encoder=encoder,
        row_source=fixture.training_rows if cfg.smoke else None,
    )


def run(cfg, stage, pretrained=None, weights_dir=None):
    cfg.output.mkdir(parents=True, exist_ok=True)
    with file_lock(cfg.output / ".suite.lock", blocking=False):
        if cfg.smoke:
            fixture.generate(cfg.data_root)
        encoder = FixtureEncoder(cfg.encoder) if cfg.smoke else None
        # Completed cached embeddings do not require loading the CLIP weights.
        if stage in ("prepare", "run", "train-sae", "fit-mapping"):
            manifest = ensure_pairs(cfg, encoder)
            if cfg.expected_dataset and manifest["identity"]["source"]["dataset_id"] != cfg.expected_dataset:
                raise ValueError(f"Expected paired training corpus {cfg.expected_dataset}")
        tables = {}
        for name in cfg.benchmarks:
            tables[name] = cfg.output / "embeddings" / name
        if stage in ("prepare", "run"):
            encoder = encoder or CLIPEncoder(cfg.encoder)
            for name, spec in cfg.benchmarks.items():
                prepare(spec, cfg.data_root, tables[name], encoder, cfg.split_seed)
            del encoder
        if stage == "prepare":
            return cfg.output / "embeddings"
        training_root = cfg.output / "sae"
        if stage in ("train-sae", "run") and pretrained is None:
            train_pairs(cfg.paired_cache, training_root, cfg.training.model_dump(), cfg.encoder.device)
        sae_root = pretrained or training_root
        if stage == "train-sae":
            return sae_root
        saes = FrozenSAEs(sae_root, cfg.encoder.device)
        if stage == "evaluate":
            if weights_dir is None:
                raise ValueError("evaluate requires --fit pointing to the saved mapping directory")
            out = Path(weights_dir).resolve()
            receipt = json.loads((out / "fit.json").read_text())
            if receipt["sae"] != saes.hashes:
                raise ValueError("Frozen SAE differs from the one used to fit this mapping")
            if receipt["encoder"] != cfg.encoder.model_dump(exclude={"device", "batch_size"}):
                raise ValueError("Frozen encoder differs from the one used to fit this mapping")
            weights = out / "mapping.npz"
            if sha256(weights) != receipt["sha256"]:
                raise ValueError("Saved mapping weights changed")
            with np.load(weights) as packed:
                return evaluate(cfg, tables, saes, dict(packed), out)

        code = implementation()
        stats_id = digest([manifest["fingerprint"], saes.hashes, code])
        method_id = digest([stats_id, cfg.mapping.model_dump()])
        out = cfg.output / "mappings" / method_id
        out.mkdir(parents=True, exist_ok=True)
        weights = out / "mapping.npz"
        if stage in ("fit-mapping", "run") and not (out / "fit.json").exists():
            statfile = cfg.output / "statistics" / (stats_id + ".npz")
            if not statfile.exists():
                stats = fit_moments(cfg.paired_cache, saes, cfg.encoder.batch_size)
                save_arrays(statfile, stats)
            with np.load(statfile) as stats:
                arrays, meta = fit(stats, cfg.mapping, cfg.paired_cache, saes, cfg.encoder.batch_size)
            save_arrays(weights, arrays)
            atomic_json(
                out / "fit.json",
                dict(
                    id=method_id,
                    sha256=sha256(weights),
                    sae=saes.hashes,
                    source=manifest["fingerprint"],
                    implementation=code,
                    encoder=cfg.encoder.model_dump(exclude={"device", "batch_size"}),
                    config=cfg.mapping.model_dump(),
                    solver=meta,
                ),
            )
        if stage == "fit-mapping":
            return weights
        if sha256(weights) != json.loads((out / "fit.json").read_text())["sha256"]:
            raise ValueError("Saved mapping weights changed")
        with np.load(weights) as packed:
            mapping = dict(packed)
        return evaluate(cfg, tables, saes, mapping, out)


def save_arrays(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(tmp, path)


def evaluate(cfg, tables, saes, mapping, out):
    manifests = {k: json.loads((v / "manifest.json").read_text()) for k, v in tables.items()}
    for name, manifest in manifests.items():
        spec = cfg.benchmarks[name]
        expected = {
            "dataset": spec.dataset,
            "files": {k: str((cfg.data_root / v).resolve()) for k, v in spec.files.items()},
            "split_seed": cfg.split_seed,
            "encoder": cfg.encoder.model_dump(exclude={"device", "batch_size"}),
        }
        if expected != manifest["contract"]:
            raise ValueError(f"Prepared benchmark differs from evaluation recipe: {name}")
    eval_id = digest([implementation(), manifests, {k: v.model_dump() for k, v in cfg.benchmarks.items()}])
    dest = out / "evaluation" / eval_id
    if dest.is_dir():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        work = Path(tmp) / "results"
        work.mkdir()
        summary = {}
        # Edited pairs require a probe that was selected using original-data validation loss.
        order = sorted(cfg.benchmarks, key=lambda name: cfg.benchmarks[name].dataset in ("syncoco", "syncub"))
        for name in order:
            spec = cfg.benchmarks[name]
            tables_dir = tables[name]
            side_scores = {}
            rows = {}
            for side, count in manifests[name]["rows"].items():
                if not count:
                    continue
                data = np.load(tables_dir / (side + ".npy"), mmap_mode="r")
                side_scores[side] = scores(
                    saes, mapping, side, data, cfg.encoder.batch_size, Path(tmp) / (side + ".npy")
                )
                rows[side] = jsonlines(tables_dir / (side + ".jsonl"))
                if spec.dataset not in ("coco", "stuff", "visual_genome", "cub"):
                    rows[side] = list(rows[side])
                if isinstance(rows[side], list) and len(rows[side]) != len(data):
                    raise ValueError("Embedding/metadata row mismatch")
            if len(rows) == 2:
                result, ranks = retrieval.evaluate(
                    side_scores["image"], side_scores["text"], rows["image"], rows["text"]
                )
                save_arrays(
                    work / (name + "-ranks.npz"),
                    dict(
                        **ranks,
                        image_ids=[r["id"] for r in rows["image"]],
                        caption_ids=[r["id"] for r in rows["text"]],
                    ),
                )
            elif spec.dataset in ("syncoco", "syncub"):
                probe = spec.probe or ("coco" if spec.dataset == "syncoco" else "cub")
                result, pairs = edits.evaluate(
                    side_scores["image"], rows["image"], summary[probe]["concepts"]
                )
                write_csv(
                    work / (name + "-pairs.csv"),
                    pairs,
                    ["pair", "role", "concept", "before", "after", "delta"],
                )
            else:
                side = next(iter(rows))
                result, details = concepts.evaluate(
                    side_scores[side],
                    rows[side],
                    spec.min_train_positives,
                    spec.diagnostics,
                    use_cuda=cfg.encoder.device == "cuda",
                )
                save_arrays(work / (name + "-probes.npz"), details)
            summary[name] = result
            atomic_json(work / (name + ".json"), result)
        metrics = []
        for name, result in summary.items():

            def collect(prefix, obj):
                for key, value in obj.items():
                    if key in {"concepts", "excluded", "split_sizes"}:
                        continue
                    metric = f"{prefix}/{key}" if prefix else key
                    if isinstance(value, dict):
                        collect(metric, value)
                    elif isinstance(value, (int, float)) or value is None:
                        metrics.append({"benchmark": name, "metric": metric, "value": value})

            collect("", result)
        write_csv(work / "summary.csv", metrics, ["benchmark", "metric", "value"])
        atomic_json(work / "recipe.json", cfg.model_dump(mode="json"))
        atomic_json(
            work / "results.json",
            {"mapping": out.name, "synthetic": cfg.smoke, "datasets": summary, "inputs": manifests},
        )
        os.rename(work, dest)
    return dest
