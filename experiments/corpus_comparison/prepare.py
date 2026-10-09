"""Build a sequential CC3M follow-up from the same COCO experiment templates."""
from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    for name in ("model_run", "annotation_run", "annotation_population", "native_parent", "native_projection", "output"):
        cfg[name] = str((path.parent / cfg[name]).resolve())
    return cfg


def preflight(cfg):
    model, annotated = Path(cfg["model_run"]), Path(cfg["annotation_run"])
    parent, projection = Path(cfg["native_parent"]), Path(cfg["native_projection"])
    required = [model / "resolved-config.json", model / "models/frozen.json",
                annotated / "run.json", annotated / "dataset.json", Path(cfg["annotation_population"])]
    required += [parent / name for name in ("moments.npz", "population.json", "selected.json", "manifest.json")]
    required += [projection / "transforms/cca_256.npz"]
    for side in ("image", "text"):
        required += [model / "models" / side / name for name in ("model.safetensors", "config.json")]
        required += [model / "activations/val2017" / f"{side}.npz"]
    for split in ("train2017", "val2017"):
        required += [annotated / "index" / split / name for name in
                     ("images.json", "captions.json", "parents.npy", "presence.npy", "mentions.npy", "concept_ids.json")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required server files are missing:\n" + "\n".join(missing))
    return len(required)


def prepare(cfg):
    out = Path(cfg["output"])
    configs = out / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    jobs = []

    def write(name, value):
        dest = configs / f"{name}.yaml"
        if (out / "signature.json").exists() and dest.exists() and yaml.safe_load(dest.read_text()) != value:
            raise ValueError(f"Existing configuration changed; use a new output directory: {dest}")
        dest.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
        return str(dest)

    def template(name, **updates):
        value = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        value.update(updates)
        return value

    def job(name, module, *args):
        if find_spec(module) is None:
            raise ModuleNotFoundError(f"Required job module is missing before launch: {module}")
        jobs.append(dict(name=name, argv=["-m", module, *map(str, args)]))

    annotated = str(out / "cc3m-sae-coco-activations")
    cache_cfg = dict(model_run=cfg["model_run"], annotation_run=cfg["annotation_run"], output=annotated,
                     device=cfg["device"], batch_size=2048, splits=["train2017", "val2017"])
    job("annotated_cache", "experiments.corpus_comparison.cache", "--config", write("cache", cache_cfg))

    control_parent, control_projection = out / "coco-fit/mapping", out / "coco-fit/ablation"
    for prefix, native in (("cc3m-fit", True), ("coco-fit", False)):
        directory = out / prefix
        parent = cfg["native_parent"] if native else str(control_parent)
        projection = cfg["native_projection"] if native else str(control_projection)
        source = cfg["model_run"] if native else annotated
        pruning, semantics = str(directory / "pruning"), str(directory / "semantics")
        common = dict(source_run=source, parent_run=parent, output=pruning,
                      device=cfg["device"], retrieval_chunk=cfg["retrieval_chunk"])
        if not native:
            mapping = template("mapping-cc3m-server", source_run=source, output=parent,
                               data_mode="annotated", skip_removal=True,
                               training_dataset="Frozen CC3M SAE; correspondence fitted on COCO train2017",
                               source_description="CC3M SAE를 고정하고 COCO 학습 자료로 대응 행렬만 학습했습니다.")
            job(prefix + "_mapping", "experiments.mapping_suite.run", "--config", write(prefix + "-mapping", mapping))
            ablation = template("mapping-ablation-server", **{**common, "output": projection})
            job(prefix + "_ablation", "experiments.mapping_ablation.run", "--config", write(prefix + "-ablation", ablation))
        diagnostic = {**common, "projection_run": projection, "output": str(directory / "diagnostics")}
        job(prefix + "_diagnostics", "experiments.corpus_comparison.diagnose", "--config",
            write(prefix + "-diagnostics", diagnostic))
        pruning_cfg = template("mapping-pruning-local", **common, projection_run=projection)
        job(prefix + "_pruning", "experiments.mapping_pruning.run", "--config", write(prefix + "-pruning", pruning_cfg))
        sinkhorn = template("sinkhorn-pruning-local", **{**common, "output": pruning + "/sinkhorn"},
                            reference_path=None)
        job(prefix + "_sinkhorn", "experiments.mapping_pruning.run_sinkhorn", "--config", write(prefix + "-sinkhorn", sinkhorn))
        job(prefix + "_pruning_report", "experiments.mapping_pruning.report", "--run", pruning)
        job(prefix + "_sign", "experiments.mapping_semantics.fit_sign",
            "--moments", parent + "/moments.npz", "--support-dir", pruning + "/transforms",
            "--support-grid", 8, 16, "--output", semantics + "/sign-fit")
        for k in cfg["sparse_supports"]:
            job(prefix + f"_sparse_{k}", "experiments.mapping_semantics.fit_sparse",
                "--moments", parent + "/moments.npz", "--initial", projection + "/transforms/cca_256.npz",
                "--output", semantics + "/sparse-fit", "--supports", k,
                "--dimensions", 256, "--max-iter", cfg["sparse_max_iter"])
        semantics_cfg = template("mapping-semantics-local", **{**common, "source_run": annotated, "output": semantics},
                                 projection_run=projection, pruning_run=pruning, sparse_k=cfg["sparse_supports"])
        if native:
            semantics_cfg["annotation_population"] = cfg["annotation_population"]
        job(prefix + "_semantics", "experiments.mapping_semantics.run", "--config", write(prefix + "-semantics", semantics_cfg))
        job(prefix + "_semantics_report", "experiments.mapping_semantics.report", "--run", semantics)
        agreement = template("representative-agreement-local", source_run=annotated, parent_run=parent,
                             output=str(directory / "representative-agreement"))
        agreement["models"] = [
            dict(key="hungarian", label="헝가리안 일대일 대응", kind="permutation",
                 path=parent + "/candidates/hungarian_000.npz"),
            dict(key="cca_256", label="CCA 256개 가중합", kind="projection", path=pruning + "/transforms/cca_full.npz"),
            *[dict(key=f"sparse_cca_{k}", label=f"Sparse CCA 가중합마다 최대 {k}개 특징", kind="projection",
                   path=semantics + f"/sparse-fit/sparse_cca_k{k}.npz") for k in cfg["sparse_supports"]],
            dict(key="cca_full_dimensions", label="CCA의 전체 공통 좌표 ({full_dimensions}개)",
                 kind="projection", candidate_control=True,
                 path=projection + "/transforms/cca_{full_dimensions}.npz"),
        ]
        agreement["protocol"] = dict(
            training="CC3M에서 학습한 SAE를 고정하고, " + ("CC3M" if native else "COCO") + "에서 대응 계수를 학습했다.",
            population_notes=dict(tune="COCO train2017 중 고정한 20% 이미지에서 독립된 두 주석 집단을 구성했다.",
                                  test="COCO val2017 5,000개 이미지를 독립된 두 주석 집단으로 나누었다."),
            differences_from_rebuttal=["COCO-Stuff 중앙 자르기 주석과 현재 고정한 모델을 사용했다.",
                                       "모든 방법에 부호 선택을 허용하고 양의 방향만 고르는 조건도 별도 평가했다."])
        if native:
            agreement["annotation_population"] = cfg["annotation_population"]
        job(prefix + "_agreement", "experiments.representative_agreement.run", "--config", write(prefix + "-agreement", agreement))
        # Report generation is separate from fitting. Protocol text comes from the config.
        job(prefix + "_agreement_report", "experiments.representative_agreement.report", "--run", agreement["output"])

    # Annotation learning and its label-free controls must share the COCO fit rows.
    directory = out / "coco-fit"
    oracle = template("oracle-sets-local", source_run=annotated, parent_run=str(control_parent),
                      semantics_run=str(directory / "semantics"), pruning_run=str(directory / "pruning"),
                      output=str(directory / "annotation-sets"), device=cfg["device"],
                      retrieval_chunk=cfg["retrieval_chunk"])
    job("annotation_sets", "experiments.oracle_sets.run", "--config", write("annotation-sets", oracle))
    job("annotation_sets_report", "experiments.oracle_sets.report", "--run", oracle["output"])
    probes = template("concept-probe-comparison-server", source_run=annotated, parent_run=str(control_parent),
                      semantics_run=str(directory / "semantics"), reference_run=oracle["output"],
                      output=str(directory / "concept-probe-comparison"), device=cfg["device"],
                      retrieval_chunk=cfg["retrieval_chunk"])
    job("probe_comparison", "experiments.concept_probe_comparison.run", "--config", write("probe-comparison", probes))
    job("probe_report", "experiments.concept_probe_comparison.report", probes["output"])
    sequence = write("sequence", dict(output=str(out), jobs=jobs))
    write("request", cfg)
    return sequence, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    if not args.prepare_only:
        n = preflight(cfg)
        if args.check:
            print(f"Required input files exist: {n}; no jobs or output files created")
            return
    sequence, jobs = prepare(cfg)
    print(json.dumps(dict(sequence=sequence, jobs=len(jobs), experiments_started=False)))


if __name__ == "__main__":
    main()
