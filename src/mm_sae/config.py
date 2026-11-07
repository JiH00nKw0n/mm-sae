"""Validated experiment recipes. Relative paths use the recipe's directory."""

from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EncoderConfig(Settings):
    backend: Literal["clip", "synthetic"] = "clip"
    model_id: str = "openai/clip-vit-base-patch32"
    revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: int = Field(128, ge=1)
    text_max_length: int = Field(77, ge=3)
    synthetic_dim: int = Field(16, ge=6)


class SharedObjective(Settings):
    method: Literal["group_sparse", "iso_align"]
    weight: float = Field(gt=0)


class Training(Settings):
    shared: SharedObjective | None = None
    holdout_pairs: int = Field(0, ge=0)
    preload_embeddings: bool = False
    latent_size: int = Field(4096, ge=2)
    top_k: int = Field(32, ge=1)
    seed: int = 0
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_width(self):
        if self.top_k > self.latent_size:
            raise ValueError("top_k exceeds latent_size")
        return self


class Mapping(Settings):
    method: Literal["hungarian", "pls", "cca", "sparse_pls", "sparse_cca", "shared_identity"] = "cca"
    rank: int = Field(256, ge=1)
    ridge: float = Field(1.0, ge=0)
    support: int = Field(16, ge=1)
    rotation: Literal["none", "covariance", "squared_score_covariance"] = "none"
    prune: int | None = Field(None, ge=1)
    max_iter: int = Field(500, ge=1)
    tolerance: float = Field(1e-6, gt=0)
    score_pairs: int = Field(200000, ge=2)
    seed: int = 0


class RegionSelection(Settings):
    images: Literal["all", "coco_trainval"] = "all"
    label_types: tuple[Literal["object", "attribute", "relation"], ...] = ("object", "attribute", "relation")
    annotated_only: bool = False


class Benchmark(Settings):
    dataset: Literal[
        "coco",
        "stuff",
        "visual_genome",
        "cub",
        "syncoco",
        "syncub",
        "coco_retrieval",
        "flickr30k",
        "docci",
        "iiw",
    ]
    # Probe name referenced by edited pairs; defaults to coco or cub.
    probe: str | None = None
    files: dict[str, str]
    min_train_positives: int = Field(2, ge=1)
    regions: RegionSelection | None = None
    min_test_images: int = Field(0, ge=0)
    diagnostics: bool = False

    @model_validator(mode="after")
    def check_region_selection(self):
        if self.regions is not None:
            if self.dataset != "visual_genome" or not self.regions.label_types:
                raise ValueError("Region selection requires Visual Genome and at least one label type")
            if self.regions.images == "coco_trainval" and "coco_train_instances" not in self.files:
                raise ValueError("Specify the COCO training annotation file for a trainval intersection")
        return self


class Recipe(Settings):
    concept_feature_budget: int | None = Field(None, ge=1)
    output: Path
    data_root: Path
    paired_cache: Path
    expected_dataset: str | None = None
    source: dict | None = None
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    training: Training = Field(default_factory=Training)
    mapping: Mapping = Field(default_factory=Mapping)
    benchmarks: dict[str, Benchmark]
    split_seed: int = 0
    smoke: bool = False

    @model_validator(mode="after")
    def check(self):
        if self.mapping.method == "shared_identity":
            if self.mapping.rotation != "none" or self.mapping.prune is not None:
                raise ValueError("Shared identity directly compares the two native codes")
        if self.smoke != (self.encoder.backend == "synthetic"):
            raise ValueError("Synthetic encoder and smoke fixture must be used together")
        if any(not k.isidentifier() for k in self.benchmarks):
            raise ValueError("Benchmark names must be identifiers")
        if not self.benchmarks:
            raise ValueError("Configure at least one benchmark")
        for name, spec in self.benchmarks.items():
            if spec.dataset in ("syncoco", "syncub"):
                kind = "coco" if spec.dataset == "syncoco" else "cub"
                dependency = spec.probe or kind
                if dependency not in self.benchmarks or self.benchmarks[dependency].dataset != kind:
                    raise ValueError(f"{name} needs an original {kind} probe named {dependency}")
        return self


def load_recipe(path, mapping_override=None):
    path = Path(path).resolve()
    current = path
    seen = set()
    data = {}
    while current is not None:
        if current in seen:
            raise ValueError(f"Circular recipe inheritance at {current}")
        seen.add(current)
        layer = yaml.safe_load(current.read_text())
        parent = layer.pop("extends", None)
        data = layer | data
        current = (current.parent / parent).resolve() if parent else None
    if mapping_override:
        data["mapping"] = yaml.safe_load(Path(mapping_override).read_text())
    for key in ("output", "data_root", "paired_cache"):
        data[key] = (path.parent / data[key]).resolve()
    return Recipe.model_validate(data)
