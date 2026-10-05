"""Verified, selective reuse of artifacts from a previous run that shares data and frozen models.

A lexicon change forces a new output directory (RunStore refuses to continue an old one). This
module lets the new run import individual artifacts from the old run, but only after proving that
the inputs of each artifact are identical:

* image-side index arrays (presence, full_presence, areas) when every image and mask file has the
  same SHA-256 as recorded by the source run and the label scope and encoder geometry agree;
* original CLIP embeddings from the source cache when the content-addressed key is identical;
* frozen SAE weights (tensor-by-tensor equality with the source models);
* original sparse activations when the embedding file bytes and the SAE weights are identical;
* per-concept image counterfactual embeddings/activations when the masked image rows, the image
  and mask hashes, the mask colour and the encoder are identical;
* per-caption text counterfactual embeddings only for rows whose caption ID, caption text and exact
  mask token positions are unchanged; every other row is encoded again.

Completion marks are never copied: every stage still runs and records its own completion. Every
decision (reused, recomputed, rejected) is written to <output>/reuse.json with the checks made.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse

from .io import atomic_json, sha256
from .progress import iter_progress

LOG = logging.getLogger(__name__)


class Reuse:
    def __init__(self, config):
        self.config = config
        self.settings = config.reuse
        self.source: Path = config.reuse.source_run
        manifest = self.source / "run.json"
        if not manifest.exists():
            raise FileNotFoundError(f"reuse.source_run has no run.json: {self.source}")
        run = json.loads(manifest.read_text())
        self.source_config = run["config"]
        self.source_signature = run["signature"]
        cache = config.reuse.source_cache or Path(self.source_config["cache"])
        self.source_cache = Path(cache)
        self.path = config.output / "reuse.json"
        self._captions: dict[str, list] = {}
        self._images: dict[str, list] = {}

    @classmethod
    def maybe(cls, config) -> "Reuse | None":
        return cls(config) if getattr(config, "reuse", None) is not None else None

    # ----------------------------------------------------------------- provenance
    def record(self, artifact: str, status: str, checks: dict, source: Path | None = None, **extra):
        value = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"source_run": str(self.source), "source_signature": self.source_signature, "items": []}
        )
        entry = {
            "artifact": artifact,
            "status": status,
            "checks": checks,
            "source": str(source) if source is not None else None,
            **extra,
        }
        value["items"] = [i for i in value["items"] if i["artifact"] != artifact] + [entry]
        atomic_json(self.path, value)
        LOG.info("reuse %s: %s", artifact, status)
        return entry

    # ----------------------------------------------------------------- helpers
    def source_captions(self, split):
        if split not in self._captions:
            self._captions[split] = json.loads((self.source / "index" / split / "captions.json").read_text())
        return self._captions[split]

    def source_images(self, split):
        if split not in self._images:
            self._images[split] = json.loads((self.source / "index" / split / "images.json").read_text())
        return self._images[split]

    def source_dataset(self):
        return json.loads((self.source / "dataset.json").read_text())

    def encoder_matches(self, fields=("backend", "model_id", "revision", "text_max_length")):
        current = self.config.encoder.model_dump(mode="json")
        old = self.source_config["encoder"]
        return {f: (current.get(f) == old.get(f)) for f in fields}

    def weights_identical(self, side: str | None = None) -> bool:
        from .training import frozen_fingerprint

        target = self.config.output / "models" / "frozen.json"
        source = self.source / "models" / "frozen.json"
        if not target.exists() or not source.exists():
            return False
        old, new = json.loads(source.read_text()), json.loads(target.read_text())
        actual = frozen_fingerprint(self.config.output)
        if side is not None:
            return old[side] == new[side] == actual[side]
        return old == new == actual

    @staticmethod
    def _copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".reuse.tmp")
        shutil.copyfile(src, tmp)
        if sha256(tmp) != sha256(src):
            tmp.unlink()
            raise RuntimeError(f"Copy of {src} is corrupt")
        tmp.replace(dst)
        return sha256(dst)

    # ----------------------------------------------------------------- image-side index
    def image_index(self, split, images, columns, image_path, mask_path):
        """Return (presence, full_presence, areas, indexed_images) or None."""
        name = f"index/{split}/image_arrays"
        if not self.settings.image_index:
            return None
        root = self.source / "index" / split
        checks: dict = {}
        needed = ["images.json", "presence.npy", "full_presence.npy", "areas.npy", "concept_ids.json"]
        checks["source_files_present"] = all((root / n).exists() for n in needed)
        if not checks["source_files_present"] or not (self.source / "dataset.json").exists():
            self.record(name, "recomputed", checks, root, reason="source index incomplete")
            return None
        checks["concept_ids_equal"] = json.loads((root / "concept_ids.json").read_text()) == list(columns)
        split_info = self.source_dataset()["splits"].get(split, {})
        checks["label_scope_equal"] = split_info.get("label_scope") == self.config.data.label_scope
        enc = self.encoder_matches(("backend", "model_id", "revision"))
        checks["encoder_geometry_equal"] = all(enc.values())
        old = self.source_images(split)
        checks["image_ids_equal_in_order"] = [r["image_id"] for r in old] == [r["id"] for r in images]
        if not all(checks.values()):
            self.record(name, "recomputed", checks, root, reason="identity checks failed")
            return None
        indexed, mismatches = [], 0
        for i, row in enumerate(
            iter_progress(images, f"Verify image/mask hashes for reuse {split}", unit="images")
        ):
            ip, mp = image_path(row["id"]), mask_path(row["id"])
            digest_i, digest_m = sha256(ip), sha256(mp)
            if digest_i != old[i]["image_sha256"] or digest_m != old[i]["mask_sha256"]:
                mismatches += 1
                if mismatches > 5:
                    break
                continue
            indexed.append(
                {
                    "image_id": row["id"],
                    "image": str(ip),
                    "mask": str(mp),
                    "image_sha256": digest_i,
                    "mask_sha256": digest_m,
                }
            )
        checks["image_and_mask_hashes_equal"] = mismatches == 0
        if mismatches:
            self.record(name, "recomputed", checks, root, reason=f"{mismatches}+ files differ")
            return None
        presence = np.load(root / "presence.npy")
        full = np.load(root / "full_presence.npy")
        areas = np.load(root / "areas.npy")
        shape = (len(images), len(columns))
        checks["array_shapes_equal"] = presence.shape == full.shape == areas.shape == shape
        if not checks["array_shapes_equal"]:
            self.record(name, "recomputed", checks, root, reason="array shape mismatch")
            return None
        self.record(
            name,
            "reused",
            checks,
            root,
            sha256={n: sha256(root / n) for n in ["presence.npy", "full_presence.npy", "areas.npy"]},
        )
        return presence, full, areas, indexed

    # ----------------------------------------------------------------- original embeddings
    def embeddings(self, split, side, target: Path, n, dim):
        name = f"embeddings/{split}/{side}"
        if target.exists() or not self.settings.embeddings:
            return
        key = target.parent.parent.name
        source = self.source_cache / "embeddings" / key / split / f"{side}.npy"
        checks = {"content_key": key, "source_has_identical_content_key": source.exists()}
        if not source.exists():
            # The legacy cache key combines both modalities. A caption replacement changes
            # that key even when the image inputs are byte-identical.
            candidate, identity = self.modality_embedding_source(split, side)
            checks.update(identity)
            if candidate is None:
                self.record(name, "recomputed", checks, source, reason="embedding inputs changed")
                return
            source = candidate
        shape = np.load(source, mmap_mode="r").shape
        checks["shape_equal"] = shape == (n, dim)
        if not checks["shape_equal"]:
            self.record(name, "recomputed", checks, source, reason=f"shape {shape} != {(n, dim)}")
            return
        digest = self._copy(source, target)
        self.record(name, "reused", checks, source, sha256=digest)

    def modality_embedding_source(self, split, side):
        """Reuse a modality only after checking its own ordered inputs and encoder runtime."""
        from importlib.metadata import version
        from .cache import embedding_directory

        root = self.config.output / "index" / split
        fields = ("backend", "model_id", "revision", "text_max_length", "synthetic_dim")
        checks = {"modality_encoder_equal": all(self.encoder_matches(fields).values())}
        libraries = ["torch", "transformers", "numpy", "Pillow"]
        old_versions = self.source_signature.get("runtime_packages", {})
        checks["runtime_equal"] = all(
            old_versions.get(p, "").split("+")[0] == version(p).split("+")[0] for p in libraries
        )
        if side == "image":
            current = json.loads((root / "images.json").read_text())
            fields = ("image_id", "image_sha256")
            original = self.source_images(split)
        else:
            current = json.loads((root / "captions.json").read_text())
            fields = ("caption_id", "text")
            original = self.source_captions(split)
        checks["ordered_modality_inputs_equal"] = (
            [tuple(r[f] for f in fields) for r in current]
            == [tuple(r[f] for f in fields) for r in original]
        )
        if not all(checks.values()):
            return None, checks
        cfg = self.config.model_copy(deep=True)
        cfg.output, cfg.cache = self.source, self.source_cache
        candidate = embedding_directory(cfg, split) / f"{side}.npy"
        checks["source_modality_embedding_exists"] = candidate.exists()
        return (candidate if candidate.exists() else None), checks

    # ----------------------------------------------------------------- original activations
    def original_activations(self, split, side, embedding: Path, target: Path, latent_size):
        name = f"activations/{split}/{side}"
        if target.exists() or not self.settings.original_activations:
            return
        source = self.source / "activations" / split / f"{side}.npz"
        checks = {"source_exists": source.exists(), "sae_weights_identical": self.weights_identical(side)}
        if not all(checks.values()):
            self.record(name, "recomputed", checks, source, reason="missing source or different weights")
            return
        key = embedding.parent.parent.name
        old_embedding = self.source_cache / "embeddings" / key / split / embedding.name
        if not old_embedding.exists():
            candidate, identity = self.modality_embedding_source(split, side)
            checks.update(identity)
            if candidate is not None:
                old_embedding = candidate
        checks["source_embedding_exists"] = old_embedding.exists()
        if old_embedding.exists():
            checks["embedding_bytes_identical"] = old_embedding.resolve() == embedding.resolve() or sha256(
                old_embedding
            ) == sha256(embedding)
        if not checks.get("embedding_bytes_identical"):
            self.record(name, "recomputed", checks, source, reason="embedding input differs")
            return
        rows = np.load(embedding, mmap_mode="r").shape[0]
        matrix = sparse.load_npz(source)
        checks["shape_equal"] = matrix.shape == (rows, latent_size)
        if not checks["shape_equal"]:
            self.record(name, "recomputed", checks, source, reason="activation shape mismatch")
            return
        digest = self._copy(source, target)
        self.record(name, "reused", checks, source, sha256=digest)

    # ----------------------------------------------------------------- image counterfactuals
    def image_counterfactual(self, split, concept, image_rows, index, target_dir: Path, dim):
        name = f"counterfactual/{split}/{concept}/image"
        if (target_dir / "image.npy").exists() or not self.settings.image_counterfactuals:
            return
        source = self.source / "counterfactual" / split / str(concept)
        checks: dict = {
            "source_exists": (source / "image.npy").exists() and (source / "image_rows.npy").exists()
        }
        if not checks["source_exists"]:
            self.record(name, "recomputed", checks, source, reason="no source counterfactual")
            return
        old_rows = np.load(source / "image_rows.npy")
        checks["image_rows_equal"] = np.array_equal(old_rows, image_rows)
        checks["mask_rgb_equal"] = list(self.source_config["features"]["mask_rgb"]) == list(
            self.config.features.mask_rgb
        )
        checks["encoder_equal"] = all(self.encoder_matches(("backend", "model_id", "revision")).values())
        old_images = self.source_images(split)
        checks["image_and_mask_hashes_equal"] = len(old_images) == len(index.images) and all(
            old_images[int(r)]["image_sha256"] == index.images[int(r)]["image_sha256"]
            and old_images[int(r)]["mask_sha256"] == index.images[int(r)]["mask_sha256"]
            for r in image_rows
        )
        if not all(checks.values()):
            self.record(name, "recomputed", checks, source, reason="identity checks failed")
            return
        shape = np.load(source / "image.npy", mmap_mode="r").shape
        checks["shape_equal"] = shape == (len(image_rows), dim)
        if not checks["shape_equal"]:
            self.record(name, "recomputed", checks, source, reason="shape mismatch")
            return
        digests = {"image.npy": self._copy(source / "image.npy", target_dir / "image.npy")}
        checks["sae_weights_identical"] = self.weights_identical("image")
        if checks["sae_weights_identical"] and (source / "image_activations.npz").exists():
            digests["image_activations.npz"] = self._copy(
                source / "image_activations.npz", target_dir / "image_activations.npz"
            )
        self.record(name, "reused", checks, source, sha256=digests)

    # ----------------------------------------------------------------- text counterfactuals
    def text_counterfactual(self, split, concept, text_rows, index, target_dir: Path, dim):
        """Return (reusable, old_embeddings, exact) where reusable maps positions in text_rows to
        rows of old_embeddings, or None when nothing can be reused."""
        name = f"counterfactual/{split}/{concept}/text"
        if (target_dir / "text.npy").exists() or not self.settings.text_counterfactual_rows:
            return None
        source = self.source / "counterfactual" / split / str(concept)
        checks: dict = {
            "source_exists": (source / "text.npy").exists() and (source / "text_rows.npy").exists()
        }
        checks["encoder_equal"] = all(self.encoder_matches().values())
        checks["text_masking_equal"] = (
            self.source_config["features"]["text_masking"] == self.config.features.text_masking
        )
        if not all(checks.values()):
            self.record(name, "recomputed", checks, source, reason="identity checks failed")
            return None
        old_captions = self.source_captions(split)
        old_rows = np.load(source / "text_rows.npy")
        old_position = {int(r): k for k, r in enumerate(old_rows)}
        key = str(concept)
        reusable = {}
        for k, r in enumerate(text_rows):
            r = int(r)
            if r not in old_position or r >= len(old_captions):
                continue
            old, new = old_captions[r], index.captions[r]
            if (
                old["caption_id"] == new["caption_id"]
                and old["text"] == new["text"]
                and old.get("text_masking") == new.get("text_masking")
                and old["mask_token_positions"].get(key) == new["mask_token_positions"].get(key)
            ):
                reusable[k] = old_position[r]
        old_embeddings = np.load(source / "text.npy", mmap_mode="r")
        checks["source_embedding_shape_ok"] = old_embeddings.shape == (len(old_rows), dim)
        exact = bool(np.array_equal(old_rows, text_rows) and len(reusable) == len(text_rows))
        checks["rows_reusable"] = len(reusable)
        checks["rows_to_encode"] = int(len(text_rows) - len(reusable))
        checks["all_rows_identical"] = exact
        if not checks["source_embedding_shape_ok"] or not reusable:
            self.record(
                name, "recomputed", checks, source, reason="no row with identical caption and mask positions"
            )
            return None
        status = "reused" if exact else "partially_reused"
        extra = {}
        if exact and self.weights_identical("text") and (source / "text_activations.npz").exists():
            extra["sha256"] = {
                "text_activations.npz": self._copy(
                    source / "text_activations.npz", target_dir / "text_activations.npz"
                )
            }
            checks["sae_weights_identical"] = True
        self.record(name, status, checks, source, **extra)
        return reusable, old_embeddings, exact

    # ----------------------------------------------------------------- SAE weights
    def sae_checkpoints(self):
        if not self.settings.sae_weights:
            return None
        return self.source / "models" / "image", self.source / "models" / "text"

    def verify_sae_weights(self):
        import torch
        from safetensors.torch import load_file

        checks = {}
        for side in ["image", "text"]:
            a = load_file(str(self.source / "models" / side / "model.safetensors"))
            b = load_file(str(self.config.output / "models" / side / "model.safetensors"))
            checks[side] = set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
        checks["frozen_fingerprint_equal"] = self.weights_identical()
        if not all(checks.values()):
            raise RuntimeError(f"Reused SAE weights are not identical to the source run: {checks}")
        self.record("models/image+text", "reused", checks, self.source / "models")

    # ----------------------------------------------------------------- panel check
    def compare_panel(self, panel):
        source = self.source / "panel.npz"
        checks: dict = {"source_exists": source.exists()}
        if source.exists():
            old = dict(np.load(source))
            checks["shape_equal"] = old["C"].shape == panel["C"].shape
            if checks["shape_equal"]:
                checks["max_abs_difference"] = float(np.max(np.abs(old["C"] - panel["C"])))
                checks["valid_flags_equal"] = bool(
                    np.array_equal(old["valid_image"], panel["valid_image"])
                    and np.array_equal(old["valid_text"], panel["valid_text"])
                )
        self.record("panel.npz", "recomputed", checks, source, note="never copied; compared with the source")
