"""Selective, verified reuse of a finished run's artifacts after a lexicon change."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse

from experiments.rq1.run import run
from mm_sae.config import Config, load_config
from mm_sae.io import RunStore

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
CUSTOM_LEXICON = {
    "concepts": [
        {"id": 0, "name": "person", "aliases": ["person"]},
        {"id": 1, "name": "bicycle", "aliases": ["bicycle"]},
        # "view 0" occurs in one fifth of the synthetic captions, so some grass rows change.
        {"id": 123, "name": "grass", "aliases": ["grass", "view 0"]},
    ]
}


def make_config(tmp_path, name, concepts_file=None, reuse=None, cache="cache"):
    raw = yaml.safe_load(SMOKE.read_text())
    raw["output"] = str(tmp_path / name)
    raw["data"]["root"] = str(tmp_path / "data")
    raw["cache"] = str(tmp_path / cache)
    if concepts_file is not None:
        raw["data"]["concepts_file"] = str(concepts_file)
    if reuse is not None:
        raw["reuse"] = reuse
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def execute(config):
    store = RunStore(config)
    with store.lock():
        run(config, store, "all")
    return config.output


def items(root):
    return {i["artifact"]: i for i in json.loads((root / "reuse.json").read_text())["items"]}


def assert_same_results(a: Path, b: Path):
    for split in ["train2017", "val2017"]:
        for concept in ["0", "1", "123"]:
            for name in ["image.npy", "text.npy", "image_rows.npy", "text_rows.npy"]:
                np.testing.assert_array_equal(
                    np.load(a / "counterfactual" / split / concept / name),
                    np.load(b / "counterfactual" / split / concept / name),
                    err_msg=f"{split}/{concept}/{name}",
                )
            for name in ["image_activations.npz", "text_activations.npz"]:
                x = sparse.load_npz(a / "counterfactual" / split / concept / name)
                y = sparse.load_npz(b / "counterfactual" / split / concept / name)
                assert (x != y).nnz == 0, f"{split}/{concept}/{name}"
        for name in ["presence.npy", "full_presence.npy", "areas.npy", "mentions.npy", "parents.npy"]:
            np.testing.assert_array_equal(np.load(a / "index" / split / name), np.load(b / "index" / split / name))
        assert json.loads((a / "index" / split / "captions.json").read_text()) == json.loads(
            (b / "index" / split / "captions.json").read_text()
        )
    np.testing.assert_array_equal(np.load(a / "panel.npz")["C"], np.load(b / "panel.npz")["C"])
    assert json.loads((a / "representatives.json").read_text()) == json.loads(
        (b / "representatives.json").read_text()
    )
    assert json.loads((a / "rq1" / "experiment2" / "summary.json").read_text()) == json.loads(
        (b / "rq1" / "experiment2" / "summary.json").read_text()
    )


def test_reuse_imports_only_verified_artifacts_and_reproduces_a_fresh_run(tmp_path):
    lexicon = tmp_path / "lexicon.yaml"
    lexicon.write_text(yaml.safe_dump(CUSTOM_LEXICON))
    original = execute(make_config(tmp_path, "run-a"))
    reused = execute(
        make_config(
            tmp_path,
            "run-b",
            concepts_file=lexicon,
            cache="cache-b",
            reuse={"source_run": str(original), "source_cache": str(tmp_path / "cache")},
        )
    )
    fresh = execute(make_config(tmp_path, "run-c", concepts_file=lexicon, cache="cache-c"))
    assert_same_results(reused, fresh)
    got = items(reused)
    for split in ["train2017", "val2017"]:
        assert got[f"index/{split}/image_arrays"]["status"] == "reused"
        assert got[f"index/{split}/image_arrays"]["checks"]["image_and_mask_hashes_equal"]
        for side in ["image", "text"]:
            assert got[f"embeddings/{split}/{side}"]["status"] == "reused"
            assert got[f"activations/{split}/{side}"]["status"] == "reused"
        for concept in ["0", "1", "123"]:
            assert got[f"counterfactual/{split}/{concept}/image"]["status"] == "reused"
    assert got["models/image+text"]["status"] == "reused"
    assert got["panel.npz"]["checks"]["max_abs_difference"] == 0.0
    text = {c: got[f"counterfactual/train2017/{c}/text"] for c in ["0", "1", "123"]}
    assert text["0"]["status"] == text["1"]["status"] == "reused"
    assert text["0"]["checks"]["all_rows_identical"]
    assert text["123"]["status"] == "partially_reused"
    assert text["123"]["checks"]["rows_reusable"] > 0 and text["123"]["checks"]["rows_to_encode"] > 0
    # The grass label set changed, so the new text rows must differ from the source run.
    assert not np.array_equal(
        np.load(original / "counterfactual" / "train2017" / "123" / "text_rows.npy"),
        np.load(reused / "counterfactual" / "train2017" / "123" / "text_rows.npy"),
    )
    # Completion marks belong to the new run only and are written by its own stages.
    marks = json.loads((reused / "completed" / "select.json").read_text())
    assert marks == json.loads((reused / "run.json").read_text())["signature"]
    assert marks != json.loads((original / "run.json").read_text())["signature"]

    # A tampered source artifact is rejected and recomputed instead of being trusted.
    rows = np.load(original / "counterfactual" / "train2017" / "1" / "image_rows.npy")
    np.save(original / "counterfactual" / "train2017" / "1" / "image_rows.npy", np.append(rows, rows[-1] + 1))
    (original / "activations" / "val2017" / "text.npz").unlink()
    tampered = execute(
        make_config(
            tmp_path,
            "run-d",
            concepts_file=lexicon,
            cache="cache-d",
            reuse={"source_run": str(original), "source_cache": str(tmp_path / "cache")},
        )
    )
    assert_same_results(tampered, fresh)
    got = items(tampered)
    assert got["counterfactual/train2017/1/image"]["status"] == "recomputed"
    assert got["counterfactual/train2017/1/image"]["checks"]["image_rows_equal"] is False
    assert got["activations/val2017/text"]["status"] == "recomputed"
    assert got["counterfactual/train2017/0/image"]["status"] == "reused"


def test_reuse_rejects_the_output_directory_as_its_own_source(tmp_path):
    with pytest.raises(ValueError, match="different, finished run"):
        Config.model_validate({"output": str(tmp_path / "x"), "reuse": {"source_run": str(tmp_path / "x")}})
    with pytest.raises(FileNotFoundError, match="run.json"):
        from mm_sae.reuse import Reuse

        Reuse(Config.model_validate({"output": str(tmp_path / "x"), "reuse": {"source_run": str(tmp_path / "y")}}))
