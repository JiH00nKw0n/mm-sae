import json
import sys

import numpy as np
import pytest
from scipy import sparse
import yaml

from experiments.mapping_suite import run
from experiments.mapping_suite.paired_data import PairedStudyData
from experiments.mapping_suite.report import source_explanation
from mm_sae.analysis.data import StudyData


@pytest.fixture
def paired_cache(tmp_path):
    source = tmp_path / "paired"
    rng = np.random.default_rng(42)
    for name, size, prefix in (("train2017", 10, "cc3m:"), ("val2017", 4, "coco:")):
        index = source / "index" / name
        activations = source / "activations" / name
        index.mkdir(parents=True)
        activations.mkdir(parents=True)
        (index / "images.json").write_text(json.dumps([{"image_id": f"{prefix}{i}"} for i in range(size)]))
        parents = np.repeat(np.arange(size), 1 if name == "train2017" else 2)
        np.save(index / "parents.npy", parents)
        image = rng.uniform(0, 2, (size, 3))
        text = image[parents] + rng.uniform(0, 0.1, (len(parents), 3))
        for side, values in (("image", image), ("text", text)):
            sparse.save_npz(activations / f"{side}.npz", sparse.csr_matrix(values))
    (source / "models").mkdir()
    (source / "models/frozen.json").write_text('{"training_dataset": "CC3M"}')
    return source


def split_options():
    return {"fit_and_tune": "train2017", "evaluation": "val2017", "seed": 7, "tune_image_fraction": 0.2}


def test_paired_split_keeps_parent_groups_and_matches_annotated_moments(paired_cache, tmp_path):
    paired = PairedStudyData(paired_cache, split_options())
    np.testing.assert_array_equal(paired.fit["text"], paired.fit["image"][paired.train.parents])
    np.testing.assert_array_equal(paired.tune["text"], paired.tune["image"][paired.train.parents])
    assert paired.fit["image"].sum() == 8
    assert paired.tune["image"].sum() == 2
    repeated = PairedStudyData(paired_cache, split_options())
    np.testing.assert_array_equal(paired.fit["image"], repeated.fit["image"])

    # Adding annotations must not change fit rows, selected features, or moments.
    (paired_cache / "dataset.json").write_text('{"concepts": []}')
    for name in ("train2017", "val2017"):
        index = paired_cache / "index" / name
        (index / "concept_ids.json").write_text("[]")
        np.save(index / "presence.npy", np.empty((len(json.loads((index / "images.json").read_text())), 0)))
        np.save(index / "mentions.npy", np.empty((len(np.load(index / "parents.npy")), 0)))
    annotated = StudyData(paired_cache, split_options())
    paired_out, annotated_out = tmp_path / "paired_out", tmp_path / "annotated_out"
    paired_out.mkdir()
    annotated_out.mkdir()
    paired_ids, paired_moments = run.prepare(paired, {}, paired_out)
    annotated_ids, annotated_moments = run.prepare(annotated, {}, annotated_out)
    for side in ("image", "text"):
        np.testing.assert_array_equal(paired_ids[side], annotated_ids[side])
        np.testing.assert_array_equal(paired.fit[side], annotated.fit[side])
    for partition in ("fit", "tune", "test"):
        assert paired_moments[partition].n == annotated_moments[partition].n
        np.testing.assert_array_equal(paired_moments[partition].mean, annotated_moments[partition].mean)
        np.testing.assert_array_equal(paired_moments[partition].second, annotated_moments[partition].second)


@pytest.mark.parametrize(
    "parents", [np.array([0.5, 1.5]), np.array([[0, 1]]), np.array([-1, 0]), np.array([0, 10]), np.arange(9)]
)
def test_paired_rejects_invalid_caption_parents(paired_cache, parents):
    np.save(paired_cache / "index/train2017/parents.npy", parents)
    with pytest.raises(ValueError):
        PairedStudyData(paired_cache, split_options())


@pytest.mark.parametrize("shape", [(9, 3), (10, 0)])
def test_paired_rejects_activation_shape_mismatch(paired_cache, shape):
    sparse.save_npz(paired_cache / "activations/train2017/image.npz", sparse.csr_matrix(shape))
    with pytest.raises(ValueError, match="row count or feature width"):
        PairedStudyData(paired_cache, split_options())


def test_paired_rejects_changed_feature_dictionary_width(paired_cache):
    sparse.save_npz(paired_cache / "activations/val2017/image.npz", sparse.csr_matrix((4, 2)))
    with pytest.raises(ValueError, match="feature widths differ"):
        PairedStudyData(paired_cache, split_options())


def test_paired_rejects_overlapping_image_ids(paired_cache):
    path = paired_cache / "index/val2017/images.json"
    records = json.loads(path.read_text())
    records[0]["image_id"] = "cc3m:0"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="overlap"):
        PairedStudyData(paired_cache, split_options())


def test_paired_manifest_tracks_only_required_unannotated_inputs(paired_cache, tmp_path):
    out = tmp_path / "mapping"
    out.mkdir()
    config = {"data_mode": "paired", "skip_removal": True}
    run.check_manifest(config, out, paired_cache)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["source_hashes"]) == 9
    assert str(paired_cache / "models/frozen.json") in manifest["source_hashes"]
    run.check_manifest(config, out, paired_cache)
    np.save(paired_cache / "index/train2017/parents.npy", np.arange(10)[::-1])
    with pytest.raises(ValueError, match="source data or code changed"):
        run.check_manifest(config, out, paired_cache)


def test_paired_mode_skips_removal_and_rejects_explicit_removal():
    assert not run.skips_removal({})
    assert run.skips_removal({"data_mode": "paired"})
    assert run.skips_removal({"skip_removal": True})
    with pytest.raises(ValueError, match="no removal annotations"):
        run.skips_removal({"data_mode": "paired", "skip_removal": False})
    with pytest.raises(ValueError, match="Unknown mapping data_mode"):
        run.skips_removal({"data_mode": "misspelled"})


def test_source_description_preserves_legacy_and_identifies_paired_training():
    population = {"fit_images": 8, "tune_images": 2, "test_images": 4, "test_captions": 8}
    assert "COCO 학습 이미지의 80%" in source_explanation(population, {})
    description = source_explanation(
        population,
        {
            "data_mode": "paired",
            "training_dataset": "CC3M",
            "evaluation_dataset": "COCO 2017 검증 자료",
            "source_description": "CC3M으로 SAE를 학습했습니다.",
        },
    )
    assert "CC3M으로 SAE를 학습했습니다." in description
    assert "COCO 학습 이미지의 80%" not in description
    assert "COCO 2017 검증 자료" in description


def test_paired_runner_completes_retrieval_and_report_without_annotations(
    paired_cache, tmp_path, monkeypatch
):
    out = tmp_path / "mapping"
    config = {
        "source_run": str(paired_cache),
        "output": str(out),
        "data_mode": "paired",
        "skip_removal": True,
        "seed": 7,
        "tune_image_fraction": 0.2,
        "device": "cpu",
        "retrieval_chunk": 2,
        "support_regression": False,
        "cca_penalties": [0.01],
        "cca_dimensions": [2],
        "families": [{"name": "hungarian", "method": "hungarian"}],
        "training_dataset": "CC3M",
        "evaluation_dataset": "COCO 2017 검증 자료",
        "source_description": "CC3M으로 SAE를 학습했습니다.",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["mapping_suite", "--config", str(path)])

    def forbidden_removal(*args):
        pytest.fail("Unannotated cache must never enter removal evaluation")

    monkeypatch.setattr(run, "removal_evaluation", forbidden_removal)
    run.main()
    progress = json.loads((out / "progress.json").read_text())
    assert progress["state"] == "completed"
    assert "removal" not in progress["stages_planned"]
    assert not (out / "removal").exists()
    population = json.loads((out / "population.json").read_text())
    assert population["fit_caption_pairs"] == 8
    assert population["tune_caption_pairs"] == 2
    evaluation = json.loads((out / "evaluation/hungarian.json").read_text())
    for metrics in evaluation["retrieval"].values():
        assert metrics["image_to_text"]["query_count"] == 4
        assert metrics["text_to_image"]["query_count"] == 8
    assert (out / "dense_references.json").exists()
    assert (out / "retrieval_summary.csv").exists()
    report = (out / "report.html").read_text()
    assert "CC3M으로 SAE를 학습했습니다." in report
    assert "COCO 학습 이미지의 80%" not in report
    assert "figures/removal_distribution.svg" not in report
    assert "범주 제거 평가 결과가 없습니다" in report
