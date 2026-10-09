import numpy as np
import json
from scipy import sparse

from experiments.corpus_comparison.isolate import fit_projection, score_diagnostics, subset_moments
from experiments.corpus_comparison.isolate import run
from experiments.corpus_comparison.isolation_report import run as report
from experiments.corpus_comparison.output_weighting import run as weighting_run
from mm_sae.metrics.regression import Moments


def test_subsetting_retains_native_feature_identity_and_block_order():
    values = np.arange(30, dtype=float).reshape(6, 5)
    moments = Moments(6, values.mean(0), values.T @ values / 6)
    ids = {"image": np.array([9, 2, 7]), "text": np.array([8, 3])}
    chosen = {"image": np.array([7, 9]), "text": np.array([3])}
    actual = subset_moments(moments, ids, chosen)
    expected = values[:, [2, 0, 4]]
    np.testing.assert_allclose(actual.mean, expected.mean(0))
    np.testing.assert_allclose(actual.second, expected.T @ expected / 6)


def test_diagnostics_excludes_every_positive_caption_from_negatives():
    image = np.eye(2)
    text = np.array([[1., 0.], [.8, .2], [0., 1.]])
    parents = np.array([0, 0, 1])
    result, rows = score_diagnostics(image, text, parents, chunk=1)
    assert result["image_to_text"]["strictly_beats_all_negatives"] == 1
    assert result["text_to_image"]["strictly_beats_all_negatives"] == 1
    assert result["image_to_text"]["best_positive_mean"] == 1
    for direction in result:
        total = sum(row["margin_contribution"] for row in rows if row["direction"] == direction)
        np.testing.assert_allclose(total, result[direction]["margin_mean"])


def test_correlated_pairs_can_fail_retrieval_and_ties_are_explicit():
    image = np.array([[1., 0.], [1., 0.]])
    result, _ = score_diagnostics(image, image, np.array([0, 1]))
    assert result["image_to_text"]["best_positive_mean"] == 1
    assert result["image_to_text"]["tied_best_positive_and_negative"] == 1
    assert result["image_to_text"]["strictly_beats_all_negatives"] == 0


def test_procrustes_preserves_each_modalities_gram_matrix():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(60, 7))
    moments = Moments(60, values.mean(0), values.T @ values / 60)
    wi, wt = fit_projection(moments, moments.scale, 4, "procrustes", 2, .01)
    standardized = (values - moments.mean) / moments.scale
    for x, w in ((standardized[:, :4], wi), (standardized[:, 4:], wt)):
        projected = np.einsum("ij,jk->ik", x, w)
        np.testing.assert_allclose(np.einsum("ik,jk->ij", projected, projected),
                                   np.einsum("ik,jk->ij", x, x), atol=1e-12)


def test_complete_small_factorial_and_report(tmp_path):
    rng = np.random.default_rng(13)
    source = tmp_path / "source"
    (source / "activations/val2017").mkdir(parents=True)
    (source / "index/val2017").mkdir(parents=True)
    for side, n in (("image", 8), ("text", 24)):
        sparse.save_npz(source / "activations/val2017" / f"{side}.npz",
                        sparse.csr_matrix(rng.normal(size=(n, 4))))
    np.save(source / "index/val2017/parents.npy", np.repeat(np.arange(8), 3))
    (source / "index/val2017/images.json").write_text(json.dumps([dict(image_id=i) for i in range(8)]))
    mappings = {}
    for name, image_ids, text_ids in (("cc3m", [0, 1, 2], [1, 2, 3]),
                                      ("coco", [1, 2, 3], [0, 1, 2])):
        root = tmp_path / name
        root.mkdir()
        mappings[name] = str(root)
        values = rng.normal(size=(40, 6))
        np.savez(root / "moments.npz", fit_n=40, fit_mean=values.mean(0),
                 fit_second=values.T @ values / 40, image_ids=image_ids, text_ids=text_ids)
        (root / "population.json").write_text(json.dumps(dict(test_image_ids=list(range(8)))))
    output = tmp_path / "out"
    cfg = dict(source_run=str(source), output=str(output), mapping_runs=mappings,
               methods=["cca", "cross_svd", "procrustes"], dimensions=2, ridge=.01,
               device="cpu", chunk_size=4)
    run(cfg)
    report(output)
    assert len(list(output.glob("*__center_*.json"))) == 30
    assert len(list(output.glob("*.npz"))) == 18  # Center replacement must not refit weights.
    assert (output / "controlled-differences.csv").exists()
    run(cfg)  # Resuming a complete run must not change or duplicate conditions.
    weighting_run(cfg)
    assert len(list((output / "output-weighting").glob("*__unit*.json"))) == 8
