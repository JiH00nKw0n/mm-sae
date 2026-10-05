import numpy as np
from scipy import sparse

from mm_sae.analysis.evaluation import auroc
from mm_sae.analysis.linear import fit_logistic
from mm_sae.analysis.probes import fit_probe, representation_statistics


def test_probe_matches_existing_objective_and_nonnegative_constraint():
    rng = np.random.default_rng(42)
    x = rng.choice([0.0, 0.0, 0.0, 1.0, 2.0], (300, 4))
    y = x[:, 0] + x[:, 1] - 0.8 * x[:, 2] + rng.normal(size=300) > 1
    mean, scale = representation_statistics(sparse.csr_matrix(x[:200]))
    for nonnegative in [False, True]:
        expected = fit_logistic(
            x[:200], y[:200], x[200:], y[200:], np.arange(4), mean, scale, [0.01, 0.1, 1.0], nonnegative
        )
        actual, trace = fit_probe(
            sparse.csr_matrix(x[:200]),
            y[:200],
            x[200:],
            y[200:],
            mean,
            scale,
            [0.01, 0.1, 1.0],
            nonnegative=nonnegative,
        )
        assert expected is not None and actual is not None
        np.testing.assert_allclose(expected.predict(x), actual.predict(x), atol=1e-8)
        assert expected.penalty == actual.penalty
        assert all(r["success"] for r in trace["penalties"])


def test_zero_fit_column_cannot_leak_tuning_only_information():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    tx = x.copy()
    tx[:, 1] = [0.0, 0.0, 100.0, 100.0]
    mean, scale = representation_statistics(x)
    model, _ = fit_probe(sparse.csr_matrix(x), [0, 0, 1, 1], tx, [0, 0, 1, 1], mean, scale, [0.1])
    assert model is not None
    np.testing.assert_array_equal(model.features, [0])
    np.testing.assert_allclose(model.predict(tx), model.predict(x))


def test_missing_tuning_class_is_not_evaluated_as_a_valid_model():
    x = np.array([[0.0], [1.0]])
    model, trace = fit_probe(x, [0, 1], x, [1, 1], np.zeros(1), np.ones(1), [0.1])
    assert model is None and trace["status"] == "missing_fit_or_tune_class"


def test_all_features_can_recover_signal_lost_by_selection_or_reconstruction():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(1000, 3))
    y = x[:, 1] + x[:, 2] > 0
    mean, scale = representation_statistics(x[:600])
    full, _ = fit_probe(x[:600], y[:600], x[600:800], y[600:800], mean, scale, [0.01, 0.1])
    chosen, _ = fit_probe(x[:600], y[:600], x[600:800], y[600:800], mean, scale, [0.01, 0.1], features=[0])
    assert full is not None and chosen is not None
    assert auroc(y[800:], full.predict(x[800:])) > 0.98
    assert auroc(y[800:], chosen.predict(x[800:])) < 0.65


def test_selected_subsets_preserve_the_existing_ranking_and_are_nested(tmp_path):
    import json
    from experiments.representation_diagnosis.run import Diagnosis, selected_count, source_representation

    ranking = np.random.default_rng(7).permutation(80).tolist()
    folder = tmp_path / 'selection'
    folder.mkdir()
    (folder / 'image_105.json').write_text(json.dumps({'ranking': ranking}))
    ctx = Diagnosis.__new__(Diagnosis)
    ctx.feature_run = tmp_path
    ctx.o = {'feature_count': 5}
    np.testing.assert_array_equal(ctx.selected(105), ranking[:5])
    for n in [8, 16, 64]:
        rep = f'selected_{n}'
        assert selected_count(rep) == n
        assert source_representation(rep) == 'all_sae'
        np.testing.assert_array_equal(ctx.selected(105, n), ranking[:n])
    import pytest
    with pytest.raises(ValueError, match='Insufficient'):
        ctx.selected(105, 81)


def test_expanding_a_fixed_subset_recovers_signal_in_a_later_ranked_coordinate():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1200, 64))
    y = x[:, 12] + .1*rng.normal(size=len(x)) > 0
    mean, scale = representation_statistics(x[:700])
    scores = {}
    for n in [8, 16, 64]:
        model, _ = fit_probe(x[:700], y[:700], x[700:900], y[700:900], mean, scale,
                             [.001, .01, .1], features=np.arange(n))
        assert model is not None
        scores[n] = auroc(y[900:], model.predict(x[900:]))
    assert scores[8] < .65 and scores[16] > .97 and scores[64] > .95
