"""ReliefSeq against McKinney et al. (2013), Eqs. (1)-(6).

``W_{a,k} = DM_{a,k} - DH_{a,k}`` with ``DM`` / ``DH`` the mean diffs to the
``k`` nearest misses / hits (Eqs. 1-3), the range-normalised diff of Eq. (4)
and Manhattan neighbour distances, then ``W_a = max_k W_{a,k}`` (Eqs. 5-6).
The oracle evaluates every ``k`` from scratch in exact rational arithmetic,
with ReliefF's class-prior weighting for more than two classes, and never calls
FastSelect internals.
"""

from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_select import ReliefF, ReliefSeq, TuRF


def _exact_differences(X, is_discrete):
    X = np.asarray(X)
    n_samples, n_features = X.shape
    columns = []
    for f in range(n_features):
        values = [Fraction(float(v)) for v in X[:, f]]
        if is_discrete[f]:
            columns.append(values)
        else:
            low, span = min(values), max(values) - min(values)
            columns.append([(v - low) / span if span else Fraction(0) for v in values])

    def diff(i, j):
        return [
            Fraction(int(columns[f][i] != columns[f][j])) if is_discrete[f] else abs(columns[f][i] - columns[f][j])
            for f in range(n_features)
        ]

    return diff, n_samples, n_features


def reliefseq_oracle(X, y, is_discrete, k_max=None):
    """Exact ``W_{a,k}`` for ``k = 1..k_max``; returns ``(W matrix (p, k_max), weights, best k)``."""
    diff, n, p = _exact_differences(X, is_discrete)
    y = list(np.asarray(y))
    classes = sorted(set(y))
    counts = {c: y.count(c) for c in classes}
    if k_max is None:
        k_max = max(1, min(counts.values()) - 1)
    prior = {c: Fraction(counts[c], n) for c in classes}
    W = [[Fraction(0)] * k_max for _ in range(p)]
    for i in range(n):
        dist = {j: sum(diff(i, j), Fraction(0)) for j in range(n) if j != i}
        ranked = {c: sorted((j for j in dist if y[j] == c), key=lambda j: (dist[j], j)) for c in classes}
        for k in range(1, k_max + 1):
            for c in classes:
                chosen = ranked[c][:k]
                if not chosen:
                    continue
                weight = Fraction(-1) if c == y[i] else prior[c] / (1 - prior[y[i]])
                for j in chosen:
                    for f, value in enumerate(diff(i, j)):
                        W[f][k - 1] += weight * value / (len(chosen) * n)
    best = [max(range(k_max), key=lambda k: (W[f][k], -k)) for f in range(p)]
    weights = [W[f][best[f]] for f in range(p)]
    return W, np.array([float(w) for w in weights]), np.array(best) + 1


def _mixed_data(n=40, seed=0, labels=(3, 7), weights=(0.6, 0.4)):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.random(n) * 7.3 + 1.1,
            rng.random(n) * 3.7,
            rng.integers(0, 3, n),
            rng.integers(0, 2, n),
        ]
    )
    y = rng.choice(np.asarray(labels), size=n, p=np.asarray(weights))
    y[: len(labels)] = labels
    X[:, 0] += 1.5 * (y == labels[0])
    return X, y, np.array([False, False, True, True])


def _fit(X, y, **kwargs):
    kwargs.setdefault("n_features_to_select", 1)
    kwargs.setdefault("n_jobs", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ReliefSeq(**kwargs).fit(X, y)


def _assert_matches(model, W, weights, best):
    exact = np.array([[float(v) for v in row] for row in W])
    assert_allclose(model.scores_by_k_, exact, rtol=1e-11, atol=1e-14)
    assert_allclose(model.feature_importances_, weights, rtol=1e-11, atol=1e-14)
    for f, row in enumerate(W):
        ordered = sorted(row, reverse=True)
        if len(ordered) == 1 or ordered[0] - ordered[1] > Fraction(1, 10**9):
            assert model.best_k_[f] == best[f]
    assert_array_equal(model.feature_importances_, model.scores_by_k_[np.arange(len(W)), model.best_k_ - 1])


# --------------------------------------------------------------------------- equation oracle


@pytest.mark.parametrize("seed", [0, 1])
def test_binary_default_sweeps_every_possible_k(seed):
    X, y, is_discrete = _mixed_data(seed=seed)
    W, weights, best = reliefseq_oracle(X, y, is_discrete)
    model = _fit(X, y, discrete_limit=3)
    assert model.max_neighbors_ == min(np.bincount(np.unique(y, return_inverse=True)[1])) - 1
    _assert_matches(model, W, weights, best)


@pytest.mark.parametrize("k_max", [1, 3, 8])
def test_explicit_max_neighbors(k_max):
    X, y, is_discrete = _mixed_data(seed=2)
    _assert_matches(_fit(X, y, max_neighbors=k_max, discrete_limit=3), *reliefseq_oracle(X, y, is_discrete, k_max))


def test_multiclass_uses_relieff_prior_weighting():
    X, y, is_discrete = _mixed_data(n=45, seed=4, labels=(-2, 5, 11), weights=(0.45, 0.35, 0.2))
    _assert_matches(_fit(X, y, max_neighbors=6, discrete_limit=3), *reliefseq_oracle(X, y, is_discrete, 6))


def test_each_column_is_relieff_with_that_k():
    X, y, _ = _mixed_data(seed=5)
    model = _fit(X, y, max_neighbors=7, discrete_limit=3)
    for k in (1, 4, 7):
        relieff = ReliefF(n_features_to_select=1, n_neighbors=k, backend="cpu", discrete_limit=3, n_jobs=1).fit(X, y)
        assert_allclose(model.scores_by_k_[:, k - 1], relieff.feature_importances_, rtol=1e-5, atol=1e-6)


def test_max_neighbors_beyond_a_class_uses_every_member_and_warns():
    X, y, is_discrete = _mixed_data(n=30, seed=6, labels=(0, 1), weights=(0.85, 0.15))
    y[:] = 0
    y[[2, 9, 17, 25]] = 1  # four class-1 instances: three hits each
    with pytest.warns(UserWarning, match="smallest class size"):
        model = ReliefSeq(max_neighbors=6, discrete_limit=3, n_jobs=1).fit(X, y)
    _assert_matches(model, *reliefseq_oracle(X, y, is_discrete, 6))


def test_constant_feature_scores_zero_with_best_k_one():
    X, y, _ = _mixed_data(seed=7)
    X = np.column_stack([X, np.full(len(y), 2.5)])
    model = _fit(X, y, discrete_limit=3)
    assert model.feature_importances_[-1] == 0.0
    assert np.all(model.scores_by_k_[-1] == 0.0)
    assert model.best_k_[-1] == 1


# --------------------------------------------------------------------------- exact distance ties

RANGES = [3, 5, 6, 7, 9]


def _integer_case(seed, mixed):
    rng = np.random.default_rng(seed)
    n, p = int(rng.integers(6, 12)), int(rng.integers(1, 4))
    X = np.column_stack([rng.integers(0, RANGES[int(rng.integers(len(RANGES)))] + 1, n) for _ in range(p)])
    y = rng.integers(0, 2, n)
    if min(np.bincount(y, minlength=2)) < 3:
        y[:3], y[3:6] = 0, 1
    discrete = np.zeros(p, dtype=bool)
    if mixed:
        discrete[::2] = True
    return X.astype(np.float64), y, discrete


def _discrete_limit_for(X, discrete):
    if not discrete.any():
        return 0
    return max(len(np.unique(X[:, f])) for f in np.flatnonzero(discrete))


def _consistent(X, discrete):
    limit = _discrete_limit_for(X, discrete)
    return all((len(np.unique(X[:, f])) <= limit) == discrete[f] for f in range(X.shape[1]))


def _rank_tie(X, y, discrete):
    """Whether some class list has two equal distances: the lower index must rank first."""
    diff, n, _ = _exact_differences(X, discrete)
    for i in range(n):
        for c in set(y):
            d = [sum(diff(i, j), Fraction(0)) for j in range(n) if j != i and y[j] == c]
            if len(d) != len(set(d)):
                return True
    return False


def _tie_cases(wanted=12, search=2000):
    found = []
    for mixed in (False, True):
        count = 0
        for seed in range(search):
            X, y, discrete = _integer_case(seed, mixed)
            if _consistent(X, discrete) and _rank_tie(X, y, discrete):
                found.append((mixed, seed))
                count += 1
            if count == wanted:
                break
    return found


TIE_CASES = _tie_cases()


def test_search_finds_rank_ties():
    assert len(TIE_CASES) >= 20


@pytest.mark.parametrize("mixed,seed", TIE_CASES)
def test_equal_distances_keep_the_lower_index_for_every_k(mixed, seed):
    X, y, discrete = _integer_case(seed, mixed)
    model = _fit(X, y, discrete_limit=_discrete_limit_for(X, discrete))
    _assert_matches(model, *reliefseq_oracle(X, y, discrete))


# --------------------------------------------------------------------------- behaviour


def test_main_effect_and_interaction_prefer_different_k():
    """Main effects peak at large k and pure interactions at small k (the paper's Figs. 2-3)."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.integers(0, 3, (n, 12)).astype(np.float64)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    flip = rng.random(n) < 0.15
    X[:, 2] = np.where(flip, 1 - y, y) * 2.0  # main effect, 85% agreement with the class
    model = _fit(X, y, max_neighbors=60, n_features_to_select=3, n_jobs=-1)
    assert set(model.top_features_) == {0, 1, 2}
    assert model.best_k_[2] > max(model.best_k_[0], model.best_k_[1])


def test_translation_relabelling_and_label_values():
    X, y, _ = _mixed_data(seed=8)
    X[:, :2] = np.round(X[:, :2] * 1000)
    reference = _fit(X, y, max_neighbors=6, discrete_limit=3)
    shifted = X.copy()
    shifted[:, 0] += 1e8
    shifted[:, 2] += 2**25
    moved = _fit(shifted, np.where(y == 3, -1, 8), max_neighbors=6, discrete_limit=3)
    assert_array_equal(moved.scores_by_k_, reference.scores_by_k_)


def test_thread_count_only_changes_reduction_order():
    X, y, _ = _mixed_data(n=160, seed=9)
    serial = _fit(X, y, max_neighbors=20, discrete_limit=3, n_jobs=1).scores_by_k_
    parallel = _fit(X, y, max_neighbors=20, discrete_limit=3, n_jobs=-1).scores_by_k_
    assert_allclose(parallel, serial, rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------- API


def test_sklearn_api_compliance():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check_estimator(ReliefSeq(max_neighbors=2, n_jobs=1))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"backend": "gpu"}, ValueError),
        ({"max_neighbors": 0}, ValueError),
        ({"max_neighbors": 40}, ValueError),
        ({"max_neighbors": 2.0}, TypeError),
        ({"n_features_to_select": 0}, ValueError),
        ({"n_features_to_select": 1.5}, ValueError),
        ({"discrete_limit": -1}, ValueError),
        ({"n_jobs": 0}, ValueError),
    ],
)
def test_parameter_validation(kwargs, error):
    X, y, _ = _mixed_data()
    with pytest.raises(error):
        ReliefSeq(**kwargs).fit(X, y)


def test_rejects_missing_values_and_continuous_targets():
    X, y, _ = _mixed_data()
    X_missing = X.copy()
    X_missing[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ReliefSeq().fit(X_missing, y)
    with pytest.raises(ValueError, match="Unknown label type"):
        ReliefSeq().fit(X, np.linspace(0, 1, len(y)))


def test_single_class_fitted_attributes_and_transform():
    X, y, _ = _mixed_data()
    single = ReliefSeq(n_features_to_select=2).fit(X, np.full(len(y), 3))
    assert_array_equal(single.feature_importances_, np.zeros(X.shape[1]))
    model = ReliefSeq(n_features_to_select=2, discrete_limit=3)
    reduced = model.fit_transform(X, y)
    assert model.effective_backend_ == "cpu"
    assert model.scores_by_k_.shape == (X.shape[1], model.max_neighbors_)
    assert_array_equal(reduced, X[:, model.top_features_])
    with pytest.raises(ValueError):
        model.transform(X[:, :3])


def test_verbose_output(capsys):
    X, y, _ = _mixed_data()
    ReliefSeq(max_neighbors=4, verbose=True).fit(X, y)
    assert "Running ReliefSeq (k = 1..4) on the CPU" in capsys.readouterr().out


def test_in_pipeline_grid_search_and_turf():
    X, y, _ = _mixed_data(n=90, seed=11)
    pipe = Pipeline([("select", ReliefSeq(max_neighbors=10, discrete_limit=3)), ("model", LogisticRegression())])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2, 3]}, cv=3).fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )
    turf = TuRF(ReliefSeq(max_neighbors=10, discrete_limit=3), n_features_to_select=2, pct_remove=0.25).fit(X, y)
    assert turf.top_features_.size == 2
