"""VLS (VLSReliefF) against its definition.

Eppstein and Haake (2008), as described by Urbanowicz et al. (2018), Section
3.4: ReliefF on random feature subsets, each feature's global weight being its
maximum subset weight.  The oracle re-scores every recorded subset with an
independent, exact-rational ReliefF and takes the maximum; it never calls
FastSelect internals.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_select import STIR, SURF, SWRF, VLS, MultiSURF, MultiSWRF, MuRelief, ReliefF, ReliefSeq, TuRF


def relieff_oracle(X, y, is_discrete, k):
    """ReliefF (k nearest hits and per-class misses, class priors), exact."""
    X = np.asarray(X)
    n, p = X.shape
    columns = []
    for f in range(p):
        values = [Fraction(float(v)) for v in X[:, f]]
        if is_discrete[f]:
            columns.append(values)
        else:
            low, span = min(values), max(values) - min(values)
            columns.append([(v - low) / span if span else Fraction(0) for v in values])

    def diff(i, j):
        return [
            Fraction(int(columns[f][i] != columns[f][j])) if is_discrete[f] else abs(columns[f][i] - columns[f][j])
            for f in range(p)
        ]

    y = list(np.asarray(y))
    classes = sorted(set(y))
    prior = {c: Fraction(y.count(c), n) for c in classes}
    scores = [Fraction(0)] * p
    for i in range(n):
        dist = {j: sum(diff(i, j), Fraction(0)) for j in range(n) if j != i}
        for c in classes:
            chosen = sorted((j for j in dist if y[j] == c), key=lambda j: (dist[j], j))[:k]
            if not chosen:
                continue
            weight = Fraction(-1) if c == y[i] else prior[c] / (1 - prior[y[i]])
            for j in chosen:
                for f, value in enumerate(diff(i, j)):
                    scores[f] += weight * value / (len(chosen) * n)
    return np.array([float(s) for s in scores])


def _data(n=40, p=9, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.random(n) * (3.0 + f) for f in range(p - 2)] + [rng.integers(0, 3, (n, 2))])
    y = (X[:, 0] / 3.0 + 0.4 * rng.random(n) > 0.7).astype(int) * 5 + 2  # labels 2 and 7
    return X, y, np.array([False] * (p - 2) + [True, True])


# --------------------------------------------------------------------------- definition


@pytest.mark.parametrize("size,count", [(3, None), (4, 5), (2, 9)])
def test_max_of_exact_relieff_over_the_recorded_subsets(size, count):
    X, y, is_discrete = _data()
    model = VLS(
        ReliefF(n_neighbors=3, backend="cpu", discrete_limit=3, n_jobs=1),
        subset_size=size,
        n_subsets=count,
        random_state=1,
    ).fit(X, y)
    expected = np.full(X.shape[1], -np.inf)
    for s, subset in enumerate(model.subsets_):
        local = relieff_oracle(X[:, subset], y, is_discrete[subset], 3)
        assert_allclose(model.subset_scores_[s], local, rtol=1e-5, atol=1e-6)
        np.maximum.at(expected, subset, local)
    assert_allclose(model.feature_importances_, expected, rtol=1e-5, atol=1e-6)
    assert_array_equal(model.top_features_, np.argsort(model.feature_importances_)[::-1][: model.top_features_.size])


def test_subsets_cover_every_feature_and_have_distinct_members():
    X, y, _ = _data(p=23)
    model = VLS(ReliefF(n_neighbors=2, backend="cpu", n_jobs=1), subset_size=5, n_subsets=9, random_state=3).fit(X, y)
    assert model.subsets_.shape == (9, 5)
    assert model.n_subsets_ == 9 and model.subset_size_ == 5
    assert set(model.subsets_.ravel()) == set(range(23))
    partition = math.ceil(23 / 5)
    assert sorted(model.subsets_[: partition - 1].ravel()) == sorted(set(model.subsets_[: partition - 1].ravel()))
    for subset in model.subsets_:
        assert len(set(subset)) == 5
    assert np.all(np.isfinite(model.feature_importances_))


def test_defaults_resolve_size_and_count():
    X, y, _ = _data(p=30)
    model = VLS(ReliefF(n_neighbors=2, backend="cpu", n_jobs=1), random_state=0).fit(X, y)
    assert model.subset_size_ == 3  # 10% of 30
    assert model.n_subsets_ == 2 * math.ceil(30 / 3)


def test_random_state_controls_the_subsets():
    X, y, _ = _data()
    base = ReliefF(n_neighbors=2, backend="cpu", n_jobs=1)
    first = VLS(base, subset_size=3, random_state=7).fit(X, y)
    again = VLS(base, subset_size=3, random_state=7).fit(X, y)
    other = VLS(base, subset_size=3, random_state=8).fit(X, y)
    assert_array_equal(first.subsets_, again.subsets_)
    assert_array_equal(first.feature_importances_, again.feature_importances_)
    assert not np.array_equal(first.subsets_, other.subsets_)


def test_one_subset_of_every_feature_is_the_base_estimator():
    X, y, _ = _data()
    base = ReliefF(n_neighbors=3, backend="cpu", discrete_limit=3, n_jobs=1)
    model = VLS(base, subset_size=X.shape[1], n_subsets=1, random_state=0).fit(X, y)
    direct = base.fit(X, y).feature_importances_
    assert_allclose(model.feature_importances_, direct, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "base",
    [
        SURF(backend="cpu", n_jobs=1),
        MultiSURF(backend="cpu", use_star=True, n_jobs=1),
        SWRF(use_star=True, n_jobs=1),
        MultiSWRF(dead_band=True, n_jobs=1),
        MuRelief(n_neighbors=3, n_jobs=1),
        STIR(n_jobs=1),
        ReliefSeq(max_neighbors=5, n_jobs=1),
    ],
    ids=["SURF", "MultiSURF*", "SWRF*", "MultiSWRFDB", "MuRelief", "STIR", "ReliefSeq"],
)
def test_any_relief_estimator_can_be_the_base(base):
    X, y, _ = _data(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = VLS(base, subset_size=4, n_subsets=5, random_state=0).fit(X, y)
        for s, subset in enumerate(model.subsets_):
            direct = base.set_params(n_features_to_select=4).fit(X[:, subset], y).feature_importances_
            assert_allclose(model.subset_scores_[s], direct, rtol=1e-12, atol=1e-12)


def test_finds_an_interaction_that_whole_space_distances_blur():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 3, (300, 40)).astype(np.float64)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    model = VLS(
        ReliefF(n_neighbors=10, backend="cpu", n_jobs=-1),
        n_features_to_select=2,
        subset_size=10,
        n_subsets=40,
        random_state=0,
    ).fit(X, y)
    together = any({0, 1} <= set(subset) for subset in model.subsets_)
    assert together
    assert set(model.top_features_) == {0, 1}


def test_missing_values_follow_the_base_estimator():
    X, y, _ = _data()
    X[[2, 5], -1] = np.nan
    VLS(ReliefF(n_neighbors=2, backend="cpu", discrete_limit=3, n_jobs=1), subset_size=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="NaN"):
        VLS(SURF(backend="cpu"), subset_size=3, random_state=0).fit(X, y)


def test_iterative_variant_through_turf():
    X, y, _ = _data(p=20, seed=5)
    vls = VLS(ReliefF(n_neighbors=3, backend="cpu", n_jobs=1), subset_size=5, random_state=0)
    turf = TuRF(vls, n_features_to_select=4, pct_remove=0.25).fit(X, y)
    assert turf.top_features_.size == 4


# --------------------------------------------------------------------------- API


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"subset_size": 0}, ValueError),
        ({"subset_size": 1.5}, ValueError),
        ({"subset_size": "3"}, TypeError),
        ({"subset_size": 3, "n_subsets": 2}, ValueError),
        ({"n_subsets": 0}, ValueError),
        ({"n_subsets": 2.5}, TypeError),
        ({"n_features_to_select": 0}, ValueError),
        ({"n_features_to_select": "1"}, TypeError),
    ],
)
def test_parameter_validation(kwargs, error):
    X, y, _ = _data()
    with pytest.raises(error):
        VLS(ReliefF(n_neighbors=2, backend="cpu"), **kwargs).fit(X, y)


def test_integer_subset_size_is_capped_at_the_feature_count():
    X, y, _ = _data()
    model = VLS(ReliefF(n_neighbors=2, backend="cpu"), subset_size=50, n_subsets=1, random_state=0).fit(X, y)
    assert model.subset_size_ == X.shape[1]


def test_sklearn_api_compliance():
    # A base that rejects NaN, as for TuRF; a NaN-tolerant ReliefF base receives NaN by design.
    check_estimator(VLS(SURF(backend="cpu", n_jobs=1), random_state=0))


def test_verbose_transform_and_pipeline(capsys):
    X, y, _ = _data(n=90, seed=6)
    model = VLS(ReliefF(n_neighbors=3, backend="cpu"), n_features_to_select=2, subset_size=3, verbose=True)
    reduced = model.fit_transform(X, y)
    assert "VLS: scored" in capsys.readouterr().out
    assert_array_equal(reduced, X[:, model.top_features_])
    assert model.effective_backend_ == "cpu"
    pipe = Pipeline(
        [("select", VLS(ReliefF(n_neighbors=3, backend="cpu"), random_state=0)), ("m", LogisticRegression())]
    )
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2], "select__subset_size": [3, 5]}, cv=3)
    search.fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )
