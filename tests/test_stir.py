"""STIR against Le, Urbanowicz, Moore and McKinney (2019), Eqs. (1)-(13).

The oracle is exact up to the final square root and t tail: differences,
distances, the MultiSURF radius test (``d < mu - sigma/2`` decided as
``mu - d > 0 and 4 (mu - d)^2 > sigma^2``), the means of Eqs. (5)-(6) and the
variances of Eqs. (12)-(13) are ``fractions.Fraction`` values.  It never calls
FastSelect internals.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_checks

from fast_select import STIR, MultiSURF, TuRF


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


def _benjamini_hochberg(p):
    """Textbook BH: q_(i) = min_{j >= i} p_(j) n / j, capped at 1."""
    n = len(p)
    order = sorted(range(n), key=lambda i: (p[i], i))
    adjusted = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * n / rank)
        adjusted[i] = running
    return np.array(adjusted)


def stir_oracle(X, y, is_discrete, neighborhood="multisurf", k=None):
    diff, n, p = _exact_differences(X, is_discrete)
    y = list(np.asarray(y))
    if k is None:
        k = max(1, n // 6)
    dist = {(i, j): sum(diff(i, j), Fraction(0)) for i in range(n) for j in range(n) if i != j}
    groups = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if neighborhood == "multisurf":
            mu = sum((dist[i, j] for j in others), Fraction(0)) / len(others)
            var = sum(((dist[i, j] - mu) ** 2 for j in others), Fraction(0)) / len(others)
            near = [j for j in others if mu - dist[i, j] > 0 and 4 * (mu - dist[i, j]) ** 2 > var]
            hits = [j for j in near if y[j] == y[i]]
            misses = [j for j in near if y[j] != y[i]]
        else:
            hits = sorted((j for j in others if y[j] == y[i]), key=lambda j: (dist[i, j], j))[:k]
            misses = sorted((j for j in others if y[j] != y[i]), key=lambda j: (dist[i, j], j))[:k]
        if hits and misses:
            groups.append((i, hits, misses))
    m = len(groups)
    if m == 0:
        return {"m": 0, "df": -1}  # undefined: no target has both a hit and a miss
    zero = [Fraction(0)] * p

    def group_mean(selector):
        total = list(zero)
        for i, hits, misses in groups:
            members = selector(hits, misses)
            for j in members:
                for f, value in enumerate(diff(i, j)):
                    total[f] += value / len(members)
        return [value / m for value in total]

    def group_var(selector, mean):
        total = list(zero)
        for i, hits, misses in groups:
            members = selector(hits, misses)
            for j in members:
                for f, value in enumerate(diff(i, j)):
                    total[f] += (value - mean[f]) ** 2 / len(members)
        return [value / m for value in total]

    def hits_of(h, _):
        return h

    def misses_of(_, mm):
        return mm

    h_mean, m_mean = group_mean(hits_of), group_mean(misses_of)
    h_var, m_var = group_var(hits_of, h_mean), group_var(misses_of, m_mean)
    n_h = sum(len(h) for _, h, _ in groups)
    n_m = sum(len(mm) for _, _, mm in groups)
    df = n_h + n_m - 2
    statistic, relief = [], []
    for f in range(p):
        pooled_sq = ((n_m - 1) * m_var[f] + (n_h - 1) * h_var[f]) / df
        difference = m_mean[f] - h_mean[f]
        relief.append(float(difference))
        if pooled_sq == 0:
            statistic.append(0.0 if difference == 0 else math.copysign(math.inf, difference))
        else:
            statistic.append(float(difference) / math.sqrt(float(pooled_sq * (Fraction(1, n_m) + Fraction(1, n_h)))))
    statistic = np.array(statistic)
    p_values = student_t.sf(statistic, df)
    return {
        "t": statistic,
        "relief": np.array(relief),
        "p": p_values,
        "p_adj": _benjamini_hochberg(list(p_values)),
        "df": df,
        "n_hits": n_h,
        "n_misses": n_m,
        "m": m,
    }


def _mixed_data(n=36, seed=0, labels=(3, 7), weights=(0.6, 0.4)):
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
    X[:, 0] += 2.0 * (y == labels[0])  # some signal so the statistics are not all noise
    return X, y, np.array([False, False, True, True])


def _assert_matches(model, expected):
    assert model.degrees_of_freedom_ == expected["df"]
    assert model.n_hit_pairs_ == expected["n_hits"]
    assert model.n_miss_pairs_ == expected["n_misses"]
    assert model.n_targets_ == expected["m"]
    assert_allclose(model.relief_scores_, expected["relief"], rtol=1e-11, atol=1e-14)
    assert_allclose(model.feature_importances_, expected["t"], rtol=1e-9, atol=1e-12)
    assert_allclose(model.p_values_, expected["p"], rtol=1e-7, atol=1e-15)
    assert_allclose(model.p_values_adjusted_, expected["p_adj"], rtol=1e-7, atol=1e-15)


# --------------------------------------------------------------------------- equation oracle


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multisurf_neighbourhood_matches_oracle(seed):
    X, y, is_discrete = _mixed_data(seed=seed)
    expected = stir_oracle(X, y, is_discrete)
    model = STIR(discrete_limit=3, n_jobs=1).fit(X, y)
    assert_array_equal(model.is_discrete_, is_discrete)
    _assert_matches(model, expected)


@pytest.mark.parametrize("k", [1, 4, None])
def test_relieff_neighbourhood_matches_oracle(k):
    X, y, is_discrete = _mixed_data(seed=3)
    expected = stir_oracle(X, y, is_discrete, "relieff", k)
    model = STIR(neighborhood="relieff", n_neighbors=k, discrete_limit=3, n_jobs=1).fit(X, y)
    _assert_matches(model, expected)
    if k is not None:
        assert expected["n_hits"] == expected["n_misses"] == expected["m"] * k


def test_relief_score_equals_multisurf_when_every_target_qualifies():
    X, y, is_discrete = _mixed_data(n=60, seed=4)
    expected = stir_oracle(X, y, is_discrete)
    assert expected["m"] == len(y)
    model = STIR(discrete_limit=3, n_jobs=1).fit(X, y)
    multisurf = MultiSURF(n_features_to_select=1, backend="cpu", discrete_limit=3, n_jobs=1).fit(X, y)
    assert_allclose(model.relief_scores_, multisurf.feature_importances_, rtol=1e-5, atol=1e-6)


def test_targets_without_a_hit_or_miss_are_left_out():
    """An outlier far from everything has no MultiSURF neighbour of one class."""
    X, y, is_discrete = _mixed_data(n=30, seed=5)
    X[0, :2] = [400.0, 200.0]
    expected = stir_oracle(X, y, is_discrete)
    assert expected["m"] < len(y)
    _assert_matches(STIR(discrete_limit=3, n_jobs=1).fit(X, y), expected)


def test_continuous_only_and_discrete_only_inputs():
    X, y, _ = _mixed_data(seed=6)
    _assert_matches(STIR(discrete_limit=0, n_jobs=1).fit(X[:, :2], y), stir_oracle(X[:, :2], y, [False, False]))
    _assert_matches(STIR(discrete_limit=3, n_jobs=1).fit(X[:, 2:], y), stir_oracle(X[:, 2:], y, [True, True]))


# --------------------------------------------------------------------------- exact boundaries and ties

RANGES = [3, 5, 6, 7, 9]


def _integer_case(seed, mixed):
    rng = np.random.default_rng(seed)
    n, p = int(rng.integers(6, 12)), int(rng.integers(1, 4))
    X = np.column_stack([rng.integers(0, RANGES[int(rng.integers(len(RANGES)))] + 1, n) for _ in range(p)])
    y = rng.integers(0, 2, n)
    if len(np.unique(y)) < 2:
        y[0], y[1] = 0, 1
    discrete = np.zeros(p, dtype=bool)
    if mixed:
        discrete[::2] = True
    return X.astype(np.float64), y, discrete, int(rng.integers(1, 4))


def _discrete_limit_for(X, discrete):
    if not discrete.any():
        return 0
    return max(len(np.unique(X[:, f])) for f in np.flatnonzero(discrete))


def _consistent(X, discrete):
    limit = _discrete_limit_for(X, discrete)
    return all((len(np.unique(X[:, f])) <= limit) == discrete[f] for f in range(X.shape[1]))


def _radius_tie(X, discrete):
    diff, n, _ = _exact_differences(X, discrete)
    for i in range(n):
        d = [sum(diff(i, j), Fraction(0)) for j in range(n) if j != i]
        mu = sum(d, Fraction(0)) / len(d)
        var = sum(((v - mu) ** 2 for v in d), Fraction(0)) / len(d)
        if var > 0 and any(v < mu and 4 * (mu - v) ** 2 == var for v in d):
            return True
    return False


def _cut_tie(X, y, discrete, k):
    diff, n, _ = _exact_differences(X, discrete)
    for i in range(n):
        for same in (True, False):
            d = sorted(sum(diff(i, j), Fraction(0)) for j in range(n) if j != i and (y[j] == y[i]) == same)
            if len(d) > k and d[k - 1] == d[k]:
                return True
    return False


def _tie_cases(kind, wanted=10, search=4000):
    found = []
    for mixed in (False, True):
        count = 0
        for seed in range(search):
            X, y, discrete, k = _integer_case(seed, mixed)
            if not _consistent(X, discrete):
                continue
            tie = _radius_tie(X, discrete) if kind == "radius" else _cut_tie(X, y, discrete, k)
            if not tie:
                continue
            expected = stir_oracle(X, y, discrete, "multisurf" if kind == "radius" else "relieff", k)
            if expected["m"] > 1 and expected["df"] > 0:
                found.append((mixed, seed))
                count += 1
            if count == wanted:
                break
    return found


RADIUS_TIES = _tie_cases("radius")
CUT_TIES = _tie_cases("cut")


def test_searches_find_exact_ties():
    assert len(RADIUS_TIES) >= 10
    assert len(CUT_TIES) >= 15


@pytest.mark.parametrize("mixed,seed", RADIUS_TIES)
def test_multisurf_radius_ties_are_excluded_exactly(mixed, seed):
    X, y, discrete, _ = _integer_case(seed, mixed)
    model = STIR(discrete_limit=_discrete_limit_for(X, discrete), n_jobs=1).fit(X, y)
    _assert_matches(model, stir_oracle(X, y, discrete))


@pytest.mark.parametrize("mixed,seed", CUT_TIES)
def test_relieff_distance_ties_keep_the_lower_index(mixed, seed):
    X, y, discrete, k = _integer_case(seed, mixed)
    model = STIR(neighborhood="relieff", n_neighbors=k, discrete_limit=_discrete_limit_for(X, discrete), n_jobs=1)
    _assert_matches(model.fit(X, y), stir_oracle(X, y, discrete, "relieff", k))


# --------------------------------------------------------------------------- zero variance


def test_constant_feature_has_t_zero_and_p_one_half():
    X, y, _ = _mixed_data(seed=7)
    X = np.column_stack([X, np.full(len(y), 4.2)])
    model = STIR(discrete_limit=3, n_jobs=1).fit(X, y)
    assert model.feature_importances_[-1] == 0.0
    assert model.relief_scores_[-1] == 0.0
    assert model.p_values_[-1] == 0.5


def test_perfectly_separating_feature_has_infinite_t_and_zero_p():
    X, y, _ = _mixed_data(seed=8)
    X = np.column_stack([X, (y == 3).astype(float)])  # hits never differ, misses always do
    for neighborhood in ("multisurf", "relieff"):
        model = STIR(neighborhood=neighborhood, n_neighbors=5, discrete_limit=3, n_jobs=1).fit(X, y)
        assert model.feature_importances_[-1] == np.inf
        assert model.relief_scores_[-1] == 1.0
        assert model.p_values_[-1] == 0.0
        assert model.top_features_[0] == X.shape[1] - 1


# --------------------------------------------------------------------------- selection and statistics


def test_benjamini_hochberg_and_alpha_selection():
    rng = np.random.default_rng(1)
    X = rng.random((200, 12))
    y = (X[:, 0] + X[:, 1] + 0.3 * rng.random(200) > 1.15).astype(int)
    model = STIR(alpha=0.05, n_jobs=-1).fit(X, y)
    assert_allclose(model.p_values_adjusted_, _benjamini_hochberg(list(model.p_values_)), rtol=1e-12)
    assert set(model.top_features_) >= {0, 1}
    assert np.all(model.p_values_adjusted_[model.top_features_] < 0.05)
    assert list(model.top_features_) == sorted(model.top_features_, key=lambda f: -model.feature_importances_[f])
    nothing = STIR(alpha=1e-300, n_jobs=-1).fit(X, y)
    assert nothing.top_features_.size == 0
    assert nothing.transform(X).shape == (200, 0)


def test_signal_is_significant_and_pure_noise_is_not():
    rng = np.random.default_rng(2)
    X = rng.random((240, 20))
    y = (X[:, 0] + 0.1 * rng.random(240) > 0.55).astype(int)
    model = STIR(n_jobs=-1).fit(X, y)
    assert model.p_values_adjusted_[0] < 1e-6
    noise = STIR(n_jobs=-1).fit(X, rng.permutation(y))
    assert np.mean(noise.p_values_adjusted_ < 0.05) <= 0.1


def test_finds_a_pure_two_way_interaction():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 3, (400, 10)).astype(np.float64)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    model = STIR(n_features_to_select=2, n_jobs=-1).fit(X, y)
    assert set(model.top_features_) == {0, 1}


# --------------------------------------------------------------------------- invariances


def test_translation_relabelling_and_label_values():
    X, y, _ = _mixed_data(seed=9)
    X[:, :2] = np.round(X[:, :2] * 1000)
    reference = STIR(discrete_limit=3, n_jobs=1).fit(X, y)
    shifted = X.copy()
    shifted[:, 0] += 1e8
    shifted[:, 2] += 2**25
    moved = STIR(discrete_limit=3, n_jobs=1).fit(shifted, np.where(y == 3, -1, 8))
    assert_array_equal(moved.feature_importances_, reference.feature_importances_)
    assert_array_equal(moved.p_values_, reference.p_values_)


def test_thread_count_only_changes_reduction_order():
    X, y, _ = _mixed_data(n=160, seed=10)
    for neighborhood in ("multisurf", "relieff"):
        serial = STIR(neighborhood=neighborhood, discrete_limit=3, n_jobs=1).fit(X, y)
        parallel = STIR(neighborhood=neighborhood, discrete_limit=3, n_jobs=-1).fit(X, y)
        assert_allclose(parallel.feature_importances_, serial.feature_importances_, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- behaviour and API


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"neighborhood": "surf"}, ValueError),
        ({"neighborhood": "relieff", "n_neighbors": 0}, ValueError),
        ({"neighborhood": "relieff", "n_neighbors": 36}, ValueError),
        ({"neighborhood": "relieff", "n_neighbors": 2.0}, TypeError),
        ({"alpha": 0.0}, ValueError),
        ({"alpha": 1.5}, ValueError),
        ({"alpha": True}, ValueError),
        ({"backend": "gpu"}, ValueError),
        ({"n_features_to_select": 0}, ValueError),
        ({"n_jobs": 0}, ValueError),
    ],
)
def test_parameter_validation(kwargs, error):
    X, y, _ = _mixed_data()
    with pytest.raises(error):
        STIR(**kwargs).fit(X, y)


def test_rejects_non_binary_targets_missing_values_and_tiny_inputs():
    X, y, _ = _mixed_data()
    with pytest.raises(ValueError, match="binary"):
        STIR().fit(X, np.arange(len(y)) % 3)
    with pytest.raises(ValueError, match="binary"):
        STIR().fit(X, np.zeros(len(y)))
    with pytest.raises(ValueError, match="Unknown label type"):
        STIR().fit(X, np.linspace(0, 1, len(y)))
    X_missing = X.copy()
    X_missing[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        STIR().fit(X_missing, y)
    with pytest.raises(ValueError):
        STIR().fit(X[:2], y[:2])


def test_fitted_attributes_and_transform():
    X, y, _ = _mixed_data()
    model = STIR(n_features_to_select=2, discrete_limit=3)
    reduced = model.fit_transform(X, y)
    assert model.effective_backend_ == "cpu"
    assert_array_equal(reduced, X[:, model.top_features_])
    assert_array_equal(model.top_features_, np.argsort(-model.feature_importances_, kind="stable")[:2])
    with pytest.raises(ValueError):
        model.transform(X[:, :3])


def test_verbose_output(capsys):
    X, y, _ = _mixed_data()
    STIR(neighborhood="relieff", verbose=True).fit(X, y)
    assert "Running STIR (relieff) on the CPU" in capsys.readouterr().out


# STIR is binary only, so the checks that feed it multiclass or regression
# targets cannot apply; the domain-independent part of the contract runs here.
CONTRACT_CHECKS = [
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_get_params_invariance",
    "check_set_params",
    "check_estimator_cloneable",
    "check_estimator_repr",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_estimators_unfitted",
    "check_estimator_tags_renamed",
    "check_valid_tag_types",
    "check_estimator_sparse_tag",
]


def test_domain_independent_sklearn_contract():
    generated = estimator_checks.estimator_checks_generator(STIR(), legacy=True, expected_failed_checks=None, mark=None)
    ran = []
    for estimator, check in generated:
        name = getattr(check, "func", check).__name__
        if name in CONTRACT_CHECKS:
            check(estimator)
            ran.append(name)
    assert len(ran) >= 8, ran


def test_in_pipeline_grid_search_and_turf():
    X, y, _ = _mixed_data(n=90, seed=11)
    pipe = Pipeline([("select", STIR(discrete_limit=3)), ("model", LogisticRegression())])
    search = GridSearchCV(
        pipe, {"select__n_features_to_select": [1, 2, 3], "select__neighborhood": ["multisurf", "relieff"]}, cv=3
    ).fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )
    turf = TuRF(STIR(discrete_limit=3), n_features_to_select=2, pct_remove=0.25).fit(X, y)
    assert turf.top_features_.size == 2
