"""μ-Relief against its definition.

Aggarwal et al. (2023), *Mean based relief*, Applied Intelligence 53, as restated
in Kazemi-Nia et al. (2026), arXiv:2608.28552, Eq. (2): for target ``R_i`` the
neighbourhood is the ``k`` hits and ``k`` misses (per class) with the largest
``|dist_ij - mu_{C_j, i}|``, where ``mu_{C_j, i}`` is the mean distance from
``R_i`` to ``R_j``'s class, scored with ReliefF's update.

The oracle is exact: distances, class means, deviations, priors and scores are
``fractions.Fraction`` values, so tied deviations are true ties.  It never calls
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

from fast_select import MuRelief, TuRF


def _exact_differences(X, is_discrete):
    """Per-feature differences as exact rationals: ``|a - b| / range`` or 0/1."""
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


def murelief_oracle(X, y, is_discrete, k, centre="class"):
    """Exact μ-Relief scores and the chosen neighbourhoods.

    ``centre="class"`` is the definition (mean over the neighbour's class);
    ``"all"`` (mean over every other instance) and ``"nearest"`` (ReliefF's
    nearest neighbours) exist only to show that the definition matters.
    """
    diff, n, p = _exact_differences(X, is_discrete)
    y = list(np.asarray(y))
    classes = sorted(set(y))
    prior = {c: Fraction(y.count(c), n) for c in classes}
    scores = [Fraction(0)] * p
    neighbourhoods = []
    for i in range(n):
        dist = {j: sum(diff(i, j), Fraction(0)) for j in range(n) if j != i}
        overall = sum(dist.values(), Fraction(0)) / len(dist)
        chosen = {}
        for c in classes:
            members = [j for j in dist if y[j] == c]
            if not members:
                chosen[c] = []
                continue
            mu = sum((dist[j] for j in members), Fraction(0)) / len(members)
            if centre == "nearest":
                ranked = sorted(members, key=lambda j: (dist[j], j))
            else:
                reference = mu if centre == "class" else overall
                ranked = sorted(members, key=lambda j: (-abs(dist[j] - reference), j))
            chosen[c] = ranked[:k]
        neighbourhoods.append(chosen)
        hits = chosen[y[i]]
        for j in hits:
            for f, value in enumerate(diff(i, j)):
                scores[f] -= value / (len(hits) * n)
        for c in classes:
            if c == y[i] or not chosen[c]:
                continue
            weight = prior[c] / (1 - prior[y[i]])
            for j in chosen[c]:
                for f, value in enumerate(diff(i, j)):
                    scores[f] += weight * value / (len(chosen[c]) * n)
    return np.array([float(s) for s in scores]), neighbourhoods


def _mixed_data(n=36, seed=0, labels=(3, 7), weights=(0.7, 0.3)):
    """Two continuous columns with non-dyadic ranges and two discrete columns."""
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
    return X, y, np.array([False, False, True, True])


def _fit(X, y, k, **kwargs):
    kwargs.setdefault("n_features_to_select", 1)
    kwargs.setdefault("n_jobs", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MuRelief(n_neighbors=k, **kwargs).fit(X, y)


# --------------------------------------------------------------------------- equation oracle


@pytest.mark.parametrize("k", [1, 3, 10])
def test_binary_matches_exact_oracle(k):
    X, y, is_discrete = _mixed_data()
    expected, _ = murelief_oracle(X, y, is_discrete, k)
    model = _fit(X, y, k, discrete_limit=3)
    assert_array_equal(model.is_discrete_, is_discrete)
    assert_allclose(model.feature_importances_, expected, rtol=1e-12, atol=1e-14)
    assert np.all(np.abs(model.feature_importances_) <= 1.0)


@pytest.mark.parametrize("k", [1, 4])
def test_multiclass_uses_relieff_prior_weighting(k):
    X, y, is_discrete = _mixed_data(n=42, seed=4, labels=(-2, 5, 11), weights=(0.5, 0.35, 0.15))
    expected, _ = murelief_oracle(X, y, is_discrete, k)
    assert_allclose(_fit(X, y, k, discrete_limit=3).feature_importances_, expected, rtol=1e-12, atol=1e-14)


def test_continuous_only_and_discrete_only_inputs():
    X, y, _ = _mixed_data(seed=6)
    expected, _ = murelief_oracle(X[:, :2], y, [False, False], 4)
    assert_allclose(_fit(X[:, :2], y, 4, discrete_limit=0).feature_importances_, expected, rtol=1e-12, atol=1e-14)
    expected, _ = murelief_oracle(X[:, 2:], y, [True, True], 4)
    assert_allclose(_fit(X[:, 2:], y, 4, discrete_limit=3).feature_importances_, expected, rtol=1e-12, atol=1e-14)


def test_neighbours_lie_on_both_sides_of_the_class_mean_and_differ_from_relieff():
    X, y, is_discrete = _mixed_data(seed=1)
    _, chosen = murelief_oracle(X, y, is_discrete, 5)
    _, nearest = murelief_oracle(X, y, is_discrete, 5, centre="nearest")
    assert chosen != nearest
    diff, n, _ = _exact_differences(X, is_discrete)
    below = above = 0
    for i in range(n):
        for c, members in chosen[i].items():
            pool = [j for j in range(n) if j != i and y[j] == c]
            mu = sum((sum(diff(i, j), Fraction(0)) for j in pool), Fraction(0)) / len(pool)
            for j in members:
                d = sum(diff(i, j), Fraction(0))
                below += d < mu
                above += d > mu
    assert below > 0 and above > 0


def test_class_means_not_the_overall_mean_define_the_deviation():
    X, y, is_discrete = _mixed_data(seed=5)
    expected, chosen = murelief_oracle(X, y, is_discrete, 3)
    overall, chosen_overall = murelief_oracle(X, y, is_discrete, 3, centre="all")
    assert chosen != chosen_overall
    actual = _fit(X, y, 3, discrete_limit=3).feature_importances_
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert not np.allclose(actual, overall, rtol=1e-6)


# --------------------------------------------------------------------------- hand calculation


def test_hand_calculated_scores():
    """Range 6; class 0 = {0, 1, 4}, class 1 = {6}.

    Target 0: hits 1 (d=1/6), 4 (d=4/6); mean 5/12, deviations 3/12 and 3/12 tie,
    so k=1 keeps the lower index 1.  Target 1: hits 0 (1/6), 4 (3/6); mean 1/3,
    deviations 1/6 tie, keep 0.  Target 4: hits 0 (4/6), 1 (3/6); mean 7/12,
    deviations 1/12 tie, keep 0.  Misses: the single class-1 instance, d = 1,
    5/6, 2/6.  Target 6 (class 1) has no hit; its misses are class 0, deviations
    from mean 13/18: 0 -> 5/18, 1 -> 2/18, 4 -> 7/18, so k=1 picks 4 (d = 2/6).
    """
    X = np.array([[0.0], [1.0], [4.0], [6.0]])
    y = np.array([0, 0, 0, 1])
    hits = -(Fraction(1, 6) + Fraction(1, 6) + Fraction(4, 6))
    misses = Fraction(1) + Fraction(5, 6) + Fraction(2, 6) + Fraction(2, 6)
    expected = float((hits + misses) / 4)
    assert_allclose(_fit(X, y, 1, discrete_limit=0).feature_importances_, [expected], rtol=1e-13)
    assert_allclose(murelief_oracle(X, y, [False], 1)[0], [expected], rtol=1e-13)


# --------------------------------------------------------------------------- exact deviation ties

RANGES = [3, 5, 6, 7, 9]


def _integer_case(seed, mixed):
    rng = np.random.default_rng(seed)
    n, p = int(rng.integers(5, 11)), int(rng.integers(1, 4))
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


def _has_cut_tie(X, y, discrete, k):
    """Whether some class list has equal deviations straddling its k-th place."""
    diff, n, _ = _exact_differences(X, discrete)
    for i in range(n):
        dist = {j: sum(diff(i, j), Fraction(0)) for j in range(n) if j != i}
        for c in set(y):
            members = [j for j in dist if y[j] == c]
            if len(members) <= k:
                continue
            mu = sum((dist[j] for j in members), Fraction(0)) / len(members)
            keys = sorted((abs(dist[j] - mu) for j in members), reverse=True)
            if keys[k - 1] == keys[k]:
                return True
    return False


def _tie_cases(wanted=15, search=3000):
    found = []
    for mixed in (False, True):
        count = 0
        for seed in range(search):
            X, y, discrete, k = _integer_case(seed, mixed)
            if _consistent(X, discrete) and _has_cut_tie(X, y, discrete, k):
                found.append((mixed, seed))
                count += 1
            if count == wanted:
                break
    return found


TIE_CASES = _tie_cases()


def test_search_finds_ties_at_the_neighbour_cut():
    assert len(TIE_CASES) >= 20


@pytest.mark.parametrize("mixed,seed", TIE_CASES)
def test_tied_deviations_keep_the_lower_index_exactly(mixed, seed):
    X, y, discrete, k = _integer_case(seed, mixed)
    expected, _ = murelief_oracle(X, y, discrete, k)
    actual = _fit(X, y, k, discrete_limit=_discrete_limit_for(X, discrete)).feature_importances_
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- small and empty groups


def test_small_class_uses_every_member_and_warns():
    X, y, is_discrete = _mixed_data(n=30, seed=2, labels=(0, 1), weights=(0.9, 0.1))
    y[:] = 0
    y[[3, 11, 20]] = 1  # three class-1 instances: two hits each, fewer than k = 5
    expected, chosen = murelief_oracle(X, y, is_discrete, 5)
    assert all(len(chosen[i][1]) == 2 for i in (3, 11, 20))
    with pytest.warns(UserWarning, match="smallest class size"):
        model = MuRelief(n_neighbors=5, discrete_limit=3, n_jobs=1).fit(X, y)
    assert_allclose(model.feature_importances_, expected, rtol=1e-12, atol=1e-14)


def test_singleton_class_has_no_hit_term():
    X, y, is_discrete = _mixed_data(n=20, seed=3)
    y[:] = 7
    y[9] = 3
    expected, chosen = murelief_oracle(X, y, is_discrete, 2)
    assert chosen[9][3] == []
    assert_allclose(_fit(X, y, 2, discrete_limit=3).feature_importances_, expected, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- invariances


def test_constant_feature_scores_exactly_zero():
    X, y, _ = _mixed_data(seed=7)
    X = np.column_stack([X, np.full(len(y), 4.2)])
    assert _fit(X, y, 4, discrete_limit=3).feature_importances_[-1] == 0.0


def test_translation_and_category_relabelling_invariance():
    X, y, _ = _mixed_data(seed=8)
    X[:, :2] = np.round(X[:, :2] * 1000)  # integer-valued, so a 1e8 shift is exact
    reference = _fit(X, y, 4, discrete_limit=3).feature_importances_
    shifted = X.copy()
    shifted[:, 0] += 1e8
    shifted[:, 2] += 2**25
    shifted[:, 3] = shifted[:, 3] * 7.0 - 2**25
    assert_array_equal(_fit(shifted, y, 4, discrete_limit=3).feature_importances_, reference)


def test_label_values_do_not_matter():
    X, y, _ = _mixed_data(seed=2)
    reference = _fit(X, y, 3, discrete_limit=3).feature_importances_
    assert_array_equal(_fit(X, np.where(y == 3, 0, 1), 3, discrete_limit=3).feature_importances_, reference)


def test_thread_count_only_changes_reduction_order():
    X, y, _ = _mixed_data(n=160, seed=9)
    serial = _fit(X, y, 10, discrete_limit=3, n_jobs=1).feature_importances_
    parallel = _fit(X, y, 10, discrete_limit=3, n_jobs=-1).feature_importances_
    assert_allclose(parallel, serial, rtol=1e-12, atol=1e-15)


def _main_effect(rng):
    X = rng.random((300, 8))
    return X, (X[:, 0] + 0.1 * rng.random(300) > 0.55).astype(int), {0}


def _two_way(rng):
    X = rng.integers(0, 3, (400, 10)).astype(np.float64)
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int), {0, 1}


def _three_way(rng):
    X = rng.integers(0, 2, (600, 10)).astype(np.float64)
    return X, ((X[:, 0] + X[:, 1] + X[:, 2]) % 2).astype(int), {0, 1, 2}


@pytest.mark.parametrize("make", [_main_effect, _two_way, _three_way], ids=["main", "2-way", "3-way"])
def test_finds_main_effects_and_pure_interactions(make):
    X, y, truth = make(np.random.default_rng(0))
    model = _fit(X, y, 10, n_features_to_select=len(truth), n_jobs=-1)
    assert set(model.top_features_) == truth


# --------------------------------------------------------------------------- external cross-check


def test_matches_pinned_scikit_rebate_on_tie_free_continuous_binary_data():
    """scikit-rebate 0.8.4's MuRelief, by the authors of the restatement, as corroboration only.

    On continuous binary data without tied deviations it uses the same
    range-normalised difference, class means and ReliefF update. (On mixed
    data it switches to a ramp difference, and it breaks ties in another
    order, so those cases are not comparable.)
    """
    skrebate = pytest.importorskip("skrebate")
    if skrebate.__version__ != "0.8.4":
        pytest.skip(f"cross-check pinned to scikit-rebate 0.8.4, found {skrebate.__version__}")
    rng = np.random.default_rng(11)
    X = rng.random((60, 6)) * np.array([1.0, 3.3, 7.1, 0.4, 2.0, 9.0])
    y = (X[:, 0] + 0.3 * X[:, 1] + 0.2 * rng.random(60) > 1.0).astype(int)
    reference = skrebate.MuRelief(n_features_to_select=2, n_neighbors=5, n_jobs=1).fit(X, y)
    model = _fit(X, y, 5, discrete_limit=0)
    assert_allclose(model.feature_importances_, reference.feature_importances_, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- behaviour and API


def test_sklearn_api_compliance():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check_estimator(MuRelief(n_neighbors=2, n_jobs=1))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"backend": "gpu"}, ValueError),
        ({"backend": "metal"}, ValueError),
        ({"n_neighbors": 0}, ValueError),
        ({"n_neighbors": 36}, ValueError),
        ({"n_neighbors": 2.0}, TypeError),
        ({"n_neighbors": True}, TypeError),
        ({"n_features_to_select": 0}, ValueError),
        ({"n_features_to_select": 1.5}, ValueError),
        ({"n_features_to_select": "2"}, TypeError),
        ({"discrete_limit": -1}, ValueError),
        ({"n_jobs": 0}, ValueError),
    ],
)
def test_parameter_validation(kwargs, error):
    X, y, _ = _mixed_data()
    with pytest.raises(error):
        MuRelief(**{"n_neighbors": 3, **kwargs}).fit(X, y)


def test_rejects_missing_values_continuous_targets_and_one_sample():
    X, y, _ = _mixed_data()
    X_missing = X.copy()
    X_missing[2, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        MuRelief(n_neighbors=3).fit(X_missing, y)
    with pytest.raises(ValueError, match="Unknown label type"):
        MuRelief(n_neighbors=3).fit(X, np.linspace(0.0, 1.0, len(y)))
    with pytest.raises(ValueError):
        MuRelief(n_neighbors=1).fit(X[:1], y[:1])


def test_single_class_scores_zero_and_transform():
    X, y, _ = _mixed_data()
    model = MuRelief(n_features_to_select=2, n_neighbors=3).fit(X, np.full(len(X), 4))
    assert_array_equal(model.feature_importances_, np.zeros(X.shape[1]))
    model = MuRelief(n_features_to_select=2, n_neighbors=3, discrete_limit=3)
    reduced = model.fit_transform(X, y)
    assert model.effective_backend_ == "cpu"
    assert model.feature_importances_.dtype == np.float64
    assert_array_equal(reduced, X[:, model.top_features_])
    with pytest.raises(ValueError):
        model.transform(X[:, :3])


def test_verbose_output(capsys):
    X, y, _ = _mixed_data()
    MuRelief(n_neighbors=3, verbose=True).fit(X, y)
    assert "Running MuRelief on the CPU" in capsys.readouterr().out


def test_in_pipeline_grid_search_and_turf():
    X, y, _ = _mixed_data(n=90, seed=1)
    pipe = Pipeline([("select", MuRelief(n_neighbors=3, discrete_limit=3)), ("model", LogisticRegression())])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2, 3], "select__n_neighbors": [3, 5]}, cv=3).fit(
        X, y
    )
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )
    turf = TuRF(MuRelief(n_neighbors=3, discrete_limit=3), n_features_to_select=2, pct_remove=0.25).fit(X, y)
    assert turf.top_features_.size == 2
