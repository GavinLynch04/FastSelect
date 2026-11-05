"""SWRF family against its defining equations.

* SWRF*: Stokes and Visweswaran (2012), BioData Mining 5:20, the Modular Relief
  Framework (MoRF) update Eqs. (1)-(4) with the sigmoid neighbour weight.
* SWRF, MultiSWRF, MultiSWRF*, MultiSWRFDB, MultiSWRFDB*: Kazemi-Nia et al.
  (2026), arXiv:2608.28552, neighbour weights Eqs. (3)-(7), used in the MoRF
  update.

The oracle transcribes those equations in their published logistic form and
never calls FastSelect internals.  Distances, means, variances and every
boundary predicate are exact rationals (``fractions.Fraction``), so a pair that
lies exactly on a boundary is a true tie; only the sigmoid values are floats.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_select import SWRF, MultiSWRF, TuRF

# name -> (per-target statistics, star, dead band)
VARIANTS = {
    "SWRF": (False, False, False),
    "SWRF*": (False, True, False),
    "MultiSWRF": (True, False, False),
    "MultiSWRF*": (True, True, False),
    "MultiSWRFDB": (True, False, True),
    "MultiSWRFDB*": (True, True, True),
}


def _estimator(name, **kwargs):
    per_target, star, dead_band = VARIANTS[name]
    kwargs.setdefault("n_features_to_select", 1)
    kwargs.setdefault("n_jobs", 1)
    if per_target:
        return MultiSWRF(use_star=star, dead_band=dead_band, **kwargs)
    return SWRF(use_star=star, **kwargs)


def _logistic(x):
    """``2 / (1 + exp(x)) - 1``, saturated where ``exp`` would overflow."""
    if x > 700.0:
        return -1.0
    if x < -700.0:
        return 1.0
    return 2.0 / (1.0 + math.exp(x)) - 1.0


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


def _neighbour_weight(name, d, mu, var):
    """The published neighbour weight and the region ``d`` falls in.

    ``d``, ``mu`` and ``var`` are exact; ``sigma = sqrt(var)`` is only needed
    as a float for the sigmoid itself, so every predicate stays exact.
    """
    per_target, star, dead_band = VARIANTS[name]
    if var == 0:
        return 0.0, "tie"  # every distance equals mu, the sigmoid's centre
    sigma = math.sqrt(var)
    offset = float(d - mu)
    if name == "SWRF*":
        # Stokes and Visweswaran (2012): f = 2 / (1 + exp(-(t - d) / (u / 4))) - 1.
        region = "near" if d < mu else "far" if d > mu else "tie"
        return _logistic(-float(mu - d) / (sigma / 4)), region
    if not dead_band:
        if d < mu:
            return _logistic(offset * 4 / sigma), "near"  # Eqs. (3)-(5)
        if d > mu:
            return (_logistic(offset * 4 / sigma) if star else 0.0), "far"  # Eq. (3)
        return 0.0, "tie"
    # Eqs. (6)-(7): boundaries mu -/+ sigma / 2, compared exactly via 4 (mu - d)^2 vs var.
    if d < mu and 4 * (mu - d) ** 2 > var:
        return 2.0 / (1.0 + math.exp((offset + sigma / 2) * 4 / sigma)) - 1.0, "near"
    if d > mu and 4 * (d - mu) ** 2 > var:
        weight = -2.0 / (1.0 + math.exp((offset - sigma / 2) * (-4 / sigma))) + 1.0
        return (weight if star else 0.0), "far"
    if 4 * (mu - d) ** 2 == var:
        return 0.0, "tie"
    return 0.0, "dead band"


def morf_oracle(X, y, is_discrete, name):
    """MoRF Eq. (1)-(2) with every instance the target once; returns ``(W, details)``.

    ``dW_i = sum_{j != i} c(i, j) f(d_ij) diff(a, i, j) / sum_{j != i} |f(d_ij)|``
    with ``c = -1`` for a hit and ``+1`` for a miss, and ``W = mean_i dW_i``.
    """
    per_target, _, _ = VARIANTS[name]
    diff, n, p = _exact_differences(X, is_discrete)
    diffs = {(i, j): diff(i, j) for i in range(n) for j in range(n) if i != j}
    dist = {key: sum(values, Fraction(0)) for key, values in diffs.items()}

    pairs = [dist[i, j] for i in range(n) for j in range(i + 1, n)]
    global_mu = sum(pairs, Fraction(0)) / len(pairs)
    global_var = sum(((d - global_mu) ** 2 for d in pairs), Fraction(0)) / len(pairs)

    scores = np.zeros(p)
    details = {"near": 0, "far": 0, "dead band": 0, "tie": 0, "global_mu": global_mu, "global_var": global_var}
    details["near_sets"] = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if per_target:
            mu = sum((dist[i, j] for j in others), Fraction(0)) / len(others)
            var = sum(((dist[i, j] - mu) ** 2 for j in others), Fraction(0)) / len(others)
        else:
            mu, var = global_mu, global_var
        numerator = np.zeros(p)
        denominator = 0.0
        near = set()
        for j in others:
            weight, region = _neighbour_weight(name, dist[i, j], mu, var)
            details[region] += 1
            if region == "near":
                near.add(j)
            c = -1.0 if y[i] == y[j] else 1.0
            numerator += c * weight * np.array([float(v) for v in diffs[i, j]])
            denominator += abs(weight)
        details["near_sets"].append(near)
        if denominator > 0:
            scores += numerator / denominator
    return scores / n, details


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
    y[: len(labels)] = labels  # every class present
    return X, y, np.array([False, False, True, True])


# --------------------------------------------------------------------------- equation oracle


@pytest.mark.parametrize("name", list(VARIANTS))
def test_binary_matches_morf_oracle(name):
    X, y, is_discrete = _mixed_data()
    expected, details = morf_oracle(X, y, is_discrete, name)
    model = _estimator(name, discrete_limit=3).fit(X, y)
    assert_array_equal(model.is_discrete_, is_discrete)
    assert_allclose(model.feature_importances_, expected, rtol=1e-10, atol=1e-13)
    assert np.all(np.abs(model.feature_importances_) <= 1.0)
    assert details["near"] > 0


@pytest.mark.parametrize("name", list(VARIANTS))
def test_multiclass_pools_every_other_class_as_a_miss(name):
    X, y, is_discrete = _mixed_data(n=42, seed=4, labels=(-2, 5, 11), weights=(0.5, 0.35, 0.15))
    expected, _ = morf_oracle(X, y, is_discrete, name)
    actual = _estimator(name, discrete_limit=3).fit(X, y).feature_importances_
    assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_label_values_do_not_matter(name):
    X, y, _ = _mixed_data(seed=2)
    reference = _estimator(name, discrete_limit=3).fit(X, y).feature_importances_
    relabelled = _estimator(name, discrete_limit=3).fit(X, np.where(y == 3, 0, 1)).feature_importances_
    assert_array_equal(relabelled, reference)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_continuous_only_and_discrete_only_inputs(name):
    X, y, _ = _mixed_data(seed=6)
    continuous = X[:, :2]
    expected, _ = morf_oracle(continuous, y, np.array([False, False]), name)
    assert_allclose(
        _estimator(name, discrete_limit=0).fit(continuous, y).feature_importances_, expected, rtol=1e-10, atol=1e-13
    )
    discrete = X[:, 2:]
    expected, _ = morf_oracle(discrete, y, np.array([True, True]), name)
    assert_allclose(
        _estimator(name, discrete_limit=3).fit(discrete, y).feature_importances_, expected, rtol=1e-10, atol=1e-13
    )


def test_global_mean_and_std_are_over_unique_pairs():
    X, y, is_discrete = _mixed_data(seed=1)
    _, details = morf_oracle(X, y, is_discrete, "SWRF")
    model = SWRF(discrete_limit=3, n_jobs=1).fit(X, y)
    assert_allclose(model.distance_mean_, float(details["global_mu"]), rtol=1e-14)
    assert_allclose(model.distance_std_, math.sqrt(details["global_var"]), rtol=1e-12)


def test_global_and_per_target_boundaries_differ():
    """The same data gives different near sets for SWRF and MultiSWRF."""
    X, y, is_discrete = _mixed_data(seed=5)
    swrf, global_details = morf_oracle(X, y, is_discrete, "SWRF")
    multi, target_details = morf_oracle(X, y, is_discrete, "MultiSWRF")
    assert global_details["near_sets"] != target_details["near_sets"]
    actual_swrf = SWRF(discrete_limit=3, n_jobs=1).fit(X, y).feature_importances_
    actual_multi = MultiSWRF(discrete_limit=3, n_jobs=1).fit(X, y).feature_importances_
    assert_allclose(actual_swrf, swrf, rtol=1e-10, atol=1e-13)
    assert_allclose(actual_multi, multi, rtol=1e-10, atol=1e-13)
    assert not np.allclose(actual_swrf, actual_multi, rtol=1e-3)


def test_dead_band_star_variant_sees_near_dead_band_and_far_instances():
    X, y, is_discrete = _mixed_data(seed=3)
    expected, details = morf_oracle(X, y, is_discrete, "MultiSWRFDB*")
    assert details["near"] > 0 and details["far"] > 0 and details["dead band"] > 0
    actual = MultiSWRF(use_star=True, dead_band=True, discrete_limit=3, n_jobs=1).fit(X, y).feature_importances_
    assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


# --------------------------------------------------------------------------- hand calculations

# Distances (range 3): d01 = 1/3, d02 = 1, d12 = 2/3.  Global mean 2/3, so pair
# (1, 2) lies exactly on the SWRF boundary.  Per target: mu_0 = 2/3, sigma_0 =
# 1/3; mu_1 = 1/2, sigma_1 = 1/6; mu_2 = 5/6, sigma_2 = 1/6.  In each case below
# the weighted instances of a target share one |weight|, so Eq. (1) is exact.
HAND_X = np.array([[0.0], [1.0], [3.0]])
HAND_Y = np.array([0, 0, 1])
HAND_EXPECTED = {
    # target 0: near hit (diff 1/3); target 1: near hit, tie; target 2: no near instance.
    "SWRF": (-1 / 3 - 1 / 3 + 0) / 3,
    # far instances keep diff with a negative weight: target 0 (-1/3 - 1)/2,
    # target 1 -1/3 (its other pair is a tie), target 2 -1 (far miss, diff 1).
    "SWRF*": (-2 / 3 - 1 / 3 - 1) / 3,
    "MultiSWRF": (-1 / 3 - 1 / 3 + 2 / 3) / 3,
    "MultiSWRF*": (-2 / 3 - 1 / 2 - 1 / 6) / 3,
    "MultiSWRFDB": (-1 / 3 - 1 / 3 + 2 / 3) / 3,
    "MultiSWRFDB*": (-2 / 3 - 1 / 2 - 1 / 6) / 3,
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_hand_calculated_scores(name):
    actual = _estimator(name, discrete_limit=0).fit(HAND_X, HAND_Y).feature_importances_
    assert_allclose(actual, [HAND_EXPECTED[name]], rtol=1e-13, atol=1e-15)
    expected, _ = morf_oracle(HAND_X, HAND_Y, np.array([False]), name)
    assert_allclose(expected, [HAND_EXPECTED[name]], rtol=1e-13, atol=1e-15)


def test_far_instances_score_diff_not_similarity():
    """SWRF* reverses diff for distant instances; scoring 1 - diff would give -1/6 for target 0."""
    model = SWRF(1, use_star=True, discrete_limit=0, n_jobs=1).fit(HAND_X, HAND_Y)
    assert_allclose(model.feature_importances_, [-2 / 3], rtol=1e-13)
    # Scoring similarity (far hits subtract 1 - diff, far misses add it) instead:
    # target 0 = (-1/3 + (1 - 1)) / 2, target 1 = -1/3, target 2 = 1 - 1 = 0.
    similarity_variant = (-1 / 6 - 1 / 3 + 0) / 3
    assert not np.isclose(model.feature_importances_[0], similarity_variant)


# --------------------------------------------------------------------------- exact boundaries

RANGES = [3, 5, 6, 7, 9]


def _integer_case(seed, mixed):
    rng = np.random.default_rng(seed)
    n, p = int(rng.integers(3, 9)), int(rng.integers(1, 4))
    X = np.column_stack([rng.integers(0, RANGES[int(rng.integers(len(RANGES)))] + 1, n) for _ in range(p)])
    y = rng.integers(0, 2, n)
    if len(np.unique(y)) < 2:
        y[0], y[1] = 0, 1
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


def _tie_seeds(name, wanted=12, search=3000):
    """Seeds whose exact boundaries coincide with at least one distance for ``name``."""
    found = []
    for mixed in (False, True):
        for seed in range(search):
            X, y, discrete = _integer_case(seed, mixed)
            if not _consistent(X, discrete):
                continue
            details = morf_oracle(X, y, discrete, name)[1]
            if details["tie"] > 0 and details["near"] > 0:
                found.append((mixed, seed))
            if len([case for case in found if case[0] == mixed]) == wanted // 2:
                break
    return found


TIE_CASES = {name: _tie_seeds(name) for name in ("SWRF", "MultiSWRF", "MultiSWRFDB")}


def test_boundary_search_finds_exact_ties():
    for name, cases in TIE_CASES.items():
        assert len(cases) >= 10, name


@pytest.mark.parametrize(
    "name,family,mixed,seed",
    [
        (name, family, mixed, seed)
        for family, cases in TIE_CASES.items()
        for mixed, seed in cases
        for name in VARIANTS
        if VARIANTS[name][0] == (family != "SWRF") and VARIANTS[name][2] == (family == "MultiSWRFDB")
    ],
)
def test_exact_boundary_ties_match_rational_oracle(name, family, mixed, seed):
    X, y, discrete = _integer_case(seed, mixed)
    expected, details = morf_oracle(X, y, discrete, name)
    assert details["tie"] > 0
    actual = _estimator(name, discrete_limit=_discrete_limit_for(X, discrete)).fit(X, y).feature_importances_
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


# Normalised rows (0, 1/3, 1), (1/3, 1, 0), (1, 0, 1/3): every pair is exactly 2
# apart, but 1/3 and 2/3 are not representable, so the float distances differ by
# rounding while sigma is zero in exact arithmetic.
EQUIDISTANT_X = np.array([[0.0, 1.0, 3.0], [1.0, 3.0, 0.0], [3.0, 0.0, 1.0]])


@pytest.mark.parametrize("name", list(VARIANTS))
def test_zero_spread_gives_zero_scores_not_rounding_noise(name):
    # Three equidistant instances, and two instances (a single pair).
    for X, y in ((EQUIDISTANT_X, [0, 0, 1]), (EQUIDISTANT_X[:2], [0, 1])):
        expected, details = morf_oracle(X, np.asarray(y), np.zeros(3, dtype=bool), name)
        assert details["near"] == 0 and details["far"] == 0
        actual = _estimator(name, discrete_limit=0).fit(X, y).feature_importances_
        assert_array_equal(actual, np.zeros(3))
        assert_array_equal(expected, np.zeros(3))


# --------------------------------------------------------------------------- invariances


@pytest.mark.parametrize("name", list(VARIANTS))
def test_constant_feature_scores_exactly_zero(name):
    X, y, _ = _mixed_data(seed=7)
    X = np.column_stack([X, np.full(len(y), 4.2)])
    model = _estimator(name, discrete_limit=3).fit(X, y)
    assert model.feature_importances_[-1] == 0.0


@pytest.mark.parametrize("name", list(VARIANTS))
def test_translation_and_category_relabelling_invariance(name):
    X, y, _ = _mixed_data(seed=8)
    X[:, :2] = np.round(X[:, :2] * 1000)  # integer-valued, so a 1e8 shift is exact
    reference = _estimator(name, discrete_limit=3).fit(X, y).feature_importances_
    shifted = X.copy()
    shifted[:, 0] += 1e8
    shifted[:, 2] += 2**25
    shifted[:, 3] = shifted[:, 3] * 7.0 - 2**25
    assert_array_equal(_estimator(name, discrete_limit=3).fit(shifted, y).feature_importances_, reference)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_thread_count_only_changes_reduction_order(name):
    X, y, _ = _mixed_data(n=160, seed=9)
    serial = _estimator(name, discrete_limit=3, n_jobs=1).fit(X, y).feature_importances_
    parallel = _estimator(name, discrete_limit=3, n_jobs=-1).fit(X, y).feature_importances_
    assert_allclose(parallel, serial, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", [name for name, (_, star, _) in VARIANTS.items() if not star])
def test_near_only_variants_rank_a_main_effect_first(name):
    """Star variants are excluded: reversing far updates penalises main effects, as for SURF*/MultiSURF*."""
    rng = np.random.default_rng(0)
    X = rng.random((240, 8))
    y = (X[:, 0] + 0.1 * rng.random(240) > 0.55).astype(int)
    model = _estimator(name, n_features_to_select=1, discrete_limit=0, n_jobs=-1).fit(X, y)
    assert model.top_features_[0] == 0


@pytest.mark.parametrize("name", list(VARIANTS))
def test_pure_two_way_interaction_ranks_first(name):
    """A pure XOR on two ternary SNP features with no main effect, among eight noise SNPs."""
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, (400, 10)).astype(np.float64)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    model = _estimator(name, n_features_to_select=2, n_jobs=-1).fit(X, y)
    assert set(model.top_features_) == {0, 1}


# --------------------------------------------------------------------------- behaviour and API


@pytest.mark.parametrize("cls", [SWRF, MultiSWRF])
def test_sklearn_api_compliance(cls):
    check_estimator(cls(n_jobs=1))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"backend": "gpu"}, ValueError),
        ({"backend": "metal"}, ValueError),
        ({"backend": "tpu"}, ValueError),
        ({"use_star": "yes"}, TypeError),
        ({"use_star": 1}, TypeError),
        ({"n_features_to_select": 0}, ValueError),
        ({"n_features_to_select": 1.5}, ValueError),
        ({"n_features_to_select": 9}, ValueError),
        ({"n_features_to_select": "2"}, TypeError),
        ({"discrete_limit": -1}, ValueError),
        ({"discrete_limit": 2.5}, TypeError),
        ({"n_jobs": 0}, ValueError),
    ],
)
@pytest.mark.parametrize("cls", [SWRF, MultiSWRF])
def test_parameter_validation(cls, kwargs, error):
    X, y, _ = _mixed_data()
    with pytest.raises(error):
        cls(**kwargs).fit(X, y)


def test_dead_band_must_be_a_bool():
    X, y, _ = _mixed_data()
    with pytest.raises(TypeError):
        MultiSWRF(dead_band=1).fit(X, y)


@pytest.mark.parametrize("cls", [SWRF, MultiSWRF])
def test_rejects_missing_values_continuous_targets_and_one_sample(cls):
    X, y, _ = _mixed_data()
    X_missing = X.copy()
    X_missing[2, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cls().fit(X_missing, y)
    with pytest.raises(ValueError, match="Unknown label type"):
        cls().fit(X, np.linspace(0.0, 1.0, len(y)))
    with pytest.raises(ValueError):
        cls().fit(X[:1], y[:1])


@pytest.mark.parametrize("cls", [SWRF, MultiSWRF])
def test_single_class_scores_zero(cls):
    X, _, _ = _mixed_data()
    model = cls(n_features_to_select=2).fit(X, np.full(len(X), 4))
    assert_array_equal(model.feature_importances_, np.zeros(X.shape[1]))
    assert model.top_features_.size == 2


@pytest.mark.parametrize("cls", [SWRF, MultiSWRF])
def test_transform_and_fitted_attributes(cls):
    X, y, _ = _mixed_data()
    model = cls(n_features_to_select=2, discrete_limit=3)
    reduced = model.fit_transform(X, y)
    assert model.effective_backend_ == "cpu"
    assert model.feature_importances_.dtype == np.float64
    assert_array_equal(model.top_features_, np.argsort(model.feature_importances_)[::-1][:2])
    assert_array_equal(reduced, X[:, model.top_features_])
    with pytest.raises(ValueError):
        model.transform(X[:, :3])


@pytest.mark.parametrize(
    "estimator, label",
    [
        (SWRF(), "SWRF"),
        (SWRF(use_star=True), "SWRF*"),
        (MultiSWRF(), "MultiSWRF"),
        (MultiSWRF(use_star=True), "MultiSWRF*"),
        (MultiSWRF(dead_band=True), "MultiSWRFDB"),
        (MultiSWRF(use_star=True, dead_band=True), "MultiSWRFDB*"),
    ],
)
def test_verbose_names_the_variant(estimator, label, capsys):
    X, y, _ = _mixed_data()
    estimator.set_params(verbose=True).fit(X, y)
    assert f"Running {label} on the CPU" in capsys.readouterr().out


@pytest.mark.parametrize("cls", [SWRF, MultiSWRF])
def test_in_pipeline_grid_search_and_turf(cls):
    X, y, _ = _mixed_data(n=90, seed=1)
    pipe = Pipeline([("select", cls(discrete_limit=3)), ("model", LogisticRegression())])
    search = GridSearchCV(
        pipe, {"select__n_features_to_select": [1, 2, 3], "select__use_star": [False, True]}, cv=3
    ).fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )
    turf = TuRF(cls(discrete_limit=3), n_features_to_select=2, pct_remove=0.25).fit(X, y)
    assert turf.top_features_.size == 2
