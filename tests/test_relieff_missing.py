"""ReliefF with missing discrete values, Robnik-Šikonja and Kononenko (2003), Eqs. (1)-(4).

The oracle re-derives every difference from the paper and never calls
FastSelect internals: Eq. (3) when one value is unknown, Eq. (4) when both are,
conditional probabilities as relative frequencies of the known training values,
the same ``diff`` for distances and scores, and ReliefF's prior-weighted update.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fast_select import SURF, MultiSURF, ReliefF, TuRF


def _relieff_missing_reference(X, y, is_discrete, k):
    """ReliefF (m = n) with the paper's missing-value ``diff``.

    Continuous columns are range-normalised first and must be complete; the
    test matrices list them before the discrete ones, matching the column order
    in which the kernel sums a distance.  Equal distances keep the lower index;
    a neighbour group smaller than ``k`` is averaged over the neighbours found.
    Explicit conventions for cases the paper leaves open: a class with no known
    value uses the feature's overall frequencies; an all-missing feature has
    difference 0.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    classes, labels = np.unique(y, return_inverse=True)
    priors = np.bincount(labels) / n_samples

    normalised = X.copy()
    for f in range(n_features):
        if not is_discrete[f]:
            span = X[:, f].max() - X[:, f].min()
            normalised[:, f] = (X[:, f] - X[:, f].min()) / span if span > 0 else 0.0

    probability = {}  # probability[f][c][value] = P(value | class c), over known values
    for f in np.flatnonzero(is_discrete):
        known = ~np.isnan(X[:, f])
        values = sorted(set(X[known, f]))
        overall = {v: np.count_nonzero(X[known, f] == v) / known.sum() for v in values} if known.any() else {}
        probability[f] = {}
        for c in range(len(classes)):
            in_class = known & (labels == c)
            if in_class.sum():
                probability[f][c] = {v: np.count_nonzero(X[in_class, f] == v) / in_class.sum() for v in values}
            else:
                probability[f][c] = overall

    def diff(f, i, j):
        a, b = normalised[i, f], normalised[j, f]
        if not is_discrete[f]:
            return abs(a - b)  # Eq. (2)
        if not probability[f][labels[i]]:
            return 0.0  # no known value of this feature anywhere
        if np.isnan(a) and np.isnan(b):  # Eq. (4)
            agreement = 0.0
            for v in sorted(probability[f][labels[i]]):
                agreement += probability[f][labels[i]][v] * probability[f][labels[j]][v]
            return 1.0 - agreement
        if np.isnan(a):  # Eq. (3), the unknown value belongs to instance i
            return 1.0 - probability[f][labels[i]][b]
        if np.isnan(b):
            return 1.0 - probability[f][labels[j]][a]
        return float(a != b)  # Eq. (1)

    scores = np.zeros(n_features)
    for i in range(n_samples):
        distance = {}
        for j in range(n_samples):
            if j != i:
                total = 0.0
                for f in range(n_features):
                    total += diff(f, i, j)
                distance[j] = total
        for c in range(len(classes)):
            group = sorted((j for j in distance if labels[j] == c), key=lambda j: (distance[j], j))[:k]
            if not group:
                continue
            if c == labels[i]:
                weight = -1.0 / (n_samples * len(group))
            else:
                weight = priors[c] / (1.0 - priors[labels[i]]) / (n_samples * len(group))
            for j in group:
                scores += weight * np.array([diff(f, i, j) for f in range(n_features)])
    return scores


def _missing_data(n=48, seed=0, missing_rate=0.2):
    """Two continuous columns (non-dyadic ranges) then three discrete ones with NaNs."""
    rng = np.random.default_rng(seed)
    y = np.repeat([4, -2, 9], [n // 2, n // 3, n - n // 2 - n // 3])  # imbalanced multiclass, non-0/1 labels
    X = np.column_stack(
        [
            rng.random(n) * 6.1 + 0.3,
            rng.random(n) * 2.9,
            (y == 4) * 2 + rng.integers(0, 2, n),  # informative genotype-like code
            rng.integers(0, 3, n),
            rng.integers(0, 2, n),
        ]
    ).astype(np.float64)
    X[:, 2:][rng.random((n, 3)) < missing_rate] = np.nan
    return X, y, np.array([False, False, True, True, True])


def _fit(X, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ReliefF(**{"discrete_limit": 4, "backend": "cpu", **kwargs}).fit(X, y)


@pytest.mark.parametrize("k", [1, 3, 8])
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_the_papers_missing_value_definition(k, seed):
    X, y, is_discrete = _missing_data(seed=seed)
    model = _fit(X, y, n_neighbors=k)
    assert_array_equal(model.is_discrete_, is_discrete)
    assert model.effective_backend_ == "cpu"
    assert_allclose(
        model.feature_importances_,
        _relieff_missing_reference(X, y, is_discrete, k),
        rtol=1e-6,
        atol=1e-7,
    )


def test_one_and_both_missing_differences_by_hand():
    """Four instances, one discrete feature: Eqs. (3)-(4) evaluated by hand.

    Class 0 knows the values {1, 1} -> P(1|0) = 1; class 1 knows {0} -> P(0|1) = 1.
    With k = 1 the nearest hit and miss of every instance are its only peers.
    """
    X = np.array([[1.0], [1.0], [0.0], [np.nan]])
    y = np.array([0, 0, 1, 1])
    # Instance 3 is class 1 with its value missing: against instances 0 and 1
    # (value 1) Eq. (3) gives 1 - P(1|1) = 1, against instance 2 (value 0) it
    # gives 1 - P(0|1) = 0. So every nearest hit agrees (diff 0) and every
    # nearest miss differs (diff 1, weight P(C)/(1-P(class)) / (m*k) = 1/4).
    expected = _relieff_missing_reference(X, y, np.array([True]), 1)
    assert_allclose(expected, [1.0])
    assert_allclose(_fit(X, y, n_neighbors=1).feature_importances_, expected, rtol=1e-6)

    both = np.array([[np.nan], [1.0], [np.nan], [0.0], [1.0], [0.0]])
    labels = np.array([0, 0, 1, 1, 0, 1])
    assert_allclose(
        _fit(both, labels, n_neighbors=2).feature_importances_,
        _relieff_missing_reference(both, labels, np.array([True]), 2),
        rtol=1e-6,
        atol=1e-7,
    )


# Dyadic probabilities: every class knows 4 values per discrete feature, so
# Eqs. (3)-(4) are multiples of 1/16 and the continuous range is 8; every
# distance is exact, and the ties below are genuine.
TIED_X = np.array(
    [
        [0, 1, 0],
        [2, np.nan, 1],
        [4, 0, np.nan],
        [6, 1, 1],
        [8, 0, 0],
        [0, 1, 0],
        [2, np.nan, 1],
        [4, 1, np.nan],
        [6, 0, 0],
        [8, 0, 1],
    ]
)
TIED_Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


@pytest.mark.parametrize("k", [1, 2, 4])
def test_exact_distance_ties_keep_the_lower_index(k):
    is_discrete = np.array([False, True, True])
    model = _fit(TIED_X, TIED_Y, n_neighbors=k, discrete_limit=2)
    assert_array_equal(model.is_discrete_, is_discrete)
    assert_allclose(
        model.feature_importances_, _relieff_missing_reference(TIED_X, TIED_Y, is_discrete, k), rtol=1e-6, atol=1e-7
    )


def test_class_without_known_values_uses_overall_frequencies():
    X, y, is_discrete = _missing_data(seed=3, missing_rate=0.1)
    X[y == -2, 3] = np.nan  # class -2 never observes feature 3
    model = _fit(X, y, n_neighbors=3)
    assert_allclose(model.feature_importances_, _relieff_missing_reference(X, y, is_discrete, 3), rtol=1e-6, atol=1e-7)


def test_all_missing_feature_scores_zero_and_leaves_the_rest_unchanged():
    X, y, is_discrete = _missing_data(seed=4)
    with_empty = np.column_stack([X, np.full(len(y), np.nan)])
    model = _fit(with_empty, y, n_neighbors=3)
    assert model.is_discrete_[-1]
    assert model.feature_importances_[-1] == 0.0
    assert_allclose(model.feature_importances_[:-1], _fit(X, y, n_neighbors=3).feature_importances_, rtol=1e-6)


def test_category_relabelling_invariance_with_missing_values():
    X, y, _ = _missing_data(seed=5)
    shifted = X.copy()
    shifted[:, 2:] += 2**25
    assert_array_equal(
        _fit(shifted, y, n_neighbors=3).feature_importances_, _fit(X, y, n_neighbors=3).feature_importances_
    )


def test_complete_data_takes_the_existing_path_and_the_same_definition():
    """Without NaNs the unchanged float32 path runs, and the oracle reduces to ReliefF."""
    X, y, is_discrete = _missing_data(seed=6, missing_rate=0.0)
    model = _fit(X, y, n_neighbors=3)
    assert not np.isnan(X).any()
    assert model.feature_importances_.dtype == np.float32
    assert_allclose(model.feature_importances_, _relieff_missing_reference(X, y, is_discrete, 3), rtol=1e-5, atol=1e-7)


def test_thread_count_only_changes_reduction_order():
    X, y, _ = _missing_data(n=120, seed=7)
    assert_allclose(
        _fit(X, y, n_neighbors=5, n_jobs=1).feature_importances_,
        _fit(X, y, n_neighbors=5, n_jobs=-1).feature_importances_,
        rtol=1e-6,
        atol=1e-8,
    )


def test_missing_values_in_continuous_features_are_rejected():
    X, y, _ = _missing_data(seed=8)
    X[2, 0] = np.nan
    with pytest.raises(ValueError, match=r"only in discrete features.*\[0\]"):
        _fit(X, y, n_neighbors=3)
    X_inf = _missing_data(seed=8)[0]
    X_inf[2, 3] = np.inf
    with pytest.raises(ValueError, match="infinity|inf"):
        _fit(X_inf, y, n_neighbors=3)


@pytest.mark.parametrize("backend", ["gpu", "metal"])
def test_gpu_backends_reject_missing_values(backend):
    X, y, _ = _missing_data(seed=9)
    with pytest.raises(ValueError, match="CPU only"):
        ReliefF(n_neighbors=3, discrete_limit=4, backend=backend).fit(X, y)


def test_auto_backend_runs_missing_values_on_cpu():
    X, y, _ = _missing_data(seed=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = ReliefF(n_neighbors=3, discrete_limit=4, backend="auto").fit(X, y)
    assert model.effective_backend_ == "cpu"


def test_transform_and_turf_accept_missing_values():
    X, y, _ = _missing_data(n=90, seed=11)
    model = _fit(X, y, n_features_to_select=2, n_neighbors=3)
    assert_array_equal(model.transform(X), X[:, model.top_features_])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        turf = TuRF(ReliefF(n_neighbors=3, discrete_limit=4, backend="cpu"), n_features_to_select=2).fit(X, y)
    assert turf.top_features_.size == 2
    assert_array_equal(turf.transform(X), X[:, turf.top_features_])


def test_transform_rejects_missing_values_in_continuous_features():
    X, y, _ = _missing_data(seed=13)
    model = _fit(X, y, n_neighbors=3)
    bad = X.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="continuous during fit"):
        model.transform(bad)


@pytest.mark.parametrize("cls", [SURF, MultiSURF])
def test_surf_family_still_rejects_missing_values(cls):
    """No SURF-family paper defines missing values, so NaN stays an input error."""
    X, y, _ = _missing_data(seed=12)
    with pytest.raises(ValueError, match="NaN"):
        cls(backend="cpu").fit(X, y)
    with pytest.raises(ValueError, match="NaN"):
        TuRF(cls(backend="cpu"), n_features_to_select=2).fit(X, y)
