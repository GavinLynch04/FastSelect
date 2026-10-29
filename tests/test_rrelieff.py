"""RReliefF against Robnik-Šikonja and Kononenko (2003), Figure 3 and Eqs. (9)-(11).

The oracles below transcribe the published pseudocode line by line (including
line 15's ``m - N_dC``) and never call FastSelect internals.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_select import RReliefF, TuRF


def _rrelieff_reference(X, y, is_discrete, k, sigma=20.0, weighting="exponential"):
    """Figure 3 with ``m = n`` (every instance is the target once).

    Continuous columns and the prediction are range-normalised first, so
    Eq. (2) is ``|a - b|`` on the normalised values; discrete columns use
    Eq. (1).  Distances are summed in column order (the test matrices list
    continuous columns first), and equal distances keep the lower index.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    normalised = X.copy()
    for f in range(n_features):
        if not is_discrete[f]:
            span = X[:, f].max() - X[:, f].min()
            normalised[:, f] = (X[:, f] - X[:, f].min()) / span if span > 0 else 0.0
    y = np.asarray(y, dtype=np.float64)
    tau = (y - y.min()) / (y.max() - y.min())

    def diff(a, b):
        return [float(a[f] != b[f]) if is_discrete[f] else abs(a[f] - b[f]) for f in range(n_features)]

    if weighting == "constant":
        d1 = [1.0 / k] * k
    else:
        d1 = [math.exp(-((rank / sigma) ** 2)) for rank in range(1, k + 1)]  # Eq. (11)
    d = [value / sum(d1) for value in d1]  # Eq. (10)

    n_dc = 0.0
    n_da = np.zeros(n_features)
    n_dc_da = np.zeros(n_features)
    for i in range(n_samples):
        distances = []
        for j in range(n_samples):
            if j != i:
                total = 0.0
                for value in diff(normalised[i], normalised[j]):
                    total += value
                distances.append((total, j))
        nearest = [j for _, j in sorted(distances)[:k]]
        for rank, j in enumerate(nearest):
            diff_c = abs(tau[i] - tau[j])
            n_dc += diff_c * d[rank]  # line 6
            diff_a = np.array(diff(normalised[i], normalised[j]))
            n_da += diff_a * d[rank]  # line 8
            n_dc_da += diff_c * diff_a * d[rank]  # lines 9-10

    m = n_samples
    first = n_dc_da / n_dc if n_dc > 0 else np.zeros(n_features)
    second = (n_da - n_dc_da) / (m - n_dc) if m - n_dc > 1e-12 else np.zeros(n_features)
    return first - second  # line 15


def _rrelieff_exact_constant(X, y, is_discrete, k):
    """Figure 3 in rational arithmetic with constant influence ``d = 1/k``."""
    rows = [[Fraction(int(v)) for v in row] for row in np.asarray(X)]
    n_samples, n_features = len(rows), len(rows[0])
    low = [min(r[f] for r in rows) for f in range(n_features)]
    span = [max(r[f] for r in rows) - low[f] for f in range(n_features)]
    target = [Fraction(int(v)) for v in y]
    t_span = max(target) - min(target)

    def diff(a, b, f):
        if is_discrete[f]:
            return Fraction(int(a[f] != b[f]))
        return abs(a[f] - b[f]) / span[f] if span[f] else Fraction(0)

    n_dc, n_da, n_dc_da = Fraction(0), [Fraction(0)] * n_features, [Fraction(0)] * n_features
    for i in range(n_samples):
        ranked = sorted(
            (sum(diff(rows[i], rows[j], f) for f in range(n_features)), j) for j in range(n_samples) if j != i
        )
        for _, j in ranked[:k]:
            diff_c = abs(target[i] - target[j]) / t_span
            n_dc += diff_c / k
            for f in range(n_features):
                n_da[f] += diff(rows[i], rows[j], f) / k
                n_dc_da[f] += diff_c * diff(rows[i], rows[j], f) / k
    m = n_samples
    return [n_dc_da[f] / n_dc - (n_da[f] - n_dc_da[f]) / (m - n_dc) for f in range(n_features)]


def _regression_data(n=60, seed=3):
    """Two continuous columns with non-dyadic ranges, then two discrete ones."""
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.random(n) * 7.3 + 1.1,
            rng.random(n) * 3.7,
            rng.integers(0, 3, n),
            rng.integers(0, 2, n),
        ]
    )
    y = 2.0 * X[:, 0] + 1.5 * X[:, 2] + 0.2 * rng.random(n)
    return X, y, np.array([False, False, True, True])


@pytest.mark.parametrize("k, sigma", [(1, 20.0), (5, 20.0), (12, 3.0), (30, 0.5)])
def test_matches_figure_3_with_exponential_rank_weights(k, sigma):
    X, y, is_discrete = _regression_data()
    expected = _rrelieff_reference(X, y, is_discrete, k, sigma)
    actual = RReliefF(n_neighbors=k, sigma=sigma, discrete_limit=3).fit(X, y).feature_importances_
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("k", [1, 3, 10])
def test_matches_figure_3_with_constant_influence(k):
    X, y, is_discrete = _regression_data(seed=8)
    expected = _rrelieff_reference(X, y, is_discrete, k, weighting="constant")
    actual = RReliefF(n_neighbors=k, weighting="constant", discrete_limit=3).fit(X, y).feature_importances_
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


# Integer columns with dyadic ranges (8 and 4), so every Eq. (2) difference and
# every distance is exact in float64 and the rank ties below are genuine ties.
TIED_X = np.array(
    [
        [0, 0, 1],
        [2, 0, 0],
        [2, 4, 1],
        [4, 2, 0],
        [6, 4, 1],
        [8, 2, 0],
        [8, 0, 1],
        [4, 4, 0],
    ]
)
TIED_Y = np.array([0, 3, 1, 4, 9, 2, 7, 5])
TIED_DISCRETE = np.array([False, False, True])


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7])
def test_rank_ties_keep_the_lower_index_exactly(k):
    """Constant influence makes Figure 3 rational, so the oracle is exact."""
    exact = _rrelieff_exact_constant(TIED_X, TIED_Y, TIED_DISCRETE, k)
    actual = RReliefF(n_neighbors=k, weighting="constant", discrete_limit=2).fit(TIED_X, TIED_Y)
    assert_array_equal(actual.is_discrete_, TIED_DISCRETE)
    assert_allclose(actual.feature_importances_, [float(v) for v in exact], rtol=1e-13, atol=1e-14)
    assert_allclose(
        actual.feature_importances_,
        _rrelieff_reference(TIED_X, TIED_Y, TIED_DISCRETE, k, weighting="constant"),
        rtol=1e-13,
        atol=1e-14,
    )


def test_empty_different_prediction_group_contributes_zero():
    """Two far-apart clusters with one target each: no neighbour pair differs in y."""
    # Within-cluster distances are at most 0.92; across clusters at least 1.96.
    X = np.array([[0.0, 0.3, 0], [0.1, 0.9, 0], [0.2, 0.0, 0], [10.0, 1.0, 1], [10.1, 0.4, 1], [10.2, 0.6, 1]])
    y = np.array([5.0, 5.0, 5.0, 9.0, 9.0, 9.0])
    model = RReliefF(n_neighbors=2, weighting="constant", discrete_limit=2).fit(X, y)
    expected = _rrelieff_reference(X, y, np.array([False, False, True]), 2, weighting="constant")
    assert_allclose(model.feature_importances_, expected, rtol=1e-12, atol=1e-15)
    assert np.all(model.feature_importances_ <= 0.0)  # only the same-prediction term remains


def test_empty_same_prediction_group_contributes_zero():
    """Every nearest neighbour has the opposite extreme target (diff(tau) = 1)."""
    X = np.array([[0.0, 1.0], [0.5, 1.0], [20.0, 3.0], [20.5, 4.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model = RReliefF(n_neighbors=1, weighting="constant", discrete_limit=0).fit(X, y)
    expected = _rrelieff_reference(X, y, np.array([False, False]), 1, weighting="constant")
    assert_allclose(model.feature_importances_, expected, rtol=1e-12, atol=1e-15)
    assert np.all(model.feature_importances_ >= 0.0)


def test_relevant_features_outrank_noise_and_constant_features_score_zero():
    rng = np.random.default_rng(0)
    X = rng.random((300, 6))
    X[:, 4] = 3.5
    y = 3 * X[:, 0] + np.sin(6 * X[:, 1]) + 0.05 * rng.random(300)
    model = RReliefF(n_features_to_select=2).fit(X, y)
    assert set(model.top_features_) == {0, 1}
    assert model.feature_importances_[4] == 0.0
    assert np.all(model.feature_importances_[[2, 3, 5]] < model.feature_importances_[[0, 1]].min())


def test_translation_scaling_and_relabelling_invariance():
    X, y, _ = _regression_data(seed=5)
    X[:, :2] = np.round(X[:, :2] * 1000)  # integer-valued, so a 1e8 shift is exact
    reference = RReliefF(n_neighbors=8, discrete_limit=3).fit(X, y).feature_importances_
    shifted = X.copy()
    shifted[:, 0] += 1e8
    shifted[:, 2] += 2**25
    assert_array_equal(RReliefF(n_neighbors=8, discrete_limit=3).fit(shifted, y).feature_importances_, reference)
    assert_allclose(
        RReliefF(n_neighbors=8, discrete_limit=3).fit(X, 1000.0 * y + 1e6).feature_importances_,
        reference,
        rtol=1e-9,
        atol=1e-12,
    )


def test_thread_count_only_changes_reduction_order():
    X, y, _ = _regression_data(n=150, seed=9)
    serial = RReliefF(n_neighbors=10, discrete_limit=3, n_jobs=1).fit(X, y).feature_importances_
    parallel = RReliefF(n_neighbors=10, discrete_limit=3, n_jobs=-1).fit(X, y).feature_importances_
    assert_allclose(parallel, serial, rtol=1e-12, atol=1e-15)


def test_large_sigma_approaches_constant_influence():
    X, y, _ = _regression_data(seed=4)
    wide = RReliefF(n_neighbors=10, sigma=1e6, discrete_limit=3).fit(X, y).feature_importances_
    constant = RReliefF(n_neighbors=10, weighting="constant", discrete_limit=3).fit(X, y).feature_importances_
    assert_allclose(wide, constant, rtol=1e-9, atol=1e-12)


def test_tiny_sigma_does_not_underflow():
    X, y, is_discrete = _regression_data(seed=6)
    model = RReliefF(n_neighbors=10, sigma=0.01, discrete_limit=3).fit(X, y)
    assert np.all(np.isfinite(model.feature_importances_))
    nearest_only = RReliefF(n_neighbors=1, discrete_limit=3).fit(X, y).feature_importances_
    assert_allclose(model.feature_importances_, nearest_only, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_neighbors": 60}, ValueError),
        ({"n_neighbors": 0}, ValueError),
        ({"n_neighbors": 2.0}, TypeError),
        ({"sigma": 0.0}, ValueError),
        ({"sigma": float("nan")}, ValueError),
        ({"sigma": True}, ValueError),
        ({"weighting": "linear"}, ValueError),
        ({"backend": "gpu"}, ValueError),
        ({"backend": "metal"}, ValueError),
        ({"n_features_to_select": 0}, ValueError),
        ({"n_features_to_select": 1.5}, ValueError),
        ({"n_jobs": 0}, ValueError),
    ],
)
def test_parameter_validation(kwargs, error):
    X, y, _ = _regression_data()
    with pytest.raises(error):
        RReliefF(**{"n_neighbors": 5, **kwargs}).fit(X, y)


def test_rejects_constant_target_and_missing_values():
    X, y, _ = _regression_data()
    with pytest.raises(ValueError, match="two distinct values"):
        RReliefF(n_neighbors=5).fit(X, np.full(len(y), 2.5))
    X_missing = X.copy()
    X_missing[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RReliefF(n_neighbors=5).fit(X_missing, y)


def test_transform_and_fitted_attributes():
    X, y, _ = _regression_data()
    model = RReliefF(n_features_to_select=2, n_neighbors=5, discrete_limit=3)
    reduced = model.fit_transform(X, y)
    assert model.effective_backend_ == "cpu"
    assert model.feature_importances_.dtype == np.float64
    assert_array_equal(reduced, X[:, model.top_features_])
    assert np.all(np.abs(model.feature_importances_) <= 1.0)


def test_sklearn_api_compliance():
    check_estimator(RReliefF(n_neighbors=3))


def test_in_pipeline_grid_search_and_turf():
    X, y, _ = _regression_data(n=90, seed=1)
    pipe = Pipeline([("select", RReliefF(n_neighbors=5, discrete_limit=3)), ("model", Ridge())])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2, 3]}, cv=3).fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )

    turf = TuRF(RReliefF(n_neighbors=5, discrete_limit=3), n_features_to_select=2, pct_remove=0.25).fit(X, y)
    assert turf.top_features_.size == 2
