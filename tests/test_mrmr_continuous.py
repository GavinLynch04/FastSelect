"""FCD / FCQ mRMR against Ding and Peng (2005), Eqs. (8)-(10) and Table 1.

Relevance is the one-way ANOVA F statistic (Eq. 8), redundancy the mean
absolute Pearson correlation with the selected features (Eq. 10), combined by
difference (FCD) or quotient (FCQ), with the first feature the most relevant.
The oracle computes every F and every squared correlation as an exact
rational; it never calls FastSelect internals.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_select import mRMR


def _exact_f(column, y):
    values = [Fraction(float(v)) for v in column]
    classes = sorted(set(y))
    n, k = len(values), len(classes)
    if min(values) == max(values):
        return 0.0
    mean = sum(values, Fraction(0)) / n
    groups = {c: [v for v, label in zip(values, y) if label == c] for c in classes}
    means = {c: sum(g, Fraction(0)) / len(g) for c, g in groups.items()}
    between = sum((len(groups[c]) * (means[c] - mean) ** 2 for c in classes), Fraction(0))
    within = sum(((v - means[c]) ** 2 for c in classes for v in groups[c]), Fraction(0))
    if within == 0:
        return math.inf if between > 0 else 0.0
    return float((between / (k - 1)) / (within / (n - k)))


def _exact_abs_correlation(a, b):
    a = [Fraction(float(v)) for v in a]
    b = [Fraction(float(v)) for v in b]
    if min(a) == max(a) or min(b) == max(b):
        return 0.0
    ma, mb = sum(a, Fraction(0)) / len(a), sum(b, Fraction(0)) / len(b)
    sab = sum(((x - ma) * (z - mb) for x, z in zip(a, b)), Fraction(0))
    saa = sum(((x - ma) ** 2 for x in a), Fraction(0))
    sbb = sum(((z - mb) ** 2 for z in b), Fraction(0))
    return math.sqrt(float(sab * sab / (saa * sbb)))


def mrmr_oracle(X, y, k, method):
    """Ding and Peng's linear incremental search (their Sec. 3), with the documented tie rule."""
    X = np.asarray(X)
    y = list(np.asarray(y))
    p = X.shape[1]
    relevance = np.array([_exact_f(X[:, f], y) for f in range(p)])
    corr = {}

    def c(i, j):
        if (i, j) not in corr:
            corr[i, j] = corr[j, i] = _exact_abs_correlation(X[:, i], X[:, j])
        return corr[i, j]

    selected = [int(np.argmax(relevance))]
    while len(selected) < k:
        candidates = [f for f in range(p) if f not in selected]
        mean_redundancy = np.array([sum(c(f, s) for s in selected) / len(selected) for f in candidates])
        rel = relevance[candidates]
        if method == "FCD":
            scores = rel - mean_redundancy
        else:
            scores = np.array([r / m if m > 0 else (math.inf if r > 0 else 0.0) for r, m in zip(rel, mean_redundancy)])
        best = np.max(scores)
        tied = [i for i in range(len(candidates)) if np.isclose(scores[i], best, atol=1e-12)]
        choice = min(tied, key=lambda i: (mean_redundancy[i], candidates[i]))
        selected.append(candidates[choice])
    rows = np.array([[c(s, f) for f in range(p)] for s in selected])  # c(s, s) is 1, or 0 if constant
    return relevance, np.array(selected), rows


def _data(n=60, p=12, seed=0, labels=("a", "b", "c"), weights=(0.5, 0.3, 0.2)):
    rng = np.random.default_rng(seed)
    y = rng.choice(np.asarray(labels), size=n, p=np.asarray(weights))
    y[: len(labels)] = labels
    signal = np.searchsorted(np.asarray(labels), y).astype(float)
    X = rng.random((n, p)) * np.linspace(1.3, 9.7, p)
    X[:, 0] += 1.1 * signal
    X[:, 1] = 0.8 * X[:, 0] + 0.3 * rng.random(n)  # redundant with feature 0
    X[:, 2] += 0.9 * signal
    X[:, 3] -= 0.6 * signal
    return X, y


# --------------------------------------------------------------------------- equation oracle


@pytest.mark.parametrize("method", ["FCD", "FCQ"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_ding_and_peng_oracle(method, seed):
    X, y = _data(seed=seed)
    relevance, selected, rows = mrmr_oracle(X, y, 6, method)
    model = mRMR(6, method=method, n_jobs=1).fit(X, y)
    assert_allclose(model.relevance_scores_, relevance, rtol=1e-12)
    assert_array_equal(model.top_features_, selected)
    assert_allclose(model.selected_correlations_, rows, rtol=1e-12, atol=1e-14)
    assert_array_equal(model.feature_importances_, model.relevance_scores_)
    assert model.effective_backend_ == "cpu"


def test_two_class_f_is_the_squared_t_statistic():
    X, y = _data(labels=(3, 7), weights=(0.6, 0.4))
    F = mRMR(1, method="FCQ").fit(X, y).relevance_scores_
    for f in range(4):
        a, b = X[y == 3, f], X[y == 7, f]
        pooled = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
        t = (a.mean() - b.mean()) / math.sqrt(pooled * (1 / a.size + 1 / b.size))
        assert_allclose(F[f], t * t, rtol=1e-11)


def test_f_agrees_with_scikit_learn_on_non_constant_columns():
    X, y = _data(seed=4)
    assert_allclose(mRMR(1, method="FCD").fit(X, y).relevance_scores_, f_classif(X, y)[0], rtol=1e-11)


def test_correlations_agree_with_numpy():
    X, y = _data(seed=5)
    model = mRMR(4, method="FCQ").fit(X, y)
    expected = np.abs(np.corrcoef(X.T))[model.top_features_]
    assert_allclose(model.selected_correlations_, expected, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- edge cases


def test_constant_feature_is_irrelevant_and_uncorrelated_exactly():
    """0.1 repeated is constant, but its rounded mean is not exactly 0.1."""
    X, y = _data(seed=6)
    X[:, 5] = 0.1
    model = mRMR(4, method="FCQ").fit(X, y)
    assert model.relevance_scores_[5] == 0.0
    assert np.all(model.selected_correlations_[:, 5] == 0.0)
    assert f_classif(X, y)[0][5] != 0.0  # scikit-learn reports rounding noise as signal here


def test_feature_constant_within_classes_has_infinite_f_and_is_chosen_first():
    X, y = _data(seed=7)
    X[:, 9] = np.searchsorted(np.array(["a", "b", "c"]), y) * 2.5 + 1.0
    for method in ("FCD", "FCQ"):
        model = mRMR(3, method=method).fit(X, y)
        assert model.relevance_scores_[9] == np.inf
        assert model.top_features_[0] == 9


def test_fcq_zero_redundancy_rules():
    """Bit columns of 0..31 are exactly uncorrelated with each other.

    The class is ``2 * bit0 + bit1``: features 0 and 1 (bits 0 and 1) are
    constant within classes (``F = inf``) and orthogonal, feature 2 (bit 2) is
    balanced in every class (``F = 0``) and orthogonal to both, feature 3 is
    noise. After feature 0, feature 1 scores ``inf / 0 = inf``; after that,
    feature 2 scores ``0 / 0 = 0`` and loses to the noise feature's positive
    quotient.
    """
    rows = np.arange(32)
    bits = np.column_stack([((rows >> b) & 1) * 2.0 - 1.0 for b in range(3)])
    noise = np.random.default_rng(8).random(32)
    X = np.column_stack([bits, noise])
    y = 2 * ((rows >> 0) & 1) + ((rows >> 1) & 1)
    model = mRMR(3, method="FCQ").fit(X, y)
    assert_array_equal(model.relevance_scores_[:3], [np.inf, np.inf, 0.0])
    assert model.relevance_scores_[3] > 0.0
    assert_array_equal(model.selected_correlations_[0, :3], [1.0, 0.0, 0.0])
    assert_array_equal(model.top_features_, [0, 1, 3])


def test_exact_ties_take_lower_redundancy_then_lower_index():
    X, y = _data(seed=9)
    X = np.column_stack([X, X[:, 2], X[:, 2]])  # features 12 and 13 duplicate feature 2
    for method in ("FCD", "FCQ"):
        model = mRMR(X.shape[1], method=method).fit(X, y)
        order = list(model.top_features_)
        assert order.index(2) < order.index(12) < order.index(13)
        relevance, selected, _ = mrmr_oracle(X, y, X.shape[1], method)
        assert_array_equal(model.top_features_, selected)


# --------------------------------------------------------------------------- invariances


@pytest.mark.parametrize("method", ["FCD", "FCQ"])
def test_affine_invariance_and_label_values(method):
    X, y = _data(seed=10)
    X = np.round(X * 1000)
    reference = mRMR(6, method=method).fit(X, y)
    moved = X * np.linspace(0.5, 7.0, X.shape[1]) + 1e8
    other = mRMR(6, method=method).fit(moved, np.where(y == "a", 11, np.where(y == "b", -4, 9)))
    assert_array_equal(other.top_features_, reference.top_features_)
    assert_allclose(other.relevance_scores_, reference.relevance_scores_, rtol=1e-9)


def test_thread_count_does_not_change_anything():
    X, y = _data(n=300, p=40, seed=11)
    serial = mRMR(10, method="FCQ", n_jobs=1).fit(X, y)
    parallel = mRMR(10, method="FCQ", n_jobs=-1).fit(X, y)
    assert_array_equal(parallel.top_features_, serial.top_features_)
    assert_array_equal(parallel.selected_correlations_, serial.selected_correlations_)


def test_fcq_skips_an_exact_duplicate_of_the_first_feature():
    X, y = _data(seed=12)
    X = np.column_stack([X, X[:, 0]])
    model = mRMR(3, method="FCQ").fit(X, y)
    assert model.top_features_[0] in (0, X.shape[1] - 1)
    assert not {0, X.shape[1] - 1} <= set(model.top_features_[:2])


# --------------------------------------------------------------------------- API


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"method": "FCD", "backend": "gpu"}, ValueError),
        ({"method": "FCQ", "backend": "metal"}, ValueError),
        ({"method": "FCQ", "n_jobs": 0}, ValueError),
        ({"method": "FCX"}, ValueError),
    ],
)
def test_parameter_validation(kwargs, error):
    X, y = _data()
    with pytest.raises(error):
        mRMR(2, **kwargs).fit(X, y)


def test_rejects_one_class_too_few_samples_nan_and_strings():
    X, y = _data()
    with pytest.raises(ValueError, match="two classes"):
        mRMR(2, method="FCQ").fit(X, np.full(len(y), "a"))
    with pytest.raises(ValueError, match="more samples than classes"):
        mRMR(2, method="FCQ").fit(X[:3], np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match="Unknown label type"):
        mRMR(2, method="FCQ").fit(X, np.linspace(0, 1, len(y)))
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        mRMR(2, method="FCD").fit(X_nan, y)
    with pytest.raises(ValueError):
        mRMR(2, method="FCD").fit(np.array([["x", "y"], ["z", "w"], ["x", "w"]]), [0, 1, 0])


@pytest.mark.parametrize("method", ["FCD", "FCQ"])
def test_sklearn_api_compliance(method):
    check_estimator(mRMR(2, method=method, n_jobs=1))


def test_in_pipeline_and_grid_search():
    X, y = _data(n=120, seed=13)
    pipe = Pipeline([("select", mRMR(2, method="FCQ")), ("model", LogisticRegression(max_iter=500))])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 3], "select__method": ["FCD", "FCQ"]}, cv=3).fit(
        X, y
    )
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )
