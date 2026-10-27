"""Independent equations and contracts for the second algorithm admission batch."""

from collections import Counter
from functools import partial

import numpy as np
import pytest
from scipy.special import digamma
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression as sklearn_mi
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.utils.estimator_checks import check_estimator

from fast_select import CIFE, DISR, MIFS, GiniIndex, mutual_info_regression


def h(*variables):
    probabilities = np.array(list(Counter(zip(*variables)).values())) / len(variables[0])
    return -np.sum(probabilities * np.log2(probabilities))


def information_oracle(X, y, criterion, k, beta=0.5):
    relevance = np.array([h(f) + h(y) - h(f, y) for f in X.T])
    remaining = list(range(X.shape[1]))
    selected, scores = [], []
    while len(selected) < k:
        objectives = []
        for f in remaining:
            x = X[:, f]
            if not selected:
                value = relevance[f]
            elif criterion == "DISR":
                value = 0.0
                for s in selected:
                    joint = h(x, X[:, s], y)
                    value += (h(x, X[:, s]) + h(y) - joint) / joint if joint else 0.0
            elif criterion == "MIFS":
                value = relevance[f] - beta * sum(h(x) + h(X[:, s]) - h(x, X[:, s]) for s in selected)
            else:
                value = relevance[f]
                for s in selected:
                    peer = X[:, s]
                    value -= h(x) + h(peer) - h(x, peer)
                    value += h(x, y) + h(peer, y) - h(y) - h(x, peer, y)
            objectives.append(value)
        i = int(np.argmax(objectives))
        chosen = remaining.pop(i)
        selected.append(chosen)
        scores.append(objectives[i])
    return np.asarray(selected), np.asarray(scores), relevance


@pytest.mark.parametrize("cls", [CIFE, DISR, MIFS])
@pytest.mark.parametrize("seed", [4, 15, 78])
def test_information_independent_oracle(cls, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, (173, 9))
    y = rng.choice([-4, 17, 800], 173, p=[0.72, 0.21, 0.07])
    result = cls(5).fit(X, y)
    selected, scores, relevance = information_oracle(X, y, cls.__name__, 5)
    np.testing.assert_array_equal(result.top_features_, selected)
    np.testing.assert_allclose(result.selection_scores_, scores, atol=8e-14, rtol=1e-12)
    np.testing.assert_allclose(result.feature_importances_, relevance, atol=8e-14)


@pytest.mark.parametrize("cls", [CIFE, DISR, MIFS])
def test_information_exact_ties_complementarity_constants_and_remapping(cls):
    a = np.tile([0, 0, 1, 1], 16)
    b = np.tile([0, 1, 0, 1], 16)
    y = a ^ b
    X = np.column_stack([a, b, y, y, np.zeros(len(y), dtype=int)])
    expected, scores, _ = information_oracle(X, y, cls.__name__, 5)
    result = cls(5).fit(X, y)
    np.testing.assert_array_equal(result.top_features_, expected)
    np.testing.assert_allclose(result.selection_scores_, scores, atol=1e-14)
    shifted = cls(5).fit(X.astype(np.uint64) + np.uint64(2**63), y + 123)
    np.testing.assert_array_equal(shifted.top_features_, result.top_features_)
    np.testing.assert_allclose(shifted.selection_scores_, scores, atol=1e-14)
    constant = cls(3).fit(np.zeros((20, 3)), np.zeros(20))
    np.testing.assert_array_equal(constant.top_features_, [0, 1, 2])
    np.testing.assert_array_equal(constant.selection_scores_, [0, 0, 0])


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.5])
def test_mifs_beta_is_summed_redundancy(beta):
    rng = np.random.default_rng(143)
    X = rng.integers(0, 4, (173, 8))
    y = rng.integers(0, 3, 173)
    result = MIFS(5, beta=beta).fit(X, y)
    selected, scores, _ = information_oracle(X, y, "MIFS", 5, beta=beta)
    np.testing.assert_array_equal(result.top_features_, selected)
    np.testing.assert_allclose(result.selection_scores_, scores, atol=8e-14)
    for invalid in (-1, np.nan, np.inf, True, "0.5"):
        with pytest.raises(ValueError, match="beta"):
            MIFS(2, beta=invalid).fit(X, y)


def gini_oracle(X, y):
    def impurity(labels):
        return 1 - sum((count / len(labels)) ** 2 for count in Counter(labels).values()) if len(labels) else 0

    scores = []
    for column in X.T:
        values = [impurity(y)]
        for threshold in np.unique(column):
            mask = column <= threshold
            values.append((mask.sum() * impurity(y[mask]) + (~mask).sum() * impurity(y[~mask])) / len(y))
        scores.append(min(values))
    return np.asarray(scores)


@pytest.mark.parametrize("classes", [1, 2, 4])
def test_gini_oracle_and_all_thresholds(classes):
    rng = np.random.default_rng(10)
    X = rng.integers(0, 8, (131, 8))
    X[:, -1] = 0
    y = rng.integers(0, classes, 131) * 7 - 13
    result = GiniIndex(4).fit(X, y)
    np.testing.assert_allclose(result.gini_scores_, gini_oracle(X, y), atol=3e-16)
    # Four-class impurity can exceed 0.5; do not copy the scikit-feature cap.
    if classes == 4:
        assert result.gini_scores_[-1] > 0.5
    shifted = GiniIndex(4).fit(X + 10**8, y)
    np.testing.assert_array_equal(result.top_features_, shifted.top_features_)
    np.testing.assert_array_equal(result.gini_scores_, shifted.gini_scores_)


def test_gini_uint64_identity_and_ties():
    X = np.tile([[0, 0, 0], [1, 1, 0]], (10, 1)).astype(np.uint64)
    y = np.tile([17, 100], 10)
    result = GiniIndex(3).fit(X + np.uint64(2**63), y)
    np.testing.assert_array_equal(result.gini_scores_, [0, 0, 0.5])
    np.testing.assert_array_equal(result.top_features_, [0, 1, 2])
    # Two float64 thresholds that a float32 CART input would merge.
    values = np.array([[1.0], [1.0 + 1e-9], [1.0 + 2e-9], [1.0 + 3e-9]])
    np.testing.assert_array_equal(GiniIndex(1).fit(values, [0, 0, 1, 1]).gini_scores_, [0])


def prepared_regression(X, y, discrete, seed):
    rng = np.random.RandomState(seed)
    values = X.astype(float, copy=True)
    if (~discrete).any():
        values[:, ~discrete] = scale(values[:, ~discrete], with_mean=False)
        means = np.maximum(1, np.mean(np.abs(values[:, ~discrete]), axis=0))
        values[:, ~discrete] += 1e-10 * means * rng.standard_normal((len(X), (~discrete).sum()))
    target = scale(y.astype(float), with_mean=False)
    target += 1e-10 * max(1, np.mean(np.abs(target))) * rng.standard_normal(len(y))
    return values, target


def regression_mi_oracle(X, y, discrete, k, seed):
    values, target = prepared_regression(X, y, discrete, seed)
    output = []
    for f, column in enumerate(values.T):
        if not discrete[f]:
            dx = np.abs(column[:, None] - column[None, :])
            dy = np.abs(target[:, None] - target[None, :])
            distance = np.maximum(dx, dy)
            np.fill_diagonal(distance, np.inf)
            radius = np.nextafter(np.sort(distance, axis=1)[:, k - 1], 0)
            nx = np.sum(dx <= radius[:, None], axis=1)
            ny = np.sum(dy <= radius[:, None], axis=1)
            estimate = digamma(len(y)) + digamma(k) - np.mean(digamma(nx)) - np.mean(digamma(ny))
        else:
            valid = np.array([np.count_nonzero(column == value) > 1 for value in column])
            terms = []
            for i in np.flatnonzero(valid):
                peers = np.flatnonzero(column == column[i])
                effective_k = min(k, len(peers) - 1)
                neighbors = np.sort(np.abs(target[peers[peers != i]] - target[i]))
                radius = np.nextafter(neighbors[effective_k - 1], 0)
                m = np.count_nonzero(np.abs(target[valid] - target[i]) <= radius)
                terms.append(digamma(effective_k) - digamma(len(peers)) - digamma(m))
            estimate = digamma(valid.sum()) + np.mean(terms)
        output.append(max(0, estimate))
    return np.array(output)


@pytest.mark.parametrize("k", [1, 3, 9, 32])
def test_regression_mi_exact_chebyshev_oracle(k):
    rng = np.random.default_rng(49)
    X = rng.normal(size=(33, 5))
    X[:, 1] = np.round(X[:, 1])
    X[:, 2] = 1
    y = X[:, 0] ** 2 + rng.normal(size=33)
    expected = regression_mi_oracle(X, y, np.zeros(5, dtype=bool), k, 7)
    np.testing.assert_allclose(mutual_info_regression(X, y, n_neighbors=k, random_state=7), expected, atol=3e-14)


@pytest.mark.parametrize("discrete", [[True, False, False], [True, True, True]])
def test_regression_mi_discrete_and_mixed(discrete):
    rng = np.random.default_rng(33)
    X = rng.integers(0, 4, (71, 3)).astype(float)
    X[-1, 0] = 999  # singleton category is removed in Ross's estimator
    y = rng.normal(size=71)
    expected = regression_mi_oracle(X, y, np.asarray(discrete), 3, 7)
    actual = mutual_info_regression(X, y, discrete_features=discrete, random_state=7)
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    np.testing.assert_allclose(actual, sklearn_mi(X, y, discrete_features=discrete, random_state=7), atol=3e-13)


def test_regression_mi_reference_large_offset_copy_and_labels():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(151, 9))
    X[:, 2] += 1e8
    y = rng.normal(size=151)
    original = X.copy()
    for k in (1, 3, 7):
        result = mutual_info_regression(X, y, n_neighbors=k, random_state=31)
        np.testing.assert_allclose(result, sklearn_mi(X, y, n_neighbors=k, random_state=31), atol=3e-13)
    np.testing.assert_array_equal(original, X)


@pytest.mark.parametrize("k", [3, 9])
def test_regression_mi_float32_inputs_have_float64_working_precision(k):
    rng = np.random.default_rng(56)
    X = rng.integers(-3, 4, (43, 5)).astype(np.float32)
    y = rng.integers(-2, 3, 43).astype(np.float32)
    expected = regression_mi_oracle(X, y, np.zeros(5, dtype=bool), k, 7)
    actual = mutual_info_regression(X, y, n_neighbors=k, random_state=7)
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    np.testing.assert_allclose(
        actual, sklearn_mi(X.astype(np.float64), y.astype(np.float64), n_neighbors=k, random_state=7), atol=3e-13
    )


@pytest.mark.parametrize("estimator", [GiniIndex(1), CIFE(1), DISR(1), MIFS(1)])
def test_second_batch_estimator_contract(estimator):
    failures = (
        {}
        if isinstance(estimator, GiniIndex)
        else {
            "check_dtype_object": "categorical symbols are accepted; unorderable dict values fail in category sorting"
        }
    )
    check_estimator(estimator, expected_failed_checks=failures)


@pytest.mark.parametrize("cls", [GiniIndex, CIFE, DISR, MIFS])
def test_second_batch_pipeline_gridsearch(cls):
    rng = np.random.default_rng(9)
    X = rng.integers(0, 4, (90, 6))
    y = (X[:, 0] > 1).astype(int)
    pipe = Pipeline([("select", cls(2)), ("model", LogisticRegression())])
    result = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2]}, cv=3, error_score="raise").fit(X, y)
    assert result.predict(X).shape == y.shape


def test_second_batch_score_function_pipeline():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(90, 6))
    y = X[:, 0] + rng.normal(size=90)
    score = partial(mutual_info_regression, random_state=7)
    pipe = Pipeline([("select", SelectKBest(score, k=2)), ("model", Ridge())])
    result = GridSearchCV(pipe, {"select__k": [1, 2]}, cv=3, error_score="raise").fit(X, y)
    assert result.predict(X).shape == y.shape


def test_second_batch_errors_and_threads():
    import numba

    X = np.arange(40).reshape(10, 4)
    y = np.tile([0, 1], 5)
    previous = numba.get_num_threads()
    for cls in (GiniIndex, CIFE, DISR, MIFS):
        model = cls(2, n_jobs=1).fit(X, y)
        assert model.effective_backend_ == "cpu"
        assert numba.get_num_threads() == previous
        with pytest.raises(ValueError, match="features"):
            model.transform(X[:, :2])
        with pytest.raises(ValueError):
            cls(2, backend="metal").fit(X, y)
    with pytest.raises(ValueError):
        mutual_info_regression(X, y, n_neighbors=10)
    mutual_info_regression(X, y, n_jobs=1, random_state=7)
    assert numba.get_num_threads() == previous


def test_second_batch_parameter_errors():
    X = np.random.default_rng(0).integers(0, 3, (20, 4))
    y = np.arange(20) % 2
    for cls in (GiniIndex, DISR, CIFE, MIFS):
        with pytest.raises(ValueError, match="must not exceed"):
            cls(n_features_to_select=5).fit(X, y)


def test_regression_mi_discrete_feature_specifications():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.standard_normal(40), rng.integers(0, 3, 40), rng.standard_normal(40)])
    y = X[:, 1] + 0.1 * rng.standard_normal(40)
    by_mask = mutual_info_regression(X, y, discrete_features=np.array([False, True, False]), random_state=0)
    by_index = mutual_info_regression(X, y, discrete_features=[1], random_state=0)
    np.testing.assert_array_equal(by_index, by_mask)
    with pytest.raises(ValueError, match="discrete_features"):
        mutual_info_regression(X, y, discrete_features="all")
    with pytest.raises(ValueError, match="mask must match"):
        mutual_info_regression(X, y, discrete_features=np.array([True, False]))
    # A discrete column whose every category is a singleton has no Ross neighbors.
    X[:, 1] = np.arange(40)
    with pytest.raises(ValueError, match="multiple samples"):
        mutual_info_regression(X, y, discrete_features=[1], random_state=0)
