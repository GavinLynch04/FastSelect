"""Paper equations, adversarial boundaries, and estimator integration.

Oracles deliberately use Counter/dense matrix/direct distance calculations;
they never call a production entropy, graph, neighbor or GPU helper.
"""

from collections import Counter
from functools import partial

import numpy as np
import pytest
from scipy.special import digamma
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif as sklearn_mi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.utils.estimator_checks import check_estimator

from fast_select import CMIM, FCBF, JMI, LaplacianScore, mutual_info_classif


def entropy(*variables):
    counts = Counter(zip(*variables))
    probabilities = np.array(list(counts.values()), dtype=float) / len(variables[0])
    return -np.sum(probabilities * np.log2(probabilities))


def mi(a, b):
    return entropy(a) + entropy(b) - entropy(a, b)


def conditional_mi(a, b, given):
    return entropy(a, given) + entropy(b, given) - entropy(given) - entropy(a, b, given)


def information_oracle(X, y, criterion, k=5, delta=0):
    relevance = np.array([mi(column, y) for column in X.T])
    if criterion == "FCBF":
        denominators = np.array([entropy(column) + entropy(y) for column in X.T])
        relevance = np.divide(2 * relevance, denominators, out=np.zeros_like(relevance), where=denominators > 0)
    active = [i for i in range(X.shape[1]) if criterion != "FCBF" or relevance[i] > delta]
    selected, objectives = [], []
    while active and (criterion == "FCBF" or len(selected) < k):
        scores = []
        for f in active:
            if not selected or criterion == "FCBF":
                score = relevance[f]
            elif criterion == "JMI":
                # Original joint-MI sum normalized by |S|, dropping I(S_i;y),
                # which is the same additive constant for all candidates.
                score = np.mean([mi(list(zip(X[:, f], X[:, s])), y) - mi(X[:, s], y) for s in selected])
            else:
                score = min(conditional_mi(X[:, f], y, X[:, s]) for s in selected)
            scores.append(score)
        chosen = active[int(np.argmax(scores))]
        selected.append(chosen)
        objectives.append(max(scores))
        active.remove(chosen)
        if criterion == "FCBF":
            remaining = []
            for f in active:
                denominator = entropy(X[:, chosen]) + entropy(X[:, f])
                pair_su = 2 * mi(X[:, chosen], X[:, f]) / denominator if denominator else 0
                if pair_su < relevance[f]:
                    remaining.append(f)
            active = remaining
    return np.array(selected, dtype=int), np.array(objectives), relevance


BACKENDS = ["cpu"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cls", [JMI, CMIM, FCBF])
@pytest.mark.parametrize("seed", [7, 19, 31])
def test_information_paper_oracle(cls, backend, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, (191, 9))
    y = rng.choice([-7, 13, 91], 191, p=[0.7, 0.2, 0.1])
    estimator = cls(backend=backend).set_params(**({} if cls is FCBF else {"n_features_to_select": 5})).fit(X, y)
    selected, objectives, relevance = information_oracle(X, y, cls.__name__)
    np.testing.assert_array_equal(estimator.top_features_, selected)
    np.testing.assert_allclose(estimator.selection_scores_, objectives, atol=8e-14, rtol=1e-12)
    np.testing.assert_allclose(estimator.feature_importances_, relevance, atol=8e-14, rtol=1e-12)
    np.testing.assert_array_equal(estimator.transform(X), X[:, selected])


@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_information_boundaries_and_constant_variables(backend):
    y = np.tile([0, 0, 1, 1], 8)
    independent = np.tile([0, 1, 0, 1], 8)
    X = np.column_stack([y, y, independent, np.zeros_like(y)])
    for cls in (JMI, CMIM):
        result = cls(4, backend=backend).fit(X, y)
        expected, scores, _ = information_oracle(X, y, cls.__name__, k=4)
        np.testing.assert_array_equal(result.top_features_, expected)
        np.testing.assert_allclose(result.selection_scores_, scores, atol=2e-14)
    result = FCBF(backend=backend).fit(X, y)
    np.testing.assert_array_equal(result.top_features_, [0])  # redundant equal-SU peer removed
    assert FCBF(delta=1, backend=backend).fit(X, y).top_features_.size == 0  # strict relevance
    assert FCBF(backend=backend).fit(X, np.zeros_like(y)).top_features_.size == 0
    for cls in (JMI, CMIM):
        np.testing.assert_array_equal(cls(4, backend=backend).fit(np.zeros((32, 4)), y).top_features_, [0, 1, 2, 3])


@pytest.mark.parametrize("cls", [JMI, CMIM, FCBF])
def test_categorical_identity_and_high_cardinality(cls):
    rng = np.random.default_rng(24)
    X = rng.integers(0, 4, (73, 7), dtype=np.uint64)
    y = rng.integers(0, 3, 73, dtype=np.uint64)
    kwargs = {} if cls is FCBF else {"n_features_to_select": 4}
    a = cls(backend="cpu", **kwargs).fit(X, y)
    b = cls(backend="cpu", **kwargs).fit(X + np.uint64(2**63), y + np.uint64(2**62))
    np.testing.assert_array_equal(a.top_features_, b.top_features_)
    np.testing.assert_allclose(a.feature_importances_, b.feature_importances_, atol=1e-14)
    X = rng.integers(0, 2000, (1200, 3))  # sparse pair/triple histogram path
    result = cls(backend="cpu", **({} if cls is FCBF else {"n_features_to_select": 2})).fit(X, rng.integers(0, 3, 1200))
    assert np.isfinite(result.selection_scores_).all()


def laplacian_oracle(X, W):
    degree = np.sum(W, axis=1)
    laplacian = np.diag(degree) - W
    scores = []
    for f in X.T:
        centered = f - np.dot(f, degree) / degree.sum() if degree.sum() else np.zeros_like(f)
        denominator = centered @ np.diag(degree) @ centered
        scores.append(centered @ laplacian @ centered / denominator if denominator > 0 else np.inf)
    return np.array(scores)


@pytest.mark.parametrize("backend", BACKENDS)
def test_laplacian_equation_and_translation(backend):
    rng = np.random.default_rng(5)
    X = rng.integers(0, 12, (31, 8)).astype(float)
    X[:, -1] = 7
    weights = rng.random((31, 31))
    W = np.maximum(weights, weights.T)
    W[W < 0.65] = 0
    expected = laplacian_oracle(X, W)
    for values in (X, X + 1e8, X * -3):
        result = LaplacianScore(4, backend=backend).fit(values, W=W)
        np.testing.assert_allclose(result.laplacian_scores_, expected, rtol=2e-11, atol=2e-12)
        np.testing.assert_array_equal(result.top_features_, np.argsort(expected, kind="stable")[:4])
        assert np.isinf(result.laplacian_scores_[-1])
    assert np.isinf(LaplacianScore(2, backend=backend).fit(X, W=np.zeros_like(W)).laplacian_scores_).all()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["binary", "heat_kernel"])
def test_laplacian_complete_graph_fit(backend, mode):
    rng = np.random.default_rng(84)
    X = rng.normal(size=(39, 6))
    squared = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    neighbors = np.argsort(squared, axis=1)[:, :4]
    W = np.zeros((len(X), len(X)))
    for i, indices in enumerate(neighbors):
        W[i, indices] = 1 if mode == "binary" else np.exp(-squared[i, indices] / (2 * 1.7**2))
    W = np.maximum(W, W.T)
    result = LaplacianScore(3, n_neighbors=3, weight_mode=mode, bandwidth=1.7, backend=backend).fit(X)
    np.testing.assert_allclose(result.laplacian_scores_, laplacian_oracle(X, W), atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_laplacian_score_ties_and_tiny_variance(backend):
    X = np.array([[0, 0, 0], [1, 1, 1e-12], [2, 2, 2e-12], [4, 4, 4e-12]], dtype=float)
    W = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=float)
    result = LaplacianScore(2, backend=backend).fit(X, W=W)
    np.testing.assert_allclose(result.laplacian_scores_, laplacian_oracle(X, W), atol=2e-12)
    np.testing.assert_array_equal(result.top_features_, [0, 1])


def test_laplacian_graph_large_offset_and_boundary_ties():
    rng = np.random.default_rng(39)
    X = rng.integers(0, 12, (37, 7)).astype(float)
    for values in (X, X + 1e8):
        squared = np.sum((values[:, None] - values[None, :]) ** 2, axis=2)
        neighbors = np.argsort(squared, axis=1)[:, :4]
        W = np.zeros((len(values), len(values)))
        for i, indices in enumerate(neighbors):
            W[i, indices] = 1
        W = np.maximum(W, W.T)
        result = LaplacianScore(3, n_neighbors=3).fit(values)
        np.testing.assert_array_equal(result.affinity_.toarray(), W)
        np.testing.assert_allclose(result.laplacian_scores_, laplacian_oracle(X, W), atol=2e-12)
    # A boundary tie spans both included/excluded points in every row.
    X = np.tile([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], (3, 1))
    squared = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    neighbors = np.argsort(squared, axis=1)[:, :5]
    W = np.zeros((len(X), len(X)))
    W[np.arange(len(X))[:, None], neighbors] = 1
    np.testing.assert_array_equal(LaplacianScore(1, n_neighbors=4).fit(X).affinity_.toarray(), np.maximum(W, W.T))


def ross_oracle(X, y, k, seed=7):
    # The public scale/jitter convention is separate from Ross's Eq. (2).
    values = scale(np.asarray(X, dtype=float), with_mean=False)
    means = np.maximum(1, np.mean(np.abs(values), axis=0))
    values += 1e-10 * means * np.random.RandomState(seed).standard_normal(values.shape)
    valid = np.array([np.count_nonzero(y == label) > 1 for label in y])
    scores = []
    for column in values.T:
        terms = []
        for i in np.flatnonzero(valid):
            peers = np.flatnonzero(y == y[i])
            neighbors = min(k, len(peers) - 1)
            distances = np.sort(np.abs(column[peers[peers != i]] - column[i]))
            radius = np.nextafter(distances[neighbors - 1], 0)
            m = np.count_nonzero(np.abs(column[valid] - column[i]) <= radius)
            terms.append(digamma(neighbors) - digamma(len(peers)) - digamma(m))
        scores.append(max(0, digamma(valid.sum()) + np.mean(terms)))
    return np.array(scores)


@pytest.mark.parametrize("k", [1, 3, 9])
def test_continuous_mi_ross_equation(k):
    rng = np.random.default_rng(29)
    X = rng.normal(size=(53, 7))
    X[:, 1] = np.round(X[:, 1])  # duplicates
    X[:, 2] = 1  # jittered using the same explicit public convention
    y = np.array([-7] * 31 + [13] * 17 + [99] * 4 + [1001])
    expected = ross_oracle(X, y, k)
    actual = mutual_info_classif(X, y, n_neighbors=k, random_state=7)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=2e-12)
    # Small imbalanced groups make sklearn choose brute-force Euclidean
    # neighbors. Its squared-distance identity cancels tiny jitter distances
    # around a constant offset; the direct Ross oracle is authoritative here.


@pytest.mark.parametrize("k", [1, 3, 9])
def test_continuous_mi_reference_nondegenerate(k):
    rng = np.random.default_rng(32)
    X = rng.normal(size=(301, 11))
    y = rng.integers(0, 3, 301)
    np.testing.assert_allclose(
        mutual_info_classif(X, y, n_neighbors=k, random_state=7),
        sklearn_mi(X, y, n_neighbors=k, random_state=7),
        atol=3e-13,
        rtol=2e-12,
    )


@pytest.mark.parametrize("seed", [0, 9, 23])
def test_continuous_mi_mixed_discrete_and_copy(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(157, 9))
    X[:, 0:2] = rng.integers(0, 4, (157, 2))
    X[:, 3] += 1e8
    y = rng.integers(-1, 3, 157)
    original = X.copy()
    for discrete in ([0, 1], [True, True] + [False] * 7):
        result = mutual_info_classif(X, y, discrete_features=discrete, random_state=seed)
        expected = sklearn_mi(X, y, discrete_features=discrete, random_state=seed)
        np.testing.assert_allclose(result, expected, atol=3e-13, rtol=2e-12)
        np.testing.assert_array_equal(X, original)
    result = mutual_info_classif(X, y, discrete_features=True, random_state=seed)
    np.testing.assert_allclose(result, [mi(f, y) * np.log(2) for f in X.T], atol=3e-13)


@pytest.mark.parametrize(
    "estimator", [JMI(1, backend="cpu"), CMIM(1, backend="cpu"), FCBF(backend="cpu"), LaplacianScore(1, backend="cpu")]
)
def test_new_estimator_contract(estimator):
    categorical_reason = "categorical symbols are accepted; unorderable dict values fail during category sorting"
    check_estimator(
        estimator,
        expected_failed_checks=(
            {} if isinstance(estimator, LaplacianScore) else {"check_dtype_object": categorical_reason}
        ),
    )


@pytest.mark.parametrize("cls", [JMI, CMIM, FCBF, LaplacianScore])
def test_new_estimator_pipeline_gridsearch(cls):
    rng = np.random.default_rng(12)
    X = rng.integers(0, 3, (90, 6))
    y = (X[:, 0] > 0).astype(int)
    kwargs = {} if cls is FCBF else {"n_features_to_select": 2}
    pipe = Pipeline([("select", cls(backend="cpu", **kwargs)), ("clf", LogisticRegression())])
    parameter = {"select__delta": [0.0, 0.05]} if cls is FCBF else {"select__n_features_to_select": [1, 2]}
    search = GridSearchCV(pipe, parameter, cv=3, error_score="raise").fit(X, y)
    assert search.predict(X).shape == y.shape


def test_continuous_mi_selectkbest_gridsearch():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(90, 6))
    y = (X[:, 0] > 0).astype(int)
    pipe = Pipeline(
        [("select", SelectKBest(partial(mutual_info_classif, random_state=3), k=2)), ("clf", LogisticRegression())]
    )
    assert GridSearchCV(pipe, {"select__k": [1, 2]}, cv=3, error_score="raise").fit(X, y).predict(X).shape == y.shape


def test_new_invalid_inputs_and_thread_restoration():
    import numba

    X = np.arange(40).reshape(10, 4)
    y = np.tile([0, 1], 5)
    previous = numba.get_num_threads()
    for cls in (JMI, CMIM, FCBF, LaplacianScore):
        kwargs = {} if cls is FCBF else {"n_features_to_select": 2}
        estimator = cls(backend="cpu", n_jobs=1, **kwargs).fit(X, y)
        assert numba.get_num_threads() == previous
        with pytest.raises(ValueError, match="features"):
            estimator.transform(X[:, :2])
        with pytest.raises(ValueError):
            cls(backend="unknown", **kwargs).fit(X, y)
    for W in (np.ones((2, 2)), -np.eye(10), np.triu(np.ones((10, 10)))):
        with pytest.raises(ValueError):
            LaplacianScore(2, backend="cpu").fit(X, W=W)
    with pytest.raises(ValueError):
        mutual_info_classif(X, np.arange(10))
    mutual_info_classif(X, y, n_jobs=1, random_state=7)
    assert numba.get_num_threads() == previous


@pytest.mark.parametrize("cls", [JMI, CMIM, FCBF, LaplacianScore])
def test_unadmitted_gpu_backends_are_not_exposed(cls):
    X = np.arange(40).reshape(10, 4)
    y = np.tile([0, 1], 5)
    kwargs = {} if cls is FCBF else {"n_features_to_select": 2}
    assert cls(backend="auto", **kwargs).fit(X, y).effective_backend_ == "cpu"
    for backend in ("metal", "gpu"):
        with pytest.raises(ValueError, match="No GPU backend was retained"):
            cls(backend=backend, **kwargs).fit(X, y)
