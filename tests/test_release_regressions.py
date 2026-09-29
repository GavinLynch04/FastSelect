"""Independent numerical controls and regression tests from the release review.

Sources: Urbanowicz et al. (2018), Peng et al. (2005), Hall (1999), and the
repository's documented complete-data MDR and count-feature chi2 contracts.

The oracles below are deliberately self-contained (plain ``collections`` /
``math`` / NumPy) and never call a production numerical helper.
"""

import heapq
import math
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone, is_classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import get_tags

from fast_select import (
    CFS,
    MDR,
    SURF,
    MultiSURF,
    ReliefF,
    TuRF,
    calculate_mi_matrices,
    calculate_mi_single_pair,
    chi2,
    mRMR,
)


def empirical_mi(a, b):
    """Discrete MI in bits, using counts over observed symbols only."""
    pairs = Counter(zip(a.tolist(), b.tolist()))
    ca, cb = Counter(a.tolist()), Counter(b.tolist())
    n = len(a)
    return sum(c / n * math.log2(c * n / (ca[x] * cb[y])) for (x, y), c in pairs.items())


def entropy(a):
    n = len(a)
    return -sum((c / n) * math.log2(c / n) for c in Counter(a.tolist()).values())


def su(a, b):
    denom = entropy(a) + entropy(b)
    return 2 * empirical_mi(a, b) / denom if denom else 0.0


def cfs_reference(X, y):
    """Hall merit and a forward best-first queue; no production helpers."""
    n = X.shape[1]
    cf = [su(X[:, i], y) for i in range(n)]
    ff = {(i, j): su(X[:, i], X[:, j]) for i, j in combinations(range(n), 2)}

    def merit(subset):
        if not subset:
            return 0.0
        return sum(cf[i] for i in subset) / math.sqrt(
            len(subset) + 2 * sum(ff[i, j] for i, j in combinations(subset, 2))
        )

    queue = [(-merit((i,)), (i,)) for i in range(n)]
    heapq.heapify(queue)
    seen = {q[1] for q in queue}
    best, best_score, stale = (), 0.0, 0
    while queue and stale < 5:
        neg_score, subset = heapq.heappop(queue)
        if -neg_score > best_score + 1e-12:
            best, best_score, stale = subset, -neg_score, 0
        else:
            stale += 1
        for i in range(n):
            if i in subset:
                continue
            child = tuple(sorted((*subset, i)))
            if child not in seen:
                heapq.heappush(queue, (-merit(child), child))
                seen.add(child)
    return best, best_score


@pytest.mark.parametrize("seed", range(20))
def test_mi_and_mrmr_independent_count_oracle(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, (80, 6))
    y = rng.integers(0, 2, 80)
    rel = np.array([empirical_mi(X[:, i], y) for i in range(6)])
    red = np.zeros((6, 6))
    for i, j in combinations(range(6), 2):
        red[i, j] = red[j, i] = empirical_mi(X[:, i], X[:, j])
    for unit, scale in [("bit", 1.0), ("nat", math.log(2.0))]:
        actual_rel, actual_red = calculate_mi_matrices(X, y, backend="cpu", unit=unit)
        assert_allclose(actual_rel, rel * scale, atol=1e-12)
        assert_allclose(actual_red, red * scale, atol=1e-12)
    for method in ["MID", "MIQ"]:
        selected = [int(np.argmax(rel))]
        for _ in range(3):
            remaining = [i for i in range(6) if i not in selected]
            means = np.array([np.mean(red[i, selected]) for i in remaining])
            scores = rel[remaining] - means if method == "MID" else rel[remaining] / (means + 1e-9)
            selected.append(remaining[int(np.argmax(scores))])
        model = mRMR(4, backend="cpu", method=method).fit(X, y)
        assert_array_equal(model.top_features_, selected)


@pytest.mark.parametrize("seed", range(20))
def test_cfs_against_independent_su_and_search(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, (80, 6))
    y = rng.integers(0, 2, 80)
    expected, merit = cfs_reference(X, y)
    model = CFS(backend="cpu", n_jobs=1).fit(X, y)
    assert_array_equal(model.selected_indices_, expected)
    assert_allclose(model.merit_, merit, atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_chi2_against_independent_count_formula(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, (50, 6))
    y = rng.integers(0, 3, 50)
    observed = np.array([X[y == c].sum(axis=0) for c in np.unique(y)])
    expected = np.array([(y == c).mean() * X.sum(axis=0) for c in np.unique(y)])
    statistic = np.sum((observed - expected) ** 2 / expected, axis=0)
    actual, pvalue = chi2(X, y)
    assert_allclose(actual, statistic, atol=1e-12)
    # Three classes => 2 degrees of freedom, whose SF is exp(-statistic/2).
    assert_allclose(pvalue, np.exp(-statistic / 2), atol=1e-12)


def mdr_predictions(train, labels, test, combo):
    cases = Counter(tuple(row[list(combo)]) for row in train[labels == 1])
    controls = Counter(tuple(row[list(combo)]) for row in train[labels == 0])
    nc, nn = int(sum(labels)), int(len(labels) - sum(labels))
    result = []
    for row in test:
        cell = tuple(row[list(combo)])
        c, n = cases[cell], controls[cell]
        # Integer cross multiplication handles exact ratio ties without rounding.
        result.append(int(c + n > 0 and c * nn >= n * nc))
    return np.array(result)


def balanced_accuracy(y, predictions):
    return ((predictions[y == 0] == 0).mean() + (predictions[y == 1] == 1).mean()) / 2


@pytest.mark.parametrize("seed", range(20))
def test_mdr_against_independent_contingency_and_fold_oracle(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, (36, 4))
    y = np.array([0] * 18 + [1] * 18)
    combos = list(combinations(range(4), 2))
    winners, test_scores = [], []
    for train, test in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        scores = [balanced_accuracy(y[train], mdr_predictions(X[train], y[train], X[train], c)) for c in combos]
        # The production training selector exposes float32 balanced accuracies.
        winner = combos[int(np.argmax(np.asarray(scores, dtype=np.float32)))]
        winners.append(winner)
        test_scores.append(balanced_accuracy(y[test], mdr_predictions(X[train], y[train], X[test], winner)))
    counts = Counter(winners)
    candidates = [c for c in counts if counts[c] == max(counts.values())]
    means = [np.mean([s for w, s in zip(winners, test_scores) if w == c]) for c in candidates]
    winner = candidates[int(np.argmax(means))]
    model = MDR(k=2, cv=3, backend="cpu").fit(X, y)
    assert tuple(model.best_interaction_) == winner
    assert model.best_cvc_ == counts[winner]
    assert_allclose(model.best_mean_testing_ba_, max(means))
    assert_array_equal(model.predict(X), mdr_predictions(X, y, X, winner))


def test_mdr_k6_lookup_table_uses_full_cell_index():
    """3**6 = 729 cells do not fit uint8 arithmetic; the all-2 cell must be cell 728."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, (60, 6))
    X[:20] = 2
    y = np.array([1] * 20 + [0] * 40)
    model = MDR(k=6, cv=2, backend="cpu").fit(X, y)
    assert model.best_model_lookup_table_.shape == (729,)
    assert model.best_model_lookup_table_[728] == 1
    assert_array_equal(model.predict(X[:20]), np.ones(20, dtype=int))


RELIEF_VARIANTS = [
    (ReliefF, {}),
    (SURF, {}),
    (SURF, {"use_star": True}),
    (MultiSURF, {}),
    (MultiSURF, {"use_star": True}),
]


@pytest.mark.parametrize("cls,extra", RELIEF_VARIANTS)
def test_continuous_translation_invariance(cls, extra):
    rng = np.random.default_rng(123)
    X = rng.uniform(0, 1, (30, 3))
    y = (X[:, 0] > 0.5).astype(int)
    kw = dict(n_features_to_select=1, backend="cpu", n_jobs=1, discrete_limit=0, **extra)
    if cls is ReliefF:
        kw["n_neighbors"] = 1
    original = cls(**kw).fit(X, y).feature_importances_
    translated = cls(**kw).fit(X + 1e8, y).feature_importances_
    assert_allclose(translated, original, atol=1e-6)


@pytest.mark.parametrize("cls,extra", RELIEF_VARIANTS)
def test_discrete_category_magnitude_invariance(cls, extra):
    """Only equality matters for discrete features, whatever the raw codes are."""
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, (40, 3)).astype(np.float64)
    y = X[:, 0].astype(int)
    kw = dict(n_features_to_select=1, backend="cpu", n_jobs=1, discrete_limit=2, **extra)
    if cls is ReliefF:
        kw["n_neighbors"] = 1
    original = cls(**kw).fit(X, y).feature_importances_
    relabelled = cls(**kw).fit(X + 2.0**25, y).feature_importances_
    assert np.any(original != 0), "the control must produce a non-trivial score"
    assert_allclose(relabelled, original, atol=1e-6)


@pytest.mark.parametrize("star,expected", [(False, -2 / 9), (True, -8 / 9)])
def test_surf_exact_global_threshold_excludes_ties(star, expected):
    # Distances: 1/3, 2/3, 1. Radius: 2/3. Pair (1,3) is excluded.
    X = np.array([[0.0], [1.0], [3.0]])
    model = SURF(1, backend="cpu", discrete_limit=0, n_jobs=1, use_star=star).fit(X, [0, 0, 1])
    assert_allclose(model.feature_importances_, [expected], atol=1e-6)


@pytest.mark.parametrize("float_input", ["X", "y"])
def test_mrmr_accepts_float_coded_discrete_values(float_input):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    expected = mRMR(1, backend="cpu").fit(X, y).relevance_scores_
    actual = (
        mRMR(1, backend="cpu")
        .fit(X.astype(float) if float_input == "X" else X, y.astype(float) if float_input == "y" else y)
        .relevance_scores_
    )
    assert_allclose(actual, expected)


def test_mrmr_category_encoding_preserves_distinct_uint64_values():
    X = np.array([[2**63], [2**63], [2**63 + 1], [2**63 + 1]], dtype=np.uint64)
    model = mRMR(1, backend="cpu").fit(X, np.array([0, 0, 1, 1], dtype=np.int64))
    assert_allclose(model.relevance_scores_, [1.0])


def test_mi_integer_conversion_does_not_alias_states():
    X = np.array([0, 2**32, 0, 2**32], dtype=np.uint64)
    assert_allclose(calculate_mi_single_pair(X, np.array([0, 1, 0, 1]), backend="cpu"), 1.0)


@pytest.mark.parametrize("value", [0.5, 256, 257, -256])
def test_mdr_rejects_invalid_genotypes_before_casting(value):
    with pytest.raises(ValueError):
        MDR(k=1, cv=2, backend="cpu").fit(np.full((8, 1), value), [0, 1] * 4)


def test_mdr_is_cloneable():
    assert clone(MDR(backend="cpu")).get_params() == MDR(backend="cpu").get_params()


def test_mdr_is_recognized_as_classifier():
    assert is_classifier(MDR())


@pytest.mark.parametrize("model", [CFS(backend="cpu", n_jobs=1), mRMR(1, backend="cpu")])
def test_selectors_expose_sklearn_transformer_tags(model):
    assert get_tags(model).transformer_tags is not None


def test_mdr_predict_rejects_fractional_genotypes():
    X = np.array([[0], [0], [1], [1]] * 2)
    model = MDR(k=1, cv=2, backend="cpu").fit(X, [0, 0, 1, 1] * 2)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.5]]))


def test_mdr_predict_rejects_out_of_range_genotype():
    X = np.array([[0], [0], [1], [1]] * 2)
    model = MDR(k=1, cv=2, backend="cpu").fit(X, [0, 0, 1, 1] * 2)
    with pytest.raises(ValueError):
        model.predict(np.array([[3]]))


def test_mdr_predict_rejects_feature_count_mismatch():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2)
    model = MDR(k=1, cv=2, backend="cpu").fit(X, [0, 0, 1, 1] * 2)
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 3), dtype=int))


def test_mdr_predict_rejects_reordered_dataframe_columns():
    X = pd.DataFrame({"a": [0, 0, 1, 1] * 2, "b": [0, 1, 0, 1] * 2})
    model = MDR(k=1, cv=2, backend="cpu").fit(X, [0, 0, 1, 1] * 2)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(X[["b", "a"]])


def test_cfs_rejects_reordered_dataframe_columns():
    X = pd.DataFrame({"signal": [0, 0, 1, 1] * 2, "noise": [0, 1, 0, 1] * 2})
    model = CFS(backend="cpu", n_jobs=1).fit(X, X["signal"])
    with pytest.raises(ValueError):
        model.transform(X[["noise", "signal"]])


def test_cfs_transform_keeps_dataframe_and_accepts_lists():
    X = pd.DataFrame({"signal": [0, 0, 1, 1] * 2, "noise": [0, 1, 0, 1] * 2})
    model = CFS(backend="cpu", n_jobs=1).fit(X, X["signal"])
    out = model.transform(X)
    assert isinstance(out, pd.DataFrame) and list(out.columns) == ["signal"]
    listed = model.transform(X.to_numpy().tolist())
    assert_array_equal(listed, X[["signal"]].to_numpy())


def test_turf_works_with_real_relief_base_estimator():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2)
    base = ReliefF(2, backend="cpu", n_neighbors=1, n_jobs=1)
    model = TuRF(base, n_features_to_select=1).fit(X, [0, 0, 1, 1] * 2)
    assert len(model.top_features_) == 1
    assert base.n_features_to_select == 2, "the user's estimator must not be mutated"


@pytest.mark.parametrize("cls,extra", RELIEF_VARIANTS)
def test_turf_wraps_every_relief_estimator(cls, extra):
    rng = np.random.default_rng(3)
    X = rng.integers(0, 3, (40, 6)).astype(float)
    y = (X[:, 0] > 0).astype(int)
    kw = dict(n_features_to_select=4, backend="cpu", n_jobs=1, **extra)
    if cls is ReliefF:
        kw["n_neighbors"] = 2
    model = TuRF(cls(**kw), n_features_to_select=2, pct_remove=0.3).fit(X, y)
    assert len(model.top_features_) == 2


def test_multisurf_selects_only_from_requested_feature_subset():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, (12, 4))
    y = rng.integers(0, 2, 12)
    model = MultiSURF(1, backend="cpu", n_jobs=1).fit(X, y, feat_idx=[0, 1])
    assert set(model.top_features_) <= {0, 1}
