"""Exact weighted Gini split ranking (Breiman et al., CART, 1984)."""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from fast_select._cpu_backend import resolve_cpu_backend
from fast_select.utils import check_integer_param, resolve_num_threads


@njit(parallel=True, cache=True, nogil=True)
def _gini(columns, labels, counts):  # pragma: no cover
    p, n = columns.shape
    scores = np.empty(p)
    total_squares = np.sum(counts * counts)
    for f in prange(p):
        order = np.argsort(columns[f])
        left = np.zeros(len(counts), dtype=np.int64)
        right = counts.copy()
        lsq, rsq = 0, total_squares
        best = 1 - total_squares / (n * n)
        for i in range(n - 1):
            label = labels[order[i]]
            lsq += 2 * left[label] + 1
            rsq -= 2 * right[label] - 1
            left[label] += 1
            right[label] -= 1
            if columns[f, order[i]] != columns[f, order[i + 1]]:
                a, b = i + 1, n - i - 1
                score = 1 - (lsq / a + rsq / b) / n
                best = min(best, score)
        scores[f] = best
    return scores


class GiniIndex(TransformerMixin, BaseEstimator):
    """Rank numeric classification features by their best weighted Gini split.

    CART impurity is 1-sum(class_probability**2). Enumerate all distinct-value
    split boundaries exactly in float64, plus the no-split parent. Lower raw
    gini_scores_ is better; feature_importances_ negates it. Supports multiclass
    and arbitrary class labels. Exact score ties keep the lower feature index.
    Only CPU/auto is supported; no GPU implementation is admitted.
    """

    def __init__(self, n_features_to_select=10, backend="auto", n_jobs=-1):
        self.n_features_to_select = n_features_to_select
        self.backend = backend
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if y is None:
            raise ValueError("GiniIndex requires y to be passed, but the target y is None")
        X, y = validate_data(self, X, y, dtype="numeric", ensure_min_samples=2)
        check_classification_targets(y)
        check_integer_param(self.n_features_to_select, "n_features_to_select", 1)
        if self.n_features_to_select > X.shape[1]:
            raise ValueError("n_features_to_select must not exceed the feature count")
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        _, labels = np.unique(y, return_inverse=True)
        counts = np.bincount(labels)
        threads = resolve_num_threads(self.n_jobs)
        previous = get_num_threads()
        try:
            set_num_threads(threads)
            self.gini_scores_ = _gini(np.ascontiguousarray(X.T), labels, counts)
        finally:
            set_num_threads(previous)
        self.feature_importances_ = -self.gini_scores_
        self.top_features_ = np.argsort(self.gini_scores_, kind="stable")[: self.n_features_to_select]
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        return X[:, self.top_features_]
