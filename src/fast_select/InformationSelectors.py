"""Independent discrete JMI, CMIM and FCBF implementations.

Yang & Moody (1999), Fleuret (2004), Brown et al. (2012), Yu & Liu (2003/2004).
All variables are categorical symbols, encoded independently without binning.
Empirical entropy uses bits. Greedy ties retain the lower original feature
index. Parallel float64 Numba CPU kernels preserve the entropy traversal.
Metal prototypes were not admitted because they did not consistently beat CPU.
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads, types
from numba.typed import Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from fast_select._cpu_backend import resolve_cpu_backend
from fast_select.mRMR import _encode_categories
from fast_select.utils import check_integer_param, resolve_num_threads

_LOG2 = math.log(2.0)
_PAIR_KEY = types.UniTuple(types.int64, 2)
_TRIPLE_KEY = types.UniTuple(types.int64, 3)


@njit(cache=True, nogil=True)
def _entropy_one(x, states):
    counts = np.zeros(states, dtype=np.int64)
    for value in x:
        counts[value] += 1
    total = 0.0
    for count in counts:
        if count:
            p = count / len(x)
            total += p * (math.log(p) / _LOG2)
    return -total


@njit(cache=True, nogil=True)
def _entropy_pair(a, b, sa, sb):
    total = 0.0
    if sa * sb <= 1_000_000:
        counts = np.zeros(sa * sb, dtype=np.int64)
        order = np.empty(min(len(a), sa * sb), dtype=np.int64)
        used = 0
        for i in range(len(a)):
            key = a[i] * sb + b[i]
            if counts[key] == 0:
                order[used] = key
                used += 1
            counts[key] += 1
        for j in range(used):
            p = counts[order[j]] / len(a)
            total += p * (math.log(p) / _LOG2)
    else:
        counts_sparse = Dict.empty(key_type=_PAIR_KEY, value_type=types.int64)
        for i in range(len(a)):
            key = (np.int64(a[i]), np.int64(b[i]))
            counts_sparse[key] = counts_sparse.get(key, 0) + 1
        for count in counts_sparse.values():
            p = count / len(a)
            total += p * (math.log(p) / _LOG2)
    return -total


@njit(cache=True, nogil=True)
def _entropy_three(a, b, c, sa, sb, sc):
    total = 0.0
    if sa * sb * sc <= 1_000_000:
        counts = np.zeros(sa * sb * sc, dtype=np.int64)
        order = np.empty(min(len(a), sa * sb * sc), dtype=np.int64)
        used = 0
        for i in range(len(a)):
            key = (a[i] * sb + b[i]) * sc + c[i]
            if counts[key] == 0:
                order[used] = key
                used += 1
            counts[key] += 1
        for j in range(used):
            p = counts[order[j]] / len(a)
            total += p * (math.log(p) / _LOG2)
    else:
        counts_sparse = Dict.empty(key_type=_TRIPLE_KEY, value_type=types.int64)
        for i in range(len(a)):
            key = (np.int64(a[i]), np.int64(b[i]), np.int64(c[i]))
            counts_sparse[key] = counts_sparse.get(key, 0) + 1
        for count in counts_sparse.values():
            p = count / len(a)
            total += p * (math.log(p) / _LOG2)
    return -total


@njit(cache=True, nogil=True)
def _first_codes(x):
    mapping = np.full(int(x.max()) + 1, -1, dtype=np.int32)
    result = np.empty(len(x), dtype=np.int32)
    next_code = 0
    for i in range(len(x)):
        value = x[i]
        if mapping[value] < 0:
            mapping[value] = next_code
            next_code += 1
        result[i] = mapping[value]
    return result


@njit(parallel=True, cache=True, nogil=True)
def _initial(columns, target, counts, sy):
    p = len(columns)
    entropy = np.empty(p, dtype=np.float64)
    joint = np.empty(p, dtype=np.float64)
    recoded = np.empty_like(columns)
    for f in prange(p):
        recoded[f] = _first_codes(columns[f])
        entropy[f] = _entropy_one(recoded[f], counts[f])
        joint[f] = _entropy_pair(recoded[f], target, counts[f], sy)
    return recoded, entropy, joint


@njit(parallel=True, cache=True, nogil=True)
def _step_cpu(columns, target, counts, sy, selected, active, conditional):
    p = len(columns)
    pair = np.zeros(p, dtype=np.float64)
    triple = np.zeros(p, dtype=np.float64)
    for f in prange(p):
        if active[f]:
            pair[f] = _entropy_pair(columns[selected], columns[f], counts[selected], counts[f])
            if conditional:
                triple[f] = _entropy_three(columns[selected], columns[f], target, counts[selected], counts[f], sy)
    return pair, triple


class _InformationSelector(TransformerMixin, BaseEstimator):
    """CPU-only categorical selector; auto chooses the admitted Numba backend.

    ``feature_importances_`` is marginal MI (SU for FCBF), while
    ``selection_scores_`` and ``top_features_`` describe the greedy selection.
    ``n_jobs=-1`` uses the configured Numba pool.
    """

    def __init__(self, n_features_to_select=10, backend="auto", n_jobs=-1):
        self.n_features_to_select = n_features_to_select
        self.backend = backend
        self.n_jobs = n_jobs

    def _fit_information(self, X, y, criterion):
        if y is None:
            raise ValueError("This selector requires y to be passed, but the target y is None")
        X, y = validate_data(self, X, y, dtype=None, ensure_min_samples=2)
        check_classification_targets(y)
        threads = resolve_num_threads(self.n_jobs)
        if criterion != "FCBF":
            check_integer_param(self.n_features_to_select, "n_features_to_select", 1)
            if self.n_features_to_select > X.shape[1]:
                raise ValueError("n_features_to_select must not exceed the feature count")
        elif isinstance(self.delta, (bool, np.bool_)) or not np.isscalar(self.delta) or not 0 <= self.delta <= 1:
            raise ValueError("delta must be a finite number in [0, 1]")
        encoded, labels, counts, sy = _encode_categories(X, y)
        labels = _first_codes(labels)
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        previous = get_num_threads()
        try:
            set_num_threads(threads)
            columns, entropy, joint = _initial(np.ascontiguousarray(encoded.T), labels, counts, sy)
            hy = _entropy_one(labels, sy)
            relevance = -joint + entropy + hy
            self.relevance_scores_ = relevance
            if criterion == "FCBF":
                denom = entropy + hy
                relevance = np.divide(2 * relevance, denom, out=np.zeros_like(relevance), where=denom > 0)
            self.feature_importances_ = relevance.copy()
            active = relevance > self.delta if criterion == "FCBF" else np.ones(len(counts), dtype=bool)
            selected, scores = [], []
            redundancy = np.zeros(len(counts), dtype=np.float64)
            conditional_sum = np.zeros(len(counts), dtype=np.float64)
            maximum = np.full(len(counts), -np.inf)
            current = relevance.copy()
            while np.any(active):
                idx = int(np.argmax(np.where(active, current, -np.inf)))
                selected.append(idx)
                scores.append(current[idx])
                active[idx] = False
                if criterion != "FCBF" and len(selected) == self.n_features_to_select:
                    break
                if not active.any():
                    break
                hp, ht = _step_cpu(columns, labels, counts, sy, idx, active, criterion != "FCBF")
                if criterion == "FCBF":
                    pair_mi = -hp + entropy[idx] + entropy
                    denom = entropy[idx] + entropy
                    su = np.divide(2 * pair_mi, denom, out=np.zeros_like(denom), where=denom > 0)
                    active &= su < relevance
                    current = relevance
                else:
                    pair_mi = -hp + entropy[idx] + entropy
                    conditional = joint + joint[idx] - ht - hy
                    if criterion == "JMI":
                        redundancy += pair_mi
                        conditional_sum += conditional
                        factor = 1.0 / len(selected)
                        current = relevance - factor * redundancy + factor * conditional_sum
                    else:
                        maximum = np.maximum(maximum, pair_mi - conditional)
                        current = relevance - maximum
            self.top_features_ = np.array(selected, dtype=np.int64)
            self.selection_scores_ = np.array(scores, dtype=np.float64)
        finally:
            set_num_threads(previous)
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=None)
        return X[:, self.top_features_]


class JMI(_InformationSelector):
    """Joint Mutual Information, Yang & Moody (1999) / Brown et al. (2012).

    Discrete classification. At each step maximize relevance minus average
    redundancy plus average within-class dependence. This is the normalized
    JMI criterion; it preserves the ranking of the original joint-MI sum.
    """

    def fit(self, X, y):
        return self._fit_information(X, y, "JMI")


class CMIM(_InformationSelector):
    """Conditional MI Maximization, Fleuret (2004), discrete extension.

    Maximize the minimum I(candidate; target | selected feature), using the
    equivalent Brown et al. (2012) entropy identity. No top-k approximation.
    This follows Fleuret's Eq. (2), which Brown et al. also use. His
    Algorithm 1 pseudocode additionally seeds each score with I(candidate;
    target), so it can differ when every conditional MI exceeds the marginal.
    """

    def fit(self, X, y):
        return self._fit_information(X, y, "CMIM")


class FCBF(_InformationSelector):
    """Yu & Liu (2003/2004) Fast Correlation Based Filter.

    Keep features with SU(feature,target) > delta. Repeatedly select the most
    relevant remaining feature and remove peers whose SU with it is >= their
    relevance. The selected feature count is determined by the filter.
    """

    def __init__(self, delta=0.0, backend="auto", n_jobs=-1):
        self.delta = delta
        self.backend = backend
        self.n_jobs = n_jobs

    def fit(self, X, y):
        return self._fit_information(X, y, "FCBF")
