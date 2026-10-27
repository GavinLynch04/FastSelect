"""Ross (2014) continuous/discrete MI with exact one-dimensional neighbors.

Dense input only. Public scaling/jitter conventions follow scikit-learn's
mutual_info_classif; the Ross estimator itself is implemented independently
using sorted class samples and exact float64 distance predicates. No binning,
approximate neighbors or changed k. Returns nat units, clipped at zero.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.special import digamma
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_X_y

from fast_select.utils import check_integer_param, resolve_num_threads


@lru_cache(maxsize=8)
def _psi_table(n):
    values = digamma(np.arange(n + 1, dtype=np.float64))
    values.flags.writeable = False
    return values


@njit(cache=True, nogil=True)
def _radius_count(ordered, value, radius):
    # Compare actual differences, not rounded value +/- radius endpoints.
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if value - ordered[mid] > radius:
            lo = mid + 1
        else:
            hi = mid
    left = lo
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if ordered[mid] - value <= radius:
            lo = mid + 1
        else:
            hi = mid
    return lo - left


@njit(parallel=True, cache=True, nogil=True)
def _continuous_mi(columns, labels, members, starts, k, psi):
    p, n = columns.shape
    result = np.empty(p, dtype=np.float64)
    sizes = starts[1:] - starts[:-1]
    valid = np.empty(n, dtype=np.bool_)
    for i in range(n):
        valid[i] = sizes[labels[i]] > 1
    retained = np.count_nonzero(valid)
    for f in prange(p):
        values = columns[f]
        radii = np.zeros(n, dtype=np.float64)
        ks = np.zeros(n, dtype=np.int64)
        for c in range(len(sizes)):
            count = sizes[c]
            if count <= 1:
                continue
            group_ids = members[starts[c] : starts[c + 1]]
            order = np.argsort(values[group_ids])
            ids = group_ids[order]
            points = values[ids]
            neighbors = min(k, count - 1)
            for position in range(count):
                left, right = position - 1, position + 1
                distance = 0.0
                for _ in range(neighbors):
                    dl = points[position] - points[left] if left >= 0 else np.inf
                    dr = points[right] - points[position] if right < count else np.inf
                    if dl <= dr:
                        distance = dl
                        left -= 1
                    else:
                        distance = dr
                        right += 1
                radii[ids[position]] = np.nextafter(distance, 0.0)
                ks[ids[position]] = neighbors
        ordered = np.sort(values[valid])
        term_k, term_label, term_radius = 0.0, 0.0, 0.0
        for i in range(n):
            if valid[i]:
                term_k += psi[ks[i]]
                term_label += psi[sizes[labels[i]]]
                count = _radius_count(ordered, values[i], radii[i])
                term_radius += psi[count]
        result[f] = max(0.0, psi[retained] + term_k / retained - term_label / retained - term_radius / retained)
    return result


def mutual_info_classif(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None, n_jobs=-1):
    """Dense Ross MI scores in nats, compatible with SelectKBest.

    Continuous features use scikit-learn's public scale/jitter convention.
    Discrete columns use empirical MI; no continuous discretization is added.
    This float64 Numba implementation uses CPU threads, not Metal arithmetic.
    """
    X, y = check_X_y(X, y, dtype="numeric")
    check_classification_targets(y)
    check_integer_param(n_neighbors, "n_neighbors", 1)
    threads = resolve_num_threads(1 if n_jobs is None else n_jobs)
    n, p = X.shape
    if isinstance(discrete_features, (str, bool, np.bool_)):
        if isinstance(discrete_features, str) and discrete_features != "auto":
            raise ValueError("discrete_features must be 'auto', bool, a mask, or column indices")
        mask = np.full(p, False if discrete_features == "auto" else discrete_features, dtype=bool)
    else:
        specification = check_array(discrete_features, ensure_2d=False, dtype=None)
        if specification.dtype == bool:
            if specification.shape != (p,):
                raise ValueError("discrete_features mask must match the feature count")
            mask = specification
        else:
            mask = np.zeros(p, dtype=bool)
            mask[specification] = True
    result = np.zeros(p, dtype=np.float64)
    if np.any(mask):
        from fast_select.mutual_information import calculate_mi_single_pair

        _, encoded_y = np.unique(y, return_inverse=True)
        for feature in np.flatnonzero(mask):
            _, codes = np.unique(X[:, feature], return_inverse=True)
            result[feature] = max(0.0, calculate_mi_single_pair(codes, encoded_y, backend="cpu", unit="nat"))
    if np.any(~mask):
        X = X.astype(np.float64, copy=copy)
        X[:, ~mask] = scale(X[:, ~mask], with_mean=False, copy=False)
        means = np.maximum(1, np.mean(np.abs(X[:, ~mask]), axis=0))
        X[:, ~mask] += 1e-10 * means * check_random_state(random_state).standard_normal((n, np.count_nonzero(~mask)))
        _, labels = np.unique(y, return_inverse=True)
        counts = np.bincount(labels)
        if not np.any(counts > 1):
            raise ValueError("Continuous MI requires at least one class with multiple samples")
        members = np.argsort(labels, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts)))
        previous = get_num_threads()
        try:
            set_num_threads(threads)
            result[~mask] = _continuous_mi(
                np.ascontiguousarray(X[:, ~mask].T), labels, members, starts, n_neighbors, _psi_table(n)
            )
        finally:
            set_num_threads(previous)
    return result
