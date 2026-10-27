"""Kraskov et al. (2004), first k-neighbor MI estimator, dense numeric inputs.

Public scale/jitter conventions match scikit-learn. Exact Chebyshev neighbor
search expands sorted x until its lower bound exceeds the current kth distance.
No approximate neighbors, discretization or changed k. Ross (2014) handles
explicitly discrete feature columns. All arithmetic is float64 on CPU.
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_X_y

from fast_select.ContinuousMI import _continuous_mi, _psi_table, _radius_count
from fast_select.utils import check_integer_param, resolve_num_threads


@njit(parallel=True, cache=True, nogil=True)
def _ksg(columns, target, k, psi):
    p, n = columns.shape
    target_sorted = np.sort(target)
    result = np.empty(p)
    for f in prange(p):
        order = np.argsort(columns[f])
        ordered = columns[f, order]
        y_ordered = target[order]
        sum_x, sum_y = 0.0, 0.0
        nearest = np.empty(k)
        for i in range(n):
            nearest[:] = np.inf
            left, right = i - 1, i + 1
            while left >= 0 or right < n:
                dl = ordered[i] - ordered[left] if left >= 0 else np.inf
                dr = ordered[right] - ordered[i] if right < n else np.inf
                if min(dl, dr) > nearest[0]:
                    break
                if dl <= dr:
                    j, dx = left, dl
                    left -= 1
                else:
                    j, dx = right, dr
                    right += 1
                distance = max(dx, abs(y_ordered[j] - y_ordered[i]))
                if distance < nearest[0]:
                    position = 0
                    while position + 1 < k and nearest[position + 1] > distance:
                        nearest[position] = nearest[position + 1]
                        position += 1
                    nearest[position] = distance
            radius = np.nextafter(nearest[0], 0.0)
            sum_x += psi[_radius_count(ordered, ordered[i], radius)]
            sum_y += psi[_radius_count(target_sorted, y_ordered[i], radius)]
        result[f] = max(0.0, psi[n] + psi[k] - sum_x / n - sum_y / n)
    return result


def mutual_info_regression(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None, n_jobs=-1):
    """Dense numeric regression MI in nats, compatible with SelectKBest.

    Uses KSG estimator 1 for continuous columns and Ross for discrete columns.
    Negative estimates are clipped at zero. Set random_state for reproducible
    jitter. CPU-only; sorted search is exact but can approach quadratic work
    for adversarial data or large n_neighbors. Sparse inputs are unsupported.
    """
    X, y = check_X_y(X, y, dtype="numeric", y_numeric=True)
    check_integer_param(n_neighbors, "n_neighbors", 1)
    n, p = X.shape
    if n_neighbors >= n:
        raise ValueError("n_neighbors must be smaller than the sample count")
    threads = resolve_num_threads(1 if n_jobs is None else n_jobs)
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
    rng = check_random_state(random_state)
    if np.any(~mask):
        X = X.astype(np.float64, copy=copy)
        X[:, ~mask] = scale(X[:, ~mask], with_mean=False, copy=False)
        means = np.maximum(1, np.mean(np.abs(X[:, ~mask]), axis=0))
        X[:, ~mask] += 1e-10 * means * rng.standard_normal((n, np.count_nonzero(~mask)))
    y = scale(np.asarray(y, dtype=np.float64), with_mean=False)
    y += 1e-10 * max(1, np.mean(np.abs(y))) * rng.standard_normal(n)
    result = np.empty(p)
    previous = get_num_threads()
    try:
        set_num_threads(threads)
        if np.any(~mask):
            result[~mask] = _ksg(np.ascontiguousarray(X[:, ~mask].T), y, n_neighbors, _psi_table(n))
        for f in np.flatnonzero(mask):
            _, labels = np.unique(X[:, f], return_inverse=True)
            counts = np.bincount(labels)
            if not np.any(counts > 1):
                raise ValueError("Discrete feature MI requires a category with multiple samples")
            members = np.argsort(labels, kind="stable")
            starts = np.concatenate(([0], np.cumsum(counts)))
            result[f] = _continuous_mi(y.reshape(1, n), labels, members, starts, n_neighbors, _psi_table(n))[0]
    finally:
        set_num_threads(previous)
    return result
