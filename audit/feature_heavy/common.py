"""Exact host preparation/reduction shared by experimental GPU paths.

Research only. Runtime package sources are frozen and unchanged. Equations,
thresholds and selection state match the retained CPU implementations.
"""

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y

from fast_select.ContinuousMI import _continuous_mi, _psi_table, _radius_count
from fast_select.InformationSelectors import _entropy_one, _first_codes, _initial
from fast_select.mRMR import _encode_categories


def keys_for_sort(columns):
    """Order-preserving 64-bit integer keys, without narrowing symbol identity."""
    if columns.dtype.kind == "u":
        bits = columns.astype(np.uint64)
    elif columns.dtype.kind in "ib":
        bits = columns.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)
    else:
        values = columns.astype(np.float64, copy=True)
        values[values == 0] = 0.0  # canonicalize negative zero
        raw = values.view(np.uint64)
        bits = np.where(raw >> np.uint64(63), ~raw, raw ^ np.uint64(1 << 63))
    return (bits >> np.uint64(32)).astype(np.uint32), bits.astype(np.uint32)


@njit(parallel=True, cache=True)
def gini_from_orders(columns, labels, counts, orders):
    p, n = columns.shape
    result = np.empty(p)
    total = np.sum(counts * counts)
    for f in prange(p):
        left = np.zeros(len(counts), np.int64)
        right = counts.copy()
        lsq, rsq = 0, total
        best = 1 - total / (n * n)
        for i in range(n - 1):
            label = labels[orders[f, i]]
            lsq += 2 * left[label] + 1
            rsq -= 2 * right[label] - 1
            left[label] += 1
            right[label] -= 1
            if columns[f, orders[f, i]] != columns[f, orders[f, i + 1]]:
                a, b = i + 1, n - i - 1
                best = min(best, 1 - (lsq / a + rsq / b) / n)
        result[f] = best
    return result


def fit_gini(X, y, provider, select=10):
    X, y = check_X_y(X, y, dtype="numeric", ensure_min_samples=2)
    check_classification_targets(y)
    _, labels = np.unique(y, return_inverse=True)
    counts = np.bincount(labels)
    columns = np.ascontiguousarray(X.T)
    high, low = keys_for_sort(columns)
    orders = provider.sort(high, low)
    scores = gini_from_orders(columns, labels, counts, orders)
    return {"scores": scores, "selected": np.argsort(scores, kind="stable")[:select]}


@njit(parallel=True, cache=True)
def entropy_ordered(pair, triple, n):
    p = len(pair)
    hp, ht = np.zeros(p), np.zeros(p)
    log2 = np.log(2.0)
    for f in prange(p):
        for count in pair[f]:
            if count:
                probability = count / n
                hp[f] -= probability * (np.log(probability) / log2)
        for count in triple[f]:
            if count:
                probability = count / n
                ht[f] -= probability * (np.log(probability) / log2)
    return hp, ht


def fit_joint(X, y, criterion, provider, select=10, beta=0.5):
    X, y = check_X_y(X, y, dtype=None, ensure_min_samples=2)
    check_classification_targets(y)
    encoded, labels, counts, sy = _encode_categories(X, y)
    labels = _first_codes(labels)
    columns, entropy, joint = _initial(np.ascontiguousarray(encoded.T), labels, counts, sy)
    hy = _entropy_one(labels, sy)
    relevance = -joint + entropy + hy
    active = np.ones(len(counts), bool)
    redundancy, conditional = np.zeros(len(counts)), np.zeros(len(counts))
    current = relevance.copy()
    device = provider.histograms(columns, labels, counts, sy)
    chosen, objectives = [], []
    for step in range(select):
        idx = int(np.argmax(np.where(active, current, -np.inf)))
        chosen.append(idx)
        objectives.append(current[idx])
        active[idx] = False
        if step + 1 == select:
            break
        pair, triple = device.step(idx, active, criterion != "MIFS")
        hp, ht = entropy_ordered(pair, triple, len(X))
        if criterion == "DISR":
            conditional += np.divide(hp + hy - ht, ht, out=np.zeros_like(ht), where=ht > 0)
            current = conditional.copy()
        elif criterion == "MIFS":
            redundancy += -hp + entropy[idx] + entropy
            current = relevance - beta * redundancy
        else:
            redundancy += -hp + entropy[idx] + entropy
            conditional += joint + joint[idx] - ht - hy
            current = relevance - redundancy + conditional
    return {
        "scores": np.asarray(objectives),
        "selected": np.asarray(chosen),
        "relevance_all": relevance,
        "relevance_selected": relevance[chosen],
    }


def prepare_regression(X, y, k, seed=17, discrete=False):
    X, y = check_X_y(X, y, dtype="numeric", y_numeric=True)
    n, p = X.shape
    if not 1 <= k < n:
        raise ValueError("k must be positive and smaller than sample count")
    mask = np.full(p, discrete, bool) if np.isscalar(discrete) else np.asarray(discrete, bool)
    rng = check_random_state(seed)
    if np.any(~mask):
        X = X.astype(np.float64, copy=True)
        X[:, ~mask] = scale(X[:, ~mask], with_mean=False, copy=False)
        means = np.maximum(1, np.mean(np.abs(X[:, ~mask]), axis=0))
        X[:, ~mask] += 1e-10 * means * rng.standard_normal((n, (~mask).sum()))
    y = scale(np.asarray(y, np.float64), with_mean=False)
    y += 1e-10 * max(1, np.mean(np.abs(y))) * rng.standard_normal(n)
    return X, y, mask


@njit(parallel=True, cache=True)
def scores_from_radii(columns, target, radii, k, psi):
    p, n = columns.shape
    ys = np.sort(target)
    result = np.empty(p)
    for f in prange(p):
        order = np.argsort(columns[f])
        xs = columns[f, order]
        sx, sy = 0.0, 0.0
        for i in range(n):
            idx = order[i]
            r = radii[f, idx]
            sx += psi[_radius_count(xs, columns[f, idx], r)]
            sy += psi[_radius_count(ys, target[idx], r)]
        result[f] = max(0.0, psi[n] + psi[k] - sx / n - sy / n)
    return result


@njit(parallel=True, cache=True)
def repair_proposals(columns, target, neighbors, cutoff, error, k):
    p, n = columns.shape
    radii = np.empty((p, n))
    repaired = np.zeros(p, np.int64)
    for f in prange(p):
        buffer = np.empty(n)
        for i in range(n):
            used = 0
            for j in neighbors[f, i]:
                if j < n:
                    buffer[used] = max(abs(columns[f, j] - columns[f, i]), abs(target[j] - target[i]))
                    used += 1
            # Tiny candidate list: insertion sort avoids a per-row allocation.
            for a in range(1, used):
                value, b = buffer[a], a - 1
                while b >= 0 and buffer[b] > value:
                    buffer[b + 1] = buffer[b]
                    b -= 1
                buffer[b + 1] = value
            radius = buffer[k - 1] if used >= k else np.inf
            certified = used == n - 1 or (np.isfinite(cutoff[f, i]) and radius < cutoff[f, i] - error[f])
            if not certified:
                repaired[f] += 1
                for j in range(n):
                    buffer[j] = (
                        max(abs(columns[f, j] - columns[f, i]), abs(target[j] - target[i])) if i != j else np.inf
                    )
                radius = np.partition(buffer, k - 1)[k - 1]
            radii[f, i] = np.nextafter(radius, 0.0)
    return radii, repaired.sum()


def fit_regression(X, y, provider, k=3, seed=17, discrete=False):
    X, y, mask = prepare_regression(X, y, k, seed, discrete)
    n, p = X.shape
    result = np.empty(p)
    diagnostics = {}
    if np.any(~mask):
        columns = np.ascontiguousarray(X[:, ~mask].T)
        radii, diagnostics = provider.radii(columns, y, k)
        result[~mask] = scores_from_radii(columns, y, radii, k, _psi_table(n))
    for f in np.flatnonzero(mask):
        _, labels = np.unique(X[:, f], return_inverse=True)
        counts = np.bincount(labels)
        members = np.argsort(labels, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts)))
        if not np.any(counts > 1):
            raise ValueError("Discrete feature MI requires a category with multiple samples")
        result[f] = _continuous_mi(y.reshape(1, n), labels, members, starts, k, _psi_table(n))[0]
    return {"scores": result, "selected": np.argsort(-result, kind="stable")[:10], "diagnostics": diagnostics}


def run(name, X, y, provider, select=10):
    previous = get_num_threads()
    try:
        set_num_threads(config.NUMBA_NUM_THREADS)
        if name == "GiniIndex":
            return fit_gini(X, y, provider, select)
        if name == "mutual_info_regression":
            return fit_regression(X, y, provider)
        return fit_joint(X, y, name, provider, select)
    finally:
        set_num_threads(previous)
