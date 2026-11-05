"""STIR: STatistical Inference Relief, a pseudo t-test on Relief's hit and miss differences.

Source
------
Le, Urbanowicz, Moore and McKinney, *STatistical Inference Relief (STIR)
feature selection*, Bioinformatics 35(8):1358-1365 (2019), DOI
10.1093/bioinformatics/bty788.  For binary (case-control) data and a
neighbourhood that gives target ``R_i`` the hits ``H_ij`` (``k_Hi`` of them)
and misses ``M_ij`` (``k_Mi``):

* Eqs. (5)-(6): ``M_a = (1/m) sum_i (1/k_Mi) sum_j diff(a, R_i, M_ij)``, and
  ``H_a`` likewise; Eq. (7): the Relief score ``W_R = M_a - H_a``.
* Eqs. (12)-(13): ``S2_M = (1/m) sum_i (1/k_Mi) sum_j (diff - M_a)^2``, and
  ``S2_H`` likewise.
* Eq. (11): ``S_p = sqrt(((|M| - 1) S2_M + (|H| - 1) S2_H) / (|M| + |H| - 2))``
  with ``|M| = sum_i k_Mi`` and ``|H| = sum_i k_Hi``.
* Eq. (10): ``W_STIR = (M_a - H_a) / (S_p sqrt(1/|M| + 1/|H|))``, compared
  with a t distribution on ``|M| + |H| - 2`` degrees of freedom.

P-values are one-sided (a larger miss difference is evidence of relevance), as
in the authors' R implementation, and adjusted with Benjamini-Hochberg, as the
paper states.
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cpu_backend import resolve_cpu_backend
from .utils import (
    BOUNDARY_RTOL,
    check_integer_param,
    discrete_feature_mask,
    prepare_relief_matrix,
    resolve_num_threads,
    split_discrete_last,
)

NEIGHBORHOODS = ("multisurf", "relieff")
_FASTMATH = {"reassoc", "contract", "nsz"}


@njit(parallel=True)
def _feature_pair_mean_diffs(x, n_cont):  # pragma: no cover
    """Mean per-feature difference over all unique pairs (the shift that keeps the variance sums exact-ish)."""
    n_samples, n_features = x.shape
    pairs = n_samples * (n_samples - 1) // 2
    out = np.zeros(n_features, dtype=np.float64)
    for f in prange(n_features):
        values = np.sort(x[:, f])
        total = 0.0
        if f >= n_cont:
            start = 0
            while start < n_samples:
                end = start + 1
                while end < n_samples and values[end] == values[start]:
                    end += 1
                run = end - start
                total += run * (n_samples - run)
                start = end
            total *= 0.5
        else:
            for i in range(n_samples):
                total += values[i] * (2 * i - n_samples + 1)
        out[f] = total / pairs if pairs > 0 else 0.0
    return out


@njit(inline="always")
def _insert_nearest(dists, keys, idx, size, k, j, tolerance):  # pragma: no cover
    """Keep the ``k`` smallest distances in ``keys``; equal ones (within tolerance) keep the lower index."""
    d = dists[j]
    if size == k:
        if not d < keys[k - 1] - tolerance:
            return size
        pos = k - 1
    else:
        pos = size
        size += 1
    while pos > 0 and d < keys[pos - 1] - tolerance:
        keys[pos] = keys[pos - 1]
        idx[pos] = idx[pos - 1]
        pos -= 1
    keys[pos] = d
    idx[pos] = j
    return size


@njit(parallel=True, fastmath=_FASTMATH)
def _stir_cpu_kernel(x, y, n_cont, use_multisurf, k, shift, out):  # pragma: no cover
    """Accumulate the STIR sums over every target instance.

    Only targets with at least one hit and one miss contribute. ``out`` rows (per
    feature): 0/1 hit/miss ``sum w (diff - shift)``, 2/3 hit/miss
    ``sum w (diff - shift)^2``, 4/5 hit/miss min diff, 6/7 hit/miss max diff, with
    ``w = 1 / k_i``.  Returns ``(|H|, |M|, m)`` where ``m`` counts the
    contributing targets.
    """
    n_samples, n_features = x.shape
    n_threads = get_num_threads()
    sums = np.zeros((n_threads, 4, n_features), dtype=np.float64)
    lows = np.full((n_threads, 2, n_features), np.inf, dtype=np.float64)
    highs = np.full((n_threads, 2, n_features), -np.inf, dtype=np.float64)
    counts = np.zeros((n_threads, 3), dtype=np.int64)
    thread_dists = np.empty((n_threads, n_samples), dtype=np.float64)
    thread_members = np.empty((n_threads, 2, n_samples), dtype=np.int64)
    thread_keys = np.empty((n_threads, 2, max(k, 1)), dtype=np.float64)

    for i in prange(n_samples):
        tid = get_thread_id()
        dists = thread_dists[tid]
        members = thread_members[tid]
        x_i = x[i]
        total = 0.0
        for j in range(n_samples):
            if j == i:
                dists[j] = 0.0
                continue
            x_j = x[j]
            d = 0.0
            for f in range(n_cont):
                d += abs(x_i[f] - x_j[f])
            for f in range(n_cont, n_features):
                if x_i[f] != x_j[f]:
                    d += 1.0
            dists[j] = d
            total += d
        mu = total / (n_samples - 1)
        tolerance = BOUNDARY_RTOL * mu

        n_group = np.zeros(2, dtype=np.int64)  # 0 = hits, 1 = misses
        if use_multisurf:
            dev_sum = 0.0
            sq_sum = 0.0
            for j in range(n_samples):
                if j != i:
                    dev = dists[j] - mu
                    dev_sum += dev
                    sq_sum += dev * dev
            mean_dev = dev_sum / (n_samples - 1)
            variance = sq_sum / (n_samples - 1) - mean_dev * mean_dev
            sigma = math.sqrt(variance) if variance > 0.0 else 0.0
            near_limit = mu - 0.5 * sigma - tolerance
            for j in range(n_samples):
                if j != i and dists[j] < near_limit:
                    g = 0 if y[j] == y[i] else 1
                    members[g, n_group[g]] = j
                    n_group[g] += 1
        else:
            keys = thread_keys[tid]
            for j in range(n_samples):
                if j != i:
                    g = 0 if y[j] == y[i] else 1
                    n_group[g] = _insert_nearest(dists, keys[g], members[g], n_group[g], k, j, tolerance)

        if n_group[0] == 0 or n_group[1] == 0:
            continue  # Eqs. (5)-(6) need both a hit and a miss mean for this target
        counts[tid, 2] += 1
        for g in range(2):
            size = n_group[g]
            weight = 1.0 / size
            counts[tid, g] += size
            acc = sums[tid, g]
            acc_sq = sums[tid, 2 + g]
            low = lows[tid, g]
            high = highs[tid, g]
            for r in range(size):
                x_j = x[members[g, r]]
                for f in range(n_cont):
                    value = abs(x_i[f] - x_j[f])
                    centred = value - shift[f]
                    acc[f] += weight * centred
                    acc_sq[f] += weight * centred * centred
                    low[f] = min(low[f], value)
                    high[f] = max(high[f], value)
                for f in range(n_cont, n_features):
                    value = 1.0 if x_i[f] != x_j[f] else 0.0
                    centred = value - shift[f]
                    acc[f] += weight * centred
                    acc_sq[f] += weight * centred * centred
                    low[f] = min(low[f], value)
                    high[f] = max(high[f], value)

    for f in range(n_features):
        for row in range(4):
            value = 0.0
            for t in range(n_threads):
                value += sums[t, row, f]
            out[row, f] = value
        for g in range(2):
            low = np.inf
            high = -np.inf
            for t in range(n_threads):
                low = min(low, lows[t, g, f])
                high = max(high, highs[t, g, f])
            out[4 + g, f] = low
            out[6 + g, f] = high
    totals = np.zeros(3, dtype=np.int64)
    for t in range(n_threads):
        for c in range(3):
            totals[c] += counts[t, c]
    return totals


def _benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values (step-up, monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(adjusted, 1.0)
    return out


class STIR(TransformerMixin, BaseEstimator):
    """STIR: Relief scores with p-values for binary classification.

    STIR (Le et al., 2019) recasts Relief's score ``W = mean miss diff - mean
    hit diff`` as a difference of two means and divides it by its pooled
    standard error, giving a pseudo t statistic per feature with a p-value.
    Features can then be chosen by significance, with a false-discovery-rate
    adjustment for testing many features, instead of an arbitrary top-k.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features (by t statistic) to select when ``alpha`` is
        None. A float selects that fraction of the features; an int that many.

    neighborhood : {'multisurf', 'relieff'}, default='multisurf'
        ``'multisurf'``: each target's neighbours are the instances closer than
        its mean distance minus half its standard deviation (MultiSURF, no far
        neighbours). ``'relieff'``: the ``n_neighbors`` nearest hits and
        nearest misses.

    n_neighbors : int or None, default=None
        ``k`` for ``neighborhood='relieff'``. None uses ``n_samples // 6``,
        which the paper reports approximates MultiSURF's neighbourhood size on
        balanced data. Ignored for ``'multisurf'``.

    alpha : float or None, default=None
        If set, select every feature whose Benjamini-Hochberg adjusted p-value
        is below ``alpha`` (possibly none), best first, instead of using
        ``n_features_to_select``.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    backend : {'auto', 'cpu'}, default='auto'
        STIR runs on the CPU only; no GPU implementation has been measured.

    n_jobs : int, default=-1
        Number of Numba threads; ``-1`` uses the configured pool.

    verbose : bool, default=False
        Controls whether progress messages are printed during fit.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The STIR t statistic of each feature (Eq. 10); higher is better.

    relief_scores_ : ndarray of shape (n_features,)
        The Relief score ``M_a - H_a`` (Eq. 7). With ``'multisurf'`` this is
        MultiSURF's score.

    p_values_ : ndarray of shape (n_features,)
        One-sided p-values ``P(T >= t)`` on ``degrees_of_freedom_``.

    p_values_adjusted_ : ndarray of shape (n_features,)
        Benjamini-Hochberg adjusted p-values.

    degrees_of_freedom_ : int
        ``|M| + |H| - 2``.

    n_hit_pairs_, n_miss_pairs_ : int
        ``|H|`` and ``|M|``, the numbers of (target, neighbour) pairs.

    n_targets_ : int
        ``m``, the number of targets with at least one hit and one miss.

    top_features_ : ndarray
        Indices of the selected features, best first.

    is_discrete_ : ndarray of shape (n_features,)
        Which features were treated as discrete.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    The paper's formalism is for binary (case-control) targets; other targets
    are rejected. Conventions where the paper is silent:

    * Eqs. (5)-(6) average over the same ``m`` targets, so a target needs at
      least one hit and one miss in its neighbourhood to contribute; ``m``
      (``n_targets_``) counts those that do (the authors' R code requires two
      of each). When every target qualifies, the Relief score with
      ``'multisurf'`` equals MultiSURF's.
    * ``sigma_i`` in the MultiSURF radius is the population standard deviation
      over the other instances, as in :class:`MultiSURF` (the authors' R code
      uses the sample form), and a distance within ``utils.BOUNDARY_RTOL`` of
      the radius is not a neighbour. Equal distances in the ``'relieff'``
      neighbourhood keep the lower index.
    * A feature whose hit and miss differences have zero pooled variance has
      ``t = +inf`` or ``-inf`` when the means differ and ``t = 0`` (p-value
      0.5) when they do not, such as a constant feature.

    The variances are accumulated in one pass, shifted by each feature's mean
    pair difference; a group whose differences are all equal is detected
    exactly. Continuous features use the range-normalised difference and the
    distance is their Manhattan sum (Eqs. 1-2). Input containing NaNs is
    rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        neighborhood: str = "multisurf",
        n_neighbors: int | None = None,
        alpha: float | None = None,
        discrete_limit: int = 10,
        backend: str = "auto",
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.n_features_to_select = n_features_to_select
        self.neighborhood = neighborhood
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.discrete_limit = discrete_limit
        self.backend = backend
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _validate_parameters(self, n_samples: int, n_features: int) -> tuple[int, int]:
        if n_samples < 3:
            raise ValueError(f"STIR requires at least 3 samples, but got n_samples = {n_samples}")
        if self.neighborhood not in NEIGHBORHOODS:
            raise ValueError(f"neighborhood must be one of {NEIGHBORHOODS}, got {self.neighborhood!r}.")
        k = 0
        if self.neighborhood == "relieff":
            if self.n_neighbors is None:
                k = max(1, n_samples // 6)
            else:
                k = check_integer_param(self.n_neighbors, "n_neighbors", 1)
                if k >= n_samples:
                    raise ValueError(f"n_neighbors ({k}) must be between 1 and n_samples - 1 ({n_samples - 1}).")
        if self.alpha is not None and (
            isinstance(self.alpha, (bool, np.bool_))
            or not isinstance(self.alpha, (int, float, np.integer, np.floating))
            or not 0.0 < float(self.alpha) <= 1.0
        ):
            raise ValueError(f"alpha must be None or a number in (0, 1], got {self.alpha!r}.")
        check_integer_param(self.discrete_limit, "discrete_limit", 0)
        resolve_num_threads(self.n_jobs)

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            return max(1, int(self.n_features_to_select * n_features)), k
        if isinstance(self.n_features_to_select, (int, np.integer)) and not isinstance(
            self.n_features_to_select, (bool, np.bool_)
        ):
            if not 0 < self.n_features_to_select <= n_features:
                raise ValueError(
                    f"If n_features_to_select is an int ({self.n_features_to_select}), "
                    f"it must be > 0 and <= n_features ({n_features})."
                )
            return int(self.n_features_to_select), k
        raise TypeError("n_features_to_select must be an int or a float.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Compute STIR statistics and p-values, then select features."""
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32], ensure_2d=True, y_numeric=True)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        n_select, k = self._validate_parameters(n_samples, self.n_features_in_)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"STIR is defined for binary targets; got {len(self.classes_)} classes.")
        self.is_discrete_ = discrete_feature_mask(X, self.discrete_limit)
        columns, n_cont = split_discrete_last(self.is_discrete_)
        X_prepared = prepare_relief_matrix(
            X, self.is_discrete_, columns, dtype=np.float64, discrete_limit=self.discrete_limit
        )
        if self.verbose:
            print(f"Running STIR ({self.neighborhood}) on the CPU now...")

        out = np.empty((8, X_prepared.shape[1]), dtype=np.float64)
        original_num_threads = get_num_threads()
        set_num_threads(resolve_num_threads(self.n_jobs))
        try:
            shift = _feature_pair_mean_diffs(X_prepared, n_cont)
            totals = _stir_cpu_kernel(
                X_prepared, y_encoded.astype(np.int64), n_cont, self.neighborhood == "multisurf", k, shift, out
            )
        finally:
            set_num_threads(original_num_threads)
        n_hits, n_misses, n_targets = (int(v) for v in totals)
        if n_targets == 0 or n_hits + n_misses <= 2:
            raise ValueError(
                "No target has both a hit and a miss in its neighbourhood (or there are too few pairs), "
                "so STIR's t statistic is undefined."
            )

        out[:4] /= n_targets  # the 1/m of Eqs. (5)-(6) and (12)-(13)
        statistic, relief, p_values = self._statistics(out, shift, n_hits, n_misses)
        self.n_targets_ = n_targets
        self.n_hit_pairs_ = n_hits
        self.n_miss_pairs_ = n_misses
        self.degrees_of_freedom_ = n_hits + n_misses - 2
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self.feature_importances_[columns] = statistic
        self.relief_scores_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self.relief_scores_[columns] = relief
        self.p_values_ = np.ones(self.n_features_in_, dtype=np.float64)
        self.p_values_[columns] = p_values
        self.p_values_adjusted_ = _benjamini_hochberg(self.p_values_)

        order = np.argsort(-self.feature_importances_, kind="stable")
        if self.alpha is None:
            self.top_features_ = order[:n_select]
        else:
            self.top_features_ = order[self.p_values_adjusted_[order] < float(self.alpha)]
        return self

    @staticmethod
    def _statistics(out, shift, n_hits, n_misses):
        """Eqs. (5)-(13) from the shifted, ``1/m``-scaled sums; returns ``(t, relief score, p-value)``."""
        means = []
        variances = []
        for g in range(2):
            centred_mean = out[g]  # (1/m) sum_i (1/k_i) sum_j (d - c) = mean - c
            mean = centred_mean + shift
            # (1/m) sum_i (1/k_i) sum_j (d - mean)^2 = E_w[(d - c)^2] - (mean - c)^2
            variance = np.maximum(out[2 + g] - centred_mean * centred_mean, 0.0)
            # Every difference in the group is the same value: that value is the mean
            # and the variance is exactly zero.
            constant = out[4 + g] == out[6 + g]
            mean = np.where(constant, out[4 + g], mean)
            variance[constant] = 0.0
            means.append(mean)
            variances.append(variance)
        hit_mean, miss_mean = means
        relief = miss_mean - hit_mean
        pooled = np.sqrt(((n_misses - 1) * variances[1] + (n_hits - 1) * variances[0]) / (n_misses + n_hits - 2))
        scale = pooled * math.sqrt(1.0 / n_misses + 1.0 / n_hits)
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = np.where(scale > 0.0, relief / scale, np.sign(relief) * np.inf)
        statistic = np.where((scale == 0.0) & (relief == 0.0), 0.0, statistic)
        p_values = student_t.sf(statistic, n_hits + n_misses - 2)
        return statistic, relief, p_values

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce ``X`` to the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
