"""RReliefF: Relief attribute estimation for regression targets.

Robnik-Šikonja and Kononenko, *An adaptation of Relief for attribute estimation
in regression* (ICML 1997); the algorithm is restated in their *Theoretical and
Empirical Analysis of ReliefF and RReliefF* (Machine Learning 53, 2003) as
Figure 3 with Eqs. (9)-(11), which this module follows.
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cpu_backend import resolve_cpu_backend
from .utils import (
    check_integer_param,
    discrete_feature_mask,
    prepare_relief_matrix,
    resolve_num_threads,
    split_discrete_last,
)

WEIGHTINGS = ("exponential", "constant")


def _rank_weights(k: int, sigma: float, weighting: str) -> np.ndarray:
    """``d(i, j)`` of Eq. (10) for the neighbours ranked ``1..k``.

    Eq. (11) defines ``d1 = exp(-(rank / sigma)**2)``; the constant alternative
    of Section 2.3 is ``d1 = 1 / k``.  ``d`` depends only on the rank, so one
    vector serves every target instance.  The exponent is shifted by the rank-1
    term, a common factor that Eq. (10)'s normalisation cancels, so the
    normalising sum is at least 1 and cannot underflow for small ``sigma``.
    """
    if weighting == "constant":
        return np.full(k, 1.0 / k)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    d1 = np.exp(-(ranks * ranks - 1.0) / (sigma * sigma))
    return d1 / d1.sum()


@njit(parallel=True)
def _rrelieff_cpu_kernel(x, target, n_cont, rank_weights):  # pragma: no cover
    """Per-thread sums behind Figure 3, line 15.

    ``x`` holds range-normalised continuous features in ``[0, n_cont)``, so
    ``|a - b|`` is Eq. (2), and exact category codes after them (Eq. (1));
    ``target`` is the range-normalised prediction, so ``|t_i - t_j|`` is
    ``diff(tau)``.  Every instance is a target once (``m = n``).  Its neighbours
    are the ``k`` smallest Manhattan distances; equal distances keep the lower
    index, and the insertion order is the rank of Eq. (11).  ``fastmath`` stays
    off so each distance is summed in column order and neighbour membership is
    reproducible.

    Returns per-thread ``N_dC``, ``N_dC&dA[A]`` and their same-prediction
    complements ``m - N_dC`` and ``N_dA[A] - N_dC&dA[A]``, accumulated directly
    (with weight ``1 - diff(tau)``) instead of by subtraction.
    """
    n_samples, n_features = x.shape
    k = rank_weights.shape[0]
    n_threads = get_num_threads()

    diff_target = np.zeros(n_threads)
    same_target = np.zeros(n_threads)
    diff_target_diff_attr = np.zeros((n_threads, n_features))
    same_target_diff_attr = np.zeros((n_threads, n_features))

    for i in prange(n_samples):
        tid = get_thread_id()
        x_i = x[i]
        near_d = np.full(k, np.inf)
        near_idx = np.full(k, -1, dtype=np.int64)

        for j in range(n_samples):
            if i == j:
                continue
            x_j = x[j]
            d = 0.0
            for f in range(n_cont):
                d += abs(x_i[f] - x_j[f])
            for f in range(n_cont, n_features):
                if x_i[f] != x_j[f]:
                    d += 1.0
            if d < near_d[k - 1]:
                pos = k - 1
                while pos > 0 and d < near_d[pos - 1]:
                    near_d[pos] = near_d[pos - 1]
                    near_idx[pos] = near_idx[pos - 1]
                    pos -= 1
                near_d[pos] = d
                near_idx[pos] = j

        for rank in range(k):
            j = near_idx[rank]
            x_j = x[j]
            diff_c = abs(target[i] - target[j])
            diff_weight = diff_c * rank_weights[rank]
            same_weight = (1.0 - diff_c) * rank_weights[rank]
            diff_target[tid] += diff_weight
            same_target[tid] += same_weight
            for f in range(n_cont):
                diff_a = abs(x_i[f] - x_j[f])
                diff_target_diff_attr[tid, f] += diff_weight * diff_a
                same_target_diff_attr[tid, f] += same_weight * diff_a
            for f in range(n_cont, n_features):
                if x_i[f] != x_j[f]:
                    diff_target_diff_attr[tid, f] += diff_weight
                    same_target_diff_attr[tid, f] += same_weight

    return diff_target, diff_target_diff_attr, same_target, same_target_diff_attr


def _rrelieff_cpu_host_caller(x, target, n_cont, rank_weights, n_jobs):
    """Run the kernel and apply Figure 3, line 15 (Eq. (9))."""
    original_num_threads = get_num_threads()
    set_num_threads(resolve_num_threads(n_jobs))
    try:
        sums = _rrelieff_cpu_kernel(x, target, n_cont, rank_weights)
    finally:
        set_num_threads(original_num_threads)

    diff_target, diff_target_diff_attr, same_target, same_target_diff_attr = sums
    n_dc = diff_target.sum()
    n_sc = same_target.sum()
    # A group with zero total weight (no neighbour pair with a different, or no
    # pair with an equal, prediction) contributes nothing, as an empty ReliefF
    # neighbour group does; the paper's ratio is 0/0 there.
    first = diff_target_diff_attr.sum(axis=0) / n_dc if n_dc > 0.0 else np.zeros(x.shape[1])
    second = same_target_diff_attr.sum(axis=0) / n_sc if n_sc > 0.0 else np.zeros(x.shape[1])
    return first - second


class RReliefF(TransformerMixin, BaseEstimator):
    """Feature selection for regression targets with RReliefF.

    RReliefF (Robnik-Šikonja and Kononenko, 1997; 2003, Figure 3) replaces
    ReliefF's hits and misses with the probability that two nearby instances
    have different predicted values, modelled by their range-normalised target
    difference. Each feature's estimate is Eq. (9),
    ``P(diff C | diff A) P(diff A) / P(diff C) - (1 - P(diff C | diff A)) P(diff A) / (1 - P(diff C))``,
    evaluated from the ``k`` nearest neighbours of every instance with the
    rank weights of Eqs. (10)-(11).

    Parameters
    ----------
    n_features_to_select : int | float, default=0.2
        The number of top features to select. A float selects that fraction of
        the features; an int selects that many.

    discrete_limit : int, default=10
        Features with at most this many distinct values are treated as discrete
        (Eq. (1)); the others are continuous (Eq. (2)).

    n_neighbors : int, default=70
        ``k``, the number of nearest instances used for every target instance.
        The paper's default for exponential weighting is 70 with ``sigma=20``;
        for constant weighting it recommends 10.

    sigma : float, default=20.0
        ``sigma`` of Eq. (11), controlling how quickly a neighbour's influence
        decays with its rank. Ignored when ``weighting='constant'``.

    weighting : {'exponential', 'constant'}, default='exponential'
        ``'exponential'`` uses Eq. (11); ``'constant'`` gives each of the ``k``
        neighbours influence ``1/k``, as ReliefF does.

    backend : {'auto', 'cpu'}, default='auto'
        RReliefF runs on the CPU only; no GPU implementation has been measured.

    verbose : bool, default=False
        Controls whether progress updates are printed during the fit.

    n_jobs : int, default=-1
        Controls the number of threads utilized by Numba.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.

    feature_importances_ : ndarray of shape (n_features,)
        ``W[A]`` for each feature, in ``[-1, 1]``; higher is better.

    top_features_ : ndarray of shape (n_select,)
        Indices of the selected features, best first.

    is_discrete_ : ndarray of shape (n_features,)
        Which features were treated as discrete.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    Every instance is used as a target once (``m = n_samples``), the usual
    choice for ReliefF. Neighbours are ranked by Manhattan distance over the
    Eq. (1)/(2) differences; equal distances are ranked by the lower instance
    index. Differences use Eq. (2) directly, not the ramp function the paper
    also discusses. The two sums of line 15 that involve ``m - N_dC`` are
    accumulated directly as same-prediction sums, an algebraically identical
    form that avoids cancellation; a sum whose total weight is zero contributes
    zero, an explicit convention for the paper's 0/0. Input containing NaNs,
    and a constant target (whose difference Eq. (2) leaves undefined), are
    rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        discrete_limit: int = 10,
        n_neighbors: int = 70,
        sigma: float = 20.0,
        weighting: str = "exponential",
        backend: str = "auto",
        verbose: bool = False,
        n_jobs: int = -1,
    ):
        self.n_features_to_select = n_features_to_select
        self.discrete_limit = discrete_limit
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.weighting = weighting
        self.backend = backend
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _validate_parameters(self, n_samples, n_features):
        """Validate all user-provided parameters."""
        if n_samples < 2:
            raise ValueError(f"RReliefF requires at least 2 samples, but got n_samples = {n_samples}")

        check_integer_param(self.n_neighbors, "n_neighbors", 1)
        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be an integer "
                f"between 1 and n_samples - 1 ({n_samples - 1})."
            )
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {self.weighting!r}.")
        if (
            isinstance(self.sigma, (bool, np.bool_))
            or not isinstance(self.sigma, (int, float, np.integer, np.floating))
            or not math.isfinite(self.sigma)
            or self.sigma <= 0
        ):
            raise ValueError(f"sigma must be a positive finite number, got {self.sigma!r}.")
        check_integer_param(self.discrete_limit, "discrete_limit", 0)
        resolve_num_threads(self.n_jobs)

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            n_select = max(1, int(self.n_features_to_select * n_features))
        elif isinstance(self.n_features_to_select, int) and not isinstance(self.n_features_to_select, bool):
            if not 0 < self.n_features_to_select <= n_features:
                raise ValueError(
                    f"If n_features_to_select is an int ({self.n_features_to_select}), "
                    f"it must be > 0 and <= n_features ({n_features})."
                )
            n_select = self.n_features_to_select
        else:
            raise TypeError("n_features_to_select must be an int or a float.")

        return n_select

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Calculates feature importances using the RReliefF algorithm."""
        x, y = validate_data(
            self,
            x,
            y,
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            y_numeric=True,
        )
        self.n_features_in_ = x.shape[1]
        n_samples = x.shape[0]
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        n_select = self._validate_parameters(n_samples, self.n_features_in_)

        y = np.asarray(y, dtype=np.float64)
        low, high = y.min(), y.max()
        if not high > low:
            raise ValueError("RReliefF needs a target with at least two distinct values.")
        target = (y - low) / (high - low)

        is_discrete = discrete_feature_mask(x, self.discrete_limit)
        self.is_discrete_ = is_discrete
        columns, n_cont = split_discrete_last(is_discrete)
        x_prepared = prepare_relief_matrix(
            x, is_discrete, columns, dtype=np.float64, discrete_limit=self.discrete_limit
        )
        rank_weights = _rank_weights(self.n_neighbors, float(self.sigma), self.weighting)

        if self.verbose:
            print("Running RReliefF on the CPU now...")
        permuted_scores = _rrelieff_cpu_host_caller(x_prepared, target, n_cont, rank_weights, self.n_jobs)

        scores = np.zeros(self.n_features_in_, dtype=np.float64)
        scores[columns] = permuted_scores
        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Reduces x to the selected features."""
        check_is_fitted(self)

        x = validate_data(self, x, reset=False, dtype=[np.float64, np.float32])

        return x[:, self.top_features_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(x, y)
        return self.transform(x)
