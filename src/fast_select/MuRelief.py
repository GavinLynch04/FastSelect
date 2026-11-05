"""μ-Relief (mean-based Relief): ReliefF with neighbours chosen far from class-mean distances.

Source
------
Aggarwal, Shukla, Saxena, Rawat, Bafila, Singh and Pundir, *Mean based relief:
An improved feature selection method based on ReliefF*, Applied Intelligence
53:23004-23028 (2023), DOI 10.1007/s10489-023-04662-w.  The original article
could not be consulted (it is behind a publisher login), so this module
follows its restatement in Kazemi-Nia, Bandhey, Freda and Urbanowicz,
arXiv:2608.28552 (2026), Eq. (2), whose authors implemented it "as described"
in the original::

    dist*_ij = |dist_ij - mu_{C_j, i}|

where ``mu_{C_j, i}`` is the mean distance from target ``R_i`` to the
instances of ``R_j``'s class.  The neighbourhood is the ``k`` hits and ``k``
misses with the largest ``dist*``, scored with ReliefF's standard (unweighted)
update.  Every rule the restatement leaves open is listed as a convention in
:class:`MuRelief`'s Notes.
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads
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

# As in SWRF.py: vectorise the feature loops without approximate maths.
_FASTMATH = {"reassoc", "contract", "nsz"}


@njit(parallel=True, fastmath=_FASTMATH)
def _murelief_cpu_kernel(x, y_enc, n_cont, k, class_probs, scores_out):  # pragma: no cover
    """μ-Relief scoring with every instance the target once.

    For target ``i``: (1) distances to every other instance and their mean per
    class, ``mu_c`` (``i`` itself excluded from its own class); (2) for each
    class, the ``k`` instances with the largest ``|d - mu_c|``, ties (within
    ``BOUNDARY_RTOL * mu_c``) keeping the lower index; (3) ReliefF's update,
    ``-diff / (h * n)`` for each of the ``h`` hits and
    ``P(c) / (1 - P(class_i)) * diff / (m_c * n)`` for each of the ``m_c``
    misses of class ``c``.
    """
    n_samples, n_features = x.shape
    n_classes = class_probs.shape[0]
    n_threads = get_num_threads()
    thread_scores = np.zeros((n_threads, n_features), dtype=np.float64)
    thread_dists = np.empty((n_threads, n_samples), dtype=np.float64)
    thread_sums = np.empty((n_threads, n_classes), dtype=np.float64)
    thread_counts = np.empty((n_threads, n_classes), dtype=np.int64)
    thread_keys = np.empty((n_threads, n_classes, k), dtype=np.float64)
    thread_idx = np.empty((n_threads, n_classes, k), dtype=np.int64)
    thread_found = np.empty((n_threads, n_classes), dtype=np.int64)

    for i in prange(n_samples):
        tid = get_thread_id()
        dists = thread_dists[tid]
        sums = thread_sums[tid]
        counts = thread_counts[tid]
        keys = thread_keys[tid]
        idx = thread_idx[tid]
        found = thread_found[tid]
        x_i = x[i]
        label_i = y_enc[i]

        for c in range(n_classes):
            sums[c] = 0.0
            counts[c] = 0
            found[c] = 0

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
            sums[y_enc[j]] += d
            counts[y_enc[j]] += 1

        # Top-k by |d - mu_c|, largest first.  j ascends, so a candidate only
        # moves ahead of an entry it beats by more than the tie tolerance and
        # equal keys stay in index order.
        for j in range(n_samples):
            if j == i:
                continue
            c = y_enc[j]
            mu = sums[c] / counts[c]
            key = abs(dists[j] - mu)
            tolerance = BOUNDARY_RTOL * mu
            size = found[c]
            if size == k:
                if not key > keys[c, k - 1] + tolerance:
                    continue
                pos = k - 1
            else:
                pos = size
                found[c] = size + 1
            while pos > 0 and key > keys[c, pos - 1] + tolerance:
                keys[c, pos] = keys[c, pos - 1]
                idx[c, pos] = idx[c, pos - 1]
                pos -= 1
            keys[c, pos] = key
            idx[c, pos] = j

        miss_denominator = 1.0 - class_probs[label_i]
        scores_i = thread_scores[tid]
        for c in range(n_classes):
            m = found[c]
            if m == 0:
                continue
            if c == label_i:
                coefficient = -1.0 / (m * n_samples)
            else:
                coefficient = class_probs[c] / miss_denominator / (m * n_samples)
            for r in range(m):
                x_n = x[idx[c, r]]
                for f in range(n_cont):
                    scores_i[f] += coefficient * abs(x_i[f] - x_n[f])
                for f in range(n_cont, n_features):
                    if x_i[f] != x_n[f]:
                        scores_i[f] += coefficient

    for f in range(n_features):
        total = 0.0
        for t in range(n_threads):
            total += thread_scores[t, f]
        scores_out[f] = total


def _murelief_cpu_host_caller(x, y_enc, n_cont, k, class_probs, n_jobs):
    scores = np.zeros(x.shape[1], dtype=np.float64)
    original_num_threads = get_num_threads()
    set_num_threads(resolve_num_threads(n_jobs))
    try:
        _murelief_cpu_kernel(x, y_enc, n_cont, k, class_probs, scores)
    finally:
        set_num_threads(original_num_threads)
    return scores


class MuRelief(TransformerMixin, BaseEstimator):
    """μ-Relief: ReliefF whose neighbours lie far from the class-mean distance.

    ReliefF scores a feature with the ``k`` nearest hits and misses of every
    instance. μ-Relief (Aggarwal et al., 2023) keeps ReliefF's update but picks
    different neighbours: for target ``R_i`` and each class ``C`` it computes
    the mean distance ``mu_C`` from ``R_i`` to that class's instances and takes
    the ``k`` instances of the class whose distance deviates most from it,
    ``|dist(R_i, R_j) - mu_C|``, in either direction. Both unusually close and
    unusually distant instances are therefore used, with ReliefF's standard
    weights.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select. A float selects that fraction of
        the features; an int selects that many.

    n_neighbors : int, default=10
        ``k``, the number of hits and of misses per class used for each target.
        The benchmark that restates the method used 10 and 100.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    backend : {'auto', 'cpu'}, default='auto'
        μ-Relief runs on the CPU only; no GPU implementation has been measured.

    n_jobs : int, default=-1
        Number of Numba threads; ``-1`` uses the configured pool.

    verbose : bool, default=False
        Controls whether progress messages are printed during fit.

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
    The original article could not be consulted; this follows the
    restatement in arXiv:2608.28552, Eq. (2), by authors who implemented it
    from the original. The rules that restatement does not spell out follow
    this package's ReliefF and are conventions:

    * Every instance is the target once, and the class mean for the target's
      own class excludes the target itself.
    * Distances are the sum of the per-feature differences (range-normalised
      ``|a - b|`` for continuous features, 0/1 for discrete ones).
    * Multiclass targets take ``k`` misses from every other class, weighted by
      ``P(C) / (1 - P(class(R_i)))`` as in ReliefF.
    * A class with fewer than ``k`` other instances contributes all of them,
      normalised by the number used (a warning is issued, as for ReliefF);
      a target with no other instance of its class has no hit term.
    * Equal deviations keep the lower instance index; deviations within
      ``utils.BOUNDARY_RTOL`` (relative to the class mean) count as equal.

    Classification targets only; input containing NaNs is rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        n_neighbors: int = 10,
        discrete_limit: int = 10,
        backend: str = "auto",
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.discrete_limit = discrete_limit
        self.backend = backend
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _validate_parameters(self, n_samples: int, n_features: int) -> int:
        if n_samples < 2:
            raise ValueError(f"MuRelief requires at least 2 samples, but got n_samples = {n_samples}")
        check_integer_param(self.n_neighbors, "n_neighbors", 1)
        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be an integer between 1 and n_samples - 1 ({n_samples - 1})."
            )
        check_integer_param(self.discrete_limit, "discrete_limit", 0)
        resolve_num_threads(self.n_jobs)

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            return max(1, int(self.n_features_to_select * n_features))
        if isinstance(self.n_features_to_select, (int, np.integer)) and not isinstance(
            self.n_features_to_select, (bool, np.bool_)
        ):
            if not 0 < self.n_features_to_select <= n_features:
                raise ValueError(
                    f"If n_features_to_select is an int ({self.n_features_to_select}), "
                    f"it must be > 0 and <= n_features ({n_features})."
                )
            return int(self.n_features_to_select)
        raise TypeError("n_features_to_select must be an int or a float.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Score every feature with μ-Relief and keep the best ``n_features_to_select``."""
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32], ensure_2d=True, y_numeric=True)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        n_select = self._validate_parameters(n_samples, self.n_features_in_)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.is_discrete_ = discrete_feature_mask(X, self.discrete_limit)
        if len(self.classes_) < 2:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
            self.top_features_ = np.arange(n_select)
            return self

        class_counts = np.bincount(y_encoded)
        if self.n_neighbors >= class_counts.min():
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than or equal to the "
                f"smallest class size ({class_counts.min()}).",
                UserWarning,
            )
        class_probs = class_counts / n_samples

        columns, n_cont = split_discrete_last(self.is_discrete_)
        X_prepared = prepare_relief_matrix(
            X, self.is_discrete_, columns, dtype=np.float64, discrete_limit=self.discrete_limit
        )
        if self.verbose:
            print("Running MuRelief on the CPU now...")
        permuted_scores = _murelief_cpu_host_caller(
            X_prepared, y_encoded.astype(np.int64), n_cont, int(self.n_neighbors), class_probs, self.n_jobs
        )

        scores = np.zeros(self.n_features_in_, dtype=np.float64)
        scores[columns] = permuted_scores
        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce ``X`` to the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
