"""ReliefSeq: ReliefF with a gene-wise (per-feature) adaptive number of neighbours.

Source
------
McKinney, White, Grill, Li, Kennedy, Poland and Oberg, *ReliefSeq: a gene-wise
adaptive-K nearest-neighbor feature selection tool for finding gene-gene
interactions and main effects in mRNA-Seq gene expression data*, PLoS ONE
8(12):e81527 (2013), DOI 10.1371/journal.pone.0081527:

* Eq. (1): ``W_{a,k} = DM_{a,k} - DH_{a,k}``, the Relief-F weight of feature
  ``a`` with ``k`` neighbours;
* Eqs. (2)-(3): ``DM_{a,k} = (1/(m k)) sum_i sum_{j<=k} diff(a, R_i, M_j(R_i))``
  over the ``k`` nearest misses, ``DH`` likewise over the ``k`` nearest hits;
* Eq. (4): ``diff = |value(a, R_i) - value(a, R_j)| / (max(a) - min(a))``,
  with neighbours found by the Manhattan sum of the diffs;
* Eqs. (5)-(6): ``k_hat(a) = argmax_k W_{a,k}`` over ``k = 1..k_max`` and the
  ReliefSeq weight ``W_a = W_{a, k_hat(a)}``.

Every ``k`` is computed from one neighbour ranking: the per-target lists of the
``k_max`` nearest hits and misses are built once, and running sums down those
lists give ``W_{a,k}`` for all ``k`` in a single pass instead of ``k_max``
separate Relief-F fits.
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

_FASTMATH = {"reassoc", "contract", "nsz"}


@njit(parallel=True, fastmath=_FASTMATH)
def _rank_neighbours_cpu(x, y_enc, n_cont, n_classes, k_max, neighbours, found):  # pragma: no cover
    """For every target, the ``k_max`` nearest instances of each class (own class = hits).

    ``neighbours[i, c, r]`` is the ``r``-th nearest instance of class ``c`` to
    ``i``; ``found[i, c]`` how many exist (at most ``k_max``).  Distances within
    ``BOUNDARY_RTOL`` of the target's mean distance count as equal, and equal
    distances keep the lower index.
    """
    n_samples, n_features = x.shape
    n_threads = get_num_threads()
    thread_dists = np.empty((n_threads, n_samples), dtype=np.float64)
    thread_keys = np.empty((n_threads, n_classes, k_max), dtype=np.float64)
    for i in prange(n_samples):
        tid = get_thread_id()
        dists = thread_dists[tid]
        keys = thread_keys[tid]
        x_i = x[i]
        total = 0.0
        for j in range(n_samples):
            if j == i:
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
        tolerance = BOUNDARY_RTOL * total / (n_samples - 1)
        for c in range(n_classes):
            found[i, c] = 0
        for j in range(n_samples):
            if j == i:
                continue
            c = y_enc[j]
            d = dists[j]
            size = found[i, c]
            if size == k_max:
                if not d < keys[c, k_max - 1] - tolerance:
                    continue
                pos = k_max - 1
            else:
                pos = size
                found[i, c] = size + 1
            while pos > 0 and d < keys[c, pos - 1] - tolerance:
                keys[c, pos] = keys[c, pos - 1]
                neighbours[i, c, pos] = neighbours[i, c, pos - 1]
                pos -= 1
            keys[c, pos] = d
            neighbours[i, c, pos] = j


@njit(parallel=True, fastmath=_FASTMATH)
def _scores_by_k_cpu(x_t, y_enc, n_cont, class_probs, neighbours, found, scores):  # pragma: no cover
    """``scores[k - 1, a] = W_{a,k}`` for every ``k`` and feature, parallel over features.

    For target ``i`` and class ``c`` the running sum of the first ``r + 1``
    ranked diffs, divided by the number used (``min(r + 1, found)``), is added
    with ReliefF's sign and class-prior weight to the row ``r``.
    """
    n_features, n_samples = x_t.shape
    n_classes = class_probs.shape[0]
    k_max = scores.shape[0]
    inv_n = 1.0 / n_samples
    for a in prange(n_features):
        column = x_t[a]
        discrete = a >= n_cont
        for i in range(n_samples):
            value_i = column[i]
            label_i = y_enc[i]
            miss_scale = 1.0 - class_probs[label_i]
            for c in range(n_classes):
                size = found[i, c]
                if size == 0:
                    continue
                if c == label_i:
                    coefficient = -inv_n
                else:
                    coefficient = class_probs[c] / miss_scale * inv_n
                running = 0.0
                for r in range(k_max):
                    if r < size:
                        other = column[neighbours[i, c, r]]
                        if discrete:
                            if other != value_i:
                                running += 1.0
                        else:
                            running += abs(value_i - other)
                        used = r + 1
                    else:
                        used = size
                    scores[r, a] += coefficient * running / used


class ReliefSeq(TransformerMixin, BaseEstimator):
    """ReliefSeq: ReliefF with the best number of neighbours chosen per feature.

    ReliefF's score depends on ``k``: main effects favour many neighbours and
    interactions fewer. ReliefSeq (McKinney et al., 2013) computes the
    Relief-F weight of every feature for every ``k`` from 1 to ``k_max`` and
    keeps, for each feature, its largest weight (Eqs. 5-6).

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select. A float selects that fraction of
        the features; an int selects that many.

    max_neighbors : int or None, default=None
        ``k_max``. None uses every possible ``k``: the smallest class size
        minus one, so that every target has ``k`` hits and ``k`` misses of each
        class (the paper swept "all 23 possible" for 24 subjects per class).
        For large samples, a smaller value limits the cost.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    backend : {'auto', 'cpu'}, default='auto'
        ReliefSeq runs on the CPU only; no GPU implementation has been measured.

    n_jobs : int, default=-1
        Number of Numba threads; ``-1`` uses the configured pool.

    verbose : bool, default=False
        Controls whether progress messages are printed during fit.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The ReliefSeq weight ``max_k W_{a,k}`` (Eq. 6); higher is better.

    best_k_ : ndarray of shape (n_features,)
        ``k_hat(a)`` (Eq. 5), the smallest ``k`` attaining the maximum.

    scores_by_k_ : ndarray of shape (n_features, ``max_neighbors_``)
        ``W_{a,k}`` with column ``k - 1``; column ``k - 1`` is ReliefF's score
        with ``n_neighbors=k``.

    max_neighbors_ : int
        The ``k_max`` used.

    top_features_ : ndarray of shape (n_select,)
        Indices of the selected features, best first.

    is_discrete_ : ndarray of shape (n_features,)
        Which features were treated as discrete.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    The paper's equations are for two classes. Multiclass targets use
    ReliefF's form, ``k`` misses from every other class weighted by
    ``P(C) / (1 - P(class(R_i)))``, which is what each column already is; for
    two classes it is Eqs. (1)-(3). Every instance is the target once
    (``m = n_samples``). Discrete features differ by 0 or 1. Equal distances
    keep the lower instance index, as in :class:`ReliefF`. If ``max_neighbors``
    exceeds what a class can supply, that class contributes all its instances,
    normalised by the number used (a warning is issued). Memory is
    ``n_samples * n_classes * max_neighbors`` neighbour indices plus the
    ``max_neighbors * n_features`` score matrix. Input containing NaNs is
    rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        max_neighbors: int | None = None,
        discrete_limit: int = 10,
        backend: str = "auto",
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.n_features_to_select = n_features_to_select
        self.max_neighbors = max_neighbors
        self.discrete_limit = discrete_limit
        self.backend = backend
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _validate_parameters(self, n_samples: int, n_features: int) -> int:
        if n_samples < 2:
            raise ValueError(f"ReliefSeq requires at least 2 samples, but got n_samples = {n_samples}")
        if self.max_neighbors is not None:
            k_max = check_integer_param(self.max_neighbors, "max_neighbors", 1)
            if k_max >= n_samples:
                raise ValueError(f"max_neighbors ({k_max}) must be between 1 and n_samples - 1 ({n_samples - 1}).")
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
        """Compute ``W_{a,k}`` for every ``k`` and keep each feature's best."""
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32], ensure_2d=True, y_numeric=True)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        n_select = self._validate_parameters(n_samples, self.n_features_in_)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.is_discrete_ = discrete_feature_mask(X, self.discrete_limit)
        if len(self.classes_) < 2:
            self.max_neighbors_ = 1 if self.max_neighbors is None else int(self.max_neighbors)
            self.scores_by_k_ = np.zeros((self.n_features_in_, self.max_neighbors_), dtype=np.float64)
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
            self.best_k_ = np.ones(self.n_features_in_, dtype=np.int64)
            self.top_features_ = np.arange(n_select)
            return self

        class_counts = np.bincount(y_encoded)
        smallest = int(class_counts.min())
        if self.max_neighbors is None:
            k_max = max(1, smallest - 1)
        else:
            k_max = int(self.max_neighbors)
            if k_max >= smallest:
                warnings.warn(
                    f"max_neighbors ({k_max}) is greater than or equal to the smallest class size ({smallest}).",
                    UserWarning,
                )
        self.max_neighbors_ = k_max
        class_probs = class_counts / n_samples
        n_classes = len(self.classes_)

        columns, n_cont = split_discrete_last(self.is_discrete_)
        X_prepared = prepare_relief_matrix(
            X, self.is_discrete_, columns, dtype=np.float64, discrete_limit=self.discrete_limit
        )
        if self.verbose:
            print(f"Running ReliefSeq (k = 1..{k_max}) on the CPU now...")
        y_enc = y_encoded.astype(np.int64)
        neighbours = np.empty((n_samples, n_classes, k_max), dtype=np.int32)
        found = np.empty((n_samples, n_classes), dtype=np.int64)
        scores = np.zeros((k_max, X_prepared.shape[1]), dtype=np.float64)
        original_num_threads = get_num_threads()
        set_num_threads(resolve_num_threads(self.n_jobs))
        try:
            _rank_neighbours_cpu(X_prepared, y_enc, n_cont, n_classes, k_max, neighbours, found)
            _scores_by_k_cpu(np.ascontiguousarray(X_prepared.T), y_enc, n_cont, class_probs, neighbours, found, scores)
        finally:
            set_num_threads(original_num_threads)

        self.scores_by_k_ = np.zeros((self.n_features_in_, k_max), dtype=np.float64)
        self.scores_by_k_[columns] = scores.T
        best = np.argmax(self.scores_by_k_, axis=1)
        self.best_k_ = best.astype(np.int64) + 1
        self.feature_importances_ = self.scores_by_k_[np.arange(self.n_features_in_), best]
        self.top_features_ = np.argsort(self.feature_importances_)[::-1][:n_select]
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
