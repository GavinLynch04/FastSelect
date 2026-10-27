"""He, Cai & Niyogi (2005) Laplacian Score on a symmetric nonnegative graph.

Binary/heat-kernel Euclidean kNN affinity, including self edges, follows the
explicit scikit-feature construct_W configuration used in comparisons.
Scoring uses the equivalent sparse edge energy, not dense n-by-n products.
Undefined zero weighted-variance features score +inf and rank last.
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, validate_data

from fast_select._cpu_backend import resolve_cpu_backend
from fast_select.utils import check_integer_param, resolve_num_threads


def _affinity(X, neighbors, mode, bandwidth):
    n = len(X)
    # Center columns with large common offsets before the BLAS distance
    # identity. Translation preserves Euclidean distances and avoids losing
    # small separations through x*x + y*y - 2*x*y cancellation.
    with np.errstate(over="ignore"):
        span = np.ptp(X, axis=0)
        offset = np.max(np.abs(X), axis=0)
        shifted = offset > 1e4 * span
    coordinates = X
    if np.any(shifted):
        coordinates = X.copy()
        coordinates[:, shifted] -= X[0, shifted]
    distances = pairwise_distances(coordinates, metric="euclidean")
    distances *= distances
    k = min(neighbors + 1, n)
    if k == n:
        indices = np.tile(np.arange(n), (n, 1))
    else:
        indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
        chosen = np.take_along_axis(distances, indices, axis=1)
        cutoff = chosen.max(axis=1)
        # Partial selection avoids sorting all n distances. If a boundary tie
        # spans selected/unselected neighbors, recover the reference argsort
        # membership exactly rather than imposing a different tie rule.
        ambiguous = np.count_nonzero(distances == cutoff[:, None], axis=1) > np.count_nonzero(
            chosen == cutoff[:, None], axis=1
        )
        for row in np.flatnonzero(ambiguous):
            indices[row] = np.argsort(distances[row])[:k]
    if mode == "binary":
        data = np.ones(indices.size)
    else:
        selected = np.take_along_axis(distances, indices, axis=1)
        data = np.exp(-selected / (2 * bandwidth * bandwidth)).ravel()
    graph = coo_matrix((data, (np.repeat(np.arange(n), k), indices.ravel())), shape=(n, n)).tocsr()
    return graph.maximum(graph.T).tocsr()


@njit(parallel=True, cache=True, nogil=True)
def _columns_and_statistics(X, degree):
    n, p = X.shape
    columns = np.empty((p, n), dtype=np.float64)
    means = np.zeros(p)
    denominator = np.zeros(p)
    total_degree = np.sum(degree)
    for f in prange(p):
        minimum, maximum = X[0, f], X[0, f]
        for i in range(n):
            minimum = min(minimum, X[i, f])
            maximum = max(maximum, X[i, f])
        span = maximum - minimum
        for i in range(n):
            if span == 0:
                columns[f, i] = 0
            elif np.isfinite(span):
                columns[f, i] = (X[i, f] - minimum) / span
            else:
                scale = max(abs(minimum), abs(maximum))
                columns[f, i] = (X[i, f] / scale - minimum / scale) / (maximum / scale - minimum / scale)
        mean = 0.0
        if total_degree > 0:
            for i in range(n):
                mean += degree[i] * columns[f, i]
            mean /= total_degree
        means[f] = mean
        for i in range(n):
            denominator[f] += degree[i] * (columns[f, i] - mean) ** 2
    return columns, means, denominator


@njit(parallel=True, cache=True, nogil=True)
def _edge_energy(columns, indptr, indices, data, features):
    result = np.empty(len(features))
    for t in prange(len(features)):
        f = features[t]
        total = 0.0
        for i in range(columns.shape[1]):
            value = columns[f, i]
            for edge in range(indptr[i], indptr[i + 1]):
                difference = value - columns[f, indices[edge]]
                total += data[edge] * difference * difference
        result[t] = total / 2
    return result


class LaplacianScore(TransformerMixin, BaseEstimator):
    """Laplacian Score (He et al., 2005), lower score means more local structure.

    Defaults to an undirected binary Euclidean 5-neighbor graph with self edges.
    ``fit(..., W=...)`` accepts a precomputed symmetric nonnegative affinity.
    ``weight_mode='heat_kernel'`` uses exp(-distance**2/(2*bandwidth**2)).
    The raw scores are ``laplacian_scores_``; ``feature_importances_`` negates
    them so larger means more important, as in the other FastSelect selectors.
    Only parallel Numba CPU execution is admitted: ``backend='auto'`` selects
    CPU. Graph construction uses quadratic memory in the sample count.
    """

    def __init__(
        self, n_features_to_select=0.2, n_neighbors=5, weight_mode="binary", bandwidth=1.0, backend="auto", n_jobs=-1
    ):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.weight_mode = weight_mode
        self.bandwidth = bandwidth
        self.backend = backend
        self.n_jobs = n_jobs

    def fit(self, X, y=None, *, W=None):
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        check_integer_param(self.n_neighbors, "n_neighbors", 1)
        if self.weight_mode not in ("binary", "heat_kernel"):
            raise ValueError("weight_mode must be 'binary' or 'heat_kernel'")
        if not np.isscalar(self.bandwidth) or not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive and finite")
        value = self.n_features_to_select
        if isinstance(value, (float, np.floating)):
            if not 0 < value <= 1:
                raise ValueError("fractional n_features_to_select must be in (0, 1]")
            count = max(1, int(value * X.shape[1]))
        else:
            count = check_integer_param(value, "n_features_to_select", 1)
            if count > X.shape[1]:
                raise ValueError("n_features_to_select must not exceed the feature count")
        threads = resolve_num_threads(self.n_jobs)
        graph = (
            _affinity(X, self.n_neighbors, self.weight_mode, self.bandwidth)
            if W is None
            else csr_matrix(W, dtype=float)
        )
        graph = graph.copy().tocsr()
        if graph.shape != (len(X), len(X)) or not np.isfinite(graph.data).all() or np.any(graph.data < 0):
            raise ValueError("W must be a finite nonnegative square affinity matching the sample count")
        difference = graph - graph.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
            raise ValueError("W must be symmetric")
        if graph.nnz and graph.data.max() > 0:
            graph.data /= graph.data.max()
        degree = np.asarray(graph.sum(axis=1)).ravel()
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        previous = get_num_threads()
        try:
            set_num_threads(threads)
            columns, _, denominator = _columns_and_statistics(X, degree)
            features = np.arange(X.shape[1])
            numerator = _edge_energy(columns, graph.indptr, graph.indices, graph.data, features)
            self.laplacian_scores_ = np.divide(
                numerator, denominator, out=np.full(len(features), np.inf), where=denominator > 0
            )
            self.feature_importances_ = -self.laplacian_scores_
            self.top_features_ = np.argsort(self.laplacian_scores_, kind="stable")[:count]
            self.affinity_ = graph
        finally:
            set_num_threads(previous)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X[:, self.top_features_]
