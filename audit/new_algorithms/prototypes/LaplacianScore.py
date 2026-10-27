"""He, Cai & Niyogi (2005) Laplacian Score on a symmetric nonnegative graph.

Binary/heat-kernel Euclidean kNN affinity, including self edges, follows the
explicit scikit-feature construct_W configuration used in comparisons.
Scoring uses the equivalent sparse edge energy, not dense n-by-n products.
Undefined zero weighted-variance features score +inf and rank last.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, validate_data

from fast_select.utils import check_integer_param, resolve_compute_backend, resolve_num_threads


def _affinity(X, neighbors, mode, bandwidth):
    n = len(X)
    distances = pairwise_distances(X, metric="euclidean")
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


@lru_cache(maxsize=1)
def _energy_kernel():
    from fast_select._metal import get_mx

    return get_mx().fast.metal_kernel(
        name="laplacian_sparse_split_energy",
        input_names=["high", "low", "indptr", "indices", "wh", "wl"],
        output_names=["partial"],
        source=r"""
        uint task = thread_position_in_grid.x;
        uint n = high_shape[0], p = high_shape[1];
        if (task >= BLOCKS*p) return;
        uint f = task % p, block = task / p;
        float sum = 0.0f, residual = 0.0f;
        for (uint i = block*32; i < min(n, (block+1)*32); ++i) {
            float a = high[i*p+f], al = low[i*p+f];
            for (uint e = indptr[i]; e < indptr[i+1]; ++e) {
                uint j = indices[e];
                float b = high[j*p+f], bl = low[j*p+f];
                float d = a-b, v = d-a;
                float dl = (a-(d-v))-(b+v) + al-bl;
                float square = d*d;
                float sl = fma(d,d,-square) + 2.0f*d*dl + dl*dl;
                float term = square*wh[e];
                float tl = fma(square,wh[e],-term) + square*wl[e] + sl*wh[e] + sl*wl[e];
                float next = sum+term, z = next-sum;
                residual += (sum-(next-z))+(term-z)+tl;
                float combined = next+residual;
                residual -= combined-next;
                sum = combined;
            }
        }
        partial[task*2] = sum;
        partial[task*2+1] = residual;
        """,
        compile_options={"math_mode": "safe"},
    )


def _metal_energy(columns, graph):
    from fast_select._metal import get_mx

    mx = get_mx()
    X = np.ascontiguousarray(columns.T)
    high = X.astype(np.float32)
    low = (X - high).astype(np.float32)
    wh = graph.data.astype(np.float32)
    wl = (graph.data - wh).astype(np.float32)
    blocks = (len(X) + 31) // 32
    with mx.stream(mx.gpu):
        parts = _energy_kernel()(
            inputs=[
                mx.array(high),
                mx.array(low),
                mx.array(graph.indptr.astype(np.uint32)),
                mx.array(graph.indices.astype(np.uint32)),
                mx.array(wh),
                mx.array(wl),
            ],
            template=[("BLOCKS", blocks)],
            grid=(blocks * X.shape[1], 1, 1),
            threadgroup=(min(256, blocks * X.shape[1]), 1, 1),
            output_shapes=[(blocks, X.shape[1], 2)],
            output_dtypes=[mx.float32],
        )[0]
        values = np.array(parts).astype(np.float64)
    return np.sum(values[..., 0] + values[..., 1], axis=0) / 2


class LaplacianScore(TransformerMixin, BaseEstimator):
    """Laplacian Score (He et al., 2005), lower score means more local structure.

    Defaults to an undirected binary Euclidean 5-neighbor graph with self edges.
    ``fit(..., W=...)`` accepts a precomputed symmetric nonnegative affinity.
    ``weight_mode='heat_kernel'`` uses exp(-distance**2/(2*bandwidth**2)).
    The raw scores are ``laplacian_scores_``; ``feature_importances_`` negates
    them so larger means more important, as in the other FastSelect selectors.
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
        supported = X.size * 8 <= 128 * 1024**2 and graph.nnz < 2**32
        self.effective_backend_ = resolve_compute_backend(self.backend, cuda_available=False, metal_supported=supported)
        previous = get_num_threads()
        try:
            set_num_threads(threads)
            columns, _, denominator = _columns_and_statistics(X, degree)
            features = np.arange(X.shape[1])
            if self.effective_backend_ == "metal":
                numerator = _metal_energy(columns, graph)
                scores = np.divide(numerator, denominator, out=np.full(len(features), np.inf), where=denominator > 0)
                order = np.argsort(scores, kind="stable")
                close = np.zeros(len(features), dtype=bool)
                for a, b in zip(order[:-1], order[1:]):
                    if np.isfinite(scores[a]) and np.isfinite(scores[b]) and abs(scores[a] - scores[b]) <= 1e-10:
                        close[a] = close[b] = True
                close |= (denominator > 0) & (denominator < 1e-12)
                if np.any(close):
                    refined = np.flatnonzero(close)
                    numerator[refined] = _edge_energy(columns, graph.indptr, graph.indices, graph.data, refined)
                self.metal_diagnostics_ = {"refined_features": int(close.sum())}
            else:
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
