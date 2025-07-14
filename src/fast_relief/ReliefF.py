from __future__ import annotations

import numpy as np
from numba import cuda, float32, int32, njit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from fast_relief.MultiSURF import _compute_ranges

TPB = 64


@cuda.jit
def _relieff_gpu_kernel(x, y, recip_full, k_neighbors, scores_out):
    """
    ReliefF scoring kernel for GPU.
    Each block processes one sample.
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # --- Step 1: Find k-nearest hits and misses ---
    # This part is simplified for the GPU. Using a simple selection process
    # within each block. More advanced parallel sorting would be faster but
    # much more complex.

    # Each thread will find its own candidate neighbors
    # Note: For simplicity, this kernel finds k-nearest misses from ALL other
    # classes combined, which is a common ReliefF variant.

    local_hit_dists = cuda.local.array(shape=10, dtype=float32)
    local_hit_idxs = cuda.local.array(shape=10, dtype=int32)
    local_miss_dists = cuda.local.array(shape=10, dtype=float32)
    local_miss_idxs = cuda.local.array(shape=10, dtype=int32)

    for ki in range(k_neighbors):
        local_hit_dists[ki] = 3.4e38
        local_miss_dists[ki] = 3.4e38

    for j in range(tid, n_samples, TPB):
        if i == j:
            continue

        dist = 0.0
        for f in range(n_features):
            diff = x[i, f] - x[j, f]
            dist += abs(diff) * recip_full[f]

        if y[i] == y[j]:
            for ki in range(k_neighbors - 1, -1, -1):
                if dist < local_hit_dists[ki]:
                    if ki < k_neighbors - 1:
                        local_hit_dists[ki + 1] = local_hit_dists[ki]
                        local_hit_idxs[ki + 1] = local_hit_idxs[ki]
                    local_hit_dists[ki] = dist
                    local_hit_idxs[ki] = j
                else:
                    break
        else:
            for ki in range(k_neighbors - 1, -1, -1):
                if dist < local_miss_dists[ki]:
                    if ki < k_neighbors - 1:
                        local_miss_dists[ki + 1] = local_miss_dists[ki]
                        local_miss_idxs[ki + 1] = local_miss_idxs[ki]
                    local_miss_dists[ki] = dist
                    local_miss_idxs[ki] = j
                else:
                    break

    cuda.syncthreads()

    if tid == 0:
        for f in range(n_features):
            hit_term = 0.0
            miss_term = 0.0

            for ki in range(k_neighbors):
                hit_idx = local_hit_idxs[ki]
                hit_term += abs(x[i, f] - x[hit_idx, f]) * recip_full[f]

            for ki in range(k_neighbors):
                miss_idx = local_miss_idxs[ki]
                miss_term += abs(x[i, f] - x[miss_idx, f]) * recip_full[f]

            update = (miss_term / k_neighbors) - (hit_term / k_neighbors)
            cuda.atomic.add(scores_out, f, update)


def _relieff_gpu_host_caller(x_d, y_d, recip_full_d, k):
    """Host helper function that launches the ReliefF kernel."""
    n_samples, n_features = x_d.shape
    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0

    _relieff_gpu_kernel[n_samples, 64](x_d, y_d, recip_full_d, k, scores_d)
    cuda.synchronize()

    return scores_d.copy_to_host() / n_samples


@njit(parallel=True, fastmath=True)
def _relieff_cpu_kernel(x, y, recip_full, k_neighbors, scores_out):
    """
    Optimized ReliefF scoring for CPU, parallelized over samples.
    """
    n_samples, n_features = x.shape

    temp_scores = np.zeros((n_samples, n_features), dtype=np.float32)

    for i in prange(n_samples):

        distances = np.empty(n_samples, dtype=np.float32)
        for j in range(n_samples):
            if i == j:
                distances[j] = np.inf
                continue
            dist = 0.0
            for f in range(n_features):
                diff = x[i, f] - x[j, f]
                dist += abs(diff) * recip_full[f]
            distances[j] = dist

        sorted_indices = np.argsort(distances)

        hits = np.empty(k_neighbors, dtype=np.int32)
        misses = np.empty(k_neighbors, dtype=np.int32)
        n_hits_found = 0
        n_miss_found = 0

        for j_idx in sorted_indices:
            if n_hits_found < k_neighbors and y[j_idx] == y[i]:
                hits[n_hits_found] = j_idx
                n_hits_found += 1
            elif n_miss_found < k_neighbors and y[j_idx] != y[i]:
                misses[n_miss_found] = j_idx
                n_miss_found += 1

            if n_hits_found == k_neighbors and n_miss_found == k_neighbors:
                break

        for f in range(n_features):
            hit_term = 0.0
            for ki in range(k_neighbors):
                hit_term += abs(x[i, f] - x[hits[ki], f]) * recip_full[f]

            miss_term = 0.0
            for ki in range(k_neighbors):
                miss_term += abs(x[i, f] - x[misses[ki], f]) * recip_full[f]

            temp_scores[i, f] = miss_term / k_neighbors
            -(hit_term / k_neighbors)

    # --- Step 3: Reduce scores from all threads ---
    for f in range(n_features):
        scores_out[f] = temp_scores[:, f].sum()


def _relieff_cpu_host_caller(x, y, recip_full, k):
    """Host caller for the optimized ReliefF CPU kernel."""
    _, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)
    _relieff_cpu_kernel(x, y, recip_full, k, scores)
    return scores / x.shape[0]


class ReliefF(BaseEstimator):
    """GPU and CPU-accelerated feature selection using the ReliefF algorithm.

    This estimator provides a unified API for running ReliefF on either
    a CPU (using Numba's parallel JIT) or a GPU (using Numba CUDA).

    Parameters
    ----------
    n_features_to_select : int, default=10
        The number of top features to select.

    k_neighbors : int, default=10
        The number of nearest neighbors to use for score calculation.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend to use.
    """

    def __init__(
        self,
        n_features_to_select: int = 10,
        k_neighbors: int = 10,
        backend: str = "auto",
    ):
        self.n_features_to_select = n_features_to_select
        self.k_neighbors = k_neighbors
        self.backend = backend

        if self.backend not in ["auto", "gpu", "cpu"]:
            raise ValueError("backend must be one of 'auto', 'gpu', or 'cpu'")

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Calculates feature importances using the ReliefF algorithm."""
        x, y = check_X_y(x, y, dtype=np.float32, ensure_2d=True)
        self.n_features_in_ = x.shape[1]

        # Determine backend
        if self.backend == "auto":
            self.effective_backend_ = "gpu" if cuda.is_available() else "cpu"
        elif self.backend == "gpu":
            if not cuda.is_available():
                raise RuntimeError(
                    "backend='gpu' selected, but no compatible GPU found."
                )
            self.effective_backend_ = "gpu"
        else:
            self.effective_backend_ = "cpu"

        # Compute feature ranges for normalization
        feature_ranges = _compute_ranges(x)
        feature_ranges[feature_ranges == 0] = 1
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        if self.effective_backend_ == "gpu":
            x_d = cuda.to_device(x)
            y_d = cuda.to_device(y.astype(np.int32))
            recip_full_d = cuda.to_device(recip_full)

            scores = _relieff_gpu_host_caller(x_d, y_d, recip_full_d, self.k_neighbors)
        else:  # CPU backend
            scores = _relieff_cpu_host_caller(x, y, recip_full, self.k_neighbors)

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][: self.n_features_to_select]

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Reduces x to the selected features."""
        check_is_fitted(self)
        x = check_array(x, dtype=np.float32)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x has {x.shape[1]} features, but "
                + "was trained with {self.n_features_in_}."
            )
        return x[:, self.top_features_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(x, y)
        return self.transform(x)
