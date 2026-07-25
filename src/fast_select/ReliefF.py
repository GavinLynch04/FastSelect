from __future__ import annotations

import warnings

import numpy as np
from numba import config, cuda, float32, get_num_threads, get_thread_id, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .utils import (
    build_kernel_matrix,
    discrete_feature_mask,
    ensure_cuda_context,
    is_cuda_ready,
    split_discrete_last,
)

TPB = 64  # Threads-per-block


@cuda.jit
def _compute_dist_matrix_gpu_kernel(x, recip_full, is_discrete, dist_matrix):  # pragma: no cover
    """
    Computes each pairwise sample distance once and mirrors the result.
    Grid: (n_samples,), Threads per block: TPB
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    if tid == 0:
        dist_matrix[i, i] = 0.0

    for j in range(i + 1, n_samples):
        local_dist = 0.0
        for f in range(tid, n_features, TPB):
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            local_dist += diff

        sh_sum = cuda.shared.array(shape=64, dtype=float32)
        sh_sum[tid] = local_dist
        cuda.syncthreads()

        off = TPB // 2
        while off > 0:
            if tid < off:
                sh_sum[tid] += sh_sum[tid + off]
            cuda.syncthreads()
            off //= 2

        if tid == 0:
            dist_matrix[i, j] = sh_sum[0]
            dist_matrix[j, i] = sh_sum[0]
        cuda.syncthreads()


@cuda.jit
def _accumulate_weighted_diffs_gpu_kernel(x, weights_matrix, recip_full, is_discrete, scores_out):  # pragma: no cover
    """
    Accumulates feature difference scores weighted by neighbor relationships on GPU.
    Grid: (n_samples,), Threads per block: TPB
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    for j in range(n_samples):
        w = weights_matrix[i, j]
        if w == 0.0:
            continue

        for f in range(tid, n_features, TPB):
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            cuda.atomic.add(scores_out, f, w * diff)


@cuda.jit
def _select_relieff_neighbors_gpu_kernel(dist_matrix, y, k, neighbor_distances, neighbor_indices):  # pragma: no cover
    """Select the nearest hits/misses for one sample per GPU thread."""
    i = cuda.grid(1)
    n_samples = dist_matrix.shape[0]
    if i >= n_samples:
        return

    n_classes = neighbor_indices.shape[1]
    for c in range(n_classes):
        for neighbor in range(k):
            neighbor_distances[i, c, neighbor] = np.inf
            neighbor_indices[i, c, neighbor] = -1

    for j in range(n_samples):
        if i == j:
            continue
        label = y[j]
        distance = dist_matrix[i, j]
        if distance < neighbor_distances[i, label, k - 1]:
            pos = k - 1
            while pos > 0 and distance < neighbor_distances[i, label, pos - 1]:
                neighbor_distances[i, label, pos] = neighbor_distances[i, label, pos - 1]
                neighbor_indices[i, label, pos] = neighbor_indices[i, label, pos - 1]
                pos -= 1
            neighbor_distances[i, label, pos] = distance
            neighbor_indices[i, label, pos] = j


@cuda.jit
def _score_relieff_neighbors_gpu_kernel(
    x,
    y,
    neighbor_indices,
    class_probs,
    recip_full,
    is_discrete,
    scores_out,
):  # pragma: no cover
    """Score only selected ReliefF neighbors, avoiding a dense weight matrix."""
    n_samples, n_features = x.shape
    n_classes = neighbor_indices.shape[1]
    k = neighbor_indices.shape[2]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    label_i = y[i]
    denom = 1.0 - class_probs[label_i]
    if denom <= 0.0:
        denom = 1.0

    # Feature-major accumulation: each thread owns a feature and sums that
    # feature's contribution over every selected neighbour in a register, so the
    # block issues one atomic per feature instead of one per (neighbour, feature).
    for f in range(tid, n_features, TPB):
        discrete_f = is_discrete[f]
        x_if = x[i, f]
        recip_f = recip_full[f]
        acc = 0.0

        for c in range(n_classes):
            count = 0
            for neighbor in range(k):
                if neighbor_indices[i, c, neighbor] >= 0:
                    count += 1
            if count == 0:
                continue

            if c == label_i:
                weight = -1.0 / (count * n_samples)
            else:
                weight = class_probs[c] / denom / (count * n_samples)

            for neighbor in range(count):
                j = neighbor_indices[i, c, neighbor]
                if discrete_f:
                    diff = 1.0 if x_if != x[j, f] else 0.0
                else:
                    diff = abs(x_if - x[j, f]) * recip_f
                acc += weight * diff

        if acc != 0.0:
            cuda.atomic.add(scores_out, f, acc)


@njit(parallel=True)
def _compute_relieff_weights(dist_matrix, y_enc, class_probs, k):  # pragma: no cover
    """
    Computes ReliefF neighbor weight matrix W of shape (n_samples, n_samples)
    using bounded insertion buffers instead of sorting a full row per class.
    """
    n_samples = dist_matrix.shape[0]
    n_classes = len(class_probs)
    weights = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in prange(n_samples):
        lbl_i = y_enc[i]
        denom = 1.0 - class_probs[lbl_i]
        if denom <= 0:
            denom = 1.0

        hit_dists = np.full(k, np.inf, dtype=np.float32)
        hit_indices = np.full(k, -1, dtype=np.int32)
        miss_dists = np.full((n_classes, k), np.inf, dtype=np.float32)
        miss_indices = np.full((n_classes, k), -1, dtype=np.int32)

        for j in range(n_samples):
            if i == j:
                continue
            distance = dist_matrix[i, j]
            label = y_enc[j]
            if label == lbl_i:
                if distance < hit_dists[k - 1]:
                    pos = k - 1
                    while pos > 0 and distance < hit_dists[pos - 1]:
                        hit_dists[pos] = hit_dists[pos - 1]
                        hit_indices[pos] = hit_indices[pos - 1]
                        pos -= 1
                    hit_dists[pos] = distance
                    hit_indices[pos] = j
            elif distance < miss_dists[label, k - 1]:
                pos = k - 1
                while pos > 0 and distance < miss_dists[label, pos - 1]:
                    miss_dists[label, pos] = miss_dists[label, pos - 1]
                    miss_indices[label, pos] = miss_indices[label, pos - 1]
                    pos -= 1
                miss_dists[label, pos] = distance
                miss_indices[label, pos] = j

        hit_count = 0
        for neighbor in range(k):
            if hit_indices[neighbor] >= 0:
                hit_count += 1
        if hit_count > 0:
            hit_weight = -1.0 / (hit_count * n_samples)
            for neighbor in range(hit_count):
                weights[i, hit_indices[neighbor]] = hit_weight

        for c in range(n_classes):
            if c == lbl_i:
                continue
            miss_count = 0
            for neighbor in range(k):
                if miss_indices[c, neighbor] >= 0:
                    miss_count += 1
            if miss_count > 0:
                miss_weight = class_probs[c] / denom / (miss_count * n_samples)
                for neighbor in range(miss_count):
                    weights[i, miss_indices[c, neighbor]] = miss_weight

    return weights


def _relieff_gpu_host_caller(x_d, y_enc, recip_full_d, is_discrete_d, class_probs, k):
    """Launch ReliefF distance, neighbor-selection, and scoring stages."""
    ensure_cuda_context()
    n_samples, n_features = x_d.shape
    dist_matrix_d = cuda.device_array((n_samples, n_samples), dtype=np.float32)

    _compute_dist_matrix_gpu_kernel[n_samples, TPB](x_d, recip_full_d, is_discrete_d, dist_matrix_d)

    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0

    n_classes = class_probs.shape[0]
    if n_classes * k <= n_samples:
        y_d = cuda.to_device(y_enc)
        class_probs_d = cuda.to_device(class_probs)
        neighbor_distances_d = cuda.device_array((n_samples, n_classes, k), dtype=np.float32)
        neighbor_indices_d = cuda.device_array((n_samples, n_classes, k), dtype=np.int32)
        selection_threads = 128
        selection_blocks = (n_samples + selection_threads - 1) // selection_threads
        _select_relieff_neighbors_gpu_kernel[selection_blocks, selection_threads](
            dist_matrix_d,
            y_d,
            k,
            neighbor_distances_d,
            neighbor_indices_d,
        )
        _score_relieff_neighbors_gpu_kernel[n_samples, TPB](
            x_d,
            y_d,
            neighbor_indices_d,
            class_probs_d,
            recip_full_d,
            is_discrete_d,
            scores_d,
        )
    else:
        # Preserve bounded memory for unusually large k/class combinations.
        dist_matrix = dist_matrix_d.copy_to_host()
        weights_matrix = _compute_relieff_weights(dist_matrix, y_enc, class_probs, k)
        weights_d = cuda.to_device(weights_matrix)
        _accumulate_weighted_diffs_gpu_kernel[n_samples, TPB](x_d, weights_d, recip_full_d, is_discrete_d, scores_d)

    return scores_d.copy_to_host()


@njit(parallel=True, fastmath=True)
def _relieff_cpu_kernel(x, y_enc, n_cont, k, class_probs, scores_out):  # pragma: no cover
    """ReliefF CPU scoring.

    ``x`` arrives with continuous features in columns ``[0, n_cont)`` and
    discrete features after them, so both inner loops are branch-free and
    vectorisable.  The distance is still the sum of the same per-feature terms.
    """
    n_samples, n_features = x.shape
    n_classes = class_probs.shape[0]
    n_threads = get_num_threads()

    thread_scores = np.zeros((n_threads, n_features), dtype=np.float32)

    for i in prange(n_samples):
        tid = get_thread_id()
        lbl_i = y_enc[i]
        x_i = x[i]

        hit_d = np.full(k, np.inf, dtype=np.float32)
        hit_idx = np.full(k, -1, dtype=np.int32)

        miss_d = np.full((n_classes, k), np.inf, dtype=np.float32)
        miss_idx = np.full((n_classes, k), -1, dtype=np.int32)

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

            lbl_j = y_enc[j]
            if lbl_j == lbl_i:
                if d < hit_d[k - 1]:
                    pos = k - 1
                    while pos > 0 and d < hit_d[pos - 1]:
                        hit_d[pos] = hit_d[pos - 1]
                        hit_idx[pos] = hit_idx[pos - 1]
                        pos -= 1
                    hit_d[pos] = d
                    hit_idx[pos] = j
            else:
                if d < miss_d[lbl_j, k - 1]:
                    pos = k - 1
                    while pos > 0 and d < miss_d[lbl_j, pos - 1]:
                        miss_d[lbl_j, pos] = miss_d[lbl_j, pos - 1]
                        miss_idx[lbl_j, pos] = miss_idx[lbl_j, pos - 1]
                        pos -= 1
                    miss_d[lbl_j, pos] = d
                    miss_idx[lbl_j, pos] = j

        h_found = 0
        for ki in range(k):
            if hit_idx[ki] != -1:
                h_found += 1
            else:
                break

        denom = 1.0 - class_probs[lbl_i]
        if denom <= 0:
            denom = 1.0

        scores_i = thread_scores[tid]
        if h_found > 0:
            scale_hit = -1.0 / (h_found * n_samples)
            for ki in range(h_found):
                x_h = x[hit_idx[ki]]
                for f in range(n_cont):
                    scores_i[f] += scale_hit * abs(x_i[f] - x_h[f])
                for f in range(n_cont, n_features):
                    if x_i[f] != x_h[f]:
                        scores_i[f] += scale_hit

        for c in range(n_classes):
            if c == lbl_i:
                continue
            m_found = 0
            for ki in range(k):
                if miss_idx[c, ki] != -1:
                    m_found += 1
                else:
                    break
            if m_found > 0:
                weight_c = class_probs[c] / denom
                scale_miss = weight_c / m_found / n_samples
                for ki in range(m_found):
                    x_m = x[miss_idx[c, ki]]
                    for f in range(n_cont):
                        scores_i[f] += scale_miss * abs(x_i[f] - x_m[f])
                    for f in range(n_cont, n_features):
                        if x_i[f] != x_m[f]:
                            scores_i[f] += scale_miss

    for f in range(n_features):
        tot = 0.0
        for t in range(n_threads):
            tot += thread_scores[t, f]
        scores_out[f] = tot


def _relieff_cpu_host_caller(x, y_enc, n_cont, k, class_probs, n_jobs):
    n_samples, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)

    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs

    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)

    try:
        _relieff_cpu_kernel(x, y_enc, n_cont, k, class_probs, scores)
    finally:
        set_num_threads(original_num_threads)

    return scores


class ReliefF(TransformerMixin, BaseEstimator):
    """GPU and CPU-accelerated feature selection using the ReliefF algorithm.

    This estimator provides a unified API for running ReliefF on either
    a CPU (using Numba's parallel JIT) or a GPU (using Numba CUDA).

    Parameters
    ----------
    n_features_to_select : int | float, default=0.2
        The number of top features to select. If variable is a float, that percent
        of features will be selected. If variable is an int, that number of features will be returned.

    discrete_limit : int, default=10
        The limit for the number of independent feature values to be considered discrete.

    n_neighbors : int, default=3
        The number of nearest neighbors to use for score calculation.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend to use.

    verbose : bool, default=False
        Controls whether progress updates are printed during the fit.

    n_jobs : int, default=-1
        Controls the number of threads utilized by Numba while running on CPU.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.

    feature_importances_ : ndarray of shape (n_features,)
        The calculated importance scores for each feature.

    effective_backend_ : str
        The backend that was actually used during `fit` ('gpu' or 'cpu').

    Notes
    -----
    This implementation follows the complete-data classification equations,
    including multiclass prior weighting. Missing-value ReliefF extensions are
    outside its supported scope; input containing NaNs is rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        discrete_limit: int = 10,
        n_neighbors: int = 3,
        backend: str = "auto",
        verbose: bool = False,
        n_jobs: int = -1,
    ):
        self.n_features_to_select = n_features_to_select
        self.discrete_limit = discrete_limit
        self.n_neighbors = n_neighbors
        self.backend = backend
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _validate_parameters(self, n_samples, n_features):
        """Validate all user-provided parameters."""
        if self.backend not in ["auto", "gpu", "cpu"]:
            raise ValueError("backend must be one of 'auto', 'gpu', or 'cpu'")

        if n_samples < 2:
            raise ValueError(f"ReliefF requires at least 2 samples, but got n_samples = {n_samples}")

        if not (0 < self.n_neighbors < n_samples):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be an integer "
                f"between 1 and n_samples - 1 ({n_samples - 1})."
            )

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            n_select = max(1, int(self.n_features_to_select * n_features))
        elif isinstance(self.n_features_to_select, int):
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
        """Calculates feature importances using the ReliefF algorithm."""
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

        n_select = self._validate_parameters(n_samples, self.n_features_in_)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float32)
            self.top_features_ = np.arange(n_select)
            self.effective_backend_ = "cpu" if self.backend != "gpu" else "gpu"
            return self

        min_class_size = np.min(np.bincount(y_encoded))
        if self.n_neighbors >= min_class_size:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than or equal to the "
                f"smallest class size ({min_class_size}).",
                UserWarning,
            )

        is_discrete = discrete_feature_mask(x, self.discrete_limit)
        self.is_discrete_ = is_discrete

        class_labels, class_counts = np.unique(y, return_counts=True)
        class_probs = (class_counts / len(y)).astype(np.float32)
        y_enc = np.searchsorted(class_labels, y).astype(np.int32)

        feature_ranges = x.max(axis=0) - x.min(axis=0)
        feature_ranges[is_discrete] = 1.0
        feature_ranges[feature_ranges == 0] = 1.0
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        if self.backend == "auto":
            self.effective_backend_ = "gpu" if is_cuda_ready() else "cpu"
        elif self.backend == "gpu" and not is_cuda_ready():
            raise RuntimeError("backend='gpu', but no CUDA-enabled GPU is available.")
        else:
            self.effective_backend_ = self.backend

        if self.effective_backend_ == "gpu":
            x_d = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
            recip_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(is_discrete.astype(np.bool_))
            if self.verbose:
                print("Running ReliefF on the GPU now...")
            scores = _relieff_gpu_host_caller(x_d, y_enc, recip_d, is_discrete_d, class_probs, self.n_neighbors)
        else:
            if self.verbose:
                print("Running ReliefF on the CPU now...")
            columns, n_cont = split_discrete_last(is_discrete)
            x_cpu = build_kernel_matrix(x, columns, recip_full, n_cont)
            permuted_scores = _relieff_cpu_host_caller(x_cpu, y_enc, n_cont, self.n_neighbors, class_probs, self.n_jobs)
            scores = np.zeros(self.n_features_in_, dtype=np.float32)
            scores[columns] = permuted_scores

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Reduces x to the selected features."""
        check_is_fitted(self)

        x = validate_data(self, x, reset=False, dtype=[np.float64, np.float32])

        return x[:, self.top_features_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(x, y)
        return self.transform(x)
