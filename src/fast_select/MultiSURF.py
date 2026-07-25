from __future__ import annotations

import math

import numpy as np
from numba import config, cuda, float32, get_num_threads, get_thread_id, int32, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .utils import (
    build_kernel_matrix,
    discrete_feature_mask,
    ensure_cuda_context,
    is_cuda_ready,
    split_discrete_last,
)

TPB = 64  # Threads Per Block

# Above this many kept features the shared-memory score accumulator stops paying
# for itself and the global-atomic kernel is faster; measured on the benchmark in
# benchmarking/compare_head_vs_current.py.
SHARED_SCORE_MAX_FEATURES = 512


@cuda.jit
def _compute_dist_matrix_multisurf_kernel(x, recip_full, feat_idx, is_discrete, dist_matrix):  # pragma: no cover
    """Computes each pair distance once and mirrors it into the distance matrix."""
    n_samples = x.shape[0]
    n_kept = feat_idx.shape[0]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    if tid == 0:
        dist_matrix[i, i] = 0.0

    for j in range(i + 1, n_samples):
        local_dist = 0.0
        for k in range(tid, n_kept, TPB):
            f = feat_idx[k]
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
def _score_multisurf_gpu_kernel(
    x, y, dist_matrix, recip_full, feat_idx, use_star, is_discrete, scores_out
):  # pragma: no cover
    """Computes MultiSURF thresholds and scores entirely on the GPU."""
    n_samples = x.shape[0]
    n_kept = feat_idx.shape[0]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    local_sum = 0.0
    local_sum2 = 0.0
    for j in range(tid, n_samples, TPB):
        if i != j:
            dist = dist_matrix[i, j]
            local_sum += dist
            local_sum2 += dist * dist

    sh_sum = cuda.shared.array(shape=64, dtype=float32)
    sh_sum2 = cuda.shared.array(shape=64, dtype=float32)
    sh_sum[tid] = local_sum
    sh_sum2[tid] = local_sum2
    cuda.syncthreads()

    off = TPB // 2
    while off > 0:
        if tid < off:
            sh_sum[tid] += sh_sum[tid + off]
            sh_sum2[tid] += sh_sum2[tid + off]
        cuda.syncthreads()
        off //= 2

    mu = sh_sum[0] / (n_samples - 1)
    variance = sh_sum2[0] / (n_samples - 1) - mu * mu
    if variance < 0.0:
        variance = 0.0
    half_sigma = 0.5 * math.sqrt(variance)
    near_threshold = mu - half_sigma
    far_threshold = mu + half_sigma

    local_near_hits = 0
    local_near_misses = 0
    local_far_hits = 0
    local_far_misses = 0
    for j in range(tid, n_samples, TPB):
        if i == j:
            continue
        dist = dist_matrix[i, j]
        is_hit = y[i] == y[j]
        if dist < near_threshold:
            if is_hit:
                local_near_hits += 1
            else:
                local_near_misses += 1
        elif use_star and dist > far_threshold:
            if is_hit:
                local_far_hits += 1
            else:
                local_far_misses += 1

    sh_near_hits = cuda.shared.array(shape=64, dtype=int32)
    sh_near_misses = cuda.shared.array(shape=64, dtype=int32)
    sh_far_hits = cuda.shared.array(shape=64, dtype=int32)
    sh_far_misses = cuda.shared.array(shape=64, dtype=int32)
    sh_near_hits[tid] = local_near_hits
    sh_near_misses[tid] = local_near_misses
    sh_far_hits[tid] = local_far_hits
    sh_far_misses[tid] = local_far_misses
    cuda.syncthreads()

    off = TPB // 2
    while off > 0:
        if tid < off:
            sh_near_hits[tid] += sh_near_hits[tid + off]
            sh_near_misses[tid] += sh_near_misses[tid + off]
            sh_far_hits[tid] += sh_far_hits[tid + off]
            sh_far_misses[tid] += sh_far_misses[tid + off]
        cuda.syncthreads()
        off //= 2

    n_near_hits = sh_near_hits[0]
    n_near_misses = sh_near_misses[0]
    n_far_hits = sh_far_hits[0]
    n_far_misses = sh_far_misses[0]
    scale = 1.0 / n_samples

    for j in range(n_samples):
        if i == j:
            continue

        is_hit = y[i] == y[j]
        dist = dist_matrix[i, j]
        weight = 0.0
        use_similarity = False
        if dist < near_threshold:
            if is_hit and n_near_hits > 0:
                weight = -scale / n_near_hits
            elif not is_hit and n_near_misses > 0:
                weight = scale / n_near_misses
        elif use_star and dist > far_threshold:
            use_similarity = True
            if is_hit and n_far_hits > 0:
                weight = -scale / n_far_hits
            elif not is_hit and n_far_misses > 0:
                weight = scale / n_far_misses

        if weight == 0.0:
            continue
        for k in range(tid, n_kept, TPB):
            f = feat_idx[k]
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            contribution = (1.0 - diff) if use_similarity else diff
            cuda.atomic.add(scores_out, k, weight * contribution)


@cuda.jit
def _score_multisurf_shared_gpu_kernel(
    x, y, dist_matrix, recip_full, feat_idx, use_star, is_discrete, scores_out
):  # pragma: no cover
    """MultiSURF scoring accumulated in shared memory.

    Identical equations to :func:`_score_multisurf_gpu_kernel`; the only
    difference is that a block keeps its per-feature running totals in shared
    memory and issues one global atomic per feature at the end instead of one
    per (neighbour, feature).  Selected by the host only when the kept features
    fit in shared memory; see ``SHARED_SCORE_MAX_FEATURES``.
    """
    n_samples = x.shape[0]
    n_kept = feat_idx.shape[0]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    local_sum = 0.0
    local_sum2 = 0.0
    for j in range(tid, n_samples, TPB):
        if i != j:
            dist = dist_matrix[i, j]
            local_sum += dist
            local_sum2 += dist * dist

    sh_sum = cuda.shared.array(shape=64, dtype=float32)
    sh_sum2 = cuda.shared.array(shape=64, dtype=float32)
    sh_sum[tid] = local_sum
    sh_sum2[tid] = local_sum2
    cuda.syncthreads()

    off = TPB // 2
    while off > 0:
        if tid < off:
            sh_sum[tid] += sh_sum[tid + off]
            sh_sum2[tid] += sh_sum2[tid + off]
        cuda.syncthreads()
        off //= 2

    mu = sh_sum[0] / (n_samples - 1)
    variance = sh_sum2[0] / (n_samples - 1) - mu * mu
    if variance < 0.0:
        variance = 0.0
    half_sigma = 0.5 * math.sqrt(variance)
    near_threshold = mu - half_sigma
    far_threshold = mu + half_sigma

    local_near_hits = 0
    local_near_misses = 0
    local_far_hits = 0
    local_far_misses = 0
    for j in range(tid, n_samples, TPB):
        if i == j:
            continue
        dist = dist_matrix[i, j]
        is_hit = y[i] == y[j]
        if dist < near_threshold:
            if is_hit:
                local_near_hits += 1
            else:
                local_near_misses += 1
        elif use_star and dist > far_threshold:
            if is_hit:
                local_far_hits += 1
            else:
                local_far_misses += 1

    sh_near_hits = cuda.shared.array(shape=64, dtype=int32)
    sh_near_misses = cuda.shared.array(shape=64, dtype=int32)
    sh_far_hits = cuda.shared.array(shape=64, dtype=int32)
    sh_far_misses = cuda.shared.array(shape=64, dtype=int32)
    sh_near_hits[tid] = local_near_hits
    sh_near_misses[tid] = local_near_misses
    sh_far_hits[tid] = local_far_hits
    sh_far_misses[tid] = local_far_misses
    cuda.syncthreads()

    off = TPB // 2
    while off > 0:
        if tid < off:
            sh_near_hits[tid] += sh_near_hits[tid + off]
            sh_near_misses[tid] += sh_near_misses[tid + off]
            sh_far_hits[tid] += sh_far_hits[tid + off]
            sh_far_misses[tid] += sh_far_misses[tid + off]
        cuda.syncthreads()
        off //= 2

    n_near_hits = sh_near_hits[0]
    n_near_misses = sh_near_misses[0]
    n_far_hits = sh_far_hits[0]
    n_far_misses = sh_far_misses[0]
    scale = 1.0 / n_samples

    sh_scores = cuda.shared.array(shape=0, dtype=float32)
    for k in range(tid, n_kept, TPB):
        sh_scores[k] = 0.0
    cuda.syncthreads()

    for j in range(n_samples):
        if i == j:
            continue

        is_hit = y[i] == y[j]
        dist = dist_matrix[i, j]
        weight = 0.0
        use_similarity = False
        if dist < near_threshold:
            if is_hit and n_near_hits > 0:
                weight = -scale / n_near_hits
            elif not is_hit and n_near_misses > 0:
                weight = scale / n_near_misses
        elif use_star and dist > far_threshold:
            use_similarity = True
            if is_hit and n_far_hits > 0:
                weight = -scale / n_far_hits
            elif not is_hit and n_far_misses > 0:
                weight = scale / n_far_misses

        if weight == 0.0:
            continue
        for k in range(tid, n_kept, TPB):
            f = feat_idx[k]
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            contribution = (1.0 - diff) if use_similarity else diff
            sh_scores[k] += weight * contribution

    cuda.syncthreads()
    for k in range(tid, n_kept, TPB):
        if sh_scores[k] != 0.0:
            cuda.atomic.add(scores_out, k, sh_scores[k])


def _multisurf_gpu_host_caller(x_d, y, recip_full_d, feat_idx: np.ndarray, use_star: bool, is_discrete_d) -> np.ndarray:
    """Launch GPU-only distance and scoring stages for MultiSURF."""
    ensure_cuda_context()
    n_samples = x_d.shape[0]
    n_kept = feat_idx.size

    feat_idx_d = cuda.to_device(feat_idx.astype(np.int32))
    y_d = cuda.to_device(y)
    dist_matrix_d = cuda.device_array((n_samples, n_samples), dtype=np.float32)

    _compute_dist_matrix_multisurf_kernel[n_samples, TPB](x_d, recip_full_d, feat_idx_d, is_discrete_d, dist_matrix_d)

    scores_d = cuda.device_array(n_kept, dtype=np.float32)
    scores_d[:] = 0.0

    if n_kept <= SHARED_SCORE_MAX_FEATURES:
        _score_multisurf_shared_gpu_kernel[n_samples, TPB, 0, n_kept * 4](
            x_d, y_d, dist_matrix_d, recip_full_d, feat_idx_d, use_star, is_discrete_d, scores_d
        )
    else:
        _score_multisurf_gpu_kernel[n_samples, TPB](
            x_d, y_d, dist_matrix_d, recip_full_d, feat_idx_d, use_star, is_discrete_d, scores_d
        )

    return scores_d.copy_to_host()


@njit(parallel=True, fastmath=True)
def _multisurf_cpu_kernel(x, y, n_cont, use_star, scores_out):  # pragma: no cover
    """
    Optimized MultiSURF CPU kernel with zero N x P matrix allocations in parallel loop.

    ``x`` holds only the kept features, continuous ones in columns
    ``[0, n_cont)`` and discrete ones after them, so both inner loops are
    branch-free and vectorisable.
    """
    n_samples, n_kept = x.shape
    n_threads = get_num_threads()
    thread_scores = np.zeros((n_threads, n_kept), dtype=np.float32)
    thread_dists = np.empty((n_threads, n_samples), dtype=np.float32)

    for i in prange(n_samples):
        tid = get_thread_id()
        dists_from_i = thread_dists[tid]
        x_i = x[i]
        sum_d = 0.0
        sum_d2 = 0.0

        for j in range(n_samples):
            if i == j:
                dists_from_i[j] = 0.0
                continue

            x_j = x[j]
            dist = 0.0
            for k in range(n_cont):
                dist += abs(x_i[k] - x_j[k])
            for k in range(n_cont, n_kept):
                if x_i[k] != x_j[k]:
                    dist += 1.0

            dists_from_i[j] = dist
            sum_d += dist
            sum_d2 += dist * dist

        mu = sum_d / (n_samples - 1) if n_samples > 1 else 0.0
        var = max(0.0, (sum_d2 / (n_samples - 1)) - (mu * mu)) if n_samples > 1 else 0.0
        sigma = math.sqrt(var)
        near_thresh = mu - 0.5 * sigma
        far_thresh = mu + 0.5 * sigma

        n_near_hits = 0
        n_near_miss = 0
        n_far_hits = 0
        n_far_miss = 0

        for j in range(n_samples):
            if i == j:
                continue
            is_hit = y[i] == y[j]
            if dists_from_i[j] < near_thresh:
                if is_hit:
                    n_near_hits += 1
                else:
                    n_near_miss += 1
            elif use_star and dists_from_i[j] > far_thresh:
                if is_hit:
                    n_far_hits += 1
                else:
                    n_far_miss += 1

        scale = 1.0 / n_samples
        w_near_hit = -scale / n_near_hits if n_near_hits > 0 else 0.0
        w_near_miss = scale / n_near_miss if n_near_miss > 0 else 0.0
        w_far_hit = -scale / n_far_hits if n_far_hits > 0 else 0.0
        w_far_miss = scale / n_far_miss if n_far_miss > 0 else 0.0

        scores_i = thread_scores[tid]
        for j in range(n_samples):
            if i == j:
                continue
            is_hit = y[i] == y[j]
            weight = 0.0
            use_similarity = False
            if dists_from_i[j] < near_thresh:
                weight = w_near_hit if is_hit else w_near_miss
            elif use_star and dists_from_i[j] > far_thresh:
                weight = w_far_hit if is_hit else w_far_miss
                use_similarity = True

            if weight != 0.0:
                x_j = x[j]
                if use_similarity:
                    # Far neighbours score feature similarity, 1 - diff.
                    for k in range(n_cont):
                        scores_i[k] += weight * (1.0 - abs(x_i[k] - x_j[k]))
                    for k in range(n_cont, n_kept):
                        if x_i[k] == x_j[k]:
                            scores_i[k] += weight
                else:
                    for k in range(n_cont):
                        scores_i[k] += weight * abs(x_i[k] - x_j[k])
                    for k in range(n_cont, n_kept):
                        if x_i[k] != x_j[k]:
                            scores_i[k] += weight

    for k in range(n_kept):
        tot = 0.0
        for t in range(n_threads):
            tot += thread_scores[t, k]
        scores_out[k] = tot


def _multisurf_cpu_host_caller(x, y, n_cont, use_star, n_jobs):
    """Host caller for MultiSURF CPU kernel."""
    scores = np.zeros(x.shape[1], dtype=np.float32)

    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs
    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)

    try:
        _multisurf_cpu_kernel(x, y, n_cont, use_star, scores)
    finally:
        set_num_threads(original_num_threads)

    return scores


class MultiSURF(TransformerMixin, BaseEstimator):
    """GPU and CPU-accelerated feature selection using the MultiSURF algorithm.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend to use.

    use_star : bool, default=False
        If True, runs MultiSURF*: near and far thresholds surround a dead
        band, and far neighbors are scored by feature-value similarity.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    n_jobs : int, default=-1
        Number of CPU threads to use for 'cpu' backend.

    verbose : bool, default=False
        Controls whether progress updates are printed during fit.

    Notes
    -----
    The paper-defined complete-data binary-classification algorithm is used.
    Multiclass targets are supported as a pooled-miss extension; that extension
    is not presented as part of the original MultiSURF/MultiSURF* definition.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        backend: str = "auto",
        use_star: bool = False,
        discrete_limit: int = 10,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.n_features_to_select = n_features_to_select
        self.backend = backend
        self.use_star = use_star
        self.discrete_limit = discrete_limit
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _validate_parameters(self, n_samples: int, n_features: int) -> int:
        if self.backend not in ["auto", "gpu", "cpu"]:
            raise ValueError("backend must be one of 'auto', 'gpu', or 'cpu'")

        if n_samples < 2:
            raise ValueError(f"MultiSURF requires at least 2 samples, but got n_samples = {n_samples}")

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

    def fit(self, X: np.ndarray, y: np.ndarray, feat_idx: np.ndarray | None = None):
        """Fits MultiSURF model."""
        X, y = validate_data(
            self,
            X,
            y,
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            y_numeric=True,
        )
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        n_select = self._validate_parameters(n_samples, self.n_features_in_)

        if feat_idx is None:
            feat_idx = np.arange(self.n_features_in_, dtype=np.int32)
        else:
            feat_idx = np.asarray(feat_idx, dtype=np.int32)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float32)
            self.top_features_ = np.arange(n_select)
            self.effective_backend_ = "cpu" if self.backend != "gpu" else "gpu"
            return self

        if self.backend == "auto":
            self.effective_backend_ = "gpu" if is_cuda_ready() else "cpu"
        elif self.backend == "gpu" and not is_cuda_ready():
            raise RuntimeError("backend='gpu', but no CUDA-enabled GPU is available.")
        else:
            self.effective_backend_ = self.backend

        self.is_discrete_ = discrete_feature_mask(X, self.discrete_limit)

        feature_ranges = X.max(axis=0) - X.min(axis=0)
        feature_ranges[self.is_discrete_] = 1.0
        feature_ranges[feature_ranges == 0] = 1.0
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        algo_name = "MultiSURF*" if self.use_star else "MultiSURF"
        if self.verbose:
            print(f"Running {algo_name} on the {self.effective_backend_.upper()} now...")

        if self.effective_backend_ == "gpu":
            X_d = cuda.to_device(np.ascontiguousarray(X, dtype=np.float32))
            recip_full_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(self.is_discrete_)
            scores = _multisurf_gpu_host_caller(
                X_d, y_encoded.astype(np.int32), recip_full_d, feat_idx, self.use_star, is_discrete_d
            )
        else:
            columns, n_cont = split_discrete_last(self.is_discrete_, feat_idx)
            X_cpu = build_kernel_matrix(X, columns, recip_full, n_cont)
            scores = _multisurf_cpu_host_caller(X_cpu, y_encoded.astype(np.int32), n_cont, self.use_star, self.n_jobs)
            feat_idx = columns

        full_scores = np.zeros(self.n_features_in_, dtype=np.float32)
        full_scores[feat_idx] = scores

        self.feature_importances_ = full_scores
        self.top_features_ = np.argsort(full_scores)[::-1][:n_select]

        if self.verbose:
            print("Feature scoring completed.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to selected features."""
        check_is_fitted(self)

        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])

        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feat_idx: np.ndarray | None = None) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y, feat_idx=feat_idx)
        return self.transform(X)
