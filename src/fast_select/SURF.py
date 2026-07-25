from __future__ import annotations

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

# Above this many features the shared-memory score accumulator stops paying for
# itself and the global-atomic kernel is faster; measured on the benchmark in
# benchmarking/compare_head_vs_current.py.
SHARED_SCORE_MAX_FEATURES = 512


@cuda.jit
def _compute_dist_matrix_surf_kernel(x, recip_full, is_discrete, dist_matrix):  # pragma: no cover
    """Computes the upper triangle once and mirrors it into the distance matrix."""
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
def _score_surf_gpu_kernel(
    x,
    y,
    dist_matrix,
    recip_full,
    global_threshold,
    use_star,
    is_discrete,
    scores_out,
):  # pragma: no cover
    """Score SURF with the single global radius defined by the algorithm."""
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    local_near_hits = 0
    local_near_misses = 0
    local_far_hits = 0
    local_far_misses = 0
    for j in range(tid, n_samples, TPB):
        if i == j:
            continue
        dist = dist_matrix[i, j]
        is_hit = y[i] == y[j]
        if dist < global_threshold:
            if is_hit:
                local_near_hits += 1
            else:
                local_near_misses += 1
        elif use_star and dist > global_threshold:
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
        if dist < global_threshold:
            if is_hit and n_near_hits > 0:
                weight = -scale / n_near_hits
            elif not is_hit and n_near_misses > 0:
                weight = scale / n_near_misses
        elif use_star and dist > global_threshold:
            if is_hit and n_far_hits > 0:
                weight = scale / n_far_hits
            elif not is_hit and n_far_misses > 0:
                weight = -scale / n_far_misses

        if weight == 0.0:
            continue
        for f in range(tid, n_features, TPB):
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            cuda.atomic.add(scores_out, f, weight * diff)


@cuda.jit
def _score_surf_shared_gpu_kernel(
    x,
    y,
    dist_matrix,
    recip_full,
    global_threshold,
    use_star,
    is_discrete,
    scores_out,
):  # pragma: no cover
    """SURF scoring accumulated in shared memory.

    Identical equations to :func:`_score_surf_gpu_kernel`; the only difference
    is that a block keeps its per-feature running totals in shared memory and
    issues one global atomic per feature at the end instead of one per
    (neighbour, feature).  Selected by the host only when the feature block fits
    in shared memory; see ``SHARED_SCORE_MAX_FEATURES``.
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    local_near_hits = 0
    local_near_misses = 0
    local_far_hits = 0
    local_far_misses = 0
    for j in range(tid, n_samples, TPB):
        if i == j:
            continue
        dist = dist_matrix[i, j]
        is_hit = y[i] == y[j]
        if dist < global_threshold:
            if is_hit:
                local_near_hits += 1
            else:
                local_near_misses += 1
        elif use_star and dist > global_threshold:
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
    for f in range(tid, n_features, TPB):
        sh_scores[f] = 0.0
    cuda.syncthreads()

    for j in range(n_samples):
        if i == j:
            continue

        is_hit = y[i] == y[j]
        dist = dist_matrix[i, j]
        weight = 0.0
        if dist < global_threshold:
            if is_hit and n_near_hits > 0:
                weight = -scale / n_near_hits
            elif not is_hit and n_near_misses > 0:
                weight = scale / n_near_misses
        elif use_star and dist > global_threshold:
            if is_hit and n_far_hits > 0:
                weight = scale / n_far_hits
            elif not is_hit and n_far_misses > 0:
                weight = -scale / n_far_misses

        if weight == 0.0:
            continue
        for f in range(tid, n_features, TPB):
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            sh_scores[f] += weight * diff

    cuda.syncthreads()
    for f in range(tid, n_features, TPB):
        if sh_scores[f] != 0.0:
            cuda.atomic.add(scores_out, f, sh_scores[f])


@njit(parallel=True, fastmath=True)
def _global_mean_distance_cpu(x, recip_full, is_discrete):  # pragma: no cover
    """Mean Manhattan distance over all unique sample pairs without an N x N matrix."""
    n_samples, n_features = x.shape
    pair_count = n_samples * (n_samples - 1) // 2
    feature_pair_sums = np.zeros(n_features, dtype=np.float64)

    for f in prange(n_features):
        values = np.sort(x[:, f])
        pair_sum = 0.0
        if is_discrete[f]:
            run_start = 0
            while run_start < n_samples:
                run_end = run_start + 1
                while run_end < n_samples and values[run_end] == values[run_start]:
                    run_end += 1
                run_length = run_end - run_start
                pair_sum += run_length * (n_samples - run_length)
                run_start = run_end
            pair_sum *= 0.5
        else:
            for i in range(n_samples):
                pair_sum += values[i] * (2 * i - n_samples + 1)
            pair_sum *= recip_full[f]
        feature_pair_sums[f] = pair_sum

    total = 0.0
    for f in range(n_features):
        total += feature_pair_sums[f]
    return total / pair_count if pair_count > 0 else 0.0


def _surf_gpu_host_caller(x_d, y, recip_full_d, global_threshold, use_star, is_discrete_d):
    """Launch GPU-only distance and scoring stages for SURF."""
    ensure_cuda_context()
    n_samples, n_features = x_d.shape
    dist_matrix_d = cuda.device_array((n_samples, n_samples), dtype=np.float32)
    y_d = cuda.to_device(y)

    _compute_dist_matrix_surf_kernel[n_samples, TPB](x_d, recip_full_d, is_discrete_d, dist_matrix_d)

    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0

    if n_features <= SHARED_SCORE_MAX_FEATURES:
        _score_surf_shared_gpu_kernel[n_samples, TPB, 0, n_features * 4](
            x_d,
            y_d,
            dist_matrix_d,
            recip_full_d,
            global_threshold,
            use_star,
            is_discrete_d,
            scores_d,
        )
    else:
        _score_surf_gpu_kernel[n_samples, TPB](
            x_d,
            y_d,
            dist_matrix_d,
            recip_full_d,
            global_threshold,
            use_star,
            is_discrete_d,
            scores_d,
        )

    return scores_d.copy_to_host()


@njit(parallel=True, fastmath=True)
def _surf_cpu_kernel(x, y, n_cont, global_threshold, use_star, scores_out):  # pragma: no cover
    """
    Optimized SURF/SURF* scoring for CPU with zero N x P temporary matrix allocations inside prange.

    ``x`` arrives with continuous features in columns ``[0, n_cont)`` and
    discrete features after them, so both inner loops are branch-free and
    vectorisable.  The distance is still the sum of the same per-feature terms.
    """
    n_samples, n_features = x.shape
    n_threads = get_num_threads()
    thread_scores = np.zeros((n_threads, n_features), dtype=np.float32)
    thread_dists = np.empty((n_threads, n_samples), dtype=np.float32)

    for i in prange(n_samples):
        tid = get_thread_id()
        dists_from_i = thread_dists[tid]
        x_i = x[i]
        for j in range(n_samples):
            if i == j:
                dists_from_i[j] = 0.0
                continue

            x_j = x[j]
            dist_ij = 0.0
            for f in range(n_cont):
                dist_ij += abs(x_i[f] - x_j[f])
            for f in range(n_cont, n_features):
                if x_i[f] != x_j[f]:
                    dist_ij += 1.0

            dists_from_i[j] = dist_ij

        near_hits = 0
        near_misses = 0
        far_hits = 0
        far_misses = 0
        for j in range(n_samples):
            if i == j:
                continue
            dist_ij = dists_from_i[j]
            is_hit = y[i] == y[j]
            if dist_ij < global_threshold:
                if is_hit:
                    near_hits += 1
                else:
                    near_misses += 1
            elif use_star and dist_ij > global_threshold:
                if is_hit:
                    far_hits += 1
                else:
                    far_misses += 1

        scale = 1.0 / n_samples
        near_hit_weight = -scale / near_hits if near_hits > 0 else 0.0
        near_miss_weight = scale / near_misses if near_misses > 0 else 0.0
        far_hit_weight = scale / far_hits if far_hits > 0 else 0.0
        far_miss_weight = -scale / far_misses if far_misses > 0 else 0.0

        scores_i = thread_scores[tid]
        for j in range(n_samples):
            if i == j:
                continue
            dist_ij = dists_from_i[j]
            is_hit = y[i] == y[j]
            weight = 0.0
            if dist_ij < global_threshold:
                weight = near_hit_weight if is_hit else near_miss_weight
            elif use_star and dist_ij > global_threshold:
                weight = far_hit_weight if is_hit else far_miss_weight

            if weight != 0.0:
                x_j = x[j]
                for f in range(n_cont):
                    scores_i[f] += weight * abs(x_i[f] - x_j[f])
                for f in range(n_cont, n_features):
                    if x_i[f] != x_j[f]:
                        scores_i[f] += weight

    for f in range(n_features):
        tot = 0.0
        for t in range(n_threads):
            tot += thread_scores[t, f]
        scores_out[f] = tot


def _surf_cpu_host_caller(x, y, n_cont, global_threshold, use_star, n_jobs):
    """Host caller for the CPU kernel."""
    n_samples, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)

    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs

    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)

    try:
        _surf_cpu_kernel(x, y, n_cont, global_threshold, use_star, scores)
    finally:
        set_num_threads(original_num_threads)

    return scores


class SURF(TransformerMixin, BaseEstimator):
    """GPU and CPU-accelerated feature selection using the SURF algorithm.

    This estimator provides a unified scikit-learn compatible API for SURF and SURF*.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend to use.

    use_star : bool, default=False
        If True, runs SURF*, including updates from far neighbors.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    n_jobs : int, default=-1
        Number of CPU threads to use for 'cpu' backend.

    verbose : bool, default=False
        Controls whether to print progress messages during fit.

    Notes
    -----
    The paper-defined complete-data binary-classification algorithm is used.
    Multiclass targets are supported as a pooled-miss extension; that extension
    is not presented as part of the original SURF/SURF* definition.
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
            raise ValueError(f"SURF requires at least 2 samples, but got n_samples = {n_samples}")

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

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits SURF to training data."""
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
        X_float32 = np.ascontiguousarray(X, dtype=np.float32)
        global_threshold = _global_mean_distance_cpu(X_float32, recip_full, self.is_discrete_)

        algo_name = "SURF*" if self.use_star else "SURF"
        if self.verbose:
            print(f"Running {algo_name} on the {self.effective_backend_.upper()} now...")

        if self.effective_backend_ == "gpu":
            X_d = cuda.to_device(X_float32)
            recip_full_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(self.is_discrete_)
            scores = _surf_gpu_host_caller(
                X_d,
                y_encoded.astype(np.int32),
                recip_full_d,
                global_threshold,
                self.use_star,
                is_discrete_d,
            )
        else:
            columns, n_cont = split_discrete_last(self.is_discrete_)
            X_cpu = build_kernel_matrix(X_float32, columns, recip_full, n_cont)
            permuted_scores = _surf_cpu_host_caller(
                X_cpu,
                y_encoded.astype(np.int32),
                n_cont,
                global_threshold,
                self.use_star,
                self.n_jobs,
            )
            scores = np.zeros(self.n_features_in_, dtype=np.float32)
            scores[columns] = permuted_scores

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]

        if self.verbose:
            print("Feature scoring completed.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to selected top features."""
        check_is_fitted(self)

        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])

        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
