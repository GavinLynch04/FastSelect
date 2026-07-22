from __future__ import annotations
import numpy as np
from numba import cuda, float32, int32, njit, prange, set_num_threads, get_num_threads, get_thread_id, config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
import warnings
from .utils import is_cuda_ready

TPB = 64  # Threads-per-block

@cuda.jit
def _compute_dist_matrix_gpu_kernel(x, recip_full, is_discrete, dist_matrix): # pragma: no cover
    """
    Computes all pairwise sample distances on GPU with memory coalescing.
    Grid: (n_samples,), Threads per block: TPB
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    for j in range(n_samples):
        if i == j:
            if tid == 0:
                dist_matrix[i, j] = 0.0
            continue

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
        cuda.syncthreads()


@cuda.jit
def _accumulate_weighted_diffs_gpu_kernel(x, weights_matrix, recip_full, is_discrete, scores_out): # pragma: no cover
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


def _compute_relieff_weights(dist_matrix, y_enc, class_probs, k):
    """
    Computes ReliefF neighbor weight matrix W of shape (n_samples, n_samples)
    incorporating multi-class probability weighting matching literature standard.
    """
    n_samples = dist_matrix.shape[0]
    n_classes = len(class_probs)
    weights = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in range(n_samples):
        lbl_i = y_enc[i]
        denom = 1.0 - class_probs[lbl_i]
        if denom <= 0:
            denom = 1.0

        dists = dist_matrix[i].copy()
        dists[i] = np.inf

        # Find top k hits (class == lbl_i)
        hit_mask = (y_enc == lbl_i)
        hit_dists = np.where(hit_mask, dists, np.inf)
        hit_indices = np.argsort(hit_dists, kind='stable')[:k]
        actual_hits = [idx for idx in hit_indices if hit_dists[idx] != np.inf]
        h_count = len(actual_hits)

        if h_count > 0:
            weight_hit = -1.0 / (h_count * n_samples)
            for h_idx in actual_hits:
                weights[i, h_idx] += weight_hit

        # Find top k misses for each class c != lbl_i
        for c in range(n_classes):
            if c == lbl_i:
                continue
            miss_mask = (y_enc == c)
            miss_dists = np.where(miss_mask, dists, np.inf)
            miss_indices = np.argsort(miss_dists, kind='stable')[:k]
            actual_misses = [idx for idx in miss_indices if miss_dists[idx] != np.inf]
            m_count = len(actual_misses)

            if m_count > 0:
                prob_weight = class_probs[c] / denom
                weight_miss = prob_weight / (m_count * n_samples)
                for m_idx in actual_misses:
                    weights[i, m_idx] += weight_miss

    return weights


from .utils import is_cuda_ready, ensure_cuda_context

def _relieff_gpu_host_caller(x_d, y_enc, recip_full_d, is_discrete_d, class_probs, k):
    """Host caller launching distance computation, weight matrix assembly, and score accumulation."""
    ensure_cuda_context()
    n_samples, n_features = x_d.shape
    dist_matrix_d = cuda.device_array((n_samples, n_samples), dtype=np.float32)

    _compute_dist_matrix_gpu_kernel[n_samples, TPB](x_d, recip_full_d, is_discrete_d, dist_matrix_d)

    dist_matrix = dist_matrix_d.copy_to_host()
    weights_matrix = _compute_relieff_weights(dist_matrix, y_enc, class_probs, k)

    weights_d = cuda.to_device(weights_matrix)
    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0

    _accumulate_weighted_diffs_gpu_kernel[n_samples, TPB](x_d, weights_d, recip_full_d, is_discrete_d, scores_d)

    return scores_d.copy_to_host()


@njit(parallel=True, fastmath=True)
def _relieff_cpu_kernel(x, y_enc, recip_full, is_discrete, k, class_probs, scores_out): # pragma: no cover
    n_samples, n_features = x.shape
    n_classes = class_probs.shape[0]
    n_threads = get_num_threads()

    thread_scores = np.zeros((n_threads, n_features), dtype=np.float32)

    for i in prange(n_samples):
        tid = get_thread_id()
        lbl_i = y_enc[i]

        hit_d = np.full(k, np.inf, dtype=np.float32)
        hit_idx = np.full(k, -1, dtype=np.int32)

        miss_d = np.full((n_classes, k), np.inf, dtype=np.float32)
        miss_idx = np.full((n_classes, k), -1, dtype=np.int32)

        for j in range(n_samples):
            if i == j:
                continue

            d = 0.0
            for f in range(n_features):
                if is_discrete[f]:
                    d += 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    d += abs(x[i, f] - x[j, f]) * recip_full[f]

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

        if h_found > 0:
            scale_hit = -1.0 / (h_found * n_samples)
            for ki in range(h_found):
                h = hit_idx[ki]
                for f in range(n_features):
                    if is_discrete[f]:
                        diff = 1.0 if x[i, f] != x[h, f] else 0.0
                    else:
                        diff = abs(x[i, f] - x[h, f]) * recip_full[f]
                    thread_scores[tid, f] += scale_hit * diff

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
                scale_miss = weight_c / (m_count if (m_count := m_found) > 0 else 1) / n_samples
                for ki in range(m_found):
                    m = miss_idx[c, ki]
                    for f in range(n_features):
                        if is_discrete[f]:
                            diff = 1.0 if x[i, f] != x[m, f] else 0.0
                        else:
                            diff = abs(x[i, f] - x[m, f]) * recip_full[f]
                        thread_scores[tid, f] += scale_miss * diff

    for f in range(n_features):
        tot = 0.0
        for t in range(n_threads):
            tot += thread_scores[t, f]
        scores_out[f] = tot


def _relieff_cpu_host_caller(x, y_enc, recip_full, is_discrete, k, class_probs, n_jobs, discrete_weights):
    n_samples, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)

    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs

    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)

    try:
        _relieff_cpu_kernel(x, y_enc, recip_full, is_discrete, k, class_probs, scores)
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
            raise ValueError(
                f"ReliefF requires at least 2 samples, but got n_samples = {n_samples}"
            )

        if not (0 < self.n_neighbors < n_samples):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be an integer "
                f"between 1 and n_samples - 1 ({n_samples - 1})."
            )

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError(
                    "If n_features_to_select is a float, it must be in (0, 1]."
                )
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
            self, x, y, dtype=np.float64, ensure_2d=True, y_numeric=True,
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
                UserWarning
            )

        is_discrete = np.array([
            np.unique(x[:, f]).size <= self.discrete_limit for f in range(self.n_features_in_)
        ], dtype=bool)
        self.is_discrete_ = is_discrete

        discrete_weights = np.ones(self.n_features_in_, dtype=np.float32)

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
            x_d = cuda.to_device(x.astype(np.float32))
            recip_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(is_discrete.astype(np.bool_))
            if self.verbose:
                print("Running ReliefF on the GPU now...")
            scores = _relieff_gpu_host_caller(
                x_d, y_enc, recip_d, is_discrete_d, class_probs, self.n_neighbors)
        else:
            if self.verbose:
                print("Running ReliefF on the CPU now...")
            scores = _relieff_cpu_host_caller(
                x.astype(np.float32), y_enc, recip_full,
                is_discrete, self.n_neighbors, class_probs,
                self.n_jobs, discrete_weights
            )

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Reduces x to the selected features."""
        check_is_fitted(self)

        x = validate_data(
            self, x,
            reset=False,
            dtype=[np.float64, np.float32]
        )

        return x[:, self.top_features_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(x, y)
        return self.transform(x)