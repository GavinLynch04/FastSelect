from __future__ import annotations
import math
import warnings
import numpy as np
from numba import cuda, float32, int32, njit, prange, config, get_num_threads, set_num_threads, get_thread_id
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from .utils import is_cuda_ready

TPB = 64  # Threads Per Block

@cuda.jit
def _compute_dist_matrix_multisurf_kernel(x, recip_full, feat_idx, is_discrete, dist_matrix): # pragma: no cover
    """Computes all pairwise distances for feature subset on GPU."""
    n_samples = x.shape[0]
    n_kept = feat_idx.shape[0]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    for j in range(n_samples):
        if i == j:
            if tid == 0:
                dist_matrix[i, j] = 0.0
            continue

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
        cuda.syncthreads()


@cuda.jit
def _accumulate_weighted_diffs_multisurf_kernel(x, weights_matrix, recip_full, feat_idx, is_discrete, scores_out): # pragma: no cover
    """Accumulates feature scores for selected feature subset on GPU."""
    n_samples = x.shape[0]
    n_kept = feat_idx.shape[0]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    for j in range(n_samples):
        w = weights_matrix[i, j]
        if w == 0.0:
            continue

        for k in range(tid, n_kept, TPB):
            f = feat_idx[k]
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            cuda.atomic.add(scores_out, k, w * diff)


def _compute_multisurf_weights(dist_matrix, y, use_star):
    """Computes MultiSURF weight matrix W of shape (n_samples, n_samples)."""
    n_samples = dist_matrix.shape[0]
    weights = np.zeros((n_samples, n_samples), dtype=np.float32)
    scale = 1.0 / n_samples

    for i in range(n_samples):
        row_dists = dist_matrix[i]
        sum_d = np.sum(row_dists) - row_dists[i]
        sum_d2 = np.sum(row_dists ** 2) - (row_dists[i] ** 2)

        mu = sum_d / (n_samples - 1) if n_samples > 1 else 0.0
        var = max(0.0, (sum_d2 / (n_samples - 1)) - (mu * mu)) if n_samples > 1 else 0.0
        sigma = math.sqrt(var)
        thresh = mu - 0.5 * sigma

        near_hits = 0
        near_misses = 0

        for j in range(n_samples):
            if i == j:
                continue
            is_hit = (y[i] == y[j])
            if row_dists[j] < thresh:
                if is_hit:
                    near_hits += 1
                else:
                    near_misses += 1

        w_hit = -scale / near_hits if near_hits > 0 else 0.0
        w_miss = scale / near_misses if near_misses > 0 else 0.0

        for j in range(n_samples):
            if i == j:
                continue
            is_hit = (y[i] == y[j])
            if row_dists[j] < thresh:
                weights[i, j] = w_hit if is_hit else w_miss
            elif use_star and not is_hit:
                weights[i, j] = -scale

    return weights


from .utils import is_cuda_ready, ensure_cuda_context

def _multisurf_gpu_host_caller(x_d, y, recip_full_d, feat_idx: np.ndarray, use_star: bool, is_discrete_d) -> np.ndarray:
    """Host caller launching GPU kernels for MultiSURF."""
    ensure_cuda_context()
    n_samples = x_d.shape[0]
    n_kept = feat_idx.size

    feat_idx_d = cuda.to_device(feat_idx.astype(np.int32))
    dist_matrix_d = cuda.device_array((n_samples, n_samples), dtype=np.float32)

    _compute_dist_matrix_multisurf_kernel[n_samples, TPB](x_d, recip_full_d, feat_idx_d, is_discrete_d, dist_matrix_d)

    dist_matrix = dist_matrix_d.copy_to_host()
    weights_matrix = _compute_multisurf_weights(dist_matrix, y, use_star)

    weights_d = cuda.to_device(weights_matrix)
    scores_d = cuda.device_array(n_kept, dtype=np.float32)
    scores_d[:] = 0.0

    _accumulate_weighted_diffs_multisurf_kernel[n_samples, TPB](x_d, weights_d, recip_full_d, feat_idx_d, is_discrete_d, scores_d)

    return scores_d.copy_to_host()


@njit(parallel=True, fastmath=True)
def _multisurf_cpu_kernel(x, y, recip_full, feat_idx, n_kept, use_star, is_discrete, scores_out): # pragma: no cover
    """
    Optimized MultiSURF CPU kernel with zero N x P matrix allocations in parallel loop.
    """
    n_samples = x.shape[0]
    n_threads = get_num_threads()
    thread_scores = np.zeros((n_threads, n_kept), dtype=np.float32)

    for i in prange(n_samples):
        tid = get_thread_id()
        dists_from_i = np.empty(n_samples, dtype=np.float32)
        sum_d = 0.0
        sum_d2 = 0.0

        for j in range(n_samples):
            if i == j:
                dists_from_i[j] = 0.0
                continue

            dist = 0.0
            for k in range(n_kept):
                f = feat_idx[k]
                if is_discrete[f]:
                    diff = 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                dist += diff

            dists_from_i[j] = dist
            sum_d += dist
            sum_d2 += dist * dist

        mu = sum_d / (n_samples - 1) if n_samples > 1 else 0.0
        var = max(0.0, (sum_d2 / (n_samples - 1)) - (mu * mu)) if n_samples > 1 else 0.0
        sigma = math.sqrt(var)
        thresh = mu - 0.5 * sigma

        n_hits = 0
        n_miss = 0

        for j in range(n_samples):
            if i == j:
                continue
            is_hit = (y[i] == y[j])
            if dists_from_i[j] < thresh:
                if is_hit:
                    n_hits += 1
                else:
                    n_miss += 1

        scale = 1.0 / n_samples
        w_hit = -scale / n_hits if n_hits > 0 else 0.0
        w_miss = scale / n_miss if n_miss > 0 else 0.0

        for j in range(n_samples):
            if i == j:
                continue
            is_hit = (y[i] == y[j])
            weight = 0.0
            if dists_from_i[j] < thresh:
                weight = w_hit if is_hit else w_miss
            elif use_star and not is_hit:
                weight = -scale

            if weight != 0.0:
                for k in range(n_kept):
                    f = feat_idx[k]
                    if is_discrete[f]:
                        diff = 1.0 if x[i, f] != x[j, f] else 0.0
                    else:
                        diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                    thread_scores[tid, k] += weight * diff

    for k in range(n_kept):
        tot = 0.0
        for t in range(n_threads):
            tot += thread_scores[t, k]
        scores_out[k] = tot


def _multisurf_cpu_host_caller(x, y, recip_full, feat_idx, use_star, is_discrete, n_jobs):
    """Host caller for MultiSURF CPU kernel."""
    n_kept = feat_idx.size
    scores = np.zeros(n_kept, dtype=np.float32)

    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs
    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)

    try:
        _multisurf_cpu_kernel(x, y, recip_full, feat_idx, n_kept, use_star, is_discrete, scores)
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
        If True, includes far miss updates in MultiSURF*.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    n_jobs : int, default=-1
        Number of CPU threads to use for 'cpu' backend.

    verbose : bool, default=False
        Controls whether progress updates are printed during fit.
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
            raise ValueError(
                f"MultiSURF requires at least 2 samples, but got n_samples = {n_samples}"
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

    def fit(self, X: np.ndarray, y: np.ndarray, feat_idx: np.ndarray | None = None):
        """Fits MultiSURF model."""
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_2d=True, y_numeric=True,
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

        self.is_discrete_ = np.array([
            np.unique(X[:, f]).size <= self.discrete_limit
            for f in range(self.n_features_in_)
        ], dtype=bool)

        feature_ranges = X.max(axis=0) - X.min(axis=0)
        feature_ranges[self.is_discrete_] = 1.0
        feature_ranges[feature_ranges == 0] = 1.0
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        algo_name = "MultiSURF*" if self.use_star else "MultiSURF"
        if self.verbose:
            print(f"Running {algo_name} on the {self.effective_backend_.upper()} now...")

        if self.effective_backend_ == "gpu":
            X_d = cuda.to_device(X.astype(np.float32))
            recip_full_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(self.is_discrete_)
            scores = _multisurf_gpu_host_caller(
                X_d, y_encoded.astype(np.int32), recip_full_d, feat_idx, self.use_star, is_discrete_d
            )
        else:
            scores = _multisurf_cpu_host_caller(
                X.astype(np.float32), y_encoded.astype(np.int32), recip_full, feat_idx, self.use_star, self.is_discrete_, self.n_jobs
            )

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

        X = validate_data(
            self, X,
            reset=False,
            dtype=[np.float64, np.float32]
        )

        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feat_idx: np.ndarray | None = None) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y, feat_idx=feat_idx)
        return self.transform(X)
