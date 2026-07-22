from __future__ import annotations
import numpy as np
from numba import cuda, float32, int32, njit, prange, config, get_num_threads, set_num_threads, get_thread_id
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
import warnings
from .utils import is_cuda_ready

TPB = 64  # Threads Per Block

@cuda.jit
def _compute_dist_matrix_surf_kernel(x, recip_full, is_discrete, dist_matrix): # pragma: no cover
    """Computes all pairwise distances on GPU with memory coalescing."""
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
def _accumulate_weighted_diffs_surf_kernel(x, weights_matrix, recip_full, is_discrete, scores_out): # pragma: no cover
    """Accumulates feature difference scores weighted by SURF neighbor relationships."""
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


def _compute_surf_weights(dist_matrix, y, use_star):
    """Computes SURF / SURF* weight matrix W of shape (n_samples, n_samples)."""
    n_samples = dist_matrix.shape[0]
    weights = np.zeros((n_samples, n_samples), dtype=np.float32)
    scale = 1.0 / n_samples

    for i in range(n_samples):
        row_dists = dist_matrix[i]
        sum_d = np.sum(row_dists) - row_dists[i]
        avg_dist = sum_d / (n_samples - 1) if n_samples > 1 else 0.0

        for j in range(n_samples):
            if i == j:
                continue
            is_hit = (y[i] == y[j])
            is_near = (row_dists[j] < avg_dist)

            if is_near:
                weights[i, j] = scale if not is_hit else -scale
            elif use_star:
                weights[i, j] = scale if is_hit else -scale

    return weights


from .utils import is_cuda_ready, ensure_cuda_context

def _surf_gpu_host_caller(x_d, y, recip_full_d, use_star, is_discrete_d):
    """Host helper function that launches GPU kernels for SURF and returns scores."""
    ensure_cuda_context()
    n_samples, n_features = x_d.shape
    dist_matrix_d = cuda.device_array((n_samples, n_samples), dtype=np.float32)

    _compute_dist_matrix_surf_kernel[n_samples, TPB](x_d, recip_full_d, is_discrete_d, dist_matrix_d)

    dist_matrix = dist_matrix_d.copy_to_host()
    weights_matrix = _compute_surf_weights(dist_matrix, y, use_star)

    weights_d = cuda.to_device(weights_matrix)
    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0

    _accumulate_weighted_diffs_surf_kernel[n_samples, TPB](x_d, weights_d, recip_full_d, is_discrete_d, scores_d)

    return scores_d.copy_to_host()


@njit(parallel=True, fastmath=True)
def _surf_cpu_kernel(x, y, recip_full, use_star, is_discrete, scores_out): # pragma: no cover
    """
    Optimized SURF/SURF* scoring for CPU with zero N x P temporary matrix allocations inside prange.
    """
    n_samples, n_features = x.shape
    n_threads = get_num_threads()
    thread_scores = np.zeros((n_threads, n_features), dtype=np.float32)

    for i in prange(n_samples):
        tid = get_thread_id()
        dists_from_i = np.empty(n_samples, dtype=np.float32)
        sum_d = 0.0

        for j in range(n_samples):
            if i == j:
                dists_from_i[j] = 0.0
                continue

            dist_ij = 0.0
            for f in range(n_features):
                if is_discrete[f]:
                    diff = 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                dist_ij += diff

            dists_from_i[j] = dist_ij
            sum_d += dist_ij

        avg_dist = sum_d / (n_samples - 1) if n_samples > 1 else 0.0
        scale = 1.0 / n_samples

        for j in range(n_samples):
            if i == j:
                continue

            dist_ij = dists_from_i[j]
            is_hit = (y[i] == y[j])
            is_near = (dist_ij < avg_dist)

            weight = 0.0
            if is_near:
                weight = scale if not is_hit else -scale
            elif use_star:
                weight = scale if is_hit else -scale

            if weight != 0.0:
                for f in range(n_features):
                    if is_discrete[f]:
                        diff = 1.0 if x[i, f] != x[j, f] else 0.0
                    else:
                        diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                    thread_scores[tid, f] += weight * diff

    for f in range(n_features):
        tot = 0.0
        for t in range(n_threads):
            tot += thread_scores[t, f]
        scores_out[f] = tot


def _surf_cpu_host_caller(x, y, recip_full, use_star, is_discrete, n_jobs):
    """Host caller for the CPU kernel."""
    n_samples, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)

    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs

    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)

    try:
        _surf_cpu_kernel(x, y, recip_full, use_star, is_discrete, scores)
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
                f"SURF requires at least 2 samples, but got n_samples = {n_samples}"
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits SURF to training data."""
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_2d=True, y_numeric=True,
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

        self.is_discrete_ = np.array([
            np.unique(X[:, f]).size <= self.discrete_limit
            for f in range(self.n_features_in_)
        ], dtype=bool)

        feature_ranges = X.max(axis=0) - X.min(axis=0)
        feature_ranges[self.is_discrete_] = 1.0
        feature_ranges[feature_ranges == 0] = 1.0
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        algo_name = "SURF*" if self.use_star else "SURF"
        if self.verbose:
            print(f"Running {algo_name} on the {self.effective_backend_.upper()} now...")

        if self.effective_backend_ == "gpu":
            X_d = cuda.to_device(X.astype(np.float32))
            recip_full_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(self.is_discrete_)
            scores = _surf_gpu_host_caller(
                X_d, y_encoded.astype(np.int32), recip_full_d, self.use_star, is_discrete_d
            )
        else:
            scores = _surf_cpu_host_caller(
                X.astype(np.float32), y_encoded.astype(np.int32), recip_full, self.use_star, self.is_discrete_, self.n_jobs
            )

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]

        if self.verbose:
            print("Feature scoring completed.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to selected top features."""
        check_is_fitted(self)

        X = validate_data(
            self, X,
            reset=False,
            dtype=[np.float64, np.float32]
        )

        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)