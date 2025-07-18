from __future__ import annotations
import math
import warnings
import numpy as np
from numba import cuda, float32, int32, njit, prange, config, get_num_threads, set_num_threads
from numba.core.errors import NumbaPerformanceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted,check_X_y
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

TPB = 64  # Threads Per Block
MAx_F_TILE = 1024  # Features loaded per shared-memory tile


@cuda.jit
def _multisurf_gpu_kernel(x, y, recip_full, feat_idx, n_kept, use_star, is_discrete, scores_out):
    """MultiSURF scoring for an _arbitrary subset_ of features."""
    n_samples = x.shape[0]
    i = cuda.blockIdx.x  # focal sample index
    tid = cuda.threadIdx.x
    # Shared scratch
    hits_tile = cuda.shared.array(shape=MAx_F_TILE, dtype=float32)
    miss_tile = cuda.shared.array(shape=MAx_F_TILE, dtype=float32)
    sh_red_f32 = cuda.shared.array(shape=TPB, dtype=float32)
    sh_red_i32 = cuda.shared.array(shape=TPB, dtype=int32)

    sum_d = 0.0
    sum_d2 = 0.0
    for j in range(tid, n_samples, TPB):
        if j == i:
            continue
        dist = 0.0
        for f0 in range(0, n_kept, MAx_F_TILE):
            tile_len = min(MAx_F_TILE, n_kept - f0)
            for f in range(tile_len):
                full_idx = feat_idx[f0 + f]
                if is_discrete[full_idx]:
                    diff = 1.0 if x[i, full_idx] != x[j, full_idx] else 0.0
                else:
                    diff = abs(x[i, full_idx] - x[j, full_idx]) * recip_full[full_idx]
                dist += diff
        sum_d += dist
        sum_d2 += dist * dist
    # Reduce for mean
    sh_red_f32[tid] = sum_d
    cuda.syncthreads()
    off = TPB // 2
    while off:
        if tid < off:
            sh_red_f32[tid] += sh_red_f32[tid + off]
        off //= 2
        cuda.syncthreads()
    mu = sh_red_f32[0] / (n_samples - 1)
    # Reduce for variance
    sh_red_f32[tid] = sum_d2
    cuda.syncthreads()
    off = TPB // 2
    while off:
        if tid < off:
            sh_red_f32[tid] += sh_red_f32[tid + off]
        off //= 2
        cuda.syncthreads()
    var = sh_red_f32[0] / (n_samples - 1) - mu * mu
    sigma = math.sqrt(max(var, 0.0))
    thresh = mu - 0.5 * sigma

    for f0 in range(0, n_kept, MAx_F_TILE):
        tile_len = min(MAx_F_TILE, n_kept - f0)
        if tid < tile_len:
            hits_tile[tid] = 0.0
            miss_tile[tid] = 0.0
        cuda.syncthreads()
        n_hit_local = 0
        n_miss_local = 0
        for j in range(tid, n_samples, TPB):
            if j == i:
                continue
            dist = 0.0
            for f in range(tile_len):
                full_idx = feat_idx[f0 + f]
                if is_discrete[full_idx]:  # Use binary differences for discrete features
                    diff = 1.0 if x[i, full_idx] != x[j, full_idx] else 0.0
                else:  # Use scaled continuous difference for continuous features
                    diff = abs(x[i, full_idx] - x[j, full_idx]) * recip_full[full_idx]
                dist += diff
            is_hit = y[i] == y[j]
            if dist < thresh:  # This is a NEAR neighbor
                for f in range(tile_len):
                    full_idx = feat_idx[f0 + f]
                    diff = abs(x[i, full_idx] - x[j, full_idx]) * recip_full[full_idx]
                    if is_hit:
                        cuda.atomic.add(hits_tile, f, diff)
                    else:
                        cuda.atomic.add(miss_tile, f, diff)
                if is_hit:
                    n_hit_local += 1
                else:
                    n_miss_local += 1
            elif use_star and not is_hit:  # This is a FAR MISS
                for f in range(tile_len):
                    full_idx = feat_idx[f0 + f]
                    diff = abs(x[i, full_idx] - x[j, full_idx]) * recip_full[full_idx]
                    cuda.atomic.add(miss_tile, f, -diff)
        cuda.syncthreads()
        # Shared reduction of neighbour counts
        sh_red_i32[tid] = n_hit_local
        cuda.syncthreads()
        off = TPB // 2
        while off:
            if tid < off:
                sh_red_i32[tid] += sh_red_i32[tid + off]
            off //= 2
            cuda.syncthreads()
        total_hits = sh_red_i32[0]
        sh_red_i32[tid] = n_miss_local
        cuda.syncthreads()
        off = TPB // 2
        while off:
            if tid < off:
                sh_red_i32[tid] += sh_red_i32[tid + off]
            off //= 2
            cuda.syncthreads()
        total_miss = sh_red_i32[0]
        if tid < tile_len:
            local_idx = f0 + tid
            term = 0.0
            if total_miss > 0:
                term += miss_tile[tid] / total_miss
            if total_hits > 0:
                term -= hits_tile[tid] / total_hits
            cuda.atomic.add(scores_out, local_idx, term)
        cuda.syncthreads()


def _compute_ranges(x: np.ndarray) -> np.ndarray:
    """Helper to compute feature ranges on the CPU."""
    ranges = (x.max(axis=0) - x.min(axis=0)).astype(np.float32)
    return ranges


def _multisurf_gpu_host_caller(
    x_d, y_d, recip_full_d, feat_idx: np.ndarray, use_star: bool, is_discrete: np.ndarray
) -> np.ndarray:
    """Host helper function that launches the kernel and returns scores."""
    n_samples, _ = x_d.shape
    n_kept = feat_idx.size
    feat_idx_d = cuda.to_device(feat_idx.astype(np.int64, copy=False))
    scores_d = cuda.device_array(n_kept, dtype=np.float32)
    scores_d[:] = 0.0  # Zero-fill on device

    _multisurf_gpu_kernel[n_samples, TPB](
        x_d, y_d, recip_full_d, feat_idx_d, n_kept, use_star, is_discrete, scores_d
    )
    cuda.synchronize()

    return scores_d.copy_to_host() / n_samples


@njit(parallel=True, fastmath=True)
def _multisurf_cpu_kernel(x, y, recip_full, feat_idx, n_kept, use_star, is_discrete, scores_out):
    """
    Optimized MultiSURF scoring for CPU.

    This version avoids large intermediate allocations inside the parallel loop
    by recalculating distances in a second pass, which is faster due to
    Numba's compilation and reduced memory pressure.
    """
    n_samples = x.shape[0]

    temp_scores = np.zeros((n_samples, n_kept), dtype=np.float32)

    for i in prange(n_samples):
        sum_d = 0.0
        sum_d2 = 0.0
        for j in range(n_samples):
            if i == j:
                continue

            dist = 0.0
            for k in range(n_kept):
                f = feat_idx[k]
                if is_discrete[f]:
                    diff = 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                dist += diff

            sum_d += dist
            sum_d2 += dist * dist

        mu = sum_d / (n_samples - 1)
        var = max(0.0, (sum_d2 / (n_samples - 1)) - (mu * mu))
        sigma = math.sqrt(var)
        thresh = mu - 0.5 * sigma

        hit_diffs = np.zeros(n_kept, dtype=np.float32)
        miss_diffs = np.zeros(n_kept, dtype=np.float32)
        n_hits = 0
        n_miss = 0

        for j in range(n_samples):
            if i == j:
                continue

            dist = 0.0
            for k in range(n_kept):
                f = feat_idx[k]
                if is_discrete[f]:
                    diff = 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    diff = abs(x[i, f] - x[j, f]) * recip_full[f]  # Continuous difference
                dist += diff

            is_hit = y[i] == y[j]
            if dist < thresh:  # NEAR neighbor
                if is_hit:
                    n_hits += 1
                    for k in range(n_kept):
                        f = feat_idx[k]
                        diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                        hit_diffs[k] += diff
                else:
                    n_miss += 1
                    for k in range(n_kept):
                        f = feat_idx[k]
                        diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                        miss_diffs[k] += diff
            elif use_star and not is_hit:  # FAR MISS
                for k in range(n_kept):
                    f = feat_idx[k]
                    diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                    miss_diffs[k] -= diff  # Subtract the contribution

        if n_hits > 0:
            hit_diffs /= n_hits
        if n_miss > 0:
            miss_diffs /= n_miss

        for k in range(n_kept):
            temp_scores[i, k] = miss_diffs[k] - hit_diffs[k]
    for k in range(n_kept):
        scores_out[k] = temp_scores[:, k].sum()


def _multisurf_cpu_host_caller(x, y, recip_full, feat_idx, use_star, is_discrete, n_jobs):
    """Host caller for the optimized CPU kernel."""
    n_samples = x.shape[0]
    n_kept = feat_idx.size
    scores = np.zeros(n_kept, dtype=np.float32)
    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs
    original_num_threads = get_num_threads()

    try:
        set_num_threads(num_threads_to_set)
        _multisurf_cpu_kernel(x, y, recip_full, feat_idx, n_kept, use_star, is_discrete, scores)
    finally:
        set_num_threads(original_num_threads)

    return scores / n_samples


class MultiSURF(BaseEstimator, TransformerMixin):
    """GPU and CPU-accelerated feature selection using the MultiSURF algorithm.

    This estimator provides a unified API for running MultiSURF on either
    a CPU (using Numba's parallel JIT) or a GPU (using Numba CUDA).

    Parameters
    ----------
    n_features_to_select : int, default=10
        The number of top features to select.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend to use.
        - 'auto': Use 'gpu' if a compatible NVIDIA GPU is detected,
        otherwise fall back to 'cpu'.
        - 'gpu': Force use of the GPU. Raises an error if not available.
        - 'cpu': Force use of the CPU.

    use_star : bool, default=False
        Whether to run the MultiSURF* adaptation of the algorithm.
        By default, the standard MultiSURF algorithm is used.
        
    discrete_limit : int, default=10
        The limit of individual feature values to determine whether or not
        a given feature is discrete or continuous. (Effects distance
        calculation)
    
    verbose : bool, default=True
        Controls whether progress updates are printed during the fit.
        Limited benefit currently, will be expanded in future versions.
        
    n_jobs : int, default=-1
        Controls the number of threads utilized by Numba while running on the cpu.
        -1 uses all threads avaliable by default. Set to a low number if experiencing
        difficulties and lagging running the script.

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
        n_features_to_select: int = 10,
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

        if self.backend not in ["auto", "gpu", "cpu"]:
            raise ValueError("backend must be one of 'auto', 'gpu', or 'cpu'")

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Calculates feature importances using the MultiSURF algorithm on a GPU/CPU.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values. Please handle missing data before passing it.")
            
        x, y = check_X_y(x, y, dtype=np.float32, ensure_2d=True)
        self.n_features_in_ = x.shape[1]
        
        if self.n_features_to_select >= self.n_features_in_:
            raise ValueError("Number of features to select must be less than the number of input features.")
        if self.n_neighbors >= self.n_features_in_:
            raise ValueError("Number of neighbors must be less than the number of input features.")
            
        if self.backend == "auto":
            if cuda.is_available():
                self.effective_backend_ = "gpu"
            else:
                self.effective_backend_ = "cpu"
        elif self.backend == "gpu":
            if not cuda.is_available():
                raise RuntimeError(
                    "backend='gpu' was selected, but no compatible "
                    "NVIDIA GPU was found or CUDA toolkit is not installed."
                )
            self.effective_backend_ = "gpu"
        else:
            self.effective_backend_ = "cpu"

        feature_ranges = _compute_ranges(x)

        feature_ranges[feature_ranges == 0] = 1
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        all_feature_indices = np.arange(self.n_features_in_, dtype=np.int64)
        
        is_discrete = np.zeros(self.n_features_in_, dtype=bool)
        self.is_discrete_ = is_discrete
        for f in range(self.n_features_in_):
            if np.unique(x[:, f]).size <= self.discrete_limit:  # Define discrete_limit
                is_discrete[f] = True

        if self.effective_backend_ == "gpu":
            x_d = cuda.to_device(x)
            y_d = cuda.to_device(y.astype(np.int32))
            recip_full_d = cuda.to_device(recip_full)
            if self.verbose and self.use_start:
                print("Running MultiSURF* on the GPU now...")
            elif self.verbose:
                print("Running MultiSURF on the GPU now...")
            scores = _multisurf_gpu_host_caller(
                x_d, y_d, recip_full_d, all_feature_indices, self.use_star, is_discrete
            )
        else:
            if self.verbose and self.use_start:
                print("Running MultiSURF* on the CPU now...")
            elif self.verbose:
                print("Running MultiSURF on the CPU now...")
            scores = _multisurf_cpu_host_caller(
                x, y, recip_full, all_feature_indices, self.use_star, is_discrete, self.n_jobs
            )

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][: self.n_features_to_select]

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Reduces x to the selected features.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns
        -------
        x_new : ndarray of shape (n_samples, n_features_to_select)
            The input samples with only the selected features.
        """
        if np.any(np.isnan(x)):
            raise ValueError("Input data contains NaN values. Please handle missing data before passing it.")
        check_is_fitted(self)

        x = check_array(x, dtype=np.float32)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x has {x.shape[1]} features, but {self.__class__.__name__} "
                f"was trained with {self.n_features_in_} features."
            )

        return x[:, self.top_features_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        A convenience method that fits the model and applies the transformation
        to the same data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        x_new : ndarray of shape (n_samples, n_features_to_select)
            The transformed input samples.
        """
        self.fit(x, y)
        return self.transform(x)
