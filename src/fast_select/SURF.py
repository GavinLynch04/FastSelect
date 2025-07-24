from __future__ import annotations
import numpy as np
from numba import cuda, float32, int32, njit, prange, config, get_num_threads, set_num_threads, get_thread_id
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

TPB = 64  # Threads Per Block

@cuda.jit
def _surf_gpu_kernel(x, y, recip_full, use_star, is_discrete, scores_out): # pragma: no cover
    """
    SURF/SURF* scoring on the GPU for all features.
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    sh_dist_sum = cuda.shared.array(shape=TPB, dtype=float32)
    sh_near_hit = cuda.shared.array(shape=1, dtype=float32)
    sh_near_miss = cuda.shared.array(shape=1, dtype=float32)
    sh_far_hit = cuda.shared.array(shape=1, dtype=float32)
    sh_far_miss = cuda.shared.array(shape=1, dtype=float32)

    local_sum_d = 0.0
    for j in range(tid, n_samples, TPB):
        if i == j:
            continue

        dist = 0.0
        for f in range(n_features):
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f]) * recip_full[f]
            dist += diff
        local_sum_d += dist

    sh_dist_sum[tid] = local_sum_d
    cuda.syncthreads()

    off = TPB // 2
    while off > 0:
        if tid < off:
            sh_dist_sum[tid] += sh_dist_sum[tid + off]
        cuda.syncthreads()
        off //= 2

    if tid == 0:
        if n_samples > 1:
            avg_dist_val = sh_dist_sum[0] / (n_samples - 1)
        else:
            avg_dist_val = 0.0
        sh_dist_sum[0] = avg_dist_val  # Reuse shared memory for broadcast

    cuda.syncthreads()
    avg_dist = sh_dist_sum[0]  # All threads read the broadcasted value

    for f in range(n_features):
        if tid == 0:
            sh_near_hit[0] = 0.0
            sh_near_miss[0] = 0.0
            if use_star:
                sh_far_hit[0] = 0.0
                sh_far_miss[0] = 0.0
        cuda.syncthreads()

        local_near_hit_sum = 0.0
        local_near_miss_sum = 0.0
        local_far_hit_sum = 0.0
        local_far_miss_sum = 0.0

        for j in range(tid, n_samples, TPB):
            if i == j:
                continue

            dist = 0.0
            for f_inner in range(n_features):
                if is_discrete[f_inner]:
                    diff = 1.0 if x[i, f_inner] != x[j, f_inner] else 0.0
                else:
                    diff = abs(x[i, f_inner] - x[j, f_inner]) * recip_full[f_inner]
                dist += diff

            is_hit = (y[i] == y[j])
            is_near = (dist < avg_dist)

            if is_discrete[f]:
                feat_diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                feat_diff = abs(x[i, f] - x[j, f]) * recip_full[f]

            if is_near:
                if is_hit:
                    local_near_hit_sum += feat_diff
                else:
                    local_near_miss_sum += feat_diff
            elif use_star:
                if is_hit:
                    local_far_hit_sum += feat_diff
                else:
                    local_far_miss_sum += feat_diff

        cuda.atomic.add(sh_near_hit, 0, local_near_hit_sum)
        cuda.atomic.add(sh_near_miss, 0, local_near_miss_sum)
        if use_star:
            cuda.atomic.add(sh_far_hit, 0, local_far_hit_sum)
            cuda.atomic.add(sh_far_miss, 0, local_far_miss_sum)
        cuda.syncthreads()

        if tid == 0:
            score_update = sh_near_miss[0] - sh_near_hit[0]
            if use_star:
                score_update += sh_far_hit[0] - sh_far_miss[0]

            cuda.atomic.add(scores_out, f, score_update)

def _surf_gpu_host_caller(x_d, y_d, recip_full_d, use_star, is_discrete_d):
    """Host helper function that launches the kernel and returns scores."""
    n_samples, n_features = x_d.shape
    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0

    _surf_gpu_kernel[n_samples, TPB](
        x_d, y_d, recip_full_d, use_star, is_discrete_d, scores_d
    )
    cuda.synchronize()

    return scores_d.copy_to_host() / n_samples


@njit(parallel=True, fastmath=True)
def _surf_cpu_kernel(x, y, recip_full, use_star, is_discrete, private_scores): # pragma: no cover
    """
    SURF/SURF* scoring for CPU.
    """
    n_samples, n_features = x.shape
    n_threads = private_scores.shape[0]

    for i in prange(n_samples):
        tid = get_thread_id()
        
        dists_from_i = np.empty(n_samples, dtype=np.float32)
        
        diffs_from_i = np.empty((n_samples, n_features), dtype=np.float32)

        for j in range(n_samples):
            if i == j:
                dists_from_i[j] = 0.0
                continue
                
            dist_ij = 0.0
            for f in range(n_features):
                if is_discrete[f]:
                    feat_diff = 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    feat_diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                
                diffs_from_i[j, f] = feat_diff
                dist_ij += feat_diff
            dists_from_i[j] = dist_ij
        
        sum_d = np.sum(dists_from_i)
        avg_dist = sum_d / (n_samples - 1)
        
        near_hit_sum = np.zeros(n_features, dtype=np.float32)
        near_miss_sum = np.zeros(n_features, dtype=np.float32)
        far_hit_sum = np.zeros(n_features, dtype=np.float32)
        far_miss_sum = np.zeros(n_features, dtype=np.float32)

        for j in range(n_samples):
            if i == j:
                continue

            dist_ij = dists_from_i[j]
            is_hit = (y[i] == y[j])
            is_near = (dist_ij < avg_dist)

            feat_diff_array = diffs_from_i[j]

            if is_near:
                if is_hit:
                    near_hit_sum += feat_diff_array
                else:
                    near_miss_sum += feat_diff_array
            elif use_star:
                if is_hit:
                    far_hit_sum += feat_diff_array
                else:
                    far_miss_sum += feat_diff_array
        
        score_update = (near_miss_sum - near_hit_sum)
        if use_star:
            score_update += (far_hit_sum - far_miss_sum)
        
        private_scores[tid] += score_update


def _surf_cpu_host_caller(x, y, recip_full, use_star, is_discrete, n_jobs):
    """
    Host caller for the CPU kernel.
    Manages thread setup and final reduction of scores.
    """
    n_samples, n_features = x.shape
    
    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs

    private_scores = np.zeros((num_threads_to_set, n_features), dtype=np.float32)
    
    original_num_threads = get_num_threads()
    try:
        set_num_threads(num_threads_to_set)
        _surf_cpu_kernel(x, y, recip_full, use_star, is_discrete, private_scores)
    finally:
        set_num_threads(original_num_threads)
        
    final_scores = private_scores.sum(axis=0)
    
    return final_scores / n_samples

class SURF(TransformerMixin, BaseEstimator):
    """GPU and CPU-accelerated feature selection using the SURF algorithm.

    This estimator provides a unified, scikit-learn compatible API for running
    SURF or SURF* on either a CPU or a GPU. The implementation is designed
    for performance and scalability, avoiding the memory bottlenecks of
    older implementations by calculating distances on-the-fly.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select.
        - If an int, the exact number of features to select.
        - If a float between (0, 1], the percentage of features to select.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend to use. 'auto' will use a GPU if available.

    use_star : bool, default=False
        If True, runs the SURF* algorithm, which includes updates from
        "far" neighbors. If False (default), runs the standard SURF algorithm.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    n_jobs : int, default=-1
        Number of CPU threads to use for the 'cpu' backend. -1 means all.
        This parameter is ignored for the 'gpu' backend.

    verbose : bool, default=False
        Controls whether to print progress messages during fit.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.

    feature_importances_ : ndarray of shape (n_features,)
        The calculated importance scores for each feature.

    top_features_ : ndarray of shape (n_features_to_select,)
        The indices of the selected top features.

    effective_backend_ : str
        The backend that was actually used during `fit` ('gpu' or 'cpu').
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
        
    def _validate_parameters(self, n_samples, n_features):
        """Validate all user-provided parameters."""
        # Backend check
        if self.backend not in ["auto", "gpu", "cpu"]:
            raise ValueError("backend must be one of 'auto', 'gpu', or 'cpu'")

        # Sample count check
        if n_samples < 2:
            raise ValueError(
                f"SURF requires at least 2 samples, but got n_samples = {n_samples}"
            )

        # n_features_to_select check (handles both int and float)
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
        """
        Calculates feature importances using the SURF or SURF* algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. NaN values are not supported.
        y : array-like of shape (n_samples,)
            The target values (class labels). Must be numeric.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(
            self, X, y, y_numeric=True, dtype=np.float64, ensure_2d=True,
        )
            
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        
        n_select = self._validate_parameters(n_samples, self.n_features_in_)

        if self.backend == "auto":
            self.effective_backend_ = "gpu" if cuda.is_available() else "cpu"
        elif self.backend == "gpu" and not cuda.is_available():
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
            X_d = cuda.to_device(X)
            y_d = cuda.to_device(y.astype(np.int32))
            recip_full_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(self.is_discrete_)
            scores = _surf_gpu_host_caller(
                X_d, y_d, recip_full_d, self.use_star, is_discrete_d
            )
        else:  # CPU
            scores = _surf_cpu_host_caller(
                X, y.astype(np.int32), recip_full, self.use_star, self.is_discrete_, self.n_jobs
            )

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]

        if self.verbose:
            print("Feature scoring completed.")

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
        check_is_fitted(self)
        x = validate_data(
            self, x,
            reset=False,
            dtype=[np.float64, np.float32]
        )

        return x[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        self.fit(X, y)
        return self.transform(X)