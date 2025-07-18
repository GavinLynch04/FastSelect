from __future__ import annotations
import numpy as np
from numba import cuda, float32, int32, njit, prange, set_num_threads, get_num_threads, config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y, validate_data
import time
import warnings

TPB = 64  # Threads‑per‑block for CUDA kernels, potentially expose to user

@cuda.jit
def _relieff_gpu_kernel(x, y, recip_full, is_discrete, k_neighbors, scores_out):
    """ReliefF scoring on the GPU (k ≤ 10).
    Multiclass weighting is still delegated to CPU path for simplicity.
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x     # one sample per block
    tid = cuda.threadIdx.x

    # local neighbour buffers (k ≤ 10)
    local_hit_d = cuda.local.array(10, float32)
    local_hit_i = cuda.local.array(10, int32)
    local_mis_d = cuda.local.array(10, float32)
    local_mis_i = cuda.local.array(10, int32)

    for k in range(k_neighbors):
        local_hit_d[k] = 3.4e38
        local_mis_d[k] = 3.4e38

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

        if y[i] == y[j]:  # hit
            for k in range(k_neighbors - 1, -1, -1):
                if dist < local_hit_d[k]:
                    if k < k_neighbors - 1:
                        local_hit_d[k + 1] = local_hit_d[k]
                        local_hit_i[k + 1] = local_hit_i[k]
                    local_hit_d[k] = dist
                    local_hit_i[k] = j
                else:
                    break
        else:  # miss
            for k in range(k_neighbors - 1, -1, -1):
                if dist < local_mis_d[k]:
                    if k < k_neighbors - 1:
                        local_mis_d[k + 1] = local_mis_d[k]
                        local_mis_i[k + 1] = local_mis_i[k]
                    local_mis_d[k] = dist
                    local_mis_i[k] = j
                else:
                    break
    cuda.syncthreads()

    if tid == 0:
        for f in range(n_features):
            hit_sum = 0.0
            miss_sum = 0.0
            for k in range(k_neighbors):
                h = local_hit_i[k]
                m = local_mis_i[k]
                if is_discrete[f]:
                    hit_sum += 1.0 if x[i, f] != x[h, f] else 0.0
                    miss_sum += 1.0 if x[i, f] != x[m, f] else 0.0
                else:
                    hit_sum += abs(x[i, f] - x[h, f]) * recip_full[f]
                    miss_sum += abs(x[i, f] - x[m, f]) * recip_full[f]

            update = (miss_sum - hit_sum) / k_neighbors
            cuda.atomic.add(scores_out, f, update)


def _relieff_gpu_host_caller(x_d, y_d, recip_full_d, is_discrete_d, k):
    """Launch the GPU kernel and collect scores."""
    n_samples, n_features = x_d.shape
    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0
    _relieff_gpu_kernel[n_samples, TPB](x_d, y_d, recip_full_d, is_discrete_d, k, scores_d)
    cuda.synchronize()
    return scores_d.copy_to_host() / n_samples


@njit(parallel=True, fastmath=True)
def _relieff_cpu_kernel(x, y_enc, recip_full, is_discrete, k, class_probs, scores_out):
    n_samples, n_features = x.shape
    n_classes = class_probs.shape[0]
    temp = np.zeros((n_samples, n_features), dtype=np.float32)

    for i in prange(n_samples):
        dists = np.empty(n_samples, dtype=np.float32)
        for j in range(n_samples):
            if i == j:
                dists[j] = np.inf
                continue
            d = 0.0
            for f in range(n_features):
                if is_discrete[f]:
                    d += 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    d += abs(x[i, f] - x[j, f]) * recip_full[f]
            dists[j] = d
        order = np.argsort(dists)
        lbl_i = y_enc[i]
        hits = np.empty(k, dtype=np.int32)
        misses = np.empty((n_classes, k), dtype=np.int32)
        h_found = 0
        m_found = np.zeros(n_classes, dtype=np.int32)
        for idx in order:
            lbl = y_enc[idx]
            if lbl == lbl_i:
                if h_found < k:
                    hits[h_found] = idx
                    h_found += 1
            else:
                if m_found[lbl] < k:
                    misses[lbl, m_found[lbl]] = idx
                    m_found[lbl] += 1
            if h_found == k and (m_found >= k).all(): break
        denom = 1.0 - class_probs[lbl_i]
        for f in range(n_features):
            h_sum = 0.0
            for ki in range(h_found):
                j = hits[ki]
                if is_discrete[f]:
                    h_sum += 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    h_sum += abs(x[i, f] - x[j, f]) * recip_full[f]
            h_avg = h_sum / k
            m_total = 0.0
            for c in range(n_classes):
                if c == lbl_i or m_found[c] == 0: continue
                weight = class_probs[c] / denom
                m_sum = 0.0
                for ki in range(m_found[c]):
                    j = misses[c, ki]
                    if is_discrete[f]:
                        m_sum += 1.0 if x[i, f] != x[j, f] else 0.0
                    else:
                        m_sum += abs(x[i, f] - x[j, f]) * recip_full[f]
                m_avg = m_sum / k
                m_total += weight * m_avg
            temp[i, f] = m_total - h_avg

    for f in range(n_features):
        scores_out[f] = temp[:, f].sum()

def _relieff_cpu_host_caller(x, y_enc, recip_full, is_discrete, k, class_probs, n_jobs, verbose):
    n_samples, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)
    
    num_threads_to_set = config.NUMBA_NUM_THREADS if n_jobs == -1 else n_jobs
    
    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)
    
    try:
        _relieff_cpu_kernel(x, y_enc, recip_full, is_discrete, k, class_probs, scores)
    finally:
        set_num_threads(original_num_threads)
        
    return scores / n_samples


class ReliefF(BaseEstimator, TransformerMixin):
    """GPU and CPU-accelerated feature selection using the ReliefF algorithm.

    This estimator provides a unified API for running ReliefF on either
    a CPU (using Numba's parallel JIT) or a GPU (using Numba CUDA).

    Parameters
    ----------
    n_features_to_select : int, default=10
        The number of top features to select.

    discrete_limit : int, default=10
        The limit for the number of independent feature values to be considered
        discrete or continuous (affects distance calculation).

    n_neighbors : int, default=10
        The number of nearest neighbors to use for score calculation.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend to use.
        
    verbose : bool, default=True
        Controls whether progress updates are printed during the fit. Only avaliable if backend='cpu'.
        
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
        discrete_limit: int = 10,
        n_neighbors: int = 10,
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


    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Calculates feature importances using the ReliefF algorithm on a GPU/CPU.

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
        x, y = validate_data(
            self, x, y,
        )
        
        if self.backend not in ["auto", "gpu", "cpu"]:
            raise ValueError("backend must be one of 'auto', 'gpu', or 'cpu'")
             
        x, y = check_X_y(x, y, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = x.shape[1]
        n_samples = x.shape[0]
        
        if not (0 < self.n_neighbors < n_samples):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be an integer "
                f"between 1 and n_samples - 1 ({n_samples - 1})."
            )
            
        _, y_encoded = np.unique(y, return_inverse=True)
        min_class_size = np.min(np.bincount(y_encoded))
        if self.n_neighbors >= min_class_size:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than or equal to the "
                f"smallest class size ({min_class_size}). The number of hits found "
                f"for each sample will be capped by (class_size - 1).",
                UserWarning
            )

        # Determine discrete features
        is_discrete = np.zeros(self.n_features_in_, dtype=np.bool_)
        self.is_discrete_ = is_discrete
        for f in range(self.n_features_in_):
            if np.unique(x[:, f]).size <= self.discrete_limit:
                is_discrete[f] = True

        # Class encoding and probabilities
        class_labels, class_counts = np.unique(y, return_counts=True)
        class_probs = class_counts / len(y)
        y_enc = np.searchsorted(class_labels, y)

        # Feature range reciprocals (continuous only)
        feature_ranges = x.max(axis=0) - x.min(axis=0)
        feature_ranges[is_discrete] = 1.0
        feature_ranges[feature_ranges == 0] = 1.0  # avoid div/0
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        # Select backend
        if self.backend == "auto":
            self.effective_backend_ = "gpu" if cuda.is_available() else "cpu"
        else:
            if self.backend == "gpu" and not cuda.is_available():
                    raise RuntimeError("CUDA GPU not detected. Install NVIDIA CUDA toolkit or "
                       "use `backend='cpu'` instead.")
            self.effective_backend_ = self.backend

        if self.effective_backend_ == "gpu":
            x_d = cuda.to_device(x.astype(np.float32))
            y_d = cuda.to_device(y_enc.astype(np.int32))
            recip_d = cuda.to_device(recip_full)
            is_discrete_d = cuda.to_device(is_discrete.astype(np.bool_))
            if self.verbose:
                print("Running ReliefF on the GPU now...")
            scores = _relieff_gpu_host_caller(
                x_d, y_d, recip_d, is_discrete_d, self.n_neighbors)
        else:
            if self.verbose:
                print("Running ReliefF on the CPU now...")
            scores = _relieff_cpu_host_caller(
                x.astype(np.float32), y_enc.astype(np.int32), recip_full,
                is_discrete, self.n_neighbors, class_probs.astype(np.float32),
                self.n_jobs, self.verbose)

        self.feature_importances_ = scores
        n_select = min(self.n_features_to_select, self.n_features_in_)
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
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
        x = validate_data(
            self, x,
        )
        check_is_fitted(self)
        x = check_array(x, dtype=np.float32)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x has {x.shape[1]} features, but "
                + "was trained with {self.n_features_in_}."
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