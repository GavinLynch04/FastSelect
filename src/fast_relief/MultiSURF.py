from __future__ import annotations
import math
import numpy as np
from numba import cuda, float32, int32, njit, prange,set_num_threads
import warnings
from numba.core.errors import NumbaPerformanceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

TPB = 64              # Threads Per Block
MAX_F_TILE = 1024     # Features loaded per shared-memory tile

@cuda.jit
def _multisurf_gpu_kernel(X, y, recip_full, feat_idx, n_kept, use_star, scores_out):
    """MultiSURF scoring for an _arbitrary subset_ of features."""
    n_samples = X.shape[0]
    i = cuda.blockIdx.x          # focal sample index
    tid = cuda.threadIdx.x
    # Shared scratch
    hits_tile = cuda.shared.array(shape=MAX_F_TILE, dtype=float32)
    miss_tile = cuda.shared.array(shape=MAX_F_TILE, dtype=float32)
    sh_red_f32 = cuda.shared.array(shape=TPB, dtype=float32)
    sh_red_i32 = cuda.shared.array(shape=TPB, dtype=int32)

    sum_d = 0.0
    sum_d2 = 0.0
    for j in range(tid, n_samples, TPB):
        if j == i:
            continue
        dist = 0.0
        for f0 in range(0, n_kept, MAX_F_TILE):
            tile_len = min(MAX_F_TILE, n_kept - f0)
            for f in range(tile_len):
                full_idx = feat_idx[f0 + f]
                diff = X[i, full_idx] - X[j, full_idx]
                dist += abs(diff) * recip_full[full_idx]
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

    for f0 in range(0, n_kept, MAX_F_TILE):
        tile_len = min(MAX_F_TILE, n_kept - f0)
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
                diff = X[i, full_idx] - X[j, full_idx]
                dist += abs(diff) * recip_full[full_idx]
            is_hit = y[i] == y[j]
            if dist < thresh:  # This is a NEAR neighbor
                for f in range(tile_len):
                    full_idx = feat_idx[f0 + f]
                    diff = abs(X[i, full_idx] - X[j, full_idx]) * recip_full[full_idx]
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
                    diff = abs(X[i, full_idx] - X[j, full_idx]) * recip_full[full_idx]
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


def _compute_ranges(X: np.ndarray) -> np.ndarray:
    """Helper to compute feature ranges on the CPU."""
    ranges = (X.max(axis=0) - X.min(axis=0)).astype(np.float32)
    return ranges

def _multisurf_gpu_host_caller(X_d, y_d, recip_full_d, feat_idx: np.ndarray, use_star: bool) -> np.ndarray:
    """Host helper function that launches the kernel and returns scores."""
    n_samples, _ = X_d.shape
    n_kept = feat_idx.size
    feat_idx_d = cuda.to_device(feat_idx.astype(np.int64, copy=False))
    scores_d = cuda.device_array(n_kept, dtype=np.float32)
    scores_d[:] = 0.0  # Zero-fill on device
    
    _multisurf_gpu_kernel[n_samples, TPB](
        X_d, y_d, recip_full_d, feat_idx_d, n_kept, use_star, scores_d
    )
    cuda.synchronize()
    
    return scores_d.copy_to_host() / n_samples


@njit(parallel=True, fastmath=True)
def _multisurf_cpu_kernel(X, y, recip_full, feat_idx, n_kept, use_star, scores_out):
    """
    Optimized MultiSURF scoring for CPU.

    This version avoids large intermediate allocations inside the parallel loop
    by recalculating distances in a second pass, which is faster due to
    Numba's compilation and reduced memory pressure.
    """
    n_samples = X.shape[0]

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
                diff = X[i, f] - X[j, f]
                dist += abs(diff) * recip_full[f]

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
                diff = X[i, f] - X[j, f]
                dist += abs(diff) * recip_full[f]

            is_hit = (y[i] == y[j])
            if dist < thresh:  # NEAR neighbor
                if is_hit:
                    n_hits += 1
                    for k in range(n_kept):
                        f = feat_idx[k]
                        diff = abs(X[i, f] - X[j, f]) * recip_full[f]
                        hit_diffs[k] += diff
                else:
                    n_miss += 1
                    for k in range(n_kept):
                        f = feat_idx[k]
                        diff = abs(X[i, f] - X[j, f]) * recip_full[f]
                        miss_diffs[k] += diff
            elif use_star and not is_hit:  # FAR MISS
                for k in range(n_kept):
                    f = feat_idx[k]
                    diff = abs(X[i, f] - X[j, f]) * recip_full[f]
                    miss_diffs[k] -= diff  # Subtract the contribution

        if n_hits > 0:
            hit_diffs /= n_hits
        if n_miss > 0:
            miss_diffs /= n_miss

        for k in range(n_kept):
            temp_scores[i, k] = miss_diffs[k] - hit_diffs[k]
    for k in range(n_kept):
        scores_out[k] = temp_scores[:, k].sum()



def _multisurf_cpu_host_caller(X, y, recip_full, feat_idx, use_star):
    """Host caller for the optimized CPU kernel."""
    n_kept = feat_idx.size
    scores = np.zeros(n_kept, dtype=np.float32)
    # Call the new optimized kernel
    _multisurf_cpu_kernel(X, y, recip_full, feat_idx, n_kept, use_star, scores)
    return scores / X.shape[0]

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

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.

    feature_importances_ : ndarray of shape (n_features,)
        The calculated importance scores for each feature.
    
    effective_backend_ : str
        The backend that was actually used during `fit` ('gpu' or 'cpu').
    """
    def __init__(self, n_features_to_select: int = 10, backend: str = 'auto', use_star: bool = False):
        self.n_features_to_select = n_features_to_select
        self.backend = backend
        self.use_star = use_star

        if self.backend not in ['auto', 'gpu', 'cpu']:
            raise ValueError("backend must be one of 'auto', 'gpu', or 'cpu'")
          
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Calculates feature importances using the MultiSURF algorithm on a GPU/CPU.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y, dtype=np.float32, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
      
        if self.backend == 'auto':
            if cuda.is_available():
                self.effective_backend_ = 'gpu'
            else:
                self.effective_backend_ = 'cpu'
        elif self.backend == 'gpu':
            if not cuda.is_available():
                raise RuntimeError("backend='gpu' was selected, but no compatible "
                                   "NVIDIA GPU was found or CUDA toolkit is not installed.")
            self.effective_backend_ = 'gpu'
        else:
            self.effective_backend_ = 'cpu'
          
        feature_ranges = _compute_ranges(X)

        feature_ranges[feature_ranges == 0] = 1
        recip_full = (1.0 / feature_ranges).astype(np.float32)

        all_feature_indices = np.arange(self.n_features_in_, dtype=np.int64)

        if self.effective_backend_ == 'gpu':
            X_d = cuda.to_device(X)
            y_d = cuda.to_device(y.astype(np.int32))
            recip_full_d = cuda.to_device(recip_full)
            
            scores = _multisurf_gpu_host_caller(
                X_d, y_d, recip_full_d, all_feature_indices, self.use_star
            )
        else:
            scores = _multisurf_cpu_host_caller(
                X, y, recip_full, all_feature_indices, self.use_star
            )

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:self.n_features_to_select]
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduces X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_to_select)
            The input samples with only the selected features.
        """
        check_is_fitted(self)

        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                             f"was trained with {self.n_features_in_} features.")

        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        A convenience method that fits the model and applies the transformation
        to the same data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_to_select)
            The transformed input samples.
        """
        self.fit(X, y)
        return self.transform(X)
