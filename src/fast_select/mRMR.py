from __future__ import annotations
import numpy as np
from math import log, ceil
from numba import njit, prange, float32, int32, int64, cuda
from numba.types import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
import time

@njit(parallel=True, cache=True)
def _encode_data_numba(X, y, unique_vals): # pragma: no cover
    """
    Encodes X and y using a precomputed sorted array of unique values.
    This is dramatically faster than np.vectorize.
    """
    n_samples, n_features = X.shape
    X_encoded = np.empty_like(X)
    y_encoded = np.empty_like(y)

    # Parallelize the encoding of X
    for i in prange(n_features):
        for j in range(n_samples):
            X_encoded[j, i] = np.searchsorted(unique_vals, X[j, i])

    for i in range(n_samples):
        y_encoded[i] = np.searchsorted(unique_vals, y[i])

    return X_encoded, y_encoded


@njit(cache=True, nogil=True)
def _calculate_mi(x1, n_states1, x2, n_states2, n_samples): # pragma: no cover
    """Calculates Mutual Information between two discrete vectors."""
    contingency_table = np.zeros((n_states1, n_states2), dtype=np.float32)
    for i in range(n_samples):
        # Ensure indices are integers
        contingency_table[int(x1[i]), int(x2[i])] += 1
    
    contingency_table /= n_samples
    
    p1 = np.sum(contingency_table, axis=1)
    p2 = np.sum(contingency_table, axis=0)
    
    mi = 0.0
    for i in range(n_states1):
        for j in range(n_states2):
            p_xy = contingency_table[i, j]
            p_x = p1[i]
            p_y = p2[j]
            # Use a small epsilon to avoid log(0)
            if p_xy > 1e-12 and p_x > 1e-12 and p_y > 1e-12:
                mi += p_xy * log(p_xy / (p_x * p_y))
    return mi

@njit(parallel=True, cache=True)
def _precompute_mi_matrices(X, y): # pragma: no cover
    """Precomputes relevance and redundancy matrices."""
    n_samples, n_features = X.shape
    
    n_states_X = np.zeros(n_features, dtype=np.int32)
    for i in prange(n_features):
        n_states_X[i] = int(np.max(X[:, i])) + 1
        
    n_states_y = int(np.max(y)) + 1
    
    relevance_scores = np.zeros(n_features, dtype=np.float32)
    for i in prange(n_features):
        relevance_scores[i] = _calculate_mi(
            X[:, i], n_states_X[i], y, n_states_y, n_samples
        )
        
    redundancy_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    for i in prange(n_features):
        for j in range(i + 1, n_features):
            mi = _calculate_mi(
                X[:, i], n_states_X[i], X[:, j], n_states_X[j], n_samples
            )
            redundancy_matrix[i, j] = mi
            redundancy_matrix[j, i] = mi
            
    return relevance_scores, redundancy_matrix

MAX_SHARED_STATES = 32
# Define a standard, efficient thread block size.
THREADS_PER_BLOCK = (16, 16)

@cuda.jit
def _relevance_kernel_gpu(X_gpu, y_gpu, relevance_out, n_samples, n_states): # pragma: no cover
    """
    Specialized CUDA kernel for calculating relevance scores I(f; y).
    Each CUDA block calculates the MI for one feature.
    """
    shared_contingency = cuda.shared.array(shape=(MAX_SHARED_STATES, MAX_SHARED_STATES), dtype=float32)

    feature_idx = cuda.blockIdx.x

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if tx < n_states and ty < n_states:
        shared_contingency[tx, ty] = 0.0
    cuda.syncthreads()

    for sample_idx in range(n_samples):
        val1 = int(X_gpu[sample_idx, feature_idx])
        val2 = int(y_gpu[sample_idx])
        cuda.atomic.add(shared_contingency, (val1, val2), 1.0)
    cuda.syncthreads()

    if tx == 0 and ty == 0:
        for r in range(n_states):
            for c in range(n_states):
                shared_contingency[r,c] /= n_samples

        p_x = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        p_y = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        for i in range(n_states):
            p_x[i] = 0.0
            p_y[i] = 0.0

        for r in range(n_states):
            for c in range(n_states):
                p_x[r] += shared_contingency[r, c]
                p_y[c] += shared_contingency[r, c]

        mi = 0.0
        for r in range(n_states):
            for c in range(n_states):
                p_xy_val = shared_contingency[r, c]
                p_x_r = p_x[r]
                p_y_c = p_y[c]
                if p_xy_val > 1e-12:
                    mi += p_xy_val * log(p_xy_val / (p_x_r * p_y_c))
        
        relevance_out[feature_idx] = mi

@cuda.jit
def _redundancy_kernel_gpu(X_gpu, redundancy_out, n_features, n_samples, n_states): # pragma: no cover
    """
    Specialized CUDA kernel for calculating redundancy matrix I(f_i; f_j).
    Each CUDA block calculates the MI for one pair of features (f_i, f_j).
    """
    shared_contingency = cuda.shared.array(shape=(MAX_SHARED_STATES, MAX_SHARED_STATES), dtype=float32)
    
    f1_idx = cuda.blockIdx.x
    f2_idx = cuda.blockIdx.y

    if f2_idx <= f1_idx:
        return

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    if tx < n_states and ty < n_states:
        shared_contingency[tx, ty] = 0.0
    cuda.syncthreads()

    for sample_idx in range(n_samples):
        val1 = int(X_gpu[sample_idx, f1_idx])
        val2 = int(X_gpu[sample_idx, f2_idx])
        cuda.atomic.add(shared_contingency, (val1, val2), 1.0)
    cuda.syncthreads()

    if tx == 0 and ty == 0:
        for r in range(n_states):
            for c in range(n_states):
                shared_contingency[r,c] /= n_samples

        p_x = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        p_y = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        for i in range(n_states):
            p_x[i] = 0.0
            p_y[i] = 0.0

        for r in range(n_states):
            for c in range(n_states):
                p_x[r] += shared_contingency[r, c]
                p_y[c] += shared_contingency[r, c]

        mi = 0.0
        for r in range(n_states):
            for c in range(n_states):
                p_xy_val = shared_contingency[r, c]
                p_x_r = p_x[r]
                p_y_c = p_y[c]
                if p_xy_val > 1e-12:
                    mi += p_xy_val * log(p_xy_val / (p_x_r * p_y_c))

        redundancy_out[f1_idx, f2_idx] = mi
        redundancy_out[f2_idx, f1_idx] = mi

def _precompute_mi_matrices_gpu(X, y, n_states):
    """Orchestrator for the GPU backend."""
    n_samples, n_features = X.shape

    if n_states > MAX_SHARED_STATES:
        raise ValueError(
            f"GPU backend supports a maximum of {MAX_SHARED_STATES} unique discrete states, "
            f"but data has {n_states}."
        )

    X_gpu = cuda.to_device(np.ascontiguousarray(X))
    y_gpu = cuda.to_device(np.ascontiguousarray(y))

    relevance_gpu = cuda.device_array(n_features, dtype=np.float32)
    redundancy_gpu = cuda.device_array((n_features, n_features), dtype=np.float32)

    blocks_per_grid_rel = (n_features,)
    _relevance_kernel_gpu[blocks_per_grid_rel, THREADS_PER_BLOCK](
        X_gpu, y_gpu, relevance_gpu, n_samples, n_states
    )

    blocks_per_grid_red = (n_features, n_features)
    _redundancy_kernel_gpu[blocks_per_grid_red, THREADS_PER_BLOCK](
        X_gpu, redundancy_gpu, n_features, n_samples, n_states
    )

    cuda.synchronize()

    return relevance_gpu.copy_to_host(), redundancy_gpu.copy_to_host()

class mRMR(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible feature selector based on the mRMR algorithm.
    
    This implementation is designed for discrete data and uses Numba for
    high-performance computation of mutual information matrices.
    
    Parameters
    ----------
    n_features_to_select : int
        The number of top features to select.
        
    method : {'MID', 'MIQ'}, default='MID'
        The mRMR selection criterion to use.
        - 'MID' (Mutual Information Difference): f_score = I(f; y) - mean(I(f; S))
        - 'MIQ' (Mutual Information Quotient): f_score = I(f; y) / mean(I(f; S))
        
    backend : {'cpu', 'gpu'}, default='cpu'
        The computational backend to use. 'gpu' requires a compatible NVIDIA GPU
        and Numba with CUDA support installed.
        
    """
    def __init__(self, n_features_to_select: int, method: str = 'MID', backend: str = 'cpu'):
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.backend = backend
        if self.method not in ['MID', 'MIQ']:
            raise ValueError("Method must be either 'MID' or 'MIQ'.")
        if self.backend not in ['cpu', 'gpu']:
            raise ValueError("Backend must be either 'cpu' or 'gpu'.")
        if self.backend == 'gpu' and not cuda.is_available():
            raise RuntimeError(
                "GPU backend was selected, but Numba could not find a usable CUDA installation. "
                "Please ensure you have an NVIDIA GPU with the latest drivers and a compatible CUDA toolkit."
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the mRMR model to select the best features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Assumed to be discrete.
        y : array-like of shape (n_samples,)
            The target values. Assumed to be discrete.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y, dtype=None, y_numeric=True, ensure_2d=True,)
        self.n_features_in_ = X.shape[1]

        if not (0 < self.n_features_to_select <= self.n_features_in_):
            raise ValueError(
                "n_features_to_select must be a positive integer less "
                "than or equal to the number of features."
            )
        start = time.time()
        unique_vals = np.unique(np.concatenate([np.unique(X), np.unique(y)]))
        self.unique_vals_ = unique_vals
        n_states = len(unique_vals)
        end = time.time()
        print(end-start)
        start = time.time()
        X_encoded, y_encoded = _encode_data_numba(X, y, unique_vals)
        end = time.time()
        print(end-start)
        
        start = time.time()
        if self.backend == 'cpu':
            relevance, redundancy = _precompute_mi_matrices_cpu(X_encoded, y_encoded)
        elif self.backend == 'gpu':
            relevance, redundancy = _precompute_mi_matrices_gpu(X_encoded, y_encoded, n_states)
        end = time.time()
        print(end-start)
        self.relevance_scores_ = relevance
        self.redundancy_matrix_ = redundancy

        
        start = time.time()
        selected_indices = np.zeros(self.n_features_to_select, dtype=np.int32)
        remaining_mask = np.ones(self.n_features_in_, dtype=bool)

        first_idx = np.argmax(self.relevance_scores_)
        selected_indices[0] = first_idx
        remaining_mask[first_idx] = False

        redundancy_sum = self.redundancy_matrix_[:, first_idx].copy()

        for i in range(1, self.n_features_to_select):
            remaining_indices_arr = np.where(remaining_mask)[0]

            if self.method == 'MID':
                scores = self.relevance_scores_[remaining_indices_arr] - (redundancy_sum[remaining_indices_arr] / i)
            else: # 'MIQ'
                scores = self.relevance_scores_[remaining_indices_arr] / ((redundancy_sum[remaining_indices_arr] / i) + 1e-9)

            best_remaining_local_idx = np.argmax(scores)
            best_feature_idx = remaining_indices_arr[best_remaining_local_idx]

            selected_indices[i] = best_feature_idx
            remaining_mask[best_feature_idx] = False

            redundancy_sum += self.redundancy_matrix_[:, best_feature_idx]

        self.top_features_ = selected_indices
        self.feature_importances_ = self.relevance_scores_
        end = time.time()
        print(end-start)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to the selected features."""
        check_is_fitted(self)
        X = validate_data(
            self, X,
            reset=False,
            ensure_2d=True,
            dtype=None
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but was trained with {self.n_features_in_}."
            )
        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)


import numpy as np
from numba import jit, prange
from math import log
from sklearn.feature_selection import chi2


@jit(nopython=True, parallel=True, cache=True)
def _precompute_redundancy_matrix(X):
    """JIT kernel to pre-compute only the feature-feature MI matrix."""
    n_samples, n_features = X.shape
    redundancy_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    for i in prange(n_features):
        for j in range(i + 1, n_features):
            # Assumes _calculate_mi is defined as before
            mi = _calculate_mi(X[:, i], X[:, j], n_samples)
            redundancy_matrix[i, j] = mi
            redundancy_matrix[j, i] = mi
    return redundancy_matrix

def fs_chi2_rmr(X, y, k):
    """
    Selects top k features using a hybrid Chi2-RMR algorithm.

    This custom method uses the chi2 statistic for relevance (feature-target)
    and Mutual Information for redundancy (feature-feature), combining the
    strengths of both approaches.
    """

    n_samples, n_features = X.shape

    if k > n_features:
        raise ValueError("k cannot be greater than the number of features.")

    relevance_scores, _ = chi2(X, y)
    
    redundancy_matrix = _precompute_redundancy_matrix(X)

    selected_features = []
    remaining_features = set(range(n_features))
    
    first_feature_idx = np.argmax(relevance_scores)
    selected_features.append(first_feature_idx)
    remaining_features.remove(first_feature_idx)
    
    for _ in range(k - 1):
        best_score = -np.inf
        best_feature = -1
        
        candidates = list(remaining_features)
        
        for candidate_idx in candidates:
            relevance_score = relevance_scores[candidate_idx]
            
            redundancy_score = 0.0
            for selected_idx in selected_features:
                redundancy_score += redundancy_matrix[candidate_idx, selected_idx]
            
            avg_redundancy = redundancy_score / len(selected_features)
            
            mrmr_score = relevance_score - avg_redundancy
            
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_feature = candidate_idx
                
        if best_feature != -1:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

    return np.sort(np.array(selected_features))