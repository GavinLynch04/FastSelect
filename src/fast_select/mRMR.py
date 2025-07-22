from __future__ import annotations
import numpy as np
from math import log, ceil
from numba import njit, prange, float32, int32, int64, cuda
from numba.types import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
import time

@njit(parallel=True, cache=True)
def _encode_data_numba(X, y, unique_vals):
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
def _calculate_mi(x1, n_states1, x2, n_states2, n_samples):
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
                # Corrected syntax here
                mi += p_xy * log(p_xy / (p_x * p_y))
    return mi

@njit(parallel=True, cache=True)
def _precompute_mi_matrices(X, y):
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

@cuda.jit
def _relevance_kernel_gpu(X_gpu, y_gpu, relevance_out, n_samples, n_states):
    """Specialized CUDA kernel for calculating relevance scores I(f; y)."""
    # Define shared memory for this block's contingency table
    shared_contingency = cuda.shared.array(shape=(32, 32), dtype=float32) # MAX_STATES = 32
    
    # Global feature index for this thread block
    feature_idx = cuda.blockIdx.x
    
    # Thread indices within the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # Initialize shared memory
    if tx < n_states and ty < n_states:
        shared_contingency[tx, ty] = 0.0
    cuda.syncthreads()

    # Parallel histogram construction for this feature against y
    for sample_idx in range(n_samples):
        val1 = int(X_gpu[sample_idx, feature_idx])
        val2 = int(y_gpu[sample_idx])
        cuda.atomic.add(shared_contingency, (val1, val2), 1.0)
    cuda.syncthreads()
    
    # One thread per block finalizes the MI calculation
    if tx == 0 and ty == 0:
        # Normalize
        for r in range(n_states):
            for c in range(n_states):
                shared_contingency[r,c] /= n_samples
        
        mi = 0.0
        for r in range(n_states):
            p_x_r = 0.0
            for c_ in range(n_states): p_x_r += shared_contingency[r, c_]
            for c in range(n_states):
                p_y_c = 0.0
                for r_ in range(n_states): p_y_c += shared_contingency[r_, c]
                p_xy_val = shared_contingency[r, c]
                if p_xy_val > 1e-12 and p_x_r > 1e-12 and p_y_c > 1e-12:
                    mi += p_xy_val * log(p_xy_val / (p_x_r * p_y_c))
        
        relevance_out[feature_idx] = mi

@cuda.jit
def _redundancy_kernel_gpu(X_gpu, redundancy_out, n_features, n_samples, n_states):
    """Specialized CUDA kernel for calculating redundancy matrix I(f_i; f_j)."""
    shared_contingency = cuda.shared.array(shape=(32, 32), dtype=float32)
    
    # Global 2D feature indices for this thread block
    f1_idx, f2_idx = cuda.grid(2)
    
    # Process only the upper triangle of the matrix to avoid redundant work
    if not (f1_idx < n_features and f2_idx > f1_idx):
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
        mi = 0.0
        for r in range(n_states):
            p_x_r = 0.0
            for c_ in range(n_states): p_x_r += shared_contingency[r, c_]
            for c in range(n_states):
                p_y_c = 0.0
                for r_ in range(n_states): p_y_c += shared_contingency[r_, c]
                p_xy_val = shared_contingency[r, c]
                if p_xy_val > 1e-12 and p_x_r > 1e-12 and p_y_c > 1e-12:
                    mi += p_xy_val * log(p_xy_val / (p_x_r * p_y_c))
        
        redundancy_out[f1_idx, f2_idx] = mi
        redundancy_out[f2_idx, f1_idx] = mi # Symmetrically fill the matrix

def _precompute_mi_matrices_gpu(X, y, n_states):
    """Orchestrator for the GPU backend."""
    n_samples, n_features = X.shape
    if n_states > 32:
        raise ValueError(f"GPU backend supports a maximum of 32 unique discrete states, but found {n_states}.")

    X_gpu = cuda.to_device(X)
    y_gpu = cuda.to_device(y)
    relevance_gpu = cuda.device_array(n_features, dtype=np.float32)
    redundancy_gpu = cuda.device_array((n_features, n_features), dtype=np.float32)

    threads_per_block = (n_states, n_states)

    # Launch Relevance Kernel (1D grid of blocks)
    blocks_per_grid_rel = (n_features,)
    _relevance_kernel_gpu[blocks_per_grid_rel, threads_per_block](X_gpu, y_gpu, relevance_gpu, n_samples, n_states)
    
    # Launch Redundancy Kernel (2D grid of blocks)
    blocks_per_grid_red_x = ceil(n_features / threads_per_block[0])
    blocks_per_grid_red_y = ceil(n_features / threads_per_block[1])
    blocks_per_grid_red = (blocks_per_grid_red_x, blocks_per_grid_red_y)
    _redundancy_kernel_gpu[blocks_per_grid_red, threads_per_block](X_gpu, redundancy_gpu, n_features, n_samples, n_states)
    
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

        selected_indices = []
        remaining_mask = np.ones(self.n_features_in_, dtype=bool)

        first_feature_idx = np.argmax(relevance)
        selected_indices.append(first_feature_idx)
        remaining_mask[first_feature_idx] = False
        
        start = time.time()
        for _ in range(self.n_features_to_select - 1):
            best_score = -np.inf
            best_feature = -1
            
            remaining_indices = np.where(remaining_mask)[0]
            
            m = len(selected_indices)

            for candidate_idx in remaining_indices:
                relevance_score = self.relevance_scores_[candidate_idx]
                redundancy_score = np.sum(self.redundancy_matrix_[candidate_idx, selected_indices])
                
                if self.method == 'MID':
                    score = relevance_score - (redundancy_score / m)
                else: # 'MIQ'
                    score = relevance_score / (redundancy_score / m + 1e-9)
                
                if score > best_score:
                    best_score = score
                    best_feature = candidate_idx
            
            if best_feature != -1:
                selected_indices.append(best_feature)
                remaining_mask[best_feature] = False
        end = time.time()
        print(end-start)
        self.top_features_ = np.array(selected_indices)
        self.feature_importances_ = self.relevance_scores_

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