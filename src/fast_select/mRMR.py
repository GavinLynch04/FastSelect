from __future__ import annotations
import numpy as np
from math import log, ceil
from numba import njit, prange, float32, int32, int64, cuda
from numba.types import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
import time

from . import mutual_information as mi

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
        unique_vals = np.unique(np.concatenate([np.unique(X), np.unique(y)]))
        self.unique_vals_ = unique_vals
        n_states = len(unique_vals)
        X_encoded, y_encoded = _encode_data_numba(X, y, unique_vals)
        
        relevance, redundancy = mi.calculate_mi_matrices(
            X_encoded, y_encoded, n_states, self.backend
        )
            
        self.relevance_scores_ = relevance
        self.redundancy_matrix_ = redundancy

        
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

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to the selected features."""
        check_is_fitted(self)
        X = validate_data(
            self, X,
            reset=False,
            dtype=None
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