from __future__ import annotations
import numpy as np
from math import log
from numba import njit, prange, float32, int32, int64
from numba.types import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


@njit(cache=True, nogil=True)
def _calculate_mi_optimized(x1, n_states1, x2, n_states2, n_samples):
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
    y_f32 = y.astype(np.float32)
    
    relevance_scores = np.zeros(n_features, dtype=np.float32)
    for i in prange(n_features):
        relevance_scores[i] = _calculate_mi_optimized(
            X[:, i], n_states_X[i], y_f32, n_states_y, n_samples
        )
        
    redundancy_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    for i in prange(n_features):
        for j in range(i + 1, n_features):
            mi = _calculate_mi_optimized(
                X[:, i], n_states_X[i], X[:, j], n_states_X[j], n_samples
            )
            redundancy_matrix[i, j] = mi
            redundancy_matrix[j, i] = mi
            
    return relevance_scores, redundancy_matrix

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
        
    """
    def __init__(self, n_features_to_select: int, method: str = 'MID'):
        self.n_features_to_select = n_features_to_select
        self.method = method
        if self.method not in ['MID', 'MIQ']:
            raise ValueError("Method must be either 'MID' or 'MIQ'.")

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
        
        unique_values = np.unique(np.concatenate([X.flatten(), y]))
        self.value_to_int_ = {value: i for i, value in enumerate(unique_values)}
        mapper = np.vectorize(self.value_to_int_.get)
        
        X_encoded = mapper(X)
        y_encoded = mapper(y)

        relevance, redundancy = _precompute_mi_matrices(X_for_numba, y_for_numba)
        
        self.relevance_scores_ = relevance
        self.redundancy_matrix_ = redundancy

        selected_indices = []
        remaining_mask = np.ones(self.n_features_in_, dtype=bool)

        first_feature_idx = np.argmax(relevance)
        selected_indices.append(first_feature_idx)
        remaining_mask[first_feature_idx] = False

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