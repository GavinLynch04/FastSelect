import numpy as np
from math import log
import numpy as np
from numba import njit, prange, float32, int32, int64
from numba.types import Tuple

MI_SIGNATURE = float32(float32[:], int32, float32[:], int32, int64)

@njit(MI_SIGNATURE, cache=True, nogil=True)
def _calculate_mi_optimized(x1, n_states1, x2, n_states2, n_samples):
    """
    Calculates Mutual Information between two discrete vectors.
    Uses float32 for better performance and memory usage.
    """
    contingency_table = np.zeros((n_states1, n_states2), dtype=np.float32)

    for i in range(n_samples):
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
            if p_xy > 1e-12 and p_x > 1e-12 and p_y > 1e-12:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

    return mi

PRECOMPUTE_SIGNATURE = Tuple((float32[:], float32[:, ::1]))(float32[:, ::1], int32[:])

@njit(PRECOMPUTE_SIGNATURE, parallel=True, cache=True)
def _precompute_mi_matrices(X, y):
    """
    Precomputes relevance (MI with target) and redundancy (MI between features)
    matrices with optimized data types and function calls.
    """
    n_samples, n_features = X.shape

    # --- Setup ---
    n_states_X = np.zeros(n_features, dtype=np.int32)
    for i in prange(n_features):
        n_states_X[i] = int(np.max(X[:, i])) + 1
    
    n_states_y = int(np.max(y)) + 1
    y_f32 = y.astype(np.float32)

    # --- Stage 1: Relevance Calculation ---
    relevance_scores = np.zeros(n_features, dtype=np.float32)
    for i in prange(n_features):
        relevance_scores[i] = _calculate_mi_optimized(
            X[:, i], n_states_X[i], y_f32, n_states_y, n_samples
        )

    # --- Stage 2: Redundancy Calculation ---
    redundancy_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    for i in prange(n_features):
        for j in range(i + 1, n_features):
            mi = _calculate_mi_optimized(
                X[:, i], n_states_X[i], X[:, j], n_states_X[j], n_samples
            )
            redundancy_matrix[i, j] = mi
            redundancy_matrix[j, i] = mi

    return relevance_scores, redundancy_matrix

def mRMR(X, y, k):
    """
    Selects top k features using mRMR, now generalized for ANY discrete data.
    This function handles non-contiguous, negative, or large integer categories
    by mapping them to zero-based integers before computation.
    
    Args:
        X (np.ndarray): Feature data (n_samples, n_features). Can contain any discrete values.
        y (np.ndarray): Target labels. Can contain any discrete values.
        k (int): The number of top features to select.
        
    Returns:
        np.ndarray: A sorted array of the top k feature indices.
    """
    n_samples, n_features = X.shape
    if not (0 < k <= n_features):
        raise ValueError("k must be a positive integer less than or equal to the number of features.")

    unique_values = np.unique(np.concatenate([X.flatten(), y]))
    value_to_int = {value: i for i, value in enumerate(unique_values)}

    mapper = np.vectorize(value_to_int.get)
    X_encoded = mapper(X)
    y_encoded = mapper(y)
    
    # Optimize data types later, numba does not like when you can pass in many different types
    X_for_numba = X_encoded.astype(np.float32)
    y_for_numba = y_encoded.astype(np.int32)
    
    relevance, redundancy = _precompute_mi_matrices(X_for_numba, y_for_numba)
    selected_features = []

    remaining_mask = np.ones(n_features, dtype=bool)

    # Select the first feature: the one with the highest relevance.
    first_feature_idx = np.argmax(relevance)
    selected_features.append(first_feature_idx)
    remaining_mask[first_feature_idx] = False

    for _ in range(k - 1):
        best_score = -np.inf
        best_feature = -1
        
        # Get the indices of features that are still available.
        remaining_indices = np.where(remaining_mask)[0]

        for candidate_idx in remaining_indices:
            # Relevance term: I(candidate; y)
            relevance_score = relevance[candidate_idx]
            
            # Redundancy term: Σ I(candidate; selected) / |S|
            # Look up pre-computed MI values. This is extremely fast.
            redundancy_score = np.sum(redundancy[candidate_idx, selected_features])
            avg_redundancy = redundancy_score / len(selected_features)
            
            # mRMR score (MID formulation: Relevance - Redundancy)
            mrmr_score = relevance_score - avg_redundancy

            if mrmr_score > best_score:
                best_score = mrmr_score
                best_feature = candidate_idx

        # Add the best feature found and update the mask
        if best_feature != -1:
            selected_features.append(best_feature)
            remaining_mask[best_feature] = False
    print("done")
    return np.sort(np.array(selected_features))


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
