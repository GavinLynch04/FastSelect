import numpy as np
from numba import jit, prange
from math import log
from Models.FeatureSelection.FSHelper import *

# ==============================================================================
#  Part 1: The High-Performance Numba JIT Kernel
# ==============================================================================

@jit(nopython=True, cache=True)
def _calculate_mi(x1, x2, n_samples):
    """
    Calculates the mutual information between two discrete vectors.
    This is a helper function called by the main parallel loop.
    """
    # Find the number of states for each variable (e.g., 3 for 0,1,2 SNP data)
    n_states1 = int(np.max(x1)) + 1
    n_states2 = int(np.max(x2)) + 1

    # Create the contingency table (joint probability distribution)
    contingency_table = np.zeros((n_states1, n_states2))
    for i in range(n_samples):
        contingency_table[int(x1[i]), int(x2[i])] += 1
    
    # Normalize to get probabilities
    contingency_table /= n_samples

    # Calculate marginal probabilities
    p1 = np.sum(contingency_table, axis=1)
    p2 = np.sum(contingency_table, axis=0)

    # Calculate mutual information
    mi = 0.0
    for i in range(n_states1):
        for j in range(n_states2):
            if contingency_table[i, j] > 1e-10: # Avoid log(0)
                mi += contingency_table[i, j] * log(contingency_table[i, j] / (p1[i] * p2[j]))
    
    return mi

@jit(nopython=True, parallel=True, cache=True)
def _precompute_mi_matrices(X, y):
    """
    JIT-compiled kernel to pre-compute all MI values in parallel.
    """
    n_samples, n_features = X.shape
    
    # 1. Calculate Relevance: I(feature; target) for all features
    relevance_scores = np.zeros(n_features)
    for i in prange(n_features):
        relevance_scores[i] = _calculate_mi(X[:, i], y, n_samples)

    # 2. Calculate Redundancy: I(feature_i; feature_j) for all pairs
    # This is an N x N matrix.
    redundancy_matrix = np.zeros((n_features, n_features))
    for i in prange(n_features):
        for j in range(i + 1, n_features):
            mi = _calculate_mi(X[:, i], X[:, j], n_samples)
            redundancy_matrix[i, j] = mi
            redundancy_matrix[j, i] = mi # Matrix is symmetric
            
    return relevance_scores, redundancy_matrix

# ==============================================================================
#  Part 2: The Main Python Wrapper Function
# ==============================================================================

def fs_mrmr(X, y, k):
    """
    Selects top k features using a highly efficient, JIT-compiled mRMR algorithm.

    This function follows the classic mRMR 'MIQ' (Mutual Information Quotient) formulation.

    Args:
        X (np.ndarray): Feature data (n_samples, n_features). MUST be discrete integers (e.g., int8).
        y (np.ndarray): Target labels. MUST be discrete integers.
        k (int): The number of top features to select.

    Returns:
        np.ndarray: A sorted array of the top k feature indices.
    """
    n_samples, n_features = X.shape

    if k > n_features:
        raise ValueError("k cannot be greater than the number of features.")

    if X.dtype != np.int8:
        X = X.astype(np.int8)
    if y.dtype != np.int8:
        y = y.astype(np.int8)

    relevance, redundancy = _precompute_mi_matrices(X, y)
    
    # --- Stage 2: Fast Sequential Greedy Selection ---
    selected_features = []
    remaining_features = list(range(n_features))
    
    # Select the first feature (the one with the highest relevance)
    first_feature_idx = np.argmax(relevance)
    selected_features.append(first_feature_idx)
    remaining_features.remove(first_feature_idx)
    
    # Iteratively select the rest
    for _ in range(k - 1):
        best_score = -np.inf
        best_feature = -1
        
        # Calculate redundancy for each candidate with the *current* selected set
        for candidate_idx in remaining_features:
            # Relevance term: I(candidate; y)
            relevance_score = relevance[candidate_idx]
            
            # Redundancy term: Σ I(candidate; selected)
            # We look up these values from our pre-computed matrix. This is very fast.
            redundancy_score = 0.0
            for selected_idx in selected_features:
                redundancy_score += redundancy[candidate_idx, selected_idx]
            
            # Average the redundancy
            avg_redundancy = redundancy_score / len(selected_features)
            
            # The mRMR score (MID formulation: Difference)
            # You could also use MIQ (Quotient): relevance_score / (avg_redundancy + 1e-10)
            mrmr_score = relevance_score - avg_redundancy
            
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_feature = candidate_idx
                
        # Add the best feature found in this round
        if best_feature != -1:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

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

    print("Running custom Chi2-RMR feature selection...")
    n_samples, n_features = X.shape

    if k > n_features:
        raise ValueError("k cannot be greater than the number of features.")

    # --- Stage 1: Pre-computation ---
    
    # 1. Calculate Relevance using Chi2 (The key change)
    # chi2 returns two arrays: the chi2 stats and the p-values. We want the stats.
    # Note: chi2 expects non-negative values, which is fine for SNP data 0,1,2.
    print("  - Calculating relevance scores using chi2...")
    relevance_scores, _ = chi2(X, y)
    
    # 2. Calculate Redundancy using Mutual Information (The parallel part)
    # Cast to int8 for performance, as MI calculation needs it.
    if X.dtype != np.int8:
        X = X.astype(np.int8)
    
    print("  - Pre-computing redundancy matrix with Numba (parallel)...")
    redundancy_matrix = _precompute_redundancy_matrix(X)

    # --- Stage 2: Fast Sequential Greedy Selection (Identical to mRMR logic) ---
    print("  - Performing greedy selection...")
    selected_features = []
    # Use a set for faster removal
    remaining_features = set(range(n_features))
    
    # Select the first feature (the one with the highest *chi2* relevance)
    first_feature_idx = np.argmax(relevance_scores)
    selected_features.append(first_feature_idx)
    remaining_features.remove(first_feature_idx)
    
    # Iteratively select the rest
    for _ in range(k - 1):
        best_score = -np.inf
        best_feature = -1
        
        # Create a list of candidates to iterate over
        candidates = list(remaining_features)
        
        for candidate_idx in candidates:
            # Relevance term: chi2(candidate; y)
            relevance_score = relevance_scores[candidate_idx]
            
            # Redundancy term: Σ I(candidate; selected)
            redundancy_score = 0.0
            for selected_idx in selected_features:
                redundancy_score += redundancy_matrix[candidate_idx, selected_idx]
            
            avg_redundancy = redundancy_score / len(selected_features)
            
            # Chi2-RMR score (using the MID formulation)
            mrmr_score = relevance_score - avg_redundancy
            
            if mrmr_score > best_score:
                best_score = mrmr_score
                best_feature = candidate_idx
                
        if best_feature != -1:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

    print("Chi2-RMR selection complete.")
    return np.sort(np.array(selected_features))
