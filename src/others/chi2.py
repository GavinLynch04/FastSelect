import numpy as np
from numba import njit, prange
from scipy.stats import chi2 as chi2_dist
@njit(fastmath=True)
def _compute_observed_matrix(X, y_mapped, n_features, n_classes):
    """
    Efficiently computes the observed frequency matrix in a single pass.
    This is not parallelized as parallel writes to the same array rows would cause
    a race condition without atomics, but this single-threaded loop is extremely fast.
    """
    observed = np.zeros((n_classes, n_features), dtype=np.float64)
    n_samples = X.shape[0]
    for i in range(n_samples):
        class_idx = y_mapped[i]
        for j in range(n_features):
            observed[class_idx, j] += X[i, j]
    return observed

@njit(parallel=True, fastmath=True)
def _chi2_core(observed, class_freqs, feature_counts, n_samples):
    """
    Calculates chi2 stats from the pre-computed observed matrix.
    The loop over features is parallelized.
    """
    n_classes, n_features = observed.shape
    chi2_stats = np.zeros(n_features, dtype=np.float64)

    # Parallel loop over features
    for i in prange(n_features):
        # Expected frequencies for feature i
        # expected = (class_frequency * feature_frequency) / total_samples
        expected_i = class_freqs * feature_counts[i] / n_samples

        # Chi-squared term calculation
        term = 0.0
        for k in range(n_classes):
            # The observed value is directly looked up from the pre-computed matrix
            observed_ik = observed[k, i]
            expected_ik = expected_i[k]
            
            if expected_ik > 1e-9: # More robust check for zero
                term += (observed_ik - expected_ik)**2 / expected_ik
        
        chi2_stats[i] = term
        
    return chi2_stats

def chi2_numba(X, y):
    """
    A more optimized Numba-parallelized implementation of the Chi-squared test.
    This version pre-computes the entire observed frequency matrix for higher efficiency.
    
    Args:
        X (np.ndarray): The input sample matrix of shape (n_samples, n_features).
                        Must contain non-negative values.
        y (np.ndarray): The target vector of shape (n_samples,).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - chi2_stats: The Chi-squared statistics for each feature.
            - p_values: The p-values for each feature.
    """
    if np.any(X < 0):
        raise ValueError("Input matrix X must contain non-negative values.")
    
    n_samples, n_features = X.shape
    labels, y_mapped = np.unique(y, return_inverse=True)
    n_classes = len(labels)

    if n_classes < 2:
        return np.zeros(n_features, dtype=np.float64), np.ones(n_features, dtype=np.float64)
    
    class_freqs = np.bincount(y_mapped).astype(np.float64)
    feature_counts = X.sum(axis=0)
    
    observed = _compute_observed_matrix(X, y_mapped, n_features, n_classes)
    chi2_stats = _chi2_core(observed, class_freqs, feature_counts, n_samples)
    
    dof = n_classes - 1
    p_values = chi2_dist.sf(chi2_stats, dof)
    
    return chi2_stats, p_values