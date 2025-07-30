from __future__ import annotations
import numpy as np
from numba import njit, prange
from scipy.stats import chi2 as chi2_dist
from sklearn.utils.validation import check_array, check_X_y

@njit(fastmath=True)
def _compute_observed_and_feature_counts(X, y_mapped, n_features, n_classes): # pragma: no cover
    """
    Efficiently computes the observed frequency matrix and feature counts
    in a single pass over the data.
    """
    observed = np.zeros((n_classes, n_features), dtype=np.float64)
    feature_counts = np.zeros(n_features, dtype=np.float64)
    n_samples = X.shape[0]
    for i in range(n_samples):
        class_idx = y_mapped[i]
        for j in range(n_features):
            val = X[i, j]
            observed[class_idx, j] += val
            feature_counts[j] += val
    return observed, feature_counts

@njit(parallel=True, fastmath=True)
def _chi2_core(observed, class_freqs, feature_counts, n_samples): # pragma: no cover
    """
    Calculates chi2 stats from the pre-computed observed matrix.
    The loop over features is parallelized.
    """
    n_classes, n_features = observed.shape
    chi2_stats = np.zeros(n_features, dtype=np.float64)
    
    for i in prange(n_features):
        # Skip features that have a total count of 0
        if feature_counts[i] == 0:
            continue
            
        expected_i = class_freqs * feature_counts[i] / n_samples
        term = 0.0
        for k in range(n_classes):
            observed_ik = observed[k, i]
            expected_ik = expected_i[k]
            if expected_ik > 1e-12:
                term += (observed_ik - expected_ik)**2 / expected_ik
        chi2_stats[i] = term
        
    return chi2_stats

def chi2(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Chi-squared statistics between each feature and the target vector.

    This function calculates the Chi-squared test for independence between each
    non-negative feature and the class labels (similar to SciKit-Learn). It is suitable for features
    that represent frequencies or counts (e.g., word counts in text classification).

    Args:
        X (np.ndarray): The input sample matrix of shape (n_samples, n_features).
                        Must contain non-negative, count-based feature values.
        y (np.ndarray): The target vector of class labels, shape (n_samples,).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - chi2_stats: The Chi-squared statistics for each feature.
            - p_values: The p-values for each feature.
    """
    X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)

    X, y = check_X_y(X, y, y_numeric=True)
    
    if np.any(X < 0):
        raise ValueError("Input matrix X must contain non-negative values.")

    n_samples, n_features = X.shape
    labels, y_mapped = np.unique(y, return_inverse=True)
    n_classes = len(labels)

    if n_classes < 2:
        return np.zeros(n_features, dtype=np.float64), np.ones(n_features, dtype=np.float64)

    class_freqs = np.bincount(y_mapped).astype(np.float64)
    
    observed, feature_counts = _compute_observed_and_feature_counts(
        X, y_mapped, n_features, n_classes
    )
    
    chi2_stats = _chi2_core(observed, class_freqs, feature_counts, n_samples)
    
    dof = n_classes - 1
    p_values = chi2_dist.sf(chi2_stats, dof)
    
    return chi2_stats, p_values