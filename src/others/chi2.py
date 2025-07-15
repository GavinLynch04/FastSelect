import numpy as np
from numba import njit, prange
from scipy.stats import chi2 as chi2_dist


def chi2_numba(X, y):
    """
    A Numba-parallelized implementation of the Chi-squared feature selection test.

    This function calculates the Chi-squared statistic between each non-negative
    feature and the target classes.

    Args:
        X (np.ndarray): The input sample matrix of shape (n_samples, n_features)
                        with non-negative values.
        y (np.ndarray): The target vector of shape (n_samples,).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - chi2_stats: The Chi-squared statistics for each feature.
            - p_values: The p-values for each feature.
    """
    # Ensure input data is non-negative, similar to scikit-learn's check
    if np.any(X < 0):
        raise ValueError("Input matrix X must contain non-negative values.")

    # This is the core numerical function that will be JIT-compiled by Numba.
    # The `parallel=True` flag enables automatic parallelization with `prange`.
    @njit(parallel=True, fastmath=True)
    def _chi2_core(X, y_mapped, class_freqs, feature_counts, n_samples, n_features, n_classes):
        chi2_stats = np.zeros(n_features, dtype=np.float64)

        # The loop over features is parallelized across multiple CPU cores.
        for i in prange(n_features):
            # Calculate observed frequencies for the i-th feature
            observed = np.zeros(n_classes, dtype=np.float64)
            for j in range(n_samples):
                observed[y_mapped[j]] += X[j, i]

            # Calculate expected frequencies
            # expected = (class_frequency * feature_frequency) / total_samples
            expected = class_freqs * feature_counts[i] / n_samples

            # Calculate the Chi-squared term: (observed - expected)^2 / expected
            terms = np.zeros(n_classes, dtype=np.float64)
            for k in range(n_classes):
                # Avoid division by zero
                if expected[k] > 0:
                    terms[k] = (observed[k] - expected[k]) ** 2 / expected[k]

            chi2_stats[i] = np.sum(terms)

        return chi2_stats

    # === Pre-computation steps (run outside the JIT-compiled function) ===
    n_samples, n_features = X.shape

    # Map class labels to integers 0, 1, 2... for easy indexing
    labels, y_mapped = np.unique(y, return_inverse=True)
    n_classes = len(labels)

    # Calculate frequencies for classes and features
    class_freqs = np.bincount(y_mapped).astype(np.float64)
    feature_counts = X.sum(axis=0)

    # === Call the core JIT-compiled function ===
    chi2_stats = _chi2_core(X, y_mapped, class_freqs, feature_counts, n_samples, n_features, n_classes)

    # === Post-computation: Calculate p-values using SciPy ===
    # The degrees of freedom is (n_classes - 1) * (2 - 1) = n_classes - 1
    dof = n_classes - 1
    p_values = chi2_dist.sf(chi2_stats, dof)

    return chi2_stats, p_values