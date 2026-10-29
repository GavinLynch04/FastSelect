from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from scipy.stats import chi2 as chi2_dist
from sklearn.utils.validation import check_array, check_X_y

from .utils import resolve_compute_backend


@njit(fastmath=True)
def _compute_observed_and_feature_counts(X, y_mapped, n_features, n_classes):  # pragma: no cover
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


@njit(parallel=True)
def _compute_observed_and_feature_counts_csc(
    data, indices, indptr, y_mapped, n_features, n_classes
):  # pragma: no cover
    """Sparse (CSC) counterpart of :func:`_compute_observed_and_feature_counts`.

    Each column is summed in row order, as in the dense loop, and absent entries
    contribute nothing, so both layouts produce identical sums.  ``fastmath`` is
    deliberately off: it would let the per-column total be reassociated.
    """
    observed = np.zeros((n_classes, n_features), dtype=np.float64)
    feature_counts = np.zeros(n_features, dtype=np.float64)
    for j in prange(n_features):
        total = 0.0
        for position in range(indptr[j], indptr[j + 1]):
            value = data[position]
            observed[y_mapped[indices[position]], j] += value
            total += value
        feature_counts[j] = total
    return observed, feature_counts


@njit(parallel=True, fastmath=True)
def _chi2_core(observed, class_freqs, feature_counts, n_samples):  # pragma: no cover
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
                term += (observed_ik - expected_ik) ** 2 / expected_ik
        chi2_stats[i] = term

    return chi2_stats


def chi2(X: np.ndarray, y: np.ndarray, *, backend="auto") -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Chi-squared statistics between each feature and the target vector.

    This function calculates the Chi-squared test for independence between each
    non-negative feature and the class labels (similar to SciKit-Learn). It is suitable for features
    that represent frequencies or counts (e.g., word counts in text classification).

    Args:
        X (np.ndarray | scipy.sparse matrix or array): The input sample matrix of
                        shape (n_samples, n_features). Must contain non-negative,
                        count-based feature values. Sparse input is counted on the
                        CPU without densifying and gives the same statistics as
                        the equivalent dense array.
        y (np.ndarray): The target vector of class labels, shape (n_samples,).
        backend (str): 'auto' and 'cpu' run on the CPU. 'metal' explicitly
                       requests MLX; it was slower than CPU in sustained
                       benchmarks, so 'auto' does not select it, and it
                       does not accept sparse input. This statistic has no
                       CUDA implementation.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - chi2_stats: The Chi-squared statistics for each feature.
            - p_values: The p-values for each feature.
    """
    is_sparse = sp.issparse(X)
    if is_sparse and backend == "metal":
        raise ValueError("Sparse input runs on the CPU only; use backend='cpu' or 'auto', or pass a dense array.")

    X = check_array(X, accept_sparse="csc", dtype=[np.float64, np.float32], ensure_2d=True)

    X, y = check_X_y(X, y, accept_sparse="csc", y_numeric=True)

    if is_sparse and not X.has_canonical_format:
        # Sorted row indices keep the per-column summation in row order.
        X = X.copy()
        X.sum_duplicates()

    if np.any((X.data if is_sparse else X) < 0):
        raise ValueError("Input matrix X must contain non-negative values.")

    n_samples, n_features = X.shape
    effective_backend = resolve_compute_backend(backend, cuda_available=False, auto_metal=False)
    labels, y_mapped = np.unique(y, return_inverse=True)
    n_classes = len(labels)

    if n_classes < 2:
        return np.zeros(n_features, dtype=np.float64), np.ones(n_features, dtype=np.float64)

    class_freqs = np.bincount(y_mapped).astype(np.float64)

    if effective_backend == "metal":
        from ._chi2_metal import chi2_counts_metal

        observed, feature_counts = chi2_counts_metal(X, y_mapped, n_classes)
    elif is_sparse:
        observed, feature_counts = _compute_observed_and_feature_counts_csc(
            X.data, X.indices, X.indptr, y_mapped, n_features, n_classes
        )
    else:
        observed, feature_counts = _compute_observed_and_feature_counts(X, y_mapped, n_features, n_classes)

    chi2_stats = _chi2_core(observed, class_freqs, feature_counts, n_samples)

    dof = n_classes - 1
    p_values = chi2_dist.sf(chi2_stats, dof)

    return chi2_stats, p_values
