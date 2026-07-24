import heapq
import math

import numba
import numpy as np
import pandas as pd
from numba import cuda
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted, check_X_y

from .utils import ensure_cuda_context, is_cuda_ready


@numba.njit(cache=True)
def _cfs_merit(sum_r_cf, k, sum_r_ff): # pragma: no cover
    """
    k              : number of features in the subset
    sum_r_cf       : sum of feature-class SUs in the subset
    sum_r_ff       : sum of all pairwise feature-feature SUs in the subset
    """
    if k == 0:
        return 0.0
    r_cf_avg = sum_r_cf / k
    r_ff_avg = (2.0 * sum_r_ff) / (k * (k - 1)) if k > 1 else 0.0
    denom = math.sqrt(k + k * (k - 1) * r_ff_avg)
    return (k * r_cf_avg / denom) if denom > 1e-12 else 0.0

@numba.njit(cache=True)
def _entropy(x, n_states):  # pragma: no cover
    """Calculates the entropy of a 1D array of discrete values."""
    n_samples = x.shape[0]
    if n_samples == 0:
        return 0.0

    counts = np.zeros(n_states, dtype=np.float32)
    for i in range(n_samples):
        counts[x[i]] += 1.0

    p = counts / n_samples
    entropy = 0.0
    for prob in p:
        if prob > 1e-12:
            entropy -= prob * np.log2(prob)
    return entropy


@numba.njit(cache=True)
def _mutual_information(x, y, n_states_x, n_states_y):  # pragma: no cover
    """Calculates the mutual information between two discrete arrays."""
    n_samples = x.shape[0]
    if n_samples == 0:
        return 0.0

    contingency_table = np.zeros((n_states_x, n_states_y), dtype=np.float32)
    for i in range(n_samples):
        contingency_table[x[i], y[i]] += 1.0

    p_xy = contingency_table / n_samples
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    mi = 0.0
    for i in range(n_states_x):
        for j in range(n_states_y):
            if p_xy[i, j] > 1e-12 and p_x[i] > 1e-12 and p_y[j] > 1e-12:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


@numba.njit(cache=True)
def _symmetrical_uncertainty(x, y, n_states_x, n_states_y):  # pragma: no cover
    """Calculates Symmetrical Uncertainty, a normalized form of mutual information."""
    h_x = _entropy(x, n_states_x)
    h_y = _entropy(y, n_states_y)

    if h_x + h_y < 1e-12:
        return 0.0

    i_xy = _mutual_information(x, y, n_states_x, n_states_y)
    return 2.0 * i_xy / (h_x + h_y)


@numba.njit(parallel=True, cache=True)
def _precompute_correlations_cpu(X_encoded, y_encoded, n_states_features, n_states_y):  # pragma: no cover
    """
    (CPU VERSION) Calculates all feature-class and feature-feature correlations in parallel.
    """
    n_features = X_encoded.shape[1]

    # Feature-class correlations (r_cf)
    r_cf_all = np.zeros(n_features, dtype=np.float32)
    for i in numba.prange(n_features):
        r_cf_all[i] = _symmetrical_uncertainty(
            X_encoded[:, i], y_encoded, n_states_features[i], n_states_y
        )

    # Feature-feature correlations (r_ff)
    r_ff_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    for i in numba.prange(n_features):
        for j in range(i + 1, n_features):
            corr = _symmetrical_uncertainty(
                X_encoded[:, i], X_encoded[:, j], n_states_features[i], n_states_features[j]
            )
            r_ff_matrix[i, j] = corr
            r_ff_matrix[j, i] = corr

    return r_cf_all, r_ff_matrix

def _best_first_search(
    n_features, r_cf_all, r_ff_matrix, max_backtracks=5
): # pragma: no cover
    """Forward best-first CFS search with the paper's stale-node stopping rule."""
    queue = []
    visited = {()}
    best_subset = ()
    best_merit = 0.0
    stale_nodes = 0

    for feature in range(n_features):
        subset = (feature,)
        visited.add(subset)
        sum_r_cf = float(r_cf_all[feature])
        subset_merit = float(_cfs_merit(sum_r_cf, 1, 0.0))
        heapq.heappush(queue, (-subset_merit, subset, sum_r_cf, 0.0))

    while queue and stale_nodes < max_backtracks:
        neg_merit, subset, sum_r_cf, sum_r_ff = heapq.heappop(queue)
        subset_merit = -neg_merit
        if subset_merit > best_merit + 1e-12:
            best_merit = subset_merit
            best_subset = subset
            stale_nodes = 0
        else:
            stale_nodes += 1

        selected = set(subset)
        for feature in range(n_features):
            if feature in selected:
                continue
            child = tuple(sorted(subset + (feature,)))
            if child in visited:
                continue
            visited.add(child)
            child_sum_r_cf = sum_r_cf + float(r_cf_all[feature])
            child_sum_r_ff = sum_r_ff
            for selected_feature in subset:
                child_sum_r_ff += float(
                    r_ff_matrix[feature, selected_feature]
                )
            child_merit = float(
                _cfs_merit(
                    child_sum_r_cf,
                    len(child),
                    child_sum_r_ff,
                )
            )
            heapq.heappush(
                queue,
                (
                    -child_merit,
                    child,
                    child_sum_r_cf,
                    child_sum_r_ff,
                ),
            )

    return list(best_subset)


@cuda.jit(device=True)
def _cu_entropy(counts, n_samples, n_states): # pragma: no cover
    """(GPU DEVICE) Calculates entropy from a counts array."""
    if n_samples == 0:
        return 0.0
    entropy = 0.0
    for i in range(n_states):
        count = counts[i]
        if count > 0:
            prob = count / n_samples
            entropy -= prob * math.log2(prob)
    return entropy


@cuda.jit(device=True)
def _cu_symmetrical_uncertainty(x, y, n_states_x, n_states_y): # pragma: no cover
    """(GPU DEVICE) Calculates Symmetrical Uncertainty."""
    n_samples = x.shape[0]

    # Using local memory arrays for contingency tables and counts
    contingency = cuda.local.array(shape=(32, 32), dtype=numba.float32)
    counts_x = cuda.local.array(shape=32, dtype=numba.float32)
    counts_y = cuda.local.array(shape=32, dtype=numba.float32)

    for i in range(n_states_x):
        counts_x[i] = 0.0
        for j in range(n_states_y):
            contingency[i, j] = 0.0

    for j in range(n_states_y):
        counts_y[j] = 0.0

    for i in range(n_samples):
        contingency[x[i], y[i]] += 1.0
        counts_x[x[i]] += 1.0
        counts_y[y[i]] += 1.0

    h_x = _cu_entropy(counts_x, n_samples, n_states_x)
    h_y = _cu_entropy(counts_y, n_samples, n_states_y)

    if h_x + h_y < 1e-12:
        return 0.0

    mi = 0.0
    for i in range(n_states_x):
        for j in range(n_states_y):
            if contingency[i, j] > 1e-12:
                p_xy = contingency[i, j] / n_samples
                p_x = counts_x[i] / n_samples
                p_y = counts_y[j] / n_samples
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

    return 2.0 * mi / (h_x + h_y)


@cuda.jit
def _precompute_correlations_gpu_kernel(X_d, y_d, n_states_features_d, n_states_y, r_cf_out,
                                        r_ff_out):  # pragma: no cover
    """
    (GPU KERNEL) Calculates all correlations on the GPU.
    Each thread block handles a single feature's calculations against the class and all other features.
    """
    i = cuda.blockIdx.x
    n_features = X_d.shape[1]

    if i >= n_features:
        return

    # 1. Feature-Class Correlation
    r_cf_out[i] = _cu_symmetrical_uncertainty(
        X_d[:, i], y_d, n_states_features_d[i], n_states_y
    )

    # 2. Feature-Feature Correlations
    for j in range(i + 1, n_features):
        corr = _cu_symmetrical_uncertainty(
            X_d[:, i], X_d[:, j], n_states_features_d[i], n_states_features_d[j]
        )
        r_ff_out[i, j] = corr
        r_ff_out[j, i] = corr


class CFS(BaseEstimator, SelectorMixin):
    """
    GPU and CPU-accelerated Correlation-based Feature Selection (CFS).

    This selector evaluates feature subsets on the hypothesis that a good subset
    contains features highly correlated with the class, yet uncorrelated with each other.
    Symmetrical Uncertainty is used as the correlation measure.

    The algorithm performs a greedy "best-first" search to find the best subset.
    It supports both CPU and GPU backends for the computationally intensive
    correlation matrix calculation.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for discretizing continuous features.

    strategy : {'uniform', 'quantile', 'kmeans'}, default='uniform'
        Strategy for binning continuous features.

    backend : {'auto', 'gpu', 'cpu'}, default='auto'
        The compute backend. 'auto' uses GPU if available.

    n_jobs : int, default=-1
        Number of CPU threads to use. Ignored for the 'gpu' backend.

    max_backtracks : int, default=5
        Stop best-first search after this many consecutive expanded subsets
        fail to improve the best CFS merit, matching the canonical default.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    selected_indices_ : ndarray of shape (n_selected_features,)
        Indices of the selected features.

    support_mask_ : ndarray of shape (n_features_in_,)
        A boolean mask of the selected features.

    merit_ : float
        The CFS merit score of the selected feature subset.
    """

    def __init__(
        self,
        n_bins=10,
        strategy='uniform',
        backend='auto',
        n_jobs=-1,
        max_backtracks=5,
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.backend = backend
        self.n_jobs = n_jobs
        self.max_backtracks = max_backtracks

    def fit(self, X, y):
        """
        Fits the CFS model to find the best feature subset by evaluating feature
        correlation with the target and inter-feature correlation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be continuous, discrete, or mixed.
        y : array-like of shape (n_samples,)
            Target values. Must be discrete (Classification).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        feature_names = np.asarray(X.columns) if hasattr(X, "columns") else None
        X, y = check_X_y(X, y, dtype=None, ensure_min_samples=2)
        self.n_features_in_ = X.shape[1]
        if self.backend not in ('auto', 'cpu', 'gpu'):
            raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
        if self.max_backtracks < 1:
            raise ValueError("max_backtracks must be at least 1")
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        # --- 1. Data Discretization and Encoding ---
        is_continuous = np.array([np.issubdtype(X[:, i].dtype, np.floating) for i in range(self.n_features_in_)])
        continuous_indices = np.where(is_continuous)[0]
        X_encoded = np.zeros_like(X, dtype=np.int32)
        n_states_features = np.zeros(self.n_features_in_, dtype=np.int32)
        if len(continuous_indices) > 0:
            discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy,
                                           subsample=None)
            X_encoded[:, continuous_indices] = discretizer.fit_transform(X[:, continuous_indices]).astype(np.int32)
            n_states_features[continuous_indices] = self.n_bins
        discrete_indices = np.where(~is_continuous)[0]
        if len(discrete_indices) > 0:
            for i in discrete_indices:
                unique_vals, encoded_col = np.unique(X[:, i], return_inverse=True)
                X_encoded[:, i] = encoded_col
                n_states_features[i] = len(unique_vals)
        unique_y, y_encoded = np.unique(y, return_inverse=True)
        n_states_y = len(unique_y)
        y_encoded = y_encoded.astype(np.int32)

        # --- 2. Backend Selection and Correlation Calculation ---
        if self.backend == 'auto':
            effective_backend = 'gpu' if is_cuda_ready() else 'cpu'
        else:
            effective_backend = self.backend

        if effective_backend == 'gpu':
            if not is_cuda_ready():
                raise RuntimeError("backend='gpu', but no CUDA-enabled GPU is available.")
            ensure_cuda_context()
            max_states = np.max(n_states_features)
            if n_states_y > 32 or max_states > 32:
                raise ValueError("GPU backend supports up to 32 unique states/bins.")
            X_d = cuda.to_device(X_encoded)
            y_d = cuda.to_device(y_encoded)
            n_states_features_d = cuda.to_device(n_states_features)
            r_cf_d = cuda.device_array(self.n_features_in_, dtype=np.float32)
            r_ff_d = cuda.device_array((self.n_features_in_, self.n_features_in_), dtype=np.float32)
            blocks_per_grid = self.n_features_in_
            threads_per_block = 1  # Each thread block handles one feature
            _precompute_correlations_gpu_kernel[blocks_per_grid, threads_per_block](
                X_d, y_d, n_states_features_d, n_states_y, r_cf_d, r_ff_d
            )
            r_cf_all = r_cf_d.copy_to_host()
            r_ff_matrix = r_ff_d.copy_to_host()

        else:  # --- CPU Backend Logic ---
            original_n_threads = numba.get_num_threads()
            n_threads = self.n_jobs if self.n_jobs != -1 else numba.config.NUMBA_DEFAULT_NUM_THREADS
            try:
                numba.set_num_threads(n_threads)
                r_cf_all, r_ff_matrix = _precompute_correlations_cpu(
                    X_encoded, y_encoded, n_states_features, n_states_y
                )
            finally:
                numba.set_num_threads(original_n_threads)

        # --- 3. Best First Search (on CPU) ---
        selected_indices_list = _best_first_search(
            self.n_features_in_,
            r_cf_all,
            r_ff_matrix,
            self.max_backtracks,
        )

        self.selected_indices_ = np.sort(np.array(list(selected_indices_list), dtype=int))
        self.support_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        if len(self.selected_indices_) > 0:
            self.support_mask_[self.selected_indices_] = True

        # --- 4. Calculate Final Merit Score ---
        k = len(self.selected_indices_)
        if k == 0:
            self.merit_ = 0.0
        else:
            sum_r_cf = np.sum(r_cf_all[self.selected_indices_])
            sum_r_ff = np.sum(np.triu(
                r_ff_matrix[np.ix_(self.selected_indices_, self.selected_indices_)], k=1)
            )
            self.merit_ = _cfs_merit(sum_r_cf, k, sum_r_ff)

        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.
        Required by the SelectorMixin.
        """
        check_is_fitted(self)
        return self.support_mask_

    def transform(self, X):
        """
        Reduces X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """

        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support_mask_]
        return X[:, self.support_mask_]
