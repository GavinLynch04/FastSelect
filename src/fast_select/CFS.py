import numpy as np
import numba
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer


@numba.njit(cache=True)
def _entropy(x, n_states):
    n_samples = x.shape[0]
    if n_samples == 0:
        return 0.0

    counts = np.zeros(n_states, dtype=np.float32)
    for i in range(n_samples):
        counts[x[i]] += 1.0

    p = counts / n_samples
    entropy = 0.0
    for prob in p:
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy

@numba.njit(cache=True)
def _mutual_information(x, y, n_states_x, n_states_y):
    n_samples = x.shape[0]
    if n_samples == 0:
        return 0.0

    # Create contingency table
    contingency_table = np.zeros((n_states_x, n_states_y), dtype=np.float32)
    for i in range(n_samples):
        contingency_table[x[i], y[i]] += 1.0

    p_xy = contingency_table / n_samples
    # Marginal probabilities
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    mi = 0.0
    for i in range(n_states_x):
        for j in range(n_states_y):
            # Corrected check for non-zero probabilities
            if p_xy[i, j] > 1e-12:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi

@numba.njit(cache=True)
def _symmetrical_uncertainty(x, y, n_states_x, n_states_y):
    h_x = _entropy(x, n_states_x)
    h_y = _entropy(y, n_states_y)
    
    if h_x + h_y == 0:
        return 0.0
        
    i_xy = _mutual_information(x, y, n_states_x, n_states_y)
    return 2.0 * i_xy / (h_x + h_y)


@numba.njit(parallel=True, cache=True)
def _precompute_correlations_parallel(X_encoded, y_encoded, n_states_features, n_states_y):
    """
    Calculates all correlations on pre-encoded integer data.
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

@numba.njit(parallel=True, cache=True)
def _best_first_search_optimized(n_features, r_cf_all, r_ff_matrix):
    """
    Performs an optimized best-first search using incremental updates.
    """
    remaining_indices = list(range(n_features))
    
    first_feature_idx = np.argmax(r_cf_all)
    selected_indices = numba.typed.List([first_feature_idx])
    remaining_indices.pop(first_feature_idx)

    current_sum_r_cf = r_cf_all[first_feature_idx]
    current_sum_r_ff = 0.0
    k = 1
    
    current_best_merit = current_sum_r_cf / np.sqrt(k + 2.0 * current_sum_r_ff)
    
    while len(remaining_indices) > 0:
        merits = np.full(len(remaining_indices), -1.0, dtype=np.float32)
        
        for i in numba.prange(len(remaining_indices)):
            candidate_idx = remaining_indices[i]
            
            next_sum_r_cf = current_sum_r_cf + r_cf_all[candidate_idx]
            
            sum_corr_with_selected = 0.0
            for sel_idx in selected_indices:
                sum_corr_with_selected += r_ff_matrix[candidate_idx, sel_idx]
                
            next_sum_r_ff = current_sum_r_ff + sum_corr_with_selected
            
            k_next = k + 1
            denominator = np.sqrt(k_next + 2.0 * next_sum_r_ff)
            if denominator > 1e-12:
                merits[i] = next_sum_r_cf / denominator

        if np.max(merits) <= -0.5:
             break

        best_candidate_local_idx = np.argmax(merits)
        merit_for_best_candidate = merits[best_candidate_local_idx]

        if merit_for_best_candidate > current_best_merit:
            current_best_merit = merit_for_best_candidate
            
            best_candidate_global_idx = remaining_indices.pop(best_candidate_local_idx)
            selected_indices.append(best_candidate_global_idx)
            
            current_sum_r_cf += r_cf_all[best_candidate_global_idx]
            sum_corr_with_selected = 0.0
            for sel_idx in selected_indices:
                if sel_idx != best_candidate_global_idx:
                    sum_corr_with_selected += r_ff_matrix[best_candidate_global_idx, sel_idx]
            current_sum_r_ff += sum_corr_with_selected
            k += 1
        else:
            break
            
    return selected_indices


class CFS(BaseEstimator, SelectorMixin):
    """
    Correlation-based Feature Selection (CFS) for discrete and continuous data.
    
    This selector evaluates subsets of features based on the hypothesis that a good
    feature subset contains features highly correlated with the class, yet
    uncorrelated with each other. Symmetrical Uncertainty is used as the
    correlation measure.

    The algorithm performs a greedy "best-first" search to find the best subset.
    Continuous features are automatically discretized using KBinsDiscretizer.
    """
    def __init__(self, n_bins=10, strategy='uniform', n_jobs=-1):
        """
        Parameters
        ----------
        n_bins : int, default=10
            The number of bins to use for discretizing continuous features.

        strategy : {'uniform', 'quantile', 'kmeans'}, default='uniform'
            Strategy used to define the widths of the bins for discretization.
            - 'uniform': All bins in each feature have identical widths.
            - 'quantile': All bins in each feature have the same number of points.
            - 'kmeans': Values in each bin have the same nearest center of a 1D
                        k-means cluster.

        n_jobs : int, default=-1
            The number of jobs to run in parallel for the computation.
            ``-1`` means using all available processors. ``1`` means no parallelism.
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None, ensure_min_samples=2, ensure_all_finite='allow-nan')
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns)
            
        is_continuous = [np.issubdtype(X[:, i].dtype, np.floating) for i in range(n_features)]
        continuous_indices = [i for i, is_cont in enumerate(is_continuous) if is_cont]
        
        X_encoded = np.zeros_like(X, dtype=np.int16)
        n_states_features = np.zeros(n_features, dtype=np.int16)

        if len(continuous_indices) > 0:
            discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
            X_encoded[:, continuous_indices] = discretizer.fit_transform(X[:, continuous_indices])
            n_states_features[continuous_indices] = self.n_bins

        discrete_indices = [i for i, is_cont in enumerate(is_continuous) if not is_cont]
        if len(discrete_indices) > 0:
            for i in discrete_indices:
                unique_vals, encoded_col = np.unique(X[:, i], return_inverse=True)
                X_encoded[:, i] = encoded_col
                n_states_features[i] = len(unique_vals)
        
        unique_y, y_encoded = np.unique(y, return_inverse=True)
        n_states_y = len(unique_y)

        original_n_threads = numba.get_num_threads()
        n_threads = self.n_jobs if self.n_jobs != -1 else numba.config.NUMBA_DEFAULT_NUM_THREADS
        try:
            numba.set_num_threads(n_threads)
            
            r_cf_all, r_ff_matrix = _precompute_correlations_parallel(
                X_encoded, y_encoded, n_states_features, n_states_y
            )
            
            if np.all(r_cf_all < 1e-12):
                selected_indices_list = []
            else:
                selected_indices_list = _best_first_search_optimized(
                    n_features, r_cf_all, r_ff_matrix
                )
        finally:
            numba.set_num_threads(original_n_threads)

        self.selected_indices_ = np.sort(np.array(list(selected_indices_list), dtype=int))
        self.support_mask_ = np.zeros(n_features, dtype=bool)
        if len(self.selected_indices_) > 0:
            self.support_mask_[self.selected_indices_] = True

        k = len(self.selected_indices_)
        if k == 0:
            self.merit_ = 0.0
        else:
            sum_r_cf = np.sum(r_cf_all[self.selected_indices_])
            sum_r_ff = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    sum_r_ff += r_ff_matrix[self.selected_indices_[i], self.selected_indices_[j]]
            denominator = np.sqrt(k + 2.0 * sum_r_ff)
            self.merit_ = sum_r_cf / denominator if denominator > 1e-12 else 0.0
            
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_mask_