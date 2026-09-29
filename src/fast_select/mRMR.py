from __future__ import annotations

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from . import mutual_information as mi
from .utils import check_integer_param, is_cuda_ready


@njit(parallel=True, cache=True)
def _encode_columns_numba(X, out):  # pragma: no cover
    """Write dense per-column category codes of numeric ``X`` into ``out``.

    Each column is encoded independently against its own sorted distinct values
    in the column's *own* dtype, so no two distinct symbols can collapse (mixing
    uint64 and int64 vocabularies would promote both to float64).  Returns the
    number of distinct symbols per column.
    """
    n_samples, n_features = X.shape
    n_states = np.empty(n_features, dtype=np.int64)
    for j in prange(n_features):
        column = X[:, j].copy()
        unique_values = np.unique(column)
        n_states[j] = unique_values.shape[0]
        for i in range(n_samples):
            out[i, j] = np.searchsorted(unique_values, column[i])
    return n_states


def _encode_categories(X, y):
    """Encode ``X`` (2-D) and ``y`` (1-D) into dense non-negative int32 codes.

    Every feature and the target are encoded independently and only by symbol
    identity; the empirical joint distribution of any pair is preserved
    exactly.  Float-coded categories (e.g. ``0.0`` / ``1.0``) are accepted.
    Returns ``(X_encoded, y_encoded, n_states_x, n_states_y)``.
    """
    n_samples, n_features = X.shape
    X_encoded = np.empty((n_samples, n_features), dtype=np.int32)
    if X.dtype.kind in "biuf":
        n_states_x = _encode_columns_numba(np.ascontiguousarray(X), X_encoded)
    else:  # strings / objects: not representable in the compiled encoder
        n_states_x = np.empty(n_features, dtype=np.int64)
        for j in range(n_features):
            uniques, inverse = np.unique(X[:, j], return_inverse=True)
            X_encoded[:, j] = inverse.reshape(-1)
            n_states_x[j] = uniques.shape[0]

    y_uniques, y_inverse = np.unique(y, return_inverse=True)
    y_encoded = y_inverse.reshape(-1).astype(np.int32)
    return X_encoded, y_encoded, n_states_x, int(y_uniques.shape[0])


class mRMR(TransformerMixin, BaseEstimator):
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

    backend : {'auto', 'cpu', 'gpu'}, default='auto'
        The computational backend to use. 'auto' runs on the GPU when a usable
        CUDA device is present and the encoded data has at most 32 distinct
        states, and falls back to the CPU otherwise. 'gpu' requires a compatible
        NVIDIA GPU and raises rather than falling back.

    Attributes
    ----------
    effective_backend_ : str
        The backend that actually ran during `fit`, 'cpu' or 'gpu'.

    """

    def __init__(self, n_features_to_select: int, method: str = "MID", backend: str = "auto"):
        # Per the scikit-learn estimator contract __init__ only stores the
        # parameters as given; every check happens in fit so that get_params,
        # set_params and clone round-trip without side effects.
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.backend = backend

    def _validate_parameters(self):
        """Validate constructor parameters. Called from fit, never from __init__."""
        if self.method not in ["MID", "MIQ"]:
            raise ValueError("Method must be either 'MID' or 'MIQ'.")
        if self.backend not in ["auto", "cpu", "gpu"]:
            raise ValueError("Backend must be 'auto', 'cpu', or 'gpu'.")
        if self.backend == "gpu" and not is_cuda_ready():
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
        self._validate_parameters()
        X, y = validate_data(
            self,
            X,
            y,
            dtype=None,
            y_numeric=True,
            ensure_2d=True,
        )
        self.n_features_in_ = X.shape[1]

        check_integer_param(self.n_features_to_select, "n_features_to_select", 1)
        if self.n_features_to_select > self.n_features_in_:
            raise ValueError(
                "n_features_to_select must be a positive integer less than or equal to the number of "
                f"features (got n_features_to_select={self.n_features_to_select}, n_features={self.n_features_in_})."
            )
        X_encoded, y_encoded, n_states_x, n_states_y = _encode_categories(X, y)

        # Same max_state that calculate_mi_matrices derives, so the reported
        # backend is the one that actually runs.
        max_state = int(max(n_states_x.max(), n_states_y))
        self.effective_backend_ = mi.resolve_backend(self.backend, max_state)

        relevance, redundancy = mi.calculate_mi_matrices(X_encoded, y_encoded, backend=self.backend, unit="bit")

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

            if self.method == "MID":
                scores = self.relevance_scores_[remaining_indices_arr] - (redundancy_sum[remaining_indices_arr] / i)
            else:  # 'MIQ'
                scores = self.relevance_scores_[remaining_indices_arr] / (
                    (redundancy_sum[remaining_indices_arr] / i) + 1e-9
                )
            max_score = np.max(scores)

            top_mask = np.isclose(scores, max_score, atol=1e-12)
            top_candidates = remaining_indices_arr[top_mask]
            if top_candidates.size > 1:
                avg_redundancy = redundancy_sum[top_candidates] / i
                best_feature_idx = top_candidates[np.argmin(avg_redundancy)]
            else:
                best_feature_idx = top_candidates[0]

            selected_indices[i] = best_feature_idx
            remaining_mask[best_feature_idx] = False

            redundancy_sum += self.redundancy_matrix_[:, best_feature_idx]

        self.top_features_ = selected_indices
        self.feature_importances_ = self.relevance_scores_

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=None)

        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
