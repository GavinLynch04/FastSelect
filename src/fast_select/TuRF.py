from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


class TuRF(TransformerMixin, BaseEstimator):
    """
    A meta-estimator that implements the Iterative Relief (TuRF) algorithm.

    TuRF iteratively removes features with the lowest scores as determined by a
    base Relief-based estimator. This process is repeated until a desired
    number of features remains, which can improve robustness against noise.

    This implementation is designed to wrap any scikit-learn compatible
    estimator that provides a `feature_importances_` attribute after fitting,
    such as the `ReliefF`, `SURF`, or `MultiSURF` classes in this library.

    Parameters
    ----------
    estimator : estimator object
        The base estimator to use for scoring features at each iteration.
        This object is cloned and not modified.
    n_features_to_select : int, default=10
        The final number of features to select.
    pct_remove : float, default=0.1
        The percentage of the remaining features to remove at each iteration.
        Must be between 0 and 1.
    n_iterations : int or None, default=None
        The number of iterations to run. If None, the process continues until
        the number of features is less than or equal to `n_features_to_select`.
    verbose : bool, default=False
        Controls whether progress updates are printed during the fit.
        Limited benefit currently, will be expanded in future versions.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.
    feature_importances_ : ndarray of shape (n_features_in_,)
        The feature importance scores calculated by the base estimator on the
        **full, original feature set** during the first iteration.
    top_features_ : ndarray of shape (n_features_to_select,)
        The indices of the selected top features, sorted by importance.
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: int = 10,
        pct_remove: float = 0.1,
        n_iterations: int | None = None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.pct_remove = pct_remove
        self.n_iterations = n_iterations
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the TuRF model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(
            self, X, y, y_numeric=True, dtype=np.float64, ensure_2d=True,
        )
        self.n_features_in_ = X.shape[1]
        if not 0 < self.pct_remove < 1:
            raise ValueError("pct_remove must be between 0 and 1.")

        active_feature_indices = np.arange(self.n_features_in_)
        base_estimator = clone(self.estimator)

        base_estimator.fit(X, y)
        self.feature_importances_ = base_estimator.feature_importances_.copy()

        current_scores = self.feature_importances_.copy()

        iteration = 0
        while True:
            if len(active_feature_indices) <= self.n_features_to_select:
                break
            if self.n_iterations is not None and iteration >= self.n_iterations:
                break

            n_to_remove = int(len(active_feature_indices) * self.pct_remove)
            n_to_remove = max(1, n_to_remove)
            if len(active_feature_indices) - n_to_remove < self.n_features_to_select:
                n_to_remove = len(active_feature_indices) - self.n_features_to_select

            indices_of_worst_in_subset = np.argsort(current_scores)[:n_to_remove]

            active_feature_indices = np.delete(active_feature_indices, indices_of_worst_in_subset)
            
            if self.verbose:
                print(f"Iteration {iteration}: {len(active_feature_indices)} features remaining.")
            X_subset = X[:, active_feature_indices]
            base_estimator.fit(X_subset, y)

            current_scores = base_estimator.feature_importances_

            iteration += 1

        sorted_indices_in_subset = np.argsort(current_scores)[::-1]
        self.top_features_ = active_feature_indices[sorted_indices_in_subset]
        self.top_features_ = np.sort(self.top_features_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to the selected features."""
        check_is_fitted(self)
        X = validate_data(
            self, X,
            reset=False,
            dtype=[np.float64, np.float32]
        )

        return X[:, self.top_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
