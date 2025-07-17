from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class TuRF(BaseEstimator, TransformerMixin):
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
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.pct_remove = pct_remove
        self.n_iterations = n_iterations

        if not 0 < self.pct_remove < 1:
            raise ValueError("pct_remove must be between 0 and 1.")

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
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
                
        # Start with all features
        active_feature_indices = np.arange(self.n_features_in_)
        
        # The base estimator is cloned to avoid modifying the original object
        base_estimator = clone(self.estimator)

        # Run the first iteration on all features to get the initial scores
        base_estimator.fit(X, y)
        # The final scores are defined by the first run on the full feature set
        self.feature_importances_ = base_estimator.feature_importances_.copy()
        
        iteration = 0
        while True:
            if len(active_feature_indices) <= self.n_features_to_select:
                print("Stopping: Target number of features reached.")
                break
            if self.n_iterations is not None and iteration >= self.n_iterations:
                print(f"Stopping: Pre-defined number of iterations ({self.n_iterations}) reached.")
                break

            print(f"Iteration {iteration}: {len(active_feature_indices)} features remaining.")

            # Get the scores for the currently active features
            current_scores = self.feature_importances_[active_feature_indices]

            # Determine how many features to remove
            n_to_remove = int(len(active_feature_indices) * self.pct_remove)
            n_to_remove = max(1, n_to_remove)

            # If removing them would go below the target, adjust
            if len(active_feature_indices) - n_to_remove < self.n_features_to_select:
                n_to_remove = len(active_feature_indices) - self.n_features_to_select

            # Find the indices of the worst features *within the current subset*
            indices_of_worst_in_subset = np.argsort(current_scores)[:n_to_remove]
            
            # Remove these features from our list of active indices
            active_feature_indices = np.delete(active_feature_indices, indices_of_worst_in_subset)
            
            # --- Re-fit the estimator on the reduced feature set ---
            X_subset = X[:, active_feature_indices]
            base_estimator.fit(X_subset, y)

            self.feature_importances_[active_feature_indices] = base_estimator.feature_importances_
            
            iteration += 1

        # --- Final Selection ---
        # After the loop, `active_feature_indices` holds the final feature set.
        # We sort these final features by their last calculated importance scores.
        final_scores = self.feature_importances_[active_feature_indices]
        sorted_indices_in_subset = np.argsort(final_scores)[::-1]
        
        # Select the top N features from the final active set
        self.top_features_ = active_feature_indices[sorted_indices_in_subset][:self.n_features_to_select]
        self.top_features_ = np.sort(self.top_features_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces X to the selected features."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but was trained with {self.n_features_in_}."
            )
        return X[:, self.top_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
