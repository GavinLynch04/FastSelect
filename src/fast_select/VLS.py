"""VLS: very-large-scale Relief, scoring random feature subsets and keeping each feature's best.

Source
------
Eppstein and Haake, *Very large scale ReliefF for genome-wide association
analysis*, IEEE CIBCB 2008, pp. 112-119, DOI 10.1109/CIBCB.2008.4675767, as
described by Urbanowicz et al., *Relief-Based Feature Selection: Introduction
and Review* (J. Biomed. Inform. 2018), Section 3.4: ReliefF is applied to
``S`` randomly selected feature subsets of size ``a_s < a`` and the global
weight of a feature is the maximum of its subset ("local") weights.  The
original article could not be consulted; how its subsets are sized, counted
and drawn is therefore a documented convention here (see :class:`VLS`).
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state, get_tags
from sklearn.utils.validation import check_is_fitted, validate_data

from .utils import check_integer_param


class VLS(TransformerMixin, BaseEstimator):
    """VLS (VLSReliefF): a wrapper that scores random feature subsets with a Relief estimator.

    Relief scores degrade as irrelevant features crowd the distance used to
    find neighbours. VLS (Eppstein and Haake, 2008) fits the base estimator on
    many random subsets of the features, where each subset's distances involve
    far fewer features, and gives every feature the largest score it received
    in any subset. Wrap it in :class:`TuRF` for the iterative variant.

    Parameters
    ----------
    estimator : estimator object
        The Relief-family estimator fitted on every subset (cloned; only its
        ``feature_importances_`` are used). If it has ``n_features_to_select``,
        the clone keeps every subset feature.

    n_features_to_select : int or float, default=0.2
        The number of top features to select. A float selects that fraction of
        the features; an int selects that many.

    subset_size : int or float, default=0.1
        Features per subset. A float is a fraction of the features (at least
        one); an int is a count, capped at the number of features.

    n_subsets : int or None, default=None
        The number of subsets. None uses ``2 * ceil(n_features / subset_size)``,
        so each feature is scored about twice. It must be at least
        ``ceil(n_features / subset_size)``, so that every feature is scored.

    random_state : int, RandomState instance or None, default=None
        Controls the random subsets.

    verbose : bool, default=False
        Controls whether progress messages are printed during fit.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        Each feature's largest score over the subsets that contain it.

    subsets_ : ndarray of shape (n_subsets_, subset_size_)
        The feature indices of every subset, in the order they were scored.

    subset_scores_ : ndarray of shape (n_subsets_, subset_size_)
        The base estimator's score for each subset feature.

    n_subsets_, subset_size_ : int
        The resolved subset count and size.

    top_features_ : ndarray of shape (n_select,)
        Indices of the selected features, best first.

    Notes
    -----
    The original article could not be consulted, so the subset scheme is a
    convention: the first ``ceil(n_features / subset_size)`` subsets are a
    random partition of the features (the last one filled up with features
    drawn from outside it, so all subsets have the same size), which
    guarantees every feature a score; each further subset is drawn uniformly
    without replacement. The paper derives the number and size of subsets from
    the interaction order to be detected; choose them for the problem rather
    than relying on the defaults. Score ties between features keep the order
    of :func:`numpy.argsort`, as in the other Relief estimators.
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: int | float = 0.2,
        subset_size: int | float = 0.1,
        n_subsets: int | None = None,
        random_state=None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.subset_size = subset_size
        self.n_subsets = n_subsets
        self.random_state = random_state
        self.verbose = verbose

    def _resolve(self, n_features: int) -> tuple[int, int, int]:
        size = self.subset_size
        if isinstance(size, (float, np.floating)) and not isinstance(size, bool):
            if not 0.0 < size <= 1.0:
                raise ValueError(f"A float subset_size must be in (0, 1], got {size!r}.")
            size = max(1, int(round(size * n_features)))
        else:
            # Capped, so VLS keeps working inside TuRF as the feature set shrinks.
            size = min(check_integer_param(size, "subset_size", 1), n_features)
        partition = math.ceil(n_features / size)
        if self.n_subsets is None:
            count = 2 * partition
        else:
            count = check_integer_param(self.n_subsets, "n_subsets", 1)
            if count < partition:
                raise ValueError(
                    f"n_subsets ({count}) must be at least ceil(n_features / subset_size) = {partition} "
                    "so that every feature is scored."
                )

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            n_select = max(1, int(self.n_features_to_select * n_features))
        elif isinstance(self.n_features_to_select, (int, np.integer)) and not isinstance(
            self.n_features_to_select, (bool, np.bool_)
        ):
            if not 0 < self.n_features_to_select <= n_features:
                raise ValueError(
                    f"If n_features_to_select is an int ({self.n_features_to_select}), "
                    f"it must be > 0 and <= n_features ({n_features})."
                )
            n_select = int(self.n_features_to_select)
        else:
            raise TypeError("n_features_to_select must be an int or a float.")
        return size, count, n_select

    @staticmethod
    def _draw_subsets(n_features, size, count, rng):
        """A random partition padded to ``size``, then uniform subsets, as an ``(count, size)`` array."""
        subsets = np.empty((count, size), dtype=np.intp)
        order = rng.permutation(n_features)
        partition = math.ceil(n_features / size)
        for s in range(partition):
            block = order[s * size : (s + 1) * size]
            if block.size < size:
                outside = np.setdiff1d(np.arange(n_features), block, assume_unique=True)
                block = np.concatenate([block, rng.choice(outside, size - block.size, replace=False)])
            subsets[s] = block
        for s in range(partition, count):
            subsets[s] = rng.choice(n_features, size, replace=False)
        return subsets

    def _base_accepts_nan(self) -> bool:
        return bool(
            getattr(self.estimator, "_accepts_nan_in_discrete_features", False)
            or get_tags(self.estimator).input_tags.allow_nan
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Score every subset with the base estimator and keep each feature's best score."""
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_2d=True, y_numeric=True, ensure_all_finite="allow-nan"
        )
        if not self._base_accepts_nan() and np.isnan(X).any():
            raise ValueError("Input X contains NaN, and the base estimator does not handle missing values.")
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        size, count, n_select = self._resolve(n_features)
        rng = check_random_state(self.random_state)
        subsets = self._draw_subsets(n_features, size, count, rng)

        base = clone(self.estimator)
        if "n_features_to_select" in base.get_params(deep=False):
            base.set_params(n_features_to_select=size)
        scores = np.empty((count, size), dtype=np.float64)
        best = np.full(n_features, -np.inf)
        for s, subset in enumerate(subsets):
            base.fit(X[:, subset], y)
            local = np.asarray(base.feature_importances_, dtype=np.float64)
            scores[s] = local
            np.maximum.at(best, subset, local)
            if self.verbose and (s + 1) % max(1, count // 10) == 0:
                print(f"VLS: scored {s + 1}/{count} subsets.")

        self.estimator_ = base
        if hasattr(base, "effective_backend_"):
            self.effective_backend_ = base.effective_backend_
        self.subsets_ = subsets
        self.subset_scores_ = scores
        self.n_subsets_ = count
        self.subset_size_ = size
        self.feature_importances_ = best
        self.top_features_ = np.argsort(best)[::-1][:n_select]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce ``X`` to the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32], ensure_all_finite="allow-nan")
        if not self._base_accepts_nan() and np.isnan(X).any():
            raise ValueError("Input X contains NaN, and the base estimator does not handle missing values.")
        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
