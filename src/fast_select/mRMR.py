from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from . import mutual_information as mi
from .utils import check_integer_param, is_cuda_ready, resolve_num_threads

MI_METHODS = ("MID", "MIQ")
F_METHODS = ("FCD", "FCQ")


@njit(parallel=True, cache=True)
def _anova_f_and_unit_rows(unit, y_enc, counts):  # pragma: no cover
    """One-way ANOVA F of every feature (Ding & Peng 2005, Eq. 8), standardising it in place.

    ``unit`` arrives as ``X.T`` (one float64 row per feature) and leaves with
    each row replaced by ``(x_f - mean) / ||x_f - mean||``, so that
    ``unit[a] . unit[b]`` is the Pearson correlation.  Constant features and
    classes are detected exactly from their min and max, not from a rounded
    variance: a constant feature has ``F = 0`` and a zero row (correlation 0
    with everything), and a feature that is constant within every class but
    differs between classes has ``F = inf``.
    """
    n_features, n_samples = unit.shape
    n_classes = counts.shape[0]
    f_stat = np.empty(n_features, dtype=np.float64)
    df_between = n_classes - 1
    df_within = n_samples - n_classes
    for f in prange(n_features):
        column = unit[f]
        sums = np.zeros(n_classes)
        lows = np.full(n_classes, np.inf)
        highs = np.full(n_classes, -np.inf)
        total = 0.0
        for i in range(n_samples):
            value = column[i]
            c = y_enc[i]
            sums[c] += value
            total += value
            lows[c] = min(lows[c], value)
            highs[c] = max(highs[c], value)
        mean = total / n_samples
        constant_within = True
        low = np.inf
        high = -np.inf
        for c in range(n_classes):
            if lows[c] != highs[c]:
                constant_within = False
            low = min(low, lows[c])
            high = max(high, highs[c])
        if low == high:
            f_stat[f] = 0.0
            for i in range(n_samples):
                unit[f, i] = 0.0
            continue
        between = 0.0
        for c in range(n_classes):
            class_mean = sums[c] / counts[c]
            between += counts[c] * (class_mean - mean) ** 2
        within = 0.0
        spread = 0.0
        for i in range(n_samples):
            value = column[i]
            d = value - sums[y_enc[i]] / counts[y_enc[i]]
            within += d * d
            e = value - mean
            spread += e * e
            unit[f, i] = e
        if constant_within:
            f_stat[f] = np.inf if between > 0.0 else 0.0
        else:
            f_stat[f] = (between / df_between) / (within / df_within)
        scale = 1.0 / math.sqrt(spread)
        for i in range(n_samples):
            unit[f, i] *= scale
    return f_stat


@njit(parallel=True, cache=True)
def _abs_correlation_with(unit, j, out):  # pragma: no cover
    """``out[f] = |Pearson(f, j)|`` from the unit columns, clipped to 1 against rounding."""
    n_features, n_samples = unit.shape
    reference = unit[j]
    for f in prange(n_features):
        row = unit[f]
        total = 0.0
        for i in range(n_samples):
            total += row[i] * reference[i]
        if f == j:
            out[f] = 1.0 if total > 0.0 else 0.0  # exactly 1 unless the column is constant
        else:
            out[f] = min(abs(total), 1.0)


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

    Minimum redundancy, maximum relevance selects features greedily: the first
    is the most relevant to the target, and each next one maximises relevance
    combined with its mean redundancy to the features already chosen. Four
    criteria are available, as in Ding and Peng (2005), Table 1:

    * ``'MID'`` / ``'MIQ'`` (discrete data; Peng, Long and Ding, 2005):
      relevance ``I(f; y)`` and redundancy ``mean_{s in S} I(f; s)``, combined
      by difference or quotient.
    * ``'FCD'`` / ``'FCQ'`` (continuous data; Ding and Peng, 2005, Eqs. 8 and
      10): relevance is the one-way ANOVA F statistic of the feature across the
      classes and redundancy is ``mean_{s in S} |Pearson(f, s)|``, combined by
      difference or quotient. FCQ is the default of the widely used
      ``mrmr_selection`` package.

    Parameters
    ----------
    n_features_to_select : int
        The number of top features to select.

    method : {'MID', 'MIQ', 'FCD', 'FCQ'}, default='MID'
        The mRMR selection criterion to use.
        - 'MID' (Mutual Information Difference): f_score = I(f; y) - mean(I(f; S))
        - 'MIQ' (Mutual Information Quotient): f_score = I(f; y) / mean(I(f; S))
        - 'FCD' (F-test Correlation Difference): F(f, y) - mean(|c(f, S)|)
        - 'FCQ' (F-test Correlation Quotient): F(f, y) / mean(|c(f, S)|)

    backend : {'auto', 'cpu', 'gpu', 'metal'}, default='auto'
        'auto' prefers CUDA (up to 32 encoded states), then optional Metal
        (up to 64 states), then CPU. 'gpu' explicitly requests CUDA; 'metal'
        explicitly requests MLX/Metal. Explicit GPU requests raise when the
        device or input is unsupported. Metal uses exact GPU joint counts and
        host precision for mutual information and the selection search. FCD
        and FCQ run on the CPU only ('auto' or 'cpu').

    n_jobs : int, default=-1
        Numba threads for FCD and FCQ; ``-1`` uses the configured pool. The
        mutual-information methods use their backend's own parallelism.

    Attributes
    ----------
    effective_backend_ : str
        The backend that actually ran during `fit`: 'cpu', 'gpu', or 'metal'.

    top_features_ : ndarray of shape (n_features_to_select,)
        The selected features in the order they were chosen.

    relevance_scores_ : ndarray of shape (n_features,)
        ``I(f; y)`` in bits for MID/MIQ, the F statistic for FCD/FCQ. Also
        exposed as ``feature_importances_``.

    redundancy_matrix_ : ndarray of shape (n_features, n_features)
        MID/MIQ only: the pairwise mutual information between features.

    selected_correlations_ : ndarray of shape (n_features_to_select, n_features)
        FCD/FCQ only: row ``r`` holds ``|Pearson|`` between the ``r``-th
        selected feature and every feature (no ``p x p`` matrix is formed).

    Notes
    -----
    Ding and Peng define FCD/FCQ for classification, which is what is
    implemented (targets with at least two classes and more samples than
    classes). Conventions the paper leaves open: a constant feature has
    ``F = 0`` and correlation 0 with every feature; a feature that is constant
    within every class but differs between classes has ``F = inf``; in FCQ a
    zero mean redundancy gives ``+inf`` when ``F > 0`` and 0 when ``F = 0``.
    The redundancy is not floored (``mrmr_selection`` floors it at 0.001) and
    zero-relevance features stay candidates (``mrmr_selection`` drops them).
    For every method, candidates whose scores tie within
    :func:`numpy.isclose` are resolved by the lower mean redundancy, then the
    lower feature index.
    """

    def __init__(self, n_features_to_select: int, method: str = "MID", backend: str = "auto", n_jobs: int = -1):
        # Per the scikit-learn estimator contract __init__ only stores the
        # parameters as given; every check happens in fit so that get_params,
        # set_params and clone round-trip without side effects.
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.backend = backend
        self.n_jobs = n_jobs

    def _validate_parameters(self):
        """Validate constructor parameters. Called from fit, never from __init__."""
        if self.method not in MI_METHODS + F_METHODS:
            raise ValueError("Method must be one of 'MID', 'MIQ', 'FCD' or 'FCQ'.")
        if self.backend not in ["auto", "cpu", "gpu", "metal"]:
            raise ValueError("Backend must be 'auto', 'cpu', 'gpu', or 'metal'.")
        if self.method in F_METHODS:
            if self.backend not in ("auto", "cpu"):
                raise ValueError("FCD and FCQ run on the CPU only; use backend='auto' or 'cpu'.")
            resolve_num_threads(self.n_jobs)
            return
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
        continuous = self.method in F_METHODS
        X, y = validate_data(
            self,
            X,
            y,
            dtype=np.float64 if continuous else None,
            y_numeric=not continuous,
            ensure_2d=True,
        )
        self.n_features_in_ = X.shape[1]

        check_integer_param(self.n_features_to_select, "n_features_to_select", 1)
        if self.n_features_to_select > self.n_features_in_:
            raise ValueError(
                "n_features_to_select must be a positive integer less than or equal to the number of "
                f"features (got n_features_to_select={self.n_features_to_select}, n_features={self.n_features_in_})."
            )
        if continuous:
            return self._fit_continuous(X, y)
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

    def _fit_continuous(self, X, y):
        """FCD / FCQ (Ding and Peng 2005, Table 1) with incremental correlations."""
        check_classification_targets(y)
        classes, y_encoded = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        if classes.size < 2:
            raise ValueError(
                "FCD and FCQ need at least two classes; got one class, for which the F statistic is undefined."
            )
        if n_samples <= classes.size:
            raise ValueError(
                f"FCD and FCQ need more samples than classes (got {n_samples} samples, {classes.size} classes)."
            )
        self.classes_ = classes
        self.effective_backend_ = "cpu"
        counts = np.bincount(y_encoded).astype(np.float64)
        k = int(self.n_features_to_select)
        unit = np.ascontiguousarray(X.T, dtype=np.float64)  # standardised in place, one row per feature
        correlations = np.empty((k, n_features), dtype=np.float64)
        original_num_threads = get_num_threads()
        set_num_threads(resolve_num_threads(self.n_jobs))
        try:
            relevance = _anova_f_and_unit_rows(unit, y_encoded.astype(np.int64), counts)
            selected = np.zeros(k, dtype=np.int32)
            remaining = np.ones(n_features, dtype=bool)
            selected[0] = np.argmax(relevance)
            remaining[selected[0]] = False
            _abs_correlation_with(unit, selected[0], correlations[0])
            redundancy_sum = correlations[0].copy()
            for step in range(1, k):
                candidates = np.flatnonzero(remaining)
                mean_redundancy = redundancy_sum[candidates] / step
                if self.method == "FCD":
                    scores = relevance[candidates] - mean_redundancy
                else:  # FCQ
                    with np.errstate(divide="ignore", invalid="ignore"):
                        scores = np.where(
                            mean_redundancy > 0.0,
                            relevance[candidates] / mean_redundancy,
                            np.where(relevance[candidates] > 0.0, np.inf, 0.0),
                        )
                top = candidates[np.isclose(scores, np.max(scores), atol=1e-12)]
                best = top[np.argmin(redundancy_sum[top])] if top.size > 1 else top[0]
                selected[step] = best
                remaining[best] = False
                _abs_correlation_with(unit, best, correlations[step])
                redundancy_sum += correlations[step]
        finally:
            set_num_threads(original_num_threads)

        self.relevance_scores_ = relevance
        self.selected_correlations_ = correlations
        self.top_features_ = selected
        self.feature_importances_ = relevance
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
