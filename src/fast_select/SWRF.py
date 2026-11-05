"""SWRF family: Relief with sigmoid (spatially weighted) neighbour influence.

Sources
-------
SWRF*
    Stokes and Visweswaran, *Application of a spatially-weighted Relief
    algorithm for ranking genetic predictors of disease*, BioData Mining 5:20
    (2012), DOI 10.1186/1756-0381-5-20.  Its Modular Relief Framework (MoRF)
    gives the update every estimator in this module uses, Eqs. (1)-(4)::

        dW_i[a] = sum_j c(i, j) f(d_ij) diff(a, i, j) / sum_j |f(d_ij)|
        W[a]    = (1 / n) sum_i dW_i[a]

    with ``c = -1`` for a hit and ``+1`` for a miss (Eq. 2), ``diff`` the
    per-feature difference (Eq. 3) and ``d_ij = sum_a diff(a, i, j)`` (Eq. 4).
    SWRF*'s neighbour weight is ``f(d) = 2 / (1 + exp(-(t - d) / (u / 4))) - 1``
    with ``t`` and ``u`` the mean and standard deviation of the pairwise
    distances.  The authors' MoRF code skips ``j = i`` in both sums.

SWRF, MultiSWRF, MultiSWRF*, MultiSWRFDB, MultiSWRFDB*
    Kazemi-Nia, Bandhey, Freda and Urbanowicz, *Advancing Interaction-Sensitive
    Feature Selection: Novel Relief-Based Algorithms, Expanded Comparisons, and
    Recommendations for Biomedical Data Mining*, arXiv:2608.28552 (2026),
    Eqs. (3)-(7).  That paper defines only the neighbour weights ``w_ij``; it
    does not state a score update, so its weights are used in MoRF Eq. (1), the
    framework the SWRF* weighting it extends was defined in.

Every weight is written here as a hyperbolic tangent, which is algebraically
identical to the papers' logistic form (``2 / (1 + exp(x)) - 1 = -tanh(x / 2)``)
and cannot overflow.
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cpu_backend import resolve_cpu_backend
from .SURF import _global_mean_distance_cpu
from .utils import (
    BOUNDARY_RTOL,
    check_integer_param,
    discrete_feature_mask,
    prepare_relief_matrix,
    resolve_num_threads,
    split_discrete_last,
)

# Reassociation lets the per-feature loops vectorise; approximate functions
# ('afn') and reciprocals ('arcp') stay off so tanh and the divisions are exact
# library results.
_FASTMATH = {"reassoc", "contract", "nsz"}


@njit(inline="always", fastmath=_FASTMATH)
def _pair_distance(x_i, x_j, n_cont, n_features):  # pragma: no cover
    """MoRF Eq. (4): the sum of the per-feature differences.

    Continuous columns ``[0, n_cont)`` are range-normalised, so their difference
    is ``|a - b|``; discrete columns after them differ by 0 or 1.
    """
    dist = 0.0
    for f in range(n_cont):
        dist += abs(x_i[f] - x_j[f])
    for f in range(n_cont, n_features):
        if x_i[f] != x_j[f]:
            dist += 1.0
    return dist


@njit(inline="always", fastmath=_FASTMATH)
def _row_deviation_sums(x, row, n_cont, n_features, mean):  # pragma: no cover
    """``sum(d - mean)`` and ``sum((d - mean)^2)`` over the pairs ``(row, j > row)``."""
    x_i = x[row]
    dev_sum = 0.0
    sq_sum = 0.0
    for j in range(row + 1, x.shape[0]):
        dev = _pair_distance(x_i, x[j], n_cont, n_features) - mean
        dev_sum += dev
        sq_sum += dev * dev
    return dev_sum, sq_sum


@njit(parallel=True, fastmath=_FASTMATH)
def _global_distance_std_cpu(x, n_cont, mean):  # pragma: no cover
    """Population standard deviation of the distances over unique sample pairs.

    ``mean`` is the exact-formula pair mean from ``_global_mean_distance_cpu``.
    The corrected two-pass form ``E[(d - m)^2] - (E[d - m])^2`` removes the
    small error left in ``mean`` instead of squaring it into the variance.
    Row ``k`` (``n - 1 - k`` pairs) is processed with row ``n - 2 - k``
    (``k + 1`` pairs) so every parallel iteration has the same work.
    """
    n_samples, n_features = x.shape
    n_pairs = n_samples * (n_samples - 1) // 2
    if n_pairs == 0:
        return 0.0
    n_threads = get_num_threads()
    partial_dev = np.zeros(n_threads, dtype=np.float64)
    partial_sq = np.zeros(n_threads, dtype=np.float64)

    for k in prange(n_samples // 2):
        tid = get_thread_id()
        dev_sum, sq_sum = _row_deviation_sums(x, k, n_cont, n_features, mean)
        partner = n_samples - 2 - k
        if partner > k:
            partner_dev, partner_sq = _row_deviation_sums(x, partner, n_cont, n_features, mean)
            dev_sum += partner_dev
            sq_sum += partner_sq
        partial_dev[tid] += dev_sum
        partial_sq[tid] += sq_sum

    dev_total = 0.0
    sq_total = 0.0
    for t in range(n_threads):
        dev_total += partial_dev[t]
        sq_total += partial_sq[t]
    mean_dev = dev_total / n_pairs
    variance = sq_total / n_pairs - mean_dev * mean_dev
    return math.sqrt(variance) if variance > 0.0 else 0.0


@njit(parallel=True, fastmath=_FASTMATH)
def _swrf_cpu_kernel(
    x, y, n_cont, per_target, global_mean, global_std, use_star, dead_band, scores_out
):  # pragma: no cover
    """MoRF Eq. (1) summed over every target instance, divided by ``n``.

    For each target ``i`` the neighbour weight is centred on ``mu`` (the global
    pair mean, or target ``i``'s mean distance to the others when
    ``per_target``) with width ``sigma``.  Without ``dead_band`` both centres
    are ``mu``; with it they are ``mu - sigma / 2`` (near) and
    ``mu + sigma / 2`` (far).  A distance strictly below the near centre has
    weight ``tanh(2 (near - d) / sigma)``; with ``use_star`` a distance strictly
    above the far centre has weight ``-tanh(2 (d - far) / sigma)``; every other
    instance has weight zero.  Distances within ``BOUNDARY_RTOL * mu`` of a
    centre are ties: the weight there is zero in exact arithmetic, and the tie
    rule keeps rounding noise from being amplified by a near-zero ``sigma``.
    """
    n_samples, n_features = x.shape
    n_threads = get_num_threads()
    thread_scores = np.zeros((n_threads, n_features), dtype=np.float64)
    thread_dists = np.empty((n_threads, n_samples), dtype=np.float64)
    thread_weights = np.empty((n_threads, n_samples), dtype=np.float64)
    inv_others = 1.0 / (n_samples - 1)

    for i in prange(n_samples):
        tid = get_thread_id()
        dists = thread_dists[tid]
        weights = thread_weights[tid]
        x_i = x[i]

        dist_sum = 0.0
        for j in range(n_samples):
            if j == i:
                dists[j] = 0.0
                continue
            d = _pair_distance(x_i, x[j], n_cont, n_features)
            dists[j] = d
            dist_sum += d

        if per_target:
            mu = dist_sum * inv_others
            dev_sum = 0.0
            sq_sum = 0.0
            for j in range(n_samples):
                if j != i:
                    dev = dists[j] - mu
                    dev_sum += dev
                    sq_sum += dev * dev
            mean_dev = dev_sum * inv_others
            variance = sq_sum * inv_others - mean_dev * mean_dev
            sigma = math.sqrt(variance) if variance > 0.0 else 0.0
        else:
            mu = global_mean
            sigma = global_std

        near_centre = mu
        far_centre = mu
        if dead_band:
            near_centre = mu - 0.5 * sigma
            far_centre = mu + 0.5 * sigma
        tolerance = BOUNDARY_RTOL * abs(mu)
        near_limit = near_centre - tolerance
        far_limit = far_centre + tolerance
        steepness = 2.0 / sigma if sigma > 0.0 else 0.0

        weight_total = 0.0
        for j in range(n_samples):
            w = 0.0
            if j != i and sigma > 0.0:
                d = dists[j]
                if d < near_limit:
                    w = math.tanh(steepness * (near_centre - d))
                elif use_star and d > far_limit:
                    w = -math.tanh(steepness * (d - far_centre))
            weights[j] = w
            weight_total += abs(w)

        if weight_total == 0.0:
            continue  # no instance carries weight: dW_i = 0
        scale = 1.0 / (weight_total * n_samples)

        scores_i = thread_scores[tid]
        y_i = y[i]
        for j in range(n_samples):
            w = weights[j]
            if w == 0.0:
                continue
            # MoRF Eq. (2): c = -1 for a hit, +1 for a miss.
            coefficient = -w * scale if y[j] == y_i else w * scale
            x_j = x[j]
            for f in range(n_cont):
                scores_i[f] += coefficient * abs(x_i[f] - x_j[f])
            for f in range(n_cont, n_features):
                if x_i[f] != x_j[f]:
                    scores_i[f] += coefficient

    for f in range(n_features):
        total = 0.0
        for t in range(n_threads):
            total += thread_scores[t, f]
        scores_out[f] = total


def _swrf_cpu_host_caller(x, y, n_cont, per_target, use_star, dead_band, discrete_columns, n_jobs):
    """Run the SWRF-family kernel; returns ``(scores, mean, std)`` (global stats or NaN)."""
    scores = np.zeros(x.shape[1], dtype=np.float64)
    original_num_threads = get_num_threads()
    set_num_threads(resolve_num_threads(n_jobs))
    try:
        if per_target:
            mean = std = float("nan")
            _swrf_cpu_kernel(x, y, n_cont, True, 0.0, 0.0, use_star, dead_band, scores)
        else:
            mean = float(_global_mean_distance_cpu(x, discrete_columns))
            std = float(_global_distance_std_cpu(x, n_cont, mean))
            _swrf_cpu_kernel(x, y, n_cont, False, mean, std, use_star, dead_band, scores)
    finally:
        set_num_threads(original_num_threads)
    return scores, mean, std


def _check_flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {value!r}.")
    return bool(value)


class _SWRFBase(TransformerMixin, BaseEstimator):
    """Shared fit/transform for the SWRF family; subclasses fix the neighbourhood."""

    _per_target = False

    def _neighbourhood(self):
        """Return ``(use_star, dead_band)`` after validating the subclass flags."""
        raise NotImplementedError

    def _algorithm_name(self, use_star, dead_band):
        raise NotImplementedError

    def _validate_parameters(self, n_samples: int, n_features: int) -> int:
        if n_samples < 2:
            raise ValueError(f"{type(self).__name__} requires at least 2 samples, but got n_samples = {n_samples}")

        check_integer_param(self.discrete_limit, "discrete_limit", 0)
        resolve_num_threads(self.n_jobs)

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            return max(1, int(self.n_features_to_select * n_features))
        if isinstance(self.n_features_to_select, (int, np.integer)) and not isinstance(
            self.n_features_to_select, (bool, np.bool_)
        ):
            if not 0 < self.n_features_to_select <= n_features:
                raise ValueError(
                    f"If n_features_to_select is an int ({self.n_features_to_select}), "
                    f"it must be > 0 and <= n_features ({n_features})."
                )
            return int(self.n_features_to_select)
        raise TypeError("n_features_to_select must be an int or a float.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Score every feature with MoRF Eq. (1) and keep the best ``n_features_to_select``."""
        X, y = validate_data(self, X, y, dtype=[np.float64, np.float32], ensure_2d=True, y_numeric=True)
        check_classification_targets(y)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        n_select = self._validate_parameters(n_samples, self.n_features_in_)
        use_star, dead_band = self._neighbourhood()

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.is_discrete_ = discrete_feature_mask(X, self.discrete_limit)
        if not self._per_target:
            self.distance_mean_ = float("nan")
            self.distance_std_ = float("nan")
        if len(self.classes_) < 2:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
            self.top_features_ = np.arange(n_select)
            return self

        columns, n_cont = split_discrete_last(self.is_discrete_)
        X_prepared = prepare_relief_matrix(
            X, self.is_discrete_, columns, dtype=np.float64, discrete_limit=self.discrete_limit
        )
        if self.verbose:
            print(f"Running {self._algorithm_name(use_star, dead_band)} on the CPU now...")
        permuted_scores, mean, std = _swrf_cpu_host_caller(
            X_prepared,
            y_encoded.astype(np.int32),
            n_cont,
            self._per_target,
            use_star,
            dead_band,
            np.ascontiguousarray(self.is_discrete_[columns]),
            self.n_jobs,
        )
        if not self._per_target:
            self.distance_mean_ = mean
            self.distance_std_ = std

        scores = np.zeros(self.n_features_in_, dtype=np.float64)
        scores[columns] = permuted_scores
        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce ``X`` to the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)


class SWRF(_SWRFBase):
    """SWRF and SWRF*: sigmoid-weighted Relief around the global mean distance.

    Every instance is the target once. Its neighbours are weighted by a sigmoid
    of their distance, centred on the mean ``mu`` of all pairwise distances with
    width set by their standard deviation ``sigma``:
    ``w(d) = 2 / (1 + exp((d - mu) * 4 / sigma)) - 1``, close to 1 for near
    instances, 0 at ``mu`` and close to -1 for distant ones. Scores follow the
    Modular Relief Framework (MoRF) update of Stokes and Visweswaran (2012),
    Eq. (1): ``sum_j c * w * diff / sum_j |w|`` per target (``c = -1`` for a
    hit, ``+1`` for a miss), averaged over the targets.

    * ``use_star=True`` is **SWRF*** (Stokes and Visweswaran, 2012): every
      instance is weighted, so distant instances (negative weight) reverse the
      update, as in SURF*.
    * ``use_star=False`` is **SWRF** (Kazemi-Nia et al., 2026, Eq. 4): only
      instances strictly closer than ``mu`` are weighted.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select. A float selects that fraction of
        the features; an int selects that many.

    use_star : bool, default=False
        If True, runs SWRF*, which also scores instances farther than the mean.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    backend : {'auto', 'cpu'}, default='auto'
        The SWRF family runs on the CPU only; no GPU implementation has been
        measured.

    n_jobs : int, default=-1
        Number of Numba threads; ``-1`` uses the configured pool.

    verbose : bool, default=False
        Controls whether progress messages are printed during fit.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.

    feature_importances_ : ndarray of shape (n_features,)
        ``W[a]`` for each feature, in ``[-1, 1]``; higher is better.

    top_features_ : ndarray of shape (n_select,)
        Indices of the selected features, best first.

    is_discrete_ : ndarray of shape (n_features,)
        Which features were treated as discrete.

    distance_mean_, distance_std_ : float
        The mean and population standard deviation of the distances over all
        unique sample pairs (the sigmoid centre and width); NaN for a
        single-class fit.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    The SWRF* paper defines the algorithm for binary classes and discrete
    (SNP) features with the Hamming distance. As in the rest of the Relief
    family here, continuous features use the range-normalised difference
    ``|a - b| / (max - min)`` and the distance is the sum of the differences;
    multiclass targets apply Eq. (2) as written (every other class is a
    miss). Both are extensions, not part of the published definitions.

    The arXiv paper that defines SWRF gives its neighbour weight but no score
    update; SWRF here uses SWRF*'s MoRF update. The papers do not say whether
    ``sigma`` is a population or sample standard deviation; the population form
    over the ``n (n - 1) / 2`` unique pairs is used, and the pair mean is the
    one SURF uses. A distance within ``utils.BOUNDARY_RTOL`` (relative) of the
    mean is a tie with weight zero (the sigmoid's value there), and a target
    whose instances all have zero weight contributes zero. Input containing
    NaNs is rejected.

    scikit-rebate (0.8.4) implements these algorithms differently, so its
    scores do not match: it scores distant instances by similarity
    (``1 - diff``), normalises hits and misses separately by their counts, and
    includes the zero self-distances in the mean and standard deviation.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        use_star: bool = False,
        discrete_limit: int = 10,
        backend: str = "auto",
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.n_features_to_select = n_features_to_select
        self.use_star = use_star
        self.discrete_limit = discrete_limit
        self.backend = backend
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _neighbourhood(self):
        return _check_flag(self.use_star, "use_star"), False

    def _algorithm_name(self, use_star, dead_band):
        return "SWRF*" if use_star else "SWRF"


class MultiSWRF(_SWRFBase):
    """MultiSWRF and MultiSWRFDB: sigmoid-weighted Relief with per-target boundaries.

    Like :class:`SWRF`, but each target instance ``i`` centres the sigmoid on its
    own mean distance to the other instances, ``mu_i``, with width set by their
    population standard deviation ``sigma_i`` (Kazemi-Nia et al., 2026). Scores
    use the Modular Relief Framework update of Stokes and Visweswaran (2012),
    Eq. (1).

    * ``use_star=True, dead_band=False`` is **MultiSWRF*** (Eq. 3): every
      instance is weighted by ``2 / (1 + exp((d - mu_i) * 4 / sigma_i)) - 1``.
    * ``use_star=False, dead_band=False`` is **MultiSWRF** (Eq. 5): only
      instances strictly closer than ``mu_i`` are weighted.
    * ``use_star=True, dead_band=True`` is **MultiSWRFDB*** (Eq. 6): instances
      strictly closer than ``mu_i - sigma_i / 2`` get a positive weight that
      falls to 0 at that boundary, instances strictly farther than
      ``mu_i + sigma_i / 2`` a negative one that rises to 0 at it, and the
      dead band between them is ignored, as in MultiSURF*.
    * ``use_star=False, dead_band=True`` is **MultiSWRFDB** (Eq. 7): the near
      part of MultiSWRFDB* only.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select. A float selects that fraction of
        the features; an int selects that many.

    use_star : bool, default=False
        If True, instances beyond the far boundary are also scored.

    dead_band : bool, default=False
        If True, the near and far boundaries are ``mu_i -/+ sigma_i / 2``
        (MultiSWRFDB); otherwise both are ``mu_i``.

    discrete_limit : int, default=10
        Features with this many or fewer unique values are treated as discrete.

    backend : {'auto', 'cpu'}, default='auto'
        The SWRF family runs on the CPU only; no GPU implementation has been
        measured.

    n_jobs : int, default=-1
        Number of Numba threads; ``-1`` uses the configured pool.

    verbose : bool, default=False
        Controls whether progress messages are printed during fit.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.

    feature_importances_ : ndarray of shape (n_features,)
        ``W[a]`` for each feature, in ``[-1, 1]``; higher is better.

    top_features_ : ndarray of shape (n_select,)
        Indices of the selected features, best first.

    is_discrete_ : ndarray of shape (n_features,)
        Which features were treated as discrete.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    The arXiv paper defining these four algorithms (arXiv:2608.28552, v1, not
    yet peer reviewed) gives their neighbour weights but no score update, so
    they use SWRF*'s MoRF update. ``sigma_i`` is the population standard
    deviation over the ``n - 1`` other instances, as in :class:`MultiSURF`; the
    paper does not say which form it uses. A distance within
    ``utils.BOUNDARY_RTOL`` (relative to ``mu_i``) of a boundary is a tie with
    weight zero (every weight is zero at its boundary), and a target with no
    weighted instance, such as one equidistant from all others, contributes
    zero. Continuous features (range-normalised differences) and multiclass
    targets (every other class is a miss) are extensions, as for
    :class:`SWRF`; input containing NaNs is rejected.

    scikit-rebate (0.8.4) scores distant instances by similarity
    (``1 - diff``) and normalises hits and misses separately by their counts,
    so its scores do not match these.
    """

    _per_target = True

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        use_star: bool = False,
        dead_band: bool = False,
        discrete_limit: int = 10,
        backend: str = "auto",
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.n_features_to_select = n_features_to_select
        self.use_star = use_star
        self.dead_band = dead_band
        self.discrete_limit = discrete_limit
        self.backend = backend
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _neighbourhood(self):
        return _check_flag(self.use_star, "use_star"), _check_flag(self.dead_band, "dead_band")

    def _algorithm_name(self, use_star, dead_band):
        return "MultiSWRF" + ("DB" if dead_band else "") + ("*" if use_star else "")
