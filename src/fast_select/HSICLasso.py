"""HSIC Lasso: nonlinear, non-redundant feature selection by a feature-wise kernelized Lasso.

Sources
-------
Yamada, Jitkrittum, Sigal, Xing and Sugiyama, *High-Dimensional Feature
Selection by Feature-Wise Kernelized Lasso*, Neural Computation 26(1):185-207
(2014), DOI 10.1162/NECO_a_00537, Eq. (3)::

    min_{alpha >= 0}  1/2 || Lbar - sum_k alpha_k Kbar^(k) ||_F^2 + lambda ||alpha||_1

with centred Gram matrices ``Kbar = Gamma K Gamma``, a Gaussian kernel on each
feature scaled to unit standard deviation (``sigma_x = 1`` in the paper's
experiments) and, for the output, a delta kernel ``1/n_y`` (classification) or
a Gaussian kernel on the unit-variance target (regression).

Climente-Gonzalez, Azencott, Kaski and Yamada, *Block HSIC Lasso: model-free
biomarker detection for ultra-high dimensional data*, Bioinformatics
35(14):i427-i435 (2019), DOI 10.1093/bioinformatics/btz333: the same problem
with Frobenius-normalised Gram matrices ``Kbar = HKH / ||HKH||_F`` ("the
normalized variant", Eq. 2), the block estimator over ``n / B`` blocks of
``B`` samples bagged over ``M`` permutations (Eqs. 5-14), and nonnegative
LARS-Lasso to select a fixed number of features.

The (n B M) x d design matrix those papers' Lasso is written with is never
formed. Every Frobenius product of two centred block Gram matrices is
streamed from kernel values through
``<HAH, HCH> = sum A*C - (2/B) r_A . r_C + s_A s_C / B^2`` (``r`` the row sums,
``s`` the total), so memory is O(n d) plus the active features' blocks.
Gaussian kernels enter as ``D = 1 - K = -expm1(-delta^2 / (2 sigma^2))``,
which centring makes equivalent (``H 1 1^T H = 0``) and which keeps nearly
constant blocks accurate.
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cpu_backend import resolve_cpu_backend
from .utils import check_integer_param, resolve_num_threads

TASKS = ("auto", "classification", "regression")
# Nonnegative LARS tolerances, as in the authors' pyHSICLasso solver.
_LARS_TOL = 1e-9


def _block_index(n_samples, block_size, n_permutations, random_state):
    """Sample indices of every block, shape ``(M, n // B, B)``.

    As in pyHSICLasso: permutation ``m`` re-permutes the previous order with
    seed ``random_state + m`` (``random_state=0`` reproduces its blocks), and
    the ``n mod B`` samples left over after each permutation are not used.
    """
    n_blocks = n_samples // block_size
    used = n_blocks * block_size
    order = np.arange(n_samples)
    blocks = np.empty((n_permutations, n_blocks, block_size), dtype=np.int64)
    for m in range(n_permutations):
        order = np.random.RandomState(random_state + m).permutation(order)
        blocks[m] = order[:used].reshape(n_blocks, block_size)
    return blocks


@njit(cache=True)
def _block_stats(values, gaussian, scale, out_block):  # pragma: no cover
    """Fill ``out_block`` (B x B) with the kernel transform and return ``(row sums, total, norm^2)``.

    Gaussian: ``D = -expm1(-(a - b)^2 * scale)``; otherwise (delta output
    kernel, given as class codes) ``E = -1/n_c`` for equal codes, the negated
    delta kernel so that ``<HDH, HEH> = <HKH, HLH>``.
    """
    b = values.shape[0]
    rows = np.zeros(b)
    if gaussian:
        for i in range(b):
            out_block[i, i] = 0.0
            for j in range(i + 1, b):
                d = values[i] - values[j]
                v = -math.expm1(-d * d * scale)
                out_block[i, j] = v
                out_block[j, i] = v
                rows[i] += v
                rows[j] += v
    else:
        n_codes = 0
        for i in range(b):
            n_codes = max(n_codes, int(values[i]) + 1)
        counts = np.zeros(n_codes)
        for i in range(b):
            counts[int(values[i])] += 1.0
        for i in range(b):
            for j in range(b):
                v = -1.0 / counts[int(values[i])] if values[i] == values[j] else 0.0
                out_block[i, j] = v
                rows[i] += v
    total = 0.0
    squares = 0.0
    row_squares = 0.0
    for i in range(b):
        total += rows[i]
        row_squares += rows[i] * rows[i]
        for j in range(b):
            squares += out_block[i, j] * out_block[i, j]
    norm_sq = squares - 2.0 * row_squares / b + total * total / (b * b)
    return rows, total, max(norm_sq, 0.0)


@njit(cache=True)
def _prepare_target(values, blocks, gaussian, scale, out_blocks, out_rows, out_totals, out_norms):  # pragma: no cover
    n_perm, n_blocks, b = blocks.shape
    for m in range(n_perm):
        for g in range(n_blocks):
            block_values = np.empty(b)
            for i in range(b):
                block_values[i] = values[blocks[m, g, i]]
            rows, total, norm_sq = _block_stats(block_values, gaussian, scale, out_blocks[m, g])
            out_rows[m, g] = rows
            out_totals[m, g] = total
            out_norms[m, g] = math.sqrt(norm_sq)


@njit(parallel=True, cache=True)
def _feature_products(
    x_t, constant, blocks, scale, ref_blocks, ref_rows, ref_totals, ref_norms, normalize, norms, out
):  # pragma: no cover
    """``out[k] = mean over blocks of <Kbar_k, Rbar>`` for one reference (the output or an active feature).

    ``x_t`` is ``(d, n)`` standardised features. ``norms[k, m, g]`` receives
    ``||H D_k H||_F`` when ``norms`` is writable for the first (output) pass; on
    later passes it is read. A block whose centred Gram matrix has zero norm
    contributes zero.
    """
    n_features = x_t.shape[0]
    n_perm, n_blocks, b = blocks.shape
    weight = 1.0 / (n_perm * n_blocks)
    for k in prange(n_features):
        if constant[k]:
            out[k] = 0.0
            for m in range(n_perm):
                for g in range(n_blocks):
                    norms[k, m, g] = 0.0
            continue
        column = x_t[k]
        rows = np.empty(b)
        total_product = 0.0
        for m in range(n_perm):
            for g in range(n_blocks):
                index = blocks[m, g]
                ref = ref_blocks[m, g]
                ref_r = ref_rows[m, g]
                for i in range(b):
                    rows[i] = 0.0
                dot = 0.0
                squares = 0.0
                for i in range(b):
                    xi = column[index[i]]
                    for j in range(i + 1, b):
                        d = xi - column[index[j]]
                        v = -math.expm1(-d * d * scale)
                        rows[i] += v
                        rows[j] += v
                        dot += 2.0 * v * ref[i, j]
                        squares += 2.0 * v * v
                total = 0.0
                row_dot = 0.0
                row_squares = 0.0
                for i in range(b):
                    total += rows[i]
                    row_dot += rows[i] * ref_r[i]
                    row_squares += rows[i] * rows[i]
                own_sq = squares - 2.0 * row_squares / b + total * total / (b * b)
                own = math.sqrt(own_sq) if own_sq > 0.0 else 0.0
                norms[k, m, g] = own
                centred = dot - 2.0 * row_dot / b + total * ref_totals[m, g] / (b * b)
                if normalize:
                    denominator = own * ref_norms[m, g]
                    if denominator > 0.0:
                        total_product += centred / denominator
                else:
                    total_product += centred
        out[k] = total_product * weight


def _nonnegative_lars(relevance, gram_column, n_select):
    """Nonnegative LARS-Lasso (Efron et al., 2004) stopped with ``n_select`` active features.

    Minimises ``1/2 ||v - U alpha||^2`` along the Lasso path with
    ``alpha >= 0``, where ``U^T v = relevance`` and ``gram_column(j)`` returns
    ``U^T U[:, j]``. Stops at the knot where a further feature would enter
    (the Lasso solution with ``n_select`` active features, as pyHSICLasso
    does), when the active correlation reaches zero, or when no feature can
    enter. Ties take the lowest index.
    """
    d = relevance.shape[0]
    beta = np.zeros(d)
    c = relevance.astype(np.float64).copy()
    columns = {}
    if not np.any(c > _LARS_TOL):
        return beta, [], 0
    active = [int(np.argmax(c))]
    steps = 0
    while True:
        for j in active:
            if j not in columns:
                columns[j] = gram_column(j)
        G = np.array([[columns[j][i] for j in active] for i in active])
        level = float(np.mean(c[active]))
        if level < _LARS_TOL:
            break
        try:
            w = np.linalg.solve(G, np.ones(len(active)))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(G, np.ones(len(active)), rcond=None)[0]
        a = np.zeros(d)
        for wj, j in zip(w, active):
            a += wj * columns[j]
        inactive = np.ones(d, dtype=bool)
        inactive[active] = False
        with np.errstate(divide="ignore", invalid="ignore"):
            enter = np.where(inactive & (1.0 - a > _LARS_TOL), (level - c) / (1.0 - a), np.inf)
        # An entry no earlier than the end of the path (lambda = 0) is the end of the path:
        # such a feature (e.g. a constant one) would enter with zero correlation and coefficient.
        enter[~np.isfinite(enter) | (enter <= _LARS_TOL) | (enter >= level * (1.0 - 1e-12))] = np.inf
        drop = np.full(len(active), np.inf)
        for position, j in enumerate(active):
            if w[position] < 0.0:
                gamma = -beta[j] / w[position]
                if gamma > _LARS_TOL:
                    drop[position] = gamma
        candidates = np.concatenate([enter, drop, [level]])
        step = int(np.argmin(candidates))
        gamma = float(candidates[step])
        if not np.isfinite(gamma):
            break
        if step < d and len(active) >= n_select:
            # The next event adds a feature beyond the requested count: stop at that knot.
            for wj, j in zip(w, active):
                beta[j] += gamma * wj
            c -= gamma * a
            steps += 1
            break
        for wj, j in zip(w, active):
            beta[j] += gamma * wj
        c -= gamma * a
        steps += 1
        if step < d:
            active.append(step)
        elif step < d + len(active):
            j = active.pop(step - d)
            beta[j] = 0.0
        else:  # the active correlations reached zero: end of the path
            break
    beta[beta < 0.0] = 0.0
    return beta, active, steps


class HSICLasso(TransformerMixin, BaseEstimator):
    """HSIC Lasso: nonlinear feature selection that also removes redundant features.

    Each feature's Gaussian Gram matrix is regressed onto the output's Gram
    matrix with a nonnegative Lasso, so a feature is chosen when it depends on
    the target (in any nonlinear way) and is not already explained by the
    features chosen before it (Yamada et al., 2014). With the default block
    estimator (Climente-Gonzalez et al., 2019), Gram matrices are computed on
    ``M`` random partitions of the samples into blocks of ``block_size``,
    which makes the cost linear in the number of samples.

    Parameters
    ----------
    n_features_to_select : int or float, default=10
        The number of features to select: the nonnegative LARS path is
        followed until this many are active. A float is a fraction of the
        features. Fewer are returned if the path ends first.

    task : {'auto', 'classification', 'regression'}, default='auto'
        The output kernel: the delta kernel for classification, a Gaussian
        kernel on the unit-variance target for regression. ``'auto'`` uses
        scikit-learn's ``type_of_target``: continuous targets are regression.

    block_size : int, default=20
        ``B``, samples per block. ``0`` (or ``None``) uses one block of all
        samples: vanilla HSIC Lasso, whose cost grows with ``n_samples ** 2``.
        The authors recommend 10 to 20.

    n_permutations : int, default=3
        ``M``, the number of random block partitions averaged (ignored, and 1,
        for a single block).

    sigma_x, sigma_y : float, default=1.0
        Gaussian kernel widths on the standardised features and (regression)
        target. Both papers use 1.

    normalize : bool, default=True
        Divide every centred block Gram matrix by its Frobenius norm, the
        normalised HSIC of Climente-Gonzalez et al. (2019) and pyHSICLasso.
        ``False`` is Yamada et al. (2014), Eq. (3), exactly.

    random_state : int, default=0
        Seeds of the block permutations (``random_state + m`` for the
        ``m``-th); 0 reproduces pyHSICLasso's blocks.

    backend : {'auto', 'cpu'}, default='auto'
        HSIC Lasso runs on the CPU only; no GPU implementation has been measured.

    n_jobs : int, default=-1
        Number of Numba threads; ``-1`` uses the configured pool.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The Lasso coefficients ``alpha`` at the stopping point (zero for
        features that are not selected).

    top_features_ : ndarray
        The selected features, largest coefficient first.

    relevance_ : ndarray of shape (n_features,)
        The (block, normalised) HSIC of each feature with the output.

    task_ : str
        ``'classification'`` or ``'regression'``.

    block_size_, n_permutations_ : int
        The block size and number of partitions used.

    n_iter_ : int
        LARS steps taken.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    Conventions from the authors' pyHSICLasso where the papers are silent:
    each block is centred and normalised on its own, the delta kernel uses
    the class counts within the block, the ``n mod B`` samples left over by
    each permutation are not used, and the path stops at the knot where one
    more feature would enter. Features are scaled by their standard deviation
    (``ddof=0``); a constant feature, or a block in which a feature or the
    output is constant, contributes zero instead of a rounding artefact. A
    feature whose kernel equals an active feature's never enters. Covariate
    adjustment and the discrete-input kernel of pyHSICLasso are not
    implemented. Input containing NaNs is rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 10,
        task: str = "auto",
        block_size: int | None = 20,
        n_permutations: int = 3,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        normalize: bool = True,
        random_state: int = 0,
        backend: str = "auto",
        n_jobs: int = -1,
    ):
        self.n_features_to_select = n_features_to_select
        self.task = task
        self.block_size = block_size
        self.n_permutations = n_permutations
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.normalize = normalize
        self.random_state = random_state
        self.backend = backend
        self.n_jobs = n_jobs

    @staticmethod
    def _positive(value, name):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
        return float(value)

    def _validate(self, n_samples, n_features):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}.")
        if self.block_size in (0, None):
            block = n_samples
        else:
            block = check_integer_param(self.block_size, "block_size", 2)
            if block > n_samples:
                raise ValueError(f"block_size ({block}) cannot exceed n_samples ({n_samples}); use 0 for one block.")
        n_perm = 1 if block == n_samples else check_integer_param(self.n_permutations, "n_permutations", 1)
        if not isinstance(self.normalize, (bool, np.bool_)):
            raise TypeError(f"normalize must be a bool, got {self.normalize!r}.")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(self.random_state, (int, np.integer)):
            raise TypeError(f"random_state must be an int, got {self.random_state!r}.")
        resolve_num_threads(self.n_jobs)
        sigma_x = self._positive(self.sigma_x, "sigma_x")
        sigma_y = self._positive(self.sigma_y, "sigma_y")
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
                    f"it must be > 0 and <= n_features (got n_features={n_features})."
                )
            n_select = int(self.n_features_to_select)
        else:
            raise TypeError("n_features_to_select must be an int or a float.")
        return n_select, block, n_perm, sigma_x, sigma_y

    def fit(self, X, y):
        """Run nonnegative LARS on the (block) HSIC Lasso problem."""
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_2d=True, y_numeric=False, ensure_min_samples=2)
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        n_select, block, n_perm, sigma_x, sigma_y = self._validate(n_samples, n_features)

        task = self.task
        if task == "auto":
            task = "regression" if type_of_target(y) in ("continuous", "continuous-multioutput") else "classification"
        self.task_ = task
        if task == "classification":
            classes, codes = np.unique(y, return_inverse=True)
            self.classes_ = classes
            target = codes.astype(np.float64)
            gaussian_y = False
        else:
            target = np.asarray(y, dtype=np.float64)
            if not np.all(np.isfinite(target)):
                raise ValueError("Input y contains NaN or infinity.")
            spread = target.std()
            if not target.max() > target.min():
                raise ValueError("HSIC Lasso regression needs a target with at least two distinct values.")
            target = target / spread
            gaussian_y = True

        blocks = _block_index(n_samples, block, n_perm, int(self.random_state))
        self.block_size_ = block
        self.n_permutations_ = n_perm
        n_blocks = blocks.shape[1]

        constant = X.max(axis=0) == X.min(axis=0)
        scales = X.std(axis=0)
        x_t = np.ascontiguousarray((X / np.where(constant, 1.0, scales)).T)
        scale_x = 1.0 / (2.0 * sigma_x * sigma_x)
        scale_y = 1.0 / (2.0 * sigma_y * sigma_y)

        out_blocks = np.empty((n_perm, n_blocks, block, block))
        out_rows = np.empty((n_perm, n_blocks, block))
        out_totals = np.empty((n_perm, n_blocks))
        out_norms = np.empty((n_perm, n_blocks))
        norms = np.empty((n_features, n_perm, n_blocks))
        relevance = np.empty(n_features)
        normalize = bool(self.normalize)

        original_num_threads = get_num_threads()
        set_num_threads(resolve_num_threads(self.n_jobs))
        try:
            _prepare_target(target, blocks, gaussian_y, scale_y, out_blocks, out_rows, out_totals, out_norms)
            _feature_products(
                x_t, constant, blocks, scale_x, out_blocks, out_rows, out_totals, out_norms, normalize, norms, relevance
            )

            def gram_column(j):
                ref_blocks = np.empty((n_perm, n_blocks, block, block))
                ref_rows = np.empty((n_perm, n_blocks, block))
                ref_totals = np.empty((n_perm, n_blocks))
                ref_norms = np.empty((n_perm, n_blocks))
                _prepare_target(x_t[j], blocks, True, scale_x, ref_blocks, ref_rows, ref_totals, ref_norms)
                column = np.empty(n_features)
                scratch = np.empty_like(norms)
                _feature_products(
                    x_t,
                    constant,
                    blocks,
                    scale_x,
                    ref_blocks,
                    ref_rows,
                    ref_totals,
                    ref_norms,
                    normalize,
                    scratch,
                    column,
                )
                return column

            beta, active, steps = _nonnegative_lars(relevance, gram_column, n_select)
        finally:
            set_num_threads(original_num_threads)

        self.relevance_ = relevance
        self.feature_importances_ = beta
        order = sorted((j for j in active if beta[j] > 0.0), key=lambda j: (-beta[j], j))
        self.top_features_ = np.array(order, dtype=np.int64)
        self.n_iter_ = steps
        return self

    def transform(self, X):
        """Reduce ``X`` to the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return X[:, self.top_features_]

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)
