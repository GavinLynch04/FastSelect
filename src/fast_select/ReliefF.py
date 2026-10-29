from __future__ import annotations

import warnings

import numpy as np
from numba import cuda, float32, get_num_threads, get_thread_id, njit, prange, set_num_threads
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .utils import (
    check_integer_param,
    discrete_feature_mask,
    effective_discrete_limit,
    ensure_cuda_context,
    is_cuda_ready,
    prepare_relief_matrix,
    resolve_compute_backend,
    resolve_num_threads,
    split_discrete_last,
)

TPB = 64  # Threads-per-block


@cuda.jit
def _compute_dist_matrix_gpu_kernel(x, is_discrete, dist_matrix):  # pragma: no cover
    """
    Computes each pairwise sample distance once and mirrors the result.
    Grid: (n_samples,), Threads per block: TPB
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    if tid == 0:
        dist_matrix[i, i] = 0.0

    for j in range(i + 1, n_samples):
        local_dist = 0.0
        for f in range(tid, n_features, TPB):
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f])
            local_dist += diff

        sh_sum = cuda.shared.array(shape=64, dtype=float32)
        sh_sum[tid] = local_dist
        cuda.syncthreads()

        off = TPB // 2
        while off > 0:
            if tid < off:
                sh_sum[tid] += sh_sum[tid + off]
            cuda.syncthreads()
            off //= 2

        if tid == 0:
            dist_matrix[i, j] = sh_sum[0]
            dist_matrix[j, i] = sh_sum[0]
        cuda.syncthreads()


@cuda.jit
def _accumulate_weighted_diffs_gpu_kernel(x, weights_matrix, is_discrete, scores_out):  # pragma: no cover
    """
    Accumulates feature difference scores weighted by neighbor relationships on GPU.
    Grid: (n_samples,), Threads per block: TPB
    """
    n_samples, n_features = x.shape
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    for j in range(n_samples):
        w = weights_matrix[i, j]
        if w == 0.0:
            continue

        for f in range(tid, n_features, TPB):
            if is_discrete[f]:
                diff = 1.0 if x[i, f] != x[j, f] else 0.0
            else:
                diff = abs(x[i, f] - x[j, f])
            cuda.atomic.add(scores_out, f, w * diff)


@cuda.jit
def _select_relieff_neighbors_gpu_kernel(dist_matrix, y, k, neighbor_distances, neighbor_indices):  # pragma: no cover
    """Select the nearest hits/misses for one sample per GPU thread."""
    i = cuda.grid(1)
    n_samples = dist_matrix.shape[0]
    if i >= n_samples:
        return

    n_classes = neighbor_indices.shape[1]
    for c in range(n_classes):
        for neighbor in range(k):
            neighbor_distances[i, c, neighbor] = np.inf
            neighbor_indices[i, c, neighbor] = -1

    for j in range(n_samples):
        if i == j:
            continue
        label = y[j]
        distance = dist_matrix[i, j]
        if distance < neighbor_distances[i, label, k - 1]:
            pos = k - 1
            while pos > 0 and distance < neighbor_distances[i, label, pos - 1]:
                neighbor_distances[i, label, pos] = neighbor_distances[i, label, pos - 1]
                neighbor_indices[i, label, pos] = neighbor_indices[i, label, pos - 1]
                pos -= 1
            neighbor_distances[i, label, pos] = distance
            neighbor_indices[i, label, pos] = j


@cuda.jit
def _score_relieff_neighbors_gpu_kernel(
    x,
    y,
    neighbor_indices,
    class_probs,
    is_discrete,
    scores_out,
):  # pragma: no cover
    """Score only selected ReliefF neighbors, avoiding a dense weight matrix."""
    n_samples, n_features = x.shape
    n_classes = neighbor_indices.shape[1]
    k = neighbor_indices.shape[2]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    label_i = y[i]
    denom = 1.0 - class_probs[label_i]
    if denom <= 0.0:
        denom = 1.0

    # Feature-major accumulation: each thread owns a feature and sums that
    # feature's contribution over every selected neighbour in a register, so the
    # block issues one atomic per feature instead of one per (neighbour, feature).
    for f in range(tid, n_features, TPB):
        discrete_f = is_discrete[f]
        x_if = x[i, f]
        acc = 0.0

        for c in range(n_classes):
            count = 0
            for neighbor in range(k):
                if neighbor_indices[i, c, neighbor] >= 0:
                    count += 1
            if count == 0:
                continue

            if c == label_i:
                weight = -1.0 / (count * n_samples)
            else:
                weight = class_probs[c] / denom / (count * n_samples)

            for neighbor in range(count):
                j = neighbor_indices[i, c, neighbor]
                if discrete_f:
                    diff = 1.0 if x_if != x[j, f] else 0.0
                else:
                    diff = abs(x_if - x[j, f])
                acc += weight * diff

        if acc != 0.0:
            cuda.atomic.add(scores_out, f, acc)


@njit(parallel=True)
def _compute_relieff_weights(dist_matrix, y_enc, class_probs, k):  # pragma: no cover
    """
    Computes ReliefF neighbor weight matrix W of shape (n_samples, n_samples)
    using bounded insertion buffers instead of sorting a full row per class.
    """
    n_samples = dist_matrix.shape[0]
    n_classes = len(class_probs)
    weights = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in prange(n_samples):
        lbl_i = y_enc[i]
        denom = 1.0 - class_probs[lbl_i]
        if denom <= 0:
            denom = 1.0

        hit_dists = np.full(k, np.inf, dtype=np.float32)
        hit_indices = np.full(k, -1, dtype=np.int32)
        miss_dists = np.full((n_classes, k), np.inf, dtype=np.float32)
        miss_indices = np.full((n_classes, k), -1, dtype=np.int32)

        for j in range(n_samples):
            if i == j:
                continue
            distance = dist_matrix[i, j]
            label = y_enc[j]
            if label == lbl_i:
                if distance < hit_dists[k - 1]:
                    pos = k - 1
                    while pos > 0 and distance < hit_dists[pos - 1]:
                        hit_dists[pos] = hit_dists[pos - 1]
                        hit_indices[pos] = hit_indices[pos - 1]
                        pos -= 1
                    hit_dists[pos] = distance
                    hit_indices[pos] = j
            elif distance < miss_dists[label, k - 1]:
                pos = k - 1
                while pos > 0 and distance < miss_dists[label, pos - 1]:
                    miss_dists[label, pos] = miss_dists[label, pos - 1]
                    miss_indices[label, pos] = miss_indices[label, pos - 1]
                    pos -= 1
                miss_dists[label, pos] = distance
                miss_indices[label, pos] = j

        hit_count = 0
        for neighbor in range(k):
            if hit_indices[neighbor] >= 0:
                hit_count += 1
        if hit_count > 0:
            hit_weight = -1.0 / (hit_count * n_samples)
            for neighbor in range(hit_count):
                weights[i, hit_indices[neighbor]] = hit_weight

        for c in range(n_classes):
            if c == lbl_i:
                continue
            miss_count = 0
            for neighbor in range(k):
                if miss_indices[c, neighbor] >= 0:
                    miss_count += 1
            if miss_count > 0:
                miss_weight = class_probs[c] / denom / (miss_count * n_samples)
                for neighbor in range(miss_count):
                    weights[i, miss_indices[c, neighbor]] = miss_weight

    return weights


def _relieff_gpu_host_caller(x_d, y_enc, is_discrete_d, class_probs, k):
    """Launch ReliefF distance, neighbor-selection, and scoring stages."""
    ensure_cuda_context()
    n_samples, n_features = x_d.shape
    dist_matrix_d = cuda.device_array((n_samples, n_samples), dtype=np.float32)

    _compute_dist_matrix_gpu_kernel[n_samples, TPB](x_d, is_discrete_d, dist_matrix_d)

    scores_d = cuda.device_array(n_features, dtype=np.float32)
    scores_d[:] = 0.0

    n_classes = class_probs.shape[0]
    if n_classes * k <= n_samples:
        y_d = cuda.to_device(y_enc)
        class_probs_d = cuda.to_device(class_probs)
        neighbor_distances_d = cuda.device_array((n_samples, n_classes, k), dtype=np.float32)
        neighbor_indices_d = cuda.device_array((n_samples, n_classes, k), dtype=np.int32)
        selection_threads = 128
        selection_blocks = (n_samples + selection_threads - 1) // selection_threads
        _select_relieff_neighbors_gpu_kernel[selection_blocks, selection_threads](
            dist_matrix_d,
            y_d,
            k,
            neighbor_distances_d,
            neighbor_indices_d,
        )
        _score_relieff_neighbors_gpu_kernel[n_samples, TPB](
            x_d,
            y_d,
            neighbor_indices_d,
            class_probs_d,
            is_discrete_d,
            scores_d,
        )
    else:
        # Preserve bounded memory for unusually large k/class combinations.
        dist_matrix = dist_matrix_d.copy_to_host()
        weights_matrix = _compute_relieff_weights(dist_matrix, y_enc, class_probs, k)
        weights_d = cuda.to_device(weights_matrix)
        _accumulate_weighted_diffs_gpu_kernel[n_samples, TPB](x_d, weights_d, is_discrete_d, scores_d)

    return scores_d.copy_to_host()


@njit(parallel=True, fastmath=True)
def _relieff_cpu_kernel(x, y_enc, n_cont, k, class_probs, scores_out):  # pragma: no cover
    """ReliefF CPU scoring.

    ``x`` arrives with continuous features in columns ``[0, n_cont)`` and
    discrete features after them, so both inner loops are branch-free and
    vectorisable.  The distance is still the sum of the same per-feature terms.
    """
    n_samples, n_features = x.shape
    n_classes = class_probs.shape[0]
    n_threads = get_num_threads()

    thread_scores = np.zeros((n_threads, n_features), dtype=np.float32)

    for i in prange(n_samples):
        tid = get_thread_id()
        lbl_i = y_enc[i]
        x_i = x[i]

        hit_d = np.full(k, np.inf, dtype=np.float32)
        hit_idx = np.full(k, -1, dtype=np.int32)

        miss_d = np.full((n_classes, k), np.inf, dtype=np.float32)
        miss_idx = np.full((n_classes, k), -1, dtype=np.int32)

        for j in range(n_samples):
            if i == j:
                continue

            x_j = x[j]
            d = 0.0
            for f in range(n_cont):
                d += abs(x_i[f] - x_j[f])
            for f in range(n_cont, n_features):
                if x_i[f] != x_j[f]:
                    d += 1.0

            lbl_j = y_enc[j]
            if lbl_j == lbl_i:
                if d < hit_d[k - 1]:
                    pos = k - 1
                    while pos > 0 and d < hit_d[pos - 1]:
                        hit_d[pos] = hit_d[pos - 1]
                        hit_idx[pos] = hit_idx[pos - 1]
                        pos -= 1
                    hit_d[pos] = d
                    hit_idx[pos] = j
            else:
                if d < miss_d[lbl_j, k - 1]:
                    pos = k - 1
                    while pos > 0 and d < miss_d[lbl_j, pos - 1]:
                        miss_d[lbl_j, pos] = miss_d[lbl_j, pos - 1]
                        miss_idx[lbl_j, pos] = miss_idx[lbl_j, pos - 1]
                        pos -= 1
                    miss_d[lbl_j, pos] = d
                    miss_idx[lbl_j, pos] = j

        h_found = 0
        for ki in range(k):
            if hit_idx[ki] != -1:
                h_found += 1
            else:
                break

        denom = 1.0 - class_probs[lbl_i]
        if denom <= 0:
            denom = 1.0

        scores_i = thread_scores[tid]
        if h_found > 0:
            scale_hit = -1.0 / (h_found * n_samples)
            for ki in range(h_found):
                x_h = x[hit_idx[ki]]
                for f in range(n_cont):
                    scores_i[f] += scale_hit * abs(x_i[f] - x_h[f])
                for f in range(n_cont, n_features):
                    if x_i[f] != x_h[f]:
                        scores_i[f] += scale_hit

        for c in range(n_classes):
            if c == lbl_i:
                continue
            m_found = 0
            for ki in range(k):
                if miss_idx[c, ki] != -1:
                    m_found += 1
                else:
                    break
            if m_found > 0:
                weight_c = class_probs[c] / denom
                scale_miss = weight_c / m_found / n_samples
                for ki in range(m_found):
                    x_m = x[miss_idx[c, ki]]
                    for f in range(n_cont):
                        scores_i[f] += scale_miss * abs(x_i[f] - x_m[f])
                    for f in range(n_cont, n_features):
                        if x_i[f] != x_m[f]:
                            scores_i[f] += scale_miss

    for f in range(n_features):
        tot = 0.0
        for t in range(n_threads):
            tot += thread_scores[t, f]
        scores_out[f] = tot


def _relieff_cpu_host_caller(x, y_enc, n_cont, k, class_probs, n_jobs):
    n_samples, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)

    num_threads_to_set = resolve_num_threads(n_jobs)

    original_num_threads = get_num_threads()
    set_num_threads(num_threads_to_set)

    try:
        _relieff_cpu_kernel(x, y_enc, n_cont, k, class_probs, scores)
    finally:
        set_num_threads(original_num_threads)

    return scores


@njit(inline="always")
def _missing_aware_diff(a, b, f, class_a, class_b, one_missing, both_missing):  # pragma: no cover
    """``diff`` of discrete feature ``f`` for category codes ``a`` and ``b``; ``-1`` is missing.

    Known codes use Eq. (1). One missing value uses Eq. (3),
    ``1 - P(known value | class of the instance whose value is missing)``, and two
    use Eq. (4), ``1 - sum_V P(V | class_a) P(V | class_b)`` (Robnik-Šikonja and
    Kononenko, 2003); both are tabulated in ``one_missing`` and ``both_missing``.
    """
    if a >= 0.0:
        if b >= 0.0:
            return 0.0 if a == b else 1.0
        return one_missing[f, class_b, int(a)]
    if b >= 0.0:
        return one_missing[f, class_a, int(b)]
    return both_missing[f, class_a, class_b]


@njit(parallel=True)
def _relieff_missing_cpu_kernel(x, y_enc, n_cont, k, class_probs, one_missing, both_missing):  # pragma: no cover
    """ReliefF CPU scoring for data with missing discrete values.

    The same neighbour selection, tie rule (lower index first) and update as
    :func:`_relieff_cpu_kernel`, with every discrete difference, in distances as
    well as in scores, taken from :func:`_missing_aware_diff`.  Kept separate so
    complete data never pays for the lookups; float64 throughout and no
    ``fastmath``, so each distance is summed in column order.
    """
    n_samples, n_features = x.shape
    n_classes = class_probs.shape[0]
    n_threads = get_num_threads()

    thread_scores = np.zeros((n_threads, n_features))

    for i in prange(n_samples):
        tid = get_thread_id()
        lbl_i = y_enc[i]
        x_i = x[i]

        hit_d = np.full(k, np.inf)
        hit_idx = np.full(k, -1, dtype=np.int32)
        miss_d = np.full((n_classes, k), np.inf)
        miss_idx = np.full((n_classes, k), -1, dtype=np.int32)

        for j in range(n_samples):
            if i == j:
                continue

            x_j = x[j]
            lbl_j = y_enc[j]
            d = 0.0
            for f in range(n_cont):
                d += abs(x_i[f] - x_j[f])
            for f in range(n_cont, n_features):
                d += _missing_aware_diff(x_i[f], x_j[f], f - n_cont, lbl_i, lbl_j, one_missing, both_missing)

            if lbl_j == lbl_i:
                if d < hit_d[k - 1]:
                    pos = k - 1
                    while pos > 0 and d < hit_d[pos - 1]:
                        hit_d[pos] = hit_d[pos - 1]
                        hit_idx[pos] = hit_idx[pos - 1]
                        pos -= 1
                    hit_d[pos] = d
                    hit_idx[pos] = j
            elif d < miss_d[lbl_j, k - 1]:
                pos = k - 1
                while pos > 0 and d < miss_d[lbl_j, pos - 1]:
                    miss_d[lbl_j, pos] = miss_d[lbl_j, pos - 1]
                    miss_idx[lbl_j, pos] = miss_idx[lbl_j, pos - 1]
                    pos -= 1
                miss_d[lbl_j, pos] = d
                miss_idx[lbl_j, pos] = j

        denom = 1.0 - class_probs[lbl_i]
        if denom <= 0:
            denom = 1.0

        scores_i = thread_scores[tid]
        for c in range(n_classes):
            found = 0
            for ki in range(k):
                if (hit_idx[ki] if c == lbl_i else miss_idx[c, ki]) != -1:
                    found += 1
                else:
                    break
            if found == 0:
                continue
            if c == lbl_i:
                scale = -1.0 / (found * n_samples)
            else:
                scale = class_probs[c] / denom / found / n_samples
            for ki in range(found):
                j = hit_idx[ki] if c == lbl_i else miss_idx[c, ki]
                x_j = x[j]
                for f in range(n_cont):
                    scores_i[f] += scale * abs(x_i[f] - x_j[f])
                for f in range(n_cont, n_features):
                    scores_i[f] += scale * _missing_aware_diff(
                        x_i[f], x_j[f], f - n_cont, lbl_i, c, one_missing, both_missing
                    )

    scores = np.zeros(n_features)
    for f in range(n_features):
        for t in range(n_threads):
            scores[f] += thread_scores[t, f]
    return scores


def _discrete_mask_with_missing(x, missing, discrete_limit):
    """:func:`discrete_feature_mask`, counting only the known values of columns with NaNs."""
    is_discrete = discrete_feature_mask(x, discrete_limit)
    limit = effective_discrete_limit(discrete_limit, x.shape[0])
    for f in np.flatnonzero(missing.any(axis=0)):
        is_discrete[f] = limit >= 1 and np.unique(x[~missing[:, f], f]).size <= limit
    return is_discrete


def _prepare_missing_matrix(x, missing, is_discrete, columns, n_cont, y_enc, n_classes, discrete_limit):
    """float64 working matrix plus the Eq. (3)/(4) tables for the discrete columns.

    Continuous columns (which contain no NaNs) use :func:`prepare_relief_matrix`.
    Each discrete column becomes dense codes of its known values, with ``-1`` for
    a missing value. ``P(V | c)`` is the relative frequency of ``V`` among the
    class-``c`` instances whose value is known. Two conventions the paper leaves
    open are explicit here: a class with no known value of a feature uses the
    feature's overall relative frequencies, and a feature with no known value at
    all has difference 0 everywhere, like a constant feature.
    """
    n_samples = x.shape[0]
    n_disc = columns.size - n_cont
    prepared = np.empty((n_samples, columns.size))
    prepared[:, :n_cont] = prepare_relief_matrix(
        x, is_discrete, columns[:n_cont], dtype=np.float64, discrete_limit=discrete_limit
    )

    codes, vocabulary_sizes = [], []
    for f in columns[n_cont:]:
        known = ~missing[:, f]
        values, inverse = np.unique(x[known, f], return_inverse=True)
        column = np.full(n_samples, -1.0)
        column[known] = inverse
        codes.append(column)
        vocabulary_sizes.append(values.size)
    if n_disc:
        prepared[:, n_cont:] = np.column_stack(codes)

    n_values = max([1, *vocabulary_sizes])
    one_missing = np.zeros((n_disc, n_classes, n_values))
    both_missing = np.zeros((n_disc, n_classes, n_classes))
    for position in range(n_disc):
        column = prepared[:, n_cont + position]
        known = column >= 0
        if not known.any():
            continue
        counts = np.zeros((n_classes, n_values))
        np.add.at(counts, (y_enc[known], column[known].astype(np.int64)), 1.0)
        overall = counts.sum(axis=0)
        probabilities = np.empty_like(counts)
        for c in range(n_classes):
            class_known = counts[c].sum()
            probabilities[c] = counts[c] / class_known if class_known > 0 else overall / overall.sum()
        one_missing[position] = 1.0 - probabilities
        agreement = np.zeros((n_classes, n_classes))
        for v in range(n_values):  # sequential over values, in sorted value order
            agreement += np.outer(probabilities[:, v], probabilities[:, v])
        both_missing[position] = 1.0 - agreement
    return prepared, one_missing, both_missing


def _relieff_missing_cpu_host_caller(x, y_enc, n_cont, k, class_probs, one_missing, both_missing, n_jobs):
    original_num_threads = get_num_threads()
    set_num_threads(resolve_num_threads(n_jobs))
    try:
        return _relieff_missing_cpu_kernel(x, y_enc, n_cont, k, class_probs, one_missing, both_missing)
    finally:
        set_num_threads(original_num_threads)


class ReliefF(TransformerMixin, BaseEstimator):
    """GPU and CPU-accelerated feature selection using the ReliefF algorithm.

    This estimator provides a unified API for running ReliefF on either
    a CPU (Numba), NVIDIA GPU (CUDA), or Apple Silicon GPU (Metal).

    Parameters
    ----------
    n_features_to_select : int | float, default=0.2
        The number of top features to select. If variable is a float, that percent
        of features will be selected. If variable is an int, that number of features will be returned.

    discrete_limit : int, default=10
        The limit for the number of independent feature values to be considered discrete.

    n_neighbors : int, default=3
        The number of nearest neighbors to use for score calculation.

    backend : {'auto', 'gpu', 'cpu', 'metal'}, default='auto'
        The compute backend. ``'auto'`` prefers CUDA, then optional Metal, then
        CPU. ``'metal'`` is the Apple Silicon GPU backend and requires
        ``fast-select[metal]``; it supports ``n_neighbors <= 64``.

    verbose : bool, default=False
        Controls whether progress updates are printed during the fit.

    n_jobs : int, default=-1
        Controls the number of threads utilized by Numba while running on CPU.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.

    feature_importances_ : ndarray of shape (n_features,)
        The calculated importance scores for each feature.

    effective_backend_ : str
        The backend that was actually used during `fit` ('gpu', 'cpu', or
        'metal'). Metal resolves numerically ambiguous neighbors on the CPU.

    Notes
    -----
    This implementation follows the classification equations, including
    multiclass prior weighting. Missing values (NaN) are accepted in discrete
    features and handled as the ReliefF papers define (Kononenko, 1994;
    Robnik-Šikonja and Kononenko, 2003, Eqs. 3-4): the difference becomes the
    class-conditional probability that the two values differ, estimated from
    relative frequencies of the known values, in distances as well as scores.
    The papers define this only for discrete features, so NaN in a continuous
    feature is rejected. Data with missing values runs on the CPU (``'auto'``
    selects it; an explicit GPU backend raises). Two conventions the papers
    leave open: a class with no known value of a feature uses the feature's
    overall frequencies, and a feature with no known value scores 0.
    """

    # Read by TuRF: NaNs in discrete features are handled (see Notes). Not an
    # ``allow_nan`` tag, because NaN in a continuous feature is still an error.
    _accepts_nan_in_discrete_features = True

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        discrete_limit: int = 10,
        n_neighbors: int = 3,
        backend: str = "auto",
        verbose: bool = False,
        n_jobs: int = -1,
    ):
        self.n_features_to_select = n_features_to_select
        self.discrete_limit = discrete_limit
        self.n_neighbors = n_neighbors
        self.backend = backend
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _validate_parameters(self, n_samples, n_features):
        """Validate all user-provided parameters."""
        if self.backend not in ["auto", "gpu", "cpu", "metal"]:
            raise ValueError("backend must be one of 'auto', 'gpu', 'cpu', or 'metal'")

        if n_samples < 2:
            raise ValueError(f"ReliefF requires at least 2 samples, but got n_samples = {n_samples}")

        check_integer_param(self.n_neighbors, "n_neighbors", 1)
        if self.backend == "metal" and self.n_neighbors > 64:
            raise ValueError("Experimental Metal ReliefF supports n_neighbors <= 64; use backend='cpu' for larger k")
        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be an integer "
                f"between 1 and n_samples - 1 ({n_samples - 1})."
            )
        check_integer_param(self.discrete_limit, "discrete_limit", 0)
        resolve_num_threads(self.n_jobs)

        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            n_select = max(1, int(self.n_features_to_select * n_features))
        elif isinstance(self.n_features_to_select, int) and not isinstance(self.n_features_to_select, bool):
            if not 0 < self.n_features_to_select <= n_features:
                raise ValueError(
                    f"If n_features_to_select is an int ({self.n_features_to_select}), "
                    f"it must be > 0 and <= n_features ({n_features})."
                )
            n_select = self.n_features_to_select
        else:
            raise TypeError("n_features_to_select must be an int or a float.")

        return n_select

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Calculates feature importances using the ReliefF algorithm."""
        x, y = validate_data(
            self,
            x,
            y,
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            y_numeric=True,
            ensure_all_finite="allow-nan",
        )
        check_classification_targets(y)
        self.n_features_in_ = x.shape[1]
        n_samples = x.shape[0]

        n_select = self._validate_parameters(n_samples, self.n_features_in_)
        missing = np.isnan(x)
        has_missing = bool(missing.any())
        if has_missing and self.backend in ("gpu", "metal"):
            raise ValueError("Missing values are handled on the CPU only; use backend='cpu' or 'auto'.")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float32)
            self.top_features_ = np.arange(n_select)
            self.effective_backend_ = "cpu" if self.backend != "gpu" else "gpu"
            return self

        min_class_size = np.min(np.bincount(y_encoded))
        if self.n_neighbors >= min_class_size:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than or equal to the "
                f"smallest class size ({min_class_size}).",
                UserWarning,
            )

        if has_missing:
            return self._fit_missing(x, y_encoded, missing, n_select)

        is_discrete = discrete_feature_mask(x, self.discrete_limit)
        self.is_discrete_ = is_discrete

        class_labels, class_counts = np.unique(y, return_counts=True)
        class_probs = (class_counts / len(y)).astype(np.float32)
        y_enc = np.searchsorted(class_labels, y).astype(np.int32)

        self.effective_backend_ = resolve_compute_backend(
            self.backend,
            cuda_available=is_cuda_ready() if self.backend in ("auto", "gpu") else False,
            metal_supported=self.n_neighbors <= 64,
        )

        # One representation for both backends: float64-normalised continuous
        # columns and exact category codes, narrowed to float32 only afterwards.
        columns, n_cont = split_discrete_last(is_discrete)
        x_prepared = prepare_relief_matrix(
            x, is_discrete, columns, dtype=np.float32, discrete_limit=self.discrete_limit
        )
        if self.effective_backend_ == "metal":
            from ._relieff_metal import relieff_metal

            if self.verbose:
                print("Running ReliefF on the Metal GPU now...")
            permuted_scores, self.metal_diagnostics_ = relieff_metal(
                x_prepared, y_enc, n_cont, self.n_neighbors, class_probs
            )
        elif self.effective_backend_ == "gpu":
            x_d = cuda.to_device(x_prepared)
            is_discrete_d = cuda.to_device(is_discrete[columns].astype(np.bool_))
            if self.verbose:
                print("Running ReliefF on the GPU now...")
            permuted_scores = _relieff_gpu_host_caller(x_d, y_enc, is_discrete_d, class_probs, self.n_neighbors)
        else:
            if self.verbose:
                print("Running ReliefF on the CPU now...")
            permuted_scores = _relieff_cpu_host_caller(
                x_prepared, y_enc, n_cont, self.n_neighbors, class_probs, self.n_jobs
            )
        scores = np.zeros(self.n_features_in_, dtype=np.float32)
        scores[columns] = permuted_scores

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
        return self

    def _fit_missing(self, x, y_enc, missing, n_select):
        """CPU fit for input with NaNs in discrete features (Eqs. 3-4)."""
        is_discrete = _discrete_mask_with_missing(x, missing, self.discrete_limit)
        continuous_with_missing = np.flatnonzero(missing.any(axis=0) & ~is_discrete)
        if continuous_with_missing.size:
            raise ValueError(
                "ReliefF handles missing values only in discrete features (at most "
                f"discrete_limit={self.discrete_limit} distinct known values), as the ReliefF papers "
                f"define them; features {continuous_with_missing.tolist()} are continuous and contain NaN."
            )
        self.is_discrete_ = is_discrete
        self.effective_backend_ = "cpu"

        n_classes = len(self.classes_)
        y_enc = np.asarray(y_enc, dtype=np.int32)
        class_probs = np.bincount(y_enc, minlength=n_classes) / len(y_enc)
        columns, n_cont = split_discrete_last(is_discrete)
        x_prepared, one_missing, both_missing = _prepare_missing_matrix(
            x, missing, is_discrete, columns, n_cont, y_enc, n_classes, self.discrete_limit
        )
        if self.verbose:
            print("Running ReliefF with missing values on the CPU now...")
        permuted_scores = _relieff_missing_cpu_host_caller(
            x_prepared, y_enc, n_cont, self.n_neighbors, class_probs, one_missing, both_missing, self.n_jobs
        )
        scores = np.zeros(self.n_features_in_, dtype=np.float32)
        scores[columns] = permuted_scores
        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:n_select]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Reduces x to the selected features."""
        check_is_fitted(self)

        x = validate_data(self, x, reset=False, dtype=[np.float64, np.float32], ensure_all_finite="allow-nan")
        missing_columns = np.isnan(x).any(axis=0)
        discrete = getattr(self, "is_discrete_", np.zeros(self.n_features_in_, dtype=bool))
        if np.any(missing_columns & ~discrete):
            raise ValueError("Input X contains NaN in features that were continuous during fit.")

        return x[:, self.top_features_]

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(x, y)
        return self.transform(x)
