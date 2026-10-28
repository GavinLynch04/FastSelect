from __future__ import annotations

import math
from typing import Literal

import numba
import numpy as np
from numba import njit, prange

from .utils import is_cuda_ready, is_metal_ready

try:
    from numba import cuda

    _CUDA_AVAILABLE = is_cuda_ready()
except Exception:  # pragma: no cover
    _CUDA_AVAILABLE = False


def _validate_discrete(arr: np.ndarray, name: str) -> np.ndarray:
    """Return *arr* as a non-negative int32 code array without aliasing symbols.

    Mutual information depends only on the empirical joint distribution, so a
    variable's symbols may be relabelled freely.  Codes that are already small
    (below ``2 * n_samples + 32``) are used as-is.  Larger codes, including
    values that do not fit in int32, are remapped to dense ``0..K-1`` codes per
    column so that distinct symbols stay distinct and the joint histograms stay
    bounded by the number of observed symbols rather than by the code magnitude.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"{name} must be an integer-coded array (got {arr.dtype}). "
            "Discretise continuous data before calling this function."
        )
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if arr.min() < 0:
        raise ValueError(f"{name} contains negative values; expected 0..K-1 codes.")

    compact_below = 2 * arr.shape[0] + 32
    if arr.max() < compact_below:
        return arr.astype(np.int32, copy=False)

    if arr.ndim == 1:
        return np.unique(arr, return_inverse=True)[1].reshape(-1).astype(np.int32)

    out = np.empty(arr.shape, dtype=np.int32)
    for column in range(arr.shape[1]):
        values = arr[:, column]
        if values.max() < compact_below:
            out[:, column] = values
        else:
            out[:, column] = np.unique(values, return_inverse=True)[1].reshape(-1)
    return out


@njit(cache=True, nogil=True, fastmath=True)
def _mi_pair_cpu(x1: np.ndarray, x2: np.ndarray, log_base: float) -> float:  # pragma: no cover
    n = x1.shape[0]
    k1 = int(x1.max()) + 1
    k2 = int(x2.max()) + 1

    table = np.zeros((k1, k2), dtype=np.float64)
    counts1 = np.zeros(k1, dtype=np.float64)
    counts2 = np.zeros(k2, dtype=np.float64)
    for sample in range(n):
        state1 = x1[sample]
        state2 = x2[sample]
        table[state1, state2] += 1.0
        counts1[state1] += 1.0
        counts2[state2] += 1.0

    return _mi_from_counts(table, counts1, counts2, n, log_base)


@njit(cache=True, nogil=True, fastmath=True)
def _mi_from_counts(table, counts1, counts2, n, log_base):  # pragma: no cover
    """``sum p(x,y) * log(p(x,y) / (p(x) p(y))) / log_base`` from exact float64 counts."""
    k1, k2 = table.shape
    mi = 0.0
    eps = 1e-12
    for i in range(k1):
        for j in range(k2):
            pxy = table[i, j] / n
            if pxy > eps:
                mi += pxy * math.log(pxy / ((counts1[i] / n) * (counts2[j] / n)))
    return mi / log_base


@njit(parallel=True, cache=True, fastmath=True)
def _batch_mi_cpu(X: np.ndarray, y: np.ndarray, log_base: float) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    n_samples, n_features = X.shape
    columns = np.ascontiguousarray(X.T)
    relevance = np.empty(n_features, dtype=np.float64)
    redundancy = np.zeros((n_features, n_features), dtype=np.float64)

    for f in prange(n_features):
        relevance[f] = _mi_pair_cpu(columns[f], y, log_base)

    for task in prange(n_features):
        i = task // 2 if task % 2 == 0 else n_features - 1 - task // 2
        for j in range(i + 1, n_features):
            mi = _mi_pair_cpu(columns[i], columns[j], log_base)
            redundancy[i, j] = mi
            redundancy[j, i] = mi
    return relevance, redundancy


# When every joint table has at most _COUNTED_MAX_CELLS cells, the matrices
# reuse each column's marginal counts and count only the joint table: up to
# _INTERLEAVED_MAX_CELLS in four interleaved integer sub-tables, so increments
# of one cell by consecutive samples do not form a serial dependency chain,
# and otherwise directly. Larger tables are dominated by the per-cell
# logarithms and keep _batch_mi_cpu, which was as fast there.
_INTERLEAVED_MAX_CELLS = 64
_COUNTED_MAX_CELLS = 256


@njit(cache=True, nogil=True)
def _joint_counts(x1, x2, k2, cells, lanes, flat):  # pragma: no cover
    """Write the exact joint counts of two code columns into ``flat[:cells]``.

    Cell ``(a, b)`` is ``flat[a * k2 + b]`` (row-major ``k1 x k2``). ``lanes``
    is ``(4, _INTERLEAVED_MAX_CELLS)`` integer scratch space. Integer sums of
    the sub-tables are exact, so every path equals the sequential float64
    counting of :func:`_mi_pair_cpu`.
    """
    n = x1.shape[0]
    if cells > _INTERLEAVED_MAX_CELLS:
        for cell in range(cells):
            flat[cell] = 0.0
        for sample in range(n):
            flat[x1[sample] * k2 + x2[sample]] += 1.0
        return
    for lane in range(4):
        for cell in range(cells):
            lanes[lane, cell] = 0
    sample = 0
    while sample + 4 <= n:
        lanes[0, x1[sample] * k2 + x2[sample]] += 1
        lanes[1, x1[sample + 1] * k2 + x2[sample + 1]] += 1
        lanes[2, x1[sample + 2] * k2 + x2[sample + 2]] += 1
        lanes[3, x1[sample + 3] * k2 + x2[sample + 3]] += 1
        sample += 4
    while sample < n:
        lanes[0, x1[sample] * k2 + x2[sample]] += 1
        sample += 1
    for cell in range(cells):
        flat[cell] = lanes[0, cell] + lanes[1, cell] + lanes[2, cell] + lanes[3, cell]


@njit(cache=True, nogil=True)
def _mi_columns(x1, x2, k1, k2, counts1, counts2, log_base, lanes, flat):  # pragma: no cover
    """``_mi_pair_cpu(x1, x2)``, identical to it, given state and marginal counts.

    ``lanes`` and ``flat`` are scratch space for a ``k1 x k2`` table of at
    most ``_COUNTED_MAX_CELLS`` cells.
    """
    cells = k1 * k2
    _joint_counts(x1, x2, k2, cells, lanes, flat)
    return _mi_from_counts(flat[:cells].reshape((k1, k2)), counts1, counts2, x1.shape[0], log_base)


@njit(parallel=True, cache=True, fastmath=True)
def _batch_mi_counted_cpu(
    X: np.ndarray, y: np.ndarray, log_base: float, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """:func:`_batch_mi_cpu` for codes below 16, with identical entries.

    ``states[f]`` is ``X[:, f].max() + 1``; every ``states`` product, including
    with the target's, is at most ``_COUNTED_MAX_CELLS``. Each column's
    marginal counts are computed once rather than once per pair.
    """
    n_samples, n_features = X.shape
    columns = np.ascontiguousarray(X.T)
    relevance = np.empty(n_features, dtype=np.float64)
    redundancy = np.zeros((n_features, n_features), dtype=np.float64)

    offsets = np.zeros(n_features + 1, dtype=np.int64)
    for f in range(n_features):
        offsets[f + 1] = offsets[f] + states[f]
    marginals = np.zeros(offsets[n_features], dtype=np.float64)
    y_states = int(y.max()) + 1
    y_counts = np.zeros(y_states, dtype=np.float64)
    for sample in range(n_samples):
        y_counts[y[sample]] += 1.0

    for f in prange(n_features):
        # Count locally, then publish: neighboring columns share cache lines.
        counts = np.zeros(states[f], dtype=np.float64)
        for sample in range(n_samples):
            counts[columns[f, sample]] += 1.0
        marginals[offsets[f] : offsets[f + 1]] = counts
        lanes = np.empty((4, _INTERLEAVED_MAX_CELLS), dtype=np.int64)
        flat = np.empty(_COUNTED_MAX_CELLS, dtype=np.float64)
        relevance[f] = _mi_columns(columns[f], y, states[f], y_states, counts, y_counts, log_base, lanes, flat)

    for task in prange(n_features):
        i = task // 2 if task % 2 == 0 else n_features - 1 - task // 2
        lanes = np.empty((4, _INTERLEAVED_MAX_CELLS), dtype=np.int64)
        flat = np.empty(_COUNTED_MAX_CELLS, dtype=np.float64)
        counts_i = marginals[offsets[i] : offsets[i + 1]]
        for j in range(i + 1, n_features):
            mi = _mi_columns(
                columns[i],
                columns[j],
                states[i],
                states[j],
                counts_i,
                marginals[offsets[j] : offsets[j + 1]],
                log_base,
                lanes,
                flat,
            )
            redundancy[i, j] = mi
            redundancy[j, i] = mi
    return relevance, redundancy


_MAX_STATES_GPU = 32
_THREADS_PER_BLOCK = (16, 16)


def resolve_backend(backend: str, max_state: int) -> str:
    """Prefer supported CUDA (32 states), then Metal (64 states), then CPU.

    Explicit ``gpu`` retains its NVIDIA/CUDA meaning. Explicit ``metal``
    requires MLX and a physical Apple GPU; it raises on unsupported input.
    """
    if backend not in ("auto", "cpu", "gpu", "metal"):
        raise ValueError("backend must be 'auto', 'cpu', 'gpu', or 'metal'")

    if backend == "cpu":
        return "cpu"

    if backend == "gpu":
        if not _CUDA_AVAILABLE:
            raise RuntimeError("backend='gpu' requested but CUDA not available")
        if max_state > _MAX_STATES_GPU:
            raise RuntimeError(
                f"GPU backend supports at most {_MAX_STATES_GPU} states " f"(got {max_state}); try backend='cpu'"
            )
        return "gpu"

    if backend == "metal":
        if max_state > 64:
            raise ValueError("Metal backend supports at most 64 states; try backend='cpu'")
        if not is_metal_ready():
            raise RuntimeError("backend='metal' requires MLX and an available Metal GPU")
        return "metal"
    if _CUDA_AVAILABLE and max_state <= _MAX_STATES_GPU:
        return "gpu"
    if max_state <= 64 and is_metal_ready():
        return "metal"
    return "cpu"


_REDUNDANCY_TPB = 128  # must stay a power of two for the block reduction


@cuda.jit
def _mi_matrix_gpu_kernel(X, out, n_states, log_base):  # pragma: no cover
    """Pairwise mutual information for every feature pair, one block per pair.

    Computes the same quantity as :func:`_mi_pair_cpu`,
    ``sum p(x,y) * log(p(x,y) / (p(x) p(y))) / log_base``, with the whole block
    cooperating on the joint histogram.  The 2-D grid addresses pairs directly;
    blocks below the diagonal exit immediately.

    ``X`` is feature-major so consecutive threads read consecutive samples.
    """
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    n_features = X.shape[0]

    if i >= n_features or j >= n_features or j <= i:
        return

    tid = cuda.threadIdx.x
    n_threads = cuda.blockDim.x
    n_samples = X.shape[1]

    joint = cuda.shared.array(shape=(_MAX_STATES_GPU, _MAX_STATES_GPU), dtype=numba.float32)
    counts_x = cuda.shared.array(shape=_MAX_STATES_GPU, dtype=numba.float32)
    counts_y = cuda.shared.array(shape=_MAX_STATES_GPU, dtype=numba.float32)
    partial = cuda.shared.array(shape=_REDUNDANCY_TPB, dtype=numba.float32)

    for cell in range(tid, n_states * n_states, n_threads):
        joint[cell // n_states, cell % n_states] = 0.0
    cuda.syncthreads()

    row_i = X[i]
    row_j = X[j]
    for s in range(tid, n_samples, n_threads):
        cuda.atomic.add(joint, (row_i[s], row_j[s]), 1.0)
    cuda.syncthreads()

    for r in range(tid, n_states, n_threads):
        total = 0.0
        for c in range(n_states):
            total += joint[r, c]
        counts_x[r] = total
    for c in range(tid, n_states, n_threads):
        total = 0.0
        for r in range(n_states):
            total += joint[r, c]
        counts_y[c] = total
    cuda.syncthreads()

    local_mi = 0.0
    eps = 1e-12
    for cell in range(tid, n_states * n_states, n_threads):
        r = cell // n_states
        c = cell - r * n_states
        count = joint[r, c]
        if count > 0.0:
            p_xy = count / n_samples
            if p_xy > eps:
                p_x = counts_x[r] / n_samples
                p_y = counts_y[c] / n_samples
                local_mi += p_xy * math.log(p_xy / (p_x * p_y))
    partial[tid] = local_mi
    cuda.syncthreads()

    offset = n_threads // 2
    while offset > 0:
        if tid < offset:
            partial[tid] += partial[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        mi = partial[0] / log_base
        out[i, j] = mi
        out[j, i] = mi


@cuda.jit
def _mi_pair_gpu_kernel(X, y, out, n_states, log_base):  # pragma: no cover
    feature_idx = cuda.blockIdx.x
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    linear_tid = tx + ty * cuda.blockDim.x
    threads_per_block = cuda.blockDim.x * cuda.blockDim.y

    cont = cuda.shared.array((_MAX_STATES_GPU, _MAX_STATES_GPU), dtype=numba.float32)

    # zero shared memory cooperatively
    for r in range(tx, n_states, cuda.blockDim.x):
        for c in range(ty, n_states, cuda.blockDim.y):
            cont[r, c] = 0.0
    cuda.syncthreads()

    n = X.shape[0]
    idx = linear_tid
    while idx < n:
        r = int(X[idx, feature_idx])
        c = int(y[idx])
        cuda.atomic.add(cont, (r, c), 1.0)
        idx += threads_per_block
    cuda.syncthreads()

    if tx == 0 and ty == 0:
        for r in range(n_states):
            for c in range(n_states):
                cont[r, c] /= n

        px = cuda.local.array(_MAX_STATES_GPU, dtype=numba.float32)
        py = cuda.local.array(_MAX_STATES_GPU, dtype=numba.float32)
        for i in range(n_states):
            px[i] = 0.0
            py[i] = 0.0
        for r in range(n_states):
            for c in range(n_states):
                px[r] += cont[r, c]
                py[c] += cont[r, c]

        mi = 0.0
        eps = 1e-12
        for r in range(n_states):
            for c in range(n_states):
                pxy = cont[r, c]
                if pxy > eps:
                    mi += pxy * math.log(pxy / (px[r] * py[c]))
        out[feature_idx] = mi / log_base


def calculate_mi_single_pair(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    backend: Literal["auto", "cpu", "gpu", "metal"] = "auto",
    unit: Literal["bit", "nat"] = "bit",
) -> float:  # pragma: no cover
    """Mutual information I(x1; x2) for *discrete* 1-D arrays.

    Raises ``ValueError`` if inputs are not integer-coded.
    """
    if x1.ndim != 1 or x2.ndim != 1 or x1.shape != x2.shape:
        raise ValueError("x1 and x2 must be 1-D arrays of equal length")

    if backend not in ("auto", "cpu", "gpu", "metal"):
        raise ValueError("backend must be 'auto', 'cpu', 'gpu', or 'metal'")
    if unit not in ("bit", "nat"):
        raise ValueError("unit must be 'bit' or 'nat'")

    x1_d = _validate_discrete(x1.ravel(), "x1")
    x2_d = _validate_discrete(x2.ravel(), "x2")

    log_base = math.log(2.0) if unit == "bit" else 1.0

    max_state = int(max(x1_d.max(), x2_d.max())) + 1

    effective_backend = resolve_backend(backend, max_state)
    if effective_backend == "metal":
        from ._information_metal import information_metal

        relevance, _ = information_metal(x1_d.reshape(-1, 1), x2_d, log_base)
        return float(relevance[0])
    if effective_backend == "gpu":
        out = cuda.device_array(1, dtype=np.float32)
        _mi_pair_gpu_kernel[1, _THREADS_PER_BLOCK](x1_d.reshape(-1, 1), x2_d, out, max_state, log_base)
        return float(out.copy_to_host()[0])

    return _mi_pair_cpu(x1_d, x2_d, log_base)


def calculate_mi_matrices(
    X: np.ndarray,
    y: np.ndarray,
    *,
    backend: Literal["auto", "cpu", "gpu", "metal"] = "auto",
    unit: Literal["bit", "nat"] = "bit",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (relevance, redundancy) using discrete data only.

    * X.shape == (n_samples, n_features)
    * y.shape == (n_samples,)
    * All values must be integer codes >=0. Codes need not be dense: they are
      remapped per column when large, so distinct symbols are never merged.
    """
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2-D and y 1-D with matching sample size")

    if backend not in ("auto", "cpu", "gpu", "metal"):
        raise ValueError("backend must be 'auto', 'cpu', 'gpu', or 'metal'")
    if unit not in ("bit", "nat"):
        raise ValueError("unit must be 'bit' or 'nat'")

    X_d = _validate_discrete(X, "X")
    y_d = _validate_discrete(y, "y")

    max_state = int(max(X_d.max(), y_d.max())) + 1
    log_base = math.log(2.0) if unit == "bit" else 1.0

    effective_backend = resolve_backend(backend, max_state)
    if effective_backend == "metal":
        from ._information_metal import information_metal

        return information_metal(X_d, y_d, log_base)
    if effective_backend == "gpu":
        n_samples, n_features = X_d.shape
        X_gpu = cuda.to_device(X_d)
        y_gpu = cuda.to_device(y_d)
        relevance_gpu = cuda.device_array(n_features, dtype=np.float32)
        _mi_pair_gpu_kernel[(n_features,), _THREADS_PER_BLOCK](X_gpu, y_gpu, relevance_gpu, max_state, log_base)
        relevance = relevance_gpu.copy_to_host().astype(np.float64)

        # The redundancy matrix is the O(n_features^2 * n_samples) term, so it
        # runs on the device too: one block per feature pair, feature-major so
        # consecutive threads read consecutive samples.
        X_columns = cuda.to_device(np.ascontiguousarray(X_d.T))
        redundancy_gpu = cuda.device_array((n_features, n_features), dtype=np.float32)
        redundancy_gpu[:] = 0.0
        _mi_matrix_gpu_kernel[(n_features, n_features), _REDUNDANCY_TPB](X_columns, redundancy_gpu, max_state, log_base)
        redundancy = redundancy_gpu.copy_to_host().astype(np.float64)
        return relevance, redundancy

    # Pure CPU path
    if max_state * max_state <= _COUNTED_MAX_CELLS:
        return _batch_mi_counted_cpu(X_d, y_d, log_base, X_d.max(axis=0).astype(np.int64) + 1)
    return _batch_mi_cpu(X_d, y_d, log_base)
