from __future__ import annotations

import math
from typing import Literal

import numba
import numpy as np
from numba import njit, prange

from .utils import is_cuda_ready

try:
    from numba import cuda

    _CUDA_AVAILABLE = is_cuda_ready()
except Exception:  # pragma: no cover
    _CUDA_AVAILABLE = False


def _validate_discrete(arr: np.ndarray, name: str) -> np.ndarray:
    """Ensure *arr* is 1-D or 2-D integer array with non-negative entries."""
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"{name} must be an integer-coded array (got {arr.dtype}). "
            "Discretise continuous data before calling this function."
        )
    if arr.min() < 0:
        raise ValueError(f"{name} contains negative values; expected 0..K-1 codes.")
    return arr.astype(np.int32, copy=False)


@njit(cache=True, nogil=True, fastmath=True)
def _mi_pair_cpu(x1: np.ndarray, x2: np.ndarray, log_base: float) -> float:  # pragma: no cover
    n = x1.shape[0]
    k1 = int(x1.max()) + 1
    k2 = int(x2.max()) + 1

    table = np.zeros((k1, k2), dtype=np.float64)
    for i in range(n):
        table[x1[i], x2[i]] += 1.0

    table /= n  # joint probabilities
    p1 = table.sum(axis=1)
    p2 = table.sum(axis=0)

    mi = 0.0
    eps = 1e-12
    for i in range(k1):
        for j in range(k2):
            pxy = table[i, j]
            if pxy > eps:
                mi += pxy * math.log(pxy / (p1[i] * p2[j]))
    return mi / log_base


@njit(parallel=True, cache=True, fastmath=True)
def _batch_mi_cpu(X: np.ndarray, y: np.ndarray, log_base: float) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    n_samples, n_features = X.shape
    relevance = np.empty(n_features, dtype=np.float64)
    redundancy = np.zeros((n_features, n_features), dtype=np.float64)

    for f in prange(n_features):
        relevance[f] = _mi_pair_cpu(X[:, f], y, log_base)

    for i in prange(n_features):
        for j in range(i + 1, n_features):
            mi = _mi_pair_cpu(X[:, i], X[:, j], log_base)
            redundancy[i, j] = mi
            redundancy[j, i] = mi
    return relevance, redundancy


_MAX_STATES_GPU = 32
_THREADS_PER_BLOCK = (16, 16)


def resolve_backend(backend: str, max_state: int) -> str:
    """Return the backend that will actually run: ``'cpu'`` or ``'gpu'``.

    ``'auto'`` prefers the GPU but falls back to the CPU when CUDA is
    unavailable or the data has more distinct states than the GPU kernels
    support.  An explicit ``'gpu'`` raises instead of falling back, so a caller
    that asked for the GPU never silently gets the CPU.

    This is the single decision point for the backend; callers that need to
    report which backend ran should use it rather than re-deriving the rule.
    """
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")

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

    if _CUDA_AVAILABLE and max_state <= _MAX_STATES_GPU:
        return "gpu"
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
    backend: Literal["auto", "cpu", "gpu"] = "auto",
    unit: Literal["bit", "nat"] = "bit",
) -> float:  # pragma: no cover
    """Mutual information I(x1; x2) for *discrete* 1-D arrays.

    Raises ``ValueError`` if inputs are not integer-coded.
    """
    if x1.ndim != 1 or x2.ndim != 1 or x1.shape != x2.shape:
        raise ValueError("x1 and x2 must be 1-D arrays of equal length")

    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
    if unit not in ("bit", "nat"):
        raise ValueError("unit must be 'bit' or 'nat'")

    x1_d = _validate_discrete(x1.ravel(), "x1")
    x2_d = _validate_discrete(x2.ravel(), "x2")

    log_base = math.log(2.0) if unit == "bit" else 1.0

    max_state = int(max(x1_d.max(), x2_d.max())) + 1

    if resolve_backend(backend, max_state) == "gpu":
        out = cuda.device_array(1, dtype=np.float32)
        _mi_pair_gpu_kernel[1, _THREADS_PER_BLOCK](x1_d.reshape(-1, 1), x2_d, out, max_state, log_base)
        return float(out.copy_to_host()[0])

    return _mi_pair_cpu(x1_d, x2_d, log_base)


def calculate_mi_matrices(
    X: np.ndarray,
    y: np.ndarray,
    *,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
    unit: Literal["bit", "nat"] = "bit",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (relevance, redundancy) using discrete data only.

    * X.shape == (n_samples, n_features)
    * y.shape == (n_samples,)
    * All values must be integer codes >=0.
    """
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2-D and y 1-D with matching sample size")

    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
    if unit not in ("bit", "nat"):
        raise ValueError("unit must be 'bit' or 'nat'")

    X_d = _validate_discrete(X, "X")
    y_d = _validate_discrete(y, "y")

    max_state = int(max(X_d.max(), y_d.max())) + 1
    log_base = math.log(2.0) if unit == "bit" else 1.0

    if resolve_backend(backend, max_state) == "gpu":
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
    return _batch_mi_cpu(X_d, y_d, log_base)
