from __future__ import annotations
from math import log
import numpy as np
from numba import njit, prange, float32, int32, cuda

MAX_SHARED_STATES = 32
THREADS_PER_BLOCK = (16, 16)


@njit(cache=True, fastmath=True, nogil=True)
def _calculate_mi_cpu_kernel(x1: np.ndarray, n_states1: int, x2: np.ndarray, n_states2: int) -> float:
    """
    Numba JIT-compiled kernel to calculate Mutual Information between two discrete vectors.
    This is the core CPU implementation.
    """
    n_samples = x1.shape[0]
    contingency_table = np.zeros((n_states1, n_states2), dtype=np.float32)

    # Populate contingency table
    for i in range(n_samples):
        val1 = int(x1[i])
        val2 = int(x2[i])
        contingency_table[val1, val2] += 1

    contingency_table /= n_samples

    # Calculate marginal probabilities
    p1 = np.sum(contingency_table, axis=1)
    p2 = np.sum(contingency_table, axis=0)

    # Calculate mutual information
    mi = 0.0
    for i in range(n_states1):
        for j in range(n_states2):
            p_xy = contingency_table[i, j]
            p_x = p1[i]
            p_y = p2[j]
            if p_xy > 1e-12:  # Use epsilon to avoid log(0)
                mi += p_xy * log(p_xy / (p_x * p_y))

    return mi


def calculate_mi_single_pair(x1: np.ndarray, x2: np.ndarray, n_states: int, backend: str = 'cpu') -> float:
    """
    Calculates the Mutual Information between two discrete vectors.
    This function is a dispatcher for CPU or GPU backends. It is best used for
    calculating single MI values iteratively. For bulk calculations (e.g., full
    redundancy matrix), use `calculate_mi_matrices`.

    Parameters
    ----------
    x1 : np.ndarray
        The first discrete data vector.
    x2 : np.ndarray
        The second discrete data vector.
    n_states : int
        The total number of unique discrete states in the dataset.
    backend : {'cpu', 'gpu'}, default='cpu'
        The computational backend to use.

    Returns
    -------
    float
        The calculated Mutual Information I(x1; x2).
    """
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("Input arrays x1 and x2 must be 1-dimensional.")
    if x1.shape != x2.shape:
        raise ValueError("Input arrays x1 and x2 must have the same shape.")

    if backend == 'cpu':
        n_states1 = int(np.max(x1)) + 1
        n_states2 = int(np.max(x2)) + 1
        return _calculate_mi_cpu_kernel(x1, n_states1, x2, n_states2)

    elif backend == 'gpu':
        raise NotImplementedError("Single-pair GPU MI is not recommended due to kernel launch overhead. "
                                  "Use the bulk `calculate_mi_matrices` for GPU efficiency.")
    else:
        raise ValueError(f"Unsupported backend: '{backend}'. Choose 'cpu' or 'gpu'.")


@cuda.jit
def _relevance_kernel_gpu(X_gpu, y_gpu, relevance_out, n_samples, n_states): # pragma: no cover
    """CUDA kernel to calculate all relevance scores I(f; y) in parallel."""
    shared_contingency = cuda.shared.array(shape=(MAX_SHARED_STATES, MAX_SHARED_STATES), dtype=float32)
    feature_idx = cuda.blockIdx.x
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    # Zero out shared memory
    if tx < n_states and ty < n_states:
        shared_contingency[tx, ty] = 0.0
    cuda.syncthreads()

    # Populate contingency table from global memory using atomic operations
    for sample_idx in range(n_samples):
        val1 = int(X_gpu[sample_idx, feature_idx])
        val2 = int(y_gpu[sample_idx])
        cuda.atomic.add(shared_contingency, (val1, val2), 1.0)
    cuda.syncthreads()

    # Single thread per block calculates the final MI score
    if tx == 0 and ty == 0:
        # Normalize to get joint probability p(x,y)
        for r in range(n_states):
            for c in range(n_states):
                shared_contingency[r,c] /= n_samples

        # Calculate marginal probabilities p(x) and p(y)
        p_x = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        p_y = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        for i in range(n_states):
            p_x[i] = 0.0
            p_y[i] = 0.0
        for r in range(n_states):
            for c in range(n_states):
                p_x[r] += shared_contingency[r, c]
                p_y[c] += shared_contingency[r, c]

        # Calculate MI
        mi = 0.0
        for r in range(n_states):
            for c in range(n_states):
                p_xy_val = shared_contingency[r, c]
                if p_xy_val > 1e-12:
                    p_x_r = p_x[r]
                    p_y_c = p_y[c]
                    mi += p_xy_val * log(p_xy_val / (p_x_r * p_y_c))
        relevance_out[feature_idx] = mi


@cuda.jit
def _redundancy_kernel_gpu(X_gpu, redundancy_out, n_features, n_samples, n_states): # pragma: no cover
    """CUDA kernel to calculate the full redundancy matrix I(f_i; f_j) in parallel."""
    shared_contingency = cuda.shared.array(shape=(MAX_SHARED_STATES, MAX_SHARED_STATES), dtype=float32)
    f1_idx, f2_idx = cuda.blockIdx.x, cuda.blockIdx.y

    # Each block handles one pair (f1, f2). Exploit symmetry.
    if f2_idx <= f1_idx:
        return

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    if tx < n_states and ty < n_states:
        shared_contingency[tx, ty] = 0.0
    cuda.syncthreads()

    for sample_idx in range(n_samples):
        val1 = int(X_gpu[sample_idx, f1_idx])
        val2 = int(X_gpu[sample_idx, f2_idx])
        cuda.atomic.add(shared_contingency, (val1, val2), 1.0)
    cuda.syncthreads()

    # The rest of the logic is identical to the relevance kernel
    if tx == 0 and ty == 0:
        for r in range(n_states):
            for c in range(n_states):
                shared_contingency[r, c] /= n_samples
        p_x = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        p_y = cuda.local.array(MAX_SHARED_STATES, dtype=float32)
        for i in range(n_states):
            p_x[i] = 0.0
            p_y[i] = 0.0
        for r in range(n_states):
            for c in range(n_states):
                p_x[r] += shared_contingency[r, c]
                p_y[c] += shared_contingency[r, c]
        mi = 0.0
        for r in range(n_states):
            for c in range(n_states):
                p_xy_val = shared_contingency[r, c]
                if p_xy_val > 1e-12:
                    p_x_r = p_x[r]
                    p_y_c = p_y[c]
                    mi += p_xy_val * log(p_xy_val / (p_x_r * p_y_c))

        # Store result symmetrically in the output matrix
        redundancy_out[f1_idx, f2_idx] = mi
        redundancy_out[f2_idx, f1_idx] = mi


@njit(parallel=True, cache=True)
def _calculate_mi_matrices_cpu(X, y): # pragma: no cover
    """Numba JIT host function to compute relevance and redundancy on the CPU."""
    n_samples, n_features = X.shape
    n_states_X = np.zeros(n_features, dtype=np.int32)
    for i in prange(n_features):
        n_states_X[i] = int(np.max(X[:, i])) + 1
    n_states_y = int(np.max(y)) + 1

    relevance_scores = np.zeros(n_features, dtype=np.float32)
    for i in prange(n_features):
        relevance_scores[i] = _calculate_mi_cpu_kernel(
            X[:, i], n_states_X[i], y, n_states_y
        )

    redundancy_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    for i in prange(n_features):
        for j in range(i + 1, n_features):
            mi = _calculate_mi_cpu_kernel(
                X[:, i], n_states_X[i], X[:, j], n_states_X[j]
            )
            redundancy_matrix[i, j] = mi
            redundancy_matrix[j, i] = mi

    return relevance_scores, redundancy_matrix


def calculate_mi_matrices(X: np.ndarray, y: np.ndarray, n_states: int, backend: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-computes the relevance vector and redundancy matrix using Mutual Information.
    This function is optimized for bulk computation and is ideal for mRMR.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The encoded, discrete training data.
    y : np.ndarray of shape (n_samples,)
        The encoded, discrete target values.
    n_states : int
        The total number of unique discrete states across X and y.
        Required for the GPU backend.
    backend : {'cpu', 'gpu'}, default='cpu'
        The computational backend to use.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - relevance_scores (ndarray of shape (n_features,)): I(f; y) for each feature f.
        - redundancy_matrix (ndarray of shape (n_features, n_features)): I(f_i; f_j).
    """
    if backend == 'cpu':
        return _calculate_mi_matrices_cpu(X, y)

    elif backend == 'gpu':
        if not cuda.is_available():
             raise RuntimeError("GPU backend selected, but no CUDA-enabled GPU found.")
        if n_states > MAX_SHARED_STATES:
            raise ValueError(
                f"GPU backend supports a maximum of {MAX_SHARED_STATES} unique discrete states, "
                f"but data has {n_states}."
            )

        n_samples, n_features = X.shape
        X_gpu = cuda.to_device(np.ascontiguousarray(X))
        y_gpu = cuda.to_device(np.ascontiguousarray(y))

        # Launch Relevance Kernel
        relevance_gpu = cuda.device_array(n_features, dtype=np.float32)
        blocks_per_grid_rel = (n_features,)
        _relevance_kernel_gpu[blocks_per_grid_rel, THREADS_PER_BLOCK[0]](
            X_gpu, y_gpu, relevance_gpu, n_samples, n_states
        )

        # Launch Redundancy Kernel
        redundancy_gpu = cuda.device_array((n_features, n_features), dtype=np.float32)
        blocks_per_grid_red = (n_features, n_features)
        _redundancy_kernel_gpu[blocks_per_grid_red, THREADS_PER_BLOCK](
            X_gpu, redundancy_gpu, n_features, n_samples, n_states
        )

        cuda.synchronize()
        return relevance_gpu.copy_to_host(), redundancy_gpu.copy_to_host()
    else:
        raise ValueError(f"Unsupported backend: '{backend}'. Choose 'cpu' or 'gpu'.")
