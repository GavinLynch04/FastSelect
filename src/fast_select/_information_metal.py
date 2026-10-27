"""Exact Metal joint counts, CPU precision for Shannon MI and Hall SU.

One-hot products are exact integer sums for n <= 2**24 (0/1 products are
exact even with reduced-mantissa multiplies). Larger allocations use bounded
integer histograms. No GPU logarithm approximation may change search ties.
"""

from functools import lru_cache

import numpy as np
from numba import njit, prange

from ._metal import get_mx, metal_entry

MAX_STATES = 64


@lru_cache(maxsize=1)
def _histogram_kernel():
    return get_mx().fast.metal_kernel(
        name="information_integer_histograms",
        input_names=["columns", "pairs"],
        output_names=["tables"],
        source=r"""
        uint pair = threadgroup_position_in_grid.x, lane = thread_index_in_threadgroup;
        uint n = columns_shape[1];
        threadgroup atomic_uint counts[S * S];
        for (uint cell = lane; cell < S*S; cell += 128)
            atomic_store_explicit(&counts[cell], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint a = pairs[pair * 2], b = pairs[pair * 2 + 1];
        for (uint i = lane; i < n; i += 128) {
            uint cell = uint(columns[a*n+i])*S + uint(columns[b*n+i]);
            atomic_fetch_add_explicit(&counts[cell], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint cell = lane; cell < S*S; cell += 128)
            tables[pair*S*S+cell] = atomic_load_explicit(&counts[cell], memory_order_relaxed);
        """,
        compile_options={"math_mode": "safe"},
    )


def _tables(columns, pairs, states):
    mx = get_mx()
    variables, samples = columns.shape
    width = variables * states
    # Bound both GPU one-hot input and its Gram matrix independently.
    if samples <= 2**24 and width * width * 4 <= 64 * 1024**2 and width * samples * 4 <= 128 * 1024**2:
        encoded = mx.array(columns)
        with mx.stream(mx.gpu):
            hot = (encoded[:, None, :] == mx.arange(states)[None, :, None]).astype(mx.float32)
            hot = hot.reshape(width, samples)
            counts = np.array(hot @ hot.T).reshape(variables, states, variables, states).transpose(0, 2, 1, 3)
        for start in range(0, len(pairs), 8192):
            batch = pairs[start : start + 8192]
            yield start, np.ascontiguousarray(counts[batch[:, 0], batch[:, 1]], dtype=np.int32)
    else:
        columns_gpu = mx.array(columns)
        batch_size = max(1, min(8192, 8 * 1024**2 // (states * states * 4)))
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            result = _histogram_kernel()(
                inputs=[columns_gpu, mx.array(batch)],
                template=[("S", states)],
                grid=(len(batch) * 128, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(len(batch), states, states)],
                output_dtypes=[mx.uint32],
                stream=mx.gpu,
            )[0]
            yield start, np.array(result).astype(np.int32)


@njit(parallel=True, cache=True)
def _reduce_tables(tables, pairs, state_counts, n, log_base, cfs):
    values = np.zeros(len(tables), dtype=np.float64)
    for pair in prange(len(tables)):
        na = state_counts[pairs[pair, 0]]
        nb = state_counts[pairs[pair, 1]]
        if cfs:
            # Match the existing CFS float32 empirical probabilities exactly.
            probabilities = tables[pair, :na, :nb].astype(np.float32) / n
            px = np.sum(probabilities, axis=1)
            py = np.sum(probabilities, axis=0)
            total = 0.0
            for a in range(na):
                for b in range(nb):
                    p = probabilities[a, b]
                    if p > 1e-12 and px[a] > 1e-12 and py[b] > 1e-12:
                        total += p * np.log2(p / (px[a] * py[b]))
            values[pair] = total
        else:
            ca = np.zeros(na, dtype=np.float64)
            cb = np.zeros(nb, dtype=np.float64)
            for a in range(na):
                for b in range(nb):
                    ca[a] += tables[pair, a, b]
                    cb[b] += tables[pair, a, b]
            total = 0.0
            for a in range(na):
                for b in range(nb):
                    p = tables[pair, a, b] / n
                    if p > 1e-12:
                        total += p * np.log(p / ((ca[a] / n) * (cb[b] / n)))
            values[pair] = total / log_base
    return values


@metal_entry
def information_metal(X, y, log_base=1.0, *, cfs=False, state_counts=None):
    """Compute every relevant pair; preserve the public zero redundancy diagonal."""
    columns = np.ascontiguousarray(np.vstack((X.T, y)), dtype=np.int32)
    p = X.shape[1]
    counts = columns.max(axis=1).astype(np.int64) + 1 if state_counts is None else state_counts
    states = int(counts.max())
    if states > MAX_STATES:
        raise ValueError(f"Metal information kernels support at most {MAX_STATES} states")
    left, right = np.triu_indices(p, k=1)
    pairs = np.concatenate((np.column_stack((np.arange(p), np.full(p, p))), np.column_stack((left, right))))
    pairs = np.ascontiguousarray(pairs, dtype=np.uint32)
    values = np.empty(len(pairs), dtype=np.float64)
    for start, tables in _tables(columns, pairs, states):
        values[start : start + len(tables)] = _reduce_tables(
            tables, pairs[start : start + len(tables)], counts, len(y), log_base, cfs
        )
    if cfs:
        from .CFS import _entropy

        entropy = np.array([_entropy(column, int(count)) for column, count in zip(columns, counts)])
        denominators = entropy[pairs[:, 0]] + entropy[pairs[:, 1]]
        values = np.divide(2 * values, denominators, out=np.zeros_like(values), where=denominators >= 1e-12)
        values = values.astype(np.float32)
    redundancy = np.zeros((p, p), dtype=values.dtype)
    redundancy[left, right] = redundancy[right, left] = values[p:]
    return values[:p], redundancy
