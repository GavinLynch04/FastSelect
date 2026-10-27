"""MDR exact integer training-cell counts and risk comparisons on Metal.

Ritchie et al. (2001), with the package's documented balanced-accuracy/CVC
variant. CV, lexicographic search ordering and final lookup construction stay
on the host. No float32 risk-ratio rounding may change cell classification.
"""

from functools import lru_cache

import numpy as np

from ._metal import get_mx, metal_entry


@lru_cache(maxsize=1)
def _kernel():
    return get_mx().fast.metal_kernel(
        name="mdr_integer_counts",
        input_names=["x", "labels", "combinations"],
        output_names=["correct"],
        source=r"""
        uint model = threadgroup_position_in_grid.x, lane = thread_index_in_threadgroup;
        uint n = x_shape[0], p = x_shape[1];
        threadgroup atomic_uint cases[CELLS];
        threadgroup atomic_uint controls[CELLS];
        for (uint c = lane; c < CELLS; c += 64) {
            atomic_store_explicit(&cases[c], 0u, memory_order_relaxed);
            atomic_store_explicit(&controls[c], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < n; i += 64) {
            uint cell = 0;
            for (uint f = 0; f < K; ++f)
                cell = cell * 3 + uint(x[i*p+combinations[model*K+f]]);
            if (labels[i]) atomic_fetch_add_explicit(&cases[cell], 1u, memory_order_relaxed);
            else atomic_fetch_add_explicit(&controls[cell], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            uint total_case = 0;
            for (uint c = 0; c < CELLS; ++c)
                total_case += atomic_load_explicit(&cases[c], memory_order_relaxed);
            uint total_control = n - total_case, tp = 0, tn = 0;
            for (uint c = 0; c < CELLS; ++c) {
                uint a = atomic_load_explicit(&cases[c], memory_order_relaxed);
                uint b = atomic_load_explicit(&controls[c], memory_order_relaxed);
                bool high = (a + b > 0) && ulong(a)*ulong(total_control) >= ulong(b)*ulong(total_case);
                if (high) tp += a; else tn += b;
            }
            correct[model*2] = tp;
            correct[model*2+1] = tn;
        }
        """,
        compile_options={"math_mode": "safe"},
    )


@metal_entry
def make_mdr_scorer(X, y, k):
    """Upload one fold once; score bounded, lexicographic combination batches."""
    mx = get_mx()
    x_gpu, y_gpu = mx.array(X), mx.array(y)
    cases = int(np.count_nonzero(y))
    controls = len(y) - cases

    def score(combinations):
        if cases == 0 or controls == 0:
            return np.zeros(len(combinations), dtype=np.float32)
        correct = _kernel()(
            inputs=[x_gpu, y_gpu, mx.array(np.ascontiguousarray(combinations, dtype=np.uint32))],
            template=[("K", k), ("CELLS", 3**k)],
            grid=(len(combinations) * 64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(len(combinations), 2)],
            output_dtypes=[mx.uint32],
            stream=mx.gpu,
        )[0]
        counts = np.array(correct).astype(np.float64)
        return ((counts[:, 0] / cases + counts[:, 1] / controls) / 2).astype(np.float32)

    return score
