"""Pearson count-feature class sums with compensated split precision on Metal.

Columns are scaled in float64, then represented by high/low float32 parts.
GPU two-sum accumulators return both parts for a final float64 host reduction;
statistics and SciPy survival probabilities retain the existing CPU formula.
"""

from functools import lru_cache

import numpy as np

from ._metal import get_mx, metal_entry


@lru_cache(maxsize=1)
def _kernel():
    return get_mx().fast.metal_kernel(
        name="chi2_split_class_sums",
        input_names=["high", "low", "labels"],
        output_names=["parts"],
        source=r"""
        uint task = thread_position_in_grid.x;
        uint n = high_shape[0], p = high_shape[1];
        if (task >= BLOCKS * CLASSES * p) return;
        uint f = task % p, c = (task / p) % CLASSES, block = task / (p*CLASSES);
        float sum = 0.0f, residual = 0.0f;
        for (uint i = block*256; i < min(n, (block+1)*256); ++i) {
            if (labels[i] != int(c)) continue;
            float value = high[i*p+f];
            float next = sum + value;
            float v = next - sum;
            float error = (sum - (next-v)) + (value-v);
            residual += error + low[i*p+f];
            float combined = next + residual;
            residual = residual - (combined-next);
            sum = combined;
        }
        parts[task*2] = sum;
        parts[task*2+1] = residual;
        """,
        compile_options={"math_mode": "safe"},
    )


@metal_entry
def chi2_counts_metal(X, labels, classes):
    mx = get_mx()
    scale = np.max(X, axis=0).astype(np.float64)
    scale[scale == 0] = 1
    normalized = X.astype(np.float64) / scale
    high = normalized.astype(np.float32)
    low = (normalized - high).astype(np.float32)
    blocks = (len(X) + 255) // 256
    shape = (blocks, classes, X.shape[1], 2)
    values = _kernel()(
        inputs=[mx.array(high), mx.array(low), mx.array(labels.astype(np.int32))],
        template=[("BLOCKS", blocks), ("CLASSES", classes)],
        grid=(blocks * classes * X.shape[1], 1, 1),
        threadgroup=(min(256, blocks * classes * X.shape[1]), 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]
    parts = np.array(values).astype(np.float64)
    observed = np.sum(parts[..., 0] + parts[..., 1], axis=0) * scale
    return observed, np.sum(observed, axis=0)
