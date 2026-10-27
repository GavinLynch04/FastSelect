"""SURF/SURF* and MultiSURF/MultiSURF* with precision-certified membership.

Moore et al. / Urbanowicz et al. (2018): strict global/per-target radii,
separate hit/miss groups, and similarity (not difference) in MultiSURF* far
updates. CPU float64 resolves uncertain predicates; Metal scores fixed groups.
"""

from functools import lru_cache

import numpy as np
from numba import njit, prange

from ._metal import get_mx, metal_entry

QUERY_TILE = 256


@njit(cache=True, parallel=True)
def _weights(x, labels, n_cont, offset, distances, multisurf, star, global_near, global_far):
    rows, n = distances.shape
    weights = np.zeros((rows, n, 2), dtype=np.float32)
    repairs = np.zeros(rows, dtype=np.int64)
    for row in prange(rows):
        i = offset + row
        d = distances[row].astype(np.float64)
        d[i] = 0.0
        bound = 0.0 if n_cont == 0 else 64 * np.finfo(np.float32).eps * max(1.0, d.max())
        bound += 4 * np.finfo(np.float32).eps * n_cont
        near, far = global_near, global_far
        if multisurf:
            mean = np.sum(d) / (n - 1)
            variance = max(0.0, np.sum(d * d) / (n - 1) - mean * mean)
            near, far = mean - np.sqrt(variance) / 2, mean + np.sqrt(variance) / 2
            uncertain = False
            for j in range(n):
                if j != i and (abs(d[j] - near) <= 8 * bound or (star and abs(d[j] - far) <= 8 * bound)):
                    uncertain = True
            if uncertain:
                total = 0.0
                total2 = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    value = 0.0
                    for f in range(n_cont):
                        value += abs(x[i, f] - x[j, f])
                    for f in range(n_cont, x.shape[1]):
                        value += x[i, f] != x[j, f]
                    total += value
                    total2 += value * value
                    # Match the shared float32 MultiSURF working-distance field.
                    d[j] = np.float32(value)
                mean = total / (n - 1)
                variance = max(0.0, total2 / (n - 1) - mean * mean)
                near, far = mean - np.sqrt(variance) / 2, mean + np.sqrt(variance) / 2
                repairs[row] = n - 1
        else:
            for j in range(n):
                if j != i and (abs(d[j] - near) <= bound or (star and abs(d[j] - far) <= bound)):
                    value = 0.0
                    for f in range(n_cont):
                        value += abs(x[i, f] - x[j, f])
                    for f in range(n_cont, x.shape[1]):
                        value += x[i, f] != x[j, f]
                    d[j] = value
                    repairs[row] += 1
        counts = np.zeros(4, dtype=np.int64)
        groups = np.full(n, -1, dtype=np.int32)
        for j in range(n):
            if j == i:
                continue
            miss = int(labels[j] != labels[i])
            if d[j] < near:
                groups[j] = miss
                counts[miss] += 1
            elif star and d[j] > far:
                groups[j] = 2 + miss
                counts[2 + miss] += 1
        for j in range(n):
            group = groups[j]
            if group >= 0:
                sign = -1.0 if group % 2 == 0 else 1.0
                if group >= 2 and not multisurf:
                    sign = -sign
                channel = 1 if group >= 2 and multisurf else 0
                weights[row, j, channel] = sign / (n * counts[group])
    return weights, repairs


@lru_cache(maxsize=1)
def _score_kernel():
    return get_mx().fast.metal_kernel(
        name="threshold_neighbor_tiled_scores",
        input_names=["high", "low", "weights", "offset"],
        output_names=["partial"],
        source=r"""
        uint lane = thread_index_in_threadgroup;
        uint row_local = lane / 32, feature_local = lane % 32;
        uint row = threadgroup_position_in_grid.y * 8 + row_local;
        uint chunk = threadgroup_position_in_grid.z;
        uint f = threadgroup_position_in_grid.x * 32 + feature_local;
        uint n = high_shape[0], p = high_shape[1], i = offset[0] + row;
        threadgroup float bh[32*32], bl[32*32];
        threadgroup float wd[8*32], ws[8*32];
        for (uint t = lane; t < 32*32; t += 256) {
            uint j = chunk*32 + t/32;
            uint g = threadgroup_position_in_grid.x*32 + t%32;
            bh[t] = j < n && g < p ? high[j*p+g] : 0.0f;
            bl[t] = j < n && g < p ? low[j*p+g] : 0.0f;
        }
        uint j = chunk*32 + feature_local;
        wd[lane] = row < N_QUERY && j < n ? weights[(row*n+j)*2] : 0.0f;
        ws[lane] = row < N_QUERY && j < n ? weights[(row*n+j)*2+1] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < N_QUERY && f < p) {
            float ah = high[i*p+f], al = low[i*p+f], total = 0.0f;
            for (uint c = 0; c < min(32u, n-chunk*32); ++c) {
                float b = bh[c*32+feature_local], l = bl[c*32+feature_local];
                float difference;
                if (f < N_CONT) {
                    float d = ah-b, v = d-ah;
                    float error = (ah-(d-v))-(b+v);
                    difference = abs(d+(error+al-l));
                } else difference = float(ah != b);
                total += wd[row_local*32+c]*difference + ws[row_local*32+c]*(1.0f-difference);
            }
            partial[(row*CHUNKS+chunk)*p+f] = total;
        }
        """,
        compile_options={"math_mode": "safe"},
    )


@metal_entry
def threshold_neighbors_metal(x, labels, n_cont, star, *, multisurf=False, near=0.0, far=0.0):
    from ._relieff_metal import _mlx_and_kernels

    mx = get_mx()
    distance = _mlx_and_kernels()[1]
    n, p = x.shape
    high = x.astype(np.float32)
    low = (x.astype(np.float64) - high).astype(np.float32)
    high_gpu, low_gpu = mx.array(high), mx.array(low)
    chunks = (n + 31) // 32
    query_tile = max(1, min(QUERY_TILE, 64 * 1024**2 // (chunks * p * 4)))
    scores = []
    repaired = 0
    for offset in range(0, n, query_tile):
        rows = min(query_tile, n - offset)
        start = mx.array([offset], dtype=mx.uint32)
        dist = distance(
            inputs=[high_gpu, start],
            template=[("N_CONT", n_cont), ("N_QUERY", rows)],
            grid=(((n + 15) // 16) * 16, ((rows + 15) // 16) * 16, 1),
            threadgroup=(16, 16, 1),
            output_shapes=[(rows, n)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )[0]
        weights, repairs = _weights(x, labels, n_cont, offset, np.array(dist), multisurf, star, near, far)
        repaired += int(repairs.sum())
        partial = _score_kernel()(
            inputs=[high_gpu, low_gpu, mx.array(weights), start],
            template=[("N_CONT", n_cont), ("N_QUERY", rows), ("CHUNKS", chunks)],
            grid=(((p + 31) // 32) * 256, (rows + 7) // 8, chunks),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, chunks, p)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )[0]
        total = mx.sum(partial, axis=(0, 1))
        mx.eval(total)
        scores.append(total)
    return np.array(mx.sum(mx.stack(scores), axis=0)), {
        "recomputed_pair_distances": repaired,
        "distance_buffer_bytes": min(n, query_tile) * n * 4,
        "query_tile": query_tile,
        "device": mx.device_info()["device_name"],
    }
