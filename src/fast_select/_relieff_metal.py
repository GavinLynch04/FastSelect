"""Experimental bounded-memory Metal ReliefF, following Kononenko (1994).

Complete-data classification only. Use the same float64 preprocessing and
float32 working representation as the CPU/CUDA backends. Distance tiles use
compensated float32 sums; uncertain k-th neighbor boundaries are recomputed on
the CPU with float64 accumulation before GPU scoring. No approximate neighbors,
changed metric, or subsampling. MLX is imported only on explicit use.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from ._metal import metal_entry

QUERY_TILE = 512
CANDIDATE_TILE = 128


@lru_cache(maxsize=1)
def _mlx_and_kernels():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "backend='metal' requires MLX and access to an Apple Silicon GPU; "
            "install fast-select[metal] and run outside a GPU-restricted sandbox."
        ) from exc
    if not mx.metal.is_available():
        raise RuntimeError("backend='metal' requires an available Metal GPU")

    distance = mx.fast.metal_kernel(
        name="relieff_tiled_distance",
        input_names=["x", "offset"],
        output_names=["dist"],
        source=r"""
        const uint n = x_shape[0], p = x_shape[1];
        const uint local_i = threadgroup_position_in_grid.y * 16 + thread_position_in_threadgroup.y;
        const uint j = threadgroup_position_in_grid.x * 16 + thread_position_in_threadgroup.x;
        const uint i = offset[0] + local_i;
        const uint lane = thread_index_in_threadgroup;
        threadgroup float a[16 * 32];
        threadgroup float b[16 * 32];
        float total = 0.0f, correction = 0.0f;
        for (uint base = 0; base < p; base += 32) {
            for (uint t = lane; t < 16 * 32; t += 256) {
                uint row = t / 32, f = base + t % 32;
                uint ai = offset[0] + threadgroup_position_in_grid.y * 16 + row;
                uint bj = threadgroup_position_in_grid.x * 16 + row;
                a[t] = ai < n && f < p ? x[ai * p + f] : 0.0f;
                b[(t % 32) * 16 + row] = bj < n && f < p ? x[bj * p + f] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 partial = float4(0.0f);
            for (uint f = 0; f < 32; f += 4) {
                float4 av = *((threadgroup float4*)(a + thread_position_in_threadgroup.y * 32 + f));
                uint j_local = thread_position_in_threadgroup.x;
                float4 bv = float4(b[f * 16 + j_local], b[(f + 1) * 16 + j_local],
                                   b[(f + 2) * 16 + j_local], b[(f + 3) * 16 + j_local]);
                float4 v = select(float4(av != bv), abs(av - bv),
                                  base + f + uint4(0, 1, 2, 3) < N_CONT);
                partial += v;
            }
            float v = (partial.x + partial.y) + (partial.z + partial.w);
            float adjusted = v - correction;
            float next = total + adjusted;
            correction = (next - total) - adjusted;
            total = next;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (local_i < N_QUERY && j < n) {
            dist[local_i * n + j] = i == j ? INFINITY : total;
        }
        """,
        compile_options={"math_mode": "safe"},
    )
    neighbors = mx.fast.metal_kernel(
        name="relieff_neighbor_chunks",
        input_names=["dist", "labels", "offset"],
        output_names=["local_indices", "local_distances"],
        source=r"""
        uint task = thread_position_in_grid.x;
        if (task >= N_QUERY * N_CLASSES * N_CHUNKS) return;
        uint group = task / N_CHUNKS, chunk = task % N_CHUNKS;
        uint row = group / N_CLASSES, c = group % N_CLASSES;
        uint n = labels_shape[0];
        float best[K + 1];
        int ids[K + 1];
        for (uint t = 0; t <= K; ++t) { best[t] = INFINITY; ids[t] = -1; }
        for (uint j = chunk * CANDIDATE_TILE; j < min(n, (chunk + 1) * CANDIDATE_TILE); ++j) {
            if (labels[j] != int(c) || j == offset[0] + row) continue;
            float d = dist[row * n + j];
            if (d < best[K]) {
                uint pos = K;
                while (pos > 0 && d < best[pos - 1]) {
                    best[pos] = best[pos - 1]; ids[pos] = ids[pos - 1]; --pos;
                }
                best[pos] = d; ids[pos] = int(j);
            }
        }
        for (uint t = 0; t <= K; ++t) {
            local_indices[task * (K + 1) + t] = ids[t];
            local_distances[task * (K + 1) + t] = best[t];
        }
        """,
        compile_options={"math_mode": "safe"},
    )
    merge = mx.fast.metal_kernel(
        name="relieff_merge_neighbors",
        input_names=["local_indices", "local_distances"],
        output_names=["indices", "uncertain"],
        source=r"""
        uint task = thread_position_in_grid.x;
        if (task >= N_QUERY * N_CLASSES) return;
        uint positions[N_CHUNKS];
        for (uint chunk = 0; chunk < N_CHUNKS; ++chunk) positions[chunk] = 0;
        float kth = INFINITY, next = INFINITY;
        int next_id = -1;
        for (uint t = 0; t <= K; ++t) {
            float best = INFINITY;
            int best_id = -1;
            uint winner = 0;
            for (uint chunk = 0; chunk < N_CHUNKS; ++chunk) {
                uint pos = positions[chunk];
                if (pos > K) continue;
                uint address = (task * N_CHUNKS + chunk) * (K + 1) + pos;
                float d = local_distances[address];
                int id = local_indices[address];
                if (id >= 0 && (d < best || (d == best && (best_id < 0 || id < best_id)))) {
                    best = d; best_id = id; winner = chunk;
                }
            }
            if (t < K) indices[task * K + t] = best_id;
            if (t == K - 1) kth = best;
            if (t == K) { next = best; next_id = best_id; }
            ++positions[winner];
        }
        // Eight adds per float4 lane, two final adds, and compensated block
        // accumulation. Widen the error bound and repair close membership.
        float error = 32.0f * 1.1920928955078125e-7f * max(1.0f, next);
        uncertain[task] = N_CONT > 0 && next_id >= 0 && next - kth <= error;
        """,
        compile_options={"math_mode": "safe"},
    )
    score = mx.fast.metal_kernel(
        name="relieff_score",
        input_names=["x", "labels", "priors", "indices", "offset"],
        output_names=["scores"],
        source=r"""
        uint task = thread_position_in_grid.x;
        uint n = x_shape[0], p = x_shape[1];
        if (task >= N_QUERY * p) return;
        uint row = task / p, f = task % p, i = offset[0] + row;
        int own = labels[i];
        float value = x[i * p + f], total = 0.0f;
        for (uint c = 0; c < N_CLASSES; ++c) {
            uint start = (row * N_CLASSES + c) * K;
            uint found = 0;
            float diffs = 0.0f;
            for (uint t = 0; t < K; ++t) {
                int j = indices[start + t];
                if (j < 0) break;
                float other = x[uint(j) * p + f];
                diffs += f < N_CONT ? abs(value - other) : float(value != other);
                ++found;
            }
            if (found) {
                float weight = int(c) == own ? -1.0f : priors[c] / (1.0f - priors[own]);
                total += weight * diffs / (float(n) * float(found));
            }
        }
        scores[task] = total;
        """,
        compile_options={"math_mode": "safe"},
    )
    return mx, distance, neighbors, merge, score


@njit(cache=True, nogil=True)
def _repair_neighbors(x, labels, n_cont, k, offset, uncertain, indices):
    """Resolve ambiguous membership with double accumulation and index ties."""
    n, p = x.shape
    n_classes = uncertain.shape[1]
    for row in range(uncertain.shape[0]):
        i = offset + row
        for c in range(n_classes):
            if not uncertain[row, c]:
                continue
            best = np.full(k, np.inf, dtype=np.float64)
            ids = np.full(k, -1, dtype=np.int32)
            for j in range(n):
                if j == i or labels[j] != c:
                    continue
                d = 0.0
                for f in range(n_cont):
                    d += abs(x[i, f] - x[j, f])
                for f in range(n_cont, p):
                    d += x[i, f] != x[j, f]
                if d < best[k - 1]:
                    pos = k - 1
                    while pos > 0 and d < best[pos - 1]:
                        best[pos] = best[pos - 1]
                        ids[pos] = ids[pos - 1]
                        pos -= 1
                    best[pos], ids[pos] = d, j
            indices[row, c] = ids


@metal_entry
def relieff_metal(x, labels, n_cont, k, priors):
    """Return synchronized host scores and observable prototype diagnostics."""
    if k > 64:
        raise ValueError("Experimental Metal ReliefF supports n_neighbors <= 64; use backend='cpu' for larger k")
    mx, distance, neighbors, merge, score = _mlx_and_kernels()
    n, p = x.shape
    n_classes = len(priors)
    chunks = (n + CANDIDATE_TILE - 1) // CANDIDATE_TILE
    x_gpu, labels_gpu, priors_gpu = mx.array(x), mx.array(labels), mx.array(priors)
    tile_scores = []
    repaired_groups = 0
    for offset in range(0, n, QUERY_TILE):
        rows = min(QUERY_TILE, n - offset)
        start = mx.array([offset], dtype=mx.uint32)
        template = [
            ("N_CONT", n_cont),
            ("N_QUERY", rows),
            ("N_CLASSES", n_classes),
            ("K", k),
            ("N_CHUNKS", chunks),
            ("CANDIDATE_TILE", CANDIDATE_TILE),
        ]
        distances = distance(
            inputs=[x_gpu, start],
            template=template,
            grid=(((n + 15) // 16) * 16, ((rows + 15) // 16) * 16, 1),
            threadgroup=(16, 16, 1),
            output_shapes=[(rows, n)],
            output_dtypes=[mx.float32],
        )[0]
        local_indices, local_distances = neighbors(
            inputs=[distances, labels_gpu, start],
            template=template,
            grid=(rows * n_classes * chunks, 1, 1),
            threadgroup=(min(128, rows * n_classes * chunks), 1, 1),
            output_shapes=[(rows, n_classes, chunks, k + 1)] * 2,
            output_dtypes=[mx.int32, mx.float32],
        )
        indices, uncertain = merge(
            inputs=[local_indices, local_distances],
            template=template,
            grid=(rows * n_classes, 1, 1),
            threadgroup=(min(128, rows * n_classes), 1, 1),
            output_shapes=[(rows, n_classes, k), (rows, n_classes)],
            output_dtypes=[mx.int32, mx.bool_],
        )
        flags = np.array(uncertain)
        if flags.any():
            host_indices = np.array(indices)
            _repair_neighbors(x, labels, n_cont, k, offset, flags, host_indices)
            indices = mx.array(host_indices)
            repaired_groups += int(flags.sum())
        contributions = score(
            inputs=[x_gpu, labels_gpu, priors_gpu, indices, start],
            template=template,
            grid=(rows * p, 1, 1),
            threadgroup=(min(256, rows * p), 1, 1),
            output_shapes=[(rows, p)],
            output_dtypes=[mx.float32],
        )[0]
        totals = mx.sum(contributions, axis=0)
        mx.eval(totals)
        tile_scores.append(totals)
    result = np.array(mx.sum(mx.stack(tile_scores), axis=0))
    return result, {
        "query_tile": QUERY_TILE,
        "distance_buffer_bytes": min(n, QUERY_TILE) * n * 4,
        "repaired_neighbor_groups": repaired_groups,
        "neighbor_groups": n * n_classes,
        "device": mx.device_info()["device_name"],
    }
