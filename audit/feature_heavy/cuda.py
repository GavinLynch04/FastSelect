"""Research CUDA kernels; simulator-tested logic, no hardware timing claim.

Run only on an NVIDIA device or under NUMBA_ENABLE_CUDASIM=1. Float64 KSG
distances avoid Metal's proposal repair. Shared-memory integer sorting and
histograms preserve exact input ordering and first-occurrence entropy order.
"""

import math

import numpy as np
from numba import config, cuda, float64, uint32


@cuda.jit
def _sort(high, low, orders, pad):
    f, lane = cuda.blockIdx.x, cuda.threadIdx.x
    n = high.shape[1]
    kh = cuda.shared.array(1024, uint32)
    kl = cuda.shared.array(1024, uint32)
    ids = cuda.shared.array(1024, uint32)
    for i in range(lane, pad, cuda.blockDim.x):
        kh[i] = high[f, i] if i < n else 0xFFFFFFFF
        kl[i] = low[f, i] if i < n else 0xFFFFFFFF
        ids[i] = i if i < n else 0xFFFFFFFF
    cuda.syncthreads()
    width = 2
    while width <= pad:
        gap = width // 2
        while gap:
            for i in range(lane, pad, cuda.blockDim.x):
                j = i ^ gap
                if j > i:
                    greater = kh[i] > kh[j] or (
                        kh[i] == kh[j] and (kl[i] > kl[j] or (kl[i] == kl[j] and ids[i] > ids[j]))
                    )
                    less = kh[i] < kh[j] or (kh[i] == kh[j] and (kl[i] < kl[j] or (kl[i] == kl[j] and ids[i] < ids[j])))
                    swap = greater if (i & width) == 0 else less
                    if swap:
                        old_high, old_low, index = kh[i], kl[i], ids[i]
                        kh[i], kl[i], ids[i] = kh[j], kl[j], ids[j]
                        kh[j], kl[j], ids[j] = old_high, old_low, index
            cuda.syncthreads()
            gap //= 2
        width *= 2
    for i in range(lane, n, cuda.blockDim.x):
        orders[f, i] = ids[i]


@cuda.jit
def _histograms(columns, target, active, selected, s, sy, conditional, pairs, triples):
    f, lane = cuda.blockIdx.x, cuda.threadIdx.x
    n, t = columns.shape[1], s * s * sy
    counts = cuda.shared.array(512, uint32)
    first = cuda.shared.array(512, uint32)
    pcounts = cuda.shared.array(64, uint32)
    pfirst = cuda.shared.array(64, uint32)
    for i in range(lane, t, cuda.blockDim.x):
        counts[i], first[i], triples[f, i] = 0, n, 0
    for i in range(lane, s * s, cuda.blockDim.x):
        pcounts[i], pfirst[i], pairs[f, i] = 0, n, 0
    cuda.syncthreads()
    if not active[f]:
        return
    for i in range(lane, n, cuda.blockDim.x):
        key = columns[selected, i] * s + columns[f, i]
        cuda.atomic.add(pcounts, key, 1)
        cuda.atomic.min(pfirst, key, i)
        if conditional:
            key = key * sy + target[i]
            cuda.atomic.add(counts, key, 1)
            cuda.atomic.min(first, key, i)
    cuda.syncthreads()
    for key in range(lane, s * s, cuda.blockDim.x):
        seen = pfirst[key]
        if seen < n:
            rank = 0
            for j in range(s * s):
                if pfirst[j] < seen:
                    rank += 1
            pairs[f, rank] = pcounts[key]
    if conditional:
        for key in range(lane, t, cuda.blockDim.x):
            seen = first[key]
            if seen < n:
                rank = 0
                for j in range(t):
                    if first[j] < seen:
                        rank += 1
                triples[f, rank] = counts[key]


@cuda.jit
def _joint_distances(columns, target, k, radii):
    task = cuda.grid(1)
    n = columns.shape[1]
    if task >= columns.shape[0] * n:
        return
    f, i = task // n, task % n
    nearest = cuda.local.array(32, float64)
    for j in range(k):
        nearest[j] = math.inf
    for j in range(n):
        if i != j:
            d = max(abs(columns[f, i] - columns[f, j]), abs(target[i] - target[j]))
            if d < nearest[0]:
                position = 0
                while position + 1 < k and nearest[position + 1] > d:
                    nearest[position] = nearest[position + 1]
                    position += 1
                nearest[position] = d
    radii[f, i] = nearest[0]


def _check():
    if not cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA device is required; simulator validates logic only")


def _lanes(default):
    return min(default, 16) if config.ENABLE_CUDASIM else default


def sort(high, low):
    _check()
    p, n = high.shape
    pad = 1 << (n - 1).bit_length()
    if pad > 1024:
        raise ValueError("Experimental sort supports up to 1024 samples")
    output = cuda.device_array((p, n), np.uint32)
    _sort[p, _lanes(min(128, pad))](cuda.to_device(high), cuda.to_device(low), output, pad)
    return output.copy_to_host()


class Histograms:
    def __init__(self, columns, target, counts, sy):
        _check()
        self.s, self.sy = int(counts.max()), int(sy)
        if self.s > 8 or self.sy > 8:
            raise ValueError("Experimental histogram supports eight states/classes")
        self.p, self.n = columns.shape
        self.columns, self.target = cuda.to_device(columns), cuda.to_device(target)
        self.pairs = cuda.device_array((self.p, self.s * self.s), np.uint32)
        self.triples = cuda.device_array((self.p, self.s * self.s * self.sy), np.uint32)

    def step(self, selected, active, conditional):
        _histograms[self.p, _lanes(128)](
            self.columns,
            self.target,
            cuda.to_device(active),
            selected,
            self.s,
            self.sy,
            conditional,
            self.pairs,
            self.triples,
        )
        triple = self.triples.copy_to_host() if conditional else np.empty((self.p, 0), dtype=np.uint32)
        return self.pairs.copy_to_host(), triple


def histograms(columns, target, counts, sy):
    return Histograms(columns, target, counts, sy)


def radii(columns, target, k):
    _check()
    if k > 32:
        raise ValueError("Experimental CUDA neighbor kernel supports k<=32")
    p, n = columns.shape
    output = cuda.device_array((p, n), np.float64)
    lanes = _lanes(128)
    _joint_distances[(p * n + lanes - 1) // lanes, lanes](cuda.to_device(columns), cuda.to_device(target), k, output)
    return np.nextafter(output.copy_to_host(), 0.0), {"float64_native_distances": True}
