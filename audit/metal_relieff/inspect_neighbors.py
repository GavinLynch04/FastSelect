import inspect
import importlib
import time

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads

from fast_select import ReliefF
from fast_select.utils import discrete_feature_mask, split_discrete_last, prepare_relief_matrix
import fast_select._relieff_metal as metal

module = importlib.import_module("fast_select.ReliefF")
source = inspect.getsource(module._relieff_cpu_kernel.py_func)
source = source[source.index("def ") :]
source = source.replace("def _relieff_cpu_kernel(", "def cpu_neighbors(")
source = source.replace(
    "    for i in prange(n_samples):",
    "    all_ids = np.full((n_samples, n_classes, k), -1, dtype=np.int32)\n    for i in prange(n_samples):",
)
source = source.replace(
    "        h_found = 0", "        all_ids[i] = miss_idx\n        all_ids[i, lbl_i] = hit_idx\n        h_found = 0"
)
source += "\n    return all_ids\n"
exec(source)
cpu_neighbors = njit(parallel=True, fastmath=True)(cpu_neighbors)
set_num_threads(8)
rng = np.random.default_rng(20261026)
n, p = 2000, 512
x = rng.standard_normal((n, p), dtype=np.float32)
x[:, ::2] = rng.integers(0, 3, (n, p // 2))
y = np.arange(n, dtype=np.int32) % 4
rng.shuffle(y)
disc = discrete_feature_mask(x, 3)
cols, ncont = split_discrete_last(disc)
prepared = prepare_relief_matrix(x, disc, cols, dtype=np.float32, discrete_limit=3)
priors = np.bincount(y).astype(np.float32) / n
cpu_scores = np.zeros(p, dtype=np.float32)
cpu_ids = cpu_neighbors(prepared, y, ncont, 10, priors, cpu_scores)
mx, *kernels = metal._mlx_and_kernels()
score = kernels[-1]
captured = []


def capture(*args, **kwargs):
    captured.append(np.array(kwargs["inputs"][3]))
    return score(*args, **kwargs)


kernels[-1] = capture
metal._mlx_and_kernels = lambda: (mx, *kernels)
metal_scores, diag = metal.relieff_metal(prepared, y, ncont, 10, priors)
gpu_ids = np.concatenate(captured)
changed = np.any(np.sort(cpu_ids, axis=-1) != np.sort(gpu_ids, axis=-1), axis=-1)
print("diagnostics", diag, "max_score_error", np.max(abs(cpu_scores - metal_scores)))
print("changed groups", np.argwhere(changed).tolist())
for i, c in np.argwhere(changed):
    cids = cpu_ids[i, c]
    gids = gpu_ids[i, c]
    ids = np.unique(np.concatenate([cids, gids]))
    print("row,class", i, c, "cpu", cids, "metal", gids)
    for j in ids:
        terms = np.abs(prepared[i, :ncont] - prepared[j, :ncont])
        dist = terms.astype(np.float64).sum() + np.count_nonzero(prepared[i, ncont:] != prepared[j, ncont:])
        print(" candidate", j, "distance", repr(dist), "float32", repr(np.float32(dist)))
print("scores sorted error", np.sort(abs(cpu_scores - metal_scores))[-10:])
