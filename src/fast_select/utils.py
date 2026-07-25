from __future__ import annotations

import numpy as np
from numba import cuda, njit, prange
from numba.cuda.cudadrv.devices import _Runtime, _runtime


# Patch Numba CUDA _Runtime._get_or_create_context_uncached to prevent ac.devnum IndexError on Windows
def _patched_get_or_create_context_uncached(self, devnum):
    ac = self._get_attached_context()
    if devnum is None:
        devnum = (
            ac.devnum
            if (hasattr(ac, "devnum") and isinstance(ac.devnum, int) and 0 <= ac.devnum < len(self.gpus))
            else 0
        )
    ctx = self.gpus[devnum].get_primary_context()
    try:
        ctx.push()
    except Exception:
        pass
    self._set_attached_context(ctx)
    return ctx


_Runtime._get_or_create_context_uncached = _patched_get_or_create_context_uncached


def ensure_cuda_context() -> bool:
    """
    Safely initializes and attaches Numba CUDA device context to current thread.
    Fixes ac.devnum IndexError and invalid context bugs on Windows.
    """
    try:
        if not cuda.is_available() or len(_runtime.gpus) == 0:
            return False
        ctx = _runtime.gpus[0].get_primary_context()
        try:
            ctx.push()
        except Exception:
            pass
        _runtime._set_attached_context(ctx)
        return True
    except Exception:
        return False


def is_cuda_ready() -> bool:
    """
    Checks if CUDA is available and initializes primary context.
    """
    return ensure_cuda_context()


@njit(parallel=True, cache=True, nogil=True)
def _discrete_mask_kernel(x, limit):  # pragma: no cover
    """Per-column test for ``np.unique(column).size <= limit``.

    Equality is tested exactly, as ``np.unique`` does, so the answer is
    identical to the sort-based count.  The scan stops as soon as a column is
    known to exceed ``limit`` distinct values, which is what makes it cheap on
    continuous data.
    """
    n_samples, n_features = x.shape
    out = np.zeros(n_features, dtype=np.bool_)

    for f in prange(n_features):
        seen = np.empty(limit, dtype=x.dtype)
        n_seen = 0
        discrete = True
        for i in range(n_samples):
            value = x[i, f]
            found = False
            for s in range(n_seen):
                if seen[s] == value:
                    found = True
                    break
            if not found:
                if n_seen == limit:
                    discrete = False
                    break
                seen[n_seen] = value
                n_seen += 1
        out[f] = discrete

    return out


def discrete_feature_mask(x: np.ndarray, discrete_limit: int) -> np.ndarray:
    """Return the Relief-family discrete-feature mask for ``x``.

    Equivalent to ``[np.unique(x[:, f]).size <= discrete_limit for f in ...]``
    but avoids sorting every column in full.
    """
    if discrete_limit < 1:
        return np.zeros(x.shape[1], dtype=bool)
    return np.asarray(_discrete_mask_kernel(np.ascontiguousarray(x), discrete_limit), dtype=bool)


def split_discrete_last(is_discrete: np.ndarray, feat_idx: np.ndarray | None = None):
    """Order features as ``[continuous..., discrete...]`` for branch-free kernels.

    Returns ``(columns, n_continuous)`` where ``columns`` indexes into the
    original feature axis.  Relative order inside each block is preserved so the
    per-block summation order is unchanged.
    """
    if feat_idx is None:
        kept = np.arange(is_discrete.shape[0], dtype=np.int32)
    else:
        kept = np.asarray(feat_idx, dtype=np.int32)

    kept_is_discrete = is_discrete[kept]
    columns = np.concatenate((kept[~kept_is_discrete], kept[kept_is_discrete])).astype(np.int32)
    return columns, int(np.count_nonzero(~kept_is_discrete))


def build_kernel_matrix(x, columns, recip_full, n_cont):
    """Return ``x[:, columns]`` as float32 with the continuous block pre-scaled.

    Values are rounded to float32 before being multiplied by the float32
    reciprocal ranges, which is the order the Relief-family kernels have always
    used.

    Both branches produce a C-contiguous result in a single allocation, so peak
    memory is one float32 copy.  ``x[:, columns]`` is deliberately avoided: it
    returns a non-C-contiguous array, and making it contiguous for the kernels
    would double the peak.  A non-float32 input is converted column by column so
    no wide temporary is materialised in the input's own dtype either.
    """
    if x.dtype == np.float32:
        out = np.take(x, columns, axis=1)
    else:
        out = np.empty((x.shape[0], columns.size), dtype=np.float32)
        for position in range(columns.size):
            np.copyto(out[:, position], x[:, columns[position]], casting="unsafe")

    if n_cont:
        out[:, :n_cont] *= recip_full[columns[:n_cont]]
    return out
