from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from numba import config, cuda, njit, prange

# Exact-boundary policy for the Relief-family distance thresholds (SURF/SURF*).
# The threshold is a mean of the very distances it is compared with, so a
# distance that equals it mathematically can differ from it by float64
# summation-order noise.  Comparisons stay strict (``<`` near, ``>`` far); the
# limits handed to the kernels are moved outward by this relative amount so a
# mathematical tie is excluded on every backend and in every reduction order.
# The value is ~4 orders of magnitude above float64 rounding noise for the
# feature counts these kernels can process, and far below any separation that
# distinct float32-representable distances can have.
BOUNDARY_RTOL = 1e-12

# Category codes travel through float32 kernels, which represent integers
# exactly only up to 2**24.
MAX_EXACT_FLOAT32_CODE = 2**24


def boundary_limits(threshold: float) -> tuple[float, float]:
    """Return ``(near_limit, far_limit)`` for a strict two-sided comparison.

    ``distance < near_limit`` is the near test and ``distance > far_limit`` the
    far test; a distance within :data:`BOUNDARY_RTOL` of ``threshold`` is a tie
    and belongs to neither.
    """
    tolerance = BOUNDARY_RTOL * abs(threshold)
    return float(threshold - tolerance), float(threshold + tolerance)


def resolve_num_threads(n_jobs) -> int:
    """Translate an ``n_jobs`` argument into a valid Numba thread count.

    ``-1`` means every thread in the configured Numba pool
    (``NUMBA_NUM_THREADS``), not ``NUMBA_DEFAULT_NUM_THREADS``.  Explicit values
    must be integers within ``[1, NUMBA_NUM_THREADS]``.
    """
    max_threads = int(config.NUMBA_NUM_THREADS)
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)):
        raise TypeError(f"n_jobs must be an integer, got {n_jobs!r}.")
    if n_jobs == -1:
        return max_threads
    if not 1 <= n_jobs <= max_threads:
        raise ValueError(
            f"n_jobs must be -1 or between 1 and {max_threads} (the configured Numba thread pool); got {n_jobs}."
        )
    return int(n_jobs)


def _install_windows_context_patch() -> None:
    """Work around a Numba CUDA context lookup ``IndexError`` seen on Windows.

    ``_Runtime._get_or_create_context_uncached`` indexes ``gpus`` with the
    active context's ``devnum``, which some Windows driver stacks report out of
    range.  The workaround replaces that private method, so it is limited to the
    platform that needs it, skipped under the CUDA simulator (which has no such
    runtime), skipped when Numba's private layout is not what it expects, and
    can be disabled with ``FAST_SELECT_DISABLE_CUDA_CONTEXT_PATCH=1``.

    This path is only exercised on Windows with a physical NVIDIA device.
    """
    if sys.platform != "win32" or config.ENABLE_CUDASIM:
        return
    if os.environ.get("FAST_SELECT_DISABLE_CUDA_CONTEXT_PATCH", "") not in ("", "0"):
        return
    try:
        from numba.cuda.cudadrv.devices import _Runtime
    except ImportError:  # pragma: no cover - private layout changed
        return
    if not hasattr(_Runtime, "_get_or_create_context_uncached"):  # pragma: no cover
        return

    def _patched_get_or_create_context_uncached(self, devnum):  # pragma: no cover
        attached = self._get_attached_context()
        if devnum is None:
            devnum = (
                attached.devnum
                if (
                    hasattr(attached, "devnum")
                    and isinstance(attached.devnum, int)
                    and 0 <= attached.devnum < len(self.gpus)
                )
                else 0
            )
        ctx = self.gpus[devnum].get_primary_context()
        try:
            ctx.push()
        except Exception as exc:
            warnings.warn(f"fast_select: pushing the CUDA primary context failed: {exc!r}", RuntimeWarning)
        self._set_attached_context(ctx)
        return ctx

    _Runtime._get_or_create_context_uncached = _patched_get_or_create_context_uncached


_install_windows_context_patch()


def ensure_cuda_context() -> bool:
    """Bind a CUDA context to the calling thread; ``False`` when no usable GPU exists.

    Uses only the public ``numba.cuda`` API, so it behaves the same on real
    devices and under the CUDA simulator.
    """
    try:
        if not cuda.is_available():
            return False
        cuda.current_context()
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


def check_integer_param(value, name: str, minimum: int) -> int:
    """Return ``value`` as ``int``, rejecting bools, non-integral numbers and values below ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)


def effective_discrete_limit(discrete_limit: int, n_samples: int) -> int:
    """Largest distinct-value count a column may have and still be coded as discrete.

    A column can never hold more than ``n_samples`` distinct values, and category
    codes must stay exactly representable in float32, so larger limits are
    equivalent to these bounds.  ``0`` means no feature is discrete.
    """
    limit = check_integer_param(discrete_limit, "discrete_limit", 0)
    return min(limit, n_samples, MAX_EXACT_FLOAT32_CODE)


def discrete_feature_mask(x: np.ndarray, discrete_limit: int) -> np.ndarray:
    """Return the Relief-family discrete-feature mask for ``x``.

    Equivalent to ``[np.unique(x[:, f]).size <= discrete_limit for f in ...]``
    but avoids sorting every column in full.
    """
    limit = effective_discrete_limit(discrete_limit, x.shape[0])
    if limit < 1:
        return np.zeros(x.shape[1], dtype=bool)
    return np.asarray(_discrete_mask_kernel(np.ascontiguousarray(x), limit), dtype=bool)


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


@njit(parallel=True, cache=True, nogil=True)
def _prepare_columns_kernel(x, columns, is_discrete, limit, out):  # pragma: no cover
    """Write the Relief-family working representation of ``x[:, columns]`` into ``out``.

    * Continuous column: ``(v - min) / (max - min)`` evaluated in float64 and only
      then rounded to ``out``'s dtype; a zero-range column becomes all zeros.
      The result is invariant to translating the column, which narrowing raw
      values to float32 first is not.
    * Discrete column: a dense integer code per distinct value (order of first
      appearance).  Only equality is ever tested, so any bijection preserves the
      ``0`` / ``1`` difference exactly, whatever the raw magnitudes were.
    """
    n_samples = x.shape[0]
    for position in prange(columns.shape[0]):
        f = columns[position]
        if is_discrete[f]:
            seen = np.empty(limit, dtype=x.dtype)
            n_seen = 0
            for i in range(n_samples):
                value = x[i, f]
                code = -1
                for s in range(n_seen):
                    if seen[s] == value:
                        code = s
                        break
                if code < 0:
                    seen[n_seen] = value
                    code = n_seen
                    n_seen += 1
                out[i, position] = code
        else:
            low = np.float64(x[0, f])
            high = low
            for i in range(1, n_samples):
                value = np.float64(x[i, f])
                if value < low:
                    low = value
                if value > high:
                    high = value
            span = high - low
            for i in range(n_samples):
                if span > 0.0:
                    out[i, position] = (np.float64(x[i, f]) - low) / span
                else:
                    out[i, position] = 0.0


def prepare_relief_matrix(x, is_discrete, columns=None, *, dtype=np.float32, discrete_limit=None):
    """Return ``x[:, columns]`` in the Relief-family working representation.

    Continuous columns are range-normalised in float64 (see
    :func:`_prepare_columns_kernel`) and discrete columns are replaced by exact
    category codes, then the matrix is narrowed to ``dtype`` in one C-contiguous
    allocation.  CPU and CUDA paths share this matrix, so both score identical
    values.  ``dtype`` is float32 unless the caller needs float64 distances.
    """
    n_samples, n_features = x.shape
    if columns is None:
        columns = np.arange(n_features, dtype=np.int32)
    columns = np.ascontiguousarray(columns, dtype=np.int32)
    limit = max(1, effective_discrete_limit(n_samples if discrete_limit is None else discrete_limit, n_samples))
    out = np.empty((n_samples, columns.size), dtype=dtype)
    _prepare_columns_kernel(
        np.ascontiguousarray(x), columns, np.ascontiguousarray(is_discrete, dtype=np.bool_), limit, out
    )
    return out
