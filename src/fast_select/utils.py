from __future__ import annotations
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devices import _runtime, _Runtime

# Patch Numba CUDA _Runtime._get_or_create_context_uncached to prevent ac.devnum IndexError on Windows
def _patched_get_or_create_context_uncached(self, devnum):
    ac = self._get_attached_context()
    if devnum is None:
        devnum = ac.devnum if (hasattr(ac, 'devnum') and isinstance(ac.devnum, int) and 0 <= ac.devnum < len(self.gpus)) else 0
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
