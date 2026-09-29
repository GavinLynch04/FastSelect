"""fast-select: Numba- and CUDA-accelerated feature selection.

Public API
----------
Estimators
    :class:`ReliefF`, :class:`SURF`, :class:`MultiSURF`, :class:`TuRF`,
    :class:`mRMR`, :class:`CFS`, :class:`MDR`
Functions
    :func:`chi2`, :func:`calculate_mi_single_pair`, :func:`calculate_mi_matrices`
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .CFS import CFS
from .Chi2 import chi2
from .MDR import MDR
from .mRMR import mRMR
from .MultiSURF import MultiSURF
from .mutual_information import calculate_mi_matrices, calculate_mi_single_pair
from .ReliefF import ReliefF
from .SURF import SURF
from .TuRF import TuRF

try:
    __version__ = _version("fast-select")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "CFS",
    "MDR",
    "MultiSURF",
    "ReliefF",
    "SURF",
    "TuRF",
    "__version__",
    "calculate_mi_matrices",
    "calculate_mi_single_pair",
    "chi2",
    "mRMR",
]
