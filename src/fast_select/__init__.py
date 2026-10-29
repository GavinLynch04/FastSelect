"""fast-select: Numba, CUDA and optional Metal feature selection.

Public API
----------
Estimators
    :class:`ReliefF`, :class:`SURF`, :class:`MultiSURF`, :class:`TuRF`,
    :class:`mRMR`, :class:`CFS`, :class:`MDR`, :class:`JMI`, :class:`CMIM`,
    :class:`FCBF`, :class:`LaplacianScore`, :class:`GiniIndex`, :class:`DISR`,
    :class:`CIFE`, :class:`MIFS`
Functions
    :func:`chi2`, :func:`calculate_mi_single_pair`, :func:`calculate_mi_matrices`,
    :func:`mutual_info_classif`, :func:`mutual_info_regression`
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .CFS import CFS
from .Chi2 import chi2
from .ContinuousMI import mutual_info_classif
from .GiniIndex import GiniIndex
from .InformationSelectors import CMIM, FCBF, JMI
from .JointInformation import CIFE, DISR, MIFS
from .LaplacianScore import LaplacianScore
from .MDR import MDR
from .mRMR import mRMR
from .MultiSURF import MultiSURF
from .mutual_information import calculate_mi_matrices, calculate_mi_single_pair
from .RegressionMI import mutual_info_regression
from .ReliefF import ReliefF
from .RReliefF import RReliefF
from .SURF import SURF
from .TuRF import TuRF

try:
    __version__ = _version("fast-select")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "CFS",
    "CIFE",
    "CMIM",
    "FCBF",
    "DISR",
    "GiniIndex",
    "JMI",
    "LaplacianScore",
    "MDR",
    "MIFS",
    "MultiSURF",
    "ReliefF",
    "RReliefF",
    "SURF",
    "TuRF",
    "__version__",
    "calculate_mi_matrices",
    "calculate_mi_single_pair",
    "chi2",
    "mRMR",
    "mutual_info_classif",
    "mutual_info_regression",
]
