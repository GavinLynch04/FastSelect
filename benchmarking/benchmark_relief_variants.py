"""Newer Relief variants: FastSelect CPU against scikit-rebate 0.8.4, runtime and memory.

``--family swrf`` compares the six SWRF variants and ``--family murelief``
compares μ-Relief.

Runtime follows ``benchmarking.py``: every series (library, variant) runs in its
own fresh process with eight Numba threads and single-threaded BLAS. FastSelect
warms every shape (JIT compilation) and scikit-rebate warms each worker pool
once; each point is then the median of repeated fits totalling at least
``--min-seconds`` (at least three fits unless one fit already takes that long).
The whole set runs ``--rounds`` times in alternating order, so every case is
measured for ``rounds * min_seconds`` cumulative seconds or more (default 6 s).
scikit-rebate is timed with ``n_jobs=-1`` and ``n_jobs=1`` and the faster
median is reported; ``n_jobs=1`` is skipped once a single fit takes longer than
``SERIAL_LIMIT_SECONDS``.

Memory runs each (library, variant, shape) in a further fresh process. After
the same 30-row warm-up fit for both libraries (which compiles FastSelect's
kernels, since Numba specialises on dtype and layout rather than shape) the
parent samples the worker's macOS physical footprint (``proc_pid_rusage``)
about every 0.5 ms while the worker repeats the fit for at least 0.5 s; the
reported figure is the peak footprint above the footprint just before the fits.
Sampling from outside the process sees Numba kernels that hold the GIL.
scikit-rebate runs with ``n_jobs=1`` here so that all of its work is inside the
sampled process. A separate fit records the ``tracemalloc`` peak, which covers
Python/NumPy host allocations only (not Numba's internal buffers).

The SWRF scores differ between the libraries by design: scikit-rebate scores
distant instances by ``1 - diff`` and normalises hits and misses by their
counts, while FastSelect follows the papers (``src/fast_select/SWRF.py``). For
μ-Relief the scores should agree on this continuous binary data, and the run
records the largest score difference for every shape.

    python benchmarking/benchmark_relief_variants.py --family swrf
    python benchmarking/benchmark_relief_variants.py --family swrf --report
    python benchmarking/benchmark_relief_variants.py --family murelief
    python benchmarking/benchmark_relief_variants.py --family murelief --report

macOS only (the memory sampler uses libproc). Inputs are seeded
``make_classification`` data.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import select
import statistics
import subprocess
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarking" / "results"
MURELIEF_NEIGHBORS = 10
RELIEFSEQ_MAX_K = 10
VLS_SUBSET_SIZE = 100
RELIEF_SCENARIOS = {
    "samples": {"fixed": ("n_features", 100), "sizes": [100, 200, 500, 1000, 2000]},
    "features": {"fixed": ("n_samples", 100), "sizes": [100, 1000, 5000, 10000]},
}
# Every family maps variant -> {label: spec}; the "fast-select" label is the
# measured FastSelect estimator and every other label is a comparison.
# Specs: ("fast" | "rebate", class name, keywords); ("fast-sweep" | "rebate-sweep",
# k_max, ReliefF keywords) fits ReliefF once per k and keeps each feature's best;
# ("fast-vls" | "rebate-vls", VLS keywords, ReliefF keywords).
FAMILIES = {
    "swrf": {
        "title": "SWRF family",
        "output": RESULTS / "swrf_vs_skrebate.json",
        "report": RESULTS / "SWRF_FAMILY.md",
        "parity": False,
        "scenarios": RELIEF_SCENARIOS,
        "note": (
            "The libraries compute different scores by design (see `src/fast_select/SWRF.py`); this compares "
            "speed and memory, not output."
        ),
        "variants": {
            name: {"fast-select": ("fast", fast, keywords), "scikit-rebate": ("rebate", rebate, {})}
            for name, fast, keywords, rebate in (
                ("SWRF", "SWRF", {}, "SWRF"),
                ("SWRF*", "SWRF", {"use_star": True}, "SWRFstar"),
                ("MultiSWRF", "MultiSWRF", {}, "MultiSWRF"),
                ("MultiSWRF*", "MultiSWRF", {"use_star": True}, "MultiSWRFstar"),
                ("MultiSWRFDB", "MultiSWRF", {"dead_band": True}, "MultiSWRFDB"),
                ("MultiSWRFDB*", "MultiSWRF", {"use_star": True, "dead_band": True}, "MultiSWRFDBstar"),
            )
        },
    },
    "murelief": {
        "title": "μ-Relief",
        "output": RESULTS / "murelief_vs_skrebate.json",
        "report": RESULTS / "MURELIEF.md",
        "parity": True,
        "scenarios": RELIEF_SCENARIOS,
        "note": (
            f"Both libraries use `n_neighbors={MURELIEF_NEIGHBORS}`. On this continuous binary data both use the "
            "range-normalised difference and ReliefF's update, so their scores should agree; the parity table "
            "records the largest difference."
        ),
        "variants": {
            "MuRelief": {
                "fast-select": ("fast", "MuRelief", {"n_neighbors": MURELIEF_NEIGHBORS}),
                "scikit-rebate": ("rebate", "MuRelief", {"n_neighbors": MURELIEF_NEIGHBORS}),
            },
        },
    },
    "reliefseq": {
        "title": "ReliefSeq",
        "output": RESULTS / "reliefseq_comparison.json",
        "report": RESULTS / "RELIEFSEQ.md",
        "parity": False,
        "scenarios": {
            "samples": {"fixed": ("n_features", 100), "sizes": [100, 200, 500, 1000]},
            "features": {"fixed": ("n_samples", 100), "sizes": [100, 1000, 5000]},
        },
        "note": (
            f"`ReliefSeq(max_neighbors={RELIEFSEQ_MAX_K})` against what ReliefSeq replaces: a separate ReliefF fit "
            f"for every k = 1..{RELIEFSEQ_MAX_K}, keeping each feature's best score, with FastSelect's and with "
            "scikit-rebate's ReliefF. scikit-rebate has no ReliefSeq, and the authors' C++ tool was not run."
        ),
        "variants": {
            f"ReliefSeq k <= {RELIEFSEQ_MAX_K}": {
                "fast-select": ("fast", "ReliefSeq", {"max_neighbors": RELIEFSEQ_MAX_K}),
                "fast-select-sweep": ("fast-sweep", RELIEFSEQ_MAX_K, {}),
                "scikit-rebate-sweep": ("rebate-sweep", RELIEFSEQ_MAX_K, {}),
            },
        },
    },
    "stir": {
        "title": "STIR",
        "output": RESULTS / "stir_comparison.json",
        "report": RESULTS / "STIR.md",
        "parity": False,
        "scenarios": RELIEF_SCENARIOS,
        "note": (
            "No Python STIR exists to compare with (the authors' implementation is R, which was not run). STIR "
            "with its default MultiSURF neighbourhood is shown against MultiSURF itself, which computes the same "
            "neighbourhood's scores without statistics, in FastSelect and in scikit-rebate."
        ),
        "variants": {
            "STIR (MultiSURF neighbourhood)": {
                "fast-select": ("fast", "STIR", {}),
                "fast-select-multisurf": ("fast", "MultiSURF", {"backend": "cpu"}),
                "scikit-rebate-multisurf": ("rebate", "MultiSURF", {}),
            },
        },
    },
    "vls": {
        "title": "VLS",
        "output": RESULTS / "vls_vs_skrebate.json",
        "report": RESULTS / "VLS.md",
        "parity": False,
        "scenarios": {
            "samples": {"fixed": ("n_features", 1000), "sizes": [100, 300, 1000]},
            "features": {"fixed": ("n_samples", 100), "sizes": [1000, 5000, 10000]},
        },
        "note": (
            f"Both wrap ReliefF (`n_neighbors=10`) and score `2 * n_features / {VLS_SUBSET_SIZE}` random subsets of "
            f"{VLS_SUBSET_SIZE} features. The subsets are random in both libraries, so scores differ; this "
            "compares speed and memory."
        ),
        "variants": {
            f"VLS + ReliefF ({VLS_SUBSET_SIZE}-feature subsets)": {
                "fast-select": ("fast-vls", {"subset_size": VLS_SUBSET_SIZE}, {"n_neighbors": 10}),
                "scikit-rebate": ("rebate-vls", {"size_feature_subset": VLS_SUBSET_SIZE}, {"n_neighbors": 10}),
            },
        },
    },
    "mrmr": {
        "title": "Continuous mRMR (FCQ)",
        "output": RESULTS / "mrmr_fcq_comparison.json",
        "report": RESULTS / "MRMR_FCQ.md",
        "parity": True,
        "scenarios": {
            "samples": {"fixed": ("n_features", 1000), "sizes": [500, 2000, 10000]},
            "features": {"fixed": ("n_samples", 500), "sizes": [1000, 5000, 20000]},
        },
        "note": (
            '`mRMR(10, method="FCQ")` against `mrmr_selection`\'s `mrmr_classif(K=10)` (its default FCQ) and '
            "against the same criterion written with scikit-learn's `f_classif` and vectorised NumPy correlations "
            "(`|Z.T @ z|` per step on standardised columns, BLAS single-threaded like every worker here). "
            "`mrmr_selection` gets the faster of `n_jobs=-1` and `n_jobs=1`. FastSelect's selections equal the "
            "NumPy FCQ's at every shape; `mrmr_selection`'s differ wherever its 0.001 floor on each correlation "
            "changes a quotient, and equal the NumPy FCQ with that floor added at every shape (checked separately)."
        ),
        "variants": {
            "FCQ": {
                "fast-select": ("fast", "mRMR", {"method": "FCQ"}),
                "mrmr-selection": ("mrmr-selection",),
                "numpy-fcq": ("numpy-fcq",),
            },
        },
    },
    "hsic": {
        "title": "HSIC Lasso (block, B = 20, M = 3)",
        "output": RESULTS / "hsic_lasso_comparison.json",
        "report": RESULTS / "HSIC_LASSO.md",
        "parity": True,
        "scenarios": {
            "samples": {"fixed": ("n_features", 1000), "sizes": [500, 1000, 2000]},
            "features": {"fixed": ("n_samples", 500), "sizes": [1000, 2000, 5000]},
        },
        "note": (
            "`HSICLasso(10)` (block size 20, 3 permutations, normalised Gram matrices) against pyHSICLasso's "
            "`classification(num_feat=10)` with the same defaults. pyHSICLasso computes its kernels in float32, so "
            "coefficients agree to roughly single precision; the parity table records it."
        ),
        "variants": {
            "block": {
                "fast-select": ("fast", "HSICLasso", {}),
                "pyhsiclasso": ("pyhsiclasso", {"B": 20, "M": 3}),
            },
        },
    },
    "hsic_vanilla": {
        "title": "HSIC Lasso (vanilla, one block)",
        "output": RESULTS / "hsic_lasso_vanilla_comparison.json",
        "report": RESULTS / "HSIC_LASSO_VANILLA.md",
        "parity": True,
        "scenarios": {
            "samples": {"fixed": ("n_features", 200), "sizes": [100, 200, 400]},
            "features": {"fixed": ("n_samples", 200), "sizes": [200, 1000, 2000]},
        },
        "note": (
            "`HSICLasso(10, block_size=0)` against pyHSICLasso's `classification(num_feat=10, B=0)`. pyHSICLasso "
            "stores every feature's n x n Gram matrix, so shapes stay small."
        ),
        "variants": {
            "vanilla": {
                "fast-select": ("fast", "HSICLasso", {"block_size": 0}),
                "pyhsiclasso": ("pyhsiclasso", {"B": 0, "M": 1}),
            },
        },
    },
    "spec": {
        "title": "SPEC",
        "output": RESULTS / "spec_comparison.json",
        "report": RESULTS / "SPEC.md",
        "parity": True,
        "scenarios": {
            "samples": {"fixed": ("n_features", 100), "sizes": [200, 500, 1000, 2000]},
            "features": {"fixed": ("n_samples", 500), "sizes": [100, 1000, 5000]},
        },
        "note": (
            "`SPEC(10, criterion=...)` on the RBF graph (`gamma = 1 / n_features`; each fit builds it, "
            "scikit-feature through `rbf_kernel` passed as `W`) against the pinned scikit-feature `spec` "
            "(commit 48cffad, one NumPy-2 fix applied in memory: `D[i, i] = X_sum[i, 0]`; all mathematics "
            "unchanged) with `style=0` (phi2) or `style=11` (phi3 over 10 nontrivial eigenpairs). The scores "
            "agree; the selections differ because scikit-feature ranks phi2 descending and phi3 ascending, "
            "the opposite of the criteria's meaning. At 500 × 5,000 FastSelect's traced peak is one centred "
            "copy of `X`, which keeps the RBF distances free of cancellation; copies above 64 MiB are built in "
            "column blocks instead."
        ),
        "variants": {
            "phi2": {
                "fast-select": ("fast", "SPEC", {"criterion": "phi2"}),
                "scikit-feature": ("skfeature-spec", {"style": 0}),
            },
            "phi3 (m = 10)": {
                "fast-select": ("fast", "SPEC", {"criterion": "phi3", "n_eigenvectors": 10}),
                "scikit-feature": ("skfeature-spec", {"style": 11}),
            },
        },
    },
    "mcfs": {
        "title": "MCFS",
        "output": RESULTS / "mcfs_comparison.json",
        "report": RESULTS / "MCFS.md",
        "parity": True,
        "scenarios": {
            "samples": {"fixed": ("n_features", 100), "sizes": [500, 1000, 2000, 4000]},
            "features": {"fixed": ("n_samples", 500), "sizes": [100, 1000, 5000]},
        },
        "note": (
            "`MCFS(10, n_clusters=5)` against the pinned scikit-feature `mcfs` (commit 48cffad) given the same "
            "Euclidean binary 5-NN graph (`construct_W(X, metric='euclidean', weight_mode='binary', k=5)`). "
            "scikit-feature solves a dense eigenproblem, fits plain LARS rather than LARS-Lasso and scores by "
            "the largest signed coefficient. The agreement table compares the largest absolute coefficient for "
            "both: it agrees to rounding unless the Lasso path drops a variable before reaching 10 nonzero "
            "coefficients, where plain LARS (which never drops one) no longer solves Eq. (30). The "
            "selections differ because of the signed score."
        ),
        "variants": {
            "MCFS (K = 5)": {
                "fast-select": ("fast", "MCFS", {"n_clusters": 5}),
                "scikit-feature": ("skfeature-mcfs", {"n_clusters": 5}),
            },
        },
    },
}
ALL_VARIANTS = {name: spec for family in FAMILIES.values() for name, spec in family["variants"].items()}
LABELS = {
    "fast-select": "FastSelect",
    "scikit-rebate": "scikit-rebate",
    "fast-select-sweep": "FastSelect ReliefF, each k",
    "scikit-rebate-sweep": "scikit-rebate ReliefF, each k",
    "fast-select-multisurf": "FastSelect MultiSURF",
    "scikit-rebate-multisurf": "scikit-rebate MultiSURF",
    "numpy-fcq": "scikit-learn + NumPy FCQ",
    "mrmr-selection": "mrmr_selection",
    "pyhsiclasso": "pyHSICLasso",
    "scikit-feature": "scikit-feature",
}
N_FEATURES_TO_SELECT = 10
SERIAL_LIMIT_SECONDS = 10.0
MEMORY_MIN_SECONDS = 0.5
RESULT_PREFIX = "RESULT_JSON="
THREADS = 8
ENVIRONMENT = {
    "NUMBA_NUM_THREADS": str(THREADS),
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _labels(family):
    labels = []
    for implementations in FAMILIES[family]["variants"].values():
        labels += [label for label in implementations if label not in labels]
    return labels


def _series(family):
    return [
        (label, variant)
        for variant, implementations in FAMILIES[family]["variants"].items()
        for label in implementations
    ]


def _is_rebate(label, variant):
    kind = ALL_VARIANTS[variant][label][0]
    return kind.startswith("rebate") or kind in ("mrmr-selection", "pyhsiclasso")


def _shapes(family):
    for scenario, spec in FAMILIES[family]["scenarios"].items():
        fixed_name, fixed_value = spec["fixed"]
        for size in spec["sizes"]:
            if fixed_name == "n_samples":
                yield scenario, {"n_samples": fixed_value, "n_features": size}
            else:
                yield scenario, {"n_samples": size, "n_features": fixed_value}


def _dataset(n_samples, n_features):
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=min(20, n_features), random_state=42
    )


class _Sweep:
    """ReliefF fitted once per ``k = 1..k_max``, keeping each feature's best score (naive ReliefSeq)."""

    def __init__(self, make, k_max):
        self.make, self.k_max = make, k_max

    def fit(self, X, y):
        import numpy as np

        best = None
        for k in range(1, self.k_max + 1):
            scores = np.asarray(self.make(k).fit(X, y).feature_importances_, dtype=np.float64)
            best = scores if best is None else np.maximum(best, scores)
        self.feature_importances_ = best
        return self


class _NumpyFCQ:
    """FCQ written directly with scikit-learn's ``f_classif`` and NumPy, as a tuned reference."""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        import numpy as np
        from sklearn.feature_selection import f_classif

        X = np.asarray(X, dtype=np.float64)
        relevance = np.nan_to_num(f_classif(X, y)[0])
        centred = X - X.mean(axis=0)
        norms = np.sqrt(np.einsum("ij,ij->j", centred, centred))
        unit = centred / np.where(norms > 0, norms, 1.0)
        remaining = np.ones(X.shape[1], dtype=bool)
        selected = [int(np.argmax(relevance))]
        remaining[selected[0]] = False
        redundancy = np.abs(unit.T @ unit[:, selected[0]])
        for step in range(1, self.k):
            mean = redundancy / step
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(mean > 0, relevance / mean, np.where(relevance > 0, np.inf, 0.0))
            scores[~remaining] = -np.inf
            best = int(np.argmax(scores))
            selected.append(best)
            remaining[best] = False
            redundancy += np.abs(unit.T @ unit[:, best])
        self.feature_importances_ = relevance
        self.top_features_ = np.array(selected)
        return self


class _MrmrSelection:
    """``mrmr_selection.mrmr_classif`` (FCQ by default), imported from ``MRMR_SELECTION_PATH``."""

    def __init__(self, k, n_jobs):
        self.k, self.n_jobs = k, n_jobs

    def fit(self, X, y):
        import numpy as np
        import pandas as pd

        sys.path.append(os.environ["MRMR_SELECTION_PATH"])  # appended: NumPy, pandas, sklearn stay the venv's
        from mrmr import mrmr_classif

        selected = mrmr_classif(pd.DataFrame(X), pd.Series(y), K=self.k, show_progress=False, n_jobs=self.n_jobs or 1)
        self.top_features_ = np.asarray(selected, dtype=np.int64)
        self.feature_importances_ = None  # mrmr_classif returns the selection only
        return self


class _PyHSICLasso:
    """pyHSICLasso's ``classification``, imported from ``PYHSICLASSO_PATH``."""

    def __init__(self, k, n_jobs, B, M):
        self.k, self.n_jobs, self.B, self.M = k, n_jobs, B, M

    def fit(self, X, y):
        import contextlib
        import io

        import numpy as np

        sys.path.append(os.environ["PYHSICLASSO_PATH"])  # appended: NumPy, SciPy and pandas stay the venv's
        from pyHSICLasso import HSICLasso

        model = HSICLasso()
        with contextlib.redirect_stdout(io.StringIO()):
            model.input(np.asarray(X, dtype=np.float64), np.asarray(y))
            model.classification(num_feat=self.k, B=self.B, M=self.M, n_jobs=self.n_jobs or 1)
        self.top_features_ = np.asarray(model.get_index(), dtype=np.int64)
        self.feature_importances_ = np.asarray(model.beta, dtype=np.float64).ravel()
        return self


SKFEATURE_ROOT = ROOT / "audit" / "new_algorithms" / "vendor" / "jundongl-scikit-feature-48cffad"


def _skfeature_spec_module():
    """The pinned scikit-feature SPEC module with its one NumPy-2 incompatibility fixed in memory."""
    import types

    sys.path.insert(0, str(SKFEATURE_ROOT))
    path = SKFEATURE_ROOT / "skfeature" / "function" / "similarity_based" / "SPEC.py"
    source = path.read_text()
    assert source.count("D[i, i] = X_sum[i]") == 1
    module = types.ModuleType("skfeature_spec_patched")
    module.__file__ = str(path)
    exec(compile(source.replace("D[i, i] = X_sum[i]", "D[i, i] = X_sum[i, 0]"), str(path), "exec"), module.__dict__)
    return module


class _SkfeatureSPEC:
    def __init__(self, style):
        self.style = style

    def fit(self, X, y=None):
        import numpy as np
        from sklearn.metrics.pairwise import rbf_kernel

        module = _skfeature_spec_module()
        X = np.asarray(X, dtype=np.float64)
        # The same graph as FastSelect's default (gamma = 1 / n_features), passed explicitly.
        scores = module.spec(X, style=self.style, W=rbf_kernel(X, gamma=1.0 / X.shape[1]))
        # Higher is better in FastSelect's convention: -phi1/-phi2, +phi3.
        self.feature_importances_ = -scores if self.style in (-1, 0) else scores
        self.top_features_ = np.asarray(module.feature_ranking(scores, style=self.style))[:N_FEATURES_TO_SELECT]
        return self


class _SkfeatureMCFS:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        import numpy as np

        sys.path.insert(0, str(SKFEATURE_ROOT))
        from skfeature.function.sparse_learning_based import MCFS as module
        from skfeature.utility.construct_W import construct_W

        X = np.asarray(X, dtype=np.float64)
        W = construct_W(X, metric="euclidean", weight_mode="binary", k=5)
        coef = module.mcfs(X, N_FEATURES_TO_SELECT, W=W, n_clusters=self.n_clusters)
        self.feature_importances_ = np.abs(coef).max(axis=1)
        self.top_features_ = np.asarray(module.feature_ranking(coef))[:N_FEATURES_TO_SELECT]
        return self


def _estimator(label, variant, n_jobs=None):
    kind, *spec = ALL_VARIANTS[variant][label]
    if kind == "skfeature-spec":
        return _SkfeatureSPEC(**spec[0])
    if kind == "skfeature-mcfs":
        return _SkfeatureMCFS(**spec[0])
    if kind == "pyhsiclasso":
        return _PyHSICLasso(N_FEATURES_TO_SELECT, n_jobs, **spec[0])
    if kind == "numpy-fcq":
        return _NumpyFCQ(N_FEATURES_TO_SELECT)
    if kind == "mrmr-selection":
        return _MrmrSelection(N_FEATURES_TO_SELECT, n_jobs)
    select = {"n_features_to_select": N_FEATURES_TO_SELECT}
    if kind.startswith("fast"):
        import fast_select as library
    else:
        import skrebate as library

        select["n_jobs"] = n_jobs
    if kind in ("fast", "rebate"):
        name, keywords = spec
        return getattr(library, name)(**select, **keywords)
    if kind in ("fast-sweep", "rebate-sweep"):
        k_max, keywords = spec
        return _Sweep(lambda k: library.ReliefF(n_neighbors=k, **select, **keywords), k_max)
    vls_keywords, relieff_keywords = spec
    base = library.ReliefF(**select, **relieff_keywords)
    if kind == "fast-vls":
        return library.VLS(base, n_features_to_select=N_FEATURES_TO_SELECT, random_state=0, **vls_keywords)

    class _RebateVLS:
        """scikit-rebate's VLS with the same subset count as FastSelect's default (2 * ceil(p / size))."""

        def fit(self, X, y):
            size = vls_keywords["size_feature_subset"]
            count = 2 * -(-X.shape[1] // size)
            self.model = library.VLS(base, num_feature_subset=count, random_state=0, **vls_keywords).fit(X, y)
            self.feature_importances_ = self.model.feature_importances_
            return self

    return _RebateVLS()


def _summary(timings):
    quartiles = statistics.quantiles(timings, n=4) if len(timings) > 1 else [timings[0]] * 3
    return {
        "fits": len(timings),
        "measured_seconds": sum(timings),
        "median_seconds": statistics.median(timings),
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "q1_seconds": quartiles[0],
        "q3_seconds": quartiles[2],
        "max_seconds": max(timings),
        "timings_seconds": timings,
    }


def _measure(fit, min_seconds):
    """Median of fits totalling ``min_seconds`` (at least three unless one fit is that slow)."""
    timings = []
    while True:
        start = time.perf_counter()
        fit()
        timings.append(time.perf_counter() - start)
        if sum(timings) >= min_seconds and (len(timings) >= 3 or timings[0] >= min_seconds):
            return _summary(timings)


# --------------------------------------------------------------------------- runtime worker


def _runtime_worker(args):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ROOT / "src"))
    points = []
    warmed = set()
    serial_allowed = {scenario: True for scenario in FAMILIES[args.family]["scenarios"]}
    for scenario, shape in _shapes(args.family):
        X, y = _dataset(**shape)
        point = {"scenario": scenario, **shape}
        rebate = _is_rebate(args.library, args.variant)
        configurations = [-1, 1] if rebate else [None]
        for n_jobs in configurations:
            if n_jobs == 1 and not serial_allowed[scenario]:
                continue

            def fit(n_jobs=n_jobs):
                return _estimator(args.library, args.variant, n_jobs).fit(X, y)

            if not rebate or n_jobs not in warmed:
                fit()  # warm-up: JIT per shape for FastSelect, the worker pool once for scikit-rebate
                warmed.add(n_jobs)
            result = _measure(fit, args.min_seconds)
            key = "fit" if n_jobs is None else f"n_jobs={n_jobs}"
            point[key] = result
            if n_jobs == 1 and result["median_seconds"] > SERIAL_LIMIT_SECONDS:
                serial_allowed[scenario] = False
            print(
                f"  {args.library} {args.variant} {shape} {key}: {result['median_seconds']:.4g}s "
                f"({result['fits']} fits)",
                file=sys.stderr,
                flush=True,
            )
        medians = {name: value["median_seconds"] for name, value in point.items() if isinstance(value, dict)}
        point["best_configuration"] = min(medians, key=medians.get)
        point["median_seconds"] = medians[point["best_configuration"]]
        points.append(point)
    print(RESULT_PREFIX + json.dumps(points), flush=True)


def _parity_worker(args):
    """Fit both libraries once per shape and report the score difference."""
    import numpy as np

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ROOT / "src"))
    rows = []
    for scenario, shape in _shapes(args.family):
        X, y = _dataset(**shape)
        other = next(label for label in ALL_VARIANTS[args.variant] if label != "fast-select")
        fast = _estimator("fast-select", args.variant).fit(X, y)
        rebate = _estimator(other, args.variant, -1).fit(X, y)
        a = np.asarray(fast.feature_importances_, dtype=np.float64)
        scored = getattr(rebate, "feature_importances_", None) is not None
        b = np.asarray(rebate.feature_importances_, dtype=np.float64) if scored else a
        if hasattr(fast, "top_features_") and hasattr(rebate, "top_features_"):  # greedy selectors
            top_fast, top_rebate = set(fast.top_features_.tolist()), set(rebate.top_features_.tolist())
        else:
            top_fast = set(np.argsort(a)[::-1][:N_FEATURES_TO_SELECT].tolist())
            top_rebate = set(np.argsort(b)[::-1][:N_FEATURES_TO_SELECT].tolist())
        rows.append(
            {
                "scenario": scenario,
                **shape,
                "max_abs_difference": float(np.max(np.abs(a - b))) if scored else None,
                "max_abs_score": float(np.max(np.abs(b))) if scored else None,
                "same_order": (
                    list(fast.top_features_) == list(rebate.top_features_)
                    if hasattr(fast, "top_features_") and hasattr(rebate, "top_features_")
                    else None
                ),
                "top_features_shared": len(top_fast & top_rebate),
            }
        )
        print(f"  parity {shape}: {rows[-1]}", file=sys.stderr, flush=True)
    print(RESULT_PREFIX + json.dumps(rows), flush=True)


# --------------------------------------------------------------------------- memory worker and sampler


class _RUsageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


_LIBPROC = None


def _footprint(pid):
    """``(phys_footprint, resident_size)`` of ``pid`` in bytes (macOS ``proc_pid_rusage``)."""
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = _RUsageInfoV2()
    if _LIBPROC.proc_pid_rusage(int(pid), 2, ctypes.byref(info)) != 0:
        raise OSError(f"proc_pid_rusage failed for pid {pid}")
    return int(info.ri_phys_footprint), int(info.ri_resident_size)


def _memory_worker(args):
    """Fit one shape under the parent's sampler; protocol READY -> GO -> DONE on stdin/stdout."""
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ROOT / "src"))
    shape = {"n_samples": args.n_samples, "n_features": args.n_features}
    X, y = _dataset(**shape)
    n_jobs = 1 if _is_rebate(args.library, args.variant) else None

    def fit():
        return _estimator(args.library, args.variant, n_jobs).fit(X, y)

    # Both libraries warm up on the same 30-row slice: Numba compiles per dtype and
    # layout, not shape, so FastSelect's JIT is done, yet neither allocator has
    # already held full-size buffers when the sampled window starts.
    warm_up = _estimator(args.library, args.variant, n_jobs)
    if hasattr(warm_up, "get_params") and "n_neighbors" in warm_up.get_params():
        warm_up.set_params(n_neighbors=5)  # 30 rows cannot supply the full neighbour count
    warm_up.fit(X[:30], y[:30])
    gc.collect()
    print("READY", flush=True)
    sys.stdin.readline()  # GO
    fits = 0
    start = time.perf_counter()
    while True:
        fit()
        fits += 1
        if time.perf_counter() - start >= MEMORY_MIN_SECONDS:
            break
    elapsed = time.perf_counter() - start
    print("DONE", flush=True)
    sys.stdin.readline()  # parent has stopped sampling
    gc.collect()
    tracemalloc.start()
    fit()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        RESULT_PREFIX + json.dumps({"fits": fits, "sampled_seconds": elapsed, "tracemalloc_peak_bytes": traced_peak}),
        flush=True,
    )


def _sample_memory(family, library, variant, shape, environment):
    process = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "--family",
            family,
            "--memory-worker",
            "--library",
            library,
            "--variant",
            variant,
            "--n-samples",
            str(shape["n_samples"]),
            "--n-features",
            str(shape["n_features"]),
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        if process.stdout.readline().strip() != "READY":
            raise RuntimeError(process.stderr.read())
        baseline_footprint, baseline_resident = _footprint(process.pid)
        peak_footprint, peak_resident = baseline_footprint, baseline_resident
        samples = 0
        process.stdin.write("GO\n")
        process.stdin.flush()
        while True:
            footprint, resident = _footprint(process.pid)
            peak_footprint = max(peak_footprint, footprint)
            peak_resident = max(peak_resident, resident)
            samples += 1
            ready, _, _ = select.select([process.stdout], [], [], 0.0005)
            if ready:
                line = process.stdout.readline().strip()
                if line != "DONE":
                    raise RuntimeError(f"unexpected worker output {line!r}: {process.stderr.read()}")
                break
        process.stdin.write("STOP\n")
        process.stdin.flush()
        stdout, stderr = process.communicate(timeout=3600)
    except BaseException:
        process.kill()
        raise
    if process.returncode:
        raise RuntimeError(f"{library} {variant} {shape} memory worker failed:\n{stderr}")
    line = next(line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX))
    return {
        **json.loads(line[len(RESULT_PREFIX) :]),
        "baseline_footprint_bytes": baseline_footprint,
        "peak_footprint_bytes": peak_footprint,
        "footprint_increase_bytes": peak_footprint - baseline_footprint,
        "baseline_resident_bytes": baseline_resident,
        "peak_resident_bytes": peak_resident,
        "resident_increase_bytes": peak_resident - baseline_resident,
        "samples": samples,
    }


# --------------------------------------------------------------------------- provenance


def _tree_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _mrmr_selection_identity():
    path = os.environ.get("MRMR_SELECTION_PATH")
    if not path:
        return {}
    sys.path.append(path)
    import mrmr

    version = None
    for info in Path(path).glob("mrmr_selection-*.dist-info"):
        version = info.name.removesuffix(".dist-info").split("-")[1]
    return {"mrmr_selection": version, "mrmr_selection_source_sha256": _tree_sha256(Path(mrmr.__file__).parent)}


def _pyhsiclasso_identity():
    path = os.environ.get("PYHSICLASSO_PATH")
    if not path:
        return {}
    version = None
    for info in Path(path).glob("pyHSICLasso-*.dist-info"):
        version = info.name.removesuffix(".dist-info").split("-")[1]
    return {"pyhsiclasso": version, "pyhsiclasso_source_sha256": _tree_sha256(Path(path) / "pyHSICLasso")}


SKFEATURE_FILES = {
    "spec": ["skfeature/function/similarity_based/SPEC.py"],
    "mcfs": ["skfeature/function/sparse_learning_based/MCFS.py", "skfeature/utility/construct_W.py"],
}


def _skfeature_identity(family):
    if family not in SKFEATURE_FILES:
        return {}
    digest = hashlib.sha256()
    for name in SKFEATURE_FILES[family]:
        digest.update(name.encode())
        digest.update((SKFEATURE_ROOT / name).read_bytes())
    commit = SKFEATURE_ROOT.name.rsplit("-", 1)[-1]
    return {"scikit_feature": commit, "scikit_feature_source_sha256": digest.hexdigest()}


def _uses_skrebate(family):
    return any(
        kind.startswith("rebate")
        for implementations in FAMILIES[family]["variants"].values()
        for kind, *_ in implementations.values()
    )


def _provenance(family):
    import numba
    import numpy
    import sklearn

    rebate = {}
    if _uses_skrebate(family):
        import skrebate

        rebate = {
            "skrebate": skrebate.__version__,
            "skrebate_source_sha256": _tree_sha256(Path(skrebate.__file__).parent),
        }

    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    return {
        "family": family,
        "git_commit": git("rev-parse", "HEAD"),
        "git_src_dirty": bool(git("status", "--porcelain", "--", "src")),
        "source_sha256": _tree_sha256(ROOT / "src" / "fast_select"),
        **rebate,
        **_skfeature_identity(family),
        **_mrmr_selection_identity(),
        **_pyhsiclasso_identity(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": brand.strip(),
        "numba_threads": THREADS,
        "environment": ENVIRONMENT,
        "n_features_to_select": N_FEATURES_TO_SELECT,
        "memory_method": (
            "peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row "
            "slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats "
            "for >= 0.5 s, minus the footprint just before"
            + ("; scikit-rebate with n_jobs=1 so all work is in-process" if rebate else "")
            + ". tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only."
        ),
    }


# --------------------------------------------------------------------------- driver


def _worker_output(arguments, environment):
    completed = subprocess.run(
        [sys.executable, __file__, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"{' '.join(arguments)} failed:\n{completed.stderr}")
    line = next(line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX))
    return json.loads(line[len(RESULT_PREFIX) :])


def _run(args):
    family = FAMILIES[args.family]
    output = args.output or family["output"]
    environment = os.environ | ENVIRONMENT
    started = time.time()
    series_order = _series(args.family)
    runs = {series: [] for series in series_order}
    for round_index in range(args.rounds):
        # Alternate the series order between rounds so drift does not favour one library.
        for library, variant in series_order if round_index % 2 == 0 else reversed(series_order):
            print(f"runtime round {round_index + 1}/{args.rounds}: {library} {variant}", flush=True)
            load_before = os.getloadavg()
            points = _worker_output(
                [
                    "--family",
                    args.family,
                    "--runtime-worker",
                    "--library",
                    library,
                    "--variant",
                    variant,
                    "--min-seconds",
                    str(args.min_seconds),
                ],
                environment,
            )
            runs[library, variant].append(
                {
                    "round": round_index + 1,
                    "load_average_before": load_before,
                    "load_average_after": os.getloadavg(),
                    "points": points,
                }
            )

    parity = {}
    if family["parity"]:
        for variant in family["variants"]:
            print(f"parity: {variant}", flush=True)
            parity[variant] = _worker_output(
                ["--family", args.family, "--parity-worker", "--variant", variant], environment
            )

    memory = []
    if not args.skip_memory:
        for library, variant in series_order:
            for scenario, shape in _shapes(args.family):
                print(f"memory: {library} {variant} {shape}", flush=True)
                load_before = os.getloadavg()
                result = _sample_memory(args.family, library, variant, shape, environment)
                memory.append(
                    {
                        "library": library,
                        "variant": variant,
                        "scenario": scenario,
                        **shape,
                        "load_average_before": load_before,
                        "load_average_after": os.getloadavg(),
                        **result,
                    }
                )

    series = []
    for (library, variant), rounds in runs.items():
        points = []
        for index, point in enumerate(rounds[0]["points"]):
            per_round = [run["points"][index]["median_seconds"] for run in rounds]
            cumulative = sum(
                value["measured_seconds"]
                for run in rounds
                for value in run["points"][index].values()
                if isinstance(value, dict)
            )
            shape = {key: point[key] for key in ("scenario", "n_samples", "n_features")}
            points.append(
                {
                    **shape,
                    "median_seconds": statistics.median(per_round),
                    "round_medians": per_round,
                    "cumulative_measured_seconds": cumulative,
                }
            )
        series.append({"library": library, "variant": variant, "points": points, "rounds": rounds})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "provenance": _provenance(args.family),
                "min_seconds_per_round": args.min_seconds,
                "rounds": args.rounds,
                "wall_seconds": time.time() - started,
                "series": series,
                "parity": parity,
                "memory": memory,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output.relative_to(ROOT)}")


# --------------------------------------------------------------------------- report


def _format_seconds(value):
    if value >= 1.0:
        return f"{value:.2f} s"
    return f"{value * 1000:.3g} ms"


def _format_bytes(value):
    return f"{value / 2**20:.1f} MiB" if value >= 2**20 else f"{value / 2**10:.0f} KiB"


def _span(values):
    low, high = _format_bytes(min(values)), _format_bytes(max(values))
    return low if low == high else f"{low} to {high}"


def _ratio(value):
    return f"{value:.0f}×" if value >= 10 else f"{value:.1f}×"


def _report(args):
    family = FAMILIES[args.family]
    output = args.output or family["output"]
    data = json.loads(output.read_text(encoding="utf-8"))
    provenance = data["provenance"]
    labels = _labels(args.family)
    references = [label for label in labels if label != "fast-select"]
    runtime = {
        (s["library"], s["variant"]): {(p["scenario"], p["n_samples"], p["n_features"]): p for p in s["points"]}
        for s in data["series"]
    }
    memory = {(m["library"], m["variant"], m["scenario"], m["n_samples"], m["n_features"]): m for m in data["memory"]}
    rebate_note = (
        " External libraries get the faster of `n_jobs=-1` and `n_jobs=1`."
        if any(_is_rebate(label, variant) for variant, impl in family["variants"].items() for label in impl)
        else ""
    )
    lines = [
        f"# {family['title']}: FastSelect CPU against {', '.join(LABELS[label] for label in references)}",
        "",
        f"Generated from [`{output.name}`]({output.name}) by "
        f"`python benchmarking/benchmark_relief_variants.py --family {args.family} --report`.",
        "",
        f"- FastSelect `{provenance['git_commit'][:7]}`"
        f"{' with uncommitted src changes' if provenance['git_src_dirty'] else ''}, "
        f"source SHA-256 `{provenance['source_sha256'][:16]}…`"
        + (
            f"; scikit-rebate {provenance['skrebate']}, source SHA-256 `{provenance['skrebate_source_sha256'][:16]}…`"
            if "skrebate" in provenance
            else ""
        )
        + (
            f"; scikit-feature {provenance['scikit_feature']} (pinned tree), source SHA-256 "
            f"`{provenance['scikit_feature_source_sha256'][:16]}…` over the files used"
            if "scikit_feature" in provenance
            else ""
        )
        + (
            f"; mrmr_selection {provenance['mrmr_selection']}, source SHA-256 "
            f"`{provenance['mrmr_selection_source_sha256'][:16]}…`"
            if "mrmr_selection" in provenance
            else ""
        )
        + (
            f"; pyHSICLasso {provenance['pyhsiclasso']}, source SHA-256 "
            f"`{provenance['pyhsiclasso_source_sha256'][:16]}…`"
            if "pyhsiclasso" in provenance
            else ""
        )
        + ".",
        f"- {provenance['cpu']}, {provenance['numba_threads']} Numba threads, single-threaded BLAS; Python "
        f"{provenance['python']}, NumPy {provenance['numpy']}, Numba {provenance['numba']}, scikit-learn "
        f"{provenance['scikit_learn']}.",
        f"- Runtime: median over {data['rounds']} alternating rounds of per-round medians; each round fits for at "
        f"least {data['min_seconds_per_round']:g} s per case (at least "
        f"{data['rounds'] * data['min_seconds_per_round']:g} s cumulative).{rebate_note}",
        f"- Memory: {provenance['memory_method']}",
        "- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.",
        f"- {family['note']}",
        "",
        "Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. "
        "Footprint growth is page-granular and does not count allocations served from memory the process "
        "already holds, so small figures understate every implementation; the traced peak excludes Numba's "
        "per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).",
        "",
    ]
    loads = [
        value
        for s in data["series"]
        for run in s["rounds"]
        for value in (*run["load_average_before"][:1], *run["load_average_after"][:1])
    ]
    lines += [f"One-minute load average around the runtime workers: {min(loads):.1f} to {max(loads):.1f}.", ""]
    shapes = list(_shapes(args.family))
    largest = max((shape for scenario, shape in shapes if scenario == "samples"), key=lambda s: s["n_samples"])
    largest_key = ("samples", largest["n_samples"], largest["n_features"])
    ratios = {label: [] for label in references}
    traced = {label: [] for label in references}
    footprints = {label: [] for label in labels}
    for variant, implementations in family["variants"].items():
        present = [label for label in labels if label in implementations]
        others = [label for label in present if label != "fast-select"]
        header = (
            ["Scenario", "n × p"]
            + [LABELS[label] for label in present]
            + [f"{LABELS[label]} / FastSelect" for label in others]
            + [f"{LABELS[label]} memory" for label in present]
        )
        lines += [f"## {variant}", "", "| " + " | ".join(header) + " |", "|---|---|" + "---:|" * (len(header) - 2)]
        for scenario, shape in shapes:
            key = (scenario, shape["n_samples"], shape["n_features"])
            times = {label: runtime[label, variant][key]["median_seconds"] for label in present}
            cells = [scenario, f"{shape['n_samples']:,} × {shape['n_features']:,}"]
            cells += [_format_seconds(times[label]) for label in present]
            for label in others:
                ratio = times[label] / times["fast-select"]
                ratios[label].append(ratio)
                cells.append(_ratio(ratio))
            fast_memory = memory.get(("fast-select", variant, *key))
            for label in present:
                record = memory.get((label, variant, *key))
                if record is None:
                    cells.append("—")
                    continue
                cells.append(
                    f"{_format_bytes(record['tracemalloc_peak_bytes'])} / "
                    f"{_format_bytes(record['footprint_increase_bytes'])}"
                )
                if key == largest_key:
                    footprints[label].append(record["footprint_increase_bytes"])
                if label != "fast-select" and fast_memory:
                    traced[label].append(
                        record["tracemalloc_peak_bytes"] / max(fast_memory["tracemalloc_peak_bytes"], 1)
                    )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if variant in data.get("parity", {}):
            lines += [
                f"Score agreement ({variant}, one fit per library):",
                "",
                "| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for row in data["parity"][variant]:
                difference = row["max_abs_difference"]
                largest_score = row["max_abs_score"]
                order = row.get("same_order")
                lines.append(
                    f"| {row['scenario']} | {row['n_samples']:,} × {row['n_features']:,} | "
                    f"{'—' if difference is None else f'{difference:.2g}'} | "
                    f"{'—' if largest_score is None else f'{largest_score:.3g}'} | "
                    f"{row['top_features_shared']} | {'—' if order is None else ('yes' if order else 'no')} |"
                )
            lines.append("")
    lines += ["## Summary", ""]
    for label in references:
        values = ratios[label]
        if not values:
            continue
        lines.append(
            f"- Runtime, {LABELS[label]} / FastSelect: {_ratio(min(values))} to {_ratio(max(values))} "
            f"(median {_ratio(statistics.median(values))}) across {len(values)} cases."
        )
        if traced[label]:
            lines.append(
                f"- Traced host allocations, {LABELS[label]} / FastSelect: {min(traced[label]):.1f}× to "
                f"{max(traced[label]):.1f}× (median {statistics.median(traced[label]):.1f}×)."
            )
    measured = [label for label in labels if footprints[label]]
    if measured:
        lines.append(
            f"- Footprint growth at {largest['n_samples']:,} × {largest['n_features']:,}: "
            + "; ".join(f"{LABELS[label]} {_span(footprints[label])}" for label in measured)
            + ". FastSelect's footprint growth is often within memory the process already holds, so no ratio "
            "is given."
        )
    lines += [
        "- The load averages include the benchmark's own workers (eight Numba threads"
        + (", or eight scikit-rebate processes with `n_jobs=-1`" if "skrebate" in provenance else "")
        + ").",
        "",
    ]
    family["report"].write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {family['report'].relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), default="swrf")
    parser.add_argument("--runtime-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parity-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--library", help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--n-samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-features", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--report", action="store_true", help="render the family's Markdown report from its JSON")
    parser.add_argument("--output", type=Path, default=None, help="results JSON (default: the family's file)")
    args = parser.parse_args()
    if args.runtime_worker:
        _runtime_worker(args)
    elif args.parity_worker:
        _parity_worker(args)
    elif args.memory_worker:
        _memory_worker(args)
    elif args.report:
        _report(args)
    else:
        if sys.platform != "darwin":
            raise SystemExit("benchmark_relief_variants.py samples memory with macOS libproc; run it on macOS.")
        _run(args)


if __name__ == "__main__":
    main()
