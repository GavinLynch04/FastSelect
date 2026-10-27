"""Diagnostic CUDA simulation only; never validates real-device execution.

The runtime shim works around a confirmed production import failure. Numerical
kernels are unmodified. Results are printed for inspection, including failures.
"""

# Imports after the shim are intentional: fast_select installs a runtime patch.
# ruff: noqa: E402
import importlib
import importlib.util
import sys
import types
from pathlib import Path

from numba import config

if not config.ENABLE_CUDASIM:
    raise SystemExit("Run with NUMBA_ENABLE_CUDASIM=1; this shim is simulator-only.")

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))
# Test-only shim: production import assumes real CUDA runtime internals.
# Let CUDASIM load; do not alter any numerical kernels.
import numba.cuda.cudadrv.devices as devices

devices._Runtime = type("_Runtime", (), {})
devices._runtime = types.SimpleNamespace()
import fast_select  # noqa: F401 -- initialize the package before availability shims

for name in ["utils", "ReliefF", "SURF", "MultiSURF", "CFS", "MDR", "mRMR", "mutual_information"]:
    mod = importlib.import_module("fast_select." + name)
    mod.ensure_cuda_context = lambda: True
    mod.is_cuda_ready = lambda: True
mi = importlib.import_module("fast_select.mutual_information")
mi._CUDA_AVAILABLE = True
import numpy as np

from fast_select import CFS, MDR, SURF, MultiSURF, ReliefF

spec = importlib.util.spec_from_file_location("oracles", str(ROOT / "tests/test_algorithm_compliance.py"))
oracles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracles)
X = oracles.MIXED_X
y = oracles.MIXED_Y
disc = oracles.MIXED_DISCRETE
for cls in [ReliefF, SURF, MultiSURF]:
    for star in ([False] if cls == ReliefF else [False, True]):
        kw = dict(n_features_to_select=2, discrete_limit=3, n_jobs=1)
        if cls == ReliefF:
            kw["n_neighbors"] = 1
        else:
            kw["use_star"] = star
        try:
            cpu = cls(backend="cpu", **kw).fit(X, y).feature_importances_
            gpu = cls(backend="gpu", **kw).fit(X, y).feature_importances_
            expected = (
                oracles._relieff_reference(X, y, disc, 1)
                if cls == ReliefF
                else (oracles._surf_reference if cls == SURF else oracles._multisurf_reference)(X, y, disc, star)
            )
            print(
                cls.__name__,
                star,
                "cpu",
                cpu.tolist(),
                "gpu",
                gpu.tolist(),
                "oracle",
                expected.tolist(),
                "maxerr",
                float(np.max(np.abs(gpu - expected))),
                flush=True,
            )
        except Exception as e:
            print(cls.__name__, star, type(e).__name__, str(e), flush=True)
xd = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2, dtype=np.int32)
yd = np.array([0, 0, 1, 1] * 2, dtype=np.int32)
for unit in ["bit", "nat"]:
    a, b = mi.calculate_mi_matrices(xd, yd, backend="gpu", unit=unit)
    print("MI", unit, a.tolist(), b.tolist(), flush=True)
for cls, kw in [(CFS, dict(n_jobs=1)), (MDR, dict(k=1, cv=2))]:
    try:
        a = cls(backend="cpu", **kw).fit(xd, yd)
        b = cls(backend="gpu", **kw).fit(xd, yd)
        print(cls.__name__, "cpu", a.transform(xd).tolist(), "gpu", b.transform(xd).tolist(), flush=True)
    except Exception as e:
        print(cls.__name__, type(e).__name__, str(e), flush=True)
# Exact global-threshold equality: normalized distances 1/3, 2/3, 1.
xt = np.array([[0.0], [1.0], [3.0]])
yt = np.array([0, 0, 1])
for star in [False, True]:
    for backend in ["cpu", "gpu"]:
        model = SURF(1, backend=backend, use_star=star, discrete_limit=0, n_jobs=1).fit(xt, yt)
        print("SURF_threshold", star, backend, model.feature_importances_.tolist(), flush=True)
# Exercise both CUDA score-accumulation branches around the 512-feature cutoff.
for p in [65, 513]:
    xwide = np.tile(X, (1, (p + 1) // 2))[:, :p]
    for cls in [SURF, MultiSURF]:
        for star in [False, True]:
            kw = dict(n_features_to_select=2, discrete_limit=3, n_jobs=1, use_star=star)
            a = cls(backend="cpu", **kw).fit(xwide, y).feature_importances_
            b = cls(backend="gpu", **kw).fit(xwide, y).feature_importances_
            print("wide", p, cls.__name__, star, "maxerr", float(np.max(np.abs(a - b))), flush=True)
