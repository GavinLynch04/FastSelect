"""Compare ordinary random inputs with the existing paper-equation oracles."""

# ruff: noqa: E402 -- explicitly import the reviewed checkout
import importlib.util
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from fast_select import SURF, MultiSURF, ReliefF

spec = importlib.util.spec_from_file_location("oracles", str(ROOT / "tests/test_algorithm_compliance.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
warnings.filterwarnings("ignore")
results = {}
for cls in [ReliefF, SURF, MultiSURF]:
    for star in ([False] if cls == ReliefF else [False, True]):
        failures = []
        worst = 0.0
        for kind in ["discrete", "continuous", "mixed"]:
            for seed in range(40):
                rng = np.random.default_rng(seed)
                X = rng.uniform(-2, 4, (12, 5))
                y = np.array([0] * 5 + [1] * 7)
                disc = (
                    np.ones(5, bool)
                    if kind == "discrete"
                    else (np.zeros(5, bool) if kind == "continuous" else np.array([1, 0, 1, 0, 0], bool))
                )
                X[:, disc] = rng.integers(0, 3, (12, int(disc.sum())))
                kw = dict(n_features_to_select=2, backend="cpu", discrete_limit=3, n_jobs=1)
                if cls == ReliefF:
                    kw["n_neighbors"] = 2
                    expected = ref._relieff_reference(X, y, disc, 2)
                else:
                    kw["use_star"] = star
                    expected = (ref._surf_reference if cls == SURF else ref._multisurf_reference)(X, y, disc, star)
                actual = cls(**kw).fit(X, y).feature_importances_
                err = float(np.max(np.abs(actual - expected)))
                worst = max(worst, err)
                if err > 2e-6:
                    failures.append((kind, seed, err))
        results[f"{cls.__name__}_{star}"] = {"cases": 120, "worst_abs_error": worst, "failures": failures}
print(json.dumps(results, indent=2))
raise SystemExit(int(any(result["failures"] for result in results.values())))
