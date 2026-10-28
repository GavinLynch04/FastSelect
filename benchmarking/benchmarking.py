"""Relief-family runtime scaling behind the README charts.

Every series (library, algorithm, backend) runs in its own fresh process with
eight Numba threads and single-threaded BLAS. FastSelect warms every size
(JIT compilation, Metal kernels) and scikit-rebate warms each process pool
once; each point is then the median of repeated fits totalling at least
``--min-seconds`` (at least three fits unless one fit already takes that long).
The whole set runs ``--rounds`` times in alternating order, and each plotted
point is the median of its round medians. ReliefF uses
``n_neighbors=10`` in both libraries. scikit-rebate is timed with ``n_jobs=1``
and ``n_jobs=-1`` and the faster median is plotted; ``n_jobs=1`` is skipped
once a single fit takes longer than ``SERIAL_LIMIT_SECONDS``.

    python benchmarking/benchmarking.py             # writes results/readme_scaling.json
    python benchmarking/plots.py                    # draws the README charts

Inputs are seeded ``make_classification`` data, not every real dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarking" / "results" / "readme_scaling.json"
ALGORITHMS = ("ReliefF", "SURF", "MultiSURF")
SCENARIOS = {
    "samples": {"fixed": ("n_features", 100), "sizes": [100, 200, 500, 1000, 2000, 5000]},
    "features": {"fixed": ("n_samples", 100), "sizes": [100, 200, 500, 1000, 2000, 5000, 10000]},
}
SERIES = [("fast-select", algorithm, backend) for algorithm in ALGORITHMS for backend in ("cpu", "metal")] + [
    ("scikit-rebate", algorithm, "cpu") for algorithm in ALGORITHMS
]
N_FEATURES_TO_SELECT = 10
N_NEIGHBORS = 10
SERIAL_LIMIT_SECONDS = 10.0
RESULT_PREFIX = "RESULT_JSON="


def _dataset(n_samples, n_features):
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=min(20, n_features), random_state=42
    )


def _estimator(library, algorithm, backend, n_jobs=None):
    keyword = {"n_neighbors": N_NEIGHBORS} if algorithm == "ReliefF" else {}
    if library == "fast-select":
        import fast_select

        return getattr(fast_select, algorithm)(n_features_to_select=N_FEATURES_TO_SELECT, backend=backend, **keyword)
    import skrebate

    return getattr(skrebate, algorithm)(n_features_to_select=N_FEATURES_TO_SELECT, n_jobs=n_jobs, **keyword)


def _measure(fit, min_seconds):
    """Median of fits totalling ``min_seconds`` (at least three unless one fit is that slow)."""
    timings = []
    while True:
        start = time.perf_counter()
        fit()
        timings.append(time.perf_counter() - start)
        if sum(timings) >= min_seconds and (len(timings) >= 3 or timings[0] >= min_seconds):
            break
    return _summary(timings)


def _summary(timings):
    """Fit count, measured duration and the timing distribution of one point."""
    quartiles = statistics.quantiles(timings, n=4) if len(timings) > 1 else [timings[0]] * 3
    return {
        "fits": len(timings),
        "measured_seconds": sum(timings),
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "q1_seconds": quartiles[0],
        "q3_seconds": quartiles[2],
        "max_seconds": max(timings),
    }


def _worker(args):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ROOT / "src"))
    points = []
    warmed = set()
    for scenario, spec in SCENARIOS.items():
        fixed_name, fixed_value = spec["fixed"]
        serial_allowed = True
        for size in spec["sizes"]:
            shape = {"n_samples": size, "n_features": fixed_value}
            if fixed_name == "n_samples":
                shape = {"n_samples": fixed_value, "n_features": size}
            X, y = _dataset(**shape)
            point = {"scenario": scenario, **shape}
            configurations = [None] if args.library == "fast-select" else [-1, 1]
            for n_jobs in configurations:
                if n_jobs == 1 and not serial_allowed:
                    continue
                fit = lambda n_jobs=n_jobs: _estimator(args.library, args.algorithm, args.backend, n_jobs).fit(X, y)
                if args.library == "fast-select" or n_jobs not in warmed:
                    fit()  # warm-up: JIT and Metal kernels per shape, the worker pool once
                    warmed.add(n_jobs)
                result = _measure(fit, args.min_seconds)
                key = "fit" if n_jobs is None else f"n_jobs={n_jobs}"
                point[key] = result
                if n_jobs == 1 and result["median_seconds"] > SERIAL_LIMIT_SECONDS:
                    serial_allowed = False
                print(
                    f"  {args.library} {args.algorithm} {args.backend} {shape} {key}: "
                    f"{result['median_seconds']:.4g}s ({result['fits']} fits)",
                    file=sys.stderr,
                    flush=True,
                )
            medians = [value["median_seconds"] for name, value in point.items() if isinstance(value, dict)]
            point["median_seconds"] = min(medians)
            points.append(point)
    print(RESULT_PREFIX + json.dumps(points), flush=True)


def _source_sha256():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "fast_select").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _provenance():
    import numba
    import numpy
    import sklearn
    import skrebate

    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_src_dirty": bool(git("status", "--porcelain", "--", "src")),
        "source_sha256": _source_sha256(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "scikit_learn": sklearn.__version__,
        "skrebate": skrebate.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numba_threads": 8,
        "n_features_to_select": N_FEATURES_TO_SELECT,
        "relieff_n_neighbors": N_NEIGHBORS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--library", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return

    environment = os.environ | {
        "NUMBA_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    runs = {spec: [] for spec in SERIES}
    for round_index in range(args.rounds):
        # Alternate the series order between rounds so drift does not favor one.
        for library, algorithm, backend in SERIES if round_index % 2 == 0 else reversed(SERIES):
            print(f"round {round_index + 1}/{args.rounds}: {library} {algorithm} ({backend})", flush=True)
            load_before = os.getloadavg()
            completed = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--library",
                    library,
                    "--algorithm",
                    algorithm,
                    "--backend",
                    backend,
                    "--min-seconds",
                    str(args.min_seconds),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(f"{library} {algorithm} {backend} failed:\n{completed.stderr}")
            line = next(line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX))
            runs[(library, algorithm, backend)].append(
                {
                    "round": round_index + 1,
                    "load_average_before": load_before,
                    "load_average_after": os.getloadavg(),
                    "points": json.loads(line[len(RESULT_PREFIX) :]),
                }
            )
    series = []
    for (library, algorithm, backend), rounds in runs.items():
        points = []
        for index, point in enumerate(rounds[0]["points"]):
            per_round = [run["points"][index]["median_seconds"] for run in rounds]
            shape = {key: point[key] for key in ("scenario", "n_samples", "n_features")}
            points.append({**shape, "median_seconds": statistics.median(per_round), "round_medians": per_round})
        series.append(
            {"library": library, "algorithm": algorithm, "backend": backend, "points": points, "rounds": rounds}
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"provenance": _provenance(), "min_seconds": args.min_seconds, "rounds": args.rounds, "series": series},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
