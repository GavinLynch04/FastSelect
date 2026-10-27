"""Second five-algorithm admission batch; independent, tuned external baselines.

--tune calibrates official MI parallelism and both Gini references. A normal
run uses fresh workers, excludes warm-up, alternates order, and admits only
complete-output parity plus >=1.03x runtime gain in every round and shape.
Prototype bytes must be copied unchanged to src after admission.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/new_algorithms_2"
CASES = {
    "regmi_medium": ("mutual_info_regression", 2000, 128),
    "regmi_large": ("mutual_info_regression", 4000, 256),
    "gini_medium": ("GiniIndex", 900, 128),
    "gini_large": ("GiniIndex", 2000, 256),
    **{
        f"{name.lower()}_{size}": (name, n, p)
        for name in ("DISR", "CIFE", "MIFS")
        for size, n, p in (("medium", 900, 128), ("large", 2000, 256))
    },
}
MODULES = ["RegressionMI.py", "GiniIndex.py", "JointInformation.py"]


def fixture(case):
    name, n, p = CASES[case]
    rng = np.random.default_rng(7931)
    if name == "mutual_info_regression":
        X = rng.normal(size=(n, p))
        y = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=0.5, size=n)
    elif name == "GiniIndex":
        # >=1e-4 gaps preserve value ordering through CART's float32 inputs and
        # its 1e-7 near-equality rule. Every distinct threshold is still scanned.
        X = np.round(rng.normal(size=(n, p)), 4)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0.2).astype(int)
    else:
        X = rng.integers(0, 3, (n, p))
        y = (X[:, 0] + X[:, 1] + rng.integers(0, 2, n)) % 3
        X[:, 2] = np.where(rng.random(n) < 0.85, y, rng.integers(0, 3, n))
        X[:, 3] = np.where(rng.random(n) < 0.75, y, rng.integers(0, 3, n))
    return X, y


def stump_score(column, y):
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=0).fit(column[:, None], y).tree_
    if tree.node_count == 1:
        return tree.impurity[0]
    return np.dot(tree.impurity[1:], tree.weighted_n_node_samples[1:]) / len(y)


def reference_call(name, X, y, configuration):
    from joblib import Parallel, delayed

    if name == "mutual_info_regression":
        from sklearn.feature_selection import mutual_info_regression

        return mutual_info_regression(X, y, random_state=17, n_jobs=configuration["n_jobs"])
    if name == "GiniIndex":
        if configuration["implementation"] == "skfeature":
            from skfeature.function.statistical_based.gini_index import gini_index

            return gini_index(X, y)
        if configuration["n_jobs"] == 1:
            return np.array([stump_score(column, y) for column in X.T])
        return np.array(Parallel(n_jobs=configuration["n_jobs"])(delayed(stump_score)(column, y) for column in X.T))
    module = importlib.import_module(f"skfeature.function.information_theoretical_based.{name}")
    return getattr(module, name.lower())(X, y, n_selected_features=10)


def tune():
    from joblib import parallel_config
    from threadpoolctl import threadpool_limits

    manifest = json.loads((AUDIT / "baseline_manifest.json").read_text())
    sys.path.insert(0, manifest["scikit_feature"]["source_directory"])
    rows = []
    for case in ("regmi_medium", "regmi_large", "gini_medium", "gini_large"):
        name, _, _ = CASES[case]
        X, y = fixture(case)
        configurations = [
            {"implementation": "sklearn", "parallel_backend": backend, "n_jobs": jobs}
            for backend, jobs in (("loky", 1), ("threading", 4), ("threading", 8), ("loky", 4), ("loky", 8))
        ]
        if name == "GiniIndex":
            configurations.append({"implementation": "skfeature", "parallel_backend": "loky", "n_jobs": 1})
        reference_outputs = []
        for configuration in configurations:
            with threadpool_limits(1), parallel_config(backend=configuration["parallel_backend"]):
                reference_call(name, X, y, configuration)
                durations = []
                for _ in range(1 if configuration["implementation"] == "skfeature" else 3):
                    start = time.perf_counter()
                    output = reference_call(name, X, y, configuration)
                    durations.append(time.perf_counter() - start)
                reference_outputs.append(output)
            row = {
                "case": case,
                "configuration": configuration,
                "seconds": durations,
                "median_seconds": statistics.median(durations),
            }
            rows.append(row)
            (AUDIT / "baseline_tuning.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
        for output in reference_outputs[1:]:
            np.testing.assert_allclose(output, reference_outputs[0], rtol=2e-12, atol=2e-13)
    print("All tuned reference alternatives preserve full scores.")


def configuration_for(case):
    if CASES[case][0] in ("mutual_info_regression", "GiniIndex"):
        rows = json.loads((AUDIT / "baseline_tuning.json").read_text())
        return min((row for row in rows if row["case"] == case), key=lambda row: row["median_seconds"])["configuration"]
    return {"implementation": "skfeature", "parallel_backend": "loky", "n_jobs": 1}


def worker(args):
    from joblib import parallel_config
    from threadpoolctl import threadpool_limits

    name, n, p = CASES[args.case]
    X, y = fixture(args.case)
    manifest = json.loads((AUDIT / "baseline_manifest.json").read_text())
    context = contextlib.nullcontext()
    configuration = configuration_for(args.case) if args.backend == "reference" else None
    if args.backend == "reference":
        sys.path.insert(0, manifest["scikit_feature"]["source_directory"])
        context = parallel_config(backend=configuration["parallel_backend"])
        run = lambda: reference_call(name, X, y, configuration)
    else:
        sys.path.insert(0, str(ROOT))
        prefix = "fast_select" if args.production else "audit.new_algorithms_2.prototypes"
        if name == "mutual_info_regression":
            function = importlib.import_module(f"{prefix}.RegressionMI").mutual_info_regression
            run = lambda: function(X, y, random_state=17, n_jobs=8)
        else:
            module = importlib.import_module(f"{prefix}.{'GiniIndex' if name == 'GiniIndex' else 'JointInformation'}")
            cls = getattr(module, name)
            run = lambda: cls(10, n_jobs=8).fit(X, y)
    with threadpool_limits(1), context:
        value = run()
        durations, total = [], 0.0
        while total < args.seconds:
            start = time.perf_counter()
            value = run()
            duration = time.perf_counter() - start
            durations.append(duration)
            total += duration
    if name == "mutual_info_regression":
        output = {"scores": value.tolist(), "selected": np.argsort(-value, kind="stable")[:10].tolist()}
    elif name == "GiniIndex":
        scores = value if args.backend == "reference" else value.gini_scores_
        output = {"scores": scores.tolist(), "selected": np.argsort(scores, kind="stable")[:10].tolist()}
    elif args.backend == "reference":
        from skfeature.utility.entropy_estimators import midd

        output = {
            "selected": value[0].tolist(),
            "scores": value[1].tolist(),
            "relevance_selected": value[2].tolist(),
            "relevance_all": [midd(column, y) for column in X.T],
        }
    else:
        output = {
            "selected": value.top_features_.tolist(),
            "scores": value.selection_scores_.tolist(),
            "relevance_selected": value.relevance_scores_[value.top_features_].tolist(),
            "relevance_all": value.feature_importances_.tolist(),
        }
    return {
        "case": args.case,
        "algorithm": name,
        "shape": [n, p],
        "seed": 7931,
        "backend": args.backend,
        "reference_configuration": configuration,
        "numba_threads": 8 if args.backend == "cpu" else None,
        "blas_threads": 1,
        "iterations": len(durations),
        "measured_seconds": total,
        "median_seconds": statistics.median(durations),
        "durations_seconds": durations,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_method": "macOS peak process RSS including imports/warmup; excludes live joblib child pool",
        "reference_full_relevance_recomputed_outside_timing": name in ("DISR", "CIFE", "MIFS"),
        "output": output,
    }


def verify(result):
    decisions = {}
    for case, (name, _, _) in CASES.items():
        ratios = []
        for backend in ("reference", "cpu"):
            group = [r for r in result["measurements"] if r["case"] == case and r["backend"] == backend]
            assert len(group) == result["rounds"]
            assert sum(r["measured_seconds"] for r in group) >= 5, (case, backend, "duration")
        for round_number in range(1, result["rounds"] + 1):
            pair = {r["backend"]: r for r in result["measurements"] if r["case"] == case and r["round"] == round_number}
            a, b = pair["reference"], pair["cpu"]
            assert a["output"]["selected"] == b["output"]["selected"], (case, "selected", round_number)
            for key in a["output"]:
                if key != "selected":
                    np.testing.assert_allclose(
                        b["output"][key],
                        a["output"][key],
                        rtol=2e-10,
                        atol=2e-10,
                        err_msg=f"{case} {key} round {round_number}",
                    )
            ratios.append(a["median_seconds"] / b["median_seconds"])
        decisions[case] = {
            "algorithm": name,
            "paired_speedups": ratios,
            "admitted": all(ratio >= 1.03 for ratio in ratios),
        }
    return decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--case", choices=list(CASES))
    parser.add_argument("--backend", choices=["reference", "cpu"])
    parser.add_argument("--seconds", type=float, default=1.05)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default=str(AUDIT / "sustained.json"))
    args = parser.parse_args()
    if args.tune:
        tune()
        return
    if args.worker:
        print(json.dumps(worker(args)))
        return
    directory = ROOT / "src/fast_select" if args.production else AUDIT / "prototypes"
    result = {
        "machine": platform.platform(),
        "processor": "Apple M1 Pro, 8 CPU cores, 16 GiB RAM",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds": args.rounds,
        "source_mode": "production" if args.production else "prototype",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in MODULES},
        "baseline_manifest": json.loads((AUDIT / "baseline_manifest.json").read_text()),
        "minimum_seconds_per_worker": args.seconds,
        "measurements": [],
    }
    env = dict(os.environ, NUMBA_NUM_THREADS="8", PYTHONPATH=str(ROOT / "src"))
    for round_number in range(1, args.rounds + 1):
        for case in CASES:
            order = ("reference", "cpu") if round_number % 2 else ("cpu", "reference")
            for backend in order:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--backend",
                    backend,
                    "--seconds",
                    str(args.seconds),
                ]
                if args.production:
                    command.append("--production")
                process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
                row = json.loads(process.stdout)
                row["round"] = round_number
                result["measurements"].append(row)
                Path(args.output).write_text(json.dumps(result, indent=2))
                print(f"round {round_number} {case} {backend}: {row['median_seconds']:.6f}s", flush=True)
    result["decisions"] = verify(result)
    result["output_parity"] = "all score arrays, full marginal relevance and selected feature indices passed"
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["decisions"], indent=2))
    if not all(row["admitted"] for row in result["decisions"].values()):
        raise SystemExit("A candidate failed admission; do not promote it")


if __name__ == "__main__":
    main()
