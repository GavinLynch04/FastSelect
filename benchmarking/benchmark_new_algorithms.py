"""External admission gate: fresh workers, warmed complete calls, alternating order.

Run from the checkout with the pinned reference tree in baseline_manifest.json.
No reference package is imported by fast_select. Prototypes stay outside src
until this gate admits them. Run without concurrent benchmarks or test jobs.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/new_algorithms"
CASES = {
    "mi_medium": ("MI", 2000, 128),
    "mi_large": ("MI", 4000, 256),
    "lap_medium": ("LaplacianScore", 900, 128),
    "lap_large": ("LaplacianScore", 1600, 256),
    **{
        f"{a.lower()}_{size}": (a, n, p)
        for a in ("JMI", "CMIM", "FCBF")
        for size, n, p in (("medium", 900, 128), ("large", 2000, 256))
    },
}


def fixture(case):
    algorithm, n, p = CASES[case]
    rng = np.random.default_rng(5401)
    if algorithm in ("MI", "LaplacianScore"):
        X = rng.normal(size=(n, p))
        y = np.digitize(X[:, 0] + 0.6 * X[:, 1] + rng.normal(size=n), [-0.7, 0.7])
    else:
        X = rng.integers(0, 3, (n, p))
        y = (X[:, 0] + X[:, 1] + rng.integers(0, 2, n)) % 3
        X[:, 2] = np.where(rng.random(n) < 0.85, y, rng.integers(0, 3, n))
        # Correlated and irrelevant peers, with informative nonidentical features.
        X[:, 3] = np.where(rng.random(n) < 0.75, y, rng.integers(0, 3, n))
    return X, y


def source_hash(directory):
    h = hashlib.sha256()
    for path in sorted(Path(directory).glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def worker(args):
    from joblib import parallel_config
    from threadpoolctl import threadpool_limits

    algorithm, n, p = CASES[args.case]
    X, y = fixture(args.case)
    manifest = json.loads((AUDIT / "baseline_manifest.json").read_text())
    context = contextlib.nullcontext()
    tuning = None
    if args.backend == "reference":
        sys.path.insert(0, manifest["source_directory"])
        if algorithm == "MI":
            from sklearn.feature_selection import mutual_info_classif

            candidates = json.loads((AUDIT / "mi_baseline_tuning.json").read_text())
            tuning = min((v for v in candidates if v["n"] == n and v["p"] == p), key=lambda v: v["median"])
            context = parallel_config(backend=tuning["backend"])
            run = lambda: mutual_info_classif(X, y, random_state=7, n_jobs=tuning["n_jobs"])
            extract = lambda value: {
                "scores": value.tolist(),
                "selected": np.argsort(-value, kind="stable")[:10].tolist(),
            }
        elif algorithm == "LaplacianScore":
            from skfeature.function.similarity_based.lap_score import lap_score
            from skfeature.utility.construct_W import construct_W

            def run():
                W = construct_W(X, metric="euclidean", weight_mode="binary", k=5)
                return lap_score(X, W=W)

            extract = lambda value: {
                "scores": value.tolist(),
                "selected": np.argsort(value, kind="stable")[:10].tolist(),
            }
        else:
            module = importlib.import_module(f"skfeature.function.information_theoretical_based.{algorithm}")
            if algorithm == "FCBF":
                # Correct only a nonfunctional NumPy keyword; retain all math.
                source = Path(module.__file__).read_text().replace("dtypes='object'", "dtype='object'")
                exec(compile(source, module.__file__, "exec"), module.__dict__)
                run = lambda: module.fcbf(X, y)
            else:
                run = lambda: getattr(module, algorithm.lower())(X, y, n_selected_features=10)
            extract = lambda v: {
                "selected": np.asarray(v[0], dtype=int).tolist(),
                "scores": np.asarray(v[1], dtype=float).tolist(),
                **({"relevance_selected": np.asarray(v[2], dtype=float).tolist()} if len(v) == 3 else {}),
            }
    else:
        sys.path.insert(0, str(ROOT))
        if algorithm == "MI":
            if args.production:
                from fast_select import mutual_info_classif
            else:
                from audit.new_algorithms.prototypes.ContinuousMI import mutual_info_classif

            run = lambda: mutual_info_classif(X, y, random_state=7, n_jobs=8)
            extract = lambda value: {
                "scores": value.tolist(),
                "selected": np.argsort(-value, kind="stable")[:10].tolist(),
            }
        else:
            module_name = "LaplacianScore" if algorithm == "LaplacianScore" else "InformationSelectors"
            prefix = "fast_select" if args.production else "audit.new_algorithms.prototypes"
            module = importlib.import_module(f"{prefix}.{module_name}")
            cls = getattr(module, algorithm)

            def run():
                estimator = cls(
                    **({} if algorithm == "FCBF" else {"n_features_to_select": 10}), backend=args.backend, n_jobs=8
                )
                return estimator.fit(X, y)

            def extract(value):
                return {
                    "selected": value.top_features_.tolist(),
                    "scores": (
                        value.laplacian_scores_ if algorithm == "LaplacianScore" else value.selection_scores_
                    ).tolist(),
                    **(
                        {"relevance_selected": value.relevance_scores_[value.top_features_].tolist()}
                        if algorithm in ("JMI", "CMIM")
                        else {}
                    ),
                }

    # BLAS=1 is chosen for information methods (no baseline GEMM) and calibrated
    # independently for the complete graph path (see lap_baseline_tuning.json).
    blas = 1
    if algorithm == "LaplacianScore":
        candidates = json.loads((AUDIT / "lap_baseline_tuning.json").read_text())
        tuning = min((v for v in candidates if v["n"] == n and v["p"] == p), key=lambda v: v["median"])
        blas = tuning["blas_threads"]
    with threadpool_limits(blas), context:
        value = run()  # warm kernels, GPU compilation, process pool, imports
        durations = []
        elapsed = 0.0
        while elapsed < args.seconds:
            start = time.perf_counter()
            value = run()
            duration = time.perf_counter() - start
            durations.append(duration)
            elapsed += duration
        output = extract(value)
    import resource

    return {
        "case": args.case,
        "algorithm": algorithm,
        "shape": [n, p],
        "backend": args.backend,
        "numba_threads": 8 if args.backend != "reference" else None,
        "blas_threads": blas,
        "baseline_tuning": tuning,
        "iterations": len(durations),
        "measured_seconds": elapsed,
        "median_seconds": float(np.median(durations)),
        "durations_seconds": durations,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_method": (
            "macOS getrusage peak process RSS; includes imports/warmup; " "Metal device allocations not isolated"
        ),
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=list(CASES))
    parser.add_argument("--backend", choices=["reference", "cpu", "metal"])
    parser.add_argument("--seconds", type=float, default=1.05)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", default=str(AUDIT / "sustained.json"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    result = {
        "machine": platform.platform(),
        "processor": "Apple M1 Pro, 8 CPU cores, 16 GiB RAM",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_manifest": json.loads((AUDIT / "baseline_manifest.json").read_text()),
        "prototype_source_sha256": source_hash(AUDIT / "prototypes"),
        "implementation": "production" if args.production else "prototype",
        "production_source_sha256": (
            {
                name: hashlib.sha256((ROOT / "src/fast_select" / name).read_bytes()).hexdigest()
                for name in ("ContinuousMI.py", "InformationSelectors.py", "LaplacianScore.py", "_cpu_backend.py")
            }
            if args.production
            else None
        ),
        "rounds": args.rounds,
        "minimum_seconds_per_worker": args.seconds,
        "measurements": [],
    }
    env = dict(os.environ, NUMBA_NUM_THREADS="8", PYTHONPATH=str(ROOT / "src"))
    for round_number in range(args.rounds):
        for case, (algorithm, _, _) in CASES.items():
            backends = ["reference", "cpu"] + ([] if algorithm == "MI" or args.cpu_only else ["metal"])
            if round_number % 2:
                backends.reverse()
            for backend in backends:
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
                completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
                row = json.loads(completed.stdout)
                row["round"] = round_number + 1
                result["measurements"].append(row)
                Path(args.output).write_text(json.dumps(result, indent=2))
                print(
                    f"round {round_number + 1} {case} {backend}: "
                    f"{row['median_seconds']:.6f}s ({row['iterations']} iterations)",
                    flush=True,
                )
    for case in CASES:
        for round_number in range(1, args.rounds + 1):
            rows = [r for r in result["measurements"] if r["case"] == case and r["round"] == round_number]
            reference = next(r for r in rows if r["backend"] == "reference")["output"]
            for row in rows:
                candidate = row["output"]
                assert candidate["selected"] == reference["selected"], (case, row["backend"], "selection")
                for name in ("scores", "relevance_selected"):
                    if name in reference:
                        np.testing.assert_allclose(
                            candidate[name],
                            reference[name],
                            rtol=2e-10,
                            atol=2e-10,
                            err_msg=f"{case} {row['backend']} {name}",
                        )
    result["output_parity"] = "all scores and selected feature indices passed in every fresh worker"
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(result["output_parity"])


if __name__ == "__main__":
    main()
