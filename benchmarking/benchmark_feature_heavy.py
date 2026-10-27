"""Feature-heavy complete CPU/Metal/CUDA research comparison.

CUDA simulation is forbidden for timings. Frozen CPU source identity, exact
output parity, warm-up, fresh processes and alternating order are recorded.
Experimental GPU paths include host preparation, transfers and exact repairs.
"""

import argparse
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
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / "audit/feature_heavy"
ALGORITHMS = ["mutual_info_regression", "GiniIndex", "DISR", "CIFE", "MIFS"]
SHAPES = {"small": (64, 1024), "medium": (128, 4096), "large": (256, 8192)}


def fixture(name, shape):
    import benchmark_more_algorithms as previous

    n, p = SHAPES[shape]
    previous.CASES["_feature_heavy"] = (name, n, p)
    return previous.fixture("_feature_heavy")


def cpu(name, X, y):
    import fast_select as fs

    if name == "mutual_info_regression":
        scores = fs.mutual_info_regression(X, y, random_state=17, n_jobs=-1)
        return {"scores": scores, "selected": np.argsort(-scores, kind="stable")[:10]}
    model = getattr(fs, name)(10, n_jobs=-1).fit(X, y)
    if name == "GiniIndex":
        return {"scores": model.gini_scores_, "selected": model.top_features_}
    return {
        "scores": model.selection_scores_,
        "selected": model.top_features_,
        "relevance_all": model.feature_importances_,
        "relevance_selected": model.relevance_scores_[model.top_features_],
    }


def worker(args):
    from numba import config
    from threadpoolctl import threadpool_limits

    if args.backend == "cuda" and config.ENABLE_CUDASIM:
        raise RuntimeError("CUDA simulator timings are not GPU performance measurements")
    load_before = list(os.getloadavg())
    if args.max_start_load is not None and load_before[0] > args.max_start_load:
        raise RuntimeError(f"System load {load_before[0]:.2f} exceeds --max-start-load={args.max_start_load}")
    X, y = fixture(args.algorithm, args.shape)
    device = None
    if args.backend == "cuda":
        from numba import cuda

        gpu = cuda.get_current_device()
        device = {"name": str(gpu.name), "compute_capability": list(gpu.compute_capability)}
    if args.backend == "cpu":
        run = lambda: cpu(args.algorithm, X, y)
    else:
        from audit.feature_heavy.common import run as run_gpu

        provider = importlib.import_module(f"audit.feature_heavy.{args.backend}")
        run = lambda: run_gpu(args.algorithm, X, y, provider)
    with threadpool_limits(1):
        output = run()
        durations = []
        total = 0.0
        while total < args.seconds:
            start = time.perf_counter()
            output = run()
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            total += elapsed
    return {
        "algorithm": args.algorithm,
        "shape_name": args.shape,
        "shape": list(X.shape),
        "backend": args.backend,
        "numba_threads": config.NUMBA_NUM_THREADS,
        "cuda_device": device,
        "blas_threads": 1,
        "load_average_before": load_before,
        "load_average_after": list(os.getloadavg()),
        "iterations": len(durations),
        "measured_seconds": total,
        "median_seconds": statistics.median(durations),
        "durations_seconds": durations,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "memory_method": "process peak RSS including imports/warm-up, not isolated GPU memory",
        "output": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in output.items()},
    }


def compare(a, b):
    assert a["output"]["selected"] == b["output"]["selected"], (a["algorithm"], a["shape"], "selected")
    for field in ["scores", "relevance_all", "relevance_selected"]:
        if field in a["output"]:
            np.testing.assert_allclose(
                a["output"][field],
                b["output"][field],
                rtol=2e-10,
                atol=2e-10,
                err_msg=f"{a['algorithm']} {a['shape']} {field}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--shape", choices=list(SHAPES))
    parser.add_argument("--backend", choices=["cpu", "metal", "cuda"])
    parser.add_argument("--backends", nargs="+", default=["cpu", "metal"])
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES))
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--max-start-load", type=float)
    parser.add_argument("--seconds", type=float, default=1.05)
    parser.add_argument("--output", default=str(AUDIT / "sustained.json"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    frozen = json.loads((AUDIT / "manifest.json").read_text())
    for name, sha in frozen["cpu_sha256"].items():
        assert hashlib.sha256((ROOT / "src/fast_select" / name).read_bytes()).hexdigest() == sha, name
    result = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_manifest": frozen,
        "experimental_source_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (AUDIT / "common.py", AUDIT / "metal.py", AUDIT / "cuda.py")
        },
        "rounds": 1 if args.screen else args.rounds,
        "screen": args.screen,
        "measurements": [],
    }
    env = dict(os.environ, NUMBA_NUM_THREADS=str(args.threads), PYTHONPATH=str(ROOT / "src"))
    for round_number in range(1, result["rounds"] + 1):
        for shape in args.shapes:
            for name in args.algorithms:
                order = list(args.backends)
                if round_number % 2 == 0:
                    order.reverse()
                pair = {}
                for backend in order:
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--algorithm",
                        name,
                        "--shape",
                        shape,
                        "--backend",
                        backend,
                        "--seconds",
                        str(0.15 if args.screen else args.seconds),
                    ]
                    if args.max_start_load is not None:
                        command.extend(["--max-start-load", str(args.max_start_load)])
                    worker_env = dict(env)
                    if backend == "cpu" and args.cpu_threads is not None:
                        worker_env["NUMBA_NUM_THREADS"] = str(args.cpu_threads)
                    completed = subprocess.run(command, cwd=ROOT, env=worker_env, capture_output=True, text=True)
                    if completed.returncode:
                        print(completed.stderr, flush=True)
                        raise SystemExit(completed.returncode)
                    row = json.loads(completed.stdout)
                    row["round"] = round_number
                    pair[backend] = row
                    result["measurements"].append(row)
                    Path(args.output).write_text(json.dumps(result, indent=2))
                    print(f"round {round_number} {name} {shape} {backend}: {row['median_seconds']:.6f}s", flush=True)
                for backend, row in pair.items():
                    if backend != "cpu":
                        compare(pair["cpu"], row)
    if not args.screen:
        for shape in args.shapes:
            for name in args.algorithms:
                for backend in args.backends:
                    group = [
                        r
                        for r in result["measurements"]
                        if r["shape_name"] == shape and r["algorithm"] == name and r["backend"] == backend
                    ]
                    assert sum(r["measured_seconds"] for r in group) >= 5
    result["output_parity"] = "complete score arrays, selected indices and marginal relevance passed"
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(result["output_parity"])


if __name__ == "__main__":
    main()
