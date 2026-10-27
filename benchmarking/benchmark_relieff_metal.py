"""Fresh-process, complete-fit CPU/Metal benchmark; never times lazy dispatch alone.

Use --explore only for preliminary tuning. Promotion measurements require at
least five rounds and five cumulative measured seconds per worker. Every worker
uses identical seeded data; comparisons include every score and exact ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "small": (128, 32, "continuous", 2),
    "medium": (512, 128, "continuous", 2),
    "reference": (900, 128, "continuous", 2),
    "large": (2000, 512, "continuous", 2),
    "sample_heavy": (4000, 128, "continuous", 2),
    "feature_heavy": (1000, 2048, "continuous", 2),
    "mixed_multiclass": (2000, 512, "mixed", 4),
    "discrete": (2000, 512, "discrete", 2),
}


def fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "fast_select").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(args):
    sys.path.insert(0, str(Path(args.source_root) / "src"))
    import numba
    import numpy as np

    from fast_select import ReliefF

    n, p, kind, classes = CASES[args.case]
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((n, p), dtype=np.float32)
    if kind == "discrete":
        x[:] = rng.integers(0, 3, (n, p))
    elif kind == "mixed":
        x[:, ::2] = rng.integers(0, 3, (n, len(range(0, p, 2))))
    y = np.arange(n, dtype=np.int32) % classes
    rng.shuffle(y)
    model = ReliefF(
        backend=args.backend,
        n_neighbors=args.neighbors,
        n_features_to_select=min(10, p),
        discrete_limit=3,
        n_jobs=args.threads,
    )
    begin = time.perf_counter()
    model.fit(x, y)
    first_fit = time.perf_counter() - begin
    for _ in range(2):
        model.fit(x, y)
    timings = []
    measured = 0.0
    while measured < args.min_seconds:
        start = time.perf_counter()
        model.fit(x, y)
        elapsed = time.perf_counter() - start
        measured += elapsed
        timings.append(elapsed)
    result = {
        "case": args.case,
        "backend": args.backend,
        "samples": n,
        "features": p,
        "kind": kind,
        "classes": classes,
        "neighbors": args.neighbors,
        "seed": args.seed,
        "dtype": str(x.dtype),
        "numba_threads": numba.get_num_threads(),
        "first_fit_seconds": first_fit,
        "iterations": len(timings),
        "measured_seconds": measured,
        "mean_seconds": statistics.fmean(timings),
        "median_seconds": statistics.median(timings),
        "timings_seconds": timings,
        "scores": model.feature_importances_.tolist(),
        "selected_indices": model.top_features_.tolist(),
        "source_sha256": fingerprint(Path(args.source_root)),
        "metal_diagnostics": getattr(model, "metal_diagnostics_", None),
        "memory_method": "Not measured; distance_buffer_bytes is one buffer, not total peak memory",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "numba", "scikit-learn", "mlx"]},
    }
    print("RESULT_JSON=" + json.dumps(result), flush=True)


def main(args):
    import numpy as np

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "method": (
            "Fresh worker per backend/case/round; alternating order; three warm-up fits; synchronized host results"
        ),
        "statistic": "Median of per-round mean complete-fit latency",
        "exploratory": args.explore,
        "threads": args.threads,
        "rounds": args.rounds,
        "minimum_measured_seconds_per_worker": args.min_seconds,
        "rtol": 1e-4,
        # The large independent oracle test bounds existing CPU float32
        # accumulation error at ~1.1e-6; Metal's error is ~2.4e-8 on that case.
        "atol": 2e-6,
        "workers": [],
        "comparisons": [],
    }
    env = dict(os.environ)
    env.update(
        NUMBA_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env.pop("NUMBA_ENABLE_CUDASIM", None)
    for round_index in range(args.rounds):
        for case in args.cases:
            pair = {}
            backends = ("cpu", "metal") if round_index % 2 == 0 else ("metal", "cpu")
            for backend in backends:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source-root",
                    args.source_root,
                    "--case",
                    case,
                    "--backend",
                    backend,
                    "--threads",
                    str(args.threads),
                    "--neighbors",
                    str(args.neighbors),
                    "--seed",
                    str(args.seed),
                    "--min-seconds",
                    str(args.min_seconds),
                    "--explore",
                ]
                completed = subprocess.run(command, env=env, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(f"{case}/{backend} failed:\n{completed.stdout}\n{completed.stderr}")
                line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT_JSON="))
                result = json.loads(line.removeprefix("RESULT_JSON="))
                result["round"] = round_index + 1
                pair[backend] = result
                report["workers"].append(result)
                print(f"round={round_index + 1} {case} {backend}: {result['mean_seconds'] * 1000:.3f} ms", flush=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
            cpu, metal = pair["cpu"], pair["metal"]
            a, b = np.array(cpu["scores"]), np.array(metal["scores"])
            finite = np.isfinite(a).all() and np.isfinite(b).all()
            comparison = {
                "case": case,
                "round": round_index + 1,
                "speedup": cpu["mean_seconds"] / metal["mean_seconds"],
                "scores_match": bool(finite and np.allclose(a, b, rtol=report["rtol"], atol=report["atol"])),
                "selected_indices_match": cpu["selected_indices"] == metal["selected_indices"],
                "max_abs_score_error": float(np.max(np.abs(a - b))),
            }
            report["comparisons"].append(comparison)
            print(f"  {comparison}", flush=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
    summaries = []
    for case in args.cases:
        cpu = [w["mean_seconds"] for w in report["workers"] if w["case"] == case and w["backend"] == "cpu"]
        metal = [w["mean_seconds"] for w in report["workers"] if w["case"] == case and w["backend"] == "metal"]
        comparisons = [c for c in report["comparisons"] if c["case"] == case]
        summaries.append(
            {
                "case": case,
                "cpu_ms": statistics.median(cpu) * 1000,
                "metal_ms": statistics.median(metal) * 1000,
                "speedup": statistics.median(cpu) / statistics.median(metal),
                "per_round_speedup_range": [
                    min(c["speedup"] for c in comparisons),
                    max(c["speedup"] for c in comparisons),
                ],
                "full_output_parity": all(c["scores_match"] and c["selected_indices_match"] for c in comparisons),
                "runtime_gate": all(c["speedup"] >= 1.03 for c in comparisons),
                "max_abs_score_error": max(c["max_abs_score_error"] for c in comparisons),
            }
        )
    report["summaries"] = summaries
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--backend", choices=["cpu", "metal"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-seconds", type=float, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output", default=str(ROOT / "benchmarking/results/m1pro_relieff_metal.json"))
    args = parser.parse_args()
    if not args.explore and (args.rounds < 5 or args.min_seconds < 5):
        parser.error("Use at least five rounds and five measured seconds, or explicitly --explore")
    if args.worker:
        worker(args)
    else:
        main(args)
