"""Sustained benchmark of the pushed HEAD tree against the working tree.

The parent process extracts a baseline git ref (HEAD by default, i.e. the code
currently pushed on this branch) and starts a fresh Python process for every
version/case pair. JIT compilation is warmed up and excluded. Each reported
timing contains at least ``--min-seconds`` seconds of measured fits, and every
case is repeated over ``--rounds`` independent rounds with the execution order
alternating between rounds.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import warnings
import zipfile
from pathlib import Path

DEFAULT_BASELINE_REF = "HEAD"
RESULT_PREFIX = "RESULT_JSON="


def _worker(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore")
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root / "src"))

    import numpy as np

    if args.estimator == "MutualInformation":
        from fast_select.mutual_information import calculate_mi_matrices

        rng = np.random.default_rng(20210721)
        x = rng.integers(0, 4, (args.samples, args.features), dtype=np.int32)
        y = rng.integers(0, 4, args.samples, dtype=np.int32)

        calculate_mi_matrices(x, y, backend=args.backend)
        timings: list[float] = []
        measured = 0.0
        while measured < args.min_seconds:
            start = time.perf_counter()
            relevance, redundancy = calculate_mi_matrices(x, y, backend=args.backend)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            measured += elapsed

        gc.collect()
        if args.backend == "gpu":
            memory_relevance = relevance
            memory_redundancy = redundancy
            peak_bytes = 0
        else:
            tracemalloc.start()
            memory_relevance, memory_redundancy = calculate_mi_matrices(x, y, backend=args.backend)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        result = {
            "version": args.version,
            "estimator": args.estimator,
            "backend": args.backend,
            "source": str(source_root),
            "module": sys.modules[calculate_mi_matrices.__module__].__file__,
            "samples": args.samples,
            "features": args.features,
            "iterations": len(timings),
            "measured_seconds": measured,
            "mean_seconds": statistics.fmean(timings),
            "median_seconds": statistics.median(timings),
            "min_seconds": min(timings),
            "peak_traced_mb": peak_bytes / (1024 * 1024),
            "score_sum": float(memory_relevance.sum()),
            "score_l2": float(np.linalg.norm(memory_redundancy)),
        }
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return

    from fast_select.CFS import CFS
    from fast_select.MDR import MDR
    from fast_select.MultiSURF import MultiSURF
    from fast_select.ReliefF import ReliefF
    from fast_select.SURF import SURF

    estimators = {
        "ReliefF": ReliefF,
        "SURF": SURF,
        "SURFStar": SURF,
        "MultiSURF": MultiSURF,
        "MultiSURFStar": MultiSURF,
        "CFS": CFS,
        "MDR": MDR,
    }
    estimator_class = estimators[args.estimator]

    rng = np.random.default_rng(20210721)
    if args.estimator == "CFS":
        x = rng.integers(0, 4, (args.samples, args.features), dtype=np.int32)
        y = rng.integers(0, 2, args.samples, dtype=np.int32)
        kwargs = {"backend": args.backend}
    elif args.estimator == "MDR":
        x = rng.integers(0, 3, (args.samples, args.features), dtype=np.uint8)
        y = rng.integers(0, 2, args.samples, dtype=np.uint8)
        kwargs = {"k": 2, "cv": 3, "backend": args.backend}
    else:
        x = rng.standard_normal((args.samples, args.features), dtype=np.float32)
        y = rng.integers(0, 2, args.samples, dtype=np.int32)
        kwargs = {
            "n_features_to_select": min(10, args.features),
            "backend": args.backend,
        }
        if args.estimator == "ReliefF":
            kwargs["n_neighbors"] = 10
        if args.estimator in ("SURFStar", "MultiSURFStar"):
            kwargs["use_star"] = True

    # Compile every relevant signature, then warm the actual problem size once.
    warm_samples = min(96, args.samples)
    warm_features = min(16 if args.estimator == "MDR" else 32, args.features)
    estimator_class(**kwargs).fit(x[:warm_samples, :warm_features], y[:warm_samples])
    estimator_class(**kwargs).fit(x, y)

    timings: list[float] = []
    measured = 0.0
    while measured < args.min_seconds:
        start = time.perf_counter()
        model = estimator_class(**kwargs).fit(x, y)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        measured += elapsed

    gc.collect()
    if args.backend == "gpu":
        # tracemalloc cannot observe device allocations, and an extra fit after
        # sustained v0.2.1 CUDA use can trigger its known context-lifetime bug.
        memory_model = model
        peak_bytes = 0
    else:
        tracemalloc.start()
        memory_model = estimator_class(**kwargs).fit(x, y)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    if hasattr(memory_model, "feature_importances_"):
        scores = memory_model.feature_importances_.astype(np.float64)
    elif args.estimator == "CFS":
        scores = np.array(
            [memory_model.merit_, len(memory_model.selected_indices_)],
            dtype=np.float64,
        )
    else:
        scores = memory_model.best_model_lookup_table_.astype(np.float64)
    result = {
        "version": args.version,
        "estimator": args.estimator,
        "backend": args.backend,
        "source": str(source_root),
        "module": sys.modules[estimator_class.__module__].__file__,
        "samples": args.samples,
        "features": args.features,
        "iterations": len(timings),
        "measured_seconds": measured,
        "mean_seconds": statistics.fmean(timings),
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "peak_traced_mb": peak_bytes / (1024 * 1024),
        "score_sum": float(scores.sum()),
        "score_l2": float(np.linalg.norm(scores)),
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def _run_worker(
    args: argparse.Namespace,
    source_root: Path,
    version: str,
    estimator: str,
    backend: str,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root),
        "--version",
        version,
        "--estimator",
        estimator,
        "--backend",
        backend,
        "--samples",
        str(args.samples),
        "--features",
        str(args.features),
        "--min-seconds",
        str(args.min_seconds),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Worker failed with exit code {completed.returncode}:\n" f"{completed.stdout}\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"Worker returned no result:\n{completed.stdout}\n{completed.stderr}")


def _extract_baseline(repo_root: Path, destination: Path, ref: str) -> None:
    archive_path = destination.parent / "fast-select-baseline.zip"
    with archive_path.open("wb") as archive:
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_root.as_posix()}",
                "archive",
                "--format=zip",
                ref,
            ],
            cwd=repo_root,
            check=True,
            stdout=archive,
        )
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)


def _parent(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = [(name, backend) for backend in args.backends for name in args.estimators]
    results: list[dict] = []

    baseline_label = f"baseline-{args.baseline_ref}"
    with tempfile.TemporaryDirectory(prefix="fast-select-baseline-") as temp:
        baseline_root = Path(temp) / "source"
        baseline_root.mkdir()
        _extract_baseline(repo_root, baseline_root, args.baseline_ref)

        for estimator, backend in cases:
            for round_index in range(args.rounds):
                print(
                    f"\n{estimator} ({backend.upper()}), round " f"{round_index + 1}/{args.rounds}",
                    flush=True,
                )
                variants = [
                    (baseline_label, baseline_root),
                    ("working-tree", repo_root),
                ]
                if round_index % 2:
                    variants.reverse()

                round_results = {}
                for version, source_root in variants:
                    print(f"  {version}", flush=True)
                    result = _run_worker(args, source_root, version, estimator, backend)
                    result["round"] = round_index + 1
                    results.append(result)
                    round_results[version] = result
                    print(
                        f"    {result['mean_seconds']:.6f}s/fit across "
                        f"{result['measured_seconds']:.2f}s "
                        f"({result['iterations']} fits)",
                        flush=True,
                    )

                speedup = round_results[baseline_label]["mean_seconds"] / round_results["working-tree"]["mean_seconds"]
                print(f"    round speedup {speedup:.3f}x", flush=True)

    print("\nSummary (lower runtime is better)")
    print("estimator   backend  baseline(s)  current(s)  speedup  rounds" "  traced-memory change")
    for estimator, backend in cases:
        matching = [item for item in results if item["estimator"] == estimator and item["backend"] == backend]
        baseline = [item for item in matching if item["version"] == baseline_label]
        current = [item for item in matching if item["version"] == "working-tree"]
        baseline_time = statistics.median(item["mean_seconds"] for item in baseline)
        current_time = statistics.median(item["mean_seconds"] for item in current)
        baseline_memory = statistics.median(item["peak_traced_mb"] for item in baseline)
        current_memory = statistics.median(item["peak_traced_mb"] for item in current)
        speedup = baseline_time / current_time
        memory_delta = current_memory - baseline_memory
        per_round = [
            b["mean_seconds"] / c["mean_seconds"]
            for b, c in zip(
                sorted(baseline, key=lambda item: item["round"]),
                sorted(current, key=lambda item: item["round"]),
            )
        ]
        print(
            f"{estimator:<11} {backend:<7} {baseline_time:>11.6f} "
            f"{current_time:>11.6f} {speedup:>8.3f}x {len(per_round):>6} "
            f"{memory_delta:>+10.2f} MiB   per-round: " + " ".join(f"{value:.2f}" for value in per_round)
        )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nDetailed results written to {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    parser.add_argument("--version", help=argparse.SUPPRESS)
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=[
            "ReliefF",
            "SURF",
            "SURFStar",
            "MultiSURF",
            "MultiSURFStar",
            "MutualInformation",
            "CFS",
            "MDR",
        ],
        default=["ReliefF", "SURF", "MultiSURF"],
    )
    parser.add_argument(
        "--estimator",
        choices=[
            "ReliefF",
            "SURF",
            "SURFStar",
            "MultiSURF",
            "MultiSURFStar",
            "MutualInformation",
            "CFS",
            "MDR",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--backends", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    parser.add_argument("--backend", choices=["cpu", "gpu"], help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=900)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.min_seconds < 5.0:
        parser.error("--min-seconds must be at least 5.0")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if not args.worker and args.rounds < 5:
        parser.error("--rounds must be at least 5 for a reportable comparison")
    return args


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker:
        _worker(parsed)
    else:
        _parent(parsed)
