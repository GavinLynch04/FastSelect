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
import hashlib
import json
import math
import os
import platform
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
GPU_MEMORY_UNAVAILABLE = "unavailable: device allocations are not observable by tracemalloc"


def _save_outputs(args, **arrays):
    if args.output_arrays:
        import numpy as np

        np.savez(args.output_arrays, **arrays)


def _compare_outputs(expected, actual, rtol, atol):
    import numpy as np

    parity = bool(expected) and expected.keys() == actual.keys()
    max_error = 0.0
    if parity:
        for name, reference in expected.items():
            candidate = actual[name]
            if candidate.shape != reference.shape:
                parity = False
                continue
            if reference.dtype.kind not in "biuf" or candidate.dtype.kind not in "biuf":
                return False, None
            if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
                return False, None
            if np.issubdtype(reference.dtype, np.integer):
                parity = parity and np.array_equal(candidate, reference)
            else:
                parity = parity and np.allclose(candidate, reference, rtol=rtol, atol=atol)
            if reference.size:
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = np.abs(candidate.astype(np.float64) - reference)
                if not np.all(np.isfinite(difference)):
                    return False, None
                max_error = max(max_error, float(np.max(difference)))
    return bool(parity), max_error


def _improvement_gate(baseline, current, margin, backend):
    baseline = sorted(baseline, key=lambda item: item["round"])
    current = sorted(current, key=lambda item: item["round"])
    if not baseline or [item["round"] for item in baseline] != [item["round"] for item in current]:
        raise ValueError("Improvement gate requires matching, nonempty rounds")
    runtime_improved = all(
        math.isfinite(reference["mean_seconds"])
        and math.isfinite(candidate["mean_seconds"])
        and reference["mean_seconds"] > 0
        and candidate["mean_seconds"] > 0
        and reference["mean_seconds"] / candidate["mean_seconds"] >= 1 + margin
        for reference, candidate in zip(baseline, current)
    )
    memory_improved = backend == "cpu" and all(
        reference["peak_traced_mb"] is not None
        and candidate["peak_traced_mb"] is not None
        and math.isfinite(reference["peak_traced_mb"])
        and math.isfinite(candidate["peak_traced_mb"])
        and reference["peak_traced_mb"] > 0
        and candidate["peak_traced_mb"] > 0
        and candidate["peak_traced_mb"] <= reference["peak_traced_mb"] * (1 - margin)
        for reference, candidate in zip(baseline, current)
    )
    parity_passed = all(item["full_output_parity"] for item in baseline + current)
    return parity_passed, runtime_improved, memory_improved


def _measurement_details(args, timings, x):
    import numba
    import scipy

    return {
        "timings_seconds": timings,
        "input_dtype": str(x.dtype),
        "states": args.states,
        "seed": args.seed,
        "threads": numba.get_num_threads(),
        "scipy": scipy.__version__,
    }


def _git(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _provenance(source_root: Path, backend: str) -> dict:
    """Identify exactly what was measured and on what."""
    import numba
    import numpy
    import sklearn

    sources = sorted((source_root / "src" / "fast_select").glob("*.py"))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    commit = _git(source_root, "rev-parse", "HEAD")
    dirty = bool(_git(source_root, "status", "--porcelain", "--", "src"))
    device = platform.processor() or platform.machine()
    if backend == "gpu":
        try:
            from numba import cuda

            device = cuda.get_current_device().name
            device = device.decode() if isinstance(device, bytes) else str(device)
        except Exception:  # pragma: no cover - no CUDA device
            device = "unavailable"
    return {
        "source_sha256": digest.hexdigest(),
        "git_commit": commit,
        "git_src_dirty": dirty,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "device": device,
    }


def _emit(result: dict, source_root: Path, backend: str) -> None:
    """Attach provenance and an explicit memory method, then print the result line."""
    result.update(_provenance(source_root, backend))
    if backend == "gpu":
        result["peak_traced_mb"] = None
        result["memory_method"] = GPU_MEMORY_UNAVAILABLE
    else:
        result["memory_method"] = "tracemalloc Python/NumPy peak of one warmed fit; excludes total RSS and CUDA memory"
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def _worker(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore")
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root / "src"))

    import numpy as np

    if args.estimator in ("MutualInformation", "Chi2"):
        from fast_select.mutual_information import calculate_mi_matrices

        if args.estimator == "Chi2":
            import inspect

            from fast_select.Chi2 import chi2

            if args.backend != "cpu":
                raise ValueError("Chi2 has only a CPU backend")

            chi2_options = {"backend": "cpu"} if "backend" in inspect.signature(chi2).parameters else {}

            def compute(features, target, backend):
                return chi2(features, target, **chi2_options)

            compute_module = chi2.__module__
        else:
            compute = calculate_mi_matrices
            compute_module = compute.__module__

        rng = np.random.default_rng(args.seed)
        x = rng.integers(0, args.states, (args.samples, args.features), dtype=np.int32)
        y = rng.integers(0, args.states, args.samples, dtype=np.int32)

        compute(x, y, backend=args.backend)
        timings: list[float] = []
        measured = 0.0
        while measured < args.min_seconds:
            start = time.perf_counter()
            relevance, redundancy = compute(x, y, backend=args.backend)
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
            memory_relevance, memory_redundancy = compute(x, y, backend=args.backend)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        result = {
            "version": args.version,
            "estimator": args.estimator,
            "backend": args.backend,
            "source": str(source_root),
            "module": sys.modules[compute_module].__file__,
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
        result.update(_measurement_details(args, timings, x))
        _save_outputs(args, first=memory_relevance, second=memory_redundancy)
        _emit(result, source_root, args.backend)
        return

    from fast_select.CFS import CFS
    from fast_select.MDR import MDR
    from fast_select.mRMR import mRMR
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
        "mRMR": mRMR,
        "mRMR-MIQ": mRMR,
    }
    estimator_class = estimators[args.estimator]

    rng = np.random.default_rng(args.seed)
    if args.estimator in ("CFS", "mRMR", "mRMR-MIQ"):
        x = rng.integers(0, args.states, (args.samples, args.features), dtype=np.int32)
        y = rng.integers(0, 2, args.samples, dtype=np.int32)
        kwargs = {"backend": args.backend}
        if args.estimator.startswith("mRMR"):
            kwargs["n_features_to_select"] = min(10, args.features)
            kwargs["method"] = "MIQ" if args.estimator == "mRMR-MIQ" else "MID"
    elif args.estimator == "MDR":
        x = rng.integers(0, 3, (args.samples, args.features), dtype=np.uint8)
        y = rng.integers(0, 2, args.samples, dtype=np.uint8)
        kwargs = {"k": 2, "cv": 3, "backend": args.backend}
    else:
        x = rng.standard_normal((args.samples, args.features), dtype=np.dtype(args.dtype))
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
    result.update(_measurement_details(args, timings, x))
    arrays = {"scores": scores}
    for attribute in ("top_features_", "selected_indices_", "relevance_scores_", "redundancy_matrix_"):
        if hasattr(memory_model, attribute):
            arrays[attribute] = getattr(memory_model, attribute)
    if args.estimator == "MDR":
        arrays["best_interaction"] = np.asarray(memory_model.best_interaction_)
        arrays["cv_scores"] = np.array([memory_model.best_cvc_, memory_model.best_mean_testing_ba_])
        arrays["predictions"] = memory_model.predict(x)
    _save_outputs(args, **arrays)
    _emit(result, source_root, args.backend)


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
        "--states",
        str(args.states),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--output-arrays",
        str(args.output_arrays),
    ]
    environment = os.environ.copy()
    environment.update(
        NUMBA_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    completed = subprocess.run(
        command,
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
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
    import numpy as np

    repo_root = Path(__file__).resolve().parents[1]
    baseline_sha = subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=repo_root, text=True).strip()
    source_hash = hashlib.sha256()
    for source in sorted((repo_root / "src" / "fast_select").glob("*.py")):
        source_hash.update(source.name.encode())
        source_hash.update(source.read_bytes())
    cases = [(name, backend) for backend in args.backends for name in args.estimators]
    results: list[dict] = []

    baseline_label = f"baseline-{args.baseline_ref}"
    with tempfile.TemporaryDirectory(prefix="fast-select-baseline-") as temp:
        baseline_root = Path(temp) / "source"
        baseline_root.mkdir()
        _extract_baseline(repo_root, baseline_root, baseline_sha)

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
                round_outputs = {}
                for version, source_root in variants:
                    print(f"  {version}", flush=True)
                    args.output_arrays = str(Path(temp) / "outputs.npz")
                    result = _run_worker(args, source_root, version, estimator, backend)
                    result["round"] = round_index + 1
                    result["baseline_sha"] = baseline_sha
                    result["working_source_sha256"] = source_hash.hexdigest()
                    result["comparison_rtol"] = args.rtol
                    result["comparison_atol"] = args.atol
                    result["minimum_improvement"] = args.min_improvement
                    with np.load(args.output_arrays, allow_pickle=False) as output:
                        round_outputs[version] = {name: output[name] for name in output.files}
                    if version == baseline_label:
                        result["git_commit"] = baseline_sha
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
                expected = round_outputs[baseline_label]
                actual = round_outputs["working-tree"]
                parity, max_error = _compare_outputs(expected, actual, args.rtol, args.atol)
                for result in round_results.values():
                    result["full_output_parity"] = bool(parity)
                    result["max_absolute_output_error"] = max_error
                error_text = "unavailable (invalid output)" if max_error is None else f"{max_error:.3g}"
                print(f"    full-output parity: {parity}, max absolute error: {error_text}", flush=True)

    print("\nSummary (lower runtime is better)")
    print("estimator   backend  baseline(s)  current(s)  speedup  rounds" "  traced-memory change")
    gate_passed = True
    for estimator, backend in cases:
        matching = [item for item in results if item["estimator"] == estimator and item["backend"] == backend]
        baseline = [item for item in matching if item["version"] == baseline_label]
        current = [item for item in matching if item["version"] == "working-tree"]
        baseline_time = statistics.median(item["mean_seconds"] for item in baseline)
        current_time = statistics.median(item["mean_seconds"] for item in current)
        speedup = baseline_time / current_time
        if any(item["peak_traced_mb"] is None for item in matching):
            memory_text = f"{'unavailable':>14}"
        else:
            baseline_memory = statistics.median(item["peak_traced_mb"] for item in baseline)
            current_memory = statistics.median(item["peak_traced_mb"] for item in current)
            memory_text = f"{current_memory - baseline_memory:>+10.2f} MiB"
        per_round = [
            b["mean_seconds"] / c["mean_seconds"]
            for b, c in zip(
                sorted(baseline, key=lambda item: item["round"]),
                sorted(current, key=lambda item: item["round"]),
            )
        ]
        parity_passed, runtime_improved, memory_improved = _improvement_gate(
            baseline, current, args.min_improvement, backend
        )
        passed = parity_passed and (runtime_improved or memory_improved)
        gate_passed = gate_passed and passed
        print(
            f"{estimator:<11} {backend:<7} {baseline_time:>11.6f} "
            f"{current_time:>11.6f} {speedup:>8.3f}x {len(per_round):>6} "
            f"{memory_text}   per-round: " + " ".join(f"{value:.2f}" for value in per_round)
        )
        print(
            f"  gate: {'PASS' if passed else 'FAIL'} (parity={parity_passed}, "
            f"runtime={runtime_improved}, traced memory={memory_improved})"
        )

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nDetailed results written to {output}")
    if args.gate and not gate_passed:
        raise SystemExit("Improvement gate failed; do not promote this experiment.")


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
            "mRMR",
            "mRMR-MIQ",
            "Chi2",
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
            "mRMR",
            "mRMR-MIQ",
            "Chi2",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--backends", nargs="+", choices=["cpu", "gpu"], default=["cpu"])
    parser.add_argument("--backend", choices=["cpu", "gpu"], help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=900)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF)
    parser.add_argument("--json-output")
    parser.add_argument("--output-arrays", help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--states",
        type=int,
        default=4,
        help="Cardinality for information/Chi2 inputs; MDR always uses genotypes 0/1/2.",
    )
    parser.add_argument("--seed", type=int, default=20210721)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--rtol", type=float, default=2e-6)
    parser.add_argument("--atol", type=float, default=2e-7)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Fail unless every case preserves full outputs and improves "
        "runtime or traced memory by the margin in every round. Does not replace oracle tests.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.min_seconds) or args.min_seconds < 5.0:
        parser.error("--min-seconds must be at least 5.0")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if not args.worker and args.rounds < 5:
        parser.error("--rounds must be at least 5 for a reportable comparison")
    if min(args.samples, args.features, args.threads, args.states) < 1:
        parser.error("samples, features, threads, and states must be positive")
    if not 0 < args.min_improvement < 1:
        parser.error("--min-improvement must be between zero and one")
    if not math.isfinite(args.rtol) or not math.isfinite(args.atol) or args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be finite and non-negative")
    return args


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker:
        _worker(parsed)
    else:
        _parent(parsed)
