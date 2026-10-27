"""Sustained full-fit Metal comparison against frozen, optimized CPU sources.

Five fresh-process rounds, alternating backend order, >=5 timed seconds each.
Full numerical outputs are retained in compressed NPZ files and compared before
reporting. This includes regressions: availability-based auto is not a speed gate.
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
    "SURF": (900, 128, "discrete"),
    "SURFStar": (900, 128, "discrete"),
    "MultiSURF": (900, 128, "discrete"),
    "MultiSURFStar": (900, 128, "discrete"),
    "MultiSURFContinuous": (900, 128, "continuous"),
    "MutualInformation": (2000, 512, "discrete"),
    "mRMR": (2000, 512, "discrete"),
    "mRMR-MIQ": (2000, 512, "discrete"),
    "CFS": (2000, 512, "discrete"),
    "MDR": (128, 512, "genotype"),
    "Chi2": (12000, 512, "counts"),
    "TuRF": (512, 128, "discrete"),
    # Feature-heavy (p > n) variants of the cases where Metal lost on tall data.
    "MultiSURFContinuousWide": (256, 4096, "continuous"),
    "CFSWide": (256, 2048, "discrete"),
    "Chi2Wide": (256, 8192, "counts"),
}


def fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted((root / "src/fast_select").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(args):
    import inspect

    sys.path.insert(0, str(Path(args.source_root) / "src"))
    import numpy as np

    from fast_select import CFS, MDR, SURF, MultiSURF, ReliefF, TuRF, calculate_mi_matrices, chi2, mRMR

    n, p, kind = CASES[args.case]
    rng = np.random.default_rng(20261026)
    if kind == "continuous":
        x = rng.standard_normal((n, p), dtype=np.float32)
    elif kind == "counts":
        x = rng.uniform(0, 10, (n, p))
    else:
        x = rng.integers(0, 3, (n, p), dtype=np.int32)
    y = np.arange(n, dtype=np.int32) % 2
    rng.shuffle(y)
    kwargs = dict(backend=args.backend)
    if args.case.startswith(("SURF", "MultiSURF")):
        C = SURF if args.case.startswith("SURF") else MultiSURF
        model = C(n_features_to_select=10, n_jobs=8, discrete_limit=3, use_star="Star" in args.case, **kwargs)
    elif args.case.startswith("mRMR"):
        model = mRMR(10, method="MIQ" if args.case.endswith("MIQ") else "MID", **kwargs)
    elif args.case.startswith("CFS"):
        model = CFS(n_jobs=8, **kwargs)
    elif args.case == "MDR":
        model = MDR(k=2, cv=3, **kwargs)
    elif args.case == "TuRF":
        model = TuRF(
            ReliefF(n_neighbors=10, n_jobs=8, **kwargs), n_features_to_select=10, pct_remove=0.5, n_iterations=2
        )

    def run():
        if args.case == "MutualInformation":
            return calculate_mi_matrices(x, y, backend=args.backend)
        if args.case.startswith("Chi2"):
            return chi2(x, y, **kwargs) if "backend" in inspect.signature(chi2).parameters else chi2(x, y)
        return model.fit(x, y)

    start = time.perf_counter()
    result = run()
    first = time.perf_counter() - start
    run()
    run()
    timings = []
    total = 0.0
    while total < args.seconds:
        start = time.perf_counter()
        result = run()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        total += elapsed
    arrays = {}
    if args.case in ("MutualInformation", "Chi2", "Chi2Wide"):
        arrays["first"], arrays["second"] = result
    elif args.case.startswith("CFS"):
        arrays.update(selected=result.selected_indices_, merit=np.array([result.merit_]))
    elif args.case == "MDR":
        arrays.update(
            selected=np.array(result.best_interaction_),
            lookup=result.best_model_lookup_table_,
            predictions=result.predict(x),
            cvc=np.array([result.best_cvc_]),
            testing_ba=np.array([result.best_mean_testing_ba_]),
        )
    else:
        arrays.update(selected=result.top_features_, scores=result.feature_importances_)
        if args.case.startswith("mRMR"):
            arrays.update(relevance=result.relevance_scores_, redundancy=result.redundancy_matrix_)
    np.savez_compressed(args.arrays, **arrays)
    result_info = dict(
        case=args.case,
        backend=args.backend,
        samples=n,
        features=p,
        kind=kind,
        seed=20261026,
        dtype=str(x.dtype),
        iterations=len(timings),
        measured_seconds=total,
        timings_seconds=timings,
        mean_seconds=statistics.fmean(timings),
        first_fit_seconds=first,
        source_sha256=fingerprint(Path(args.source_root)),
        effective_backend=getattr(result, "effective_backend_", args.backend),
        diagnostics=getattr(result, "metal_diagnostics_", None),
        platform=platform.platform(),
        python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in ["numpy", "numba", "mlx", "scikit-learn"]},
        memory_method="Peak memory not measured",
    )
    print("RESULT_JSON=" + json.dumps(result_info), flush=True)


def main(args):
    import numpy as np

    path = Path(args.output).resolve()
    folder = path.parent / "metal_all_arrays"
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(
        method="Five isolated-process rounds, alternating order, three warmups, synchronized complete fits",
        baseline_root=str(Path(args.baseline_root).resolve()),
        baseline_sha256=fingerprint(Path(args.baseline_root)),
        current_sha256=fingerprint(ROOT),
        threads=8,
        rounds=args.rounds,
        min_seconds=args.seconds,
        workers=[],
        comparisons=[],
        complete=False,
    )
    env = dict(
        os.environ,
        NUMBA_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env.pop("NUMBA_ENABLE_CUDASIM", None)
    for round_index in range(args.rounds):
        for case in args.cases:
            pair = {}
            outputs = {}
            for backend in (("cpu", "metal") if round_index % 2 == 0 else ("metal", "cpu")):
                arrays = folder / f"{case}_{backend}_{round_index+1}.npz"
                source = args.baseline_root if backend == "cpu" else str(ROOT)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--backend",
                    backend,
                    "--source-root",
                    source,
                    "--arrays",
                    str(arrays),
                    "--seconds",
                    str(args.seconds),
                ]
                completed = subprocess.run(command, env=env, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(completed.stdout + completed.stderr)
                result = json.loads(
                    next(s.split("=", 1)[1] for s in completed.stdout.splitlines() if s.startswith("RESULT_JSON="))
                )
                result.update(round=round_index + 1, arrays=str(arrays.relative_to(path.parent)))
                report["workers"].append(result)
                pair[backend] = result
                with np.load(arrays) as data:
                    outputs[backend] = {k: data[k] for k in data.files}
                print(f"round {round_index+1} {case} {backend}: {result['mean_seconds']*1000:.3f} ms", flush=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
            a, b = outputs["cpu"], outputs["metal"]
            tolerance = 2e-10 if case.startswith("Chi2") else 2e-6
            exact = True
            close = a.keys() == b.keys()
            error = 0.0
            for key in a:
                if key not in b or a[key].shape != b[key].shape:
                    close = False
                    continue
                if not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
                    close = False
                    continue
                error = max(error, float(np.max(np.abs(a[key].astype(float) - b[key]))) if a[key].size else 0)
                if a[key].dtype.kind in "bui":
                    exact = exact and np.array_equal(a[key], b[key])
                else:
                    close = close and np.allclose(a[key], b[key], rtol=1e-5, atol=tolerance)
            comparison = dict(
                case=case,
                round=round_index + 1,
                full_output_parity=bool(close and exact),
                max_abs_error=error,
                rtol=1e-5,
                atol=tolerance,
                speedup=pair["cpu"]["mean_seconds"] / pair["metal"]["mean_seconds"],
            )
            report["comparisons"].append(comparison)
            path.write_text(json.dumps(report, indent=2) + "\n")
            if not comparison["full_output_parity"]:
                raise RuntimeError(f"Full-output comparison failed: {comparison}")
    report["summaries"] = []
    for case in args.cases:
        times = {
            b: statistics.median(
                w["mean_seconds"] for w in report["workers"] if w["case"] == case and w["backend"] == b
            )
            for b in ("cpu", "metal")
        }
        ratios = [c["speedup"] for c in report["comparisons"] if c["case"] == case]
        report["summaries"].append(
            dict(
                case=case,
                cpu_ms=times["cpu"] * 1000,
                metal_ms=times["metal"] * 1000,
                speedup=times["cpu"] / times["metal"],
                round_range=[min(ratios), max(ratios)],
                runtime_gate=all(v >= 1.03 for v in ratios),
            )
        )
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--backend", choices=["cpu", "metal"])
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--baseline-root", default=str(ROOT / "audit/metal_all/baseline"))
    parser.add_argument("--arrays")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "benchmarking/results/m1pro_all_metal.json"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.rounds < 5 or args.seconds < 5:
        parser.error("Sustained comparison requires at least five rounds and five measured seconds per worker")
    else:
        main(args)
