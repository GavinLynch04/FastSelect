"""Verify admission, pin measured sources, and write the external-baseline report."""

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/new_algorithms"
final = json.loads((AUDIT / "final_admission.json").read_text())
trial = json.loads((AUDIT / "sustained.json").read_text())
assert final["implementation"] == "production"
assert final["rounds"] >= 5
assert final["output_parity"]
prototype_hash = hashlib.sha256()
for path in sorted((AUDIT / "prototypes").glob("*.py")):
    prototype_hash.update(path.name.encode())
    prototype_hash.update(path.read_bytes())
assert prototype_hash.hexdigest() == trial["prototype_source_sha256"]
for name, expected in final["production_source_sha256"].items():
    assert hashlib.sha256((ROOT / "src/fast_select" / name).read_bytes()).hexdigest() == expected, name


def group_rows(result, case, backend):
    return sorted(
        (r for r in result["measurements"] if r["case"] == case and r["backend"] == backend), key=lambda r: r["round"]
    )


def verify_case(result, case, backend):
    baseline = group_rows(result, case, "reference")
    candidate = group_rows(result, case, backend)
    assert len(candidate) == result["rounds"]
    assert sum(r["measured_seconds"] for r in candidate) >= 5
    assert sum(r["measured_seconds"] for r in baseline) >= 5
    ratios = []
    for a, b in zip(baseline, candidate):
        assert a["round"] == b["round"]
        assert a["output"]["selected"] == b["output"]["selected"]
        for key in ("scores", "relevance_selected"):
            if key in a["output"]:
                np.testing.assert_allclose(b["output"][key], a["output"][key], rtol=2e-10, atol=2e-10)
        ratios.append(a["median_seconds"] / b["median_seconds"])
    return baseline, candidate, ratios


cases = list(dict.fromkeys(row["case"] for row in final["measurements"]))
decisions = {
    "criterion": "scores/selections agree and >=1.03x CPU speedup in every round on both shapes",
    "algorithms": {},
    "gpu_trials": {},
}
table = []
for case in cases:
    reference, candidate, ratios = verify_case(final, case, "cpu")
    algorithm = candidate[0]["algorithm"]
    assert all(ratio >= 1.03 for ratio in ratios), (case, ratios)
    a = statistics.median(row["median_seconds"] for row in reference)
    b = statistics.median(row["median_seconds"] for row in candidate)
    decision = {
        "shape": candidate[0]["shape"],
        "reference_ms": a * 1000,
        "cpu_ms": b * 1000,
        "central_speedup": a / b,
        "round_speedups": ratios,
        "admitted": True,
    }
    decisions["algorithms"].setdefault(algorithm, {})[case] = decision
    n, p = candidate[0]["shape"]
    table.append(
        f"| {algorithm} | {n:,} × {p} | {a*1000:.2f} | {b*1000:.2f} | {a/b:.2f}× | "
        f"{min(ratios):.2f}–{max(ratios):.2f}× |"
    )

gpu_table = []
for case in cases:
    metal = group_rows(trial, case, "metal")
    if not metal:
        continue
    verify_case(trial, case, "metal")
    cpu = group_rows(trial, case, "cpu")
    ratios = [a["median_seconds"] / b["median_seconds"] for a, b in zip(cpu, metal)]
    retain = all(ratio >= 1.03 for ratio in ratios)
    assert not retain, "Review and admit a GPU winner instead of discarding it"
    decisions["gpu_trials"][case] = {"cpu_over_metal_round_speedups": ratios, "retained": False}
    gpu_table.append(
        f"| {case} | {statistics.median(ratios):.2f}× | " f"{min(ratios):.2f}–{max(ratios):.2f}× | Not retained |"
    )

measured = sum(row["measured_seconds"] for row in final["measurements"])
trial_measured = sum(row["measured_seconds"] for row in trial["measurements"])
hashes = "\n".join(f"- `{name}`: `{value}`" for name, value in final["production_source_sha256"].items())
report = f"""# Five new algorithms: external-baseline admission

All five candidates earned a place in the development package as **parallel
Numba CPU implementations**. Each beat its external baseline by at least 3%
in every one of five rounds on both tested shapes, while preserving complete
reference API outputs and selected feature indices. Their experimental Metal
paths did not consistently beat these new CPU paths and remain outside the
distributed package. Continuous MI has no Metal prototype.

## Results on this M1 Pro

Apple M1 Pro, 8 CPU cores, 16 GiB RAM; macOS 27.0.1 arm64; Python 3.13.17,
NumPy 2.5.4, Numba 0.68.0, SciPy 1.18.1, scikit-learn 1.9.1, MLX 0.32.4.
Run started {final['timestamp_utc']}. Times are **complete calls in milliseconds**:
median of the five per-round medians. Central speedup divides these central
times; the final column gives the minimum and maximum paired round speedup.

| Algorithm | Samples × features | External ms | FastSelect CPU ms | Central speedup | Paired round range |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

These are measured workload results, not guarantees for every input or machine.
JMI/CMIM select ten features. FCBF selects the count implied by delta=0.
Categorical fixtures use three states per feature and a three-class target,
with noisy target-linked features among noise; continuous fixtures are Gaussian with a three-class
target. The exact seed and recipes live in the harness. Laplacian fits include
Euclidean binary 5-neighbor graph construction and scoring; MI calls include
scaling, jitter and all features. No binning, approximate neighbors, smaller k,
reduced candidate set or changed stopping rule supplies these gains.

## Why these five

- **Ross continuous/discrete MI** exposes independent feature jobs. Sorting
  one-dimensional samples replaces generic neighbor-estimator objects and
  permits exact float64 neighbor/count predicates in Numba.
- **JMI and CMIM** parallelize candidate entropy calculations at each greedy
  step, reuse marginal entropies and maintain accumulated objectives. The
  greedy selection sequence remains sequential.
- **FCBF** parallelizes relevance and redundancy calculations while retaining
  the strict relevance threshold and inclusive redundancy inequality.
- **Laplacian Score** uses a sparse edge-energy identity and parallel feature
  reductions. Partial neighbor selection avoids sorting whole distance rows;
  boundary ties fall back to the reference full-sort membership. It still
  constructs an n-by-n distance matrix and is not a solution to quadratic
  graph-construction memory.

## Popular baseline choice and provenance

[scikit-learn](https://github.com/scikit-learn/scikit-learn) supplies continuous
MI, version 1.9.1. Its baseline is tuned per shape over serial, four/eight
joblib threads and four/eight processes; the fastest measured option is used.
For 2,000 × 128 this is eight processes; for 4,000 × 256 it is four processes.
Process pools are warmed before timing. The public function and random seed
are identical in each comparison.

[Original scikit-feature](https://github.com/jundongl/scikit-feature) supplies
the other four methods, pinned commit
`48cffad4e88ff4b9d2f1c7baffb314d1b3303792`. Popularity is assessed by GitHub
stars among reviewed implementations: approximately 67.9k for scikit-learn,
1.6k for scikit-feature, 286 for [mifs](https://github.com/danielhomola/mifs),
and 148 for [ITMO_FS](https://github.com/ctlab/ITMO_FS). This is a transparent
popularity proxy, not a census of installations.

The current `skfeature-chappers==1.2.1` fork was inspected, but its JMI wrapper
does not activate the normalized JMI coefficients and its FCBF truncates SU
scores to integers. The original upstream implementations are used instead.
Original FCBF's nonfunctional `dtypes='object'` keyword is changed mechanically
to `dtype='object'` at import time; no formula or loop is changed. Reference
GPL source is fetched into an ignored audit directory and is not copied into
the runtime package. The new implementations follow paper equations.

Baseline identity and archive hash: [manifest](../../audit/new_algorithms/baseline_manifest.json).
Installed versions and reference Python/KDTree source hashes:
[environment](../../audit/new_algorithms/environment.json).
Tuning measurements: [MI](../../audit/new_algorithms/mi_baseline_tuning.json),
[Laplacian BLAS](../../audit/new_algorithms/lap_baseline_tuning.json).
BLAS limits are the same for paired Laplacian calls, chosen from 1/2/4/8
threads. Numba uses eight threads; original scikit-feature has no parallel API.

## Metal decision

The first complete CPU/Metal prototype matrix ran before package admission.
Each row below compares Metal against the **new Numba CPU implementation**,
not v0.2.1 or an unoptimized serial baseline. Values above 1 favor Metal.
No case passed a 1.03× runtime gain in all five rounds. Sporadic GPU wins did
not establish a sustained improvement, so no new GPU backend is shipped or
auto-selected for these five additions.

| Prototype case | Median CPU/Metal ratio | Paired round range | Decision |
|---|---:|---:|---|
{chr(10).join(gpu_table)}

The Metal trial used exact integer GPU joint counts with float64 host entropy
for JMI/CMIM/FCBF, and a sparse high/low float32 edge-energy kernel plus host
float64 denominator/ambiguity repair for Laplacian Score. Precision and
transfer overhead limited gains. These remain reproducible experiments under
`audit/new_algorithms/prototypes`, outside setuptools' `src` package search.

## Correctness and API

The [independent tests](../../tests/test_new_algorithms.py) use empirical
Counter-based entropy/joint/conditional-MI identities, dense graph quadratic
forms, and direct pairwise class-neighbor distances for Ross's equation. They
cover constant features, duplicate samples, singleton/imbalanced classes,
multiclass and remapped labels, large categorical codes, strict delta equality,
inclusive SU redundancy, graph ties, zero weighted variance, translation by
1e8, mixed continuous/discrete MI, estimator contracts and Pipeline/GridSearchCV.
The complete suite passed: **784 passed, 76 CUDA-only skipped**. Wheel and
source-distribution builds passed; an extracted-wheel import/fit smoke check
also passed with MLX and scikit-feature blocked. Neither distribution includes
the external reference or experimental GPU source. The documentation builds
without warnings. See [validation record](../../audit/new_algorithms/validation.json).

Two numerical/scope distinctions are explicit. A constant column with small
imbalanced classes can trigger distance cancellation in scikit-learn's
automatic brute-force neighbor path; the new MI agrees with the direct Ross
oracle there. Laplacian zero weighted-variance scores are undefined and are
represented as +infinity/ranked last, rather than the reference sentinel.
Those cases are judged by the independent oracle, not included in timing
parity fixtures. Large common offsets are recentered before Euclidean graph
distance evaluation. The ordinary benchmark graphs and full outputs agree.

JMI/CMIM/FCBF accept categorical classification inputs without automatic
binning. CMIM's arbitrary categorical domain is a stated extension of its
original binary formulation. Laplacian Score is unsupervised and supports a
supplied symmetric nonnegative affinity. Continuous MI is dense-only,
numeric-input/classification-only, returns nats and supports SelectKBest. The four estimators
accept CPU/auto and expose `effective_backend_='cpu'`; explicit GPU requests
raise. Existing algorithms' backend selection is unchanged.

## Measurement and reproduction

The final gate comprises {len(final['measurements'])} fresh workers and
{measured:.1f} measured seconds. The archived prototype matrix comprises
{len(trial['measurements'])} fresh workers and {trial_measured:.1f} measured
seconds. Five rounds alternate backend order. Each worker warms compilation
and a complete call, then measures at least 1.05 seconds; every case/backend
therefore totals at least 5.25 seconds. Slow reference calls run longer.
Compilation/imports are excluded; validation, preprocessing, allocation,
graph building, device transfer and result materialization are included.
No benchmark or test workload was intentionally run concurrently.

Raw final data (individual durations, iterations, outputs, selected indices,
thread settings): [final_admission.json](../../audit/new_algorithms/final_admission.json).
GPU experiment: [sustained.json](../../audit/new_algorithms/sustained.json).
Memory is macOS peak process RSS including imports/warm-up. It excludes
separate live joblib workers and does not isolate GPU allocation; no memory
improvement is claimed. This is warmed-call throughput, not first-call latency.

Frozen final package SHA256 hashes:

{hashes}

```sh
.venv/bin/python benchmarking/fetch_new_algorithm_baselines.py
.venv/bin/python benchmarking/benchmark_new_algorithms.py --production --cpu-only \\
  --rounds 5 --seconds 1.05 --output audit/new_algorithms/final_admission.json
.venv/bin/python benchmarking/report_new_algorithms.py
```

The report generator rechecks all score/selection outputs, cumulative timing,
per-round admission gains and current package source hashes. See
[audit notes and papers](../../audit/new_algorithms/README.md) for primary
equations, limitations, baseline shim details and prototype reproduction.
"""
(AUDIT / "admission_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
(ROOT / "benchmarking/results/NEW_ALGORITHMS.md").write_text(report)
print("All five CPU algorithms admitted; GPU prototypes rejected; report written.")
