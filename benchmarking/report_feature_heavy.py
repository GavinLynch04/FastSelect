"""Verify feature-heavy experiment and summarize physical Metal/CUDA status."""

import hashlib
import json
import statistics
from pathlib import Path

from benchmark_feature_heavy import ALGORITHMS, SHAPES, compare

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/feature_heavy"
result = json.loads((AUDIT / "sustained.json").read_text())
assert result["rounds"] >= 5 and not result["screen"] and result["output_parity"]
for name, sha in result["cpu_manifest"]["cpu_sha256"].items():
    assert hashlib.sha256((ROOT / "src/fast_select" / name).read_bytes()).hexdigest() == sha, name
for name, sha in result["experimental_source_sha256"].items():
    assert hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() == sha, name
rows = result["measurements"]
summary = {}
table = []
for shape, (n, p) in SHAPES.items():
    for name in ALGORITHMS:
        groups = {
            backend: sorted(
                (r for r in rows if r["shape_name"] == shape and r["algorithm"] == name and r["backend"] == backend),
                key=lambda r: r["round"],
            )
            for backend in ["cpu", "metal"]
        }
        for group in groups.values():
            assert len(group) == 5
            assert sum(r["measured_seconds"] for r in group) >= 5
        ratios = []
        for cpu, metal in zip(groups["cpu"], groups["metal"]):
            compare(cpu, metal)
            ratios.append(cpu["median_seconds"] / metal["median_seconds"])
        a = statistics.median(r["median_seconds"] for r in groups["cpu"])
        b = statistics.median(r["median_seconds"] for r in groups["metal"])
        stable = all(r >= 1.03 for r in ratios)
        label = "gain in every round" if stable else "not a consistent gain"
        summary[f"{shape}_{name}"] = {
            "shape": [n, p],
            "algorithm": name,
            "cpu_ms": a * 1000,
            "metal_ms": b * 1000,
            "central_speedup": a / b,
            "paired_speedups": ratios,
            "passed_runtime_gate": stable,
        }
        table.append(
            f"| {n} × {p:,} | {name} | {a*1000:.2f} | {b*1000:.2f} | {a/b:.2f}× | "
            f"{min(ratios):.3f}–{max(ratios):.3f}× | {label} |"
        )
validation_path = AUDIT / "validation.json"
validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
measured = sum(r["measured_seconds"] for r in rows)
hashes = "\n".join(f"- `{name}`: `{sha}`" for name, sha in result["experimental_source_sha256"].items())
text = f"""# Feature-heavy inputs: retained CPU versus physical Metal, with CUDA preparation

The earlier five-algorithm admission reports measured **more samples than
features**. They did not establish results for p > n, and the second batch did
not test GPU implementations. This experiment closes that coverage gap for
the latest five algorithms: regression MI, GiniIndex, DISR, CIFE and MIFS.

## Results

Apple M1 Pro, 8 CPU cores, 16 GiB RAM; physical Apple Metal GPU through MLX.
Run started {result['timestamp_utc']}. The baseline is the **retained optimized
Numba CPU implementation**, frozen before the experiment, with eight threads.
It is not v0.2.1 or an older serial reference package. Earlier popular-package
comparisons remain in the admission reports; those external references were
not remeasured by this GPU-focused experiment.

Times below are milliseconds, medians of five per-round medians. A ratio >1
favors Metal. The paired range gives same-round minimum/maximum speedup. The
strict runtime gate requires >=1.03× in every round of a case.

**Load caveat:** later CPU-thread retests became unstable while another
CPU-intensive job saturated this shared desktop (observed load average 88.6
on eight cores). Those retests are excluded; they are not used to inflate GPU
ratios. The table retains the original complete eight-thread paired run.
A complete system-load history was not captured for that run, so these are
observed desktop prototype ratios, pending clean thread-count confirmation
before package integration or automatic selection.

| Samples × features | Algorithm | Numba CPU ms | Metal prototype ms | Central ratio | Paired range | Consistency |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(table)}

The shapes have 16–32 times as many features as samples. Results are
workload-specific; p > n alone does not establish a GPU win. The small case
also exposes GPU overhead rather than hiding it. MIFS has less per-step work
than CIFE/DISR, since it needs pair counts without a conditional/triple term.
The k-step greedy selection remains sequential even when features are processed
in parallel.

These timings compare actual public CPU fits/score calls with equivalent
end-to-end experimental GPU pipelines. GPU timing includes validation, category
encoding or scale/jitter, allocations, GPU input/output, float64 host
reductions, exact boundary repairs, ranking and result materialization. The
GPU research wrappers do not implement the public estimators' metadata/clone
interface; these are **experimental backends**, not newly enabled package
backends. Small estimator metadata differences are not isolated by this
experiment. No automatic GPU-selection change is made on the basis of this
prototype benchmark alone.

## What was tested on Metal

- **Gini:** exact order-preserving 64-bit keys, parallel bitonic sorting across
  features, then the same integer-count and float64 impurity reduction on CPU.
  Large uint64 codes, signed values, float64 near-ties, padding and duplicate
  values preserve numeric ordering. No histogram binning or float32 narrowing
  of numeric thresholds is used. Current sort cap: 1,024 samples.
- **DISR/CIFE/MIFS:** GPU shared-memory integer joint histograms record both
  counts and first occurrence. Counts are returned in the original traversal
  order so CPU float64 entropy reductions preserve search ties. No GPU
  logarithm approximation changes the criterion. Current prototype cap: eight
  encoded states per variable and eight target classes.
- **Regression MI:** float32 GPU searches propose k+2 neighbors, with a
  conservative error bound for coordinate quantization and subtraction. CPU
  float64 recomputation certifies the true kth distance against every omitted
  candidate's lower bound; uncertified queries fall back to an exhaustive
  float64 search. Strict marginal radius/count predicates and final digamma
  scores stay float64. This does not change k or use approximate neighbors.
  k<=16 and finite float32 proposal coordinates are required by this prototype.

All paths preserve the defining equations, not just scalar timing checks.
Every paired worker compares full scores and selected feature indices;
discrete selectors additionally compare all marginal relevance and the
selected relevance array. The same seed 7931 and fixture recipes from the
second admission batch are used at the new shapes. Regression MI uses k=3;
selectors choose ten features. Gini's 1e-4 fixture spacing keeps its numeric
ordering well-defined, while separate oracle tests exercise much closer
float64 boundaries and large integer identities.

## CUDA: prepared locally, not performance-tested

The Mac reports no CUDA device, `nvidia-smi` or `nvcc`. The user confirmed no
NVIDIA host is available and requested local preparation. A separate CUDA
provider now implements the same exact-key shared-memory sort and ordered
integer histograms, plus native float64 KSG joint-distance search (k<=32).
The provider is outside the installable package, alongside the Metal research
code.

**12 physical Metal oracle tests and 12 CUDA simulator oracle tests passed.**
They cover exhaustive Gini split equations, random and exact-tie categorical
criteria, multiclass/remapped labels, constants, large integer codes, float64
near-ties, strict KSG radii, ambiguous Metal proposal repairs and mixed
discrete/continuous regression-MI columns. CUDA simulation validates kernel
logic only; native PTX compilation, device races, real-device precision and
throughput remain unverified. It is not evidence of a CUDA speedup.
[Numba's simulator documentation](https://numba.readthedocs.io/en/stable/cuda/simulator.html)
explicitly states that meaningful GPU timing and type inference are unavailable
there. The benchmark refuses to produce CUDA timings under the simulator.

An actual NVIDIA measurement needs a compatible GPU, driver and CUDA
development environment, as described in the
[NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
On that host, tune CPU threads for the machine and run the same full comparison;
do not assume the M1/Metal ratios transfer to another GPU. No expected CUDA
speedup is invented here.

## Method, identities and reproduction

There are {len(rows)} fresh-process workers and {measured:.1f} measured seconds.
Five rounds alternate backend order. Each case/backend/round warms a complete
call and compilation first, then measures at least 1.05 seconds, totaling at
least 5.25 seconds per case/backend. No concurrent test/benchmark workload was
intentionally run. BLAS is limited to one thread; Numba uses eight. Metal
arrays are materialized on the host before timing ends, so asynchronous GPU
launches are not mistaken for completed work. Peak process RSS includes
imports/warm-up and is not isolated GPU allocation; no memory gain is claimed.

CPU sources were copied and hashed before any prototype edits. The package
sources are unchanged and are rechecked against that snapshot by the harness
and this report generator. Source and test evidence:

- [CPU manifest](../../audit/feature_heavy/manifest.json)
- [Raw sustained data and full outputs](../../audit/feature_heavy/sustained.json)
- [Summary](../../audit/feature_heavy/summary.json)
- [Exploratory fresh-process thread tuning](../../audit/feature_heavy/cpu_thread_tuning_fresh.json)
- [Excluded loaded thread retests](../../audit/feature_heavy/retuned_small.json)
- [Oracle tests](../../audit/feature_heavy/test_prototypes.py)
- [Validation record](../../audit/feature_heavy/validation.json)
- [Research notes and CUDA commands](../../audit/feature_heavy/README.md)

Frozen experimental SHA256 hashes:

{hashes}

```sh
# Physical Apple GPU:
.venv/bin/python benchmarking/benchmark_feature_heavy.py --backends cpu metal

# CUDA logic checks on this Mac (no timing claim):
NUMBA_ENABLE_CUDASIM=1 FEATURE_GPU_TEST_BACKEND=cuda \\
  .venv/bin/python -m pytest audit/feature_heavy/test_prototypes.py --no-cov

# On an NVIDIA host with CUDA installed; --threads can be tuned for that CPU:
.venv/bin/python benchmarking/benchmark_feature_heavy.py --backends cpu cuda \\
  --threads 8 --rounds 5 --seconds 1.05 --output audit/feature_heavy/cuda_hardware.json
```
"""
(AUDIT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(ROOT / "benchmarking/results/FEATURE_HEAVY_GPU.md").write_text(text)
print("Feature-heavy report verified and written.")
