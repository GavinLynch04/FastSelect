# Feature-heavy inputs: retained CPU versus physical Metal, with CUDA preparation

The earlier five-algorithm admission reports measured **more samples than
features**. They did not establish results for p > n, and the second batch did
not test GPU implementations. This experiment closes that coverage gap for
the latest five algorithms: regression MI, GiniIndex, DISR, CIFE and MIFS.

## Results

Apple M1 Pro, 8 CPU cores, 16 GiB RAM; physical Apple Metal GPU through MLX.
Run started 2026-10-27T17:01:08Z. The baseline is the **retained optimized
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
| 64 × 1,024 | mutual_info_regression | 6.81 | 5.68 | 1.20× | 1.098–1.574× | gain in every round |
| 64 × 1,024 | GiniIndex | 1.08 | 1.02 | 1.06× | 1.025–1.127× | not a consistent gain |
| 64 × 1,024 | DISR | 4.45 | 5.14 | 0.87× | 0.844–1.070× | not a consistent gain |
| 64 × 1,024 | CIFE | 5.32 | 5.31 | 1.00× | 0.858–1.042× | not a consistent gain |
| 64 × 1,024 | MIFS | 3.89 | 4.77 | 0.81× | 0.738–0.870× | not a consistent gain |
| 128 × 4,096 | mutual_info_regression | 57.28 | 41.09 | 1.39× | 1.373–1.553× | gain in every round |
| 128 × 4,096 | GiniIndex | 6.36 | 4.28 | 1.49× | 1.357–1.564× | gain in every round |
| 128 × 4,096 | DISR | 15.58 | 11.80 | 1.32× | 0.948–1.508× | not a consistent gain |
| 128 × 4,096 | CIFE | 15.14 | 12.47 | 1.21× | 1.132–1.372× | gain in every round |
| 128 × 4,096 | MIFS | 10.67 | 10.39 | 1.03× | 0.943–1.564× | not a consistent gain |
| 256 × 8,192 | mutual_info_regression | 264.74 | 168.51 | 1.57× | 1.363–1.639× | gain in every round |
| 256 × 8,192 | GiniIndex | 25.99 | 15.59 | 1.67× | 1.561–1.979× | gain in every round |
| 256 × 8,192 | DISR | 42.77 | 27.91 | 1.53× | 1.281–1.788× | gain in every round |
| 256 × 8,192 | CIFE | 49.12 | 27.98 | 1.76× | 1.424–3.377× | gain in every round |
| 256 × 8,192 | MIFS | 28.41 | 25.98 | 1.09× | 1.019–1.209× | not a consistent gain |

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

There are 150 fresh-process workers and 160.2 measured seconds.
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

- `common.py`: `fc7750a9ea3e382fb14c3f87d5655af78106c9ba92da6b604ae85be79ddaae6b`
- `metal.py`: `9e3c533274d9e1242da96d71eed478fcaafdb7338fce85f6baf912e478907ca6`
- `cuda.py`: `e64fb72a2f9cf79c0a2d46f28778221d5e696ce9d4ef135e2356f71ed8cb6b8c`

```sh
# Physical Apple GPU:
.venv/bin/python benchmarking/benchmark_feature_heavy.py --backends cpu metal

# CUDA logic checks on this Mac (no timing claim):
NUMBA_ENABLE_CUDASIM=1 FEATURE_GPU_TEST_BACKEND=cuda \
  .venv/bin/python -m pytest audit/feature_heavy/test_prototypes.py --no-cov

# On an NVIDIA host with CUDA installed; --threads can be tuned for that CPU:
.venv/bin/python benchmarking/benchmark_feature_heavy.py --backends cpu cuda \
  --threads 8 --rounds 5 --seconds 1.05 --output audit/feature_heavy/cuda_hardware.json
```
