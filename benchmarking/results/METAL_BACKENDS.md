# Metal backends and automatic selection

Metal is supported for every Relief-family and information selector/statistic: ReliefF,
SURF/SURF*, MultiSURF/MultiSURF*, MI, mRMR MID/MIQ, CFS, MDR and Chi2. TuRF
refits its configured base selector at every iteration and exposes the last
fitted clone as `estimator_` and its backend as `effective_backend_`.

`auto` prefers a supported CUDA backend, then optional MLX/Metal, then CPU.
`gpu` retains its explicit CUDA meaning. `metal` explicitly requests Metal and
raises on unavailable/unsupported input. Missing MLX does not break imports or
CPU execution. Chi2 and CFS are the exceptions: their Metal paths failed the
sustained runtime gate on both tall and feature-heavy inputs (see *Feature-heavy
follow-up* below), so `auto` runs them on CPU and Metal is explicit opt-in.
Elsewhere routing uses capability and availability; it does not predict which
device is faster.

Install the optional dependency:

```bash
pip install "fast-select[metal]"
```

Metal requires Python 3.10+, a supported Apple Silicon Mac, and physical GPU
access. ReliefF currently supports k <= 64; MI/mRMR/CFS support <=64 encoded
state slots (CUDA <=32). `auto` skips unsupported backends and uses CPU when
necessary. Explicit unsupported requests raise instead of silently falling back.

## Equations, precision, and host/device placement

- **ReliefF:** Kononenko (1994), unchanged nearest-hit/per-class-miss equations,
  class priors and actual-count normalization. Existing tiled Metal kernels and
  ambiguous-neighbor CPU repair are retained.
- **SURF/SURF*:** Moore et al. / Urbanowicz et al. (2018), the global radius is
  still computed by the existing CPU float64 method. Metal estimates tiled
  pair distances. A conservative error interval certifies unambiguous strict
  comparisons; CPU float64 recomputes uncertain pairs, including mathematical
  ties. Metal scores the fixed hit/miss groups, reversing far difference updates
  in SURF*. Normalized float64 coordinates are split into high/low float32 parts
  for GPU scoring. There is no change to `BOUNDARY_RTOL` or radius membership.
- **MultiSURF/MultiSURF*:** target-specific mean +/- half standard deviation;
  uncertain target boundaries are recomputed with CPU float64 accumulation and
  the existing float32 comparison field. Separate hit/miss normalization is
  preserved. Far MultiSURF* scores similarity, not differences.
- **TuRF:** complete refitting in each reduced feature space; no reused scores
  from an earlier feature space. Device support comes from the base selector.
- **Shannon MI:** GPU joint counts are exact integers. Bounded one-hot matrix
  products count 0/1 values exactly for <=2**24 observations; larger allocations
  use batched integer histogram kernels. Float64 host MI preserves bit/nat units.
  The one-hot input and Gram matrix budgets are 128 MiB and 64 MiB respectively;
  fallback histogram batches target 8 MiB.
- **Peng et al. (2005) mRMR / Hall (1999) CFS:** use GPU counts, then preserve
  host precision and the existing MID/MIQ or best-first search and tie rules.
  CFS retains the CPU float32 empirical probabilities and entropy calculation.
- **MDR:** Ritchie et al. (2001), with the package's documented balanced-accuracy
  and CVC variant. GPU contingency counts and integer cross-products classify
  risk cells without rounded ratios. Host float64 balanced-accuracy arithmetic
  is narrowed to the existing float32 training score before selecting winners.
  Combination batches are lexicographic, at most 65,536, and uploaded fold data
  are reused; strict updates preserve first-winner ties across batches. CV,
  testing, model selection, and final lookup/prediction remain on CPU. The
  existing k <=6 ternary-genotype/binary-label contract is unchanged.
- **Pearson Chi2:** scaled high/low float32 coordinates and GPU compensated
  class sums return both parts to a float64 host reduction. The original
  expected-count formula, degrees of freedom, and SciPy p-values remain on CPU.

SURF-family score partials target a 64 MiB buffer through adaptive query batches.
These are individual allocation limits, not measurements of total process/GPU
peak memory. All Metal helpers explicitly select a GPU stream even if an
application has set MLX's default device to CPU.

## Correctness

The complete suite passes **736 tests, with 76 CUDA-only tests skipped** on the
M1 Pro. Added tests cover CUDA/Metal/CPU precedence, unsupported capabilities,
missing devices, exact rational SURF boundaries, mixed/shifted data, MultiSURF*
far similarity and feature subsets, empirical MI units through 64 states,
the integer histogram allocation fallback, CFS search, MDR k=1/2/3/6 and ties
across batch boundaries, non-0/1 labels, Chi2 scale extremes, Pipeline,
GridSearchCV, and TuRF across all Relief base selectors. The previous physical
ReliefF oracle suite is retained. No physical CUDA parity is claimed on this Mac.

The complete suite also passes with MLX imports deliberately blocked (physical
Metal tests skip), verifying optional-dependency CPU fallback. Four focused
CUDA simulator tests pass for MI, mRMR, CFS and MDR host/kernel integration.
The broader simulator run passed its initial Relief cases and was interrupted
because Python-threaded simulation was slow; no full simulator-suite result is
claimed. Simulation is not physical NVIDIA validation.

Ruff, Black, and `git diff --check` pass. Source and wheel distributions build;
the wheel contains every Metal helper and the optional dependency metadata.
An extracted-wheel smoke test passes with MLX blocked and checks that explicit
Metal requests fail clearly when unavailable.

## Benchmark method

The exact optimized CPU working sources were frozen **before these changes** in
`audit/metal_all/baseline/src/fast_select` (including prior local improvements).
Baseline fingerprint:
`154863aca34a39448efefc76eb19e11e650340bbda5c9b26d9a8bb7563f3801d`.
This is not a v0.2.1 comparison. The frozen source and provenance are retained.

Measurements use this Mac's physical Apple M1 Pro GPU, eight CPU threads,
16 GiB RAM, native arm64 macOS 27.0.1, Python 3.13.17 and MLX 0.32.4. Every
case/backend/round runs in a fresh process, uses three warmups, and measures
at least five cumulative seconds of complete fits (or complete statistic calls).
Five rounds alternate CPU/Metal order; no other tests or benchmarks run
concurrently. Latency is the median of per-round mean latencies.

Full score arrays, relevance/redundancy matrices, p-values, selected indices,
and MDR lookup/CV/prediction outputs as applicable are stored in compressed
NPZ files and compared before results are accepted. Floating tolerance is
`rtol=1e-5, atol=2e-6` (Chi2 absolute tolerance 2e-10); integer selections and
predictions must match exactly. Independent oracles remain a separate gate.
The run stops on output disagreement. No peak-memory improvement is claimed.

Inputs are seeded (`20261026`). Threshold selectors use 900 ×128 ternary data,
plus a continuous normal-data MultiSURF control. MI/mRMR/CFS use 2,000 ×512
ternary features. MDR uses 128 ×512 ternary features, k=2 and 3-fold CV. Chi2
uses 12,000 ×512 uniform nonnegative float64 values. All labels are balanced
binary. TuRF uses 512 ×128 ternary features, a ReliefF k=10 base, 50% removal
and two removal/refit iterations; that explicit iteration cap leaves 32 active
features. These synthetic cases do not establish performance on every dataset.

## Measured results

| Case | Optimized CPU (ms) | Metal (ms) | CPU time / Metal time | Per-round range | Consistent runtime gain |
|---|---:|---:|---:|---:|---|
| SURF, discrete | 71.643 | 12.278 | 5.84× | 3.41–6.34× | Yes |
| SURF*, discrete | 142.084 | 10.532 | 13.49× | 10.23–13.49× | Yes |
| MultiSURF, discrete | 53.822 | 10.756 | 5.00× | 4.47–9.70× | Yes |
| MultiSURF*, discrete | 71.768 | 10.781 | 6.66× | 5.66–7.62× | Yes |
| MultiSURF, continuous | 6.999 | 20.361 | 0.34× | 0.34–0.83× | No |
| Mutual information | 147.978 | 30.324 | 4.88× | 3.89–7.58× | Yes |
| mRMR MID | 146.130 | 41.788 | 3.50× | 2.84–4.68× | Yes |
| mRMR MIQ | 146.715 | 30.867 | 4.75× | 2.40–5.02× | Yes |
| CFS | 263.874 | 203.857 | 1.29× | 0.89–1.36× | No |
| MDR k=2, CV=3 | 54.454 | 32.202 | 1.69× | 1.50–1.78× | Yes |
| Chi2, float64 counts | 8.133 | 23.089 | 0.35× | 0.28–0.43× | No |
| TuRF / ReliefF, 2 iterations | 17.681 | 5.230 | 3.38× | 2.28–4.10× | Yes |

All **60 paired comparisons** passed complete-output checks, including exact integer selections/predictions. The 120 workers accumulated **605.1 seconds of measured calls**. Nine of twelve cases improved runtime by >=3% in every round.

**Regressions are retained and visible.** Continuous MultiSURF was about 2.91× slower and Chi2 about 2.84× slower by median latency. CFS had a 1.29× median speedup but its final round was slower (0.89×), so it failed the consistency gate. It is not a demonstrated across-round optimization win. Automatic routing now keeps Chi2 and CFS on CPU (see the feature-heavy follow-up); `backend="cpu"` overrides Metal elsewhere.

Absolute timings varied substantially across rounds on this interactive Mac. Backend order alternated and no other tests or benchmarks were run concurrently by this experiment; system load and thermals were not controlled. The per-round ranges above are essential context. These are synthetic-dataset results, not universal speed guarantees.

Current measured source fingerprint: `7aa06fd31b01ac00fd3e9ffc980092e7d7d8d30ab6daee8af147984706b5c50d`.

Raw timing/first-fit samples and comparisons: [`m1pro_all_metal.json`](m1pro_all_metal.json). Full NPZ outputs are retained in `metal_all_arrays/`. Peak memory was not measured.


```bash
python benchmarking/benchmark_all_metal.py \
  --baseline-root audit/metal_all/baseline --rounds 5 --seconds 5 \
  --output benchmarking/results/m1pro_all_metal.json
```

Metal backends that regress are retained for explicit `backend="metal"` use
and reported as slowdowns, not optimization wins. Chi2 and CFS, which never
passed the runtime gate, are excluded from `auto`. Force `backend="cpu"` when
it better fits the workload.


## Feature-heavy follow-up (p > n)

The table above only measured inputs with more samples than features. The
three cases that did not pass the gate were re-measured with 16–32 times more
features than samples, using the same script, frozen baseline
(`154863ac…3f801d`), eight CPU threads, five alternating fresh-process rounds
and at least five measured seconds per worker (30 workers, 155.6 measured
seconds). All 15 paired comparisons passed full-output parity. The 1-minute
load average was 4.9 when the run started; the machine was otherwise idle.

| Case | Optimized CPU (ms) | Metal (ms) | CPU time / Metal time | Per-round range | Consistent runtime gain |
|---|---:|---:|---:|---:|---|
| Chi2, 256 × 8,192 counts | 7.931 | 11.074 | 0.72× | 0.62–1.35× | No |
| CFS, 256 × 2,048 discrete | 801.506 | 899.644 | 0.89× | 0.86–1.02× | No |
| MultiSURF, 256 × 4,096 continuous | 16.551 | 59.973 | 0.28× | 0.21–0.82× | No |

Wide inputs do not rescue these paths. Chi2 and CFS were therefore removed
from automatic Metal selection (explicit `backend="metal"` is unchanged).
MultiSURF keeps automatic Metal because its discrete cases gain 5–7×; pass
`backend="cpu"` for continuous data.

Measured source fingerprint: `ac2606628399f4ba2364eec0408fa69dea390075b2c90a2cadcf5ad03ca39343`.
Raw results: [`m1pro_metal_feature_heavy.json`](m1pro_metal_feature_heavy.json).

```bash
python benchmarking/benchmark_all_metal.py \
  --baseline-root audit/metal_all/baseline --rounds 5 --seconds 5 \
  --cases Chi2Wide CFSWide MultiSURFContinuousWide \
  --output benchmarking/results/m1pro_metal_feature_heavy.json
```
