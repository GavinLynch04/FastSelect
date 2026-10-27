# Experimental ReliefF on the Apple M1 Pro GPU

This report records the initial opt-in `ReliefF(backend="metal")` experiment
using MLX custom Metal kernels. At that stage, `auto` selected CUDA or CPU.
The subsequent [all-algorithm implementation](METAL_BACKENDS.md) adds automatic
CUDA → Metal → CPU selection and support for the other selectors. The original
measurements below are retained unchanged; no release, commit or push was made.

## Implementation and mathematical contract

The target is Kononenko (1994) ReliefF for complete-data classification:
range-normalized continuous differences, equality-based discrete differences,
nearest hits and per-class misses, class-prior miss weighting, and normalization
by each group's actual neighbor count. Missing values remain unsupported.

The backend uses the existing float64 normalization/category encoding before
narrowing to the common float32 working matrix. Its distance kernel reuses
16-by-32 feature tiles in threadgroup memory; candidate neighbor searches run
in chunks of 128 and merge with lower-input-index tie ordering. Query batches
contain at most 512 samples. No subsampling or approximate-neighbor search is
used. Scoring and reductions run on the GPU; validation, input preparation,
ranking, and ambiguous-neighbor repair run on the CPU.

Metal has no native float64 arithmetic. The distance kernel uses short float4
sums followed by compensated block accumulation in safe math mode. If the
k-th and next candidate are within a conservative 32-float32-epsilon relative
bound (with an absolute floor), it recomputes that class group's neighbors
using CPU float64 accumulation. Prototype support is limited to
`n_neighbors <= 64`. This does not establish Metal support for SURF's much
tighter double-precision global-radius contract.

The distance buffer occupies `4 * min(n_samples, 512) * n_samples` bytes,
rather than storing a full distance matrix for large sample counts. Additional
input, partial-neighbor, score, framework, and CPU buffers are required. This
is an allocation formula, **not a measured peak-memory result** or a claim of
less memory than the streaming Numba CPU implementation.

## Correctness and precision investigation

The complete suite passed: **632 passed, 76 skipped**. All **65 new physical
Metal tests** passed; the skipped cases need CUDA. Tests include independent
Kononenko equation oracles, multiclass imbalance, singleton classes, continuous
and discrete mixtures, constants, large offsets/category labels, tied neighbors,
C/Fortran/strided inputs, query/feature tile tails, k=1 through k=64, cloning,
transformation validation, Pipeline, and GridSearchCV.

An initial benchmark used `rtol=1e-4, atol=2e-7` for CPU/Metal comparison. It
stopped after finding a 1.05e-6 score difference on the 2,000-by-512 mixed,
four-class dataset; the partial run is preserved in
[`m1pro_relieff_metal_initial.json`](m1pro_relieff_metal_initial.json).
Instrumenting the two paths found identical neighbor membership in all 8,000
sample/class groups. A separate raw-input float64 equation oracle gave:

| Backend | Maximum absolute score error versus oracle | Selected indices |
|---|---:|---|
| Existing Numba CPU | 1.074e-6 | Exact match |
| Metal | 2.329e-8 | Exact match |

The difference comes from serial float32 score accumulation in the CPU kernel;
the Metal path sums within each sample before its final reduction. The new
large independent oracle regression retains separate bounds for the two paths.
The final performance comparison uses `rtol=1e-4, atol=2e-6` and exact selected
indices. This is tighter than the existing CUDA parity test's 1e-5 absolute
tolerance. Full oracle scores and errors are saved in
[`m1pro_relieff_metal_oracle.json`](m1pro_relieff_metal_oracle.json); the neighbor
instrumentation is retained in `audit/metal_relieff/inspect_neighbors.py`.

## Benchmark method

Apple M1 Pro, 8 CPU cores, 16 GiB unified RAM; native arm64 macOS 27.0.1;
Python 3.13.17 and MLX/MLX-Metal 0.32.4. The JSON records remaining library
versions and the exact source fingerprint. CPU uses eight Numba threads;
BLAS/OpenMP limits are one. No CUDA simulator is used.

Each backend/case/round runs in a fresh process with identical seeded input.
Five rounds alternate CPU/Metal order. Each worker performs three warm-up fits
and then at least five cumulative seconds of measured complete `.fit()` calls.
The timed call includes validation, preparation, GPU array creation, distance
and neighbor computation, any CPU precision repair, scoring, synchronized host
results, and final feature ranking. MLX lazy dispatch alone is never timed.
Displayed latency is the median of the five per-round mean latencies.

Inputs are seeded synthetic float32 matrices (`seed=20261026`), k=10, ten
selected features, and `discrete_limit=3`. Continuous cases use normal random
features and two balanced classes. The mixed case has half ternary discrete
features and four balanced classes; the discrete case is entirely ternary with
two balanced classes. Real dataset performance and other M-series chips are
not established by these measurements.

The first `.fit()` in each fresh worker is also recorded separately. It
includes backend initialization and any compilation/cache loading, but excludes
the initial package import. Persistent compiler caches are not erased, so these
are first-process-use measurements, not pristine-cache cold-start benchmarks.

## Measured results

| Samples × features / type | CPU (ms) | Metal (ms) | CPU time / Metal time | Per-round range |
|---|---:|---:|---:|---:|
| 128 × 32, continuous | 0.707 | 1.179 | 0.60× | 0.60–0.63× |
| 512 × 128, continuous | 2.217 | 1.590 | 1.39× | 1.28–1.46× |
| 900 × 128, continuous | 5.041 | 3.468 | 1.45× | 1.28–1.86× |
| 2,000 × 512, continuous | 51.699 | 26.885 | 1.92× | 1.80–1.93× |
| 4,000 × 128, continuous | 62.745 | 29.381 | 2.14× | 1.53–2.41× |
| 1,000 × 2,048, continuous | 60.134 | 39.106 | 1.54× | 1.45–1.73× |
| 2,000 × 512, mixed / 4 classes | 213.782 | 27.743 | 7.71× | 6.63–8.05× |
| 2,000 × 512, discrete | 300.508 | 18.637 | 16.12× | 14.95–16.58× |

All 40 paired comparisons passed full-score tolerance and exact selected-index checks. The 80 fresh workers accumulated **402.3 seconds of measured fits**, excluding warm-up, compilation, and process startup. Seven of eight cases improved runtime by at least 3% in every round; the small case failed the runtime gate and remains a demonstrated slowdown.

These results support explicit Metal selection for larger tested workloads. They do not justify automatically preferring Metal for all input sizes. CPU/GPU speed ratios above one favor Metal; ratios below one favor CPU.

Raw per-fit timings, every score array, exact selected indices, first-fit times, precision-repair counts, library versions, and per-round comparisons are retained in [`m1pro_relieff_metal.json`](m1pro_relieff_metal.json).

Source fingerprint: `154863aca34a39448efefc76eb19e11e650340bbda5c9b26d9a8bb7563f3801d`.

Versions: `{'numpy': '2.5.4', 'numba': '0.68.0', 'scikit-learn': '1.9.1', 'mlx': '0.32.4'}`.

## First fit in a fresh process

Medians of the five first-fit observations, with existing persistent caches:

| Case | CPU (ms) | Metal (ms) |
|---|---:|---:|
| 128 × 32, continuous | 1774.1 | 166.4 |
| 512 × 128, continuous | 1842.0 | 170.3 |
| 900 × 128, continuous | 1947.3 | 192.7 |
| 2,000 × 512, continuous | 1851.8 | 206.4 |
| 4,000 × 128, continuous | 1847.3 | 216.5 |
| 1,000 × 2,048, continuous | 1952.4 | 255.1 |
| 2,000 × 512, mixed / 4 classes | 2038.7 | 209.6 |
| 2,000 × 512, discrete | 2050.4 | 192.2 |

These first-fit observations can favor Metal even when its warmed latency is slower. The Numba ReliefF CPU kernel currently recompiles in each new process. Package imports precede the timer, and caches were not cleared; this is not a claim about complete application startup.


## Historical reference and packaging

Under the previous repository policy, a separate comparison against a clean
published `v0.2.1` tree ran in five fresh-process rounds of at least five
measured seconds per version. At 900 × 128, the published CPU took 26.570 ms versus 4.904 ms for the
current CPU (median round means, 5.42×). Every round matched full output;
maximum absolute score error was 3.15e-9. This reflects existing CPU improvements
since v0.2.1, **not a speedup attributable to this Metal prototype**. The main
table compares Metal against the current, faster CPU implementation. Raw data:
[`m1pro_relieff_v021_reference.json`](m1pro_relieff_v021_reference.json).
The updated policy makes historical release comparisons optional; the current
optimized CPU remains the baseline for Metal speedups.

Ruff, Black, and `git diff --check` pass. Local source and wheel distributions
build successfully, including the new module and optional dependency metadata.
A smoke test imports and fits from the extracted wheel with all MLX imports
deliberately blocked, and verifies that explicit Metal use fails clearly when
MLX is unavailable.

## Reproduction

From a development checkout on Apple Silicon with Python 3.10+:

```bash
pip install -e ".[metal,test]"
python -m pytest --no-cov
python benchmarking/benchmark_relieff_metal.py \
  --rounds 5 --min-seconds 5 --threads 8 \
  --output benchmarking/results/m1pro_relieff_metal.json
```

Run in a process with physical Metal GPU access. In a sandbox that hides the
GPU, MLX raises an unavailable-device error. The package's default import does
not load MLX, and CPU/CUDA users do not need this optional dependency.

```python
from fast_select import ReliefF

selector = ReliefF(backend="metal", n_neighbors=10, n_features_to_select=10)
selector.fit(X, y)
print(selector.metal_diagnostics_)
```
