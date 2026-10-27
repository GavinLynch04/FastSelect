# Apple M1 Pro optimization pass — October 26, 2026

## Scope and baseline

Local `development` was fast-forwarded from `bce53d1` to released `main`
at `v0.3.0` before retaining changes. The raw JSON files record that commit as
`eafab5f17eda7ec4a75e7517e144d119ac112d8b`; on 2026-10-27 the history was
rewritten to edit commit messages only, and the same source tree is now
`bb29886c7342c64e05c3ad344bee047e093a3582` (tag `v0.3.0`).
The older development tip predates confirmed numerical/API fixes; optimizing it
and later merging it would not be an acceptable correctness strategy. No new
commit, push, release, or merge into `main` was performed.

Machine: Apple M1 Pro, 8 CPU cores, 16 GiB RAM, native arm64 macOS 27.0.1.
Environment: Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, SciPy 1.18.1,
scikit-learn 1.9.1. Benchmarks use 8 Numba threads and single-threaded BLAS/OpenMP
limits. No NVIDIA device, Metal backend, or CUDA simulator was used.

Working source fingerprint shared by **all five artifacts**:
`9265e036ede279a7e4e8c13abafffbbb876540e7e180c82bf49db6e0737eb206`.
It hashes the sorted names and contents of `src/fast_select/*.py`.

Each comparison used five independent rounds, fresh subprocesses, warmed JIT,
and alternating execution order. Each version/case/round accumulated at least
five seconds of measured fits; the MIQ confirmation used fifteen. Across all
artifacts, about 704 seconds of actual fitting were measured. Latencies below
are medians of the five round-mean latencies, not cold-start times. Inputs are
seeded synthetic matrices, not a claim about every real-world dataset.

## Retained improvements versus corrected release

| Case | Shape (samples × features) | Baseline | Candidate | Result |
|---|---|---|---|---|
| Mutual information | 900 × 128 | 6.790 ms | 4.841 ms | 1.40x runtime speedup |
| Mutual information | 2,000 × 512 | 234.458 ms | 128.054 ms | 1.83x runtime speedup |
| CFS | 900 × 128 | 17.723 ms | 11.668 ms | 1.52x runtime speedup |
| CFS | 2,000 × 512 | 493.203 ms | 254.421 ms | 1.94x runtime speedup |
| mRMR MID | 900 × 128 | 7.615 ms | 5.484 ms | 1.39x runtime speedup |
| mRMR MIQ, 15-second rounds | 900 × 128 | 7.458 ms | 5.428 ms | 1.37x runtime speedup |
| MDR, k=2, CV=3 | 128 × 512 | 13.082 MiB traced peak | 0.756 MiB traced peak | 94.2% less traced allocation |

Every listed case passed complete measured-output comparisons and improved its
claimed resource in every round by the gate's margin. For the corrected-release
comparisons, `rtol=1e-12`, `atol=1e-13`; selected indices must match exactly.
The largest MI/relevance/redundancy error was `3.85e-16`; CFS public results and
MDR results were unchanged. MDR comparisons include selected interactions, CV
statistics, lookup tables, and predictions.

### Changes and mathematical contracts

- **MI / mRMR:** make feature columns contiguous, balance parallel triangular
  pair work, and accumulate histogram marginals while filling the joint table.
  Shannon's empirical `sum p(x,y) log(p(x,y)/(p(x)p(y)))` remains unchanged,
  including bit/nat units. Peng et al. (2005) MID/MIQ scoring, tie rules, and
  selection stay unchanged. Inputs remain discrete variables; this does not
  introduce a continuous MI estimator.
- **CFS:** cache each variable's entropy once and balance pair work. Hall (1999)
  symmetrical uncertainty `2 I(x;y)/(H(x)+H(y))`, subset merit, best-first
  traversal, and stopping/tie rules remain unchanged. Continuous features still
  use the existing discretization; the search is heuristic, not exhaustive.
- **MDR:** retain complete, lexicographic exhaustive CPU search but stream
  combination batches of at most 4,096. Preserve the package's documented
  complete-data MDR variant (Ritchie et al., 2001): ternary genotypes, arbitrary
  binary labels, dataset case/control risk threshold, balanced-accuracy fold
  scoring, CV consistency and testing-score model selection. Strict `>` updates
  between batches preserve the original first-winner tie behavior. The CUDA
  search still uses its original full combination matrix. CPU combination
  storage is bounded; search time remains combinatorial. Balanced-accuracy
  scoring and the consistency tie policy are the package's declared variant,
  not a claim to reproduce every detail of the original 2001 procedure.

### Tradeoffs, failed cases, and discarded work

- **MI exchanges memory for speed.** The feature-major copy increases traced
  allocation by about 0.44 MiB at 900 × 128 and 3.91 MiB at 2,000 × 512. These
  candidates pass on runtime, not memory. No total-process/native-memory
  improvement is claimed.
- **MDR exchanges time for memory on the feature-heavy case.** Latency increased
  from 45.470 to 53.091 ms, approximately **17% slower**, while traced peak fell
  by **12.33 MiB**. This is retained under the requested runtime-*or*-memory
  rule, not presented as a speedup.
- At **900 × 128**, MDR's traced peak was unchanged at about 1.324 MiB; its
  runtime gain was inconsistent. That case fails the resource gate and remains
  in the first artifact. Streaming is not a demonstrated whole-fit peak-memory
  improvement at every shape.
- The first five-second MIQ run had one round at **0.976x**, so the combined
  first run correctly failed its gate. The separately retained fifteen-second
  confirmation passed all five rounds, with per-round speedups **1.31–1.41x**.
  The initial failure has not been deleted or relabeled as a pass.
- An initial SURF duplicate-buffer experiment was discarded when the baseline
  investigation showed released `main` already addresses it. No SURF kernel
  change is retained; only additional input-layout oracle coverage remains.
- Independent review caught a gate accepting matching infinities. The gate now
  fails closed on empty/nonfinite outputs, overflowed differences, nonfinite
  tolerances, and invalid resource measurements; regression tests cover these.

## Raw evidence and reproduction

All commands use `python benchmarking/compare_head_vs_current.py`, `--backends
cpu --threads 8 --rounds 5`. Corrected-release commands additionally use
`--baseline-ref v0.3.0 --rtol 1e-12 --atol 1e-13` (recorded in the JSON as the
pre-rewrite hash `eafab5f`, which has the same source tree).

| Artifact | Additional arguments | Outcome |
|---|---|---|
| [Initial 900 × 128](m1pro_information_mdr_900x128.json) | `--estimators MutualInformation CFS mRMR mRMR-MIQ MDR --samples 900 --features 128 --min-seconds 5 --gate` | Combined gate fails on MIQ consistency and MDR resource metrics; other cases pass |
| [Larger MI / CFS](m1pro_information_2000x512.json) | `--estimators MutualInformation CFS --samples 2000 --features 512 --min-seconds 5 --gate` | Pass |
| [Feature-heavy MDR](m1pro_mdr_128x512.json) | `--estimators MDR --samples 128 --features 512 --min-seconds 5 --gate` | Pass on traced memory |
| [Longer MIQ confirmation](m1pro_miq_900x128_15s.json) | `--estimators mRMR-MIQ --samples 900 --features 128 --min-seconds 15 --gate` | Pass on runtime |
| [Published v0.2.1 reference](m1pro_published_v021_reference.json) | `--baseline-ref v0.2.1 --estimators MutualInformation CFS MDR --samples 900 --features 128 --min-seconds 5` | Historical reference only; no promotion gate requested |

Set `--json-output` to the artifact path when reproducing. Hardware/environment
changes may change performance; the gate should be reevaluated, not bypassed.
The `states` field records the CLI setting for information/Chi2 generation;
MDR's generator always uses three genotype states regardless of that setting.

Published-v0.2.1 comparison was performed as required by repository policy.
MI was 1.52x faster and CFS 1.22x faster in that reference run. CFS outputs do
**not** match the old release because the intervening released fixes changed
search behavior; its traced memory is also about 2.26 MiB higher. MDR's small
runtime improvement was not consistent and its traced memory was about
0.51 MiB higher. These are not all attributed to this patch. The corrected
release plus independent oracles, not historical output agreement, determine
whether this optimization introduces numerical drift.

## Validation and promotion limits

- Final full test suite: **567 passed, 76 skipped**; no simulator enabled.
- Independent MI and CFS distribution oracles cover constants, dependence,
  imbalance, multiclass/gapped labels, bit/nat units, higher cardinality,
  contiguous/Fortran/strided inputs, and multiple CPU thread counts.
- Independent MDR fold/model oracles run twenty seeds with batch sizes 1, 4,
  and 4,096; explicit ties span batch boundaries.
- Full-output gates also require exact selected-index agreement. Oracle tests
  remain a separate required layer rather than trusting old-version parity.
- Ruff, Black, and `git diff --check` pass.
- `tracemalloc` measures observable Python/NumPy allocations during an extra
  warmed fit, **not total RSS, every native allocation, or GPU memory**.
- CUDA kernels were not modified or benchmarked. Real CPU/CUDA parity remains
  **unverified without NVIDIA hardware**. GPU conformance tests remain ready
  for a physical device; matching CPU tests cannot prove GPU race freedom or
  device precision. These are validated CPU candidates, not a blanket release
  approval, and nothing was promoted to `main`.
