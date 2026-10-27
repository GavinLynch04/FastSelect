# Measured optimization loop

The five externally admitted selectors have a separate reproducible comparison
against tuned scikit-learn and pinned original scikit-feature, rather than a
historical FastSelect release. See [the report](results/NEW_ALGORITHMS.md),
[provenance and scope](../audit/new_algorithms/README.md), and
`benchmark_new_algorithms.py --production --cpu-only`. Its five rounds each
measure at least 1.05 seconds, totaling at least 5.25 seconds per case/backend;
the raw JSON retains every iteration and complete reference API outputs.

The [second batch](results/NEW_ALGORITHMS_2.md) compares regression MI and Gini
against tuned scikit-learn, plus DISR/CIFE/MIFS against pinned original
scikit-feature. `benchmark_more_algorithms.py --tune` records reference tuning;
the normal run measures frozen prototypes before admission, while
`--production` loads their byte-identical promoted modules. The losing ANOVA
prototype stays outside the package. See
[batch-two provenance](../audit/new_algorithms_2/README.md).

The [feature-heavy follow-up](results/FEATURE_HEAVY_GPU.md) compares the latest
five retained CPU paths to physical Metal prototypes at p > n. Run
`benchmark_feature_heavy.py` for the fixed eight-thread comparison, or tune
with `tune_feature_heavy_cpu.py` and `--cpu-threads`. Loaded CPU retests are
kept separately and excluded. CUDA simulator timing is explicitly rejected;
real CUDA throughput needs an NVIDIA host. These remain research backends.

Work on `development`. Before optimizing, incorporate the released fixes from
`main`; an older development tip is not a correctness baseline. Use the current
fastest correct implementation, including retained local optimizations, as the
performance baseline. Freeze its revision or source snapshot and record its
source hash before making further changes. The earlier October 2026 experiments
used `eafab5f` (release `v0.3.0`); that historical choice does not make it the
baseline for future work. Nothing in this workflow pushes, merges, publishes,
or changes branch protection.

## Iterate and retain only demonstrated gains

1. Identify a bottleneck and change one allocation, data layout, or kernel at a
   time. Preserve the equations and tie rules documented in `../CLAUDE.md` and
   `../src/fast_select/CLAUDE.md`.
2. Add independent equation-oracle tests and relevant adversarial inputs. CPU
   agreement with an old version is not sufficient: both can share a bug.
3. Run correctness checks, then benchmark without other CPU-intensive work:

   ```bash
   NUMBA_NUM_THREADS=8 python -m pytest --no-cov
   python benchmarking/compare_head_vs_current.py \
     --baseline-ref origin/main \
     --estimators MutualInformation CFS mRMR \
     --backends cpu --threads 8 --samples 900 --features 128 \
     --rounds 5 --min-seconds 5 --rtol 1e-12 --atol 1e-13 \
     --gate --json-output benchmarking/results/experiment.json
   ```

   Set `--baseline-ref` to the exact revision containing the best retained
   implementation. `origin/main` is suitable only when it contains that
   implementation. The default is `HEAD`, which excludes uncommitted retained
   optimizations. If those optimizations are the best baseline, preserve their
   sources before editing and use a harness that measures the frozen snapshot;
   do not silently substitute an older commit. For a new GPU backend, compare
   against the current optimized CPU on the same machine; report comparisons
   against existing GPU backends separately when available.
   Select a thread count supported by the benchmark machine. The script
   defaults to **CPU only**; do not enable
   `NUMBA_ENABLE_CUDASIM` for performance measurements.
4. Repeat with both sample-heavy and feature-heavy shapes, different seeds and
   cardinalities (`--states`), and small inputs. `--dtype float64` exercises
   float64 input preprocessing for the Relief-family estimators; information
   benchmarks intentionally generate integer-coded data.
   `--states` configures the information/Chi2 datasets; MDR always uses ternary
   genotypes and the Relief-family cases use continuous random features.
5. Revert experiments that do not improve their target workload or change
   scores/selections outside the numerical contract. Retain JSON evidence and
   document tradeoffs, regressions, and unsupported hardware before proposing
   promotion to `main`.

`--gate` exits nonzero unless **every requested case** preserves complete
measured score arrays and selected feature indices, and improves at least one
measured resource in **every round**. By default this means at least a 1.03x
runtime speedup or a 3% reduction in traced host-allocation peak. Indices must
match exactly; floating arrays use `--rtol`/`--atol`. MDR also compares its
interaction, cross-validation statistics, lookup table, and predictions. This
is a conservative consistency check, not a statistical significance test or a
replacement for the independent oracles. A win on one shape does not imply a
win on every dataset.

## Measurement details and limitations

- Each version/case/round runs in a fresh subprocess. The version-comparison
  script extracts its baseline from Git rather than importing the installed
  package. Backend comparisons use the same identified source snapshot for
  CPU and GPU.
- JIT signatures and the actual dataset are warmed before timing. Each worker
  records at least five cumulative seconds of fits. Five or more rounds are
  required, with version order alternating between rounds.
- Numba threads are fixed by `--threads`; OpenBLAS, OpenMP and Accelerate limits
  are set to one in worker environments. CPU temperatures and unrelated system
  activity can still affect results.
- JSON records individual timings, iteration count, measured duration, mean and
  median latency, source hashes, baseline SHA, environment/library versions,
  dataset shape/dtype/seed, thread count, and full-output comparison results.
- `peak_traced_mb` is the `tracemalloc` host-allocation peak of an additional
  warmed fit. It includes observable Python/NumPy allocations, **not total
  process RSS, every native JIT allocation, or GPU memory**. Do not describe a
  traced-memory reduction as an equivalent reduction in total system RAM.
- CUDA memory is reported as unavailable and cannot pass a memory gate.
  Physical NVIDIA hardware is required for trustworthy GPU throughput and
  real-device parity validation. For Metal, `benchmark_all_metal.py` compares
  complete fits against the frozen optimized CPU sources, retaining full-output
  NPZ files; see [the all-algorithm report](results/METAL_BACKENDS.md).
  `benchmark_relieff_metal.py` retains the initial ReliefF experiment.

## Optional historical reference

Older releases such as `v0.2.1` may be measured for historical context. This is
optional and cannot replace the current optimized baseline or establish a new
speedup claim:

```bash
python benchmarking/compare_head_vs_current.py \
  --baseline-ref v0.2.1 --estimators MutualInformation CFS MDR \
  --backends cpu --threads 8 --samples 900 --features 128 \
  --rounds 5 --min-seconds 5 \
  --json-output benchmarking/results/published_v021_reference.json
```

Do not use historical output agreement as the correctness authority. Released
fixes intentionally changed some results since `v0.2.1`; a failed historical
parity gate is therefore not automatically an optimization regression. Use
the current correct baseline and independent oracles for that decision.
