# More features than samples: GPU research

This experiment answers the coverage question for the latest five additions:
regression MI, GiniIndex, DISR, CIFE and MIFS. Earlier admission measurements
were n > p. The new cases are 64×1024, 128×4096 and 256×8192 (samples×features).
The retained CPU implementations are frozen under `cpu/` with SHA256 identity
in `manifest.json`. No runtime package source is changed by these experiments.

`common.py` performs full mathematical fit preparation/reduction/selection;
`metal.py` supplies physical M1 GPU kernels; `cuda.py` supplies experimental
NVIDIA kernels. Source files stay outside setuptools' `src` package search.
These wrappers do not provide the public estimators' metadata/clone interface,
so measured GPU results are end-to-end **prototype** results pending package
integration and its separate validation/performance gate.

The mathematical authorities are unchanged: Breiman et al. (1984) for Gini,
Kraskov et al. (2004) and Ross (2014) for regression MI, Meyer & Bontempi for
DISR, Lin & Tang for CIFE, Battiti for MIFS, and Brown et al. (2012) for the
discrete entropy identities. See the primary citations and exact invariants
in `../new_algorithms_2/README.md` and the root policy.

GPU sorting compares exact order-preserving uint64 keys represented by two
uint32 parts; it never casts raw thresholds to float32. Sorted values feed the
same host float64 Gini sweep. GPU categorical histograms retain first-sample
indices and emit count lists in that order, preserving float64 entropy sums
and greedy ties. Greedy selection still runs sequentially. MIFS omits the
unused conditional histogram transfer/reduction.

Metal KSG proposals use float32 solely to find a candidate set. If each
coordinate-rounding error is <=e and coordinates have magnitude <=M, the
subtraction error is bounded by coordinate error plus float32 rounding. The
implementation uses the conservative joint bound
`2*max(e_x,e_y) + 16*eps32*M + 8*tiny32`. An omitted candidate's true distance
is at least `approx_cutoff - error`. The candidate kth distance is accepted
only when strictly below this lower bound, otherwise every distance in that
query is recomputed in float64. The strict radius uses `nextafter(...,0)`;
marginal counts and digamma reduction are float64 in original sorted-x order.
No approximate neighbor or changed k is allowed.

CUDA uses native float64 joint distances, with the same strict host radius and
marginal reduction. Its shared-memory sort and histogram kernels use integer
arithmetic. The user confirmed no NVIDIA host is available; only simulator
logic can be tested locally. Simulator success does not establish native
compilation, hardware races/rounding or performance. The benchmark explicitly
rejects CUDA simulator timing runs. No CUDA speedup is claimed or auto-enabled.

```sh
# Physical Metal correctness:
.venv/bin/python -m pytest audit/feature_heavy/test_prototypes.py --no-cov

# CUDA logic only (must set the environment before importing Numba):
NUMBA_ENABLE_CUDASIM=1 FEATURE_GPU_TEST_BACKEND=cuda \
  .venv/bin/python -m pytest audit/feature_heavy/test_prototypes.py --no-cov

# Physical Metal sustained comparison to the retained CPU:
.venv/bin/python benchmarking/benchmark_feature_heavy.py --backends cpu metal
.venv/bin/python benchmarking/report_feature_heavy.py

# On a real NVIDIA host with a compatible driver and CUDA toolkit:
.venv/bin/python benchmarking/benchmark_feature_heavy.py --backends cpu cuda \
  --threads 8 --rounds 5 --seconds 1.05 --output audit/feature_heavy/cuda_hardware.json
```

The selected CUDA stack must supply `numba.cuda`. Numba's built-in CUDA target
is deprecated in favor of NVIDIA's `numba-cuda`; consult the official NVIDIA
and Numba installation documentation for the target host. No CUDA packages or
drivers are installed into this Mac to pretend it can execute NVIDIA kernels.
The command above is suitable for Linux CUDA hosts; the benchmark's RSS
collection uses Unix `resource.getrusage`.

Every sustained case runs in a fresh process, warms first, alternates order
for five rounds and totals >=5.25 measured seconds per backend/case. Full
outputs and selected indices must agree. Results compare the current
optimized Numba CPU with the GPU pipeline, not v0.2.1 or an old Python loop.
The prior popular external packages are not remeasured by this GPU-focused
experiment. See `benchmarking/results/FEATURE_HEAVY_GPU.md` for all timings,
small-input regressions and CUDA status.

Follow-up CPU thread calibration tried 1/4/8 in fresh workers. Eight won all
medium/large cases in that short calibration; four appeared faster for small
Gini/CIFE/MIFS, which were repeated in five paired rounds. Those retests became
highly unstable under a separate CPU-intensive workload (system load average
88.6 on eight cores), so they are excluded from the primary result. Do not use
their inflated CPU latencies to claim larger GPU gains. Files are retained as
`cpu_thread_tuning_fresh.json` and `retuned_small.json`; the original run is
also preserved as `sustained_threads8.json`. The earlier same-process thread
screen is exploratory and not a baseline selector.

The original eight-thread `sustained.json` is unchanged. Its full system-load
history was not captured, so a clean CPU-thread confirmation is still required
before package integration or auto-selection. This is deliberately a research
result, not a production-backend admission. `--cpu-threads` can select a
different CPU pool from the GPU's `--threads` host pool on the eventual target.
The harness now records load averages before/after each worker and accepts
`--max-start-load` to reject obviously busy starts. For a clean M1 recheck,
wait for the other CPU job to finish and use `--max-start-load 12`; the original
run did not have those load samples. A load guard supplements rather than
proves an otherwise idle machine.
