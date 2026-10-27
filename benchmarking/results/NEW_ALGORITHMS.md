# Five new algorithms: external-baseline admission

All five candidates earned a place in the development package as **parallel
Numba CPU implementations**. Each beat its external baseline by at least 3%
in every one of five rounds on both tested shapes, while preserving complete
reference API outputs and selected feature indices. Their experimental Metal
paths did not consistently beat these new CPU paths and remain outside the
distributed package. Continuous MI has no Metal prototype.

## Results on this M1 Pro

Apple M1 Pro, 8 CPU cores, 16 GiB RAM; macOS 27.0.1 arm64; Python 3.13.17,
NumPy 2.5.4, Numba 0.68.0, SciPy 1.18.1, scikit-learn 1.9.1, MLX 0.32.4.
Run started 2026-10-27T03:17:08Z. Times are **complete calls in milliseconds**:
median of the five per-round medians. Central speedup divides these central
times; the final column gives the minimum and maximum paired round speedup.

| Algorithm | Samples × features | External ms | FastSelect CPU ms | Central speedup | Paired round range |
|---|---:|---:|---:|---:|---:|
| MI | 2,000 × 128 | 208.35 | 18.36 | 11.35× | 2.53–11.51× |
| MI | 4,000 × 256 | 948.49 | 78.21 | 12.13× | 9.25–18.59× |
| LaplacianScore | 900 × 128 | 54.55 | 12.41 | 4.40× | 2.49–4.55× |
| LaplacianScore | 1,600 × 256 | 188.15 | 33.99 | 5.54× | 5.36–5.72× |
| JMI | 900 × 128 | 963.34 | 4.27 | 225.55× | 153.59–263.14× |
| JMI | 2,000 × 256 | 4869.01 | 10.27 | 473.92× | 165.03–488.03× |
| CMIM | 900 × 128 | 981.33 | 4.58 | 214.31× | 64.55–229.59× |
| CMIM | 2,000 × 256 | 5512.53 | 10.56 | 522.24× | 115.46–626.05× |
| FCBF | 900 × 128 | 263.19 | 4.30 | 61.28× | 22.67–71.75× |
| FCBF | 2,000 × 256 | 1252.31 | 8.82 | 141.91× | 55.11–154.46× |

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
| lap_medium | 0.94× | 0.69–1.23× | Not retained |
| lap_large | 1.01× | 0.74–1.18× | Not retained |
| jmi_medium | 0.65× | 0.21–4.90× | Not retained |
| jmi_large | 0.63× | 0.49–0.75× | Not retained |
| cmim_medium | 0.42× | 0.33–1.35× | Not retained |
| cmim_large | 0.88× | 0.46–1.32× | Not retained |
| fcbf_medium | 0.54× | 0.45–0.77× | Not retained |
| fcbf_large | 0.56× | 0.24–0.66× | Not retained |

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

The final gate comprises 100 fresh workers and
158.4 measured seconds. The archived prototype matrix comprises
140 fresh workers and 201.7 measured
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

- `ContinuousMI.py`: `d3158664d524a2bb7f5a393ab3ff06201bc8a5df46193b7bc4f18c4813932de2`
- `InformationSelectors.py`: `cbe207169c4125614cefc5b61b1d0dd81f59b0fa8b04ccffb8fd886f0ef793a2`
- `LaplacianScore.py`: `ce84632bf777ce380663e0fbe4e2e2772ee54847af469156372f7dd857f2170d`
- `_cpu_backend.py`: `12d1b843e14218067f9d06ae9c5b236e9705f7dc77ba4b622e0eaea1eb6679d2`

```sh
.venv/bin/python benchmarking/fetch_new_algorithm_baselines.py
.venv/bin/python benchmarking/benchmark_new_algorithms.py --production --cpu-only \
  --rounds 5 --seconds 1.05 --output audit/new_algorithms/final_admission.json
.venv/bin/python benchmarking/report_new_algorithms.py
```

The report generator rechecks all score/selection outputs, cumulative timing,
per-round admission gains and current package source hashes. See
[audit notes and papers](../../audit/new_algorithms/README.md) for primary
equations, limitations, baseline shim details and prototype reproduction.
