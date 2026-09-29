# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-28

The first release since 0.2.1 (the last version on PyPI). It was briefly
prepared as 1.0.0; a pre-release review found wrong-result and input-validation
defects that a stable-API declaration should not paper over, so the version is
0.3.0. A 1.0.0 release is planned once the CUDA backend has been verified on
physical NVIDIA hardware and the support matrix is final.

### Breaking

- **Scores change.** The algorithm corrections listed under *Corrected algorithm
  definitions* and *Fixed in the release review* are intentional. ReliefF, SURF,
  SURF\*, MultiSURF, MultiSURF\*, CFS, MDR, mRMR, and GPU mutual-information
  results produced by 0.2.1 and earlier may differ. Re-run any selection you
  intend to compare against older output.
- **Stricter input validation.** MDR rejects genotypes other than exactly
  0/1/2 (previously `0.5`, `256`, `-256` were silently cast) in both `fit` and
  `predict`, and `predict`/`transform` on MDR, CFS, and mRMR now check the feature
  count and column names seen during fit. The classification-only Relief
  estimators, CFS, and MDR reject continuous targets instead of treating every
  distinct value as a class. Boolean or non-integer `n_neighbors`,
  `discrete_limit`, `k`, `cv`, `n_features_to_select`, and out-of-range `n_jobs`
  raise `TypeError`/`ValueError`.
- **`MultiSURF.fit(feat_idx=...)`** now restricts ranking as well as scoring:
  `top_features_` only contains evaluated features and `n_features_to_select`
  refers to that subset. `feat_idx` is validated (non-empty, 1-D, integer,
  in range, unique).
- **`mRMR.unique_vals_` was removed.** Each feature and the target are now
  encoded independently, so there is no shared vocabulary.
- **`MDR` is now a `TransformerMixin` as well as a classifier** (it already had
  `transform`), and `MDR(backend=...)` stores the argument unchanged.
- **`pandas` is no longer a runtime dependency.** `CFS.transform` now detects
  DataFrames by duck typing, so a DataFrame in still returns a DataFrame out;
  only the mandatory install shrank. Install `fast-select[test]`, or pandas
  directly, if you relied on it being pulled in transitively.
- **The `gpu` extra is now empty.** It previously installed `cupy-cuda11x`, which
  this library has never imported. The CUDA backend runs on `numba.cuda` and
  needs an NVIDIA driver and CUDA toolkit on the machine, not a Python package.
  `pip install fast-select[gpu]` still resolves, so no command breaks.
- **Building from source requires `setuptools>=77`** for the PEP 639 license
  metadata. Installing from a wheel or sdist is unaffected.

### Fixed in the release review

- **SURF / SURF\* threshold ties.** The global radius was computed from
  unscaled float32 values while the CPU compared distances of scaled values, so a
  pair whose distance equals the radius could fall on either side (for
  `X=[[0],[1],[3]]`, CPU SURF scored `+0.222` instead of `-0.222`). Radius and
  all pair distances are now float64 and identical on both backends, and a
  distance within a relative `1e-12` of the radius is a tie excluded from the
  near and far sets. This makes the CUDA distance matrix float64 (twice the
  memory of the previous float32 matrix) for SURF.
- **Premature float32 conversion in the Relief family.** Adding `1e8` to a
  continuous feature zeroed every score, and relabelling categories `0/1` as
  `2**25`/`2**25+1` collapsed them. Continuous columns are now normalised in
  float64 (`(x - min) / (max - min)`) before narrowing, discrete columns are
  replaced by exact category codes, and both backends score the same prepared
  matrix.
- **mRMR category encoding** rejected float-coded categories (`0.0`/`1.0`) and
  merged distinct `uint64`/`int64` symbols (`2**63` and `2**63 + 1`) through a
  shared float64 vocabulary. Features and target are now encoded independently
  into int32; string categories are accepted.
- **Mutual information aliasing.** `calculate_mi_*` cast large non-negative
  codes to int32 without checking (`[0, 2**32, ...]` scored 0 bits instead of
  1). Large codes are now remapped to dense per-column codes; empty input raises.
- **MDR** validates genotypes before casting, records and validates the training
  schema, clones (`backend` is no longer rewritten in `__init__`), is a
  classifier for `sklearn.base.is_classifier`, and raises `NotFittedError` from
  `predict_proba` before fit. MDR lookup-table cell indices for `k >= 5`
  overflowed `uint8` arithmetic on NumPy 2 (`k=6` indexed cell 216 instead of
  728); they are now computed in int64.
- **CFS / mRMR scikit-learn compatibility.** Mixins now precede `BaseEstimator`,
  so both expose transformer tags and pass scikit-learn's full estimator checks.
- **CFS silently selected the wrong column** when a DataFrame's columns were
  reordered at transform time; the schema is now validated. List input works.
- **TuRF with a real Relief estimator** crashed when the retained subset became
  smaller than the base estimator's `n_features_to_select`. The clone is now set
  to keep every active feature, since only scores are used.
- **MultiSURF `feat_idx`** could select features it never evaluated (their zero
  scores outranked negative real ones).
- **CFS `n_jobs=-1`** requested `NUMBA_DEFAULT_NUM_THREADS`, raising under
  `NUMBA_NUM_THREADS=2`. All parallel estimators now honour the configured Numba
  pool and validate explicit `n_jobs`.
- **Import under the CUDA simulator.** Importing the package no longer touches
  private Numba runtime internals, so `NUMBA_ENABLE_CUDASIM=1 python -c "import
  fast_select"` works. The Windows-only CUDA context workaround is installed only
  on Windows, can be disabled with `FAST_SELECT_DISABLE_CUDA_CONTEXT_PATCH=1`, and
  warns instead of silently swallowing a failed context push. That Windows
  path could not be exercised in this review (no Windows or physical-GPU
  hardware).

### Added

- Python 3.13 and 3.14 support, with matching package classifiers,
  version-specific Numba minimums, and CI test/build coverage.
- CI jobs that run the CUDA kernels in the Numba CUDA simulator (kernel logic
  only, not a substitute for physical-GPU testing) and build the documentation
  with warnings as errors. Publishing to PyPI now waits for the full check suite
  on the tagged commit, strict `twine check`, an assertion that the release tag,
  package metadata, built artifacts, and changelog agree, and a smoke install of
  the built wheel.
- Regression and oracle tests from the release review: an exact rational
  SURF/SURF\* oracle with true radius ties, translation and category-relabelling
  invariance, independent MI/mRMR/CFS/chi2/MDR controls, scikit-learn contract
  checks for every estimator, and Pipeline/GridSearchCV smoke tests.
- `fast_select.__version__`, resolved from installed package metadata.
- `calculate_mi_single_pair` and `calculate_mi_matrices` are exported at the top
  level and documented in the API reference. Both take `backend` and `unit`
  (`"bit"` or `"nat"`) and back the mRMR criterion.
- CI now enforces `ruff` and `black` and builds/validates the distribution with
  `twine check --strict` on every push and pull request.

### Corrected algorithm definitions

- **SURF / SURF***: Replaced target-specific radii with the published global
  mean pair-distance radius and restored separate hit/miss neighbor-count
  normalization. SURF* now excludes threshold ties and normalizes its reversed
  far-neighbor updates separately.
- **MultiSURF***: Added the upper `mean + standard_deviation / 2` boundary,
  restored the dead band, and replaced the former non-near-miss update with
  published far-hit/far-miss feature-similarity scoring.
- **CFS**: Replaced greedy forward selection, the unpublished `0.1` relevance
  cutoff, and post-search pruning with forward best-first search and the
  canonical consecutive-non-improvement stopping rule.
- **MDR**: Restored the inclusive high-risk ratio threshold, treats empty cells
  as low risk, and now maps arbitrary binary class labels to and from internal
  case/control codes.
- **Mutual information CUDA**: Fixed duplicated and overlapping sample counts,
  honored both bit and natural-log units, and validates GPU state limits before
  launching.
- **CFS CUDA**: Entropy calculations now read only initialized categorical
  states.

### Compliance and verification

- Added independent, equation-driven regression tests for SURF, SURF*,
  MultiSURF, MultiSURF*, CFS merit/search, mutual-information units, and MDR
  threshold/label behavior.
- Expanded CPU/GPU parity coverage to both star variants and mutual information
  in bit and natural-log units.
- Added repository, production-code, and test `CLAUDE.md` policies that make
  defining papers authoritative over external implementations and require
  sustained five-second v0.2.1 benchmarks for performance changes.

### Performance

- **Cost of the release-review corrections** (CPU only; Apple M1 Pro, 900 samples
  x 128 features, Python 3.14.7, NumPy 2.5.3, Numba 0.67.0; fresh process per
  case, JIT warmed, at least 5 s measured per case, 5 rounds with alternating
  order, previous tree vs this one; raw results in
  `benchmarking/v030_head_vs_release_review_cpu.json`, median of per-round mean
  fit time). **SURF is 0.887x and SURF\* 0.922x, i.e. about 13% and 8% slower**,
  plus 0.47 MiB more traced host memory: float64 distances are a required
  correctness fix, not an optimization. Neutral within run-to-run noise:
  MultiSURF 0.983x, MultiSURF\* 1.026x, mutual information 1.022x, CFS 0.987x.
  ReliefF measured 1.051x and MDR 1.142x (MDR's lookup-table construction is now
  vectorised); neither was a goal of this work. Against published v0.2.1 (2
  rounds, `benchmarking/v030_release_review_cpu.json`) every Relief-family
  estimator is still faster, but that gain comes from earlier committed
  optimizations, and CFS is 0.77x (slower); that slowdown predates this work
  (the previous tree measures the same as this one). **CUDA was not measured** (no NVIDIA device);
  SURF's CUDA distance matrix is now float64 and therefore twice the previous
  size.
- Fused CUDA distance/scoring stages for the Relief family and removed
  host-side distance/weight round trips and redundant pair-distance work.
- Reused thread-local CPU buffers and pre-normalized continuous columns.
- Made canonical CFS best-first child evaluation incremental, reducing the
  corrected search benchmark from about 0.264 to 0.051 seconds per CPU fit on
  the recorded 900-by-128 case.

### Fixed

- Removed a `RuntimeWarning: nopython is set for njit and is ignored` emitted on
  every `import fast_select`, caused by a redundant `nopython=True` on an MDR
  kernel.
- Registered the `slow` and `gpu` pytest markers, clearing
  `PytestUnknownMarkWarning` during collection.

### Packaging and tooling

- Marked `Development Status :: 4 - Beta`.
- Populated the previously empty `docs` extra with the Sphinx toolchain, and
  pointed Read the Docs at it instead of the full `dev` extra (which pulled a
  CUDA wheel into the docs build).
- Moved the deprecated top-level `ruff` settings into `[tool.ruff.lint]`, and
  ignored the naming rules that the paper-matching module, class, and function
  names (`ReliefF.py`, `mRMR`, `chi2`, ...) intentionally violate.
- Applied `black` and `ruff` across `src/`, `tests/`, and `benchmarking/`;
  the repository is now clean under both, matching the badges in the README.
- `CITATION.cff` no longer carries a version, DOI, or release date (per-release
  values that had drifted and pointed at an earlier immutable Zenodo record), and
  the Sphinx `release` is read from the installed package metadata.
- Strict documentation build fixed (missing `_static`, scikit-learn
  metadata-routing cross-references).
- Benchmark results now record source hash, commit, dependency versions, device,
  and an explicit "unavailable" marker (not `0`) for unmeasured GPU memory.

### Documentation

- Corrected README claims that the library uses **Joblib** for CPU parallelism;
  the CPU kernels are thread-parallel Numba `prange`, and joblib is not a
  dependency.
- Documented the real GPU prerequisites (NVIDIA driver plus CUDA toolkit, no
  Python package) and added a `is_cuda_ready()` check to the install section.
- Added a *Versioning and Stability* section with an explicit 0.2.x upgrade
  warning and the numerical policy, and contribution instructions covering the lint/test commands and
  the paper-citation requirement for algorithm changes.
- Rewrote `paper.md`, which described infrastructure the project does not have
  (a Cython shim, `mypy`, Docker images, macOS/Windows CI), overstated coverage,
  and cited a 88x/12x scikit-rebate speed-up on a 30000x200000 dataset for which
  no supporting measurement exists in the repository. Claims are now limited to
  what the repository can substantiate, and a new *Correctness* section describes
  the paper-oracle verification approach.
- Added the missing `paper.bib` referenced by the `paper.md` front matter,
  populated with the defining papers for every implemented algorithm.
- Fixed docstring cross-references that made the Sphinx build emit errors, and
  documented CFS's search as Hall's forward best-first rather than "greedy", and
  MDR's high-risk rule as inclusive with empty cells treated as low risk.

## [0.2.1] - 2026-07-22

### Fixed

- **CUDA Runtime Context Initialization**: Added Numba CUDA attached context monkey-patching in `utils.py` to prevent `ac.devnum` `IndexError` (`0x10000003`) and `CUDA_ERROR_INVALID_CONTEXT` on Windows host environments.
- **GPU Grid Launching**: Corrected kernel block grid calculation across `ReliefF`, `SURF`, and `MultiSURF` (`blocks = (n_samples + TPB - 1) // TPB`), preventing CUDA memory access violations on large sample sizes ($N \ge 1000$).
- **ReliefF Multi-Class Weighting**: Aligned CPU and GPU multi-class probability weighting ($P(c)/(1-P(y_i))$) and dynamic $k$-neighbor searching with literature standard (Kononenko 1994).
- **Deterministic Distance Tie-Breaking**: Synchronized CPU insertion sort (`d < hit_d[k-1]`) and GPU distance matrix sorting (`np.argsort(..., kind='stable')`) so discrete feature Hamming distance ties break identically across backends.

### Optimized

- **Zero-Allocation CPU Kernels**: Rebuilt `_relieff_cpu_kernel`, `_surf_cpu_kernel`, and `_multisurf_cpu_kernel` with `@njit(parallel=True, fastmath=True)`, replacing sample-wise matrix allocations ($N \times P$) with thread-local scalar aggregations.
- **Stream-Safe Host Callers**: Replaced explicit `cuda.synchronize()` calls across all Relief, CFS, and MDR host callers with native `device_array.copy_to_host()`, eliminating `CUDA_ERROR_CONTEXT_IS_DESTROYED` crashes.

### Testing & Parity

- **Parity Test Suite**: Added `tests/test_gpu_cpu_parity.py` testing CPU vs GPU numerical alignment across 5 synthetic dataset types (`standard`, `single_feature`, `all_discrete`, `p_dominant`, `zero_variance`), verifying max relative difference $\le 2.08 \times 10^{-7}$.
- **Code Coverage**: Reported 95% line coverage (114 passing unit tests). This counted host-side Python only: numerical kernels are excluded from line coverage, so the figure is not evidence that 95% of numerical or GPU behaviour was verified.

## [0.2.0] - 2025-07-30

### Implemented

-   **Chi2**: Implemented a SKLearn-style chi2 feature ranking function with GPU and CPU parallelization.
-   **mRMR**: Implemented the Minimum Redundancy Maximum Relevance (mRMR) algorithm, with both gpu and
              cpu acceleration. Offers support for either mutual information difference or quotient (more info in docs).
-   **CFS**: Implemented the Correlation-based Feature Selection (CFS) algorithm, with GPU and CPU acceleration.
-   **MDR**: Implemented Multifactor Dimensionality Reduction (MDR), an algorithm used primarily in SNP-based feature
             selection and ranking. Only accepts 0, 1, or 2 as X variable (common SNP encoding scheme).
-   **User-Guide**: Added a user guide for when to use which feature selection tool. Provides information on both the
                    strengths and weaknesses of each algorithm.

### Fixed

-   **MultiSURF**: Issue with discrete distance calculation in GPU implementation.
-   **All Relief-Based Algos**: Fixed data verification issues, and implemented error checking improvements.

### Testing

-   **Tests**: Significantly increased code coverage for all models.

## [0.1.6] - 2025-07-24

### Fixed

-   **MultiSURF**: Issue with discrete distance calculation in GPU implementation.

### Testing

-   **Tests**: Significantly increased code coverage for all models.

## [0.1.5] - 2025-07-21

### Updated

-   **README**: Clarifying details

### Testing

-   **Automation**: GitHub workflow

## [0.1.4] - 2025-07-21

### Updated

-   **README**: Added Zenodo DOI, other edits.

## [0.1.3] - 2025-07-21

### Fixed

-   **Package API**: Fixed a bug with TuRF transform function.

## [0.1.2] - 2025-07-21

### Fixed

-   **Package API**: Fixed a bug with TuRF relating to variable names.

## [0.1.1] - 2025-07-21

### Fixed

-   **Package API**: Corrected the package's `__init__.py` to expose estimator classes (`ReliefF`, `TuRF`, etc.) at the top level. This fixes `TypeError: 'module' object is not callable` when using standard imports like `from fast_select import TuRF`.

## [0.1.0] - 2025-07-21

### Added

-   **Initial Release of `fast-select`**: A high-performance, Numba-accelerated library for various feature selection.
-   **Core Algorithms**: Implementations of ReliefF, SURF, SURF*, MultiSURF, MultiSURF*, and TuRF.
-   **Dual Execution Backends**:
    -   Thread-safe, parallelized CPU kernels for high-speed execution on multi-core processors.
    -   Correct and performant CUDA kernels for massive parallelism on NVIDIA GPUs.
-   **Benchmarking Suite**: A comprehensive suite to measure and compare the runtime and memory performance of `fast-select` algorithms against other libraries.
-   **Project Renaming**: The project identity was established as `fast-select` to better reflect its purpose.

