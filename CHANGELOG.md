# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-23

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

These are intentional score-changing corrections. SURF-family, CFS, MDR, and
GPU mRMR results produced by earlier releases may differ.

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

- Fused CUDA distance/scoring stages for the Relief family and removed
  host-side distance/weight round trips and redundant pair-distance work.
- Reused thread-local CPU buffers and pre-normalized continuous columns.
- Made canonical CFS best-first child evaluation incremental, reducing the
  corrected search benchmark from about 0.264 to 0.051 seconds per CPU fit on
  the recorded 900-by-128 case.

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
- **Code Coverage**: Achieved 95% total test coverage across the library (114 passing unit tests).

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

