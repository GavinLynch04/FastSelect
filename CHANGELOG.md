# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

-   **MultiSURF / MultiSURF\***: Corrected CPU and CUDA scoring for MultiSURF\* by applying the proper near/far dead-band thresholds and far-hit/far-miss updates.
-   **MultiSURF CUDA**: Fixed GPU feature tiling so datasets with more features than the CUDA thread block size score every feature, and added missing shared-memory synchronization for deterministic CPU/GPU agreement.
-   **ReliefF**: Corrected miss-class averaging in the CPU implementation and made the CUDA backend explicit about its current binary-classification and neighbor-count limits.
-   **Mutual Information CUDA**: Fixed GPU mutual-information relevance scoring so each sample is counted once and the requested output unit (`bit` or `nat`) is respected.
-   **MDR**: Preserved arbitrary binary class labels in predictions while continuing to use encoded `0/1` labels internally.
-   **mRMR**: Ensured encoded feature and target arrays are integer-coded for both integer and float-coded discrete inputs, and moved parameter validation out of `__init__` for better scikit-learn compatibility.
-   **CFS**: Initialized the GPU feature-feature correlation matrix and added validation for backend, `n_jobs`, and transform-time feature counts.

### Changed

-   **SURF CPU**: Reduced memory pressure by removing per-sample `n_samples x n_features` scratch allocation during scoring.
-   **Packaging**: Tightened dependency bounds to avoid incompatible NumPy/Numba/SciPy installs and updated Ruff configuration to the current `[tool.ruff.lint]` layout.

### Testing

-   Added reference tests for MultiSURF and MultiSURF\* scoring.
-   Added CPU/GPU equivalence tests for MultiSURF, ReliefF, and mutual-information CUDA paths, including a feature-tile boundary regression.
-   Added regression coverage for MDR label preservation and mRMR float-coded discrete inputs.
-   Verified the suite in a clean local virtual environment and checked CUDA kernels with Numba's CUDA simulator.

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
