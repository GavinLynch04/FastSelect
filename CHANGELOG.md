# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

