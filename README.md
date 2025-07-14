# Fast-Relief: Numba Optimized Relief-Based Feature Selection

[![PyPI version](https://img.shields.io/pypi/v/fast-relief.svg)](https://pypi.org/project/fast-relief/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-username/fast-relief/ci.yml?branch=main)](https://github.com/your-username/fast-relief/actions)
[![Python Versions](https://img.shields.io/pypi/pyversions/fast-relief.svg)](https://pypi.org/project/fast-relief/)
[![License](https://img.shields.io/pypi/l/fast-relief.svg)](https://github.com/your-username/fast-relief/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/your-zenodo-doi.svg)](https://doi.org/your-zenodo-doi)

A high-performance, Numba and CUDA-accelerated Python implementation of the complete Relief family of feature selection algorithms. `fast-relief` is designed for speed and scalability, making it possible to apply these powerful algorithms to modern, large-scale bioinformatics datasets.

![Benchmark Performance Figure](https://raw.githubusercontent.com/your-username/fast-relief/main/docs/assets/benchmark_figure.png)
*(This figure shows a **50-100x speed-up** over existing libraries on a large-scale dataset.)*

## Key Features

*   **Fast Performance:** Utilizes **Numba** and **Job-lib** for multi-core CPU acceleration and **Numba CUDA** for massive GPU parallelization, dramatically outperforming existing implementations.
*   **Scikit-Learn Compatible API:** Designed as a drop-in replacement for `sklearn` transformers with a familiar `.fit()`, `.transform()`, and `.fit_transform()` interface. Easily integrates into existing ML pipelines.
*   **Dual CPU/GPU Backends:** Intelligently auto-detects a compatible GPU or allows the user to explicitly select the `'cpu'` or `'gpu'` backend.
*   **Comprehensive Algorithm Support:** Provides optimized implementations for ReliefF, SURF, SURF*, MultiSURF, and MultiSURF*.
*   **Lightweight & Accessible:** Avoids heavy dependencies like PyTorch or TensorFlow, making it easy to install and use for any bioinformatician.

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Backend Selection](#backend-selection-cpu-vs-gpu)
- [Benchmarking Highlights](#benchmarking-highlights)
- [Algorithm Implementations](#algorithm-implementations)
- [Contributing](#contributing)
- [License](#license)
- [How to Cite](#how-to-cite)
- [Acknowledgments](#acknowledgments)

## Installation

You can install `fast-relief` directly from PyPI:

```bash
pip install fast-relief
```

