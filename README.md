# Fast-Relief: Numba Optimized Relief-Based Feature Selection

[![PyPI version](https://img.shields.io/pypi/v/fast-relief.svg)](https://pypi.org/project/fast-relief/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-username/fast-relief/ci.yml?branch=main)](https://github.com/GavinLynch04/FastRelief/actions)
[![Python Versions](https://img.shields.io/pypi/pyversions/fast-relief.svg)](https://pypi.org/project/fast-relief/)
[![License](https://img.shields.io/pypi/l/fast-relief.svg)](https://github.com/GavinLynch04/FastRelief/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/your-zenodo-doi.svg)](https://doi.org/your-zenodo-doi)
<!-- start-include -->
A high-performance, Numba and CUDA-accelerated Python implementation of the complete Relief family of feature selection algorithms. `fast-relief` is designed for speed and scalability, making it possible to apply these powerful algorithms to modern, large-scale bioinformatics datasets.

![Benchmark Performance Figure](https://raw.githubusercontent.com/your-username/fast-relief/main/docs/assets/benchmark_figure.png)
*(This figure shows a **50-100x speed-up** over existing libraries on a large-scale dataset.)*

## Key Features

*   **Fast Performance:** Utilizes **Numba** and **Job-lib** for multi-core CPU acceleration and **Numba CUDA** for massive GPU parallelization, dramatically outperforming existing implementations.
*   **Scikit-Learn Compatible API:** Designed as a drop-in replacement for `sklearn` transformers with a familiar `.fit()`, `.transform()`, and `.fit_transform()` interface. Easily integrates into existing ML pipelines.
*   **Dual CPU/GPU Backends:** Intelligently auto-detects a compatible GPU or allows the user to explicitly select the `'cpu'` or `'gpu'` backend.
*   **Comprehensive Algorithm Support:** Provides optimized implementations for ReliefF, SURF, SURF*, MultiSURF, and MultiSURF*.
*   **Lightweight & Accessible:** Avoids heavy dependencies like PyTorch or TensorFlow, making it easy to install and use for any bioinformatician.
<!-- end-include -->
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

For developers, install with all testing and documentation dependencies:
```bash
git clone https://github.com/GavinLynch04/FastRelief.git
cd fast-relief
pip install -e .[dev]
```

## Quickstart

Using `fast-relief` is designed to be simple and familiar for anyone who has used scikit-learn.

```python
from fast_relief.estimators import MultiSURF
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Example classifier

# 1. Generate a synthetic dataset
X, y = make_classification(
    n_samples=500,
    n_features=1000,
    n_informative=20,
    n_redundant=100,
    random_state=42
)

# 2. Use the estimator to select the top 15 features
# The backend will default to 'auto' (uses GPU if available)
selector = MultiSURF(n_features_to_select=15)

X_selected = selector.fit_transform(X, y)

print(f"Original feature count: {X.shape[1]}")
print(f"Selected feature count: {X_selected.shape[1]}")
print(f"Top 15 feature indices: {selector.top_features_}")

# 3. Integrate directly into a scikit-learn Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selector', MultiSURF(n_features_to_select=10, backend='cpu')),
    ('classifier', LogisticRegression())
])

# The pipeline now uses fast-relief for feature selection!
# pipeline.fit(X, y)
```

## Backend Selection (CPU vs. GPU)

You can control the computational backend using the `backend` parameter during initialization.

*   `backend='auto'` (Default): `fast-relief` will automatically detect if a compatible NVIDIA GPU is available via Numba. If so, it will run on the GPU. Otherwise, it will seamlessly fall back to the multi-core CPU implementation.
*   `backend='gpu'`: Forces the use of the GPU. Will raise a `RuntimeError` if a compatible GPU is not found.
*   `backend='cpu'`: Forces the use of the CPU, even if a GPU is available.

```python
# Force CPU usage
cpu_selector = MultiSURF(n_features_to_select=10, backend='cpu')

# Force GPU usage
gpu_selector = MultiSURF(n_features_to_select=10, backend='gpu')
```

## Benchmarking Highlights

`fast-relief` provides a significant performance leap, enabling analysis on datasets that were previously impossible for Relief-based methods.

![Benchmark Performance Figure](https://raw.githubusercontent.com/your-username/fast-relief/main/docs/assets/benchmark_figure.png)

Our benchmarks against `scikit-rebate` and R's `CORElearn` package show **up to a 50-100x reduction in runtime** and a significant decrease in peak memory usage, especially on large datasets (>10,000 samples/features). Full benchmarking scripts can be found in the `/benchmarks` directory.

## Algorithm Implementations

This library provides optimized versions of the most common Relief-based algorithms. We have paid careful attention to the original academic definitions and the practical implementations in popular libraries.

## Contributing

Contributions are welcome and greatly appreciated. Please feel free to submit a pull request that adheres to the testing standards and is well documented. These algorithms have been highly optimized over previous implementations, but there is certainly huge room for improvement.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## How to Cite

If you use `fast-relief` in your research, please cite both the software and our publication.

**1. Citing the Paper (once published):**
```bibtex
@article{yourname_2024_fastrelief,
  author  = {Your Name},
  title   = {{fast-relief: A high-performance Python package for Relief-based feature selection}},
  journal = {Journal of Open Source Software},
  year    = {2024},
  doi     = {your_paper_doi},
  url     = {https://your_paper_url}
}
```

**2. Citing the Software (specific version):**
Please cite the specific version of the software you used, which can be found using the Zenodo DOI provided on our GitHub releases page.

## Acknowledgments

This work would not be possible without the foundational contributions of the following projects:
*   The **Numba** team for creating an incredible JIT compiler.
*   The **scikit-rebate** authors for their excellent and feature-rich library, which served as the primary benchmark.
*   The original authors of the Relief family of algorithms for their pioneering work in feature selection.
```
