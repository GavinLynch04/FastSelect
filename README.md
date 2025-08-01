# **Fast-Select: Accelerated Feature Selection for Modern Datasets**
[![PyPI version](https://img.shields.io/pypi/v/fast-select?color=blue)](https://pypi.org/project/fast-select/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/GavinLynch04/FastSelect/python-tests.yml?branch=main)](https://github.com/GavinLynch04/FastSelect/actions)
[![codecov](https://codecov.io/gh/GavinLynch04/FastSelect/branch/main/graph/badge.svg?token=3LKYFCFSB4)](https://codecov.io/gh/GavinLynch04/FastSelect)
[![Python Versions](https://img.shields.io/pypi/pyversions/fast-select.svg)](https://pypi.org/project/fast-select/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/GavinLynch04/FastSelect/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/1018195486.svg)](https://doi.org/10.5281/zenodo.16285073)
<!-- start-include -->
A high-performance Python library powered by **Numba** and **CUDA**, offering accelerated algorithms for feature selection. Initially built to optimize the complete Relief family of algorithms, `fast-select` aims to expand and accelerate a wide range of feature selection methods to empower machine learning on large-scale datasets.

---

## **Key Features**

- **Fast Performance:** Leverages **Numba** for JIT compilation, **Joblib** for multi-core parallelism, and **Numba CUDA** for GPU acceleration, providing unmatched performance while scaling with modern hardware.
  
- **ML Pipeline Integration:** Fully compatible with **Scikit-Learn**, making it easy to fit into any machine learning pipeline with a familiar `.fit()`, `.transform()`, `.fit_transform()` interface.
  
- **Flexible Backends:** Offers dual execution modes for both CPU (`Joblib`) and GPU (`CUDA`). Automatically detects hardware with an easy-to-use `backend` parameter.
  
- **Feature-Rich Implementation:** Provides highly optimized implementations of ReliefF, SURF, SURF*, MultiSURF, MultiSURF*, and TuRF, with plans to support additional feature selection algorithms in future releases.
  
- **Lightweight & Simple:** Avoids heavy dependencies like TensorFlow or PyTorch while delivering significant speedups for feature selection workflows.
  
<!-- end-include -->

---

## **Table of Contents**

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Backend Selection](#backend-selection-cpu-vs-gpu)
4. [Benchmarking Highlights](#benchmarking-highlights)
5. [Algorithm Implementations](#algorithm-implementations)
6. [Contributing](#contributing)
7. [License](#license)
8. [How to Cite](#how-to-cite)
9. [Acknowledgments](#acknowledgments)

---

## **Installation**
<!-- start-installation-section -->

Install `fast-select` directly from PyPI:

```bash
pip install fast-select
```

For development versions (with testing and documentation dependencies):

```bash
git clone https://github.com/GavinLynch04/FastSelect.git
cd fast-select
pip install -e .[dev]
```

<!-- end-installation-section -->

---

## **Quickstart**
<!-- start-quickstart-section -->

Using `fast-select` is simple and seamless for anyone familiar with Scikit-Learn.

```python
from fast_select import MultiSURF
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Generate a synthetic dataset
X, y = make_classification(
    n_samples=500, 
    n_features=1000, 
    n_informative=20, 
    n_redundant=100, 
    random_state=42
)

# 2. Use the MultiSURF estimator to select the top 15 features
selector = MultiSURF(n_features_to_select=15)
X_selected = selector.fit_transform(X, y)
print(f"Original feature count: {X.shape[1]}")
print(f"Selected feature count: {X_selected.shape[1]}")
print(f"Top 15 feature indices: {selector.top_features_}")

# 3. Integrate into a Scikit-Learn Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selector', MultiSURF(n_features_to_select=10, backend='cpu')),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
# pipeline.fit(X, y)
```
<!-- end-quickstart-section -->

---

## **Backend Selection (CPU vs. GPU)**

You can control the computational backend with the `backend` parameter during initialization:

- **`backend='auto'`**: Automatically detects if an NVIDIA GPU is available. Falls back to CPU if a GPU is not available.
  
- **`backend='gpu'`**: Explicitly runs on GPU. Will raise a `RuntimeError` if no compatible GPU is found.
  
- **`backend='cpu'`**: Forces CPU computations, even if a GPU is available.

Example usage:

```python
# Force CPU usage
cpu_selector = MultiSURF(n_features_to_select=10, backend='cpu')

# Force GPU usage
gpu_selector = MultiSURF(n_features_to_select=10, backend='gpu')
```

---

## **Benchmarking Highlights**

Fast-Select delivers groundbreaking improvements in runtime and memory efficiency. Benchmarks show **50-100x speed-ups** compared to `scikit-rebate` and R's `CORElearn` library, particularly on large datasets exceeding 10,000 samples and features. [Benchmarking scripts](./benchmarking) are available in the repository for further testing.

#### Runtime vs. Number of Samples (n >> p)

<p align="center">
  <img alt="Runtime Benchmark N-Dominant" width="700" src="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_n_dominant_runtime.png">
</p>

#### Runtime vs. Number of Features (p >> n)

<p align="center">
  <img alt="Memory Benchmark P-Dominant" width="700" src="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_p_dominant_runtime.png">
</p>

---

## **Algorithm Implementations**

Currently supported:

- **Relief-Family Algorithms:**
  - ReliefF
  - SURF
  - SURF*
  - MultiSURF
  - MultiSURF*
  - TuRF
- **Correlation-Based Feature Selection (CFS)**
- **Multifactor Dimensionality Reduction (MDR)**
- **Minimum Redundancy Maximum Relevance (mRMR)**
- **Chi Squared (Chi2)**

Future plans include additional feature selection algorithms, such as wrappers, embedded methods, and more filter-based approaches.

---

## **Contributing**

Contributions are highly encouraged. Whether you're fixing bugs, improving performance, or proposing new algorithms, your work is invaluable. Please ensure your submissions include relevant test cases and documentation updates.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for full details.

---

## Citing `fast-select`

If you use `fast-select` in your research or work, please cite it using the following DOI. This helps to track the impact of the work and ensures its continued development.

> Gavin Lynch. (2025). GavinLynch04/FastSelect: v0.2.0 (0.2.0). Zenodo. [https://doi.org/10.5281/zenodo.16285073](https://doi.org/10.5281/zenodo.16285073)

You can use the following BibTeX entry:

```bibtex
@software{gavin_lynch_2025,
  author       = {Gavin Lynch},
  title        = {{GavinLynch04/FastSelect: v0.2.0}},
  month        = aug,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.2.0},
  doi          = {10.5281/zenodo.16285073},
  url          = {https://doi.org/10.5281/zenodo.16285073}
}
```

---

## **Acknowledgments**

This library builds on the exceptional work of the following:

- The **Numba** team for enabling JIT compilation and GPU acceleration.
- The **scikit-rebate** authors for their inspiring Relief-based library.
- The original researchers behind the Relief algorithms for their foundational contributions to feature selection.
- The original authors and researchers behind the various algorithms implemented in this library.
