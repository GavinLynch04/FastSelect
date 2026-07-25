# **Fast-Select: Accelerated Feature Selection for Modern Datasets**
[![PyPI version](https://img.shields.io/pypi/v/fast-select?color=blue)](https://pypi.org/project/fast-select/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/GavinLynch04/FastSelect/python-tests.yml?branch=main)](https://github.com/GavinLynch04/FastSelect/actions)
[![codecov](https://codecov.io/gh/GavinLynch04/FastSelect/branch/main/graph/badge.svg?token=3LKYFCFSB4)](https://codecov.io/gh/GavinLynch04/FastSelect)
[![Python Versions](https://img.shields.io/pypi/pyversions/fast-select.svg)](https://pypi.org/project/fast-select/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/GavinLynch04/FastSelect/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/1018195486.svg)](https://doi.org/10.5281/zenodo.16285073)
[![Downloads](https://static.pepy.tech/badge/fast-select)](https://pepy.tech/project/fast-select)
<!-- start-include -->
A high-performance Python library powered by **Numba** and **CUDA**, offering accelerated algorithms for feature selection. Initially built to optimize the complete Relief family of algorithms, `fast-select` aims to expand and accelerate a wide range of feature selection methods to empower machine learning on large-scale datasets.

---

## **Key Features**

- **Fast Performance:** Leverages **Numba** for JIT compilation and thread-parallel CPU kernels, and **Numba CUDA** for GPU acceleration, scaling with modern hardware.
  
- **ML Pipeline Integration:** Fully compatible with **Scikit-Learn**, making it easy to fit into any machine learning pipeline with a familiar `.fit()`, `.transform()`, `.fit_transform()` interface.
  
- **Flexible Backends:** Offers dual execution modes for both CPU (Numba `prange`) and GPU (`CUDA`). Automatically detects hardware with an easy-to-use `backend` parameter.
  
- **Feature-Rich Implementation:** Provides highly optimized implementations of ReliefF, SURF, SURF*, MultiSURF, MultiSURF*, and TuRF, with plans to support additional feature selection algorithms in future releases.
  
- **Paper-Faithful:** Every algorithm is verified against independent, equation-driven tests derived from its defining paper, not from another library's output. Where an extension goes beyond the published algorithm, the docstring says so.
  
- **Lightweight & Simple:** Depends only on NumPy, Numba, SciPy, and scikit-learn. No TensorFlow, no PyTorch, no CuPy.
  
<!-- end-include -->

---

## **Table of Contents**

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Backend Selection](#backend-selection-cpu-vs-gpu)
4. [Benchmarking Highlights](#benchmarking-highlights)
5. [Algorithm Implementations](#algorithm-implementations)
6. [Versioning and Stability](#versioning-and-stability)
7. [Contributing](#contributing)
8. [License](#license)
9. [How to Cite](#citing-fast-select)
10. [Acknowledgments](#acknowledgments)

---

## **Installation**
<!-- start-installation-section -->

Install `fast-select` directly from PyPI:

```bash
pip install fast-select
```

That single command gives you the full library, CPU backend included.

**GPU support** needs no extra Python package. The CUDA kernels are compiled by
`numba.cuda`, so all that is required is an NVIDIA GPU with a current driver and
a CUDA toolkit that your installed Numba supports. Check that the backend is
visible with:

```bash
python -c "from fast_select.utils import is_cuda_ready; print(is_cuda_ready())"
```

For a development checkout (tests, linters, and documentation tooling):

```bash
git clone https://github.com/GavinLynch04/FastSelect.git
cd FastSelect
pip install -e ".[dev]"
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

For the mutual-information selectors (`mRMR`) the CUDA kernels support up to 32
distinct states in the encoded data. With `backend='auto'` more states than that
fall back to the CPU, and `backend='gpu'` raises instead of falling back. After
`fit`, `effective_backend_` reports which backend actually ran.

Example usage:

```python
# Force CPU usage
cpu_selector = MultiSURF(n_features_to_select=10, backend='cpu')

# Force GPU usage
gpu_selector = MultiSURF(n_features_to_select=10, backend='gpu')
```

---

## **Benchmarking Highlights**

Fast-Select delivers substantial runtime and memory improvements through its
Numba and CUDA kernels. CPU and GPU paths are checked against independent,
equation-driven algorithm tests rather than assuming an external implementation
is authoritative. Sustained comparison scripts and raw results are available in
the [benchmarking directory](./benchmarking).

#### Runtime vs. Number of Samples (n >> p)

<p align="center">
  <img alt="Runtime Benchmark N-Dominant" width="700" src="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_n_dominant_runtime.png">
</p>

#### Runtime vs. Number of Features (p >> n)

<p align="center">
  <img alt="Runtime Benchmark P-Dominant" width="700" src="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_p_dominant_runtime.png">
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
- **Discrete mutual information** — `calculate_mi_single_pair` and `calculate_mi_matrices`, with the same CPU/GPU backend selection and a choice of bit or nat units.

Each implementation is held to its defining paper. Where `fast-select` supports
more than the paper does — multiclass targets for the SURF family, for
instance — the docstring names it as an extension rather than presenting it as
the original algorithm. The rules are written down in
[`CLAUDE.md`](./CLAUDE.md) and enforced by `tests/test_algorithm_compliance.py`,
whose oracles are computed straight from the published equations and never call
production kernels.

Future plans include additional feature selection algorithms, such as wrappers, embedded methods, and more filter-based approaches.

---

## **Versioning and Stability**

`fast-select` follows [Semantic Versioning](https://semver.org/). As of v1.0.0
the public API — estimator names, constructor parameters, and fitted attributes —
is stable, and breaking changes require a major version bump.

**Upgrading from 0.2.x:** v1.0.0 corrects several algorithms that had drifted
from their defining papers, so SURF, SURF*, MultiSURF*, CFS, MDR, and GPU mRMR
can return different scores and different selected features than 0.2.1 did.
`pandas` is also no longer a required dependency. See the
[changelog](./CHANGELOG.md) for the full list before comparing new output
against results you generated with an earlier release.

---

## **Contributing**

Contributions are highly encouraged. Whether you're fixing bugs, improving performance, or proposing new algorithms, your work is invaluable.

Before opening a pull request:

```bash
pip install -e ".[dev]"
ruff check src tests benchmarking
black --check src tests benchmarking
pytest
```

Algorithm changes carry an extra requirement: cite the paper and the specific
equation or pseudocode rule you are implementing, and add an independent oracle
test that derives the expected result from that equation rather than from
`fast-select`'s own output. [`CLAUDE.md`](./CLAUDE.md) has the details.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for full details.

---

## Citing `fast-select`

If you use `fast-select` in your research or work, please cite it using the following DOI. This helps to track the impact of the work and ensures its continued development.

> Gavin Lynch. (2026). GavinLynch04/FastSelect: v1.0.0 (1.0.0). Zenodo. [https://doi.org/10.5281/zenodo.16285073](https://doi.org/10.5281/zenodo.16285073)

You can use the following BibTeX entry:

```bibtex
@software{gavin_lynch_2026,
  author       = {Gavin Lynch},
  title        = {{GavinLynch04/FastSelect: v1.0.0}},
  month        = jul,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
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
