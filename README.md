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
A high-performance Python library powered by **Numba**, **CUDA**, and optional **Metal/MLX**, offering accelerated algorithms for feature selection. Initially built to optimize the complete Relief family of algorithms, `fast-select` aims to expand and accelerate a wide range of feature selection methods to empower machine learning on large-scale datasets.

---

## **Key Features**

- **Fast Performance:** Leverages **Numba** for JIT compilation and thread-parallel CPU kernels, **Numba CUDA** for NVIDIA GPUs, and optional **MLX/Metal** for Apple Silicon GPUs.
  
- **ML Pipeline Integration:** Fully compatible with **Scikit-Learn**, making it easy to fit into any machine learning pipeline with a familiar `.fit()`, `.transform()`, `.fit_transform()` interface.
  
- **Flexible Backends:** Supports CPU (Numba `prange`), NVIDIA GPUs (CUDA), and Apple Silicon GPUs (optional Metal/MLX). `backend="auto"` picks a GPU only where it has been shown to help.
  
- **Feature-Rich Implementation:** The complete Relief family (ReliefF, SURF, SURF*, MultiSURF, MultiSURF*, TuRF), information-theoretic selectors (mRMR, JMI, CMIM, FCBF, MIFS, CIFE, DISR, CFS), MDR, Gini index, Laplacian Score, Chi2, and continuous/regression mutual-information score functions.
  
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

Supports standard CPython 3.9–3.14. CI runs the test suite on each of these
versions. Python 3.13 requires Numba 0.61 or newer, and Python 3.14 requires
Numba 0.63 or newer; pip selects these automatically. Python prereleases and
free-threaded builds are not currently part of the supported test matrix.

Install `fast-select` directly from PyPI:

```bash
pip install fast-select
```

That single command gives you the full library, CPU backend included.

**Apple Silicon GPU support** (Python 3.10+, arm64 macOS) is an optional extra
that installs [MLX](https://github.com/ml-explore/mlx):

```bash
pip install "fast-select[metal]"
```

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

- **`backend='auto'`**: Prefers a usable CUDA backend, then an installed and usable Metal backend (where that algorithm's Metal path is enabled for `auto`), then CPU.
  
- **`backend='gpu'`**: Explicitly requests NVIDIA/CUDA. Raises if unavailable; it does not silently switch to Metal.
  
- **`backend='cpu'`**: Forces CPU computations, even if a GPU is available.

- **`backend='metal'`**: Explicitly requests Apple Silicon GPU execution through MLX. Raises if unavailable or unsupported.

Metal is available for **ReliefF, SURF/SURF*, MultiSURF/MultiSURF*, discrete
mutual information, mRMR (MID/MIQ), CFS, MDR, and Chi2**; TuRF uses its wrapped
selector's backend on every refit. With `fast-select[metal]` installed and no
CUDA device, `backend="auto"` selects Metal for ReliefF, the SURF/MultiSURF
family, mutual information, mRMR and MDR. **Chi2 and CFS stay on CPU under
`auto`**: their Metal paths were slower than the CPU in sustained benchmarks on
both tall and wide inputs, so they run on Metal only with `backend="metal"`.
Without MLX, `auto` works exactly as before.

Metal results match the CPU backend: precision-sensitive neighbor and boundary
decisions are certified on the GPU or recomputed on the CPU in float64.
ReliefF supports up to 64 neighbors per class; information/CFS kernels support
up to 64 encoded state slots. Unsupported inputs fall back under `auto`, while
explicit `metal` requests raise. After fitting, estimators report the actual
backend in `effective_backend_`.

Continuous-feature MultiSURF and small inputs are faster on CPU; use
`backend="cpu"` for them. See the [all-algorithm Metal report](benchmarking/results/METAL_BACKENDS.md)
and [ReliefF measurements](benchmarking/results/RELIEFF_METAL.md).

The newer **JMI, CMIM, FCBF, LaplacianScore and continuous `mutual_info_classif`**
use parallel Numba CPU kernels. Their GPU prototypes did not consistently beat
those CPU implementations and remain outside the package. For these four
estimators, `backend="auto"` selects CPU; explicit GPU requests raise. See the
[external-baseline admission report](benchmarking/results/NEW_ALGORITHMS.md).

The second admission batch adds **GiniIndex, DISR, CIFE, MIFS and
`mutual_info_regression`**, also using parallel Numba CPU execution. The three
new discrete selectors and GiniIndex accept CPU/auto; the regression score
function is CPU-only. These methods passed comparisons against tuned
scikit-learn or pinned original scikit-feature. An ANOVA F prototype lost to
scikit-learn and was excluded. See the
[second admission report](benchmarking/results/NEW_ALGORITHMS_2.md).

A [feature-heavy follow-up](benchmarking/results/FEATURE_HEAVY_GPU.md) tests
64×1,024 through 256×8,192 inputs against experimental Metal paths. It shows
GPU opportunities and small-input regressions; later CPU-thread retests were
disrupted by background load, so clean confirmation remains necessary before
package integration. CUDA kernels have simulator correctness checks but no
NVIDIA hardware timing. These research paths are outside the public backends.

For information selectors, CUDA supports up to 32 encoded states. When that
limit is exceeded, `auto` tries Metal (up to 64), then CPU.

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
- **Continuous/discrete classification MI** — `mutual_info_classif`, Ross's nearest-neighbor estimator in nats, for dense numeric inputs and `SelectKBest`.
- **Joint Mutual Information (JMI)** — categorical classification, normalized joint-information criterion.
- **Conditional MI Maximization (CMIM)** — categorical classification, maximizing the minimum conditional MI.
- **Fast Correlation Based Filter (FCBF)** — categorical classification, symmetrical-uncertainty relevance and redundancy filtering.
- **Laplacian Score** — unsupervised local-structure ranking, with Euclidean binary/heat-kernel kNN or a supplied symmetric affinity graph.
- **Regression mutual information** — `mutual_info_regression`, exact KSG nearest-neighbor MI in nats for dense numeric regression inputs, with optional discrete feature columns handled by Ross's estimator.
- **Gini Index** — `GiniIndex`, minimum weighted class impurity over every distinct numeric split; raw `gini_scores_` is lower-is-better and importances negate it.
- **DISR** — Double Input Symmetrical Relevance, summed normalized joint MI for categorical classification.
- **CIFE** — Conditional Informative Feature Extraction criterion, relevance minus summed redundancy plus summed conditional dependence.
- **MIFS** — Battiti's Mutual Information Feature Selection, relevance minus `beta` times summed redundancy (default `beta=0.5`).

```python
from fast_select import JMI, CMIM, FCBF, LaplacianScore, mutual_info_classif
from sklearn.feature_selection import SelectKBest

categorical_selector = JMI(n_features_to_select=10, n_jobs=-1)
unsupervised_selector = LaplacianScore(n_features_to_select=10, n_neighbors=5)
continuous_selector = SelectKBest(mutual_info_classif, k=10)
```

JMI/CMIM use empirical categorical distributions without automatic binning;
encode or discretize continuous measurements deliberately before using them.
Their `top_features_` contains greedy selection order and `selection_scores_`
contains each selected objective; `feature_importances_` contains marginal MI
(SU for FCBF). Laplacian raw `laplacian_scores_` are lower-is-better;
`feature_importances_` negates them. Constant/zero weighted-variance features
score infinity and rank last.

DISR, CIFE and MIFS also take categorical inputs without automatic binning.
They retain the fixed requested feature count and report marginal MI in bits
as importances, with greedy criterion values in `selection_scores_`. DISR's
zero joint-entropy contribution is explicitly defined as zero. MIFS can have
negative selection scores; its redundancy sum is not divided by subset size.
GiniIndex preserves float64 boundaries and integer ordering, including large
uint64 values. Regression MI requires `n_neighbors < n_samples`, returns nats,
and can be passed to `SelectKBest`; set `random_state` for reproducible jitter.
Its exact sorted search can approach quadratic work on adversarial data or
large neighbor counts. No new GPU backend is claimed by this second batch.

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

`fast-select` follows [Semantic Versioning](https://semver.org/) and is currently a
0.x release series: estimator names, constructor parameters, and fitted
attributes may still change between minor versions, and every change is listed
in the [changelog](./CHANGELOG.md). A 1.0.0 release will follow once the CUDA
backend has been verified on physical NVIDIA hardware and the support matrix is
final.

**Upgrading from 0.3.x:** v0.4.0 adds selectors and the optional Metal backend
without changing any existing algorithm; CPU results match 0.3.0. If you install
`fast-select[metal]`, `backend="auto"` may now run some estimators on the Apple
GPU (results match CPU); pass `backend="cpu"` to keep the old placement.

**Upgrading from 0.2.x:** v0.3.0 corrects several algorithms that had drifted
from their defining papers and fixes numerical-precision and input-validation
defects, so ReliefF, SURF, SURF*, MultiSURF, MultiSURF*, CFS, MDR, mRMR, and GPU
mutual information can return different scores and different selected features
than 0.2.1 did. `pandas` is also no longer a required dependency. See the
[changelog](./CHANGELOG.md) for the full list before comparing new output
against results you generated with an earlier release.

**Support matrix.**

| Area | Verified in CI | Not verified |
|---|---|---|
| Python | CPython 3.9–3.14, Linux x86-64 | Free-threaded builds, prereleases, minimum-dependency pins |
| Operating system | Linux | Windows, macOS (the Windows CUDA context workaround is untested in CI) |
| CPU backend | Full test suite, independent equation oracles | — |
| CUDA backend | Kernel logic in the Numba CUDA simulator, against the CPU backend and oracles | Physical NVIDIA GPUs: drivers, races, device-level rounding, performance |
| Metal backend | Not in CI. Full suite and oracle/parity tests run locally on an Apple M1 Pro | Other Apple GPUs, CI coverage |
| Documentation | Strict Sphinx build (warnings are errors) | — |

Reported test coverage is host-side Python line coverage only; the compiled
numerical kernels are excluded from it and are verified by the oracle tests.

**Numerical policy.** Relief-family features are range-normalised in float64
(continuous) or replaced by exact category codes (discrete) before any narrowing
to float32, on both the CPU and CUDA backends. SURF and SURF* compute the global
radius and all pair distances in float64 and treat a distance within a relative
`1e-12` of the radius as a tie, which is excluded from both the near and far
sets; the remaining Relief-family kernels score in float32.

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

Performance changes also need sustained, machine-readable evidence of a runtime
or memory improvement. The [optimization loop](./benchmarking/README.md) documents
clean-process comparisons, complete-output checks, and the `--gate` command that
rejects unrepeatable improvements. A benchmark win does not replace equation
oracles or physical-GPU conformance checks.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for full details.

---

## Citing `fast-select`

If you use `fast-select` in your research or work, please cite it and state the version you
used (`fast_select.__version__`). The DOI below is the Zenodo *concept* DOI: it always
resolves to the newest archived release. Each release also has its own version-specific DOI
on the Zenodo record, which you can cite instead for exact reproducibility.

> Gavin Lynch. FastSelect. Zenodo. [https://doi.org/10.5281/zenodo.16285073](https://doi.org/10.5281/zenodo.16285073)

```bibtex
@software{lynch_fast_select,
  author  = {Gavin Lynch},
  title   = {{FastSelect}},
  year    = {2026},
  version = {<the version you used>},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.16285073},
  url     = {https://doi.org/10.5281/zenodo.16285073}
}
```

---

## **Acknowledgments**

This library builds on the exceptional work of the following:

- The **Numba** team for enabling JIT compilation and GPU acceleration.
- The **scikit-rebate** authors for their inspiring Relief-based library.
- The original researchers behind the Relief algorithms for their foundational contributions to feature selection.
- The original authors and researchers behind the various algorithms implemented in this library.
