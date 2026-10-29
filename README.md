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
  
- **Feature-Rich Implementation:** The complete Relief family (ReliefF, RReliefF for regression, SURF, SURF*, MultiSURF, MultiSURF*, TuRF), information-theoretic selectors (mRMR, JMI, CMIM, FCBF, MIFS, CIFE, DISR, CFS), MDR, Gini index, Laplacian Score, Chi2, and continuous/regression mutual-information score functions.
  
- **Paper-Faithful:** Every algorithm is verified against independent, equation-driven tests derived from its defining paper, not from another library's output. Where an extension goes beyond the published algorithm, the docstring says so.
  
- **Lightweight & Simple:** Depends only on NumPy, Numba, SciPy, and scikit-learn. No TensorFlow, no PyTorch, no CuPy.
  
<!-- end-include -->

---

## **Table of Contents**

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Switching from scikit-rebate or scikit-learn](#switching-from-scikit-rebate-or-scikit-learn)
4. [Backend Selection](#backend-selection-cpu-vs-gpu)
5. [Benchmarking Highlights](#benchmarking-highlights)
6. [Algorithm Implementations](#algorithm-implementations)
7. [Versioning and Stability](#versioning-and-stability)
8. [Contributing](#contributing)
9. [License](#license)
10. [How to Cite](#citing-fast-select)
11. [Acknowledgments](#acknowledgments)

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

## **Switching from scikit-rebate or scikit-learn**

**From scikit-rebate.** The Relief estimators use the same fitted attributes
(`feature_importances_`, `top_features_`) and the same scikit-learn interface,
but some parameter names and defaults differ:

| scikit-rebate | fast-select | Note |
|---|---|---|
| `ReliefF`, `SURF`, `MultiSURF` | `ReliefF`, `SURF`, `MultiSURF` | Same names. |
| `SURFstar`, `MultiSURFstar` | `SURF(use_star=True)`, `MultiSURF(use_star=True)` | One class per method with a `use_star` switch. |
| `categorical_threshold=10` | `discrete_limit=10` | Same rule: a feature with at most this many distinct values is discrete. |
| `n_neighbors=100` (default) | `n_neighbors=3` (default) | Pass `n_neighbors=100` to keep the scikit-rebate setting. |
| `n_features_to_select=10` (default) | `n_features_to_select=0.2` (default) | An int is a count; a float is a fraction of features. |
| `n_jobs=1` (default) | `n_jobs=-1` (default) | `-1` uses every Numba thread. |
| `skrebate.turf.TURF(relief_object, pct=0.5, num_scores_to_return=100)` | `TuRF(estimator, pct_remove=0.5, n_features_to_select=100)` | Both remove a fraction of features per iteration; the fast-select default is `pct_remove=0.1`. An integer `pct` has no equivalent. |
| Continuous endpoint (`ReliefF` on a regression target) | `RReliefF` | A separate estimator for regression targets; see below. |
| Missing values (NaN) | `ReliefF` only, discrete features only | Handled as the ReliefF papers define them; see below. |
| `multiclass_threshold`, `weight_final_scores`, `rank_absolute`, `label_type` | — | No equivalent. |

```python
# Before: from skrebate import ReliefF
from fast_select import ReliefF

selector = ReliefF(n_features_to_select=10, n_neighbors=100, discrete_limit=10)
```

fast-select implements each method from its defining paper, so scores can
differ slightly from scikit-rebate where the two implementations differ.

For a continuous target, use `RReliefF` (Robnik-Šikonja and Kononenko), the
regression form of ReliefF. Its defaults are the paper's: 70 neighbours whose
influence decays with rank (`sigma=20`); `weighting="constant"` gives every
neighbour equal weight, for which the paper recommends `n_neighbors=10`. It
runs on the CPU and works inside `TuRF`.

`ReliefF` accepts missing values (NaN) in discrete features, such as missing
genotype calls. As the ReliefF papers define it, a missing value is compared
through the class-conditional probability that the two values differ,
estimated from the known values; this applies to the neighbor distances as
well as the scores. The papers define this only for discrete features, so NaN
in a continuous feature is an error, as is NaN for `SURF` and `MultiSURF`.
Data with missing values runs on the CPU.

```python
from fast_select import RReliefF

selector = RReliefF(n_features_to_select=10).fit(X, y_continuous)
```

**From scikit-learn.** `fast_select.mutual_info_classif` and
`fast_select.mutual_info_regression` take the same arguments as their
`sklearn.feature_selection` namesakes and work with `SelectKBest`; only the
`n_jobs` default differs (`-1` instead of `None`). scikit-learn has no
greedy relevance–redundancy selectors; for those, use `mRMR`, `JMI`, `CMIM`,
`FCBF`, `MIFS`, `CIFE`, `DISR` or `CFS` directly.

```python
# Before: from sklearn.feature_selection import mutual_info_classif
from fast_select import mutual_info_classif
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(mutual_info_classif, k=10)
```

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
from fast_select import MultiSURF

# Force CPU usage
cpu_selector = MultiSURF(n_features_to_select=10, backend='cpu')

# Force GPU usage
gpu_selector = MultiSURF(n_features_to_select=10, backend='gpu')
```

---

## **Benchmarking Highlights**

Fast-Select delivers substantial runtime and memory improvements through its
Numba, CUDA and Metal kernels. CPU and GPU paths are checked against independent,
equation-driven algorithm tests rather than assuming an external implementation
is authoritative. Sustained comparison scripts and raw results are available in
the [benchmarking directory](./benchmarking).

The charts below time the Relief-family selectors against
[scikit-rebate](https://github.com/EpistasisLab/scikit-rebate) 0.8.4 on one
Apple M1 Pro with 8 threads. FastSelect's CPU backend is 61–117× faster at
5,000 samples and 294–878× faster at 10,000 features. ReliefF uses 10 neighbors in
both libraries, and scikit-rebate gets the faster of `n_jobs=1` and all cores
at every size. Each point is the median of three rounds of repeated fits in
fresh processes; the raw timings are in
[`benchmarking/results/readme_scaling.json`](benchmarking/results/readme_scaling.json).
Rerun with `python benchmarking/benchmarking.py`, then (with matplotlib)
`python benchmarking/plots.py`. The CUDA backend needs NVIDIA hardware and is
not shown. Other measurements are in the
[Metal backend report](benchmarking/results/METAL_BACKENDS.md), the
[ReliefF Metal report](benchmarking/results/RELIEFF_METAL.md), the
[first](benchmarking/results/NEW_ALGORITHMS.md) and
[second](benchmarking/results/NEW_ALGORITHMS_2.md) new-algorithm reports, and
the [v0.4.1 CPU optimization report](benchmarking/results/V041_OPTIMIZATIONS.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_n_dominant_runtime_dark.png">
    <img alt="Runtime against sample count for ReliefF, SURF and MultiSURF on log-log axes. At 5,000 samples FastSelect on CPU takes 90 to 190 ms and scikit-rebate 8 to 15 s, 61 to 117 times slower." width="100%" src="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_n_dominant_runtime.png">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_p_dominant_runtime_dark.png">
    <img alt="Runtime against feature count for ReliefF, SURF and MultiSURF on log-log axes. At 10,000 features FastSelect on CPU takes 7 to 18 ms and scikit-rebate 5 to 6 s, 294 to 878 times slower." width="100%" src="https://raw.githubusercontent.com/GavinLynch04/FastSelect/main/benchmarking/benchmark_p_dominant_runtime.png">
  </picture>
</p>

---

## **Algorithm Implementations**

Currently supported:

- **Relief-Family Algorithms:**
  - ReliefF
  - RReliefF (regression targets, CPU)
  - SURF
  - SURF*
  - MultiSURF
  - MultiSURF*
  - TuRF
- **Correlation-Based Feature Selection (CFS)**
- **Multifactor Dimensionality Reduction (MDR)**
- **Minimum Redundancy Maximum Relevance (mRMR)**
- **Chi Squared (Chi2)** — dense or SciPy sparse count matrices; sparse input is counted on the CPU without densifying and gives the same statistics as the dense array.
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
the original algorithm. The rules are written down in the
[algorithm policy](./docs/development/algorithm-policy.md) and enforced by `tests/test_algorithm_compliance.py`,
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

Reported test coverage is host-side Python line coverage from the Linux CI
runners only. Numba-compiled kernels (marked `# pragma: no cover`) cannot be
traced by coverage tools and are verified by the equation-oracle tests instead.
The Metal modules (`fast_select/_*metal.py`) are omitted from the report
because the Linux runners have no Apple GPU; they are tested locally on Apple
Silicon against the CPU backend and the oracles. CUDA-only and Metal-only
branches in the estimators still count as uncovered.

**Numerical policy.** Relief-family features are range-normalised in float64
(continuous) or replaced by exact category codes (discrete) before any narrowing
to float32, on the CPU, CUDA and Metal backends. SURF and SURF* compute the global
radius and all pair distances in float64 and treat a distance within a relative
`1e-12` of the radius as a tie, which is excluded from both the near and far
sets; the remaining Relief-family kernels score in float32. Metal computes
distances in float32 with a conservative error bound and recomputes any
neighbor or threshold decision inside that bound on the CPU in float64, so it
makes the same decisions as the CPU backend.

---

## **Contributing**

Contributions are highly encouraged. Whether you're fixing bugs, improving performance, or proposing new algorithms, your work is invaluable.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the development setup, the checks to run, and the extra requirements for algorithm and performance changes: every algorithm change cites its defining paper and adds an independent equation-derived oracle test, and every performance change needs sustained, machine-readable benchmark evidence. Everyone taking part is expected to follow the [Code of Conduct](./CODE_OF_CONDUCT.md).

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
