---
title: "Fast‑Select: Accelerated Feature Selection for Modern Datasets"

tags:
- python
- feature selection
- bioinformatics
- GPU
- machine learning

authors:
- name: Gavin Lynch
  orcid: "0009-0006-0097-4157"
  affiliation: 1

affiliations:
- name: Department of Computer Science, California Polytechnic State University San Luis Obispo, San Luis Obispo, USA
  index: 1
  ror: 001gpfp45
date: 04 August 2025

bibliography: paper.bib

---

# Summary

`Fast‑Select` is an open‑source Python package that delivers *order‑of‑magnitude* speed‑ups for classical feature‑selection algorithms by combining Just‑In‑Time (JIT) compilation via **Numba**, optional **CUDA** kernels, and multi‑processing with **Joblib**.  The library currently ships highly‑optimized implementations of the Relief family (ReliefF, SURF, SURF\*, MultiSURF, TuRF) plus CFS, mRMR, Chi‑squared and MDR, wrapped in a **scikit‑learn‑compatible API**.  It targets modern, high‑dimensional biological datasets—such as whole‑genome variant matrices or single‑cell expression counts—where traditional CPU‑bound methods become a bottleneck.

# Statement of Need

Typical omics studies now profile **10⁴–10⁶ features** across thousands of samples.  Existing Python toolkits (e.g. *scikit‑rebate*) scale poorly beyond \~50k features and lack GPU support, limiting their utility in genomics and metagenomics.  `Fast‑Select` fills this gap by:

* providing drop‑in replacements for widely‑used filter methods with GPU acceleration;
* exposing an identical API for CPU and GPU back‑ends, easing adoption in reproducible pipelines;
* offering benchmarks and container images that enable transparent performance evaluation.

# Implementation and Architecture

The package is implemented in Python ≥3.9.  Core numerical kernels are written in Numba‑typed functions that compile to machine‑code at runtime.  When an NVIDIA GPU is detected, memory‑bound computations are transparently off‑loaded to CUDA kernels.  A lightweight Cython shim provides high‑level wrappers conforming to `sklearn.base.BaseEstimator`.  Continuous integration (GitHub Actions) runs unit tests across Linux, macOS and Windows.

# Performance

On a 30000‑sample ×200000‑feature synthetic dataset, `Fast‑Select`’s MultiSURF implementation runs **88× faster on GPU** and **12× faster on 16‑core CPU** relative to *scikit‑rebate* (v1.3).  Detailed benchmark scripts and raw results are hosted in the repository.

# Quality Control

* 96% branch coverage via `pytest` and `coverage.py`.
* Static type checking with `mypy` and style enforcement with `ruff` & `black`.
* Continuous deployment builds PyPI wheels and pushes version‑tagged Docker images.

# (Optional) Example

```python
from fast_select import MultiSURF
X_selected = MultiSURF(n_features_to_select=50, backend="gpu").fit_transform(X, y)
```

# Acknowledgements

We thank the *Numba* developers and the maintainers of *scikit‑rebate* for foundational contributions.  Early adopters at the Example Genomics Lab provided invaluable beta feedback.

# References

```bibtex
@software{lynch_fastselect_2025,
  author    = {Gavin Lynch},
  title     = {FastSelect: v0.2.0},
  month     = aug,
  year      = 2025,
  publisher = {Zenodo},
  version   = {0.2.0},
  doi       = {10.5281/zenodo.16285073},
  url       = {https://doi.org/10.5281/zenodo.16285073}
}
```
