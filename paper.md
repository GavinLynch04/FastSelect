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
date: 24 July 2026

bibliography: paper.bib

---

# Summary

`Fast‑Select` is an open‑source Python package that accelerates classical feature‑selection algorithms by combining Just‑In‑Time (JIT) compilation via **Numba**, thread‑parallel CPU kernels, and optional **CUDA** kernels.  The library ships implementations of the Relief family — ReliefF [@kononenko1994relieff], SURF and SURF\* [@greene2009surf], MultiSURF and MultiSURF\* [@urbanowicz2018benchmarking], and TuRF — alongside CFS [@hall1999cfs], mRMR [@peng2005mrmr], Chi‑squared, and MDR [@ritchie2001mdr], all wrapped in a **scikit‑learn‑compatible API**.  It targets modern, high‑dimensional biological datasets—such as whole‑genome variant matrices or single‑cell expression counts—where traditional CPU‑bound methods become a bottleneck.

# Statement of Need

Typical omics studies now profile **10⁴–10⁶ features** across thousands of samples.  Relief‑based methods are attractive here because they detect feature interactions without exhaustively enumerating them [@kira1992relief; @urbanowicz2018review], but their instance‑pair distance computation is quadratic in the sample count.  Existing Python toolkits, notably *scikit‑rebate* [@urbanowicz2018skrebate], are CPU‑bound and offer no GPU path, which limits their utility as dataset sizes grow.  `Fast‑Select` fills this gap by:

* providing scikit‑learn‑compatible filter methods with both JIT‑compiled CPU and CUDA back‑ends;
* exposing an identical API for CPU and GPU back‑ends, easing adoption in reproducible pipelines;
* verifying every algorithm against its defining paper rather than against another implementation, so that a speed‑up never comes at the cost of a silent change in what is being computed;
* shipping benchmark scripts and machine‑readable results that enable transparent performance evaluation.

# Implementation and Architecture

The package is implemented in pure Python (≥3.9) with no compiled build step; wheels are therefore platform‑independent.  Core numerical kernels are written as Numba‑typed functions [@lam2015numba] that compile to machine code at runtime, with thread‑level parallelism expressed through `numba.prange`.  When an NVIDIA GPU is detected, the distance and scoring stages are off‑loaded to `numba.cuda` kernels; the CUDA path requires only a driver and toolkit on the host, not an additional Python package.  Estimators subclass `sklearn.base.BaseEstimator` directly [@pedregosa2011sklearn], so they compose with `Pipeline`, `GridSearchCV`, and the rest of the scikit‑learn ecosystem.  Continuous integration (GitHub Actions) runs the test suite on Linux across Python 3.9–3.12, and separately enforces linting and distribution metadata checks.

# Correctness

Feature‑selection implementations are easy to get subtly wrong, and a wrong implementation is difficult to detect from downstream accuracy alone.  `Fast‑Select` therefore treats the defining paper — not a reference implementation — as the semantic authority for every algorithm.  Each method is checked against an independent oracle computed directly from the published equations, which never calls the production kernel; CPU results are compared to the oracle, and GPU results to both the oracle and the CPU path.  Tests cover threshold equality, empty neighbour groups, constant features, mixed discrete/continuous inputs, class imbalance, and non‑`0/1` binary labels.

This process surfaced and corrected several genuine deviations prior to the v1.0.0 release, including SURF's use of a per‑target rather than global mean pair‑distance radius, a missing dead band and far‑neighbour similarity term in MultiSURF\*, an unpublished relevance cutoff in CFS, and an exclusive rather than inclusive high‑risk threshold in MDR.  Where the library supports more than the source paper defines — multiclass targets for the SURF family, for example — this is documented as an extension rather than presented as the original algorithm.

# Performance

Benchmark scripts and machine‑readable results are included in the repository, covering both sample‑dominant (`n >> p`) and feature‑dominant (`p >> n`) regimes for the Relief family on CPU and GPU backends.  Performance work in the project is gated: a change is retained only when a sustained measurement — JIT warm‑up excluded, each case measured for at least five cumulative seconds in a fresh process — shows a real runtime or memory improvement, and never when it would alter a distance definition, neighbour membership, tie rule, or normalisation.

# Quality Control

* 94% line coverage via `pytest` and `coverage.py`, reported to Codecov on every push.
* Independent, equation‑derived compliance tests plus CPU/GPU parity tests for each algorithm.
* Style enforcement with `ruff` and `black`, checked in CI.
* Distributions are built and validated with `twine check --strict` in CI, and published to PyPI on tagged releases.

# Acknowledgements

We thank the *Numba* developers and the maintainers of *scikit‑rebate* [@urbanowicz2018skrebate] for foundational contributions, and the authors of the Relief‑based methods reviewed in [@urbanowicz2018review] whose descriptions made independent verification possible.

# References
