---
title: "Fast‑Select: Accelerated Feature Selection for Modern Datasets"

tags:
- python
- feature selection
- bioinformatics
- GPU
- machine learning
- mutual information

authors:
- name: Gavin Lynch
  orcid: "0009-0006-0097-4157"
  affiliation: 1

affiliations:
- name: Department of Computer Science, California Polytechnic State University San Luis Obispo, San Luis Obispo, USA
  index: 1
  ror: 001gpfp45
date: 27 October 2026

bibliography: paper.bib

---

# Summary

`Fast‑Select` is an open‑source Python package that accelerates classical feature‑selection algorithms by combining Just‑In‑Time (JIT) compilation via **Numba**, thread‑parallel CPU kernels, optional **CUDA** kernels for NVIDIA GPUs, and an optional **Metal** backend for Apple Silicon GPUs.  The library ships implementations of the Relief family — ReliefF [@kononenko1994relieff], SURF and SURF\* [@greene2009surf], MultiSURF and MultiSURF\* [@urbanowicz2018benchmarking], and TuRF — alongside CFS [@hall1999cfs], mRMR [@peng2005mrmr], Chi‑squared, and MDR [@ritchie2001mdr].  Version 0.4.0 adds the greedy information‑theoretic selectors JMI [@yang1999jmi], CMIM [@fleuret2004cmim], FCBF [@yu2004fcbf], MIFS [@battiti1994mifs], CIFE [@lin2006cife], and DISR [@meyer2006disr; @meyer2008disr]; the Gini index [@breiman1984cart]; the unsupervised Laplacian Score [@he2005laplacian]; and nearest‑neighbour mutual‑information estimators for continuous classification [@ross2014mi] and regression [@kraskov2004mi] targets.  Every method is wrapped in a **scikit‑learn‑compatible API**.  It targets modern, high‑dimensional biological datasets—such as whole‑genome variant matrices or single‑cell expression counts—where traditional CPU‑bound methods become a bottleneck.

# Statement of Need

Typical omics studies now profile **10⁴–10⁶ features** across thousands of samples.  Relief‑based methods are attractive here because they detect feature interactions without exhaustively enumerating them [@kira1992relief; @urbanowicz2018review], but their instance‑pair distance computation is quadratic in the sample count.  Existing Python toolkits, notably *scikit‑rebate* [@urbanowicz2018skrebate], are CPU‑bound and offer no GPU path, which limits their utility as dataset sizes grow.  Likewise, *scikit‑learn* [@pedregosa2011sklearn] provides mutual‑information estimators but none of the greedy relevance–redundancy selectors unified by @brown2012clm.  `Fast‑Select` fills these gaps by:

* providing scikit‑learn‑compatible filter methods with JIT‑compiled CPU, CUDA, and Apple Silicon Metal back‑ends;
* exposing an identical API for CPU and GPU back‑ends, easing adoption in reproducible pipelines;
* verifying every algorithm against its defining paper rather than against another implementation, so that a speed‑up never comes at the cost of a silent change in what is being computed;
* shipping benchmark scripts and machine‑readable results that enable transparent performance evaluation.

# Implementation and Architecture

The package is implemented in pure Python (≥3.9) with no compiled build step; wheels are therefore platform‑independent.  Core numerical kernels are written as Numba‑typed functions [@lam2015numba] that compile to machine code at runtime, with thread‑level parallelism expressed through `numba.prange`.  When an NVIDIA GPU is detected, the distance and scoring stages are off‑loaded to `numba.cuda` kernels; the CUDA path requires only a driver and toolkit on the host, not an additional Python package.  On Apple Silicon, an optional extra (`fast-select[metal]`, Python 3.10+) installs MLX [@hannun2023mlx] and enables Metal kernels for the Relief family, discrete mutual information, mRMR, CFS, MDR, and Chi‑squared.  Metal arithmetic is float32, so neighbour and threshold decisions are certified against a conservative error bound and any decision inside that bound is recomputed on the CPU in float64; the Metal path therefore makes the same decisions as the CPU path.  With `backend="auto"`, the package prefers CUDA, then Metal, then CPU, except that Chi‑squared and CFS stay on the CPU because their Metal paths did not consistently outperform it.  The selectors added in 0.4.0 run on parallel CPU kernels only, because their GPU prototypes did not consistently beat those kernels.  Estimators subclass `sklearn.base.BaseEstimator` directly [@pedregosa2011sklearn], so they compose with `Pipeline`, `GridSearchCV`, and the rest of the scikit‑learn ecosystem.  Continuous integration (GitHub Actions) runs the test suite on Linux across Python 3.9–3.14, runs the CUDA kernels under the Numba CUDA simulator, builds the documentation with warnings treated as errors, and separately enforces linting and distribution metadata checks.  Publication to PyPI is gated on the same checks passing for the tagged commit.  Physical-GPU behaviour (driver and context handling, races, device-level accuracy) is not covered by hosted CI.

# Correctness

Feature‑selection implementations are easy to get subtly wrong, and a wrong implementation is difficult to detect from downstream accuracy alone.  `Fast‑Select` therefore treats the defining paper — not a reference implementation — as the semantic authority for every algorithm.  Each method is checked against an independent oracle computed directly from the published equations, which never calls the production kernel; CPU results are compared to the oracle, and GPU results (CUDA and Metal) to both the oracle and the CPU path.  Tests cover threshold equality, empty neighbour groups, constant features, mixed discrete/continuous inputs, class imbalance, and non‑`0/1` binary labels.

This process surfaced and corrected several genuine deviations prior to the v0.3.0 release, including SURF's use of a per‑target rather than global mean pair‑distance radius, a missing dead band and far‑neighbour similarity term in MultiSURF\*, an unpublished relevance cutoff in CFS, and an exclusive rather than inclusive high‑risk threshold in MDR.  Where the library supports more than the source paper defines — multiclass targets for the SURF family, or arbitrary categorical features for CMIM, which @fleuret2004cmim formulates for binary features — this is documented as an extension rather than presented as the original algorithm.  Where a paper is internally ambiguous, the choice is stated: CMIM follows Fleuret's formal criterion, as @brown2012clm do, rather than the variant implied by his pseudocode.

# Performance

Benchmark scripts and machine‑readable results are included in the repository, covering both sample‑dominant (`n >> p`) and feature‑dominant (`p >> n`) regimes.  Performance work in the project is gated: a change is retained only when a sustained measurement — JIT warm‑up excluded, each case measured for at least five cumulative seconds in a fresh process — shows a real runtime or memory improvement, and never when it would alter a distance definition, neighbour membership, tie rule, or normalisation.  Regressions are reported rather than hidden: on an Apple M1 Pro, Metal ran discrete SURF‑family, mutual‑information, and mRMR workloads 3.5–13.5× faster than the optimised CPU path, but ran Chi‑squared and continuous‑feature MultiSURF slower on both tall and wide inputs.  Each algorithm added in 0.4.0 was admitted only after complete‑call comparisons against a tuned external baseline (scikit‑learn or the original *scikit‑feature* code); an ANOVA F‑test prototype that lost this comparison was left out.

# Quality Control

* Line coverage via `pytest` and `coverage.py`, reported to Codecov on every push from the Linux CI runners.  This figure covers host-side Python only: Numba-compiled numerical kernels are excluded from line coverage and are verified by the independent oracle tests instead, so coverage is not evidence of kernel correctness.  The Metal modules are omitted from the report because the CI runners have no Apple GPU; they are tested locally on Apple Silicon against the CPU backend and the oracles.
* Independent, equation‑derived compliance tests plus CPU/CUDA/Metal parity tests for each algorithm with a GPU path.
* Style enforcement with `ruff` and `black`, checked in CI.
* Distributions are built and validated with `twine check --strict` in CI, and published to PyPI on tagged releases.

# Acknowledgements

We thank the *Numba* developers and the maintainers of *scikit‑rebate* [@urbanowicz2018skrebate] for foundational contributions, and the authors of the Relief‑based methods reviewed in [@urbanowicz2018review] whose descriptions made independent verification possible.

# Funding

This work received no specific grant from any funding agency in the public, commercial, or not‑for‑profit sectors.

# Conflict of Interest

The author declares no competing financial or non‑financial interests.

# AI Usage Disclosure

Generative AI assistants were used in developing `Fast‑Select`, in preparing its documentation, and in revising parts of this manuscript.  The tools, versions, and scope of use were:

* **Before v0.3.0**, AI assistance was limited to performance optimisation, using OpenAI GPT 6.1 sol; the algorithm implementations were written by the author.
* **OpenAI GPT 6.1 sol**, including through OpenAI Codex (2025–2026): performance optimisation alongside the author; Python 3.13/3.14 support and fixes for the v0.3.0 release; and first implementations of the algorithms added in v0.4.0 (JMI, CMIM, FCBF, MIFS, CIFE, DISR, Gini index, Laplacian Score, and the continuous and regression mutual‑information estimators), the optional Metal backend, their oracle tests, and their benchmark harnesses.
* **Anthropic Claude Code** with Claude Sonnet 5.5 and Claude Opus 5.5 (2026): code generation and refactoring for the v0.3.0 release review, the v0.4.0 release, and the v0.4.1 CPU kernel optimisations and benchmark charts, writing tests, running and reporting benchmarks, reviewing code against the source papers, and drafting documentation, changelog entries, and release notes.
* **Manuscript:** the original text of this paper was written by the author, with Anthropic Claude Opus used only for grammar and rewording.  For the v0.4.0 revision, Claude Opus 5.5 drafted the Funding, Conflict of Interest, and AI Usage Disclosure sections and the passages describing the v0.4.0 algorithms, the Metal backend, and the coverage policy; the author reviewed them.

All AI‑generated code was constrained by a verification process designed by the author and written down, with AI assistance, in the repository's algorithm policy (`docs/development/algorithm-policy.md`).  That process treats each algorithm's defining paper as the semantic authority; requires every algorithm change to cite the equation or pseudocode it implements; requires an independent oracle test, computed directly from the published equations, that never calls the production kernel; and admits a performance change only after sustained, fresh‑process benchmarks against a frozen baseline show a real improvement.  Reference libraries are used only as corroborating evidence.

The author made the primary design and architectural decisions, including the choice of algorithms, the paper‑as‑authority verification policy, the scikit‑learn API design, and the CPU/CUDA/Metal backend structure.  The author reviewed, modified where necessary, and validated all AI‑generated code, tests, documentation, and text before it was merged, and takes full responsibility for the accuracy, originality, and licensing of the submission.

# References
