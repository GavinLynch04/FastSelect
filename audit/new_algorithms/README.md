# External algorithm admission experiment

Five candidates were implemented independently from the papers, initially in
`prototypes/` outside the distributed package. A prototype is eligible for the
package only after independent correctness oracles and sustained comparisons
against an established external implementation pass.

## Choice of baselines

- Ross continuous/discrete MI: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html),
  installed version 1.9.1. Tune serial, 4/8 threads and 4/8 joblib processes;
  use the fastest measured setting for each benchmark shape.
- JMI, CMIM, FCBF, Laplacian Score: original
  [scikit-feature](https://github.com/jundongl/scikit-feature), pinned commit
  `48cffad4e88ff4b9d2f1c7baffb314d1b3303792`. The benchmark manifest records the
  archive SHA256. References are never runtime dependencies.

GitHub stars are a popularity proxy among reviewed projects, not proof of
global installed usage: scikit-learn ~67.9k, scikit-feature ~1.6k, mifs 286,
ITMO_FS 148 at research time. See `research.json`. The original upstream family
is the reference for the four methods that scikit-learn does not provide.

The latest PyPI fork `skfeature-chappers==1.2.1` was inspected first. Its JMI
wrapper does not activate the adaptive JMI coefficients, and its FCBF truncates
SU scores to integers. We therefore use the original algorithms, rather than
copying those behaviors. Original FCBF has a nonfunctional NumPy keyword
`dtypes='object'`; the benchmark changes only that keyword to `dtype='object'`
at import time. All formulas, loops and stopping decisions remain intact.

## Mathematical authorities and scope

- [Ross (2014)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357):
  Eq. (2), continuous/discrete nearest-neighbor MI, nat units, negative estimates
  clipped at zero. Dense input and classification targets; the public
  scikit-learn scale/jitter convention is preserved. Singleton-label samples
  are excluded. Exact sorted float64 differences replace general neighbor
  objects; k is unchanged. Discrete columns use empirical MI.
- [He, Cai & Niyogi (2005)](https://proceedings.neurips.cc/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html):
  degree-centered quadratic ratio. Use the equivalent sparse edge energy and
  per-feature affine normalization. Complete fits include Euclidean binary
  5-neighbor graph construction, self edges and symmetric maximum. Both
  implementations receive this explicit configuration. Heat-kernel and
  precomputed symmetric nonnegative graphs are supported. Undefined zero
  weighted-variance features explicitly score infinity/rank last; the reference
  uses a sentinel convention, so those cases are judged by an independent
  oracle and excluded from performance parity fixtures.
- [Yang & Moody (1999)](https://papers.nips.cc/paper_files/paper/1999/file/8c01a75941549a705cf7275e41b21f0d-Paper.pdf),
  [Brown et al. (2012)](https://www.jmlr.org/papers/v13/brown12a.html):
  JMI, using relevance minus average redundancy plus average conditional
  dependence. Scores use the normalized form; its ranking is equivalent to the
  original joint-MI sum. All features and targets are categorical, with no binning.
- [Fleuret (2004)](https://www.jmlr.org/papers/v5/fleuret04a.html), Brown et al.:
  CMIM maximizes minimum conditional MI. Arbitrary finite categorical features
  are a stated extension of Fleuret's binary formulation.
- [Yu & Liu (2004), Fig. 4](https://jmlr.org/papers/volume5/yu04a/yu04a.pdf):
  FCBF keeps SU strictly above delta, then removes a peer when pairwise SU is
  greater than or equal to its target SU. Delta equality is excluded.

Discrete entropies use bits. Greedy selection retains the lower original
feature index on exact floating-point ties. CPU and Metal prototypes use the
same float64 entropy-reduction order; Metal computes integer counts only.

## Reproduction

Use the recorded Python/NumPy/Numba/scikit-learn versions and physical machine
for comparable numbers. Install the project and optional Metal dependency,
then fetch the pinned external tree (Python 3.12+ for this helper):

```sh
.venv/bin/python benchmarking/fetch_new_algorithm_baselines.py
.venv/bin/python benchmarking/benchmark_new_algorithms.py --production --cpu-only \
  --output audit/new_algorithms/final_admission.json
# Optional archived CPU/Metal experiment, not a runtime GPU backend:
.venv/bin/python benchmarking/benchmark_new_algorithms.py \
  --output audit/new_algorithms/sustained.json
.venv/bin/python benchmarking/report_new_algorithms.py
```

`mi_baseline_tuning.json` and `lap_baseline_tuning.json` record baseline tuning.
Each benchmark worker is a fresh process. Compilation and one complete call
warm up before measurement. Five rounds alternate backend order; each worker
runs for at least 1.05 measured seconds, so every case/backend totals at least
5.25 seconds (slow reference calls run longer). Complete outputs and selected
indices are compared, and durations/iteration counts are retained.

Timing includes estimator construction, validation, category encoding or MI
preprocessing, graph construction, allocations, host/device transfers and
result materialization. It excludes imports, JIT and GPU compilation. No
concurrent benchmark/test processes are intentionally run. Memory is macOS
process peak RSS including imports/warm-up; it is not isolated GPU allocation
or total joblib pool memory and does not establish a memory speedup.

## Numerical reference exception

The tiny imbalanced Ross fixture in `tests/test_new_algorithms.py` includes a
constant column and singleton class. Scikit-learn's automatic neighbor method
can choose brute-force Euclidean distances on small classes; squared-distance
cancellation loses the tiny jitter separations at a nonzero offset. The sorted
direct-difference implementation agrees with the independent Ross oracle.
These degenerate cases are correctness tests, not speedup fixtures; ordinary
complete-call benchmarks must agree with the external scores and selections.

The final Laplacian graph path also recenters columns with large common offsets
before the public Euclidean-distance routine to avoid squared-distance
cancellation. The independent graph oracle checks translation by 1e8 and
boundary ties; this correction does not affect the ordinary benchmark graphs.

## Package admission

All five algorithms passed the external-baseline gate and have parallel Numba
CPU implementations in `src/fast_select`. The archived Metal variants failed
the additional sustained gain gate against those new CPU paths, and are not
shipped or selected automatically. The `prototypes` directory preserves the
exact source hash measured by `sustained.json`; it is research provenance.
`final_admission.json` measures the final distributed CPU modules and records
their exact individual source hashes. No code from scikit-feature was copied
into the production implementation.
