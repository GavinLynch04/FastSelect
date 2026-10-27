# Second algorithm admission batch

This batch looks for **five additional algorithms**, excluding the five already
admitted in `../new_algorithms`. New implementations begin under `prototypes/`,
outside the installable `src` package. Only candidates with independent equation
oracles and sustained wins against the selected external reference can enter
the library. No reference-library source is copied into runtime modules.

## Candidates and primary definitions

- **Regression mutual information:** [Kraskov, Stögbauer & Grassberger
  (2004)](https://arxiv.org/abs/cond-mat/0305641), estimator 1, Chebyshev joint
  k-neighbor distance, strict marginal radius, digamma counts and nat units.
  Dense numeric regression data; scikit-learn's public scaling/jitter and
  negative-estimate clipping conventions are retained. Explicit discrete
  feature columns use Ross (2014), reusing the previously tested exact CPU
  kernel after the same regression-target preprocessing. `n_neighbors` must
  be smaller than sample count. Singleton categories are excluded for Ross;
  float32 inputs use float64 working values, including target jitter.
  all-singleton discrete columns raise. No approximate neighbors or binning.
  Sorted-x expansion stops only when the next x distance exceeds the current
  kth joint distance, an exact lower bound. Worst-case work can approach
  quadratic in sample count; this is not a universal KDTree replacement.
- **GiniIndex:** Breiman et al., *Classification and Regression Trees* (1984).
  Minimize weighted child impurity, `1 - sum(p_class**2)`, over every distinct
  numeric threshold and the no-split parent. Sort once per feature, sweep exact
  class counts and update their squared sums. Supports binary/multiclass
  classification and arbitrary target labels. Raw scores are lower-is-better;
  importances negate them. Do not replace this with whole-tree feature
  importance or bin thresholds.
- **DISR:** [Meyer & Bontempi (2006); Meyer et al. corrected extension
  (2008/2009)](https://orbi.uliege.be/handle/2268/163912), sum of
  `I((candidate,selected);target)/H(candidate,selected,target)`.
- **CIFE:** [Lin & Tang (2006)](https://mmlab.ie.cuhk.edu.hk/2006/39510068.pdf),
  marginal relevance minus summed redundancy plus summed within-class
  dependence; the feature-selection entropy identity is also given in
  [Brown et al. (2012)](https://www.jmlr.org/papers/v13/brown12a.html).
- **MIFS:** [Battiti (1994)](https://iris.unitn.it/handle/11572/53878), relevance
  minus beta times summed redundancy, default beta 0.5. It does not average by
  selected subset size, and has no conditional-dependence term.

The three discrete selectors use empirical categorical entropies in bits,
initialize with maximum marginal target MI, select the requested fixed count,
and break exact floating-point ties by original feature index. Their
`feature_importances_` stores marginal MI; `selection_scores_` stores the
criterion at each selection. Categorical encoding preserves symbol identity,
including large uint64 codes; it performs no implicit discretization. DISR
zero joint entropy explicitly contributes zero, a constant-variable extension.
Greedy selection is sequential; candidate calculations are parallel.

## Popular references, and stronger alternatives

Popularity uses GitHub stars among reviewed projects, not a census of Python
installations. [scikit-learn](https://github.com/scikit-learn/scikit-learn)
(~67.9k) is the regression-MI and compiled Gini baseline. Original
[scikit-feature](https://github.com/jundongl/scikit-feature) (~1.6k) provides
DISR/CIFE/MIFS; alternatives reviewed include mifs (~286), ITMO_FS (~148), and
pyHSICLasso (~197). The `mifs` package implements other continuous MI selector
criteria, not Battiti's discrete MIFS criterion.

Scikit-learn version is 1.9.1. Scikit-feature is pinned to commit
`48cffad4e88ff4b9d2f1c7baffb314d1b3303792`, reusing the unchanged archive and
source path from the first batch. This batch needs no reference source shim.
`baseline_manifest.json` records identity and hashes of all previously retained
runtime sources; `environment.json` records versions and relevant reference
Python/compiled-module hashes.

For Gini, measure both scikit-feature and a stronger, more popular alternative:
one scikit-learn `DecisionTreeClassifier(criterion='gini', max_depth=1)` per
feature. Read the weighted leaf impurity, not normalized feature importance.
Tune serial, 4/8 joblib threads and 4/8 processes. Parallel compiled CART is
much faster than scikit-feature and is the actual admission baseline. Fixtures
have numeric values rounded to 1e-4, ensuring CART's float32 conversion and
1e-7 near-equality rule preserve every distinct-value split. Full score arrays
agree across both references. Float64 near-ties/large integer identities are
tested against an independent exact split oracle, not CART's narrowed input.

Scikit-feature initializes Gini at 0.5, incorrectly capping multiclass values.
The binary benchmark fixture avoids that defect; our multiclass oracle checks
the uncapped definition. Its ICAP implementation uses a maximum where the
paper requires a sum, and TraceRatio prunes its search space before iterating;
neither is used as a candidate or a same-algorithm speed baseline.

The latest `skfeature-chappers==1.2.1` fork was inspected: its DISR changes to
another formula after the first step; CIFE/MIFS wrappers consume LCSI ranks as
though they were selected indices. The pinned original is the correct
reference. Research sources and decisions are recorded in `research.json`.

## Rejected ANOVA prototype

ANOVA F (`f_classif`) was the initial fifth candidate. Its centered float64
Numba implementation passed rational-equation, affine-invariance and SciPy
probability checks, but lost to the official scikit-learn function at all four
screened sizes (900×128 through 8000×2048), including the best of 1/4/8 Numba
threads. See `anova_screen.json` and `test_anova_research.py`. It is not
exported or copied into the installable package. MIFS replaces it.

## Benchmark and reproduction

```sh
# Fetches the same pinned reference tree if needed:
.venv/bin/python benchmarking/fetch_new_algorithm_baselines.py
.venv/bin/python benchmarking/benchmark_more_algorithms.py --tune
.venv/bin/python benchmarking/benchmark_more_algorithms.py
```

The tuning phase compares serial/thread/process alternatives on exactly the
benchmark fixtures. The sustained phase uses the fastest recorded configuration
per shape. Each case/version/round gets a fresh process; compilation, imports
and one complete warm-up call precede timing. Five rounds alternate order;
each worker measures at least 1.05 seconds, totaling at least 5.25 seconds per
case/backend. Complete score arrays, selected indices and discrete-selector
marginal relevance are checked. The reference's full relevance array is
recomputed outside timing solely for comparison; its native returned scores,
selected indices and selected relevance are compared as well.

No concurrent benchmark/test workload is intentionally run. Timings include
construction, validation, preprocessing, encoding, allocations and complete
result materialization. They exclude JIT/imports and joblib pool startup.
Memory is peak macOS process RSS including warm-up/imports, excluding separate
live joblib workers; it does not establish a total-memory improvement. Numba
uses eight threads and BLAS is limited to one. This batch has no new CUDA or
Metal implementation and makes no GPU-speed claim.

Only exact, measured winner module bytes may be promoted to `src/fast_select`.
The final report verifies their source hashes; `--production` can rerun the
same harness against the public modules. See `benchmarking/results/NEW_ALGORITHMS_2.md`
for the admission decisions and full timings once measurements finish.
