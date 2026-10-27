# fast-select algorithm policy

These are the rules every change to fast-select follows: which sources are
authoritative, what performance work may and may not change, the invariants
each algorithm must preserve, and how algorithms are verified. The contributor
guide ([`CONTRIBUTING.md`](../../CONTRIBUTING.md)) summarizes them.

## Algorithm Compliance and Performance Policy

This policy applies to the entire repository. The implementation invariants
and verification standards below add rules for the package code and the tests.

### Authority order

1. The defining paper and its equations/pseudocode are the semantic authority.
2. Author-maintained errata or a later paper that explicitly corrects the
   defining paper may override it; record the reason and citation.
3. Reference libraries such as scikit-rebate and scikit-learn are corroborating
   evidence only. Never copy their behavior when it conflicts with a paper.
4. Existing FastSelect behavior is not an authority. Backward compatibility
   must not preserve a confirmed algorithm error.

Every algorithm change must name the paper, equation/pseudocode rule, supported
problem type, and any intentional limitation. Do not describe an extension as
the original algorithm.

### No-drift rule

Performance work may change data layout, traversal order, parallel reduction,
allocation strategy, precision where tolerances permit, or host/device
placement. It must not change:

- distance or similarity definitions;
- neighbor membership, boundary inequalities, or tie rules;
- hit/miss, class, fold, or subset normalization;
- update signs or the quantity being scored (`diff` versus `1 - diff`);
- class priors, risk thresholds, entropy/logarithm units, or degrees of freedom;
- subset-search state, stopping criteria, or tie-breaking;
- public interpretation of fitted scores and selected features.

When floating-point reordering is the only difference, demonstrate bounded
error on adversarial and ordinary inputs. CPU/GPU agreement alone is
insufficient because both paths can share the same error.

### Required change gate

Before merging an algorithm or kernel change:

1. Add or retain an independent, equation-driven oracle test. The oracle must
   not call the production kernel or a helper that implements the same logic.
2. Cover relevant boundaries: exact threshold equality, empty neighbor groups,
   constant features, mixed discrete/continuous inputs, imbalanced labels,
   multiclass behavior where supported, and non-`0/1` binary labels.
3. Test CPU against the oracle and GPU against both the oracle and CPU.
4. Run the complete test suite.
5. For a performance change, use the current fastest correct implementation
   available in this repository as the baseline, including retained local
   optimizations. Freeze that baseline before changing it, using an exact
   revision or a preserved source snapshot and source hash. CPU changes must
   beat that CPU baseline; new GPU backends must be compared against that CPU
   baseline on the same machine, with existing GPU comparisons reported
   separately when available. Run in fresh processes, warm JIT compilation
   first, alternate execution order when multiple rounds are used, and measure
   each version/backend/case for at least five cumulative seconds. Comparisons
   against older releases, including v0.2.1, are optional historical references
   and must not replace the current baseline or establish a new speedup claim.
6. Retain a performance change only when the sustained result demonstrates a
   real runtime or memory improvement. Keep a required correctness fix even
   when it is slower, and report that regression without calling it a speedup.
7. Store machine-readable benchmark results and report baseline identity/source
   hash, dataset shape, backend, thread count, iterations, measured duration,
   central runtime statistic, and memory method.

### Primary sources

- Relief family: Kira and Rendell (1992); Kononenko (1994); Moore et al. for
  SURF/SURF*; Urbanowicz et al., *Benchmarking Relief-Based Feature Selection
  Methods for Bioinformatics Data Mining* (2018), PMCID: PMC6299838; and
  Urbanowicz et al., *Relief-Based Feature Selection: Introduction and Review*
  (2018).
- CFS: Mark A. Hall, *Correlation-based Feature Selection for Machine
  Learning* (PhD thesis, 1999).
- mRMR: Peng, Long, and Ding, *Feature Selection Based on Mutual Information:
  Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy* (TPAMI 2005),
  DOI: 10.1109/TPAMI.2005.159.
- MDR: Ritchie et al., *Multifactor-Dimensionality Reduction Reveals
  High-Order Interactions among Estrogen-Metabolism Genes in Sporadic Breast
  Cancer* (2001), DOI: 10.1086/321276.
- Mutual information: Shannon, *A Mathematical Theory of Communication*
  (1948).
- Chi-squared independence scoring: Pearson's chi-squared statistic; the
  package's count-feature API intentionally follows scikit-learn's documented
  `feature_selection.chi2` formulation.
- Continuous/discrete MI: Ross, *Mutual Information between Discrete and
  Continuous Data Sets* (2014), DOI: 10.1371/journal.pone.0087357, Eq. (2).
- Laplacian Score: He, Cai and Niyogi, *Laplacian Score for Feature Selection*
  (NIPS 2005), degree-centered graph-Laplacian quadratic ratio.
- JMI: Yang and Moody (1999), normalized entropy identity in Brown et al.,
  *Conditional Likelihood Maximisation* (JMLR 2012).
- CMIM: Fleuret, *Fast Binary Feature Selection with Conditional Mutual
  Information* (JMLR 2004); categorical extension using Brown et al. (2012).
- FCBF: Yu and Liu (ICML 2003; JMLR 2004 extension, Figure 4).
- Regression MI: Kraskov, Stögbauer and Grassberger, *Estimating Mutual
  Information* (2004), estimator 1; Ross (2014) for explicit discrete columns.
- GiniIndex: Breiman et al., *Classification and Regression Trees* (1984),
  minimum weighted class impurity over distinct numeric splits.
- MIFS: Battiti, *Using Mutual Information for Selecting Features in
  Supervised Neural Net Learning* (1994), DOI: 10.1109/72.298224.
- CIFE: Lin and Tang, *Conditional Infomax Learning* (ECCV 2006), with the
  feature-selection entropy identity in Brown et al. (JMLR 2012).
- DISR: Meyer and Bontempi (2006); Meyer, Schretter and Bontempi (2008,
  corrected 2009), normalized joint-information criterion.

## Production Algorithm Invariants

The repository-level compliance policy is mandatory. Preserve the following
invariants in every CPU, CUDA, helper, and fallback path.

### Relief family

- Continuous feature difference is `abs(a-b)/(max-min)`; zero-range features
  contribute zero. Discrete difference is `0` for equality and `1` otherwise.
  Instance distance is the sum of feature differences.
- ReliefF uses nearest hits and per-class nearest misses. Multiclass miss
  contributions use the defining class-prior normalization. A reduced
  neighbor count must be explicit and consistently normalized.
- SURF uses one global mean over unique off-diagonal instance-pair distances.
  It does not use a separate radius per target.
- SURF near hits subtract differences and near misses add differences. SURF*
  additionally reverses those difference updates for distances strictly above
  the global radius. Normalize hit and miss groups separately and divide each
  target contribution by the number of training instances.
- MultiSURF uses a target-specific near boundary
  `mean(distance_i) - std(distance_i)/2`.
- MultiSURF* also uses the far boundary
  `mean(distance_i) + std(distance_i)/2`; distances inside the two boundaries
  are a dead band. Far updates score feature similarity: continuous
  `1 - normalized_diff`, discrete equality. Far hits subtract similarity and
  far misses add it, with separate hit/miss count normalization.
- Threshold comparisons are strict (`< near`, `> far`). Equality is excluded.
- Working representation (all Relief-family estimators, CPU, CUDA and Metal): each
  continuous column is normalised as `(x - min) / (max - min)` in float64 and
  only then narrowed; each discrete column is replaced by exact dense category
  codes. Never cast raw feature values to float32 before normalising or coding
  (large offsets or large category codes silently collapse otherwise). Both
  backends must score the same prepared matrix (`utils.prepare_relief_matrix`);
  Metal may represent float64 coordinates by high/low float32 parts.
- SURF/SURF* retain float64 global-radius and membership semantics. CPU/CUDA
  compute pair distances in float64. Metal may certify an unambiguous predicate
  with a conservative distance-error interval; ambiguous pairs must be
  recomputed on the CPU in float64 before classifying them. This must produce
  the same strict predicate; it must not widen the algorithm's radius or treat
  uncertain pairs as ties without recomputation. A
  distance within `utils.BOUNDARY_RTOL` (relative) of the radius is a tie and is
  neither near nor far; the kernels keep strict `<` / `>` against the widened
  `near_limit` / `far_limit`. Keep exact-boundary regression cases (the
  rational-arithmetic oracle in `tests/test_exact_boundaries.py`).
- `MultiSURF.fit(feat_idx=...)` restricts distances, scoring *and* ranking to
  the listed features; unevaluated features must never be ranked.
- TuRF repeatedly re-fits its Relief-family base estimator after removing the
  requested fraction of the currently lowest-scoring features. Never reuse
  scores computed for a superseded feature space.

### Other selectors and statistics

- CFS merit is
  `(k * mean(r_cf)) / sqrt(k + k*(k-1)*mean(r_ff))`.
  Use forward best-first subset search and its configured consecutive
  non-improvement stopping count. Do not add an unpublished minimum relevance
  cutoff or post-search redundancy pruning.
- CFS symmetrical uncertainty is `2*I(X;Y)/(H(X)+H(Y))`. CUDA entropy must read
  only initialized states.
- mRMR relevance is `I(feature; target)` and redundancy is the mean mutual
  information with already selected features. MID subtracts redundancy; MIQ
  divides by it. Encoding may be optimized but must preserve each variable's
  empirical joint distribution: encode every feature and the target
  independently, by symbol identity, into an explicit integer dtype (never mix
  vocabularies of different dtypes, which promotes to float64).
- Discrete codes passed to the mutual-information functions are remapped to
  dense per-column codes when large; distinct symbols must never alias through
  a narrowing cast.
- Mutual information is
  `sum p(x,y) * log(p(x,y)/(p(x)*p(y)))`. Count every sample exactly once.
  `unit="bit"` uses log base 2 and `unit="nat"` uses the natural logarithm on
  both CPU and GPU.
- Chi-squared count-feature scoring uses class-wise feature sums as observed
  counts, class frequency times the feature total divided by sample count as
  expected counts, and `n_classes - 1` degrees of freedom.
- MDR marks a nonempty cell high risk when its case/control ratio is greater
  than or equal to the dataset case/control ratio. Empty cells are low risk.
  Encode arbitrary binary labels internally and map predictions back to the
  original labels. Model selection uses cross-validation consistency, then
  testing balanced accuracy to break consistency ties. Genotypes must be
  exactly 0, 1 or 2 and are validated (in `fit` and `predict`) *before* any
  narrowing cast; cell indices are computed in int64.
- JMI maximizes relevance minus mean redundancy plus mean conditional
  dependence. CMIM maximizes the minimum conditional MI over selected features.
  Their inputs are categorical symbols, their entropy unit is bits, and exact
  floating-point ties retain the lower original feature index. Do not add
  automatic binning or approximate candidate pruning.
- FCBF relevance is SU strictly greater than delta; remove a peer when its SU
  with the selected feature is >= its target relevance. Its selected count may
  be zero. Do not impose a fixed top-k or change equality behavior.
- Laplacian Score is f_centered.T @ (D-W) @ f_centered divided by
  f_centered.T @ D @ f_centered, with degree-weighted centering. Sparse edge
  energy and affine column normalization preserve this ratio. Zero weighted
  variance is explicitly undefined (+inf, ranked last). Partial graph-neighbor
  selection must repair boundary ties with the full argsort membership.
- Dense continuous MI uses Ross's estimator with exact float64 class-neighbor
  distances and strict radius via nextafter(kth_distance, 0), singleton-label
  removal, unchanged k, nat units and clipping at zero. Preserve the public
  scale/jitter convention and random-state behavior. Do not copy cancellation
  errors from external neighbor implementations.
- The first admission batch retains CPU only. Its GPU prototypes failed the
  sustained gain gate against their own Numba CPU paths. Keep experimental
  GPU code under audit/, rather than enabling it through availability alone.
- The second batch (regression MI, GiniIndex, DISR, CIFE, MIFS) also retains
  CPU only; no GPU implementation was attempted or measured in that batch.
- Regression MI uses KSG estimator 1: exact Chebyshev kth joint distance,
  strict marginal counts via nextafter(radius, 0), digamma formula in nats,
  and negative-estimate clipping. The sorted-x lower bound may prune only
  points whose x distance exceeds the current kth joint distance. Preserve
  preprocessing/jitter order, k and full float64 distance predicates. Explicit
  discrete columns use the existing Ross kernel after regression preprocessing.
- GiniIndex minimizes weighted impurity 1-sum(class_probability**2) over all
  distinct-value splits plus the no-split parent. Never bin thresholds, narrow
  numeric values before sorting, or copy scikit-feature's 0.5 multiclass cap.
- MIFS is marginal MI minus beta times summed redundancy. CIFE subtracts
  summed redundancy and adds summed conditional dependence. Do not divide
  either sum by subset size or clip valid negative selection objectives.
- DISR sums I((candidate,selected);target)/H(candidate,selected,target),
  initializing with marginal MI. Zero joint entropy contributes zero as an
  explicit constant-variable extension. These three criteria use categorical
  inputs, bits, exact floating-point ties by original index, and fixed counts.

### scikit-learn contract

- Estimators store constructor arguments unchanged; normalise and validate in
  `fit`. Mixins precede `BaseEstimator` in the base-class list.
- Validate feature count and names on every `transform`/`predict` through
  `validate_data(..., reset=False)`; never index by position without it.
- `n_jobs=-1` means the configured Numba pool (`NUMBA_NUM_THREADS`); use
  `utils.resolve_num_threads`.
- The package must import under `NUMBA_ENABLE_CUDASIM=1`. Use only public
  `numba.cuda` API for context handling; the Windows-only context workaround in
  `utils.py` must stay scoped to that platform.

### Automatic backend routing

- `backend="auto"` may route an algorithm to a GPU backend only when that path
  passed the sustained runtime gate against the CPU baseline. Chi2 and CFS Metal
  failed it on tall and feature-heavy inputs, so they pass `auto_metal=False` to
  `utils.resolve_compute_backend`; explicit `backend="metal"` still works.

Keep shared semantic constants and inequalities visibly identical across CPU,
CUDA and Metal code. If duplication is required for compilation, pair it with an
independent compliance test.

## Algorithm Verification Standards

Follow the repository policy and keep three distinct test layers:

1. **Equation oracle:** a tiny NumPy/Python calculation derived directly from
   the paper. It must not import private production helpers.
2. **Backend conformance:** compare CPU, CUDA and Metal results to the oracle
   on available hardware, then to each other with a tolerance justified by
   reduction order and precision. Metal's certified boundary decisions need
   exact-boundary tests that exercise float64 repair, not just random parity.
3. **Behavior/API:** validate fitted attributes, transformations, errors, and
   scikit-learn compatibility separately from mathematical correctness.

Every neighbor algorithm needs a dataset for which global and per-target
thresholds differ, plus threshold-equality and empty-group cases. Star variants
must include examples that distinguish `diff` from `1 - diff` and exercise
near, dead-band, and far observations.

Boundary tests must not depend on floating-point luck: use integer-valued data
whose radius/threshold ties are exact in rational arithmetic
(`fractions.Fraction`) and whose column ranges are non-dyadic, so that
`|a - b| / range` is not exactly representable. Also keep translation and
category-relabelling invariance cases: the score must not change when a column
is shifted by `1e8` or its category codes by `2**25`.

CUDA kernels are exercised without hardware by `NUMBA_ENABLE_CUDASIM=1` (see
`test_cuda_simulator.py`). That validates kernel logic only; it says nothing
about drivers, races, or device rounding, and must not be reported as GPU
verification.

Every exported estimator needs a scikit-learn contract check
(`test_estimator_contracts.py`) and a real `Pipeline`/`GridSearchCV` smoke test.

Statistical algorithms need hand-calculated distributions with known results:
perfect dependence, independence, constant variables, class imbalance, and
label remapping. Search algorithms need a case where greedy hill climbing and
best-first traversal choose different exploration paths.

Do not approve a semantic change solely because it matches scikit-rebate,
scikit-learn, CPU/GPU parity, or a previous FastSelect release. When using an
external implementation as a cross-check, pin its version and retain the
paper-derived oracle.

Performance tests are not correctness tests. Sustained benchmarks must exclude
JIT warm-up, measure each compared case for at least five cumulative seconds,
run versions in isolated processes, and retain raw JSON output.
