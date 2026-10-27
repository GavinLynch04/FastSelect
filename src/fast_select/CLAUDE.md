# Production Algorithm Invariants

The repository-level compliance policy is mandatory. Preserve the following
invariants in every CPU, CUDA, helper, and fallback path.

## Relief family

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

## Other selectors and statistics

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

## scikit-learn contract

- Estimators store constructor arguments unchanged; normalise and validate in
  `fit`. Mixins precede `BaseEstimator` in the base-class list.
- Validate feature count and names on every `transform`/`predict` through
  `validate_data(..., reset=False)`; never index by position without it.
- `n_jobs=-1` means the configured Numba pool (`NUMBA_NUM_THREADS`); use
  `utils.resolve_num_threads`.
- The package must import under `NUMBA_ENABLE_CUDASIM=1`. Use only public
  `numba.cuda` API for context handling; the Windows-only context workaround in
  `utils.py` must stay scoped to that platform.

## Automatic backend routing

- `backend="auto"` may route an algorithm to a GPU backend only when that path
  passed the sustained runtime gate against the CPU baseline. Chi2 and CFS Metal
  failed it on tall and feature-heavy inputs, so they pass `auto_metal=False` to
  `utils.resolve_compute_backend`; explicit `backend="metal"` still works.

Keep shared semantic constants and inequalities visibly identical across CPU,
CUDA and Metal code. If duplication is required for compilation, pair it with an
independent compliance test.

