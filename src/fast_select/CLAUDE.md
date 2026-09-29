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
- Working representation (all Relief-family estimators, CPU and CUDA): each
  continuous column is normalised as `(x - min) / (max - min)` in float64 and
  only then narrowed; each discrete column is replaced by exact dense category
  codes. Never cast raw feature values to float32 before normalising or coding
  (large offsets or large category codes silently collapse otherwise). Both
  backends must score the same prepared matrix (`utils.prepare_relief_matrix`).
- SURF/SURF* compute the global radius and every pair distance in float64. A
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

Keep shared semantic constants and inequalities visibly identical across CPU
and CUDA code. If duplication is required for compilation, pair it with an
independent compliance test.

