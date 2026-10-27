# Five more algorithms: second admission batch

**Regression MI, GiniIndex, DISR, CIFE and MIFS passed admission and were added
as parallel Numba CPU implementations.** Each preserved scores and selected
indices and beat its tuned external baseline by at least 3% in every round on
both measured shapes. The initial ANOVA F prototype lost to scikit-learn and
was excluded; MIFS replaced it. No new Metal/CUDA backend is implemented or
claimed by this batch.

## Measured complete-call performance

Apple M1 Pro, 8 CPU cores, 16 GiB RAM, macOS 27.0.1 arm64; Python 3.13.17,
NumPy 2.5.4, Numba 0.68.0, SciPy 1.18.1 and scikit-learn 1.9.1.
Run started 2026-10-27T16:04:22Z. Times are milliseconds, using the median
of five per-round medians. Central speedup divides these central times;
the paired range shows the minimum and maximum same-round speedup.

| Algorithm | Samples × features | Baseline | External ms | Numba ms | Central gain | Paired round range |
|---|---:|---|---:|---:|---:|---:|
| mutual_info_regression | 2,000 × 128 | sklearn MI, 8 processes | 131.06 | 48.35 | 2.71× | 2.46–3.10× |
| mutual_info_regression | 4,000 × 256 | sklearn MI, 8 processes | 555.15 | 224.64 | 2.47× | 2.15–3.48× |
| GiniIndex | 900 × 128 | sklearn CART, 4 processes | 24.93 | 1.97 | 12.63× | 12.28–13.07× |
| GiniIndex | 2,000 × 256 | sklearn CART, 4 processes | 49.29 | 8.35 | 5.90× | 5.82–6.97× |
| DISR | 900 × 128 | original scikit-feature | 2365.44 | 3.47 | 680.82× | 633.04–724.29× |
| DISR | 2,000 × 256 | original scikit-feature | 11191.43 | 7.94 | 1410.17× | 1345.47–1438.01× |
| CIFE | 900 × 128 | original scikit-feature | 943.20 | 3.52 | 267.90× | 260.87–285.62× |
| CIFE | 2,000 × 256 | original scikit-feature | 4309.22 | 7.95 | 542.17× | 520.54–649.92× |
| MIFS | 900 × 128 | original scikit-feature | 947.66 | 3.05 | 310.94× | 306.07–318.22× |
| MIFS | 2,000 × 256 | original scikit-feature | 4419.38 | 6.34 | 696.57× | 655.29–749.08× |

These are warmed throughput measurements for the stated workloads, not a
universal speed guarantee. All discrete selectors request ten features; Gini
also ranks all features and selects ten. Regression MI scores every continuous
column with k=3. The optional discrete-column regression-MI path is tested for
correctness but is not part of these speedup claims. The exact fixtures and
seed 7931 are in the harness.

## References and fairness

[scikit-learn](https://github.com/scikit-learn/scikit-learn) is the most popular
reviewed Python package for the regression-MI and Gini primitives (~67.9k
GitHub stars). Its public
[mutual_info_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)
was tuned over serial, four/eight threads and four/eight processes. Eight
processes won both shapes; pools are warmed before measurement.

Scikit-learn has no dedicated univariate Gini ranking function. The stronger
Gini baseline uses one compiled
[CART stump](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
per feature and reads weighted child impurity. The full function is tuned over
the same parallel options; four processes won both shapes. Original
scikit-feature's dedicated Gini scorer was also timed and cross-checked, but
was much slower, so **its larger apparent speedup is not the headline result**.
This compares the same minimum weighted-impurity statistic, not whole-tree
normalized feature importances.

[Original scikit-feature](https://github.com/jundongl/scikit-feature), about
1.6k stars, is the largest reviewed Python implementation family providing
DISR/CIFE/MIFS. Pin: `48cffad4e88ff4b9d2f1c7baffb314d1b3303792`; no source
shim is used for these methods. Alternatives reviewed include mifs (~286
stars), ITMO_FS (~148), and pyHSICLasso (~197). Stars are an explicit popularity
proxy among reviewed projects, not an installation census. The `mifs` package
name does not mean that it implements Battiti's MIFS criterion.

The current `skfeature-chappers==1.2.1` fork was inspected. Its DISR switches
formulas after the first step; its CIFE/MIFS wrappers consume LCSI ranks as
though they were selected indices. We use the original correct implementations.
ICAP and TraceRatio were dropped from consideration because the reviewed
reference code changed the paper's criterion/search. No reference source is
copied into FastSelect; external GPL code remains in an ignored audit tree.

For Gini's binary benchmark inputs, values are rounded to 1e-4 before either
implementation receives them. This ensures every distinct boundary survives
CART's float32 conversion and near-equality threshold. **The FastSelect
algorithm does not bin inputs:** it sorts original values and evaluates every
distinct-value split. Independent tests cover float64 separations of 1e-9 and
uint64 offsets of 2**63 that the CART input representation would merge.
Scikit-feature's erroneous 0.5 multiclass cap is not copied; our multiclass
definition is checked against the direct impurity oracle.

Baseline identity and prior retained source hashes:
[manifest](../../audit/new_algorithms_2/baseline_manifest.json).
Versions and reference module hashes:
[environment](../../audit/new_algorithms_2/environment.json).
All tuning measurements:
[baseline_tuning.json](../../audit/new_algorithms_2/baseline_tuning.json).

## Why the implementations improve

- Regression MI uses exact sorted-x lower bounds for the two-dimensional
  Chebyshev neighbor search, and binary searches for marginal counts. Numba
  parallelizes features. Public scaling/jitter, k and digamma formula are
  unchanged; no approximate neighbor search is used. Adversarial data/large k
  can approach quadratic search work, so this is not a universal KDTree claim.
- Gini sorts once per feature, then sweeps class counts and squared-count sums
  instead of repeatedly fitting estimators or rescanning every partition.
  Numeric symbol identity and the no-split parent are preserved.
- DISR/CIFE/MIFS reuse tested empirical entropy kernels and cache marginal
  terms within each fit. Candidate calculations run in parallel, while the
  greedy selection loop and original objectives remain sequential/exact.

MIFS subtracts beta times **summed**, not mean, redundancy (beta=0.5 by default).
CIFE adds summed conditional dependence, with no JMI subset-size averaging.
DISR sums normalized joint MI. Those distinctions are tested explicitly; they
are not aliases for the previously admitted mRMR/JMI/CMIM selectors.

## A candidate that did not enter the package

The ANOVA `f_classif` prototype passed independent rational-equation,
affine-invariance and probability checks, but scikit-learn was faster at all
screened shapes and all 1/4/8 Numba thread counts. Below is the best Numba
thread choice for each shape. These are early warmed six-call screening
medians, not sustained admission claims.

| Samples × features | sklearn ms | Best prototype ms | Threads | sklearn/prototype |
|---|---:|---:|---:|---:|
| 900 × 128 | 0.37 | 0.59 | 8 | 0.63× |
| 2,000 × 512 | 1.93 | 3.45 | 4 | 0.56× |
| 4,000 × 1024 | 7.35 | 18.77 | 8 | 0.39× |
| 8,000 × 2048 | 24.71 | 195.29 | 4 | 0.13× |

The prototype and its oracle remain outside `src` for provenance. Neither
`f_classif` nor its experimental module is exported or shipped. See
[screening data](../../audit/new_algorithms_2/anova_screen.json).

## Correctness, scope and integration

The full repository suite passed: **829 passed, 76 skipped**. Wheel/source builds and extracted-wheel fits passed with MLX and scikit-feature imports blocked. Rejected prototypes and external reference source are absent from both distributions. Documentation builds without warnings.
The [new tests](../../tests/test_more_algorithms.py) use Counter-based empirical
joint/conditional entropy, exhaustive numeric Gini partitions and direct
all-pairs Chebyshev KSG/Ross equations. They cover three random seeds,
multiclass/imbalanced/remapped labels, constants, exact ties, XOR
complementarity, beta=0/0.5/1.5, large categorical codes and numeric offsets,
duplicate samples, k=1 through n-1, mixed/discrete regression-MI columns and
singleton categories. Full estimator contract checks and real
Pipeline/GridSearchCV tests cover every addition.

The three discrete selectors use categorical inputs without automatic
discretization and return marginal MI in `feature_importances_` (bits), greedy
indices in `top_features_` and criterion values in `selection_scores_`.
DISR zero joint entropy explicitly contributes zero, a stated
constant-variable extension. MIFS can legitimately return negative selection
objectives; these are not clipped. Gini returns lower-is-better `gini_scores_`
and negates them for importances. Exact floating-point feature-score ties use
lower original indices. Regression MI accepts dense numeric regression input,
returns nats, supports SelectKBest, and requires k smaller than sample count.
Float32 inputs use float64 working values, including target jitter; conformance
there is checked against the reference using the same float64 representation.
All new estimators accept CPU/auto; explicit GPU requests raise. Existing
algorithms and their automatic CUDA/Metal choices are unchanged.

## Measurement, identity and reproduction

There are 100 fresh-process workers and 208.0 measured
seconds. Five rounds alternate order. Each worker warms imports, compilation,
pool startup and one complete call, then measures for at least 1.05 seconds;
each case/backend therefore totals at least 5.25 seconds. Validation, encoding,
scaling/jitter, sorting, allocation, selector construction and materialized
outputs are included. Full score arrays, selected indices and marginal
relevance agree in every paired round. The reference's additional full
relevance array is recomputed outside timing solely for verification.
Numba uses eight threads; BLAS is limited to one. No benchmark/test workload
was intentionally run concurrently.

Memory is macOS peak process RSS including imports/warm-up and excluding
separate live joblib children. It is not total pool/GPU allocation, and no
memory improvement is claimed. First-call latency includes JIT and is not
represented by warmed throughput.

The frozen winning prototype modules were promoted **byte-for-byte** to the
public package after admission. This report checks both copies against the
measured hashes, and checks that prior algorithm modules were unchanged.

- `RegressionMI.py`: `7dbcb379527a313d88219707bc83f8c223a20655b1419256567a13acc80a7066`
- `GiniIndex.py`: `e51fe741b25c62a13b2a62fe2f8feee06cdc5fd4ff6c8ed73a814b6f122d369e`
- `JointInformation.py`: `ee4988d9c98f82ed1f648e07ad3c59126c4d8e15b7c8f6fcc3195120306c299f`

Raw timings, every iteration, full outputs and decisions:
[sustained.json](../../audit/new_algorithms_2/sustained.json).
Summary: [admission_summary.json](../../audit/new_algorithms_2/admission_summary.json).
Validation record: [validation.json](../../audit/new_algorithms_2/validation.json).
Primary papers, scope and rejected directions:
[audit notes](../../audit/new_algorithms_2/README.md).

```sh
.venv/bin/python benchmarking/fetch_new_algorithm_baselines.py
.venv/bin/python benchmarking/benchmark_more_algorithms.py --tune
.venv/bin/python benchmarking/benchmark_more_algorithms.py --production \
  --output audit/new_algorithms_2/production_rerun.json
.venv/bin/python benchmarking/report_more_algorithms.py
```

The default harness reproduces the frozen prototype run; `--production` loads
the byte-identical public modules. Only sustained winners are included in the
installable package. The Python baseline is the tuned popular current
implementation, never FastSelect v0.2.1.
