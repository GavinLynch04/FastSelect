"""Recheck batch-two evidence and promoted source identity, then write report."""

import hashlib
import json
import statistics
from pathlib import Path

from benchmark_more_algorithms import CASES, MODULES, verify

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/new_algorithms_2"
result = json.loads((AUDIT / "sustained.json").read_text())
assert result["rounds"] >= 5
decisions = verify(result)
assert all(row["admitted"] for row in decisions.values())
for name in MODULES:
    expected = result["source_sha256"][name]
    for directory in (AUDIT / "prototypes", ROOT / "src/fast_select"):
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, (name, directory)
for name, expected in result["baseline_manifest"]["retained_prior_source_sha256"].items():
    if name != "__init__.py":  # public exports are intentionally extended
        assert hashlib.sha256((ROOT / "src/fast_select" / name).read_bytes()).hexdigest() == expected, name

rows = result["measurements"]
table, summaries = [], {}
for case, (name, n, p) in CASES.items():
    group = {
        backend: [r for r in rows if r["case"] == case and r["backend"] == backend] for backend in ("reference", "cpu")
    }
    reference = statistics.median(r["median_seconds"] for r in group["reference"])
    candidate = statistics.median(r["median_seconds"] for r in group["cpu"])
    ratios = decisions[case]["paired_speedups"]
    baseline = (
        "sklearn MI, 8 processes"
        if name == "mutual_info_regression"
        else "sklearn CART, 4 processes" if name == "GiniIndex" else "original scikit-feature"
    )
    table.append(
        f"| {name} | {n:,} × {p} | {baseline} | {reference*1000:.2f} | "
        f"{candidate*1000:.2f} | {reference/candidate:.2f}× | {min(ratios):.2f}–{max(ratios):.2f}× |"
    )
    summaries[case] = {
        "algorithm": name,
        "shape": [n, p],
        "reference_ms": reference * 1000,
        "numba_ms": candidate * 1000,
        "central_speedup": reference / candidate,
        **decisions[case],
    }

screen = json.loads((AUDIT / "anova_screen.json").read_text())
anova_table = []
for n, p in dict.fromkeys((r["n"], r["p"]) for r in screen):
    shape = [r for r in screen if r["n"] == n and r["p"] == p]
    reference = next(r for r in shape if r["numba_threads"] == 0)
    candidate = min((r for r in shape if r["numba_threads"]), key=lambda r: r["median"])
    anova_table.append(
        f"| {n:,} × {p} | {reference['median']*1000:.2f} | {candidate['median']*1000:.2f} | "
        f"{candidate['numba_threads']} | {reference['median']/candidate['median']:.2f}× |"
    )

validation_path = AUDIT / "validation.json"
validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
suite = validation.get("full_suite", {})
validation_text = (
    f"The full repository suite passed: **{suite['passed']} passed, {suite['skipped']} skipped**."
    if suite
    else "The candidate equation/contract/integration checks passed before benchmarking."
)
if validation.get("wheel_optional_dependency_blocking_smoke") == "passed":
    validation_text += (
        " Wheel/source builds and extracted-wheel fits passed with MLX and scikit-feature imports blocked."
        " Rejected prototypes and external reference source are absent from both distributions."
    )
if validation.get("documentation_build") == "passed with warnings treated as errors":
    validation_text += " Documentation builds without warnings."
measured = sum(r["measured_seconds"] for r in rows)
hashes = "\n".join(f"- `{name}`: `{sha}`" for name, sha in result["source_sha256"].items())
report = f"""# Five more algorithms: second admission batch

**Regression MI, GiniIndex, DISR, CIFE and MIFS passed admission and were added
as parallel Numba CPU implementations.** Each preserved scores and selected
indices and beat its tuned external baseline by at least 3% in every round on
both measured shapes. The initial ANOVA F prototype lost to scikit-learn and
was excluded; MIFS replaced it. No new Metal/CUDA backend is implemented or
claimed by this batch.

## Measured complete-call performance

Apple M1 Pro, 8 CPU cores, 16 GiB RAM, macOS 27.0.1 arm64; Python 3.13.17,
NumPy 2.5.4, Numba 0.68.0, SciPy 1.18.1 and scikit-learn 1.9.1.
Run started {result['timestamp_utc']}. Times are milliseconds, using the median
of five per-round medians. Central speedup divides these central times;
the paired range shows the minimum and maximum same-round speedup.

| Algorithm | Samples × features | Baseline | External ms | Numba ms | Central gain | Paired round range |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(table)}

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
{chr(10).join(anova_table)}

The prototype and its oracle remain outside `src` for provenance. Neither
`f_classif` nor its experimental module is exported or shipped. See
[screening data](../../audit/new_algorithms_2/anova_screen.json).

## Correctness, scope and integration

{validation_text}
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

There are {len(rows)} fresh-process workers and {measured:.1f} measured
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

{hashes}

Raw timings, every iteration, full outputs and decisions:
[sustained.json](../../audit/new_algorithms_2/sustained.json).
Summary: [admission_summary.json](../../audit/new_algorithms_2/admission_summary.json).
Validation record: [validation.json](../../audit/new_algorithms_2/validation.json).
Primary papers, scope and rejected directions:
[audit notes](../../audit/new_algorithms_2/README.md).

```sh
.venv/bin/python benchmarking/fetch_new_algorithm_baselines.py
.venv/bin/python benchmarking/benchmark_more_algorithms.py --tune
.venv/bin/python benchmarking/benchmark_more_algorithms.py --production \\
  --output audit/new_algorithms_2/production_rerun.json
.venv/bin/python benchmarking/report_more_algorithms.py
```

The default harness reproduces the frozen prototype run; `--production` loads
the byte-identical public modules. Only sustained winners are included in the
installable package. The Python baseline is the tuned popular current
implementation, never FastSelect v0.2.1.
"""
(AUDIT / "admission_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
(ROOT / "benchmarking/results/NEW_ALGORITHMS_2.md").write_text(report)
print("Five additional algorithms admitted; measured and public source hashes match.")
