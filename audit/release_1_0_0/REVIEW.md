**FastSelect release review — 28 September 2026**

**Recommendation: do not publish the current tree as 1.0.0. Fix the confirmed correctness failures, then release 0.3.0 and use that release to establish the corrected scoring and API behavior.** A 1.0.0 release could follow once the regression cases and real CUDA verification are green. Changing the version alone would not make the current defects acceptable.

This updates the earlier, narrower readiness assessment: the existing CI was green, but the deeper review found failures that its tests do not cover.

Reviewed base: `bce53d13933a26e6b4a6659337efbab3fec9500b` from `development`, on local branch `codex/python-313-314-support`, including the pending Python 3.13/3.14 packaging and CI edits. No production implementation or release version was changed during this review. The new files in this directory are review evidence and tests. Source hashes are recorded in [results.json](results.json).

**Verification performed**

| Check | Result |
|---|---|
| Existing full suite, fresh Python 3.13 run | 109 passed, 33 CUDA tests skipped |
| Existing full suite, Python 3.14 | 109 passed, 33 skipped in the preceding compatibility work; production code unchanged |
| New independent numerical controls | 80 passed on each of Python 3.13.15 and 3.14.7 |
| New expected-behavior regression cases | **24 failed on each Python version**, across nine finding groups below |
| Relief-family ordinary-data comparisons | 600 comparisons passed; worst absolute deviation below 3.1e-7 |
| CUDA simulation, with a test-only import/availability shim | Ordinary mixed-data cases, MI units, CFS, MDR, and 65/513-feature scoring paths agreed; SURF threshold case differed from CPU |
| Ruff / Black | Existing maintained `src`, `tests`, and `benchmarking` passed; archived notebooks are excluded |
| Sphinx strict build | **Failed: 9 warnings** |
| Wheel, sdist, strict Twine checks | Passed on Python 3.13 and 3.14 in the preceding compatibility work |
| Physical NVIDIA GPU | Not available; driver/context behavior, races, and real-device accuracy remain unverified |

The 80 controls independently compute discrete MI and mRMR selection, CFS symmetrical uncertainty and search, count-feature chi-squared statistics/p-values, and MDR contingency tables and fold selection. They do not call private numerical production helpers. The 600 additional Relief comparisons use the repository's existing equation-oracle functions on 40 seeds each of discrete, continuous, and mixed data for five algorithm variants. Agreement with those oracles is evidence under their chosen conventions, not proof of every possible interpretation of a paper.

**Confirmed findings, ordered for remediation**

**R1 — P1: SURF/SURF* threshold ties change membership and scores between CPU and GPU.**

Location: [SURF.py:533](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/SURF.py:533), [SURF.py:350](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/SURF.py:350).

For `X=[[0.],[1.],[3.]]`, `y=[0,0,1]`, and `discrete_limit=0`, the normalized pair distances are 1/3, 2/3, and 1. The global radius is 2/3, so the middle-distance pair must be excluded under the documented strict rule. The threshold is computed from unscaled float32 values, while CPU scoring subtracts values after scaling, producing a slightly smaller distance for that pair. This is a membership error, not merely a small score-rounding difference.

| Mode | Expected | CPU | Simulated GPU |
|---|---:|---:|---:|
| SURF | -0.222222 | **+0.222222** | -0.222222 |
| SURF* | -0.888889 | **-0.444444** | -0.888889 |

Use a consistent, sufficiently precise distance representation for threshold calculation and scoring on both backends. Retain exact-boundary regression cases; widening tolerances on final scores would conceal the defect. The global-radius definition is described in [Urbanowicz et al., sections 2.1.2–2.1.5](https://arxiv.org/pdf/1711.08477).

**R2 — P1: premature float32 conversion destroys valid feature information.**

Location: [utils.py:130](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/utils.py:130), plus raw float32 transfers in all Relief-family GPU callers and SURF preprocessing.

Adding `1e8` to a valid float64 continuous dataset turns all scores into zero for ReliefF, SURF, SURF*, MultiSURF, and MultiSURF*. For example, ReliefF's first-feature score changes from about `0.310895` to `0`. Range-normalized differences should be invariant to this translation. Discrete values also collide: relabeling categories 0/1 as `2**25` / `2**25+1` changes ReliefF's perfect-signal score from approximately 1 to 0.

Normalize continuous columns in adequate precision, including subtraction of the column minimum, before narrowing. Encode discrete categories into compact exact integer codes rather than casting their raw values. Apply the same representation to CPU and CUDA. This is a change in numerical correctness that should receive both oracle and backend-boundary tests.

**R3 — P1: mRMR category encoding can reject valid discrete representations or silently lose information.**

Location: [mRMR.py:19](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/mRMR.py:19), [mRMR.py:114](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/mRMR.py:114).

The encoder uses `empty_like`, leaving float-coded categories as floats; the downstream MI validator rejects them. This affects either float X or float y, including exact 0.0/1.0 values. More seriously, combining uint64 X and signed integer y into one vocabulary promotes values to float64: two distinct categories `2**63` and `2**63+1` collapse, and a perfect predictor receives 0 bits instead of 1.

Encode each feature and the target independently, preserving symbol identity and writing codes to an explicitly adequate integer dtype. The ordinary MID/MIQ control cases passed; the defect is upstream of the selection equations. Those equations were checked against [Peng, Long and Ding, equation 7](https://home.penglab.com/papersall/docpdf/2005_TPAMI_FeaSel.pdf).

**R4 — P1: the public MI functions silently alias large nonnegative integer codes.**

Location: [mutual_information.py:20](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/mutual_information.py:20).

`_validate_discrete` checks negativity before converting to int32 and never checks representability. For `x=[0,2**32,0,2**32]` and `y=[0,1,0,1]`, the public function returns 0 bits instead of 1. Other oversized values can become negative indices after conversion. Remap observed symbols or reject unsupported ranges before entering JIT/CUDA kernels. Also define empty-input behavior and avoid allocating histograms from sparse, arbitrarily large code values.

**R5 — P1: MDR validates genotypes after lossy conversion, and prediction lacks schema validation.**

Location: [MDR.py:226](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/MDR.py:226), [MDR.py:326](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/MDR.py:326).

Training accepts `0.5`, `256`, `257`, and `-256` as genotypes because uint8 conversion truncates or wraps them before checking 0/1/2 membership. Prediction likewise accepts fractional genotypes and a different feature count. Invalid prediction indices can reach the compiled lookup-table indexing path; the observed probe returned labels for genotype 3 instead of rejecting it.

Validate finite, integral values and membership in `{0,1,2}` before casting in both fit and predict. Record and validate training feature count/names. Do not rely on compiled indexing to catch errors. The independent MDR controls passed for valid data, including inclusive risk-table comparisons and final predictions; these controls implement the repository's balanced-accuracy variant, not every MDR variant in the literature.

**R6 — P1: advertised scikit-learn compatibility is broken for MDR, CFS, and mRMR.**

Locations: [MDR.py:148](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/MDR.py:148), [MDR.py:172](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/MDR.py:172), class declarations in CFS and mRMR.

`clone(MDR(backend='cpu'))` raises because the constructor replaces the backend parameter with `backend.lower()`. `is_classifier(MDR())` returns false. CFS and mRMR expose no transformer tags; scikit-learn's full checker stops immediately with a tag/inheritance-order error for each. Existing checks exercise ReliefF/SURF/MultiSURF and a mock-based TuRF, so these problems escape CI.

Store constructor parameters unchanged, normalize them at fit time, and put mixins before BaseEstimator. Add estimator-contract checks for every exported estimator, plus real Pipeline/GridSearchCV smoke tests. These requirements are explicit in [scikit-learn's estimator development guide](https://scikit-learn.org/stable/developers/develop.html).

**R7 — P1: CFS silently selects the wrong named feature when DataFrame columns are reordered.**

Location: [CFS.py:505](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/CFS.py:505).

Fit on `['signal','noise']`, then transform `['noise','signal']`: the fitted selector returns `noise` because transform uses positional `.iloc` without checking names. This can silently corrupt inference pipelines. Array-like list input also fails with a raw indexing TypeError. Validate schema before selecting, while retaining the intended DataFrame return type.

**R8 — P2: TuRF fails with a real Relief estimator when the retained subset becomes smaller than the base selector's requested output count.**

Location: [TuRF.py:116](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/TuRF.py:116).

`TuRF(ReliefF(n_features_to_select=2, ...), n_features_to_select=1)` crashes at the final refit. TuRF only needs scores, but the cloned base estimator still requests two output features from a one-feature input. Use a scoring configuration compatible with each active feature set, without mutating the user's original estimator, and test actual wrapped selectors. The current mock estimator does not enforce the restriction and hides this integration failure.

**R9 — P2: MultiSURF can select features outside `feat_idx`.**

Location: [MultiSURF.py:594](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/MultiSURF.py:594).

Unscored features receive zero, then all features are ranked. With seed 1 in the retained test, `feat_idx=[0,1]` returns feature 3. The evaluated features scored approximately -3.9e-9 and -0.05556, so the unscored zeros win. Rank only the requested subset and validate index shape, bounds, duplicates, and requested selection count. Document this public argument or remove it from the public contract before declaring the API stable.

**R10 — P2: CFS's default thread setting ignores the configured Numba cap.**

Location: [CFS.py:449](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/CFS.py:449).

In a fresh process with `NUMBA_NUM_THREADS=2`, default `CFS(backend='cpu')` raises `ValueError: The number of threads must be between 1 and 2`. It requests `NUMBA_DEFAULT_NUM_THREADS` rather than the configured pool limit. This was also the cause of an initial constrained-thread run failing during the earlier compatibility work; the normal run passed after removing the cap. Respect the configured maximum and validate explicit `n_jobs` values. Relief-family callers already use the configured limit.

**R11 — P2: importing the package assumes private CUDA runtime internals and patches them globally.**

Location: [utils.py:5](/Users/gavinlynch/Desktop/FastSelect/src/fast_select/utils.py:5).

`NUMBA_ENABLE_CUDASIM=1 python -c 'import fast_select'` fails because the simulator has no `_Runtime`. Import also installs the Windows-motivated monkey patch on every platform/version, and context-push exceptions are swallowed. The simulator failure is confirmed; the effect on real multi-GPU/context behavior remains a risk requiring hardware testing, not a claimed reproduced driver failure.

Scope or remove the workaround using supported APIs, supported versions, and targeted regression tests. The audit simulation shim bypasses only this import/availability machinery and leaves numerical kernels unchanged; it must not be interpreted as validation of the production context code.

**Release and documentation findings**

- **The publishing workflow bypasses release checks.** [publish-to-pypi.yml](/Users/gavinlynch/Desktop/FastSelect/.github/workflows/publish-to-pypi.yml) builds and uploads on release publication without running/waiting for tests, strict Twine validation, or checking that the tag agrees with package metadata. The test workflow only triggers on named branch pushes and PRs. Gate publication on successful checks of the exact release commit/artifact, including a tag/version assertion.
- **Strict docs fail.** There is a missing `_static` directory and eight unresolved scikit-learn metadata-routing references. The current workflow has no documentation build job. The generated pages exist, but the `-W --keep-going` run exits unsuccessfully. See [docs-build-results.txt](docs-build-results.txt).
- **Coverage claims need correction.** This CPU run covered 673/814 counted lines (82.68%), with **1,712 lines excluded**, many of them numerical kernels via `pragma: no cover`. The paper's 94% and historical changelog's 95% are not evidence that 94–95% of numerical/GPU behavior was checked in this review. Report environment-specific host-code coverage separately from kernel/oracle verification.
- **Release metadata still needs reconciliation.** `CITATION.cff` contains literal version 1.0.0 and date July 24 even though the changelog says the version is no longer hand-maintained. The paper still states CI ends at Python 3.12. The README citation still describes the older Zenodo release. Choose a version/date and use the correct DOI record before publishing; do not relabel an old immutable DOI record as a new release.
- **Benchmark provenance can improve.** The stored timed rows inspected all meet the five-second minimum. They record paths and broad labels such as `working-tree`, but no exact commit SHA; several files use zero for unmeasured GPU memory. Record source hashes, runtime dependencies, device model, and an explicit unavailable value for memory. I did not rerun performance benchmarks because no performance implementation was changed and there is no NVIDIA device here.

**Scientific scope and remaining questions**

CFS's SU formula, merit computation, and ordinary-data best-first results passed independent checks based on [Hall's thesis, chapters 3–4](https://ml.cms.waikato.ac.nz/publications/1999/99MH-Thesis.pdf). This is not a guarantee that heuristic search finds the global optimum. The implementation uses its documented strict-improvement/tie policy; that policy should be stated alongside the chosen CFS variant.

There is a source ambiguity worth recording for MultiSURF*: section 2.1.4 prose and the worked far-hit/far-miss tables in the [2018 benchmarking paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6299838/) give inconsistent sign descriptions. The current implementation and oracle follow the tables' negative-hit/positive-miss similarity convention. I have **not** classified this as a confirmed reversed-sign bug. Cite the precise convention and resolve the prose discrepancy with an authoritative source/erratum before claiming unqualified paper identity.

MDR's `best_mean_testing_ba_` averages only folds in which the selected combination won. It is not a full out-of-fold performance estimate of that fixed combination. Document that meaning and use an outer evaluation when reporting predictive performance. The use of balanced accuracy and a case/control-ratio threshold should be cited as the applicable MDR variant; [Ritchie and colleagues' later methodological discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC3500181/) distinguishes these choices for imbalanced data.

The classification-only Relief estimators also accept continuous numeric targets as many distinct classes rather than rejecting them. Input/parameter validation needs systematic boundary tests, including noninteger neighbor counts, unsupported target types, empty inputs, and invalid feature subsets. These are additional API-hardening gaps, separate from the scored-data defects above.

No physical CUDA, Windows, Linux, Python 3.9–3.12 rerun, minimum-dependency matrix, or new throughput benchmark was performed in this local audit. Earlier GitHub CI passed on 3.9–3.12, but that does not replace testing the eventual release commit. Ordinary CUDA simulation cannot prove race freedom or driver compatibility.

**Version decision and completion criteria**

SemVer 1.0.0 declares a stable public API; 0.x permits continued evolution. It does not require perfection, but the confirmed changes still needed here affect observable scores, accepted inputs, selected features, and estimator integration. That makes **0.3.0 the more honest next release after fixes**, with an explicit note that corrected selections can differ from 0.2.1. See [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Before either release, fix R1–R7's wrong-result/data-validation/integration failures and resolve the supported behavior for R8–R11. Move the applicable regression tests into the normal suite. Before calling the library 1.0.0, require the full suite, independent oracle cases, exact-boundary CPU/GPU tests on NVIDIA hardware, every estimator's integration checks, strict documentation/build validation, and publication checks for the exact tagged commit. Publish a clear support matrix and variant/precision policy so future changes can honor a stable contract.

**Reproducing this audit**

From the repository root in an environment with `fast-select[test]` installed:

```bash
PYTHONPATH=src python -m pytest audit/release_1_0_0/test_release_review.py --no-cov --tb=short
python audit/release_1_0_0/run_relief_oracles.py
NUMBA_ENABLE_CUDASIM=1 NUMBA_NUM_THREADS=2 python audit/release_1_0_0/run_cuda_simulator.py
```

The first command intentionally fails against the reviewed tree: these tests express the behavior that fixes must satisfy. They live outside the default `tests/` collection so the review does not silently alter the existing CI gate. The two script outputs and both Python-version regression outputs are preserved alongside this report. The CUDA runner clearly contains test-only runtime shims and must not be used as a production workaround.
