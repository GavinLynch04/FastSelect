# v0.4.1 CPU optimization pass — Apple M1 Pro, October 28, 2026

## Scope and baseline

Every algorithm was profiled; candidate changes were kept only when they were
byte-identical to the baseline **and** faster in sustained, alternating,
fresh-process rounds. Experiments that did not win were reverted (listed
below).

- Baseline: `HEAD` of `development`, commit
  `6707dbca50d7813c319245a0184923cf6cf4593d` (0.4.0 plus documentation), the
  fastest correct implementation at the start. Source fingerprint
  `12190bf1d906834abb9f5bdd4054c4befec671583cb0048b1da7b268f083ca7b`.
- Candidates were frozen as commit objects with `git stash create` and measured
  with `--candidate-ref`, so later edits could not leak into a running
  comparison:
  - `ffc370aef77d25a9e872e9384185e789e6e47206`: MI, mRMR and MDR results.
  - `4d8fa485a66b7e496bfb665164636e0522eb6368`: CFS and LaplacianScore
    results. Its source fingerprint,
    `c8d820f01f108e2805df5516074328f8ea34dfd113a528de66b6d2bae911b64b`, is the
    final source. The first snapshot differs from it only in `CFS.py`.
- The fingerprint hashes the sorted names and contents of
  `src/fast_select/*.py`. No commit, push, merge or release was made.

Machine: Apple M1 Pro (8 cores), macOS 27.0.1 arm64. Python 3.13.17, NumPy
2.5.4, Numba 0.68.0, scikit-learn 1.9.1. Eight Numba threads; BLAS, OpenMP and
Accelerate limited to one thread in workers.

Method: `compare_head_vs_current.py`, 8 rounds per case, version order
alternating each round, a fresh process per version and round, JIT and the
actual shape warmed first, and at least 5 s of measured fits per version per
round (about 80 s of fitting per case). Full outputs (scores, selections,
redundancy matrices, MDR models and predictions) were compared with
`--rtol 0 --atol 0`. Speedup is the ratio of median round-mean latencies. The
sign test and the t-test on log ratios use the 8 paired rounds (one-sided; the
sign test's floor at 8 rounds is p = 0.0039). Mann-Whitney pools every
individual fit, which is high n but not fully independent. Background system
load varied (load averages from about 4 to 37 on 8 cores); alternation and
pairing are what make the comparisons robust to it. Traced memory is the
`tracemalloc` peak of one warmed fit (Python/NumPy allocations only).

## Retained changes

| Selector (data) | Shape | Baseline | Final | Speedup | Faster rounds | Sign p | t-test p | Traced memory |
|---|---|---|---|---|---|---|---|---|
| LaplacianScore | 900 × 128 | 11.66 ms | 4.53 ms | **2.57x** | 8/8 | 0.0039 | 1e-14 | 13.3 → 6.5 MiB |
| LaplacianScore | 3,000 × 64 | 121.5 ms | 32.9 ms | **3.70x** | 8/8 | 0.0039 | 2e-9 | 146.1 → 69.8 MiB |
| MDR, k=2, cv=3 | 900 × 128 | 8.27 ms | 5.31 ms | **1.56x** | 8/8 | 0.0039 | 1e-5 | unchanged |
| MDR, k=2, cv=3 | 2,000 × 256 | 65.9 ms | 17.9 ms | **3.67x** | 8/8 | 0.0039 | 8e-12 | unchanged |
| MDR, k=2, cv=3 (per-sample path) | 128 × 512 | 52.1 ms | 51.9 ms | 1.00x (no regression) | 6/8 | 0.14 | 0.59 | unchanged |
| MI matrices, 4 states | 900 × 128 | 4.83 ms | 2.01 ms | **2.40x** | 8/8 | 0.0039 | 9e-9 | +0.03 MiB |
| MI matrices, 4 states | 2,000 × 512 | 122.6 ms | 35.1 ms | **3.50x** | 8/8 | 0.0039 | 3e-12 | +0.05 MiB |
| MI matrices, 16 states | 900 × 128 | 8.56 ms | 7.45 ms | 1.15x (weak) | 7/8 | 0.035 | 0.060 | +0.03 MiB |
| mRMR MID, 4 states | 900 × 128 | 6.15 ms | 3.18 ms | **1.93x** | 8/8 | 0.0039 | 4e-8 | +0.03 MiB |
| CFS, 4 states | 900 × 128 | 12.01 ms | 7.94 ms | **1.51x** | 8/8 | 0.0039 | 7e-10 | unchanged |
| CFS, 4 states | 2,000 × 512 | 245.1 ms | 187.2 ms | **1.31x** | 8/8 | 0.0039 | 5e-4 | unchanged |
| CFS, 16 states | 900 × 128 | 16.17 ms | 12.81 ms | **1.26x** | 8/8 | 0.0039 | 2e-6 | unchanged |

Raw per-fit timings, provenance and significance are in the matching
`v041_*.json` files. The MI traced-memory increase is the small NumPy column
maximum reduction now done before the kernel.

What changed, and why the output cannot move:

- **LaplacianScore** (`_nearest_rows`): a parallel per-row insertion selection
  of the k smallest distances replaces `np.argpartition` over the dense matrix
  (which allocated an n × n index array) and the n × n tie-check arrays. Where
  the k-smallest set is unique it equals any partial selection's set. Rows
  with a boundary tie still take the reference `np.argsort` membership, and
  rows holding NaN keep the original whole-row `argpartition` logic. COO→CSR
  conversion sorts row indices, so the graph is byte-identical. Checked on
  4,000 graphs (ties, offsets, heat kernel, overflow).
- **MDR** (`_batch_balanced_accuracy_bits`): genotypes and cases become
  per-locus bitmasks; a cell's case/control counts are popcounts of ANDed
  masks. The counts are exact integers and feed the same high-risk rule
  (`_balanced_accuracy_from_counts`, inlined into both kernels). Used only
  where calibration showed it faster (at least 160 fold samples and
  `3**k * ceil(n/64) * 3 <= n * k`); otherwise the unchanged per-sample
  kernel runs. Checked on 1,177 kernel batches (k = 1..6) and 639 fits.
- **Discrete MI** (`_batch_mi_counted_cpu`): when every code is below 16, each
  column's marginal counts are computed once instead of once per pair, and
  small joint tables are counted in four interleaved integer sub-tables. The
  per-pair log sum is the same compiled helper `_mi_pair_cpu` uses. Larger
  codes run the 0.4.0 kernel. Checked on 600 matrices (bits and nats, sparse
  and remapped codes, constant columns), 870 single pairs and 584 mRMR fits.
- **CFS**: integer inputs are encoded by the existing parallel
  `_encode_columns_numba` (the same codes as per-column `np.unique`); small
  contingency tables use interleaved integer counts below 2**24 samples; the
  best-first search converts candidate merits once with `tolist()` and tracks
  insertion points by a sorted walk instead of `bisect`. Heap entries,
  traversal and tie order are unchanged. Checked on 6,000 searches (many
  exact ties) and 400 fits.

## Tried and not retained

| Experiment | Result | Action |
|---|---|---|
| MDR: feature-major layout and branch-free per-sample counts | 0.92-0.97x at 600 × 128, 1.17-1.32x only at 1,300 × 512 | Replaced by the bitwise path |
| MDR: shared risk-rule helper without forced inlining | Per-sample kernel 0.75x (call not inlined); snapshot `62406fd` measured 0.80x at 128 × 512 (console only; its JSON was superseded by the re-run) | Fixed with `inline="always"`; 1.00x |
| MI: interleaved counting for every table size | 0.85-0.94x at 16-32 states | Restricted to codes below 16 |
| MI: helpers inlined into the parallel kernel | Last-bit (1 ulp) differences vs 0.4.0 on 29% of entries | Reverted; equation helper stays a separate compiled function |
| JMI/CMIM/FCBF/MIFS/CIFE/DISR: interleaved entropy counting | 0.97-1.03x (per-step overhead dominates) | Reverted |
| SURF: cache-blocked column gather for the global radius | 0.98-1.04x (the sort dominates) | Not applied |

## Not changed

- **ReliefF, SURF/SURF\*, MultiSURF/MultiSURF\*, TuRF** (CPU): runtime is
  dominated by the n²·p pairwise-distance loops, already parallel and
  vectorised. The remaining algorithmic gain, computing each symmetric pair
  once, needs an n × n float64 distance matrix (800 MB at n = 10,000), and
  ReliefF's float32 neighbour buffers make its ties order-dependent, so a
  symmetric traversal could change neighbour membership. TuRF only refits
  these estimators.
- **GiniIndex, `mutual_info_classif` (Ross), `mutual_info_regression` (KSG),
  Chi2**: no exact, profitable change was found. Gini and the Ross/KSG
  estimators are dominated by per-feature sorts and exact float64 neighbour
  predicates; Chi2 takes about 0.5 ms, mostly input validation.
- **GPU paths** (CUDA, Metal) are untouched. CPU/GPU agreement is unchanged
  because CPU outputs are byte-identical: the Metal parity tests pass, and the
  CUDA kernels pass `tests/test_cuda_simulator.py` under
  `NUMBA_ENABLE_CUDASIM=1`, which validates kernel logic, not NVIDIA hardware.

## Reproduce

```sh
SNAPSHOT=$(git stash create)   # or any commit containing the candidate
python benchmarking/compare_head_vs_current.py --baseline-ref 6707dbc \
  --candidate-ref "$SNAPSHOT" --estimators LaplacianScore CFS mRMR MDR \
  --threads 8 --samples 900 --features 128 --rounds 8 --min-seconds 5 \
  --rtol 0 --atol 0 --gate --json-output benchmarking/results/rerun.json
```
