# MCFS: FastSelect CPU against scikit-feature

Generated from [`mcfs_comparison.json`](mcfs_comparison.json) by `python benchmarking/benchmark_relief_variants.py --family mcfs --report`.

- FastSelect `b37a455` with uncommitted src changes, source SHA-256 `d1cbfeded87ec4d6…`; scikit-feature 48cffad (pinned tree), source SHA-256 `4277e17ece987f06…` over the files used.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative).
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- `MCFS(10, n_clusters=5)` against the pinned scikit-feature `mcfs` (commit 48cffad) given the same Euclidean binary 5-NN graph (`construct_W(X, metric='euclidean', weight_mode='binary', k=5)`). scikit-feature solves a dense eigenproblem, fits plain LARS rather than LARS-Lasso and scores by the largest signed coefficient. The agreement table compares the largest absolute coefficient for both: it agrees to rounding unless the Lasso path drops a variable before reaching 10 nonzero coefficients, where plain LARS (which never drops one) no longer solves Eq. (30). The selections differ because of the signed score.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 2.2 to 3.5.

## MCFS (K = 5)

| Scenario | n × p | FastSelect | scikit-feature | scikit-feature / FastSelect | FastSelect memory | scikit-feature memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 500 × 100 | 9.64 ms | 45.4 ms | 4.7× | 2.1 MiB / 3.2 MiB | 7.8 MiB / 19.7 MiB |
| samples | 1,000 × 100 | 14.3 ms | 243 ms | 17× | 8.0 MiB / 7.9 MiB | 30.9 MiB / 37.7 MiB |
| samples | 2,000 × 100 | 32.7 ms | 1.65 s | 50× | 31.3 MiB / 32.8 MiB | 122.9 MiB / 130.0 MiB |
| samples | 4,000 × 100 | 94.7 ms | 14.03 s | 148× | 123.6 MiB / 127.7 MiB | 489.9 MiB / 518.9 MiB |
| features | 500 × 100 | 9.6 ms | 45.3 ms | 4.7× | 2.1 MiB / 4.4 MiB | 7.8 MiB / 20.2 MiB |
| features | 500 × 1,000 | 14.6 ms | 54.4 ms | 3.7× | 4.1 MiB / 4.8 MiB | 13.7 MiB / 19.9 MiB |
| features | 500 × 5,000 | 63.3 ms | 105 ms | 1.7× | 20.1 MiB / 7.0 MiB | 44.9 MiB / 21.2 MiB |

Score agreement (MCFS (K = 5), one fit per library):

| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |
|---|---|---:|---:|---:|---:|
| samples | 500 × 100 | 0.0017 | 0.00287 | 6 | no |
| samples | 1,000 × 100 | 1.9e-16 | 0.00109 | 6 | no |
| samples | 2,000 × 100 | 6e-17 | 0.00105 | 4 | no |
| samples | 4,000 × 100 | 2.1e-16 | 0.000656 | 6 | no |
| features | 500 × 100 | 0.0017 | 0.00287 | 6 | no |
| features | 500 × 1,000 | 4.1e-17 | 0.00166 | 5 | no |
| features | 500 × 5,000 | 2.6e-16 | 0.00137 | 7 | no |

## Summary

- Runtime, scikit-feature / FastSelect: 1.7× to 148× (median 4.7×) across 7 cases.
- Traced host allocations, scikit-feature / FastSelect: 2.2× to 4.0× (median 3.7×).
- Footprint growth at 4,000 × 100: FastSelect 127.7 MiB; scikit-feature 518.9 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads).
