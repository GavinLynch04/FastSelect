# Continuous mRMR (FCQ): FastSelect CPU against mrmr_selection, scikit-learn + NumPy FCQ

Generated from [`mrmr_fcq_comparison.json`](mrmr_fcq_comparison.json) by `python benchmarking/benchmark_relief_variants.py --family mrmr --report`.

- FastSelect `9cfd044` with uncommitted src changes, source SHA-256 `fceea7810d73393c…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`; mrmr_selection 0.2.8, source SHA-256 `9cc75eed8057dd5b…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- `mRMR(10, method="FCQ")` against `mrmr_selection`'s `mrmr_classif(K=10)` (its default FCQ) and against the same criterion written with scikit-learn's `f_classif` and vectorised NumPy correlations (`|Z.T @ z|` per step on standardised columns, BLAS single-threaded like every worker here). `mrmr_selection` gets the faster of `n_jobs=-1` and `n_jobs=1`. FastSelect's selections equal the NumPy FCQ's at every shape; `mrmr_selection`'s differ wherever its 0.001 floor on each correlation changes a quotient, and equal the NumPy FCQ with that floor added at every shape (checked separately).

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 2.9 to 3.8.

## FCQ

| Scenario | n × p | FastSelect | mrmr_selection | scikit-learn + NumPy FCQ | mrmr_selection / FastSelect | scikit-learn + NumPy FCQ / FastSelect | FastSelect memory | mrmr_selection memory | scikit-learn + NumPy FCQ memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| samples | 500 × 1,000 | 3.31 ms | 454 ms | 2.42 ms | 137× | 0.7× | 4.0 MiB / 240 KiB | 16.0 MiB / 32 KiB | 9.6 MiB / 17.5 MiB |
| samples | 2,000 × 1,000 | 8.79 ms | 896 ms | 11.7 ms | 102× | 1.3× | 15.4 MiB / 176 KiB | 38.8 MiB / 48 KiB | 38.2 MiB / 23.0 MiB |
| samples | 10,000 × 1,000 | 53.1 ms | 1.26 s | 73.2 ms | 24× | 1.4× | 76.5 MiB / 0 KiB | 161.0 MiB / 80 KiB | 190.8 MiB / 114.5 MiB |
| features | 500 × 1,000 | 3.21 ms | 460 ms | 2.43 ms | 143× | 0.8× | 4.0 MiB / 32 KiB | 16.0 MiB / 0 KiB | 9.6 MiB / 17.4 MiB |
| features | 500 × 5,000 | 10.7 ms | 2.76 s | 13.4 ms | 258× | 1.3× | 19.8 MiB / 480 KiB | 231.1 MiB / 128 KiB | 47.8 MiB / 29.2 MiB |
| features | 500 × 20,000 | 38.3 ms | 7.40 s | 64.8 ms | 193× | 1.7× | 79.1 MiB / 2.8 MiB | 3212.4 MiB / 2.2 MiB | 191.0 MiB / 115.2 MiB |

Score agreement (FCQ, one fit per library):

| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |
|---|---|---:|---:|---:|---:|
| samples | 500 × 1,000 | — | — | 5 | no |
| samples | 2,000 × 1,000 | — | — | 10 | yes |
| samples | 10,000 × 1,000 | — | — | 9 | no |
| features | 500 × 1,000 | — | — | 5 | no |
| features | 500 × 5,000 | — | — | 4 | no |
| features | 500 × 20,000 | — | — | 3 | no |

## Summary

- Runtime, mrmr_selection / FastSelect: 24× to 258× (median 140×) across 6 cases.
- Traced host allocations, mrmr_selection / FastSelect: 2.1× to 40.6× (median 4.0×).
- Runtime, scikit-learn + NumPy FCQ / FastSelect: 0.7× to 1.7× (median 1.3×) across 6 cases.
- Traced host allocations, scikit-learn + NumPy FCQ / FastSelect: 2.4× to 2.5× (median 2.4×).
- Footprint growth at 10,000 × 1,000: FastSelect 0 KiB; mrmr_selection 80 KiB; scikit-learn + NumPy FCQ 114.5 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
