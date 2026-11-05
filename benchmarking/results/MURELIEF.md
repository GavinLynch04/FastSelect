# μ-Relief: FastSelect CPU against scikit-rebate

Generated from [`murelief_vs_skrebate.json`](murelief_vs_skrebate.json) by `python benchmarking/benchmark_relief_variants.py --family murelief --report`.

- FastSelect `9cfd044` with uncommitted src changes, source SHA-256 `c87dbc6ed90b5b35…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- Both libraries use `n_neighbors=10`. On this continuous binary data both use the range-normalised difference and ReliefF's update, so their scores should agree; the parity table records the largest difference.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 2.9 to 4.9.

## MuRelief

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 0.66 ms | 60 ms | 91× | 102 KiB / 32 KiB | 428 KiB / 272 KiB |
| samples | 200 × 100 | 1.17 ms | 104 ms | 88× | 188 KiB / 48 KiB | 958 KiB / 624 KiB |
| samples | 500 × 100 | 3.76 ms | 298 ms | 79× | 446 KiB / 288 KiB | 3.4 MiB / 3.3 MiB |
| samples | 1,000 × 100 | 12.3 ms | 882 ms | 72× | 875 KiB / 416 KiB | 12.2 MiB / 12.9 MiB |
| samples | 2,000 × 100 | 43.4 ms | 2.95 s | 68× | 1.7 MiB / 400 KiB | 47.3 MiB / 48.8 MiB |
| features | 100 × 100 | 0.675 ms | 58.7 ms | 87× | 102 KiB / 32 KiB | 428 KiB / 208 KiB |
| features | 100 × 1,000 | 1.33 ms | 355 ms | 267× | 873 KiB / 16 KiB | 2.8 MiB / 1.0 MiB |
| features | 100 × 5,000 | 3.82 ms | 1.74 s | 456× | 4.2 MiB / 80 KiB | 13.4 MiB / 2.3 MiB |
| features | 100 × 10,000 | 8.11 ms | 4.78 s | 589× | 8.4 MiB / 0 KiB | 26.7 MiB / 8.3 MiB |

Score agreement (MuRelief, one fit per library):

| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |
|---|---|---:|---:|---:|---:|
| samples | 100 × 100 | 2.4e-17 | 0.0245 | 10 | — |
| samples | 200 × 100 | 1.7e-17 | 0.0188 | 10 | — |
| samples | 500 × 100 | 1.4e-17 | 0.0368 | 10 | — |
| samples | 1,000 × 100 | 2.4e-17 | 0.0391 | 10 | — |
| samples | 2,000 × 100 | 6.2e-17 | 0.0215 | 10 | — |
| features | 100 × 100 | 2.4e-17 | 0.0245 | 10 | — |
| features | 100 × 1,000 | 2.3e-17 | 0.032 | 10 | — |
| features | 100 × 5,000 | 3.1e-17 | 0.0358 | 10 | — |
| features | 100 × 10,000 | 4.9e-17 | 0.0453 | 10 | — |

## Summary

- Runtime, scikit-rebate / FastSelect: 68× to 589× (median 88×) across 9 cases.
- Traced host allocations, scikit-rebate / FastSelect: 3.2× to 27.9× (median 4.2×).
- Footprint growth at 2,000 × 100: FastSelect 400 KiB; scikit-rebate 48.8 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
