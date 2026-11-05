# VLS: FastSelect CPU against scikit-rebate

Generated from [`vls_vs_skrebate.json`](vls_vs_skrebate.json) by `python benchmarking/benchmark_relief_variants.py --family vls --report`.

- FastSelect `9cfd044` with uncommitted src changes, source SHA-256 `fceea7810d73393c…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- Both wrap ReliefF (`n_neighbors=10`) and score `2 * n_features / 100` random subsets of 100 features. The subsets are random in both libraries, so scores differ; this compares speed and memory.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 3.1 to 5.0.

## VLS + ReliefF (100-feature subsets)

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 1,000 | 22.4 ms | 1.05 s | 47× | 274 KiB / 1.3 MiB | 808 KiB / 96 KiB |
| samples | 300 × 1,000 | 26.3 ms | 2.62 s | 99× | 685 KiB / 1.8 MiB | 2.1 MiB / 1.3 MiB |
| samples | 1,000 × 1,000 | 72.3 ms | 9.18 s | 127× | 2.1 MiB / 7.7 MiB | 13.3 MiB / 4.2 MiB |
| features | 100 × 1,000 | 22.2 ms | 1.05 s | 47× | 273 KiB / 1.4 MiB | 808 KiB / 128 KiB |
| features | 100 × 5,000 | 113 ms | 5.28 s | 47× | 444 KiB / 656 KiB | 8.3 MiB / 1.4 MiB |
| features | 100 × 10,000 | 233 ms | 10.54 s | 45× | 638 KiB / 448 KiB | 31.6 MiB / 1.1 MiB |

## Summary

- Runtime, scikit-rebate / FastSelect: 45× to 127× (median 47×) across 6 cases.
- Traced host allocations, scikit-rebate / FastSelect: 2.9× to 50.6× (median 4.7×).
- Footprint growth at 1,000 × 1,000: FastSelect 7.7 MiB; scikit-rebate 4.2 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
