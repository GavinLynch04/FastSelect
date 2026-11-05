# STIR: FastSelect CPU against FastSelect MultiSURF, scikit-rebate MultiSURF

Generated from [`stir_comparison.json`](stir_comparison.json) by `python benchmarking/benchmark_relief_variants.py --family stir --report`.

- FastSelect `9cfd044` with uncommitted src changes, source SHA-256 `fceea7810d73393c…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- No Python STIR exists to compare with (the authors' implementation is R, which was not run). STIR with its default MultiSURF neighbourhood is shown against MultiSURF itself, which computes the same neighbourhood's scores without statistics, in FastSelect and in scikit-rebate.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 3.9 to 9.3.

## STIR (MultiSURF neighbourhood)

| Scenario | n × p | FastSelect | FastSelect MultiSURF | scikit-rebate MultiSURF | FastSelect MultiSURF / FastSelect | scikit-rebate MultiSURF / FastSelect | FastSelect memory | FastSelect MultiSURF memory | scikit-rebate MultiSURF memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 1.33 ms | 0.68 ms | 56.5 ms | 0.5× | 42× | 163 KiB / 80 KiB | 53 KiB / 96 KiB | 423 KiB / 176 KiB |
| samples | 200 × 100 | 1.75 ms | 0.902 ms | 102 ms | 0.5× | 58× | 261 KiB / 96 KiB | 96 KiB / 32 KiB | 1002 KiB / 496 KiB |
| samples | 500 × 100 | 4.4 ms | 2.41 ms | 286 ms | 0.5× | 65× | 557 KiB / 464 KiB | 226 KiB / 0 KiB | 3.9 MiB / 4.3 MiB |
| samples | 1,000 × 100 | 12.2 ms | 7.32 ms | 851 ms | 0.6× | 70× | 1.0 MiB / 48 KiB | 443 KiB / 16 KiB | 12.7 MiB / 15.9 MiB |
| samples | 2,000 × 100 | 39.3 ms | 24.6 ms | 2.85 s | 0.6× | 72× | 2.0 MiB / 1.0 MiB | 876 KiB / 400 KiB | 47.3 MiB / 61.5 MiB |
| features | 100 × 100 | 1.33 ms | 0.682 ms | 57 ms | 0.5× | 43× | 162 KiB / 64 KiB | 52 KiB / 0 KiB | 423 KiB / 208 KiB |
| features | 100 × 1,000 | 3.26 ms | 1.22 ms | 394 ms | 0.4× | 121× | 1.4 MiB / 32 KiB | 444 KiB / 0 KiB | 2.8 MiB / 896 KiB |
| features | 100 × 5,000 | 12.6 ms | 3.59 ms | 2.01 s | 0.3× | 160× | 6.6 MiB / 2.4 MiB | 2.1 MiB / 3.9 MiB | 13.4 MiB / 3.0 MiB |
| features | 100 × 10,000 | 23.8 ms | 6.89 ms | 5.21 s | 0.3× | 219× | 13.3 MiB / 1.3 MiB | 4.3 MiB / 16 KiB | 26.7 MiB / 9.4 MiB |

## Summary

- Runtime, FastSelect MultiSURF / FastSelect: 0.3× to 0.6× (median 0.5×) across 9 cases.
- Traced host allocations, FastSelect MultiSURF / FastSelect: 0.3× to 0.4× (median 0.3×).
- Runtime, scikit-rebate MultiSURF / FastSelect: 42× to 219× (median 70×) across 9 cases.
- Traced host allocations, scikit-rebate MultiSURF / FastSelect: 2.0× to 23.8× (median 2.6×).
- Footprint growth at 2,000 × 100: FastSelect 1.0 MiB; FastSelect MultiSURF 400 KiB; scikit-rebate MultiSURF 61.5 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
