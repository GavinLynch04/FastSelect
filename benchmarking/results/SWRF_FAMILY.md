# SWRF family: FastSelect CPU against scikit-rebate

Generated from [`swrf_vs_skrebate.json`](swrf_vs_skrebate.json) by `python benchmarking/benchmark_relief_variants.py --family swrf --report`.

- FastSelect `9cfd044` with uncommitted src changes, source SHA-256 `8665a1f05c5a2e8d…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- The libraries compute different scores by design (see `src/fast_select/SWRF.py`); this compares speed and memory, not output.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 3.2 to 15.6.

## SWRF

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 1.09 ms | 53.9 ms | 50× | 104 KiB / 32 KiB | 568 KiB / 368 KiB |
| samples | 200 × 100 | 1.52 ms | 88.8 ms | 58× | 196 KiB / 80 KiB | 1.5 MiB / 2.2 MiB |
| samples | 500 × 100 | 4.13 ms | 254 ms | 61× | 472 KiB / 96 KiB | 7.2 MiB / 11.7 MiB |
| samples | 1,000 × 100 | 12.7 ms | 725 ms | 57× | 931 KiB / 560 KiB | 25.8 MiB / 45.6 MiB |
| samples | 2,000 × 100 | 42.5 ms | 2.38 s | 56× | 1.8 MiB / 1.6 MiB | 97.6 MiB / 177.5 MiB |
| features | 100 × 100 | 1.08 ms | 53.9 ms | 50× | 104 KiB / 32 KiB | 568 KiB / 416 KiB |
| features | 100 × 1,000 | 2.55 ms | 389 ms | 153× | 876 KiB / 608 KiB | 2.9 MiB / 1.2 MiB |
| features | 100 × 5,000 | 9.06 ms | 1.85 s | 204× | 4.2 MiB / 112 KiB | 13.5 MiB / 1.8 MiB |
| features | 100 × 10,000 | 19.4 ms | 4.67 s | 240× | 8.4 MiB / 80 KiB | 26.8 MiB / 6.8 MiB |

## SWRF*

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 1.11 ms | 77.5 ms | 70× | 104 KiB / 176 KiB | 674 KiB / 464 KiB |
| samples | 200 × 100 | 1.64 ms | 143 ms | 87× | 196 KiB / 64 KiB | 1.9 MiB / 2.2 MiB |
| samples | 500 × 100 | 4.74 ms | 427 ms | 90× | 472 KiB / 416 KiB | 9.2 MiB / 13.9 MiB |
| samples | 1,000 × 100 | 15.2 ms | 1.14 s | 75× | 931 KiB / 576 KiB | 33.6 MiB / 54.6 MiB |
| samples | 2,000 × 100 | 52 ms | 3.68 s | 71× | 1.8 MiB / 1.5 MiB | 128.2 MiB / 212.0 MiB |
| features | 100 × 100 | 1.11 ms | 70 ms | 63× | 104 KiB / 96 KiB | 674 KiB / 448 KiB |
| features | 100 × 1,000 | 2.81 ms | 562 ms | 200× | 876 KiB / 560 KiB | 3.1 MiB / 1.3 MiB |
| features | 100 × 5,000 | 10.1 ms | 3.26 s | 322× | 4.2 MiB / 112 KiB | 13.7 MiB / 2.3 MiB |
| features | 100 × 10,000 | 22 ms | 6.23 s | 283× | 8.4 MiB / 656 KiB | 26.9 MiB / 7.2 MiB |

## MultiSWRF

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 0.639 ms | 56.8 ms | 89× | 104 KiB / 0 KiB | 576 KiB / 304 KiB |
| samples | 200 × 100 | 0.96 ms | 103 ms | 107× | 196 KiB / 16 KiB | 1.5 MiB / 2.1 MiB |
| samples | 500 × 100 | 2.99 ms | 309 ms | 103× | 472 KiB / 32 KiB | 7.2 MiB / 10.3 MiB |
| samples | 1,000 × 100 | 10.1 ms | 943 ms | 94× | 931 KiB / 64 KiB | 25.9 MiB / 45.2 MiB |
| samples | 2,000 × 100 | 34.6 ms | 3.16 s | 91× | 1.8 MiB / 112 KiB | 97.5 MiB / 176.4 MiB |
| features | 100 × 100 | 0.637 ms | 57.2 ms | 90× | 104 KiB / 0 KiB | 576 KiB / 464 KiB |
| features | 100 × 1,000 | 1.34 ms | 383 ms | 286× | 876 KiB / 0 KiB | 3.0 MiB / 912 KiB |
| features | 100 × 5,000 | 4.47 ms | 1.88 s | 421× | 4.2 MiB / 80 KiB | 13.6 MiB / 2.4 MiB |
| features | 100 × 10,000 | 9.74 ms | 4.64 s | 476× | 8.4 MiB / 176 KiB | 26.8 MiB / 7.1 MiB |

## MultiSWRF*

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 0.664 ms | 78.9 ms | 119× | 104 KiB / 32 KiB | 674 KiB / 496 KiB |
| samples | 200 × 100 | 1.08 ms | 154 ms | 143× | 196 KiB / 288 KiB | 1.9 MiB / 2.1 MiB |
| samples | 500 × 100 | 3.74 ms | 499 ms | 134× | 472 KiB / 48 KiB | 9.2 MiB / 12.5 MiB |
| samples | 1,000 × 100 | 12.8 ms | 1.33 s | 104× | 930 KiB / 656 KiB | 33.6 MiB / 52.6 MiB |
| samples | 2,000 × 100 | 42.9 ms | 4.50 s | 105× | 1.8 MiB / 16 KiB | 128.2 MiB / 207.7 MiB |
| features | 100 × 100 | 0.661 ms | 78.5 ms | 119× | 104 KiB / 48 KiB | 674 KiB / 480 KiB |
| features | 100 × 1,000 | 1.57 ms | 576 ms | 367× | 876 KiB / 176 KiB | 3.1 MiB / 2.0 MiB |
| features | 100 × 5,000 | 5.69 ms | 3.34 s | 587× | 4.2 MiB / 32 KiB | 13.7 MiB / 1.2 MiB |
| features | 100 × 10,000 | 13.7 ms | 6.58 s | 481× | 8.4 MiB / 0 KiB | 26.9 MiB / 7.1 MiB |

## MultiSWRFDB

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 0.62 ms | 56.8 ms | 92× | 104 KiB / 32 KiB | 547 KiB / 448 KiB |
| samples | 200 × 100 | 0.863 ms | 100 ms | 116× | 196 KiB / 32 KiB | 1.4 MiB / 1.6 MiB |
| samples | 500 × 100 | 2.5 ms | 292 ms | 117× | 472 KiB / 96 KiB | 6.5 MiB / 10.8 MiB |
| samples | 1,000 × 100 | 8.1 ms | 847 ms | 105× | 930 KiB / 688 KiB | 23.9 MiB / 41.5 MiB |
| samples | 2,000 × 100 | 28.3 ms | 2.87 s | 101× | 1.8 MiB / 48 KiB | 93.5 MiB / 163.5 MiB |
| features | 100 × 100 | 0.623 ms | 57.3 ms | 92× | 104 KiB / 32 KiB | 547 KiB / 336 KiB |
| features | 100 × 1,000 | 1.22 ms | 370 ms | 303× | 876 KiB / 816 KiB | 2.9 MiB / 816 KiB |
| features | 100 × 5,000 | 4.04 ms | 1.86 s | 459× | 4.2 MiB / 96 KiB | 13.5 MiB / 1.9 MiB |
| features | 100 × 10,000 | 8.78 ms | 4.64 s | 529× | 8.4 MiB / 0 KiB | 26.8 MiB / 7.0 MiB |

## MultiSWRFDB*

| Scenario | n × p | FastSelect | scikit-rebate | scikit-rebate / FastSelect | FastSelect memory | scikit-rebate memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 0.649 ms | 72.8 ms | 112× | 104 KiB / 32 KiB | 617 KiB / 272 KiB |
| samples | 200 × 100 | 1 ms | 150 ms | 150× | 196 KiB / 48 KiB | 1.6 MiB / 1.4 MiB |
| samples | 500 × 100 | 3.28 ms | 434 ms | 132× | 472 KiB / 32 KiB | 7.7 MiB / 11.2 MiB |
| samples | 1,000 × 100 | 11.3 ms | 1.19 s | 106× | 931 KiB / 272 KiB | 27.8 MiB / 47.9 MiB |
| samples | 2,000 × 100 | 39.8 ms | 3.87 s | 97× | 1.8 MiB / 1.3 MiB | 104.8 MiB / 182.5 MiB |
| features | 100 × 100 | 0.647 ms | 72 ms | 111× | 104 KiB / 32 KiB | 617 KiB / 464 KiB |
| features | 100 × 1,000 | 1.41 ms | 575 ms | 409× | 876 KiB / 16 KiB | 3.0 MiB / 864 KiB |
| features | 100 × 5,000 | 4.83 ms | 3.37 s | 698× | 4.2 MiB / 80 KiB | 13.6 MiB / 2.2 MiB |
| features | 100 × 10,000 | 11 ms | 6.44 s | 587× | 8.4 MiB / 352 KiB | 26.9 MiB / 7.0 MiB |

## Summary

- Runtime, scikit-rebate / FastSelect: 50× to 698× (median 112×) across 54 cases.
- Traced host allocations, scikit-rebate / FastSelect: 3.2× to 71.0× (median 6.5×).
- Footprint growth at 2,000 × 100: FastSelect 16 KiB to 1.6 MiB; scikit-rebate 163.5 MiB to 212.0 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
