# ReliefSeq: FastSelect CPU against FastSelect ReliefF, each k, scikit-rebate ReliefF, each k

Generated from [`reliefseq_comparison.json`](reliefseq_comparison.json) by `python benchmarking/benchmark_relief_variants.py --family reliefseq --report`.

- FastSelect `9cfd044` with uncommitted src changes, source SHA-256 `fceea7810d73393c…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- `ReliefSeq(max_neighbors=10)` against what ReliefSeq replaces: a separate ReliefF fit for every k = 1..10, keeping each feature's best score, with FastSelect's and with scikit-rebate's ReliefF. scikit-rebate has no ReliefSeq, and the authors' C++ tool was not run.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 4.3 to 11.4.

## ReliefSeq k <= 10

| Scenario | n × p | FastSelect | FastSelect ReliefF, each k | scikit-rebate ReliefF, each k | FastSelect ReliefF, each k / FastSelect | scikit-rebate ReliefF, each k / FastSelect | FastSelect memory | FastSelect ReliefF, each k memory | scikit-rebate ReliefF, each k memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| samples | 100 × 100 | 0.836 ms | 12.8 ms | 582 ms | 15× | 696× | 181 KiB / 32 KiB | 75 KiB / 2.3 MiB | 458 KiB / 64 KiB |
| samples | 200 × 100 | 1.17 ms | 13.5 ms | 967 ms | 12× | 828× | 348 KiB / 112 KiB | 136 KiB / 2.4 MiB | 952 KiB / 176 KiB |
| samples | 500 × 100 | 2.67 ms | 18 ms | 2.00 s | 6.7× | 750× | 849 KiB / 608 KiB | 316 KiB / 5.4 MiB | 3.4 MiB / 5.8 MiB |
| samples | 1,000 × 100 | 6.82 ms | 38.9 ms | 5.21 s | 5.7× | 763× | 1.6 MiB / 1.1 MiB | 598 KiB / 11.8 MiB | 12.3 MiB / 13.7 MiB |
| features | 100 × 100 | 0.826 ms | 12.7 ms | 616 ms | 15× | 746× | 181 KiB / 128 KiB | 74 KiB / 2.2 MiB | 458 KiB / 176 KiB |
| features | 100 × 1,000 | 2.1 ms | 16.3 ms | 4.11 s | 7.7× | 1954× | 1.6 MiB / 1.7 MiB | 547 KiB / 2.5 MiB | 2.9 MiB / 1.3 MiB |
| features | 100 × 5,000 | 7.84 ms | 32.8 ms | 21.49 s | 4.2× | 2741× | 8.1 MiB / 768 KiB | 2.6 MiB / 8.7 MiB | 13.5 MiB / 368 KiB |

## Summary

- Runtime, FastSelect ReliefF, each k / FastSelect: 4.2× to 15× (median 7.7×) across 7 cases.
- Traced host allocations, FastSelect ReliefF, each k / FastSelect: 0.3× to 0.4× (median 0.4×).
- Runtime, scikit-rebate ReliefF, each k / FastSelect: 696× to 2741× (median 763×) across 7 cases.
- Traced host allocations, scikit-rebate ReliefF, each k / FastSelect: 1.7× to 7.5× (median 2.5×).
- Footprint growth at 1,000 × 100: FastSelect 1.1 MiB; FastSelect ReliefF, each k 11.8 MiB; scikit-rebate ReliefF, each k 13.7 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
