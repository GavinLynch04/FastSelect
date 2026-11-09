# HSIC Lasso (vanilla, one block): FastSelect CPU against pyHSICLasso

Generated from [`hsic_lasso_vanilla_comparison.json`](hsic_lasso_vanilla_comparison.json) by `python benchmarking/benchmark_relief_variants.py --family hsic_vanilla --report`.

- FastSelect `b37a455` with uncommitted src changes, source SHA-256 `ff29f2e2134223dd…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`; pyHSICLasso 1.4.2, source SHA-256 `d4fa6125e1d237c8…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- `HSICLasso(10, block_size=0)` against pyHSICLasso's `classification(num_feat=10, B=0)`. pyHSICLasso stores every feature's n x n Gram matrix, so shapes stay small.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 3.5 to 9.7.

## vanilla

| Scenario | n × p | FastSelect | pyHSICLasso | pyHSICLasso / FastSelect | FastSelect memory | pyHSICLasso memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 100 × 200 | 27.2 ms | 40.4 ms | 1.5× | 365 KiB / 160 KiB | 24.6 MiB / 38.2 MiB |
| samples | 200 × 200 | 93.5 ms | 119 ms | 1.3× | 999 KiB / 992 KiB | 97.9 MiB / 173.0 MiB |
| samples | 400 × 200 | 336 ms | 503 ms | 1.5× | 3.1 MiB / 2.6 MiB | 391.4 MiB / 631.5 MiB |
| features | 200 × 200 | 92.6 ms | 127 ms | 1.4× | 1001 KiB / 1.5 MiB | 97.9 MiB / 173.0 MiB |
| features | 200 × 1,000 | 400 ms | 525 ms | 1.3× | 3.1 MiB / 704 KiB | 464.6 MiB / 630.1 MiB |
| features | 200 × 2,000 | 757 ms | 1.18 s | 1.6× | 6.1 MiB / 256 KiB | 922.8 MiB / 1228.5 MiB |

Score agreement (vanilla, one fit per library):

| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |
|---|---|---:|---:|---:|---:|
| samples | 100 × 200 | 2.4e-08 | 0.0795 | 10 | yes |
| samples | 200 × 200 | 2.9e-08 | 0.0542 | 10 | yes |
| samples | 400 × 200 | 3.5e-08 | 0.0907 | 10 | yes |
| features | 200 × 200 | 2.9e-08 | 0.0542 | 10 | yes |
| features | 200 × 1,000 | 3.4e-08 | 0.0378 | 10 | yes |
| features | 200 × 2,000 | 3.1e-08 | 0.0296 | 10 | yes |

## Summary

- Runtime, pyHSICLasso / FastSelect: 1.3× to 1.6× (median 1.4×) across 6 cases.
- Traced host allocations, pyHSICLasso / FastSelect: 68.9× to 151.4× (median 112.7×).
- Footprint growth at 400 × 200: FastSelect 2.6 MiB; pyHSICLasso 631.5 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
