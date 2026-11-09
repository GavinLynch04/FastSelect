# HSIC Lasso (block, B = 20, M = 3): FastSelect CPU against pyHSICLasso

Generated from [`hsic_lasso_comparison.json`](hsic_lasso_comparison.json) by `python benchmarking/benchmark_relief_variants.py --family hsic --report`.

- FastSelect `b37a455` with uncommitted src changes, source SHA-256 `ff29f2e2134223dd…`; scikit-rebate 0.8.4, source SHA-256 `dbefb308b91a7ee7…`; pyHSICLasso 1.4.2, source SHA-256 `d4fa6125e1d237c8…`.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative). External libraries get the faster of `n_jobs=-1` and `n_jobs=1`.
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before; scikit-rebate with n_jobs=1 so all work is in-process. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- `HSICLasso(10)` (block size 20, 3 permutations, normalised Gram matrices) against pyHSICLasso's `classification(num_feat=10)` with the same defaults. pyHSICLasso computes its kernels in float32, so coefficients agree to roughly single precision; the parity table records it.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 3.2 to 7.9.

## block

| Scenario | n × p | FastSelect | pyHSICLasso | pyHSICLasso / FastSelect | FastSelect memory | pyHSICLasso memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 500 × 1,000 | 374 ms | 630 ms | 1.7× | 7.7 MiB / 1.8 MiB | 348.6 MiB / 471.0 MiB |
| samples | 1,000 × 1,000 | 707 ms | 1.23 s | 1.7× | 15.3 MiB / 2.8 MiB | 696.6 MiB / 953.5 MiB |
| samples | 2,000 × 1,000 | 1.37 s | 2.96 s | 2.2× | 30.6 MiB / 6.2 MiB | 1392.6 MiB / 1889.8 MiB |
| features | 500 × 1,000 | 376 ms | 678 ms | 1.8× | 7.7 MiB / 1.5 MiB | 348.6 MiB / 471.3 MiB |
| features | 500 × 2,000 | 694 ms | 1.38 s | 2.0× | 15.3 MiB / 2.4 MiB | 692.4 MiB / 936.9 MiB |
| features | 500 × 5,000 | 1.68 s | 3.88 s | 2.3× | 38.2 MiB / 6.1 MiB | 1723.2 MiB / 1855.5 MiB |

Score agreement (block, one fit per library):

| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |
|---|---|---:|---:|---:|---:|
| samples | 500 × 1,000 | 3.8e-08 | 0.0987 | 10 | yes |
| samples | 1,000 × 1,000 | 8.9e-08 | 0.111 | 10 | yes |
| samples | 2,000 × 1,000 | 1.7e-07 | 0.0532 | 10 | yes |
| features | 500 × 1,000 | 3.8e-08 | 0.0987 | 10 | yes |
| features | 500 × 2,000 | 3.8e-08 | 0.111 | 10 | yes |
| features | 500 × 5,000 | 4.3e-08 | 0.105 | 10 | yes |

## Summary

- Runtime, pyHSICLasso / FastSelect: 1.7× to 2.3× (median 1.9×) across 6 cases.
- Traced host allocations, pyHSICLasso / FastSelect: 45.1× to 45.5× (median 45.5×).
- Footprint growth at 2,000 × 1,000: FastSelect 6.2 MiB; pyHSICLasso 1889.8 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads, or eight scikit-rebate processes with `n_jobs=-1`).
