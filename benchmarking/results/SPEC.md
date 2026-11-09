# SPEC: FastSelect CPU against scikit-feature

Generated from [`spec_comparison.json`](spec_comparison.json) by `python benchmarking/benchmark_relief_variants.py --family spec --report`.

- FastSelect `b37a455` with uncommitted src changes, source SHA-256 `d1cbfeded87ec4d6…`; scikit-feature 48cffad (pinned tree), source SHA-256 `3d280cfb899e5716…` over the files used.
- Apple M1 Pro, 8 Numba threads, single-threaded BLAS; Python 3.13.17, NumPy 2.5.4, Numba 0.68.0, scikit-learn 1.9.1.
- Runtime: median over 3 alternating rounds of per-round medians; each round fits for at least 2 s per case (at least 6 s cumulative).
- Memory: peak macOS phys_footprint of a fresh worker process (both libraries warmed on the same 30-row slice), sampled by the parent with proc_pid_rusage about every 0.5 ms while the fit repeats for >= 0.5 s, minus the footprint just before. tracemalloc_peak_bytes is a separate fit and covers Python/NumPy host allocations only.
- `make_classification` data (seed 42, up to 20 informative features), `n_features_to_select=10`.
- `SPEC(10, criterion=...)` on the RBF graph (`gamma = 1 / n_features`; each fit builds it, scikit-feature through `rbf_kernel` passed as `W`) against the pinned scikit-feature `spec` (commit 48cffad, one NumPy-2 fix applied in memory: `D[i, i] = X_sum[i, 0]`; all mathematics unchanged) with `style=0` (phi2) or `style=11` (phi3 over 10 nontrivial eigenpairs). The scores agree; the selections differ because scikit-feature ranks phi2 descending and phi3 ascending, the opposite of the criteria's meaning. At 500 × 5,000 FastSelect's traced peak is one centred copy of `X`, which keeps the RBF distances free of cancellation; copies above 64 MiB are built in column blocks instead.

Memory cells give the `tracemalloc` peak and, after the slash, the growth of the process footprint. Footprint growth is page-granular and does not count allocations served from memory the process already holds, so small figures understate every implementation; the traced peak excludes Numba's per-thread buffers, a few times `8 * threads * (n + p)` bytes for FastSelect (under 1 MiB here).

One-minute load average around the runtime workers: 2.8 to 8.3.

## phi2

| Scenario | n × p | FastSelect | scikit-feature | scikit-feature / FastSelect | FastSelect memory | scikit-feature memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 200 × 100 | 0.397 ms | 8.54 ms | 22× | 698 KiB / 80 KiB | 2.9 MiB / 6.2 MiB |
| samples | 500 × 100 | 1.73 ms | 40.9 ms | 24× | 2.7 MiB / 2.3 MiB | 18.2 MiB / 37.9 MiB |
| samples | 1,000 × 100 | 6.55 ms | 206 ms | 31× | 9.2 MiB / 7.8 MiB | 72.6 MiB / 89.0 MiB |
| samples | 2,000 × 100 | 25.9 ms | 1.36 s | 52× | 33.7 MiB / 35.2 MiB | 290.2 MiB / 389.9 MiB |
| features | 500 × 100 | 1.73 ms | 41 ms | 24× | 2.7 MiB / 4.4 MiB | 18.2 MiB / 25.1 MiB |
| features | 500 × 1,000 | 6.79 ms | 199 ms | 29× | 6.0 MiB / 9.5 MiB | 18.2 MiB / 35.1 MiB |
| features | 500 × 5,000 | 28.7 ms | 890 ms | 31× | 21.1 MiB / 10.1 MiB | 18.2 MiB / 30.3 MiB |

Score agreement (phi2, one fit per library):

| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |
|---|---|---:|---:|---:|---:|
| samples | 200 × 100 | 1e-15 | 0.653 | 0 | no |
| samples | 500 × 100 | 1e-15 | 0.769 | 0 | no |
| samples | 1,000 × 100 | 1e-15 | 0.871 | 0 | no |
| samples | 2,000 × 100 | 1.1e-15 | 0.927 | 0 | no |
| features | 500 × 100 | 1e-15 | 0.769 | 0 | no |
| features | 500 × 1,000 | 1.1e-15 | 0.979 | 0 | no |
| features | 500 × 5,000 | 1.7e-15 | 0.986 | 0 | no |

## phi3 (m = 10)

| Scenario | n × p | FastSelect | scikit-feature | scikit-feature / FastSelect | FastSelect memory | scikit-feature memory |
|---|---|---:|---:|---:|---:|---:|
| samples | 200 × 100 | 2.41 ms | 8.4 ms | 3.5× | 1.4 MiB / 1.3 MiB | 2.9 MiB / 6.0 MiB |
| samples | 500 × 100 | 8.22 ms | 40.9 ms | 5.0× | 2.5 MiB / 2.8 MiB | 18.2 MiB / 40.8 MiB |
| samples | 1,000 × 100 | 20.7 ms | 207 ms | 10.0× | 8.6 MiB / 9.0 MiB | 72.6 MiB / 88.8 MiB |
| samples | 2,000 × 100 | 115 ms | 1.37 s | 12× | 32.5 MiB / 31.0 MiB | 290.2 MiB / 388.3 MiB |
| features | 500 × 100 | 7.96 ms | 40.9 ms | 5.1× | 2.5 MiB / 4.4 MiB | 18.2 MiB / 39.4 MiB |
| features | 500 × 1,000 | 7.88 ms | 197 ms | 25× | 5.8 MiB / 8.5 MiB | 18.2 MiB / 33.1 MiB |
| features | 500 × 5,000 | 19.4 ms | 883 ms | 45× | 21.1 MiB / 11.9 MiB | 18.2 MiB / 29.4 MiB |

Score agreement (phi3 (m = 10), one fit per library):

| Scenario | n × p | Largest score difference | Largest score | Shared top-10 | Same order |
|---|---|---:|---:|---:|---:|
| samples | 200 × 100 | 3.6e-15 | 0.99 | 0 | no |
| samples | 500 × 100 | 3.7e-15 | 0.248 | 0 | no |
| samples | 1,000 × 100 | 3.4e-15 | 0.751 | 0 | no |
| samples | 2,000 × 100 | 8e-15 | 1.54 | 0 | no |
| features | 500 × 100 | 3.7e-15 | 0.248 | 0 | no |
| features | 500 × 1,000 | 2.8e-14 | 1.12 | 0 | no |
| features | 500 × 5,000 | 6.8e-14 | 1.03 | 0 | no |

## Summary

- Runtime, scikit-feature / FastSelect: 3.5× to 52× (median 24×) across 14 cases.
- Traced host allocations, scikit-feature / FastSelect: 0.9× to 8.9× (median 6.6×).
- Footprint growth at 2,000 × 100: FastSelect 31.0 MiB to 35.2 MiB; scikit-feature 388.3 MiB to 389.9 MiB. FastSelect's footprint growth is often within memory the process already holds, so no ratio is given.
- The load averages include the benchmark's own workers (eight Numba threads).
