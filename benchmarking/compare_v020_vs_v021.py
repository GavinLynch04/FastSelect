import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
from numba import njit, prange

from fast_select.MultiSURF import MultiSURF
from fast_select.ReliefF import ReliefF
from fast_select.SURF import SURF
from fast_select.utils import is_cuda_ready

warnings.filterwarnings("ignore")


# --- Legacy v0.2.0 Unoptimized CPU Kernel ---
@njit(parallel=True)
def _relieff_cpu_kernel_v020(x, y, recip_full, is_discrete, class_probs, k, scores_out):
    n_samples, n_features = x.shape
    n_classes = len(class_probs)

    for i in prange(n_samples):
        # UNOPTIMIZED (v0.2.0): Heap allocation inside parallel loop
        dists_from_i = np.empty(n_samples, dtype=np.float32)

        for j in range(n_samples):
            if i == j:
                dists_from_i[j] = 0.0
                continue
            dist_ij = 0.0
            for f in range(n_features):
                if is_discrete[f]:
                    diff = 1.0 if x[i, f] != x[j, f] else 0.0
                else:
                    diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                dist_ij += diff
            dists_from_i[j] = dist_ij

        for c in range(n_classes):
            c_indices = np.where(y == c)[0]
            c_dists = dists_from_i[c_indices]
            sorted_idx = np.argsort(c_dists)

            n_select = k if c != y[i] else k + 1
            selected = sorted_idx[:n_select]

            for idx in selected:
                j = c_indices[idx]
                if i == j:
                    continue
                weight = 1.0 / (n_samples * k) if c != y[i] else -1.0 / (n_samples * k)
                for f in range(n_features):
                    if is_discrete[f]:
                        diff = 1.0 if x[i, f] != x[j, f] else 0.0
                    else:
                        diff = abs(x[i, f] - x[j, f]) * recip_full[f]
                    scores_out[f] += weight * diff


def run_benchmark_v020(X, y, k=10):
    n_samples, n_features = X.shape
    classes, y_encoded = np.unique(y, return_inverse=True)
    class_probs = np.bincount(y_encoded) / n_samples

    is_discrete = np.zeros(n_features, dtype=bool)
    recip_full = np.ones(n_features, dtype=np.float32)
    scores = np.zeros(n_features, dtype=np.float32)

    tracemalloc.start()
    t0 = time.perf_counter()
    _relieff_cpu_kernel_v020(
        X.astype(np.float32),
        y_encoded.astype(np.int32),
        recip_full,
        is_discrete,
        class_probs.astype(np.float32),
        k,
        scores,
    )
    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return t1 - t0, peak_bytes / (1024 * 1024)


def run_benchmark_v021(estimator_class, X, y, backend="cpu"):
    if backend == "gpu":
        from fast_select.utils import ensure_cuda_context

        ensure_cuda_context()
        t0 = time.perf_counter()
        estimator_class(n_features_to_select=10, backend=backend).fit(X, y)
        t1 = time.perf_counter()
        peak_ram_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
    else:
        tracemalloc.start()
        t0 = time.perf_counter()
        estimator_class(n_features_to_select=10, backend=backend).fit(X, y)
        t1 = time.perf_counter()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak_bytes / (1024 * 1024)

    return t1 - t0, peak_ram_mb


def main():
    print("=========================================================================")
    print("           FastSelect v0.2.0 vs v0.2.1 Performance Comparison           ")
    print("=========================================================================")

    n_samples = 3000
    n_features = 800
    print(f"\nGenerating Synthetic Dataset: {n_samples} samples x {n_features} features...")

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)

    print("\nWarming up JIT compilers...")
    ReliefF(n_features_to_select=10, backend="cpu").fit(X[:50, :50], y[:50])
    if is_cuda_ready():
        ReliefF(n_features_to_select=10, backend="gpu").fit(X[:50, :50], y[:50])

    print("\n--- Running Benchmarks (Runtimes > 1.0s) ---")

    # 1. v0.2.0 Unoptimized CPU
    print("\n1. Running v0.2.0 Unoptimized ReliefF (CPU)...")
    time_v020, ram_v020 = run_benchmark_v020(X, y)
    print(f"   > v0.2.0 ReliefF (CPU) Runtime : {time_v020:.4f} seconds | Peak RAM: {ram_v020:.2f} MB")

    # 2. v0.2.1 CPU
    print("\n2. Running v0.2.1 Optimized ReliefF (CPU)...")
    time_v021_cpu, ram_v021_cpu = run_benchmark_v021(ReliefF, X, y, backend="cpu")
    print(f"   > v0.2.1 ReliefF (CPU) Runtime : {time_v021_cpu:.4f} seconds | Peak RAM: {ram_v021_cpu:.2f} MB")

    # 3. v0.2.1 GPU
    if is_cuda_ready():
        print("\n3. Running v0.2.1 Optimized ReliefF (GPU)...")
        time_v021_gpu, ram_v021_gpu = run_benchmark_v021(ReliefF, X, y, backend="gpu")
        print(f"   > v0.2.1 ReliefF (GPU) Runtime : {time_v021_gpu:.4f} seconds | Peak RAM: {ram_v021_gpu:.2f} MB")

    # 4. v0.2.1 SURF CPU & GPU
    print("\n4. Running v0.2.1 Optimized SURF (CPU)...")
    time_surf_cpu, ram_surf_cpu = run_benchmark_v021(SURF, X, y, backend="cpu")
    print(f"   > v0.2.1 SURF (CPU) Runtime    : {time_surf_cpu:.4f} seconds | Peak RAM: {ram_surf_cpu:.2f} MB")

    if is_cuda_ready():
        print("\n5. Running v0.2.1 Optimized SURF (GPU)...")
        time_surf_gpu, ram_surf_gpu = run_benchmark_v021(SURF, X, y, backend="gpu")
        print(f"   > v0.2.1 SURF (GPU) Runtime    : {time_surf_gpu:.4f} seconds | Peak RAM: {ram_surf_gpu:.2f} MB")

    # 6. v0.2.1 MultiSURF CPU & GPU
    print("\n6. Running v0.2.1 Optimized MultiSURF (CPU)...")
    time_msurf_cpu, ram_msurf_cpu = run_benchmark_v021(MultiSURF, X, y, backend="cpu")
    print(f"   > v0.2.1 MultiSURF (CPU) Runtime : {time_msurf_cpu:.4f} seconds | Peak RAM: {ram_msurf_cpu:.2f} MB")

    if is_cuda_ready():
        print("\n7. Running v0.2.1 Optimized MultiSURF (GPU)...")
        time_msurf_gpu, ram_msurf_gpu = run_benchmark_v021(MultiSURF, X, y, backend="gpu")
        print(f"   > v0.2.1 MultiSURF (GPU) Runtime : {time_msurf_gpu:.4f} seconds | Peak RAM: {ram_msurf_gpu:.2f} MB")

    results = [
        {
            "Version": "v0.2.0 (Unoptimized)",
            "Backend": "CPU",
            "Estimator": "ReliefF",
            "Dataset": f"{n_samples}x{n_features}",
            "Runtime (s)": round(time_v020, 4),
            "Peak RAM (MB)": round(ram_v020, 2),
            "Speedup vs v0.2.0": "1.00x (Baseline)",
        },
        {
            "Version": "v0.2.1 (Optimized)",
            "Backend": "CPU",
            "Estimator": "ReliefF",
            "Dataset": f"{n_samples}x{n_features}",
            "Runtime (s)": round(time_v021_cpu, 4),
            "Peak RAM (MB)": round(ram_v021_cpu, 2),
            "Speedup vs v0.2.0": f"{(time_v020 / time_v021_cpu):.2f}x",
        },
    ]

    if is_cuda_ready():
        results.append(
            {
                "Version": "v0.2.1 (Optimized)",
                "Backend": "GPU",
                "Estimator": "ReliefF",
                "Dataset": f"{n_samples}x{n_features}",
                "Runtime (s)": round(time_v021_gpu, 4),
                "Peak RAM (MB)": round(ram_v021_gpu, 2),
                "Speedup vs v0.2.0": f"{(time_v020 / time_v021_gpu):.2f}x",
            }
        )

    results.append(
        {
            "Version": "v0.2.1 (Optimized)",
            "Backend": "CPU",
            "Estimator": "SURF",
            "Dataset": f"{n_samples}x{n_features}",
            "Runtime (s)": round(time_surf_cpu, 4),
            "Peak RAM (MB)": round(ram_surf_cpu, 2),
            "Speedup vs v0.2.0": f"{(time_v020 / time_surf_cpu):.2f}x",
        }
    )

    if is_cuda_ready():
        results.append(
            {
                "Version": "v0.2.1 (Optimized)",
                "Backend": "GPU",
                "Estimator": "SURF",
                "Dataset": f"{n_samples}x{n_features}",
                "Runtime (s)": round(time_surf_gpu, 4),
                "Peak RAM (MB)": round(ram_surf_gpu, 2),
                "Speedup vs v0.2.0": f"{(time_v020 / time_surf_gpu):.2f}x",
            }
        )

    results.append(
        {
            "Version": "v0.2.1 (Optimized)",
            "Backend": "CPU",
            "Estimator": "MultiSURF",
            "Dataset": f"{n_samples}x{n_features}",
            "Runtime (s)": round(time_msurf_cpu, 4),
            "Peak RAM (MB)": round(ram_msurf_cpu, 2),
            "Speedup vs v0.2.0": f"{(time_v020 / time_msurf_cpu):.2f}x",
        }
    )

    if is_cuda_ready():
        results.append(
            {
                "Version": "v0.2.1 (Optimized)",
                "Backend": "GPU",
                "Estimator": "MultiSURF",
                "Dataset": f"{n_samples}x{n_features}",
                "Runtime (s)": round(time_msurf_gpu, 4),
                "Peak RAM (MB)": round(ram_msurf_gpu, 2),
                "Speedup vs v0.2.0": f"{(time_v020 / time_msurf_gpu):.2f}x",
            }
        )

    df_summary = pd.DataFrame(results)

    print("\n=========================================================================")
    print("                         SUMMARY COMPARISON TABLE                        ")
    print("=========================================================================")
    print(df_summary.to_string(index=False))
    print("=========================================================================")


if __name__ == "__main__":
    main()
