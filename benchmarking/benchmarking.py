import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification

from fast_select.MultiSURF import MultiSURF as FastMultiSURF
from fast_select.ReliefF import ReliefF as FastReliefF
from fast_select.SURF import SURF as FastSURF
from fast_select.utils import is_cuda_ready

# --- Benchmark Configuration ---
P_DOMINANT_SCENARIOS = {"n_samples": 100, "n_features_range": [200, 500, 1000]}
N_DOMINANT_SCENARIOS = {"n_features": 100, "n_samples_range": [200, 500, 1000]}
N_FEATURES_TO_SELECT = 10
N_REPEATS = 1

GPU_AVAILABLE = is_cuda_ready()

# --- Estimators to Test ---
estimators = {
    # fast-select CPU estimators
    "fast_select.ReliefF (CPU)": lambda: FastReliefF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
    "fast_select.SURF (CPU)": lambda: FastSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
    "fast_select.SURF* (CPU)": lambda: FastSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu", use_star=True),
    "fast_select.MultiSURF (CPU)": lambda: FastMultiSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
    "fast_select.MultiSURF* (CPU)": lambda: FastMultiSURF(
        n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu", use_star=True
    ),
}

if GPU_AVAILABLE:
    print("NVIDIA GPU detected. Including GPU benchmarks.")
    estimators.update(
        {
            "fast_select.ReliefF (GPU)": lambda: FastReliefF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast_select.SURF (GPU)": lambda: FastSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast_select.SURF* (GPU)": lambda: FastSURF(
                n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu", use_star=True
            ),
            "fast_select.MultiSURF (GPU)": lambda: FastMultiSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast_select.MultiSURF* (GPU)": lambda: FastMultiSURF(
                n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu", use_star=True
            ),
        }
    )
else:
    print("No NVIDIA GPU detected. Skipping GPU benchmarks.")


def run_single_benchmark(estimator, X, y):
    """Measures execution time and peak memory overhead of a single estimator fit."""
    import gc
    from fast_select.utils import ensure_cuda_context
    gc.collect()
    is_gpu = getattr(estimator, 'backend', None) == 'gpu'
    if is_gpu:
        ensure_cuda_context()

    if not is_gpu:
        tracemalloc.start()

    start_time = time.perf_counter()
    estimator.fit(X, y)
    end_time = time.perf_counter()

    if not is_gpu:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak_bytes / (1024 * 1024)
    else:
        peak_ram_mb = (X.nbytes + y.nbytes) / (1024 * 1024)

    runtime = end_time - start_time
    return runtime, peak_ram_mb


def warmup_jit_compilers(estimators_dict):
    """Performs a warm-up run to trigger Numba JIT compilation."""
    print("\n--- Warming up JIT compilers ---")
    X_warmup, y_warmup = make_classification(n_samples=20, n_features=10, random_state=42)

    for name, estimator_fn in estimators_dict.items():
        print(f"  Warming up {name}...")
        try:
            estimator_fn().fit(X_warmup, y_warmup)
        except Exception as e:
            warnings.warn(f"  > Warm-up FAILED for {name}. Reason: {e}")
    print("--- Warm-up complete ---")


def main():
    """Main benchmark execution."""
    results = []
    warmup_jit_compilers(estimators)

    # --- Run p >> n scenario ---
    print("\n--- Running Scenario: p >> n (Many Features) ---")
    n_samples = P_DOMINANT_SCENARIOS["n_samples"]
    for n_features in P_DOMINANT_SCENARIOS["n_features_range"]:
        print(f"\nGenerating data: {n_samples} samples, {n_features} features")
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=min(20, n_features), random_state=42
        )

        for name, estimator_fn in estimators.items():
            for i in range(N_REPEATS):
                print(f"  Benchmarking {name} (Run {i+1}/{N_REPEATS})...")
                try:
                    runtime, peak_ram = run_single_benchmark(estimator_fn(), X, y)
                    results.append(
                        {
                            "scenario": "p >> n",
                            "algorithm": name,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "runtime": runtime,
                            "peak_ram_mb": peak_ram,
                        }
                    )
                except Exception as e:
                    warnings.warn(f"  > FAILED: {name} on {n_samples}x{n_features}. Reason: {e}")

    # --- Run n >> p scenario ---
    print("\n--- Running Scenario: n >> p (Many Samples) ---")
    n_features = N_DOMINANT_SCENARIOS["n_features"]
    for n_samples in N_DOMINANT_SCENARIOS["n_samples_range"]:
        print(f"\nGenerating data: {n_samples} samples, {n_features} features")
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=20, random_state=42
        )

        for name, estimator_fn in estimators.items():
            for i in range(N_REPEATS):
                print(f"  Benchmarking {name} (Run {i+1}/{N_REPEATS})...")
                try:
                    runtime, peak_ram = run_single_benchmark(estimator_fn(), X, y)
                    results.append(
                        {
                            "scenario": "n >> p",
                            "algorithm": name,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "runtime": runtime,
                            "peak_ram_mb": peak_ram,
                        }
                    )
                except Exception as e:
                    warnings.warn(f"  > FAILED: {name} on {n_samples}x{n_features}. Reason: {e}")

    df = pd.DataFrame(results)
    output_file = "benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nBenchmarking complete. Results saved to '{output_file}'")
    print(df.to_string())


if __name__ == "__main__":
    main()
