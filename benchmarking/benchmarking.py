import gc
import time
import tracemalloc
import warnings

import pandas as pd
from sklearn.datasets import make_classification

from fast_select.MultiSURF import MultiSURF as FastMultiSURF
from fast_select.ReliefF import ReliefF as FastReliefF
from fast_select.SURF import SURF as FastSURF
from fast_select.utils import ensure_cuda_context, is_cuda_ready

try:
    from skrebate import MultiSURF as SkMultiSURF
    from skrebate import ReliefF as SkReliefF

    SKREBATE_AVAILABLE = True
except ImportError:
    SKREBATE_AVAILABLE = False

warnings.filterwarnings("ignore")

# Benchmark Configuration
P_DOMINANT_SCENARIOS = {"n_samples": 100, "n_features_range": [100, 200, 500, 1000, 2000]}
N_DOMINANT_SCENARIOS = {"n_features": 100, "n_samples_range": [100, 200, 500, 1000, 2000]}
N_FEATURES_TO_SELECT = 10
N_REPEATS = 1

GPU_AVAILABLE = is_cuda_ready()

# Estimator Factories
estimators = {
    "fast-select.ReliefF (CPU)": lambda: FastReliefF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
    "fast-select.SURF (CPU)": lambda: FastSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
    "fast-select.MultiSURF (CPU)": lambda: FastMultiSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
}

if GPU_AVAILABLE:
    print("NVIDIA GPU detected. Including GPU benchmarks.", flush=True)
    estimators.update(
        {
            "fast-select.ReliefF (GPU)": lambda: FastReliefF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast-select.SURF (GPU)": lambda: FastSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast-select.MultiSURF (GPU)": lambda: FastMultiSURF(
                n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"
            ),
        }
    )

if SKREBATE_AVAILABLE:
    print("scikit-rebate detected. Including baseline CPU benchmarks.", flush=True)
    estimators.update(
        {
            "scikit-rebate.ReliefF": lambda: SkReliefF(n_features_to_select=N_FEATURES_TO_SELECT, n_jobs=1),
            "scikit-rebate.MultiSURF": lambda: SkMultiSURF(n_features_to_select=N_FEATURES_TO_SELECT, n_jobs=1),
        }
    )


def run_single_benchmark(estimator, X, y):
    """Measures execution time and peak RAM overhead of a single estimator fit."""
    gc.collect()
    is_gpu = getattr(estimator, "backend", None) == "gpu"
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
    """Performs a warm-up run to trigger JIT compilation and memory allocations."""
    print("\n--- Warming up JIT compilers & baselines ---", flush=True)
    X_warmup, y_warmup = make_classification(n_samples=20, n_features=10, random_state=42)

    for name, estimator_fn in estimators_dict.items():
        print(f"  Warming up {name}...", flush=True)
        try:
            estimator_fn().fit(X_warmup, y_warmup)
        except Exception as e:
            warnings.warn(f"  > Warm-up FAILED for {name}. Reason: {e}")
    print("--- Warm-up complete ---", flush=True)


def main():
    """Main benchmark execution."""
    results = []
    warmup_jit_compilers(estimators)

    # --- Run p >> n scenario ---
    print("\n--- Running Scenario: p >> n (Many Features) ---", flush=True)
    n_samples = P_DOMINANT_SCENARIOS["n_samples"]
    for n_features in P_DOMINANT_SCENARIOS["n_features_range"]:
        print(f"\nGenerating data: {n_samples} samples, {n_features} features", flush=True)
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=min(20, n_features), random_state=42
        )

        for name, estimator_fn in estimators.items():
            if "scikit-rebate" in name and n_features > 300:
                continue
            for i in range(N_REPEATS):
                print(f"  Benchmarking {name} (Run {i+1}/{N_REPEATS})...", flush=True)
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
    print("\n--- Running Scenario: n >> p (Many Samples) ---", flush=True)
    n_features = N_DOMINANT_SCENARIOS["n_features"]
    for n_samples in N_DOMINANT_SCENARIOS["n_samples_range"]:
        print(f"\nGenerating data: {n_samples} samples, {n_features} features", flush=True)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=20, random_state=42)

        for name, estimator_fn in estimators.items():
            if "scikit-rebate" in name and n_samples > 300:
                continue
            for i in range(N_REPEATS):
                print(f"  Benchmarking {name} (Run {i+1}/{N_REPEATS})...", flush=True)
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
    output_file = "benchmarking/benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nBenchmarking complete. Results saved to '{output_file}'", flush=True)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()
