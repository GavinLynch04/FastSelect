import time
import warnings

import pandas as pd
from fast_relief.MultiSURF import MultiSURF as FastMultiSURF
from fast_relief.ReliefF import ReliefF as FastReliefF
from fast_relief.SURF import SURF as FastSURF
from sklearn.base import clone

# Import the estimators to compare
from sklearn.datasets import make_classification
from skrebate import SURF, MultiSURFstar, ReliefF, SURFstar
from skrebate import MultiSURF as SkrebateMultiSURF

# Try to import CUDA to see if GPU is available
try:
    from numba import cuda

    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# --- Benchmark Configuration ---
P_DOMINANT_SCENARIOS = {"n_samples": 100, "n_features_range": [200, 400, 600, 800, 1000]}
N_DOMINANT_SCENARIOS = {"n_features": 100, "n_samples_range": [200, 400, 600, 800, 1000]}
N_FEATURES_TO_SELECT = 10
N_REPEATS = 1

# --- Estimators to Test ---
estimators = {
    # skrebate estimators
    "skrebate.ReliefF": ReliefF(n_features_to_select=N_FEATURES_TO_SELECT, n_neighbors=10, n_jobs=-1),
    "skrebate.SURF": SURF(n_features_to_select=N_FEATURES_TO_SELECT, n_jobs=-1),
    "skrebate.SURF*": SURFstar(n_features_to_select=N_FEATURES_TO_SELECT, n_jobs=-1),
    "skrebate.MultiSURF": SkrebateMultiSURF(n_features_to_select=N_FEATURES_TO_SELECT, n_jobs=-1),
    "skrebate.MultiSURF*": MultiSURFstar(n_features_to_select=N_FEATURES_TO_SELECT, n_jobs=-1),
    # fast-relief CPU estimators
    "fast_relief.ReliefF (CPU)": FastReliefF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
    "fast_relief.SURF (CPU)": FastSURF(n_features_to_select=N_FEATURES_TO_SELECT),
    "fast_relief.SURF* (CPU)": FastSURF(n_features_to_select=N_FEATURES_TO_SELECT, use_star=True),
    "fast_relief.MultiSURF (CPU)": FastMultiSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu"),
    "fast_relief.MultiSURF* (CPU)": FastMultiSURF(
        n_features_to_select=N_FEATURES_TO_SELECT, backend="cpu", use_star=True
    ),
}

if GPU_AVAILABLE:
    print("NVIDIA GPU detected. Including GPU benchmarks.")
    estimators.update(
        {
            "fast_relief.ReliefF (GPU)": FastReliefF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast_relief.SURF (GPU)": FastSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast_relief.SURF* (GPU)": FastSURF(
                n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu", use_star=True
            ),
            "fast_relief.MultiSURF (GPU)": FastMultiSURF(n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu"),
            "fast_relief.MultiSURF* (GPU)": FastMultiSURF(
                n_features_to_select=N_FEATURES_TO_SELECT, backend="gpu", use_star=True
            ),
        }
    )
else:
    print("No NVIDIA GPU detected. Skipping GPU benchmarks.")


def run_single_benchmark(estimator, X, y):
    """Measures the runtime of a single estimator fit."""
    start_time = time.perf_counter()
    estimator.fit(X, y)
    end_time = time.perf_counter()
    return end_time - start_time


def warmup_jit_compilers(estimators_dict):
    """Performs a 'warm-up' run on a small dataset to compile JIT functions."""
    print("\n--- Warming up JIT compilers ---")
    X_warmup, y_warmup = make_classification(n_samples=10, n_features=10, random_state=42)

    for name, estimator in estimators_dict.items():
        if "fast_relief" in name:
            print(f"  Warming up {name}...")
            try:
                clone(estimator).fit(X_warmup, y_warmup)
            except Exception as e:
                warnings.warn(f"  > Warm-up FAILED for {name}. Reason: {e}")
    print("--- Warm-up complete ---")


def main():
    """Main function to run all benchmark scenarios."""
    results = []
    warmup_jit_compilers(estimators)

    # --- Run p >> n scenario ---
    print("\n--- Running Scenario: p >> n (Many Features) ---")
    n_samples = P_DOMINANT_SCENARIOS["n_samples"]
    for n_features in P_DOMINANT_SCENARIOS["n_features_range"]:
        print(f"\nGenerating data: {n_samples} samples, {n_features} features")
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=20, n_redundant=100, random_state=42
        )

        for name, estimator in estimators.items():
            for i in range(N_REPEATS):
                print(f"  Benchmarking {name} (Run {i+1}/{N_REPEATS})...")
                try:
                    runtime = run_single_benchmark(clone(estimator), X, y)
                    results.append(
                        {
                            "scenario": "p >> n",
                            "algorithm": name,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "runtime": runtime,
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
            n_samples=n_samples, n_features=n_features, n_informative=20, n_redundant=50, random_state=42
        )

        for name, estimator in estimators.items():
            for i in range(N_REPEATS):
                print(f"  Benchmarking {name} (Run {i+1}/{N_REPEATS})...")
                try:
                    runtime = run_single_benchmark(clone(estimator), X, y)
                    results.append(
                        {
                            "scenario": "n >> p",
                            "algorithm": name,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "runtime": runtime,
                        }
                    )
                except Exception as e:
                    warnings.warn(f"  > FAILED: {name} on {n_samples}x{n_features}. Reason: {e}")

    # --- Save results to CSV ---
    df = pd.DataFrame(results)
    output_file = "benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nBenchmarking complete. Results saved to '{output_file}'")


if __name__ == "__main__":
    main()
