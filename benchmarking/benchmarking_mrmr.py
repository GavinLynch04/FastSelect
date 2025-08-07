import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fast_select.mRMR import mRMR as FastMRMR
from mrmr import mrmr_classif

def run_benchmarks():
    """
    Benchmarks Fast-Select's mRMR against the 'mrmr' package,
    prints the results, and saves a performance plot. Includes a
    warm-up run for JIT compilation.
    """
    # --- Configuration ---
    feature_counts = [100, 500, 1000, 2000, 5000]
    n_samples = 1000
    k_features = 10  # Number of features to select
    results = []

    # --- JIT Warm-up Run ---
    print("Performing JIT warm-up run...")
    # Create a small, throwaway dataset to trigger compilation
    n_warmup_samples = 10
    n_warmup_features = 10
    X_warmup = np.random.randint(0, 5, size=(n_warmup_samples, n_warmup_features),  dtype=np.int32)
    y_warmup = np.random.randint(0, 2, n_warmup_samples)
    X_df_warmup = pd.DataFrame(X_warmup)
    y_s_warmup = pd.Series(y_warmup)

    # Warm-up Fast-Select
    fs_mrmr_warmup = FastMRMR(n_features_to_select=2)
    fs_mrmr_warmup.fit(X_warmup, y_warmup)

    # Warm-up mrmr package
    mrmr_classif(X=X_df_warmup, y=y_s_warmup, K=2)
    print("Warm-up complete.\n")

    # --- Main Benchmark Loop ---
    print("Running mRMR Benchmarks...")
    print("-" * 52)
    print(f"{'Features':<10} | {'Fast-Select Time (s)':<22} | {'mrmr-selection Time (s)':<22}")
    print("-" * 52)

    for n_features in feature_counts:
        # Generate synthetic data
        X = np.random.randint(0, 5, size=(n_samples, n_features), dtype=np.int32)

        y = np.random.randint(0, 2, n_samples)

        # Create pandas DataFrame for mrmr package
        feature_names = [f'f_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_s = pd.Series(y, name='target')

        # Benchmark Fast-Select mRMR
        fs_mrmr = FastMRMR(n_features_to_select=k_features)
        start_time = time.time()
        fs_mrmr.fit(X, y)
        fs_time = time.time() - start_time

        # Benchmark mrmr package
        start_time = time.time()
        mrmr_classif(X=X_df, y=y_s, K=k_features)
        mrmr_time = time.time() - start_time

        results.append({
            'features': n_features,
            'fast_select_time': fs_time,
            'mrmr_time': mrmr_time
        })
        print(f"{n_features:<10} | {fs_time:<22.4f} | {mrmr_time:<22.4f}")

    # --- Plotting Results ---
    df = pd.DataFrame(results)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['features'], df['fast_select_time'], marker='o', linestyle='-', label='Fast-Select mRMR')
    ax.plot(df['features'], df['mrmr_time'], marker='s', linestyle='--', label='mrmr-selection')
    ax.set_title('mRMR Performance Comparison (after warm-up)')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_yscale('log') # Log scale is useful for wide time ranges
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('mrmr_benchmark.png')
    print("\nBenchmark complete. Saved plot to mrmr_benchmark.png")

if __name__ == '__main__':
    run_benchmarks()