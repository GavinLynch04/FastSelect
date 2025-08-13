import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fast_select.CFS import CFS
from skfeature.function.statistical_based import CFS as sk_cfs

def run_benchmarks():
    """
    Benchmarks Fast-Select's CFS against skfeature implementation,
    prints the results, and saves a performance plot.
    """
    feature_counts = [100, 200, 300, 400, 500]
    n_samples = 1000
    results = []

    print("Running CFS Benchmarks...")
    print("-" * 30)
    print(f"{'Features':<10} | {'Fast-Select (s)':<18} | {'skfeature (s)':<15}")
    print("-" * 30)

    for n_features in feature_counts:
        # Generate synthetic data
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Benchmark Fast-Select CFS
        fs_cfs = CFS()
        start_time = time.time()
        fs_cfs.fit(X, y)
        fs_time = time.time() - start_time

        # Benchmark skfeature CFS
        start_time = time.time()
        sk_cfs.cfs(X, y)
        sk_time = time.time() - start_time

        results.append({
            'features': n_features,
            'fast_select_time': fs_time,
            'skfeature_time': sk_time
        })
        print(f"{n_features:<10} | {fs_time:<18.4f} | {sk_time:<15.4f}")

    # Plotting results
    df = pd.DataFrame(results)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['features'], df['fast_select_time'], marker='o', linestyle='-', label='Fast-Select CFS')
    ax.plot(df['features'], df['skfeature_time'], marker='s', linestyle='--', label='skfeature CFS')
    ax.set_title('CFS Performance Comparison')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('cfs_benchmark.png')
    print("\nSaved plot to cfs_benchmark.png")

if __name__ == '__main__':
    run_benchmarks()