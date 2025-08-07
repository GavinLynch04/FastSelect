import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fast_select.MDR import MDR as FastMDR
from skrebate import MDR as SkrebateMDR

def run_benchmarks():
    """
    Benchmarks Fast-Select's MDR against scikit-rebate's implementation,
    prints the results, and saves a performance plot.
    """
    feature_counts = [100, 250, 500, 750, 1000]
    n_samples = 500
    results = []

    print("Running MDR Benchmarks...")
    print("-" * 30)
    print(f"{'Features':<10} | {'Fast-Select (s)':<18} | {'scikit-rebate (s)':<20}")
    print("-" * 30)

    for n_features in feature_counts:
        # Generate synthetic data
        X = np.random.randint(0, 3, size=(n_samples, n_features))
        y = np.random.randint(0, 2, size=n_samples)

        # Benchmark Fast-Select MDR
        fs_mdr = FastMDR()
        start_time = time.time()
        fs_mdr.fit(X, y)
        fs_time = time.time() - start_time

        # Benchmark scikit-rebate MDR
        sk_mdr = SkrebateMDR()
        start_time = time.time()
        sk_mdr.fit(X, y)
        sk_time = time.time() - start_time

        results.append({
            'features': n_features,
            'fast_select_time': fs_time,
            'skrebate_time': sk_time
        })
        print(f"{n_features:<10} | {fs_time:<18.4f} | {sk_time:<20.4f}")

    # Plotting results
    df = pd.DataFrame(results)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['features'], df['fast_select_time'], marker='o', linestyle='-', label='Fast-Select MDR')
    ax.plot(df['features'], df['skrebate_time'], marker='s', linestyle='--', label='scikit-rebate MDR')
    ax.set_title('MDR Performance Comparison')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('mdr_benchmark.png')
    print("\nSaved plot to mdr_benchmark.png")

if __name__ == '__main__':
    run_benchmarks()