import timeit
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import chi2 as chi2_sklearn
from src.others.chi2 import chi2_numba

# Set up the test parameters
N_SAMPLES = 2000
N_FEATURES = 200000
N_CLASSES = 5
RANDOM_STATE = 42

print("📊 Chi-Squared Implementation Benchmark")
print("-" * 40)
print(f"Dataset shape: Samples={N_SAMPLES}, Features={N_FEATURES}, Classes={N_CLASSES}")
print("-" * 40)

# 1. Generate synthetic data
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=500,
    n_redundant=0,
    n_classes=N_CLASSES,
    n_clusters_per_class=1,
    random_state=RANDOM_STATE
)
# The Chi-squared test requires non-negative features (e.g., counts)
X = np.abs(X * 100).astype(np.int64)

# 2. Run the Numba implementation
# First run is for JIT compilation ("warm-up") and is not timed.
print("Compiling Numba function...")
chi2_numba(X, y)
print("Compilation complete.\n")

# Time the Numba implementation
print("⏱️  Timing Numba implementation...")
numba_time = timeit.timeit(lambda: chi2_numba(X, y), number=10)
print(f"Done.")

# 3. Run the scikit-learn implementation
print("\n⏱️  Timing scikit-learn implementation...")
sklearn_time = timeit.timeit(lambda: chi2_sklearn(X, y), number=10)
print(f"Done.")

# 4. Verify that the results are the same
chi2_n, p_n = chi2_numba(X, y)
chi2_s, p_s = chi2_sklearn(X, y)

assert np.allclose(chi2_n, chi2_s), "Chi2 statistics do not match!"
assert np.allclose(p_n, p_s), "P-values do not match!"
print("\n✅ Correctness check passed: Results are identical.")

# 5. Report the results
print("\n\n--- 🚀 Benchmark Results ---")
print(f"Scikit-learn time: {sklearn_time:.4f} seconds")
print(f"Numba time:        {numba_time:.4f} seconds")

speedup = sklearn_time / numba_time
print(f"\n✨ Numba implementation is {speedup:.2f}x faster. ✨")