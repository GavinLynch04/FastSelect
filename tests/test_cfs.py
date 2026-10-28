import math

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from fast_select.CFS import CFS
from fast_select.utils import is_cuda_ready


@pytest.fixture(scope="module")
def sample_data():
    """
    Creates a predictable dataset for testing.

    - feature_0: Highly correlated with the target y. (Should be selected)
    - feature_1: Redundant with feature_0. (Should NOT be selected)
    - feature_2: Moderately correlated with y, uncorrelated with other features. (Should be selected)
    - feature_3: Pure random noise. (Should NOT be selected)
    - feature_4: A constant feature. (Should NOT be selected)
    - feature_5: A discrete feature with high cardinality for GPU testing.
    """
    np.random.seed(42)
    n_samples = 200
    y = np.random.randint(0, 2, n_samples)

    # feature_0: highly correlated
    feature_0 = y + np.random.normal(0, 0.1, n_samples)

    # feature_1: redundant with feature_0
    feature_1 = feature_0 + np.random.normal(0, 0.05, n_samples)

    # feature_2: moderately correlated, but independent
    feature_2 = y + np.random.normal(0, 0.5, n_samples)
    feature_2[y == 0] -= 0.5  # Add some separation

    # feature_3: noise
    feature_3 = np.random.rand(n_samples) * 10

    # feature_4: constant
    feature_4 = np.full(n_samples, 5.0)

    # feature_5: discrete with high cardinality
    feature_5 = np.random.randint(0, 40, n_samples)

    X = np.vstack([feature_0, feature_1, feature_2, feature_3, feature_4, feature_5]).T

    # The canonical CFS merit slightly favors retaining feature 1 as well.
    # Earlier FastSelect releases removed it with an unpublished post-search
    # pruning rule.
    expected_selection = [0, 1, 2]

    return {
        "X_numpy": X,
        "X_pandas": pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]),
        "y": y,
        "expected": expected_selection,
    }


GPU_UNAVAILABLE = not is_cuda_ready()


def test_initialization():
    """Tests if the estimator initializes with the correct parameters."""
    cfs = CFS(n_bins=5, strategy="quantile", backend="cpu", n_jobs=4)
    assert cfs.n_bins == 5
    assert cfs.strategy == "quantile"
    assert cfs.backend == "cpu"
    assert cfs.n_jobs == 4


def test_not_fitted_error(sample_data):
    """Tests that transform raises NotFittedError if called before fit."""
    cfs = CFS()
    with pytest.raises(NotFittedError):
        cfs.transform(sample_data["X_numpy"])
    with pytest.raises(NotFittedError):
        cfs._get_support_mask()


@pytest.mark.parametrize("backend", ["cpu", "auto"])
def test_fit_transform_cpu_and_auto(sample_data, backend):
    """
    Tests the full fit-transform pipeline on CPU and auto-backend selection.
    This is the primary "happy path" test.
    """
    X, y = sample_data["X_numpy"], sample_data["y"]
    X_subset = X[:, :5]

    cfs = CFS(backend=backend, n_bins=10)
    cfs.fit(X_subset, y)

    # Check fitted attributes
    assert hasattr(cfs, "selected_indices_")
    assert hasattr(cfs, "support_mask_")
    assert hasattr(cfs, "merit_")
    assert cfs.n_features_in_ == X_subset.shape[1]

    # Check selection correctness
    np.testing.assert_array_equal(cfs.selected_indices_, sample_data["expected"])
    assert cfs.merit_ > 0

    # Check transform
    X_transformed = cfs.transform(X_subset)
    assert X_transformed.shape[0] == X_subset.shape[0]
    assert X_transformed.shape[1] == len(sample_data["expected"])

    # Check that the correct columns were selected
    np.testing.assert_array_equal(X_transformed, X_subset[:, sample_data["expected"]])


@pytest.mark.skipif(GPU_UNAVAILABLE, reason="CUDA GPU not available for testing.")
def test_fit_transform_gpu(sample_data):
    """Tests the full fit-transform pipeline on the GPU backend."""
    X, y = sample_data["X_numpy"], sample_data["y"]
    # Exclude the high-cardinality feature which should fail in the next test
    X_subset = X[:, :5]

    cfs = CFS(backend="gpu", n_bins=10)
    cfs.fit(X_subset, y)

    # Check fitted attributes
    assert hasattr(cfs, "selected_indices_")
    assert cfs.n_features_in_ == X_subset.shape[1]

    # Check selection correctness
    np.testing.assert_array_equal(cfs.selected_indices_, sample_data["expected"])
    assert cfs.merit_ > 0

    # Check transform
    X_transformed = cfs.transform(X_subset)
    assert X_transformed.shape[1] == len(sample_data["expected"])
    np.testing.assert_array_equal(X_transformed, X_subset[:, sample_data["expected"]])


def test_pandas_integration(sample_data):
    """Tests that the selector works correctly with pandas DataFrames."""
    X_df, y = sample_data["X_pandas"], sample_data["y"]
    X_df_subset = X_df.iloc[:, :5]

    cfs = CFS(backend="cpu")
    cfs.fit(X_df_subset, y)

    # Check that feature names are stored
    assert hasattr(cfs, "feature_names_in_")
    expected_names = [f"feature_{i}" for i in sample_data["expected"]]

    # Check transform output
    X_transformed_df = cfs.transform(X_df_subset)
    assert isinstance(X_transformed_df, pd.DataFrame)
    assert list(X_transformed_df.columns) == expected_names


def test_edge_case_no_features_selected(sample_data):
    """Tests behavior when no features have correlation with the target."""
    X_noise = np.tile(
        np.array([[0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.int32),
        (25, 1),
    )
    y = np.tile(np.array([0, 1, 0, 1], dtype=np.int32), 25)

    cfs = CFS(backend="cpu")
    cfs.fit(X_noise, y)

    assert len(cfs.selected_indices_) == 0
    assert cfs.merit_ == 0.0
    assert np.sum(cfs.support_mask_) == 0

    X_transformed = cfs.transform(X_noise)
    assert X_transformed.shape[1] == 0


def test_edge_case_single_feature(sample_data):
    """Tests behavior with only one input feature."""
    X = sample_data["X_numpy"][:, [0]]  # The best feature
    y = sample_data["y"]

    cfs = CFS(backend="cpu")
    cfs.fit(X, y)

    np.testing.assert_array_equal(cfs.selected_indices_, [0])
    assert cfs.merit_ > 0
    assert cfs.transform(X).shape[1] == 1


def test_n_jobs_parameter(sample_data):
    """Tests that the n_jobs parameter runs without error."""
    X, y = sample_data["X_numpy"][:, :5], sample_data["y"]

    # Test with a single job
    cfs_1 = CFS(backend="cpu", n_jobs=1)
    cfs_1.fit(X, y)
    np.testing.assert_array_equal(cfs_1.selected_indices_, sample_data["expected"])

    # Test with multiple jobs
    cfs_multi = CFS(backend="cpu", n_jobs=-1)
    cfs_multi.fit(X, y)
    np.testing.assert_array_equal(cfs_multi.selected_indices_, sample_data["expected"])


@pytest.mark.skipif(not GPU_UNAVAILABLE, reason="This test is only for environments WITHOUT a GPU.")
def test_gpu_raises_error_if_unavailable(sample_data):
    """Checks that backend='gpu' raises a RuntimeError if no GPU is present."""
    X, y = sample_data["X_numpy"], sample_data["y"]
    cfs = CFS(backend="gpu")
    with pytest.raises(RuntimeError, match="backend='gpu', but no CUDA-enabled GPU is available"):
        cfs.fit(X, y)


@pytest.mark.skipif(GPU_UNAVAILABLE, reason="CUDA GPU not available for testing.")
def test_gpu_state_limit_handling(sample_data):
    """
    Tests that the GPU backend correctly handles features with more than 32 states.
    This tests the critical bug fix for fixed-size local arrays.
    """
    X_high_cardinality = np.arange(35, dtype=np.float64).reshape(35, 1).repeat(2, axis=1)
    y_high = np.random.choice([0, 1], size=35)

    cfs = CFS(backend="gpu", n_bins=35)
    # This should raise a ValueError due to the check for n_states > 32
    with pytest.raises(ValueError, match="GPU backend supports up to 32 unique states/bins."):
        cfs.fit(X_high_cardinality, y_high)


def test_transform_bug_fix(sample_data):
    """
    Explicitly tests the fix for the recursive transform method.
    This test will fail with a RecursionError on the original code.
    """
    X, y = sample_data["X_numpy"][:, :5], sample_data["y"]
    cfs = CFS().fit(X, y)

    transformed_X = cfs.transform(X)

    assert isinstance(transformed_X, np.ndarray)
    assert transformed_X.shape[1] == len(cfs.selected_indices_)


def _best_first_oracle(r_cf, r_ff, max_backtracks):
    """Hall (1999) forward best-first search, every merit computed from scratch.

    Correlations are multiples of 1/8 here, so every sum is exact and ties are
    broken only by the documented ascending subset-tuple order.
    """
    import heapq

    def merit(subset):
        k = len(subset)
        mean_cf = sum(float(r_cf[f]) for f in subset) / k
        pairs = [float(r_ff[a, b]) for i, a in enumerate(subset) for b in subset[i + 1 :]]
        mean_ff = 2 * sum(pairs) / (k * (k - 1)) if k > 1 else 0.0
        denominator = math.sqrt(k + k * (k - 1) * mean_ff)
        return k * mean_cf / denominator if denominator > 1e-12 else 0.0

    n_features = len(r_cf)
    queue = [(-merit((f,)), (f,)) for f in range(n_features)]
    heapq.heapify(queue)
    seen = {subset for _, subset in queue}
    best, best_merit, stale = (), 0.0, 0
    while queue and stale < max_backtracks:
        negative, subset = heapq.heappop(queue)
        if -negative > best_merit + 1e-12:
            best, best_merit, stale = subset, -negative, 0
        else:
            stale += 1
        for f in range(n_features):
            child = tuple(sorted(subset + (f,)))
            if f not in subset and child not in seen:
                seen.add(child)
                heapq.heappush(queue, (-merit(child), child))
    return list(best)


@pytest.mark.parametrize("seed", range(40))
def test_best_first_search_matches_oracle_with_exact_ties(seed):
    from fast_select.CFS import _best_first_search

    rng = np.random.default_rng(seed)
    n_features = int(rng.integers(1, 12))
    r_cf = (rng.integers(0, 5, n_features) / 8).astype(np.float32)
    r_ff = np.triu(rng.integers(0, 3, (n_features, n_features)) / 8, 1)
    r_ff = (r_ff + r_ff.T).astype(np.float32)
    for max_backtracks in (1, 3, 5):
        expected = _best_first_oracle(r_cf, r_ff, max_backtracks)
        assert _best_first_search(n_features, r_cf, r_ff, max_backtracks) == expected


@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.int64, np.uint64])
def test_parallel_integer_encoding_matches_unique_encoding(dtype):
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 5, (120, 7))
    X = (codes * 3).astype(dtype) + (dtype(2**63) if dtype == np.uint64 else dtype(0))
    y = rng.integers(0, 3, 120)
    integer = CFS(backend="cpu").fit(X, y)
    reference = CFS(backend="cpu").fit(X.astype(object), y)  # per-column np.unique path
    np.testing.assert_array_equal(integer.selected_indices_, reference.selected_indices_)
    assert integer.merit_ == reference.merit_
