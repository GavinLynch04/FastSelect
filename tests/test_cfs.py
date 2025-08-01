import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from numba import cuda
from fast_select.CFS import CFS

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

    X = np.vstack([
        feature_0, feature_1, feature_2, feature_3, feature_4, feature_5
    ]).T

    # Expected outcome for most tests: features 0 and 2 are the best subset.
    expected_selection = [0, 2]

    return {
        "X_numpy": X,
        "X_pandas": pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]),
        "y": y,
        "expected": expected_selection
    }

GPU_UNAVAILABLE = not cuda.is_available()


def test_initialization():
    """Tests if the estimator initializes with the correct parameters."""
    cfs = CFS(n_bins=5, strategy='quantile', backend='cpu', n_jobs=4)
    assert cfs.n_bins == 5
    assert cfs.strategy == 'quantile'
    assert cfs.backend == 'cpu'
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
    assert hasattr(cfs, 'selected_indices_')
    assert hasattr(cfs, 'support_mask_')
    assert hasattr(cfs, 'merit_')
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

    cfs = CFS(backend='gpu', n_bins=10)
    cfs.fit(X_subset, y)

    # Check fitted attributes
    assert hasattr(cfs, 'selected_indices_')
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

    cfs = CFS(backend='cpu')
    cfs.fit(X_df_subset, y)

    # Check that feature names are stored
    assert hasattr(cfs, 'feature_names_in_')
    expected_names = [f"feature_{i}" for i in sample_data["expected"]]

    # Check transform output
    X_transformed_df = cfs.transform(X_df_subset)
    assert isinstance(X_transformed_df, pd.DataFrame)
    assert list(X_transformed_df.columns) == expected_names


def test_edge_case_no_features_selected(sample_data):
    """Tests behavior when no features have correlation with the target."""
    X, y = sample_data["X_numpy"], sample_data["y"]
    # Use only the random noise and constant features
    X_noise = X[:, 3:5]

    cfs = CFS(backend='cpu')
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

    cfs = CFS(backend='cpu')
    cfs.fit(X, y)

    np.testing.assert_array_equal(cfs.selected_indices_, [0])
    assert cfs.merit_ > 0
    assert cfs.transform(X).shape[1] == 1


def test_n_jobs_parameter(sample_data):
    """Tests that the n_jobs parameter runs without error."""
    X, y = sample_data["X_numpy"][:, :5], sample_data["y"]

    # Test with a single job
    cfs_1 = CFS(backend='cpu', n_jobs=1)
    cfs_1.fit(X, y)
    np.testing.assert_array_equal(cfs_1.selected_indices_, sample_data["expected"])

    # Test with multiple jobs
    cfs_multi = CFS(backend='cpu', n_jobs=-1)
    cfs_multi.fit(X, y)
    np.testing.assert_array_equal(cfs_multi.selected_indices_, sample_data["expected"])


@pytest.mark.skipif(not GPU_UNAVAILABLE, reason="This test is only for environments WITHOUT a GPU.")
def test_gpu_raises_error_if_unavailable(sample_data):
    """Checks that backend='gpu' raises a RuntimeError if no GPU is present."""
    X, y = sample_data["X_numpy"], sample_data["y"]
    cfs = CFS(backend='gpu')
    with pytest.raises(RuntimeError, match="backend='gpu', but no CUDA-enabled GPU is available"):
        cfs.fit(X, y)


@pytest.mark.skipif(GPU_UNAVAILABLE, reason="CUDA GPU not available for testing.")
def test_gpu_state_limit_handling(sample_data):
    """
    Tests that the GPU backend correctly handles features with more than 32 states.
    This tests the critical bug fix for fixed-size local arrays.
    """
    X, y = sample_data["X_numpy"], sample_data["y"]
    # Feature 5 has cardinality of 40, which is > 32
    X_high_cardinality = X[:, [0, 5]]

    cfs = CFS(backend='gpu', n_bins=10)
    # This should raise a ValueError due to the check for n_states > 32
    with pytest.raises(ValueError,
                       match="GPU backend supports up to 32 unique states/bins."):
        cfs.fit(X_high_cardinality, y)


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