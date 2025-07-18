import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from fast_select import Chi2 as chi2_numba


@pytest.fixture(scope="module")
def random_data_factory():
    """
    Pytest fixture factory to generate random data for testing.
    Using a factory allows creating multiple datasets with different parameters.
    """
    def _create_data(n_samples, n_features, n_classes, seed=42):
        """Generates a random dataset."""
        rng = np.random.RandomState(seed)
        X = rng.randint(0, 100, size=(n_samples, n_features))
        y = rng.randint(0, n_classes, size=n_samples)
        return X, y
    return _create_data



@pytest.mark.parametrize(
    "n_samples, n_features, n_classes",
    [
        (100, 10, 2),   # Standard binary classification
        (200, 50, 5),   # Standard multi-class
        (50, 5, 3),     # Small dataset
        (10, 2, 2),     # Tiny dataset for easy debugging
    ]
)
def test_correctness_against_sklearn(random_data_factory, n_samples, n_features, n_classes):
    """
    Tests that the output of chi2_numba exactly matches scikit-learn's implementation
    across a variety of data shapes. This is the most critical test.
    """
    X, y = random_data_factory(n_samples, n_features, n_classes)

    # Ground truth from scikit-learn
    sk_chi2, sk_p_values = sklearn_chi2(X, y)

    # Our implementation's result
    numba_chi2, numba_p_values = chi2_numba(X, y)

    # Assert that the results are almost equal (to handle minor float discrepancies)
    np.testing.assert_allclose(numba_chi2, sk_chi2, rtol=1e-6, atol=1e-6,
                               err_msg="Chi-squared statistics do not match scikit-learn")
    np.testing.assert_allclose(numba_p_values, sk_p_values, rtol=1e-6, atol=1e-6,
                               err_msg="P-values do not match scikit-learn")



def test_edge_case_single_class(random_data_factory):
    """
    Tests the function's documented behavior when only one class is provided in y.
    It should return chi2 stats of 0 and p-values of 1.
    """
    X, _ = random_data_factory(n_samples=50, n_features=10, n_classes=3)
    # Create a target vector with only one class
    y_single_class = np.zeros(50, dtype=int)

    chi2_stats, p_values = chi2_numba(X, y_single_class)

    assert chi2_stats.shape == (X.shape[1],)
    assert p_values.shape == (X.shape[1],)
    np.testing.assert_array_equal(chi2_stats, np.zeros(X.shape[1]))
    np.testing.assert_array_equal(p_values, np.ones(X.shape[1]))

def test_edge_case_zero_feature(random_data_factory):
    """
    Tests behavior when a feature column contains only zeros.
    The chi2 statistic for this feature should be 0.
    """
    X, y = random_data_factory(n_samples=100, n_features=10, n_classes=3)
    
    # Set one feature column to all zeros
    zero_feature_idx = 3
    X[:, zero_feature_idx] = 0

    sk_chi2, _ = sklearn_chi2(X, y)
    numba_chi2, _ = chi2_numba(X, y)
    
    # The chi2 value for the zeroed-out feature should be 0
    assert numba_chi2[zero_feature_idx] == 0.0
    # And the overall result should still match scikit-learn
    np.testing.assert_allclose(numba_chi2, sk_chi2, rtol=1e-6)

def test_edge_case_constant_feature(random_data_factory):
    """
    Tests behavior with a feature column that is a non-zero constant.
    This should still produce valid results and match scikit-learn.
    """
    X, y = random_data_factory(n_samples=100, n_features=10, n_classes=4)
    
    # Set one feature to a constant value
    constant_feature_idx = 5
    X[:, constant_feature_idx] = 42

    sk_chi2, sk_p = sklearn_chi2(X, y)
    numba_chi2, numba_p = chi2_numba(X, y)
    
    np.testing.assert_allclose(numba_chi2, sk_chi2, rtol=1e-6)
    np.testing.assert_allclose(numba_p, sk_p, rtol=1e-6)



def test_error_on_negative_input(random_data_factory):
    """
    Ensures that a ValueError is raised if the input matrix X contains negative values.
    """
    X, y = random_data_factory(n_samples=50, n_features=10, n_classes=2)
    X[10, 3] = -1  # Introduce a negative value

    with pytest.raises(ValueError, match="Input matrix X must contain non-negative values."):
        chi2_numba(X, y)



@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_input_dtypes(random_data_factory, dtype):
    """
    Verifies that the function handles various common NumPy dtypes for the input matrix X
    and produces float64 output as expected.
    """
    X, y = random_data_factory(n_samples=50, n_features=5, n_classes=2)
    X = X.astype(dtype)

    # Just run the function to ensure it doesn't crash
    chi2_stats, p_values = chi2_numba(X, y)

    # Check output types
    assert chi2_stats.dtype == np.float64, f"Chi2 stats should be float64 for input dtype {dtype}"
    assert p_values.dtype == np.float64, f"P-values should be float64 for input dtype {dtype}"

    # Quick check against sklearn for this type
    sk_chi2, sk_p = sklearn_chi2(X, y)
    np.testing.assert_allclose(chi2_stats, sk_chi2, rtol=1e-6)



@pytest.mark.slow  
def test_large_data_smoke_test(random_data_factory):
    """
    A "smoke test" with a large dataset to ensure the parallel implementation
    is stable, completes without errors, and produces outputs of the correct shape and type.
    This is not a correctness test, but a stability check for the parallel code.
    """
    n_samples, n_features, n_classes = 5000, 500, 10
    X, y = random_data_factory(n_samples, n_features, n_classes, seed=123)
    
    # This might take a few seconds to run
    chi2_stats, p_values = chi2_numba(X, y)

    # Basic sanity checks on the output
    assert chi2_stats.shape == (n_features,), "Output chi2_stats has incorrect shape"
    assert p_values.shape == (n_features,), "Output p_values has incorrect shape"
    
    assert np.all(np.isfinite(chi2_stats)), "Found NaN or Inf in chi2_stats"
    assert np.all(np.isfinite(p_values)), "Found NaN or Inf in p_values"
    
    assert np.all(chi2_stats >= 0), "Chi-squared statistics must be non-negative"
    assert np.all((p_values >= 0) & (p_values <= 1)), "P-values must be between 0 and 1"