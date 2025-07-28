import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from fast_select import chi2 as chi2_numba
from sklearn.feature_selection import chi2 as sklearn_chi2
from fast_select.Chi2 import _compute_observed_and_feature_counts, _chi2_core


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
    
    sk_chi2, sk_p_values = sklearn_chi2(X, y)

    numba_chi2, numba_p_values = chi2_numba(X, y)
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
    Tests behavior when a feature column is all zeros.
    The chi2 statistic for this feature should be 0 in our implementation,
    while scikit-learn produces NaN.
    """
    X, y = random_data_factory(n_samples=100, n_features=10, n_classes=3)
    zero_feature_idx = 3
    X[:, zero_feature_idx] = 0

    sk_chi2, _ = sklearn_chi2(X, y)
    numba_chi2, _ = chi2_numba(X, y)

    assert numba_chi2[zero_feature_idx] == 0.0

    assert np.isnan(sk_chi2[zero_feature_idx])

    sk_chi2_corrected = np.nan_to_num(sk_chi2, nan=0.0)
    np.testing.assert_allclose(numba_chi2, sk_chi2_corrected, rtol=1e-6)

def test_edge_case_constant_feature(random_data_factory):
    """
    Tests behavior with a non-zero constant feature.
    Our implementation should produce 0.0, while scikit-learn produces a
    tiny float due to precision errors. We use atol to compare.
    """
    X, y = random_data_factory(n_samples=100, n_features=10, n_classes=4)
    constant_feature_idx = 5
    X[:, constant_feature_idx] = 42.0

    sk_chi2, _ = sklearn_chi2(X, y)
    numba_chi2, _ = chi2_numba(X, y)

    assert numba_chi2[constant_feature_idx] == 0.0

    np.testing.assert_allclose(numba_chi2, sk_chi2, rtol=1e-6, atol=1e-9)



def test_error_on_negative_input(random_data_factory):
    """
    Ensures that a ValueError is raised if the input matrix X contains negative values.
    """
    X, y = random_data_factory(n_samples=50, n_features=10, n_classes=2)
    X[10, 3] = -1

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

    chi2_stats, p_values = chi2_numba(X, y)

    assert chi2_stats.dtype == np.float64, f"Chi2 stats should be float64 for input dtype {dtype}"
    assert p_values.dtype == np.float64, f"P-values should be float64 for input dtype {dtype}"

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
    
    chi2_stats, p_values = chi2_numba(X, y)

    assert chi2_stats.shape == (n_features,), "Output chi2_stats has incorrect shape"
    assert p_values.shape == (n_features,), "Output p_values has incorrect shape"
    
    assert np.all(np.isfinite(chi2_stats)), "Found NaN or Inf in chi2_stats"
    assert np.all(np.isfinite(p_values)), "Found NaN or Inf in p_values"
    
    assert np.all(chi2_stats >= 0), "Chi-squared statistics must be non-negative"
    assert np.all((p_values >= 0) & (p_values <= 1)), "P-values must be between 0 and 1"
    
def test_compute_observed_and_feature_counts():
    """
    Unit test for the internal _compute_observed_and_feature_counts function.
    Verifies that it correctly calculates the observed frequency matrix and
    the per-feature totals from a simple, known dataset.
    """
    X = np.array([
        [1, 2, 0],   # Class 0
        [3, 0, 5],   # Class 1
        [0, 4, 1],   # Class 0
        [2, 2, 3],   # Class 1
    ], dtype=np.float64)

    y_mapped = np.array([0, 1, 0, 1], dtype=np.int64)
    n_features = 3
    n_classes = 2

    observed, feature_counts = _compute_observed_and_feature_counts(
        X, y_mapped, n_features, n_classes
    )

    expected_observed = np.array([
        [1., 6., 1.],
        [5., 2., 8.]
    ], dtype=np.float64)

    expected_feature_counts = np.array([6., 8., 9.], dtype=np.float64)

    np.testing.assert_array_equal(observed, expected_observed)
    np.testing.assert_array_equal(feature_counts, expected_feature_counts)
    
def test_chi2_core_calculation():
    """
    Unit test for the internal, parallelized _chi2_core function.
    Verifies the correctness of the chi-squared statistic calculation using
    a pre-computed contingency table. Also checks the handling of zero-count features.
    """
    n_samples = 100
    n_classes = 2
    n_features = 3

    observed = np.array([
        [30., 10., 20.],
        [10., 30., 0.]
    ], dtype=np.float64)

    feature_counts = observed.sum(axis=0)
    class_freqs = observed.sum(axis=1)

    chi2_stats = _chi2_core(observed, class_freqs, feature_counts, n_samples)

    expected_chi2 = np.zeros(n_features)

    expected_chi2[0] = 3.75

    expected_chi2[1] = 20.416666666

    expected_chi2[2] = 13.333333333

    np.testing.assert_allclose(chi2_stats, expected_chi2)


def test_chi2_core_with_zero_feature_count():
    """
    Ensures that the _chi2_core function correctly handles a feature that has
    a total count of zero, which should result in a chi2 statistic of 0 for that feature.
    """
    observed = np.array([[10., 0., 20.], [15., 0., 5.]])
    feature_counts = observed.sum(axis=0)  # [25., 0., 25.]
    class_freqs = observed.sum(axis=1)     # [30., 20.]
    n_samples = 50

    chi2_stats = _chi2_core(observed, class_freqs, feature_counts, n_samples)

    assert chi2_stats[1] == 0.0
    assert chi2_stats[0] > 0.0
    assert chi2_stats[2] > 0.0
