import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numba import cuda
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from src.fast_select.MultiSURF import MultiSURF as FastMultiSURF


@pytest.fixture(scope="module")
def simple_classification_data():
    """
    Creates a small, well-defined dataset for testing the core logic.
    'scope="module"' ensures this expensive setup runs only once per test file.

    - feature 0 (continuous): Highly relevant. Linearly separates classes.
    - feature 1 (continuous): Irrelevant noise.
    - feature 2 (discrete): Perfectly relevant. Value 10 for class 0, 20 for class 1.
    - feature 3 (continuous): Irrelevant, has zero range (all values are the same).
    - feature 4 (discrete): Irrelevant noise.
    """
    X = np.array([
        # Class 0
        [0.1, 5.0, 10, 3.0, 1],
        [0.2, 4.0, 10, 3.0, 2],
        [0.3, 6.0, 10, 3.0, 3],
        # Class 1
        [0.8, 5.0, 20, 3.0, 1],
        [0.9, 4.0, 20, 3.0, 2],
        [1.0, 6.0, 20, 3.0, 3],
    ], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    return X, y


# --- Core Logic and Behavior Tests ---

def test_feature_importance_ranking(simple_classification_data):
    """
    Tests if the algorithm correctly identifies relevant vs. irrelevant features
    by checking the *ranking* of their importance scores, not the exact values.
    """
    X, y = simple_classification_data
    model = FastMultiSURF(n_features_to_select=2, backend="cpu")
    model.fit(X, y)
    scores = model.feature_importances_

    # Assertions based on the designed data:
    # Relevant features (0, 2) should have higher scores than irrelevant ones (1, 4).
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
    assert scores[0] > scores[4]
    assert scores[2] > scores[4]
    
    # Feature 3 (zero range) provides no information and should have a score of 0.
    assert_allclose(scores[3], 0.0, atol=1e-7)

    # The top 2 selected features must be the relevant ones (0 and 2).
    assert set(model.top_features_) == {0, 2}


@pytest.mark.parametrize("use_star", [False, True])
def test_internal_consistency_cpu_gpu(simple_classification_data, use_star):
    """
    CRITICAL: Tests that the CPU and GPU backends produce identical results
    for both MultiSURF and MultiSURF*. This is our primary correctness check.
    """
    if not cuda.is_available():
        pytest.skip("Skipping CPU/GPU consistency test: No CUDA-enabled GPU found.")

    X, y = simple_classification_data

    # Run on CPU
    cpu_model = FastMultiSURF(backend="cpu", use_star=use_star)
    cpu_model.fit(X, y)
    scores_cpu = cpu_model.feature_importances_

    # Run on GPU
    gpu_model = FastMultiSURF(backend="gpu", use_star=use_star)
    gpu_model.fit(X, y)
    scores_gpu = gpu_model.feature_importances_

    # The scores should be extremely close (allowing for minor float precision diffs)
    assert_allclose(
        scores_cpu,
        scores_gpu,
        rtol=1e-5,
        atol=1e-7,
        err_msg=f"CPU and GPU scores do not match for use_star={use_star}",
    )


# --- Scikit-learn API Compliance and Parameter Tests ---

def test_sklearn_api_compliance():
    """
    Uses scikit-learn's built-in checker to validate the estimator's compliance.
    This is a powerful test that checks for dozens of common API requirements.
    """
    # Note: This will only test the default backend ('auto').
    check_estimator(FastMultiSURF())


def test_fit_transform_output_shape(simple_classification_data):
    """Tests that fit_transform returns a matrix of the correct shape."""
    X, y = simple_classification_data
    k_select = 3
    model = FastMultiSURF(n_features_to_select=k_select, backend="cpu")
    X_transformed = model.fit_transform(X, y)

    assert X_transformed.shape == (X.shape[0], k_select)


def test_discrete_limit_parameter():
    """Tests that `discrete_limit` correctly identifies discrete vs. continuous features."""
    # Feature 0 has 11 unique values. Feature 1 has 3.
    X = np.array([[i, i % 3] for i in range(11)] * 2, dtype=np.float32)
    y = np.array([0] * 11 + [1] * 11, dtype=np.int32)

    # With discrete_limit=10, feature 0 should be continuous, feature 1 discrete.
    # The internal `is_discrete_` attribute should reflect this.
    model_cont = FastMultiSURF(discrete_limit=10, backend="cpu")
    model_cont.fit(X, y)
    assert_array_equal(model_cont.is_discrete_, [False, True])

    # With discrete_limit=12, both features should be considered discrete.
    model_disc = FastMultiSURF(discrete_limit=12, backend="cpu")
    model_disc.fit(X, y)
    assert_array_equal(model_disc.is_discrete_, [True, True])


# --- Error Handling and Edge Case Tests ---

def test_not_fitted_error(simple_classification_data):
    """Tests that a NotFittedError is raised if transform is called before fit."""
    X, _ = simple_classification_data
    model = FastMultiSURF()
    with pytest.raises(NotFittedError):
        model.transform(X)


def test_backend_error_handling(simple_classification_data):
    """Tests that requesting the GPU backend without a GPU raises a RuntimeError."""
    if cuda.is_available():
        pytest.skip("Skipping GPU error test: GPU is available.")
    
    X, y = simple_classification_data
    with pytest.raises(RuntimeError, match="no compatible NVIDIA GPU"):
        model = FastMultiSURF(backend="gpu")
        model.fit(X, y)


def test_nan_input_raises_error(simple_classification_data):
    """Tests that the estimator raises a ValueError for data containing NaNs."""
    X, y = simple_classification_data
    X[0, 0] = np.nan
    
    model = FastMultiSURF(backend="cpu")
    with pytest.raises(ValueError, match="Input data contains NaN values"):
        model.fit(X, y)


def test_single_class_input(simple_classification_data):
    """
    Tests behavior with only one class label. Scores should be zero as there
    are no "misses" to learn from.
    """
    X, _ = simple_classification_data
    y_single_class = np.zeros(X.shape[0])
    
    model = FastMultiSURF(backend="cpu")
    model.fit(X, y_single_class)
    
    # With no misses, all feature importances should be zero.
    assert_allclose(model.feature_importances_, 0.0, atol=1e-7)
