import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numba import cuda
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from fast_select import SURF as FastSURF


@pytest.fixture
def simple_classification_data():
    """
    Creates a simple, well-defined dataset for ReliefF testing.
    Classes are made very distinct to ensure positive scores for relevant features.

    - feature 0 (continuous): Highly relevant. Low for class 0, high for class 1.
    - feature 1 (continuous): Irrelevant noise.
    - feature 2 (discrete): Perfectly relevant. Value 10 for class 0, 20 for class 1.
    - feature 3 (continuous): Irrelevant, has zero range (constant).
    """
    X = np.array([
        # Class 0 - values are low
        [0.1, 5.0, 10, 3.0],
        [0.2, 4.0, 10, 3.0],
        [0.3, 6.0, 10, 3.0],
        # Class 1 - values are high and far away
        [10.8, 5.0, 20, 3.0],
        [10.9, 4.0, 20, 3.0],
        [11.0, 6.0, 20, 3.0],
    ], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    return X, y



def test_feature_importance_ranking(simple_classification_data):
    """
    Tests if the algorithm correctly identifies relevant vs. irrelevant features
    by checking the *ranking* of their importance scores.
    """
    X, y = simple_classification_data
    model = FastSURF(n_features_to_select=2, backend="cpu", discrete_limit=3)
    model.fit(X, y)
    scores = model.feature_importances_

    assert scores[0] > scores[1]
    assert scores[2] > scores[1]

    assert_allclose(scores[3], 0.0, atol=1e-7)

    assert set(model.top_features_) == {0, 2}


@pytest.mark.parametrize("use_star", [False, True])
def test_internal_consistency_cpu_gpu(simple_classification_data, use_star):
    """
    CRITICAL: Tests that the CPU and GPU backends produce identical results
    for both SURF and SURF*.
    """
    if not cuda.is_available():
        pytest.skip("Skipping CPU/GPU consistency test: No CUDA-enabled GPU found.")

    X, y = simple_classification_data

    cpu_model = FastSURF(backend="cpu", use_star=use_star)
    cpu_model.fit(X, y)
    scores_cpu = cpu_model.feature_importances_

    gpu_model = FastSURF(backend="gpu", use_star=use_star)
    gpu_model.fit(X, y)
    scores_gpu = gpu_model.feature_importances_

    assert_allclose(
        scores_cpu,
        scores_gpu,
        rtol=1e-5,
        atol=1e-7,
        err_msg=f"CPU and GPU scores do not match for use_star={use_star}",
    )



def test_sklearn_api_compliance():
    """
    Uses scikit-learn's built-in checker to validate the estimator's compliance.
    """
    check_estimator(FastSURF())


def test_fit_transform_output_shape(simple_classification_data):
    """Tests that fit_transform returns a matrix of the correct shape."""
    X, y = simple_classification_data
    k_select = 2
    model = FastSURF(n_features_to_select=k_select, backend="cpu")
    X_transformed = model.fit_transform(X, y)

    assert X_transformed.shape == (X.shape[0], k_select)


def test_discrete_limit_parameter():
    """Tests that `discrete_limit` correctly identifies discrete vs. continuous features."""
    # Feature 0 has 11 unique values. Feature 1 has 3.
    X = np.array([[i, i % 3] for i in range(11)] * 2, dtype=np.float32)
    y = np.array([0] * 11 + [1] * 11, dtype=np.int32)

    model_cont = FastSURF(discrete_limit=10, backend="cpu")
    model_cont.fit(X, y)
    assert_array_equal(model_cont.is_discrete_, [False, True])

    model_disc = FastSURF(discrete_limit=12, backend="cpu")
    model_disc.fit(X, y)
    assert_array_equal(model_disc.is_discrete_, [True, True])


def test_not_fitted_error(simple_classification_data):
    """Tests that a NotFittedError is raised if transform is called before fit."""
    X, _ = simple_classification_data
    model = FastSURF()
    with pytest.raises(NotFittedError):
        model.transform(X)


def test_backend_error_handling(simple_classification_data):
    """Tests that requesting the GPU backend without a GPU raises a RuntimeError."""
    if cuda.is_available():
        pytest.skip("Skipping GPU error test: GPU is available.")
    
    X, y = simple_classification_data
    with pytest.raises(RuntimeError, match="no CUDA-enabled GPU is available"):
        model = FastSURF(backend="gpu")
        model.fit(X, y)

def test_verbose_output(simple_classification_data, capsys):
    """Check that verbose=True prints to stdout."""
    X, y = simple_classification_data
    model = FastSURF(verbose=True)
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "Running SURF" in captured.out
    
    model = FastSURF(verbose=True, backend='cpu', use_star=True)
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "Running SURF*" in captured.out
        
def test_backend(simple_classification_data):
    """
    Tests that transform raises a ValueError if backend is not auto, cpu, or gpu
    """
    X, y = simple_classification_data
        
    with pytest.raises(ValueError):
        transformer = FastSURF(n_features_to_select=4, backend='tpu').fit(X, y)

def test_nan_input_raises_error(simple_classification_data):
    """Tests that the estimator raises a ValueError for data containing NaNs."""
    X_orig, y = simple_classification_data

    X = X_orig.copy()
    X[0, 0] = np.nan

    model = FastSURF(backend="cpu")
    with pytest.raises(ValueError, match="Input X contains NaN."):
        model.fit(X, y)
        
@pytest.mark.parametrize("bad_k_select", [-1, 0, 100])
def test_invalid_n_features_to_select_raises_error(simple_classification_data, bad_k_select):
    """
    Tests that an invalid n_features_to_select value raises a ValueError.
    """
    X, y = simple_classification_data

    with pytest.raises(ValueError):
        FastSURF(n_features_to_select=bad_k_select).fit(X, y)
    with pytest.raises(ValueError):
        FastSURF(n_features_to_select=1.1).fit(X, y)
    with pytest.raises(TypeError):
        FastSURF(n_features_to_select='hi').fit(X, y)


def test_single_class_input(simple_classification_data):
    """
    Tests behavior with only one class label. Scores should be less than zero as there
    are no "misses" to learn from. A negative score is the expected penalty for intra-class variation.
    """
    X, _ = simple_classification_data
    y_single_class = np.zeros(X.shape[0])
    
    model = FastSURF(backend="cpu")
    model.fit(X, y_single_class)
    
    assert np.all(model.feature_importances_ <= 1e-7)


