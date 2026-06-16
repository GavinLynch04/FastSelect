import pytest
import numpy as np
from numba import cuda
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from fast_select.mRMR import mRMR, _encode_data_numba


@pytest.fixture(scope="module")
def discrete_classification_data():
    """
    Pytest fixture to create a reproducible, discrete classification dataset.
    This fixture is created once and shared across all tests in the module.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_classes=4,  # More than 2 classes
        random_state=42,
    )
    
    bin_edges = np.percentile(X, [25, 50, 75], axis=0)
    
    X_discrete = np.empty_like(X, dtype=np.int64)
    
    for i in range(X.shape[1]):
        X_discrete[:, i] = np.digitize(X[:, i], bins=bin_edges[:, i])
        
    return X_discrete, y



def test_init_parameter_validation():
    """Verify that the constructor raises errors for invalid parameters."""
    X = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    assert mRMR(n_features_to_select=1, method='INVALID_METHOD').method == 'INVALID_METHOD'
    with pytest.raises(ValueError, match="Method must be either 'MID' or 'MIQ'"):
        mRMR(n_features_to_select=1, method='INVALID_METHOD').fit(X, y)

    assert mRMR(n_features_to_select=1, backend='tpu').backend == 'tpu'
    with pytest.raises(ValueError, match="Backend must be either 'cpu' or 'gpu'"):
        mRMR(n_features_to_select=1, backend='tpu').fit(X, y)

@pytest.mark.skipif(cuda.is_available(), reason="This test is for when CUDA is NOT available")
def test_init_gpu_backend_fails_without_cuda():
    """Verify that selecting 'gpu' backend fails gracefully if CUDA is not found."""
    X = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(RuntimeError, match="Numba could not find a usable CUDA installation"):
        mRMR(n_features_to_select=1, backend='gpu').fit(X, y)


@pytest.mark.parametrize("method", ['MID', 'MIQ'])
def test_fit_transform_cpu(discrete_classification_data, method):
    """
    Test the full fit and transform cycle on the CPU backend for both methods.
    This covers the CPU host caller and JIT kernels.
    """
    X, y = discrete_classification_data
    n_samples, n_features = X.shape
    n_select = 5

    model = mRMR(n_features_to_select=n_select, method=method, backend='cpu')

    # Test fit()
    model.fit(X, y)
    assert hasattr(model, 'top_features_')
    assert hasattr(model, 'relevance_scores_')
    assert hasattr(model, 'redundancy_matrix_')
    assert model.top_features_.shape == (n_select,)
    assert model.relevance_scores_.shape == (n_features,)
    assert model.redundancy_matrix_.shape == (n_features, n_features)

    # Test transform()
    X_transformed = model.transform(X)
    assert X_transformed.shape == (n_samples, n_select)

    model2 = mRMR(n_features_to_select=n_select, method=method, backend='cpu')
    X_ft = model2.fit_transform(X, y)
    assert X_ft.shape == (n_samples, n_select)


@pytest.mark.skipif(not cuda.is_available(), reason="NVIDIA GPU with CUDA not available")
@pytest.mark.parametrize("method", ['MID', 'MIQ'])
def test_fit_transform_gpu(discrete_classification_data, method):
    """
    Test the full fit and transform cycle on the GPU backend for both methods.
    This covers the GPU host caller and CUDA kernels. Skipped if no GPU.
    """
    X, y = discrete_classification_data
    n_samples, n_features = X.shape
    n_select = 5

    model = mRMR(n_features_to_select=n_select, method=method, backend='gpu')
    
    # Test fit()
    model.fit(X, y)
    assert hasattr(model, 'top_features_')
    assert model.top_features_.shape == (n_select,)

    # Test transform()
    X_transformed = model.transform(X)
    assert X_transformed.shape == (n_samples, n_select)



@pytest.mark.parametrize("backend", ["cpu"])
def test_selects_correct_features(backend):
    """
    Verify that mRMR (MID) prefers a relevant-but-less-redundant feature
    over an exact duplicate of an already-selected one.

    Ground-truth:
      * Feature 0  – noisy copy of y (10 % flips)  → highly relevant
      * Feature 1  – exact duplicate of feature 0 → totally redundant
      * Feature 9  – cleaner copy of y (5 % flips)→ relevant, less redundant
    Expected selection order: 0 then 9.
    """
    rng = np.random.default_rng(42)   # reproducible across runs

    n_samples = 200
    n_features = 10

    y = rng.integers(0, 2, n_samples)

    X = rng.integers(0, 3, size=(n_samples, n_features))

    # feature 0:  noisy (10 % flips) copy of y
    flip0 = (rng.random(n_samples) < 0.10).astype(int)
    X[:, 0] = (y + flip0) % 2

    # feature 1: exact duplicate of feature 0
    X[:, 1] = X[:, 0]

    # feature 9: cleaner (5 % flips) copy of y still relevant, less redundant
    flip9 = (rng.random(n_samples) < 0.05).astype(int)
    X[:, 9] = (y + flip9) % 2

    model = mRMR(n_features_to_select=2, method="MID", backend=backend)
    model.fit(X, y)

    selected = set(model.top_features_)
    expected = {0, 9}

    assert (
        selected == expected
    ), f"Expected features {expected}, but got {selected}"


def test_sklearn_pipeline_compatibility(discrete_classification_data):
    """Ensures that mRMR works as a step within a scikit-learn Pipeline."""
    X, y = discrete_classification_data
    n_select = 3
    
    pipeline = Pipeline([
        ('mrmr_selector', mRMR(n_features_to_select=n_select, backend='cpu')),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # If this runs without errors, the pipeline compatibility is confirmed
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_input_validation_errors(discrete_classification_data):
    """Test for errors raised on invalid input shapes or incorrect usage."""
    X, y = discrete_classification_data
    n_features = X.shape[1]
    
    model = mRMR(n_features_to_select=5, backend='cpu')

    # 1. Calling transform before fit should raise NotFittedError
    with pytest.raises(NotFittedError):
        model.transform(X)
        
    # 2. n_features_to_select > n_features should raise ValueError
    bad_model = mRMR(n_features_to_select=n_features + 1, backend='cpu')
    with pytest.raises(ValueError, match="n_features_to_select must be a positive integer"):
        bad_model.fit(X, y)

    # 3. Calling transform with X of a different shape should raise ValueError
    model.fit(X, y)
    X_wrong_shape = np.delete(X, 0, axis=1) # Remove one feature
    with pytest.raises(ValueError, match="X has 19 features, but mRMR is expecting 20 features as input."):
        model.transform(X_wrong_shape)


def test_encode_data_numba(discrete_classification_data):
    """Test the standalone JIT-compiled data encoder."""
    X, y = discrete_classification_data
    unique_vals = np.unique(np.concatenate([np.unique(X), np.unique(y)]))
    
    X_encoded, y_encoded = _encode_data_numba(X, y, unique_vals)
    
    assert X_encoded.shape == X.shape
    assert y_encoded.shape == y.shape
    assert np.max(X_encoded) < len(unique_vals)
    assert np.max(y_encoded) < len(unique_vals)
    assert X_encoded.dtype == np.int32
    assert y_encoded.dtype == np.int32


def test_float_coded_discrete_data(discrete_classification_data):
    X, y = discrete_classification_data
    X = X.astype(np.float64)

    model = mRMR(n_features_to_select=3, backend='cpu')
    model.fit(X, y)

    assert model.top_features_.shape == (3,)
    assert model.redundancy_matrix_.shape == (X.shape[1], X.shape[1])
