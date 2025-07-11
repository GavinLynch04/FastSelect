import time

import pytest
import numpy as np
from numpy.testing import assert_allclose

from skrebate import MultiSURF as SkrebateMultiSURF
from sklearn.datasets import make_classification
from numba import cuda

from src.fast_relief.MultiSURF import MultiSURF as FastMultiSURF


@pytest.fixture(scope="module")
def synthetic_data():
    """
    Creates a standard, moderately-sized synthetic dataset for testing.
    'scope="module"' means this function runs only once per test file.
    """
    X, y = make_classification(
        n_samples=4000,
        n_features=4000,
        n_informative=100,
        n_redundant=85,
        random_state=42
    )
    return X, y


def test_multisurf_agrees_with_skrebate(synthetic_data):
    X, y = synthetic_data

    skrebate_model = SkrebateMultiSURF(n_features_to_select=10)
    start_time = time.perf_counter()
    skrebate_model.fit(X, y)
    end_time = time.perf_counter()
    scores_skrebate = skrebate_model.feature_importances_
    print(f"\nskrebate CPU time: {end_time - start_time:.4f}s")

    fast_cpu_model = FastMultiSURF(n_features_to_select=10, backend='cpu')
    start_time_fast = time.perf_counter()
    fast_cpu_model.fit(X, y)
    end_time_fast = time.perf_counter()
    scores_fast_cpu = fast_cpu_model.feature_importances_
    print(f"FastRelief MultiSURF CPU time: {end_time_fast - start_time_fast:.4f}s")

    assert_allclose(scores_fast_cpu, scores_skrebate, rtol=1e-5, atol=1e-8,
                    err_msg="CPU implementation scores do not match skrebate scores.")
    print("\nCPU implementation scores match skrebate.")

    if cuda.is_available():
        fast_gpu_model = FastMultiSURF(n_features_to_select=10, backend='gpu')
        fast_gpu_model.fit(X, y)
        scores_fast_gpu = fast_gpu_model.feature_importances_
        
        assert_allclose(scores_fast_gpu, scores_skrebate, rtol=1e-5, atol=1e-8,
                        err_msg="GPU implementation scores do not match skrebate scores.")
        print("GPU implementation scores match skrebate.")
    else:
        pytest.skip("Skipping GPU agreement test: No CUDA-enabled GPU found.")


def test_sklearn_api_compatibility(synthetic_data):
    """
    Tests if the estimator adheres to the basic scikit-learn API contract.
    """
    X, y = synthetic_data
    n_select = 5
    
    model = FastMultiSURF(n_features_to_select=n_select, backend='cpu')
    
    X_transformed = model.fit(X, y).transform(X)
    assert X_transformed.shape == (X.shape[0], n_select), "Transform output shape is incorrect."
    
    X_transformed_fit = model.fit_transform(X, y)
    assert X_transformed_fit.shape == (X.shape[0], n_select), "fit_transform output shape is incorrect."

    assert len(model.top_features_) == n_select

def test_backend_error_handling():
    """
    Tests that requesting the GPU backend without a GPU raises an error.
    """
    if not cuda.is_available():
        with pytest.raises(RuntimeError, match="no compatible NVIDIA GPU was found"):
            model = FastMultiSURF(backend='gpu')
            model.fit(np.array([[1,2],[3,4]]), np.array([0,1]))
    else:
        pytest.skip("Skipping GPU error test: GPU is available.")
