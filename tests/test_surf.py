import time
import pytest
import numpy as np
from numpy.testing import assert_allclose
from skrebate import SURF as SkrebateSURF
from sklearn.datasets import make_classification
from numba import cuda
from src.fast_relief.SURF import SURF as FastSURF


@pytest.fixture(scope="module")
def synthetic_data():
    """
    Creates a standard, large-scale synthetic dataset for testing.
    'scope="module"' means this function runs only once per test file.
    """
    X, y = make_classification(
        n_samples=200,
        n_features=200,
        n_informative=100,
        n_redundant=85,
        random_state=42
    )
    return X, y


def test_surf_agrees_with_skrebate(synthetic_data):
    """
    Compares the feature scores from our CPU and GPU SURF implementations
    against the scores produced by the skrebate library to ensure correctness.
    """
    X, y = synthetic_data

    skrebate_model = SkrebateSURF(n_features_to_select=10)
    start_time = time.perf_counter()
    skrebate_model.fit(X, y)
    end_time = time.perf_counter()
    scores_skrebate = skrebate_model.feature_importances_
    print(f"\nskrebate SURF CPU time: {end_time - start_time:.4f}s")

    fast_cpu_model = FastSURF(n_features_to_select=10, backend='cpu')
    start_time_fast = time.perf_counter()
    fast_cpu_model.fit(X, y)
    end_time_fast = time.perf_counter()
    scores_fast_cpu = fast_cpu_model.feature_importances_
    print(f"FastRelief SURF CPU time: {end_time_fast - start_time_fast:.4f}s")

    assert_allclose(scores_fast_cpu.astype(np.float64), scores_skrebate, rtol=1e-5, atol=1e-8,
                    err_msg="CPU SURF implementation scores do not match skrebate scores.")
    print("\nCPU SURF implementation scores match skrebate.")

    if cuda.is_available():
        fast_gpu_model = FastSURF(n_features_to_select=10, backend='gpu')
        start_time_gpu = time.perf_counter()
        fast_gpu_model.fit(X, y)
        end_time_gpu = time.perf_counter()
        scores_fast_gpu = fast_gpu_model.feature_importances_
        print(f"FastRelief SURF GPU time: {end_time_gpu - start_time_gpu:.4f}s")

        assert_allclose(scores_fast_gpu, scores_skrebate, rtol=1e-5, atol=1e-8,
                        err_msg="GPU SURF implementation scores do not match skrebate scores.")
        print("GPU SURF implementation scores match skrebate.")
    else:
        pytest.skip("Skipping GPU agreement test: No CUDA-enabled GPU found.")


def test_sklearn_api_compatibility(synthetic_data):
    """
    Tests if the SURF estimator adheres to the basic scikit-learn API contract.
    """
    X, y = synthetic_data
    n_select = 5

    model = FastSURF(n_features_to_select=n_select, backend='cpu')

    X_transformed = model.fit(X, y).transform(X)
    assert X_transformed.shape == (X.shape[0], n_select), "Transform output shape is incorrect."

    X_transformed_fit = model.fit_transform(X, y)
    assert X_transformed_fit.shape == (X.shape[0], n_select), "fit_transform output shape is incorrect."

    assert hasattr(model, 'top_features_')
    assert len(model.top_features_) == n_select


def test_backend_error_handling():
    """
    Tests that requesting the GPU backend without a GPU raises an error.
    """
    if not cuda.is_available():
        with pytest.raises(RuntimeError, match="no compatible GPU found"):
            model = FastSURF(backend='gpu')
            model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    else:
        pytest.skip("Skipping GPU error test: GPU is available.")

