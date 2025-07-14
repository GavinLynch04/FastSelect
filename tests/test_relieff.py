import time

import numpy as np
import pandas as pd
import pytest
from numba import cuda
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from skrebate import ReliefF as SkrebateReliefF

from src.fast_relief.ReliefF import ReliefF as FastReliefF


@pytest.fixture(scope="module")
def synthetic_data():
    """
    Creates a standard, moderately sized synthetic dataset for testing.
    'scope="module"' means this function runs only once per test file.
    """
    x, y = make_classification(
        n_samples=200,
        n_features=200,
        n_informative=100,
        n_redundant=85,
        random_state=42,
    )
    pd.DataFrame(x).to_csv("benchmark_x.csv", index=False)
    pd.DataFrame(y, columns=["target"]).to_csv("benchmark_y.csv", index=False)
    return x, y


def test_relieff_agrees_with_skrebate(synthetic_data):
    """
    Compares the feature scores from our CPU and GPU ReliefF implementations
    against the scores produced by the skrebate library to ensure correctness.
    """
    x, y = synthetic_data
    k = 10

    skrebate_model = SkrebateReliefF(n_features_to_select=10, n_neighbors=k)
    start_time = time.perf_counter()
    skrebate_model.fit(x, y)
    end_time = time.perf_counter()
    scores_skrebate = skrebate_model.feature_importances_
    print(f"\nskrebate ReliefF CPU time: {end_time - start_time:.4f}s")

    fast_cpu_model = FastReliefF(n_features_to_select=10, k_neighbors=k, backend="cpu")
    start_time_fast = time.perf_counter()
    fast_cpu_model.fit(x, y)
    end_time_fast = time.perf_counter()
    scores_fast_cpu = fast_cpu_model.feature_importances_
    print(f"FastRelief ReliefF CPU time: {end_time_fast - start_time_fast:.4f}s")

    assert_allclose(
        scores_fast_cpu,
        scores_skrebate,
        rtol=1e-5,
        atol=1e-8,
        err_msg="CPU ReliefF implementation scores do not match skrebate scores.",
    )
    print("\nCPU ReliefF implementation scores match skrebate.")

    if cuda.is_available():
        fast_gpu_model = FastReliefF(
            n_features_to_select=10, k_neighbors=k, backend="gpu"
        )
        start_time_gpu = time.perf_counter()
        fast_gpu_model.fit(x, y)
        end_time_gpu = time.perf_counter()
        scores_fast_gpu = fast_gpu_model.feature_importances_
        print(f"FastRelief ReliefF GPU time: {end_time_gpu - start_time_gpu:.4f}s")

        assert_allclose(
            scores_fast_gpu,
            scores_skrebate,
            rtol=1e-5,
            atol=1e-8,
            err_msg="GPU ReliefF implementation scores do not match skrebate scores.",
        )
        print("GPU ReliefF implementation scores match skrebate.")
    else:
        pytest.skip("Skipping GPU agreement test: No CUDA-enabled GPU found.")


def test_sklearn_api_compatibility(synthetic_data):
    """
    Tests if the ReliefF estimator adheres to the basic scikit-learn API contract.
    """
    x, y = synthetic_data
    n_select = 5

    model = FastReliefF(n_features_to_select=n_select, backend="cpu")

    x_transformed = model.fit(x, y).transform(x)
    assert x_transformed.shape == (
        x.shape[0],
        n_select,
    ), "Transform output shape is incorrect."

    x_transformed_fit = model.fit_transform(x, y)
    assert x_transformed_fit.shape == (
        x.shape[0],
        n_select,
    ), "fit_transform output shape is incorrect."

    assert hasattr(model, "top_features_")
    assert len(model.top_features_) == n_select


def test_backend_error_handling():
    """
    Tests that requesting the GPU backend without a GPU raises an error.
    """
    if not cuda.is_available():
        with pytest.raises(RuntimeError, match="no compatible GPU found"):
            model = FastReliefF(backend="gpu")
            model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    else:
        pytest.skip("Skipping GPU error test: GPU is available.")
