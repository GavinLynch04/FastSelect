import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fast_select.mRMR import _encode_data_numba, mRMR
from fast_select.mutual_information import calculate_mi_matrices
from fast_select.utils import is_cuda_ready


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


def test_parameter_validation_happens_in_fit(discrete_classification_data):
    """Invalid parameters must be reported by fit, not by the constructor.

    scikit-learn requires __init__ to store its arguments unmodified and
    without side effects so that get_params/set_params/clone round-trip.
    """
    X, y = discrete_classification_data

    # Constructing with invalid values must succeed and round-trip.
    bad_method = mRMR(n_features_to_select=5, method="INVALID_METHOD")
    assert bad_method.get_params()["method"] == "INVALID_METHOD"
    with pytest.raises(ValueError, match="Method must be either 'MID' or 'MIQ'"):
        bad_method.fit(X, y)

    bad_backend = mRMR(n_features_to_select=5, backend="tpu")
    assert bad_backend.get_params()["backend"] == "tpu"
    with pytest.raises(ValueError, match="Backend must be 'auto', 'cpu', or 'gpu'"):
        bad_backend.fit(X, y)


def test_default_backend_is_auto_and_matches_explicit_backends(discrete_classification_data):
    """'auto' is the default and must agree with whichever backend it picks."""
    X, y = discrete_classification_data
    assert mRMR(n_features_to_select=5).get_params()["backend"] == "auto"

    auto = mRMR(n_features_to_select=5).fit(X, y)
    assert auto.effective_backend_ in ("cpu", "gpu")
    assert auto.effective_backend_ == ("gpu" if is_cuda_ready() else "cpu")

    explicit = mRMR(n_features_to_select=5, backend=auto.effective_backend_).fit(X, y)
    np.testing.assert_array_equal(auto.top_features_, explicit.top_features_)
    np.testing.assert_allclose(auto.relevance_scores_, explicit.relevance_scores_, rtol=1e-12, atol=1e-12)


def test_auto_backend_falls_back_to_cpu_when_states_exceed_gpu_limit():
    """More than 32 distinct states must fall back rather than raise."""
    from fast_select.mutual_information import _MAX_STATES_GPU

    rng = np.random.default_rng(11)
    X = rng.integers(0, _MAX_STATES_GPU + 8, size=(120, 6), dtype=np.int64)
    y = rng.integers(0, 3, size=120, dtype=np.int64)

    model = mRMR(n_features_to_select=3, backend="auto").fit(X, y)
    assert model.effective_backend_ == "cpu"

    # An explicit 'gpu' must refuse instead of silently falling back.
    if is_cuda_ready():
        with pytest.raises(RuntimeError, match="supports at most"):
            mRMR(n_features_to_select=3, backend="gpu").fit(X, y)


def test_init_stores_parameters_unmodified_and_clones():
    """__init__ must not mutate its arguments; clone must reproduce them."""
    from sklearn.base import clone

    estimator = mRMR(n_features_to_select=7, method="MIQ", backend="gpu")
    params = estimator.get_params()
    assert params == {
        "n_features_to_select": 7,
        "method": "MIQ",
        "backend": "gpu",
    }
    # clone must work even for a backend this machine cannot run.
    assert clone(estimator).get_params() == params


@pytest.mark.skipif(is_cuda_ready(), reason="This test is for when CUDA is NOT available")
def test_gpu_backend_fails_without_cuda(discrete_classification_data):
    """Selecting the 'gpu' backend fails gracefully at fit time if CUDA is absent."""
    X, y = discrete_classification_data
    with pytest.raises(RuntimeError, match="Numba could not find a usable CUDA installation"):
        mRMR(n_features_to_select=5, backend="gpu").fit(X, y)


@pytest.mark.parametrize("method", ["MID", "MIQ"])
def test_fit_transform_cpu(discrete_classification_data, method):
    """
    Test the full fit and transform cycle on the CPU backend for both methods.
    This covers the CPU host caller and JIT kernels.
    """
    X, y = discrete_classification_data
    n_samples, n_features = X.shape
    n_select = 5

    model = mRMR(n_features_to_select=n_select, method=method, backend="cpu")

    # Test fit()
    model.fit(X, y)
    assert hasattr(model, "top_features_")
    assert hasattr(model, "relevance_scores_")
    assert hasattr(model, "redundancy_matrix_")
    assert model.top_features_.shape == (n_select,)
    assert model.relevance_scores_.shape == (n_features,)
    assert model.redundancy_matrix_.shape == (n_features, n_features)

    # Test transform()
    X_transformed = model.transform(X)
    assert X_transformed.shape == (n_samples, n_select)

    model2 = mRMR(n_features_to_select=n_select, method=method, backend="cpu")
    X_ft = model2.fit_transform(X, y)
    assert X_ft.shape == (n_samples, n_select)


@pytest.mark.skipif(not is_cuda_ready(), reason="NVIDIA GPU with CUDA not available")
@pytest.mark.parametrize("method", ["MID", "MIQ"])
def test_fit_transform_gpu(discrete_classification_data, method):
    """
    Test the full fit and transform cycle on the GPU backend for both methods.
    This covers the GPU host caller and CUDA kernels. Skipped if no GPU.
    """
    X, y = discrete_classification_data
    n_samples, n_features = X.shape
    n_select = 5

    model = mRMR(n_features_to_select=n_select, method=method, backend="gpu")

    # Test fit()
    model.fit(X, y)
    assert hasattr(model, "top_features_")
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
    rng = np.random.default_rng(42)  # reproducible across runs

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

    assert selected == expected, f"Expected features {expected}, but got {selected}"


def test_sklearn_pipeline_compatibility(discrete_classification_data):
    """Ensures that mRMR works as a step within a scikit-learn Pipeline."""
    X, y = discrete_classification_data
    n_select = 3

    pipeline = Pipeline(
        [
            ("mrmr_selector", mRMR(n_features_to_select=n_select, backend="cpu")),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    # If this runs without errors, the pipeline compatibility is confirmed
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_input_validation_errors(discrete_classification_data):
    """Test for errors raised on invalid input shapes or incorrect usage."""
    X, y = discrete_classification_data
    n_features = X.shape[1]

    model = mRMR(n_features_to_select=5, backend="cpu")

    # 1. Calling transform before fit should raise NotFittedError
    with pytest.raises(NotFittedError):
        model.transform(X)

    # 2. n_features_to_select > n_features should raise ValueError
    bad_model = mRMR(n_features_to_select=n_features + 1, backend="cpu")
    with pytest.raises(ValueError, match="n_features_to_select must be a positive integer"):
        bad_model.fit(X, y)

    # 3. Calling transform with X of a different shape should raise ValueError
    model.fit(X, y)
    X_wrong_shape = np.delete(X, 0, axis=1)  # Remove one feature
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
    assert X_encoded.dtype == X.dtype


@pytest.mark.skipif(not is_cuda_ready(), reason="NVIDIA GPU with CUDA not available")
@pytest.mark.parametrize("unit", ["bit", "nat"])
def test_mutual_information_gpu_matches_cpu(unit):
    rng = np.random.default_rng(31)
    X = rng.integers(0, 5, size=(513, 12), dtype=np.int32)
    y = rng.integers(0, 4, size=513, dtype=np.int32)

    cpu_relevance, cpu_redundancy = calculate_mi_matrices(X, y, backend="cpu", unit=unit)
    gpu_relevance, gpu_redundancy = calculate_mi_matrices(X, y, backend="gpu", unit=unit)

    np.testing.assert_allclose(gpu_relevance, cpu_relevance, rtol=1e-5, atol=1e-6)
    # The redundancy matrix is now computed by its own CUDA kernel rather than
    # handed back from the CPU path, so it carries genuine float32 reduction
    # error and takes the same tolerance as the relevance vector.  A tighter
    # bound here would only be asserting that one array was copied from another.
    np.testing.assert_allclose(gpu_redundancy, cpu_redundancy, rtol=1e-5, atol=1e-6)

    # Structural guarantees the CPU path provides and the kernel must preserve.
    np.testing.assert_array_equal(gpu_redundancy, gpu_redundancy.T)
    np.testing.assert_array_equal(np.diag(gpu_redundancy), np.zeros(X.shape[1]))
