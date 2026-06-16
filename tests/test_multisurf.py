import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numba import cuda
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from fast_select import MultiSURF as FastMultiSURF


@pytest.fixture
def simple_classification_data():
    """
    Creates a dataset where classes are distinct but have some overlap.
    - Feature 0: Highly relevant continuous.
    - Feature 1: Irrelevant noise.
    - Feature 2: Highly relevant but discrete.
    - Feature 3: Irrelevant constant.
    """
    X = np.array([
    # Class 0
    [1.1, 5.0, 10, 3.0],
    [1.2, 4.0, 10, 3.0],
    [2.3, 6.0, 10, 3.0],
    [2.5, 5.5, 10, 3.0],
    [1.5, 4.5, 20, 3.0],
    # Class 1
    [8.8, 5.0, 20, 3.0],
    [8.9, 4.0, 20, 3.0],
    [9.5, 6.0, 20, 3.0],
    [10.5, 4.5, 20, 3.0],
    [10.5, 4.5, 10, 3.0],
    ], dtype=np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    return X, y

def test_feature_importance_ranking(simple_classification_data):
    X, y = simple_classification_data
    model = FastMultiSURF(n_features_to_select=1, backend="cpu", discrete_limit=4, verbose=False)
    model.fit(X, y)
    scores = model.feature_importances_

    expected_top_features = {0}
    assert set(model.top_features_) == expected_top_features

    assert_allclose(scores[3], 0.0, atol=1e-7)

def _reference_multisurf_scores(X, y, use_star=False, discrete_limit=10):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    is_discrete = np.array([
        np.unique(X[:, f]).size <= discrete_limit for f in range(n_features)
    ])
    ranges = X.max(axis=0) - X.min(axis=0)
    ranges[ranges == 0] = 1.0
    recip = 1.0 / ranges
    scores = np.zeros(n_features, dtype=np.float64)

    for i in range(n_samples):
        dists = []
        for j in range(n_samples):
            if i == j:
                continue
            dist = 0.0
            for f in range(n_features):
                if is_discrete[f]:
                    dist += 1.0 if X[i, f] != X[j, f] else 0.0
                else:
                    dist += abs(X[i, f] - X[j, f]) * recip[f]
            dists.append((j, dist))

        dist_values = np.array([dist for _, dist in dists])
        mu = dist_values.mean()
        sigma = dist_values.std()
        near_thresh = mu - 0.5 * sigma
        far_thresh = mu + 0.5 * sigma

        near_hit = np.zeros(n_features)
        near_miss = np.zeros(n_features)
        far_hit = np.zeros(n_features)
        far_miss = np.zeros(n_features)
        n_near_hit = n_near_miss = n_far_hit = n_far_miss = 0

        for j, dist in dists:
            diffs = np.empty(n_features)
            for f in range(n_features):
                if is_discrete[f]:
                    diffs[f] = 1.0 if X[i, f] != X[j, f] else 0.0
                else:
                    diffs[f] = abs(X[i, f] - X[j, f]) * recip[f]
            is_hit = y[i] == y[j]
            if dist < near_thresh:
                if is_hit:
                    near_hit += diffs
                    n_near_hit += 1
                else:
                    near_miss += diffs
                    n_near_miss += 1
            elif use_star and dist > far_thresh:
                if is_hit:
                    far_hit += diffs
                    n_far_hit += 1
                else:
                    far_miss += diffs
                    n_far_miss += 1

        if n_near_hit:
            scores -= near_hit / n_near_hit
        if n_near_miss:
            scores += near_miss / n_near_miss
        if use_star:
            if n_far_hit:
                scores += far_hit / n_far_hit
            if n_far_miss:
                scores -= far_miss / n_far_miss

    return scores / n_samples


@pytest.mark.parametrize("use_star", [False, True])
def test_multisurf_matches_reference(use_star):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10, 6)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)

    model = FastMultiSURF(
        n_features_to_select=2,
        backend="cpu",
        discrete_limit=0,
        use_star=use_star,
    ).fit(X, y)

    expected = _reference_multisurf_scores(
        X, y, use_star=use_star, discrete_limit=0
    )
    assert_allclose(model.feature_importances_, expected, rtol=1e-6, atol=1e-6)


def test_multisurf_star_uses_deadband_for_equal_distances():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10, 6)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)

    model = FastMultiSURF(
        n_features_to_select=2,
        backend="cpu",
        discrete_limit=10,
        use_star=True,
    ).fit(X, y)

    assert_allclose(model.feature_importances_, 0.0, atol=1e-7)


@pytest.mark.skipif(not cuda.is_available(), reason="NVIDIA GPU with CUDA not available")
@pytest.mark.parametrize("use_star", [False, True])
def test_gpu_vs_cpu_consistency(use_star):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10, 6)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)

    cpu_model = FastMultiSURF(
        n_features_to_select=3,
        backend="cpu",
        discrete_limit=0,
        use_star=use_star,
    ).fit(X, y)
    gpu_model = FastMultiSURF(
        n_features_to_select=3,
        backend="gpu",
        discrete_limit=0,
        use_star=use_star,
    ).fit(X, y)

    assert_allclose(
        gpu_model.feature_importances_,
        cpu_model.feature_importances_,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not cuda.is_available(), reason="NVIDIA GPU with CUDA not available")
def test_gpu_vs_cpu_consistency_across_feature_tiles():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(6, 70)).astype(np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    cpu_model = FastMultiSURF(
        n_features_to_select=3,
        backend="cpu",
        discrete_limit=0,
    ).fit(X, y)
    gpu_model = FastMultiSURF(
        n_features_to_select=3,
        backend="gpu",
        discrete_limit=0,
    ).fit(X, y)

    assert_allclose(
        gpu_model.feature_importances_,
        cpu_model.feature_importances_,
        rtol=1e-5,
        atol=1e-6,
    )


'''
@pytest.mark.gpu  # Custom mark to only run if a GPU is available
def test_gpu_vs_cpu_consistency(simple_classification_data):
    """
    Tests that the GPU and CPU backends produce approximately equal results.
    We expect small differences due to parallel summation order and float precision,
    so we use a tolerance for comparison.
    """
    X, y = simple_classification_data

    cpu_surf = FastMultiSURF(n_features_to_select=3, backend='cpu')
    cpu_surf.fit(X, y)
    cpu_scores = cpu_surf.feature_importances_

    try:
        gpu_surf = FastMultiSURF(n_features_to_select=3, backend='gpu')
        gpu_surf.fit(X, y)
        gpu_scores = gpu_surf.feature_importances_
    except RuntimeError as e:
        pytest.skip(f"Skipping GPU test: {e}")

    np.testing.assert_allclose(
        cpu_scores,
        gpu_scores,
        rtol=1e-5,
        atol=1e-8,
        err_msg="GPU and CPU feature scores diverged more than expected."
    )'''


def test_sklearn_api_compliance():
    """
    Uses scikit-learn's built-in checker to validate the estimator's compliance.
    This is a powerful test that checks for dozens of common API requirements.
    """
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
    model_cont = FastMultiSURF(discrete_limit=10, backend="cpu", n_features_to_select=2)
    model_cont.fit(X, y)
    assert_array_equal(model_cont.is_discrete_, [False, True])

    # With discrete_limit=12, both features should be considered discrete.
    model_disc = FastMultiSURF(discrete_limit=12, backend="cpu", n_features_to_select=2)
    model_disc.fit(X, y)
    assert_array_equal(model_disc.is_discrete_, [True, True])



def test_not_fitted_error(simple_classification_data):
    """Tests that a NotFittedError is raised if transform is called before fit."""
    X, _ = simple_classification_data
    model = FastMultiSURF()
    with pytest.raises(NotFittedError):
        model.transform(X)
        
@pytest.mark.parametrize("bad_k_select", [-1, 0, 100])
def test_invalid_n_features_to_select_raises_error(simple_classification_data, bad_k_select):
    """
    Tests that an invalid n_features_to_select value raises a ValueError.
    """
    X, y = simple_classification_data

    with pytest.raises(ValueError):
        FastMultiSURF(n_features_to_select=bad_k_select).fit(X, y)
    with pytest.raises(ValueError):
        FastMultiSURF(n_features_to_select=1.1).fit(X, y)
    with pytest.raises(TypeError):
        FastMultiSURF(n_features_to_select='hi').fit(X, y)
        
def test_verbose_output(simple_classification_data, capsys):
    """Check that verbose=True prints to stdout."""
    X, y = simple_classification_data
    model = FastMultiSURF(verbose=True)
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "Running MultiSURF" in captured.out
    
    model = FastMultiSURF(verbose=True, use_star=True)
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "Running MultiSURF*" in captured.out
    model = FastMultiSURF(verbose=True, backend='cpu')
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "Running MultiSURF" in captured.out
    
    model = FastMultiSURF(verbose=True, use_star=True, backend='cpu')
    model.fit(X, y)

    captured = capsys.readouterr()
    assert "Running MultiSURF*" in captured.out
        
def test_backend(simple_classification_data):
    """
    Tests that transform raises a ValueError if backend is not auto, cpu, or gpu
    """
    X, y = simple_classification_data
        
    with pytest.raises(ValueError):
        transformer = FastMultiSURF(n_features_to_select=4, backend='tpu').fit(X, y)

def test_backend_error_handling(simple_classification_data):
    """Tests that requesting the GPU backend without a GPU raises a RuntimeError."""
    if cuda.is_available():
        pytest.skip("Skipping GPU error test: GPU is available.")
    
    X, y = simple_classification_data
    with pytest.raises(RuntimeError, match="no compatible NVIDIA GPU"):
        model = FastMultiSURF(backend="gpu", n_features_to_select=2)
        model.fit(X, y)


def test_nan_input_raises_error(simple_classification_data):
    """Tests that the estimator raises a ValueError for data containing NaNs."""
    X_orig, y = simple_classification_data

    X = X_orig.copy()
    X[0, 0] = np.nan

    model = FastMultiSURF(backend="cpu", n_features_to_select=2)
    with pytest.raises(ValueError, match="Input X contains NaN"):
        model.fit(X, y)


def test_single_class_input(simple_classification_data):
    """
    Tests behavior with only one class label. Scores should be less than zero as there
    are no "misses" to learn from. A negative score is the expected penalty for intra-class variation.
    """
    X, _ = simple_classification_data
    y_single_class = np.zeros(X.shape[0])
    
    model = FastMultiSURF(backend="cpu", n_features_to_select=4)
    model.fit(X, y_single_class)
    
    # With no misses, all feature importances should be less than zero.
    assert np.all(model.feature_importances_ <= 1e-7)
