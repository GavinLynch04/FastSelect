import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from fast_select import ReliefF


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


# --- Core Logic and Behavior Tests ---

def test_feature_importance_ranking(simple_classification_data):
    """
    Tests if the algorithm correctly identifies relevant vs. irrelevant features.
    We don't check exact values, but we check the *ranking* of the importances.
    """
    X, y = simple_classification_data
    transformer = ReliefF(n_neighbors=1, n_features_to_select=2)
    transformer.fit(X, y)
    
    scores = transformer.feature_importances_
    
    # Assertions based on the designed data:
    # Feature 0 (relevant) should have a higher score than feature 1 (noise).
    assert scores[0] > scores[1]
    
    # Feature 2 (perfectly relevant) should have a higher score than feature 1 (noise).
    assert scores[2] > scores[1]
    
    # Feature 3 (zero range) should have an importance of 0, as it provides no information.
    assert_allclose(scores[3], 0.0)
    
    # The top 2 features selected should be the relevant ones (0 and 2).
    # We use a set to ignore the order.
    assert set(transformer.top_features_) == {0, 2}

def test_zero_range_feature_handling(simple_classification_data):
    """
    Explicitly test that a feature with zero variance has zero importance.
    """
    X, y = simple_classification_data
    transformer = ReliefF(n_neighbors=1, n_features_to_select=4)
    transformer.fit(X, y)
    
    # Feature 3 was designed to have a range of 0.
    assert_allclose(transformer.feature_importances_[3], 0.0)
    # This also implicitly tests that no division-by-zero errors occurred.


# --- Scikit-learn API and Parameter Tests ---

def test_sklearn_api_compliance():
    """
    Uses scikit-learn's built-in checker to validate the estimator's compliance.
    This runs a large suite of tests for things like get_params/set_params,
    clonability, and expected behavior on various inputs.
    """
    # check_estimator will instantiate the class, so we pass the class itself.
    check_estimator(ReliefF())

def test_fit_transform_output_shape(simple_classification_data):
    """
    Tests that the fit_transform method returns a matrix of the correct shape.
    """
    X, y = simple_classification_data
    k_select = 2
    transformer = ReliefF(n_features_to_select=k_select, n_neighbors=2)
    
    X_transformed = transformer.fit_transform(X, y)
    
    # Check that the number of samples is unchanged.
    assert X_transformed.shape[0] == X.shape[0]
    # Check that the number of features matches n_features_to_select.
    assert X_transformed.shape[1] == k_select

def test_n_neighbors_parameter(simple_classification_data):
    """
    Ensures the n_neighbors parameter is accepted and runs without error.
    """
    X, y = simple_classification_data
    # Test with k=2. The fixture has 3 samples per class, so this is valid.
    transformer = ReliefF(n_neighbors=2, n_features_to_select=2)
    transformer.fit(X, y)
    
    assert hasattr(transformer, 'feature_importances_')
    assert transformer.feature_importances_ is not None

def test_discrete_limit_parameter():
    """
    Tests that the discrete_limit parameter correctly classifies features.
    """
    # Feature 0 has 11 unique values. Feature 1 has 3.
    X = np.array([[i, i % 3] for i in range(11)] * 2)
    y = np.array([0]*11 + [1]*11)

    # With discrete_limit=10, feature 0 should be continuous.
    # The internal `is_discrete` array should be [False, True]
    rf_cont = ReliefF(discrete_limit=10, n_features_to_select=2, n_neighbors=1)
    rf_cont.fit(X, y)
    assert_array_equal(rf_cont.is_discrete_, [False, True])

    # With discrete_limit=12, both features should be discrete.
    # The internal `is_discrete` array should be [True, True]
    rf_disc = ReliefF(discrete_limit=12, n_features_to_select=2, n_neighbors=1)
    rf_disc.fit(X, y)
    assert_array_equal(rf_disc.is_discrete_, [True, True])


# --- Error Handling and Edge Case Tests ---

def test_not_fitted_error(simple_classification_data):
    """
    Tests that a NotFittedError is raised if transform is called before fit.
    """
    X, y = simple_classification_data
    transformer = ReliefF()
    
    with pytest.raises(NotFittedError):
        transformer.transform(X)

@pytest.mark.parametrize("bad_k", [-1, 0])
def test_invalid_n_neighbors_raises_error(simple_classification_data, bad_k):
    """
    Tests that an invalid n_neighbors value raises a ValueError.
    """
    X, y = simple_classification_data
    with pytest.raises(ValueError):
        ReliefF(n_neighbors=bad_k).fit(X, y)

@pytest.mark.parametrize("bad_k_select", [-1, 0, 100])
def test_invalid_n_features_to_select_raises_error(simple_classification_data, bad_k_select):
    """
    Tests that an invalid n_features_to_select value raises a ValueError.
    """
    X, y = simple_classification_data

    with pytest.raises(ValueError):
        ReliefF(n_features_to_select=bad_k_select).fit(X, y)

def test_transform_with_wrong_n_features(simple_classification_data):
    """
    Tests that transform raises a ValueError if X has a different number of
    features than the data used for fitting.
    """
    X, y = simple_classification_data
    transformer = ReliefF(n_features_to_select=4, n_neighbors=2).fit(X, y)
    
    # Create a new X with one fewer feature column
    X_bad_shape = X[:, :-1]
    
    with pytest.raises(ValueError):
        transformer.transform(X_bad_shape)

def test_insufficient_neighbors_in_class(simple_classification_data):
    """
    Tests that a UserWarning is raised when k is larger than the number of
    available samples in a class.
    """
    X, y = simple_classification_data
    # n_neighbors=5 is > (samples_in_class - 1) which is 3-1=2
    transformer = ReliefF(n_neighbors=5)
    with pytest.warns(UserWarning, match="is greater than or equal to the smallest class size"):
        transformer.fit(X, y)

def test_single_class_input(simple_classification_data):
    """
    Tests ReliefF with single-class data. Scores should be <= 0,
    as there are no "misses" to provide positive updates.
    """
    X, _ = simple_classification_data
    y_single_class = np.zeros(X.shape[0])

    model = ReliefF(backend="cpu", n_neighbors=2)
    model.fit(X, y_single_class)

    # The test is that it runs and produces finite, non-positive scores.
    assert np.all(np.isfinite(model.feature_importances_))
    assert np.all(model.feature_importances_ <= 0)
