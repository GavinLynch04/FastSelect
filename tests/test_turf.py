import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from fast_select.TuRF import TuRF

class MockReliefEstimator(BaseEstimator, TransformerMixin):
    """A mock estimator that creates predictable feature importances."""
    def fit(self, X, y=None):
        n_features = X.shape[1]
        self.feature_importances_ = np.linspace(1, 0, n_features)
        return self

    def transform(self, X):
        return X

@pytest.fixture
def test_data():
    """Provides consistent test data for all tests."""
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_sklearn_compatibility():
    """
    Uses scikit-learn's built-in checker to verify API compliance.
    This is a powerful test that covers dozens of common issues.
    """
    estimator = TuRF(estimator=MockReliefEstimator(), n_features_to_select=2)
    check_estimator(estimator)

def test_basic_fit_transform(test_data):
    """Test the basic fit and transform functionality."""
    X, y = test_data
    n_select = 5
    turf = TuRF(estimator=MockReliefEstimator(), n_features_to_select=n_select)

    turf.fit(X, y)
    Xt = turf.transform(X)

    assert turf.n_features_in_ == X.shape[1]
    assert hasattr(turf, "feature_importances_")
    assert hasattr(turf, "top_features_")
    assert len(turf.top_features_) == n_select
    assert Xt.shape == (X.shape[0], n_select)

    Xt_fit_transform = turf.fit_transform(X, y)
    np.testing.assert_array_equal(Xt, Xt_fit_transform)


def test_attributes_after_fit(test_data):
    """Verify that all public attributes are correctly set after fitting."""
    X, y = test_data
    n_select = 7
    turf = TuRF(estimator=MockReliefEstimator(), n_features_to_select=n_select)
    turf.fit(X, y)

    # n_features_in_
    assert turf.n_features_in_ == 20

    assert turf.feature_importances_.shape == (20,)
    assert turf.feature_importances_[0] > turf.feature_importances_[-1]

    assert turf.top_features_.shape == (n_select,)
    expected_top_features = np.arange(n_select)
    np.testing.assert_array_equal(turf.top_features_, expected_top_features)
    np.testing.assert_array_equal(turf.top_features_, np.sort(turf.top_features_))


def test_n_iterations_parameter(test_data):
    """Test that the n_iterations parameter correctly stops the process."""
    X, y = test_data
    turf = TuRF(
        estimator=MockReliefEstimator(),
        n_features_to_select=10,
        n_iterations=1,
        pct_remove=0.1
    )
    turf.fit(X, y)

    assert len(turf.top_features_) == 18
    assert len(turf.top_features_) > turf.n_features_to_select


def test_pct_remove_edge_case_removes_at_least_one(test_data):
    """Test that at least one feature is removed even with a tiny pct_remove."""
    X, y = test_data
    turf = TuRF(
        estimator=MockReliefEstimator(),
        n_features_to_select=1,
        pct_remove=0.001
    )
    turf.fit(X, y)

    assert len(turf.top_features_) == 1


def test_avoids_overshooting_n_features_to_select():
    """Test the logic that prevents removing too many features near the end."""
    X = np.random.rand(50, 11)
    y = np.random.randint(0, 2, 50)
    turf = TuRF(
        estimator=MockReliefEstimator(),
        n_features_to_select=10,
        pct_remove=0.2
    )
    turf.fit(X, y)

    assert len(turf.top_features_) == 10


def test_verbose_output(test_data, capsys):
    """Check that verbose=True prints to stdout."""
    X, y = test_data
    turf = TuRF(estimator=MockReliefEstimator(), n_features_to_select=15, verbose=True)
    turf.fit(X, y)

    captured = capsys.readouterr()
    assert "Iteration" in captured.out
    assert "features remaining" in captured.out


def test_fit_errors(test_data):
    """Test that fit raises errors on invalid parameters."""
    X, y = test_data
    with pytest.raises(ValueError, match="pct_remove must be between 0 and 1"):
        TuRF(estimator=MockReliefEstimator(), pct_remove=0).fit(X, y)
    with pytest.raises(ValueError, match="pct_remove must be between 0 and 1"):
        TuRF(estimator=MockReliefEstimator(), pct_remove=1).fit(X, y)
    with pytest.raises(ValueError, match="pct_remove must be between 0 and 1"):
        TuRF(estimator=MockReliefEstimator(), pct_remove=1.1).fit(X, y)


def test_transform_before_fit(test_data):
    """Test that calling transform before fit raises a NotFittedError."""
    X, y = test_data
    turf = TuRF(estimator=MockReliefEstimator())
    with pytest.raises(NotFittedError):
        turf.transform(X)


def test_transform_incorrect_dimensions(test_data):
    """Test that transform raises an error if feature number mismatches."""
    X, y = test_data
    turf = TuRF(estimator=MockReliefEstimator(), n_features_to_select=5)
    turf.fit(X, y)

    X_wrong_dim = np.random.rand(10, X.shape[1] + 1)

    with pytest.raises(ValueError, match="X has 21 features, but TuRF is expecting 20 features as input."):
        turf.transform(X_wrong_dim)