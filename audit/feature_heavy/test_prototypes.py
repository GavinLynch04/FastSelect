"""Independent oracles for physical Metal or CUDA/simulator research kernels."""

import os

import numpy as np
import pytest
from numba import config

from fast_select import CIFE, DISR, MIFS, GiniIndex, mutual_info_regression
from tests.test_more_algorithms import gini_oracle, information_oracle, regression_mi_oracle

from . import common

if os.environ.get("FEATURE_GPU_TEST_BACKEND", "metal") == "cuda":
    from . import cuda as provider
else:
    from . import metal as provider


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64])
def test_sort_and_gini_identity(dtype):
    X = np.array(
        [[0, 1, 2, 3], [1, 0, 3, 2], [1, 1, 0, 0], [2, 2, 1, 1], [3, 1, 3, 2], [2, 3, 2, 3], [1, 0, 1, 2]], dtype=dtype
    )
    y = np.array([-7, -7, 13, 13, 19, 19, -7])
    if dtype == np.uint64:
        X += np.uint64(2**63)
    if dtype == np.float64:
        X[:, 0] = [1, 1 + 1e-9, 1 + 2e-9, 1 + 3e-9, 1 + 4e-9, 1 + 5e-9, 1 + 6e-9]
        X[:, 1] -= 1e8
    result = common.fit_gini(X, y, provider, 3)
    np.testing.assert_allclose(result["scores"], gini_oracle(X, y), atol=3e-16)
    cpu = GiniIndex(3).fit(X, y)
    np.testing.assert_array_equal(result["selected"], cpu.top_features_)
    np.testing.assert_array_equal(result["scores"], cpu.gini_scores_)


@pytest.mark.parametrize("criterion", ["CIFE", "DISR", "MIFS"])
def test_histogram_first_occurrence_and_criterion(criterion):
    rng = np.random.default_rng(8)
    X = rng.integers(0, 3, (13, 5))
    y = np.array([-7] * 8 + [15] * 4 + [97])
    result = common.fit_joint(X, y, criterion, provider, 3)
    selected, scores, rel = information_oracle(X, y, criterion, 3)
    np.testing.assert_array_equal(result["selected"], selected)
    np.testing.assert_allclose(result["scores"], scores, atol=4e-14)
    np.testing.assert_allclose(result["relevance_all"], rel, atol=4e-14)
    cpu = {"CIFE": CIFE, "DISR": DISR, "MIFS": MIFS}[criterion](3).fit(X, y)
    np.testing.assert_array_equal(result["selected"], cpu.top_features_)
    np.testing.assert_allclose(result["scores"], cpu.selection_scores_, atol=4e-14)


@pytest.mark.parametrize("criterion", ["CIFE", "DISR", "MIFS"])
def test_histogram_exact_ties_constant_labels(criterion):
    X = np.tile([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], (2, 1))
    y = X[:, 2]
    for target in [y, np.zeros(len(y), int)]:
        result = common.fit_joint(X, target, criterion, provider, 3)
        selected, scores, _ = information_oracle(X, target, criterion, 3)
        np.testing.assert_array_equal(result["selected"], selected)
        np.testing.assert_allclose(result["scores"], scores, atol=4e-14)


@pytest.mark.parametrize("k", [1, 3])
def test_ksg_float64_oracle(k):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(9, 4))
    y = X[:, 0] ** 2 + rng.normal(size=9)
    result = common.fit_regression(X, y, provider, k=k, seed=7)
    expected = regression_mi_oracle(X, y, np.zeros(4, bool), k, 7)
    np.testing.assert_allclose(result["scores"], expected, atol=3e-14)
    np.testing.assert_allclose(
        result["scores"], mutual_info_regression(X, y, n_neighbors=k, random_state=7), atol=3e-14
    )


def test_ksg_ambiguous_boundaries_and_discrete_columns():
    X = np.tile([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], (2, 1))
    y = np.arange(8.0) % 3
    result = common.fit_regression(X, y, provider, k=3, seed=7)
    expected = regression_mi_oracle(X, y, np.zeros(3, bool), 3, 7)
    np.testing.assert_allclose(result["scores"], expected, atol=3e-14)
    if not config.ENABLE_CUDASIM and provider.__name__.endswith("metal"):
        assert result["diagnostics"]["float64_fallback_queries"] > 0
    result = common.fit_regression(X, y, provider, k=3, seed=7, discrete=[True, False, False])
    expected = regression_mi_oracle(X, y, np.array([True, False, False]), 3, 7)
    np.testing.assert_allclose(result["scores"], expected, atol=3e-14)
