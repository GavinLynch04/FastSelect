"""Independent empirical-distribution oracles for Shannon MI and Hall's CFS SU."""

from collections import Counter
from math import log

import numba
import numpy as np
import pytest

from fast_select.CFS import _precompute_correlations_cpu
from fast_select.mutual_information import calculate_mi_matrices, calculate_mi_single_pair
from fast_select.utils import is_cuda_ready


def _information_oracle(left, right, unit="bit"):
    sample_count = len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    joint_counts = Counter(zip(left, right))
    log_base = log(2) if unit == "bit" else 1.0
    return (
        sum(
            count / sample_count * log(count * sample_count / (left_counts[left_value] * right_counts[right_value]))
            for (left_value, right_value), count in joint_counts.items()
        )
        / log_base
    )


def _entropy_oracle(values):
    return -sum(count / len(values) * log(count / len(values), 2) for count in Counter(values).values())


def _uncertainty_oracle(left, right):
    entropy_sum = _entropy_oracle(left) + _entropy_oracle(right)
    return 2 * _information_oracle(left, right) / entropy_sum if entropy_sum else 0.0


def _discrete_case(states, layout):
    rng = np.random.default_rng(9351)
    features = rng.integers(0, states, size=(57, 11), dtype=np.int32)
    target = np.repeat([0, 2, 5], [41, 11, 5]).astype(np.int32)
    features[:, 0] = target
    features[:, 1] = 0
    features[:, 2] = features[:, 3]
    if layout == "fortran":
        features = np.asfortranarray(features)
    elif layout == "strided":
        backing = np.zeros((114, 22), dtype=np.int64)
        backing[::2, ::2] = features
        features = backing[::2, ::2]
        target = target[::-1]
    return features, target


@pytest.mark.parametrize("unit", ["bit", "nat"])
@pytest.mark.parametrize("states", [2, 7, 37])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_mi_matrices_match_empirical_distribution(unit, states, layout):
    features, target = _discrete_case(states, layout)
    original = features.copy()
    relevance, redundancy = calculate_mi_matrices(features, target, backend="cpu", unit=unit)
    expected_relevance = [_information_oracle(column, target, unit) for column in features.T]
    expected_redundancy = np.zeros_like(redundancy)
    for left in range(features.shape[1]):
        for right in range(left + 1, features.shape[1]):
            expected_redundancy[left, right] = _information_oracle(features[:, left], features[:, right], unit)
            expected_redundancy[right, left] = expected_redundancy[left, right]
    np.testing.assert_allclose(relevance, expected_relevance, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(redundancy, expected_redundancy, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(features, original)


@pytest.mark.parametrize("sample_count", [1, 2, 51])
@pytest.mark.parametrize("feature_count", [1, 2, 9, 10])
def test_mi_constant_and_small_shapes(sample_count, feature_count):
    features = np.zeros((sample_count, feature_count), dtype=np.int32)
    target = np.zeros(sample_count, dtype=np.int32)
    relevance, redundancy = calculate_mi_matrices(features, target, backend="cpu")
    np.testing.assert_array_equal(relevance, np.zeros(feature_count))
    np.testing.assert_array_equal(redundancy, np.zeros((feature_count, feature_count)))
    assert calculate_mi_single_pair(features[:, 0], target, backend="cpu") == 0.0


@pytest.mark.parametrize("states", [2, 7, 37])
def test_cached_cfs_entropies_match_hall_uncertainty(states):
    features, target = _discrete_case(states, "c")
    columns = np.ascontiguousarray(features.T)
    state_counts = columns.max(axis=1) + 1
    relevance, redundancy = _precompute_correlations_cpu(columns, target, state_counts, int(target.max()) + 1)
    expected_relevance = [_uncertainty_oracle(column, target) for column in columns]
    expected_redundancy = np.zeros_like(redundancy)
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            expected_redundancy[left, right] = _uncertainty_oracle(columns[left], columns[right])
            expected_redundancy[right, left] = expected_redundancy[left, right]
    np.testing.assert_allclose(relevance, expected_relevance, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(redundancy, expected_redundancy, atol=2e-7, rtol=2e-6)


def test_information_thread_counts_preserve_results():
    features, target = _discrete_case(7, "c")
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        expected = calculate_mi_matrices(features, target, backend="cpu")
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        actual = calculate_mi_matrices(features, target, backend="cpu")
    finally:
        numba.set_num_threads(original_threads)
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


@pytest.mark.gpu
@pytest.mark.skipif(
    numba.config.ENABLE_CUDASIM or not is_cuda_ready(),
    reason="A real CUDA GPU is required; do not enable the simulator.",
)
@pytest.mark.parametrize("unit", ["bit", "nat"])
def test_gpu_information_matches_distribution_oracle_and_cpu(unit):
    features, target = _discrete_case(7, "c")
    expected = calculate_mi_matrices(features, target, backend="cpu", unit=unit)
    actual = calculate_mi_matrices(features, target, backend="gpu", unit=unit)
    np.testing.assert_allclose(
        actual[0], [_information_oracle(column, target, unit) for column in features.T], atol=2e-6
    )
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_allclose(actual_array, expected_array, atol=2e-6, rtol=2e-5)
