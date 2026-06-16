import numpy as np
import pytest
from numba import cuda
from numpy.testing import assert_allclose

from fast_select import mutual_information as mi


def test_calculate_mi_single_pair_matches_known_value():
    x = np.array([0, 0, 1, 1], dtype=np.int32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    assert_allclose(mi.calculate_mi_single_pair(x, y, backend="cpu"), 1.0)


@pytest.mark.skipif(not cuda.is_available(), reason="NVIDIA GPU with CUDA not available")
@pytest.mark.parametrize("unit", ["bit", "nat"])
def test_gpu_relevance_matches_cpu(unit):
    X = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [2, 0, 1],
            [2, 1, 1],
        ],
        dtype=np.int32,
    )
    y = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)

    cpu_relevance, _ = mi.calculate_mi_matrices(X, y, backend="cpu", unit=unit)
    gpu_relevance, _ = mi.calculate_mi_matrices(X, y, backend="gpu", unit=unit)

    assert_allclose(gpu_relevance, cpu_relevance, rtol=1e-6, atol=1e-6)
