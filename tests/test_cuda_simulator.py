"""CUDA-kernel conformance on small inputs.

These tests run on a physical NVIDIA GPU *or* under ``NUMBA_ENABLE_CUDASIM=1``.
The simulator executes the kernels' own Python source, so it validates indexing,
reductions, inequalities and host-side plumbing, but it cannot reveal driver
problems, data races or real-device rounding.  Sizes are kept tiny because the
simulator runs one Python thread per CUDA thread.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fast_select import CFS, MDR, SURF, MultiSURF, ReliefF, calculate_mi_matrices, calculate_mi_single_pair, mRMR
from fast_select.utils import is_cuda_ready

pytestmark = pytest.mark.skipif(not is_cuda_ready(), reason="NVIDIA GPU with CUDA (or the CUDA simulator) required")

VARIANTS = [
    (ReliefF, {}),
    (SURF, {}),
    (SURF, {"use_star": True}),
    (MultiSURF, {}),
    (MultiSURF, {"use_star": True}),
]


def mixed_data(n=14, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.uniform(0, 5, n),  # continuous
            rng.integers(0, 3, n),  # discrete
            rng.uniform(-1, 1, n),  # continuous
            rng.integers(0, 2, n),  # discrete
        ]
    ).astype(np.float64)
    y = (X[:, 1] + (X[:, 0] > 2.5)) % 2
    y[:2] = [0, 1]
    return X, y.astype(int)


def kwargs(cls, extra, backend):
    kw = dict(n_features_to_select=2, backend=backend, discrete_limit=3, n_jobs=1, **extra)
    if cls is ReliefF:
        kw["n_neighbors"] = 2
    return kw


@pytest.mark.parametrize("cls,extra", VARIANTS)
def test_relief_gpu_matches_cpu_on_mixed_data(cls, extra):
    X, y = mixed_data()
    cpu = cls(**kwargs(cls, extra, "cpu")).fit(X, y).feature_importances_
    gpu = cls(**kwargs(cls, extra, "gpu")).fit(X, y).feature_importances_
    assert_allclose(gpu, cpu, atol=1e-5)


@pytest.mark.parametrize("cls,extra", VARIANTS)
def test_relief_gpu_is_translation_and_category_invariant(cls, extra):
    X, y = mixed_data(seed=3)
    base = cls(**kwargs(cls, extra, "gpu")).fit(X, y).feature_importances_
    shifted = X.copy()
    shifted[:, [0, 2]] += 1e8  # continuous columns
    shifted[:, [1, 3]] += 2.0**25  # discrete category codes
    moved = cls(**kwargs(cls, extra, "gpu")).fit(shifted, y).feature_importances_
    assert_allclose(moved, base, atol=1e-5)


@pytest.mark.parametrize("cls", [SURF, MultiSURF])
def test_wide_feature_block_uses_global_atomic_kernel(cls):
    """More than SHARED_SCORE_MAX_FEATURES columns selects the non-shared scoring kernel."""
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(8, 513))
    y = np.array([0, 1] * 4)
    kw = dict(n_features_to_select=3, discrete_limit=0, n_jobs=1, use_star=True)
    cpu = cls(backend="cpu", **kw).fit(X, y).feature_importances_
    gpu = cls(backend="gpu", **kw).fit(X, y).feature_importances_
    assert_allclose(gpu, cpu, atol=1e-5)


def test_multisurf_gpu_feat_idx_matches_cpu_and_stays_in_subset():
    X, y = mixed_data(seed=4)
    kw = dict(n_features_to_select=1, discrete_limit=3, n_jobs=1)
    cpu = MultiSURF(backend="cpu", **kw).fit(X, y, feat_idx=[3, 0])
    gpu = MultiSURF(backend="gpu", **kw).fit(X, y, feat_idx=[3, 0])
    assert_allclose(gpu.feature_importances_, cpu.feature_importances_, atol=1e-5)
    assert set(gpu.top_features_) <= {0, 3}


def test_mutual_information_gpu_matches_cpu_with_sparse_codes():
    rng = np.random.default_rng(6)
    X = rng.choice(np.array([0, 7, 2**40], dtype=np.uint64), size=(40, 4))
    y = rng.integers(0, 3, 40)
    for unit in ("bit", "nat"):
        cpu = calculate_mi_matrices(X, y, backend="cpu", unit=unit)
        gpu = calculate_mi_matrices(X, y, backend="gpu", unit=unit)
        assert_allclose(gpu[0], cpu[0], atol=1e-5)
        assert_allclose(gpu[1], cpu[1], atol=1e-5)
        assert_allclose(
            calculate_mi_single_pair(X[:, 0], y, backend="gpu", unit=unit),
            calculate_mi_single_pair(X[:, 0], y, backend="cpu", unit=unit),
            atol=1e-5,
        )


def test_mrmr_gpu_accepts_float_coded_categories():
    rng = np.random.default_rng(8)
    X = rng.integers(0, 3, (40, 5)).astype(float)
    y = (X[:, 0] > 0).astype(float)
    cpu = mRMR(3, backend="cpu").fit(X, y)
    gpu = mRMR(3, backend="gpu").fit(X, y)
    assert_allclose(gpu.relevance_scores_, cpu.relevance_scores_, atol=1e-5)
    assert_array_equal(gpu.top_features_, cpu.top_features_)


def test_cfs_gpu_matches_cpu():
    rng = np.random.default_rng(9)
    X = rng.integers(0, 3, (40, 5))
    y = ((X[:, 0] + X[:, 2]) % 2).astype(int)
    cpu = CFS(backend="cpu", n_jobs=1).fit(X, y)
    gpu = CFS(backend="gpu").fit(X, y)
    assert_array_equal(gpu.selected_indices_, cpu.selected_indices_)
    assert_allclose(gpu.merit_, cpu.merit_, atol=1e-5)


def test_mdr_gpu_matches_cpu_and_uppercase_backend():
    rng = np.random.default_rng(10)
    X = rng.integers(0, 3, (36, 4))
    y = np.array([0] * 18 + [1] * 18)
    cpu = MDR(k=2, cv=3, backend="cpu").fit(X, y)
    gpu = MDR(k=2, cv=3, backend="GPU").fit(X, y)
    assert tuple(gpu.best_interaction_) == tuple(cpu.best_interaction_)
    assert_array_equal(gpu.predict(X), cpu.predict(X))
