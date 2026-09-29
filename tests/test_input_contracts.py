"""Input validation, parameter hardening and runtime-environment behaviour."""

import os
import subprocess
import sys

import numpy as np
import pytest
from numba import config
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from fast_select import (
    CFS,
    MDR,
    SURF,
    MultiSURF,
    ReliefF,
    calculate_mi_matrices,
    calculate_mi_single_pair,
    mRMR,
)
from fast_select.utils import resolve_num_threads

RELIEF = [ReliefF, SURF, MultiSURF]


def _data(n=30, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, (n, p)).astype(float)
    y = (X[:, 0] > 0).astype(int)
    if len(np.unique(y)) < 2:
        y[0] = 1 - y[0]
    return X, y


def _kw(cls, **extra):
    kw = dict(backend="cpu", n_jobs=1)
    if cls is ReliefF:
        kw["n_neighbors"] = 2
    kw.update(extra)
    return kw


# --------------------------------------------------------------------------- Relief family


@pytest.mark.parametrize("cls", RELIEF)
def test_relief_rejects_continuous_targets(cls):
    X, _ = _data()
    y = np.random.default_rng(0).uniform(size=X.shape[0])
    with pytest.raises(ValueError, match="Unknown label type"):
        cls(1, **_kw(cls)).fit(X, y)


@pytest.mark.parametrize("cls", RELIEF)
def test_relief_rejects_empty_input(cls):
    with pytest.raises(ValueError):
        cls(1, **_kw(cls)).fit(np.empty((0, 3)), np.empty(0))


@pytest.mark.parametrize("cls", RELIEF)
@pytest.mark.parametrize("bad", [True, 2.5, "3"])
def test_relief_rejects_noninteger_discrete_limit_and_selection(cls, bad):
    X, y = _data()
    with pytest.raises(TypeError):
        cls(1, **_kw(cls, discrete_limit=bad)).fit(X, y)
    if not isinstance(bad, float):  # a float selection count is a legitimate fraction
        with pytest.raises(TypeError):
            cls(bad, **_kw(cls)).fit(X, y)


@pytest.mark.parametrize("bad", [2.5, True, "3", 0, -1])
def test_relieff_rejects_invalid_neighbor_counts(bad):
    X, y = _data()
    with pytest.raises((TypeError, ValueError)):
        ReliefF(1, backend="cpu", n_neighbors=bad, n_jobs=1).fit(X, y)


@pytest.mark.parametrize("cls", RELIEF)
@pytest.mark.parametrize("bad", [0, -2, 10_000, 1.5, True])
def test_relief_rejects_invalid_n_jobs(cls, bad):
    X, y = _data()
    with pytest.raises((TypeError, ValueError)):
        cls(1, **_kw(cls, n_jobs=bad)).fit(X, y)


@pytest.mark.parametrize(
    "bad,error",
    [
        (np.zeros((2, 2), dtype=int), ValueError),  # not 1-D
        ([], ValueError),  # empty
        ([0.0, 1.0], TypeError),  # not integer
        ([True, False], TypeError),  # boolean mask
        ([-1, 0], ValueError),  # negative
        ([0, 4], ValueError),  # out of range
        ([1, 1], ValueError),  # duplicate
    ],
)
def test_multisurf_rejects_invalid_feat_idx(bad, error):
    X, y = _data()
    with pytest.raises(error):
        MultiSURF(1, **_kw(MultiSURF)).fit(X, y, feat_idx=bad)


def test_multisurf_feat_idx_semantics():
    X, y = _data(40, 5, seed=2)
    subset = np.array([3, 1, 4])
    model = MultiSURF(2, **_kw(MultiSURF)).fit(X, y, feat_idx=subset)
    reference = MultiSURF(2, **_kw(MultiSURF)).fit(X[:, subset], y)
    # Scores and ranking equal fitting on the sub-matrix; unevaluated features are zero.
    assert_allclose(model.feature_importances_[subset], reference.feature_importances_)
    assert np.all(model.feature_importances_[[0, 2]] == 0)
    assert_allclose(model.top_features_, subset[reference.top_features_])
    with pytest.raises(ValueError, match="n_features_to_select"):
        MultiSURF(4, **_kw(MultiSURF)).fit(X, y, feat_idx=subset)


# ----------------------------------------------------------------- mutual information / mRMR


def test_mutual_information_validation():
    with pytest.raises(ValueError, match="empty"):
        calculate_mi_single_pair(np.empty(0, dtype=int), np.empty(0, dtype=int), backend="cpu")
    with pytest.raises(ValueError, match="negative"):
        calculate_mi_single_pair(np.array([0, -1]), np.array([0, 1]), backend="cpu")
    with pytest.raises(ValueError, match="integer-coded"):
        calculate_mi_single_pair(np.array([0.0, 1.0]), np.array([0, 1]), backend="cpu")


def test_mutual_information_sparse_large_codes_are_remapped_per_column():
    big = 2**40
    X = np.array([[0, 5], [big, 5], [0, 7], [big, 7]] * 5, dtype=np.uint64)
    y = np.array([0, 1, 0, 1] * 5)
    relevance, redundancy = calculate_mi_matrices(X, y, backend="cpu")
    assert_allclose(relevance, [1.0, 0.0], atol=1e-12)
    assert_allclose(redundancy[0, 1], 0.0, atol=1e-12)
    assert_allclose(calculate_mi_single_pair(X[:, 0], y, backend="cpu", unit="nat"), np.log(2.0))


def test_mrmr_accepts_string_categories_and_rejects_bad_selection_counts():
    X = np.array([["a", "x"], ["a", "y"], ["b", "x"], ["b", "y"]] * 4)
    y = np.array([0, 0, 1, 1] * 4)
    model = mRMR(1, backend="cpu").fit(X, y)
    assert_allclose(model.relevance_scores_, [1.0, 0.0], atol=1e-12)
    assert model.top_features_.tolist() == [0]
    for bad in (0, 3, 1.5, True):
        with pytest.raises((TypeError, ValueError)):
            mRMR(bad, backend="cpu").fit(X, y)


# ------------------------------------------------------------------------------------ MDR


def _genotypes():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 4)
    y = np.array([0, 0, 1, 1] * 4)
    return X, y


@pytest.mark.parametrize("bad", [1.5, True, "2"])
def test_mdr_rejects_noninteger_k(bad):
    X, y = _genotypes()
    with pytest.raises(TypeError):
        MDR(k=bad, cv=2, backend="cpu").fit(X, y)


def test_mdr_rejects_bad_cv_backend_and_multiclass():
    X, y = _genotypes()
    with pytest.raises(ValueError):
        MDR(k=1, cv=1, backend="cpu").fit(X, y)
    with pytest.raises(ValueError, match="backend"):
        MDR(k=1, cv=2, backend="tpu").fit(X, y)
    with pytest.raises(ValueError, match="binary"):
        MDR(k=1, cv=2, backend="cpu").fit(X, np.arange(len(y)) % 3)
    with pytest.raises(ValueError, match="Unknown label type"):
        MDR(k=1, cv=2, backend="cpu").fit(X, np.linspace(0, 1, len(y)))


def test_mdr_backend_is_case_insensitive_and_unmodified():
    X, y = _genotypes()
    model = MDR(k=1, cv=2, backend="CPU").fit(X, y)
    assert model.backend == "CPU"
    assert model.predict(X).shape == (len(y),)
    assert model.transform(X).shape == (len(y), 1)


def test_mdr_predict_proba_requires_fit_then_is_unsupported():
    X, y = _genotypes()
    with pytest.raises(NotFittedError):
        MDR(backend="cpu").predict_proba(X)
    with pytest.raises(NotImplementedError):
        MDR(k=1, cv=2, backend="cpu").fit(X, y).predict_proba(X)


# ------------------------------------------------------------------------------------ CFS


@pytest.mark.parametrize("kwargs", [{"n_jobs": 0}, {"n_jobs": 10_000}, {"n_bins": 1}, {"max_backtracks": 0}])
def test_cfs_rejects_invalid_parameters(kwargs):
    X, y = _data()
    with pytest.raises(ValueError):
        CFS(backend="cpu", **kwargs).fit(X, y)


def test_cfs_rejects_continuous_targets():
    X, _ = _data()
    with pytest.raises(ValueError, match="Unknown label type"):
        CFS(backend="cpu", n_jobs=1).fit(X, np.random.default_rng(0).uniform(size=X.shape[0]))


def test_cfs_transform_validates_shape_and_values():
    X, y = _data()
    model = CFS(backend="cpu", n_jobs=1).fit(X, y)
    with pytest.raises(ValueError):
        model.transform(X[:, :-1])
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.transform(bad)


# -------------------------------------------------------------------- threads and CUDA import


def test_resolve_num_threads_honours_configured_pool():
    assert resolve_num_threads(-1) == config.NUMBA_NUM_THREADS
    assert resolve_num_threads(1) == 1
    for bad in (0, -2, config.NUMBA_NUM_THREADS + 1):
        with pytest.raises(ValueError, match="n_jobs"):
            resolve_num_threads(bad)
    for bad in (1.0, True, "1", None):
        with pytest.raises(TypeError):
            resolve_num_threads(bad)


def _run_python(code, **env):
    full_env = {**os.environ, **env}
    return subprocess.run([sys.executable, "-c", code], env=full_env, capture_output=True, text=True, timeout=300)


def test_default_cfs_respects_capped_numba_thread_pool():
    code = (
        "import numpy as np\n"
        "from fast_select import CFS\n"
        "rng = np.random.default_rng(0)\n"
        "X = rng.integers(0, 3, (60, 4)); y = rng.integers(0, 2, 60)\n"
        "CFS(backend='cpu').fit(X, y)\n"
        "print('ok')\n"
    )
    result = _run_python(code, NUMBA_NUM_THREADS="2", NUMBA_DEFAULT_NUM_THREADS="8")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_package_imports_under_the_cuda_simulator():
    code = "import fast_select\nfrom fast_select.utils import is_cuda_ready\nprint(is_cuda_ready())\n"
    result = _run_python(code, NUMBA_ENABLE_CUDASIM="1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


@pytest.mark.skipif(sys.platform == "win32", reason="the workaround is intentionally installed on Windows")
def test_import_does_not_patch_numba_cuda_runtime_off_windows():
    code = (
        "import fast_select\n"
        "from numba.cuda.cudadrv import devices\n"
        "method = devices._Runtime._get_or_create_context_uncached\n"
        "print(method.__module__)\n"
    )
    result = _run_python(code)
    if "No module named" in result.stderr or "AttributeError" in result.stderr:  # pragma: no cover
        pytest.skip("this Numba build has no CUDA driver runtime module")
    assert result.returncode == 0, result.stderr
    assert not result.stdout.strip().startswith("fast_select"), "the Windows-only workaround leaked"
