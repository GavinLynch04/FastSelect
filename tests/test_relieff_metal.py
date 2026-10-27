"""Physical Metal conformance against Kononenko (1994)'s independent oracle."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from test_algorithm_compliance import RELIEFF_DISCRETE, RELIEFF_X, RELIEFF_Y, _relieff_reference

from fast_select import ReliefF


@pytest.fixture(scope="module", autouse=True)
def physical_metal():
    try:
        import mlx.core as mx
    except ImportError as exc:
        pytest.skip(f"Physical Metal GPU/MLX unavailable: {exc}")
    if not mx.metal.is_available():
        pytest.skip("No physical Metal GPU")


@pytest.mark.parametrize("k", [1, 2, 3])
def test_metal_paper_class_priors(k):
    expected = _relieff_reference(RELIEFF_X, RELIEFF_Y, RELIEFF_DISCRETE, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = ReliefF(backend="metal", discrete_limit=3, n_neighbors=k).fit(RELIEFF_X, RELIEFF_Y)
    assert_allclose(model.feature_importances_, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kind", ["continuous", "discrete", "mixed"])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_metal_random_oracle_and_cpu(seed, kind, layout):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(27, 9))
    discrete = np.zeros(9, dtype=bool)
    if kind == "discrete":
        discrete[:] = True
    elif kind == "mixed":
        discrete[::2] = True
    x[:, discrete] = rng.integers(0, 3, (len(x), discrete.sum())) + 2**25
    x[:, ~discrete] += 1e8
    x[:, -1] = 42
    discrete[-1] = kind != "continuous"
    # Non-0/1, multiclass, imbalanced labels with a singleton class.
    y = np.array([7] * 16 + [19] * 10 + [103])
    rng.shuffle(y)
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "strided":
        backing = np.empty((54, 18))
        backing[::2, ::2] = x
        x = backing[::2, ::2]
    original = x.copy()
    limit = 0 if kind == "continuous" else 3
    expected = _relieff_reference(x, y, discrete, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cpu = ReliefF(backend="cpu", n_neighbors=3, discrete_limit=limit, n_jobs=8).fit(x, y)
        metal = ReliefF(backend="metal", n_neighbors=3, discrete_limit=limit).fit(x, y)
    assert_allclose(metal.feature_importances_, expected, rtol=1e-5, atol=2e-7)
    assert_allclose(metal.feature_importances_, cpu.feature_importances_, rtol=1e-5, atol=2e-7)
    assert_array_equal(x, original)
    assert metal.effective_backend_ == "metal"


def test_metal_ambiguous_neighbors_keep_index_ties():
    # All off-diagonal distances equal one or two, exactly in binary.
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2, dtype=np.float64)
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    expected = _relieff_reference(x, y, np.zeros(2, dtype=bool), 1)
    model = ReliefF(backend="metal", discrete_limit=0, n_neighbors=1).fit(x, y)
    assert model.metal_diagnostics_["repaired_neighbor_groups"] > 0
    assert_allclose(model.feature_importances_, expected, atol=1e-7)


def test_metal_tiles_tail_and_feature_tail():
    rng = np.random.default_rng(103)
    x = rng.normal(size=(529, 37))
    y = np.arange(len(x)) % 3
    cpu = ReliefF(backend="cpu", n_jobs=8).fit(x, y)
    metal = ReliefF(backend="metal").fit(x, y)
    assert_allclose(metal.feature_importances_, cpu.feature_importances_, rtol=1e-4, atol=2e-7)
    assert_array_equal(metal.top_features_, cpu.top_features_)
    assert metal.metal_diagnostics_["distance_buffer_bytes"] == 512 * len(x) * 4


@pytest.mark.parametrize("k", [10, 32, 64])
def test_metal_chunk_merge_large_k(k):
    rng = np.random.default_rng(101)
    x = rng.normal(size=(261, 17))
    y = np.arange(len(x)) % 3
    cpu = ReliefF(backend="cpu", n_neighbors=k, n_jobs=8).fit(x, y)
    metal = ReliefF(backend="metal", n_neighbors=k).fit(x, y)
    assert_allclose(metal.feature_importances_, cpu.feature_importances_, rtol=1e-4, atol=2e-7)
    expected = _relieff_reference(x, y, np.zeros(x.shape[1], dtype=bool), k)
    assert_allclose(metal.feature_importances_, expected, rtol=1e-4, atol=2e-7)


def test_metal_explicit_neighbor_limit():
    x = np.random.default_rng(101).normal(size=(140, 2))
    with pytest.raises(ValueError, match="n_neighbors <= 64"):
        ReliefF(backend="metal", n_neighbors=65).fit(x, np.arange(140) % 2)


@pytest.mark.slow
def test_large_mixed_multiclass_independent_float64_oracle():
    """Bound CPU cancellation error separately from Metal/oracle correctness.

    Kononenko (1994): sum normalized differences to select nearest hits/misses,
    normalize each class group by its actual count and each miss by its prior
    divided by the other-class prior. This oracle uses raw input and NumPy
    float64 throughout, with no production preprocessing/distance helpers.
    """
    rng = np.random.default_rng(20261026)
    n, p = 2000, 512
    x = rng.standard_normal((n, p), dtype=np.float32)
    x[:, ::2] = rng.integers(0, 3, (n, p // 2))
    y = np.arange(n, dtype=np.int32) % 4
    rng.shuffle(y)
    raw = x.astype(np.float64)
    discrete = np.zeros(p, dtype=bool)
    discrete[::2] = True
    spans = np.ptp(raw, axis=0)
    spans[spans == 0] = 1
    classes, counts = np.unique(y, return_counts=True)
    priors = counts / n
    members = [np.flatnonzero(y == c) for c in classes]
    expected = np.zeros(p, dtype=np.float64)
    for i in range(n):
        differences = np.abs(raw - raw[i]) / spans
        differences[:, discrete] = raw[:, discrete] != raw[i, discrete]
        distances = differences.sum(axis=1)
        distances[i] = np.inf
        own = np.flatnonzero(classes == y[i])[0]
        for ci, candidates in enumerate(members):
            candidates = candidates[candidates != i]
            ids = candidates[np.lexsort((candidates, distances[candidates]))[:10]]
            weight = -1 if ci == own else priors[ci] / (1 - priors[own])
            expected += weight * differences[ids].mean(axis=0) / n
    params = dict(n_neighbors=10, discrete_limit=3, n_features_to_select=10, n_jobs=8)
    cpu = ReliefF(backend="cpu", **params).fit(x, y)
    metal = ReliefF(backend="metal", **params).fit(x, y)
    assert_allclose(cpu.feature_importances_, expected, rtol=1e-4, atol=2e-6)
    assert_allclose(metal.feature_importances_, expected, rtol=1e-5, atol=2e-7)
    assert_array_equal(cpu.top_features_, np.argsort(expected)[::-1][:10])
    assert_array_equal(metal.top_features_, np.argsort(expected)[::-1][:10])


def test_metal_estimator_pipeline():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 8))
    y = (x[:, 0] > 0).astype(int)
    model = ReliefF(backend="metal", n_features_to_select=3)
    fitted = clone(model).fit(x, y)
    assert fitted.transform(x).shape == (48, 3)
    with pytest.raises(ValueError):
        fitted.transform(x[:, :-1])
    pipeline = make_pipeline(clone(model), LogisticRegression()).fit(x, y)
    assert pipeline.predict(x).shape == y.shape
    search = GridSearchCV(pipeline, {"relieff__n_neighbors": [1, 3]}, cv=2).fit(x, y)
    assert search.predict(x).shape == y.shape
