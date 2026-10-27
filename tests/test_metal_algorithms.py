"""Physical Metal conformance against independent equations and CPU backends."""

from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from test_algorithm_compliance import _multisurf_reference
from test_exact_boundaries import (
    CONTINUOUS_TIE_SEEDS,
    MIXED_TIE_SEEDS,
    discrete_limit_for,
    make_case,
    surf_exact,
)
from test_information_oracles import _information_oracle
from test_release_regressions import cfs_reference, mdr_predictions

from fast_select import CFS, MDR, SURF, MultiSURF, ReliefF, TuRF, calculate_mi_matrices, chi2, mRMR


@pytest.fixture(scope="module", autouse=True)
def physical_metal():
    from fast_select.utils import is_metal_ready

    if not is_metal_ready():
        pytest.skip("Physical Metal GPU and optional MLX required")


@pytest.mark.parametrize("star", [False, True])
@pytest.mark.parametrize(
    "mixed,seed", [(False, s) for s in CONTINUOUS_TIE_SEEDS[:12]] + [(True, s) for s in MIXED_TIE_SEEDS[:12]]
)
def test_surf_exact_rational_boundaries(star, mixed, seed):
    x, y, discrete = make_case(seed, mixed)
    expected, _ = surf_exact(x, y, star, discrete)
    model = SURF(1, backend="metal", discrete_limit=discrete_limit_for(x, discrete), use_star=star).fit(x, y)
    assert_allclose(model.feature_importances_, expected, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize("star", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_multisurf_mixed_oracle_and_subset(star, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(31, 7))
    x[:, 1] = rng.integers(0, 3, len(x)) + 2**25
    x[:, 6] = 42
    x[:, [0, 2, 3, 4, 5]] += 1e8
    y = np.array([7] * 22 + [19] * 9)
    discrete = np.array([False, True, False, False, False, False, True])
    expected = _multisurf_reference(x, y, discrete, star)
    cpu = MultiSURF(backend="cpu", discrete_limit=3, use_star=star).fit(x, y)
    metal = MultiSURF(backend="metal", discrete_limit=3, use_star=star).fit(x, y)
    assert_allclose(metal.feature_importances_, expected, atol=2e-7, rtol=1e-5)
    assert_allclose(metal.feature_importances_, cpu.feature_importances_, atol=2e-7, rtol=1e-5)
    subset = np.array([1, 3, 5])
    limited = MultiSURF(backend="metal", discrete_limit=3, use_star=star).fit(x, y, feat_idx=subset)
    assert set(limited.top_features_).issubset(set(subset))


@pytest.mark.parametrize("unit", ["bit", "nat"])
@pytest.mark.parametrize("states", [2, 7, 37, 64])
def test_information_distribution_oracles(unit, states):
    rng = np.random.default_rng(204)
    x = rng.integers(0, states, (79, 9), dtype=np.int64)
    y = np.array([0] * 50 + [3] * 20 + [7] * 9)
    x[:, 0] = 0
    x[:, 2] = x[:, 1]
    rel, red = calculate_mi_matrices(x, y, backend="metal", unit=unit)
    assert_allclose(rel, [_information_oracle(column, y, unit) for column in x.T], rtol=1e-12, atol=2e-14)
    for i, j in combinations(range(x.shape[1]), 2):
        assert_allclose(red[i, j], _information_oracle(x[:, i], x[:, j], unit), rtol=1e-12, atol=2e-14)
    cpu = calculate_mi_matrices(x, y, backend="cpu", unit=unit)
    assert_allclose(rel, cpu[0], rtol=1e-12, atol=2e-14)
    assert_allclose(red, cpu[1], rtol=1e-12, atol=2e-14)


def test_integer_histogram_memory_fallback():
    # 130 variables * 64 states would exceed the 64 MiB Gram-matrix budget.
    rng = np.random.default_rng(61)
    x = rng.integers(0, 64, (77, 129), dtype=np.int32)
    y = rng.integers(0, 3, 77, dtype=np.int32)
    cpu = calculate_mi_matrices(x, y, backend="cpu")
    metal = calculate_mi_matrices(x, y, backend="metal")
    for a, b in zip(cpu, metal):
        assert_allclose(a, b, rtol=1e-12, atol=2e-14)
    assert_allclose(metal[1][3, 11], _information_oracle(x[:, 3], x[:, 11]), atol=2e-14)


def test_explicit_metal_overrides_mlx_cpu_default():
    import mlx.core as mx

    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        x = np.random.default_rng(42).normal(size=(23, 5))
        y = np.arange(len(x)) % 2
        for cls in (ReliefF, SURF, MultiSURF):
            model = cls(backend="metal").fit(x, y)
            assert model.effective_backend_ == "metal"
    finally:
        mx.set_default_device(previous)


def test_mdr_ties_across_metal_batches():
    x = np.zeros((12, 365), dtype=np.uint8)
    y = np.arange(12) % 2
    model = MDR(k=2, cv=2, backend="metal").fit(x, y)
    assert tuple(model.best_interaction_) == (0, 1)


@pytest.mark.parametrize("seed", range(6))
def test_cfs_search_and_mrmr(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, (80, 6))
    y = rng.integers(0, 2, 80)
    subset, merit = cfs_reference(x, y)
    actual = CFS(backend="metal").fit(x, y)
    assert_array_equal(actual.selected_indices_, sorted(subset))
    assert_allclose(actual.merit_, merit, atol=2e-7)
    for method in ("MID", "MIQ"):
        cpu = mRMR(4, backend="cpu", method=method).fit(x, y)
        metal = mRMR(4, backend="metal", method=method).fit(x, y)
        assert_array_equal(cpu.top_features_, metal.top_features_)


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_mdr_integer_risk_oracle(k):
    from fast_select._mdr_metal import make_mdr_scorer

    rng = np.random.default_rng(103)
    x = rng.integers(0, 3, (51, 6), dtype=np.uint8)
    y = np.array([0] * 37 + [1] * 14, dtype=np.uint8)
    combos = np.array(list(combinations(range(6), k)), dtype=np.uint32)
    actual = make_mdr_scorer(x, y, k)(combos)
    expected = []
    for combo in combos:
        predicted = mdr_predictions(x, y, x, tuple(combo))
        expected.append(((predicted[y == 0] == 0).mean() + (predicted[y == 1] == 1).mean()) / 2)
    assert_array_equal(actual, np.array(expected, dtype=np.float32))
    cpu = MDR(k=k, cv=3, backend="cpu").fit(x, y * 19 + 7)
    metal = MDR(k=k, cv=3, backend="metal").fit(x, y * 19 + 7)
    assert cpu.best_interaction_ == metal.best_interaction_
    assert cpu.best_cvc_ == metal.best_cvc_
    assert_array_equal(cpu.predict(x), metal.predict(x))


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_chi2_float64_count_oracle(scale):
    rng = np.random.default_rng(452)
    x = rng.uniform(0, 2, (513, 11)) * scale
    x[:, 0] = 0
    y = np.array([0] * 412 + [3] * 99 + [7] * 2)
    observed = np.array([x[y == c].sum(axis=0) for c in np.unique(y)])
    expected = np.array([(y == c).mean() * x.sum(axis=0) for c in np.unique(y)])
    statistic = np.sum(
        np.divide((observed - expected) ** 2, expected, out=np.zeros_like(expected), where=expected > 1e-12), axis=0
    )
    scores, p = chi2(x, y, backend="metal")
    assert_allclose(scores, statistic, rtol=2e-11, atol=1e-11)
    assert_allclose(p, np.exp(-statistic / 2), atol=1e-11)


@pytest.mark.parametrize(
    "selector", [ReliefF(), SURF(), SURF(use_star=True), MultiSURF(), MultiSURF(use_star=True), CFS(), mRMR(2)]
)
def test_auto_metal_pipeline_and_turf(selector):
    rng = np.random.default_rng(42)
    x = rng.integers(0, 3, (40, 7))
    y = x[:, 0] % 2
    fitted = clone(selector).fit(x, y)
    from fast_select.utils import is_cuda_ready

    expected = "gpu" if is_cuda_ready() else ("cpu" if isinstance(selector, CFS) else "metal")
    assert fitted.effective_backend_ == expected
    search = GridSearchCV(make_pipeline(clone(selector), LogisticRegression()), {}, cv=2).fit(x, y)
    assert search.predict(x).shape == y.shape
    if isinstance(selector, (ReliefF, SURF, MultiSURF)):
        cpu = TuRF(clone(selector).set_params(backend="cpu"), n_features_to_select=2, pct_remove=0.4).fit(x, y)
        metal = TuRF(clone(selector).set_params(backend="metal"), n_features_to_select=2, pct_remove=0.4).fit(x, y)
        assert_array_equal(cpu.top_features_, metal.top_features_)
        assert metal.effective_backend_ == "metal"
