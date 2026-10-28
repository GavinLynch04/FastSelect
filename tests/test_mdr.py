import math
from itertools import combinations

import numpy as np
import pytest

try:
    from hypothesis import given, settings  # type: ignore
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ModuleNotFoundError:
    HYPOTHESIS_AVAILABLE = False

from fast_select.MDR import MAX_K_FOR_KERNEL, MDR, _combination_batches
from fast_select.utils import is_cuda_ready

try:
    CUDA_AVAILABLE = is_cuda_ready()
except Exception:
    CUDA_AVAILABLE = False


@pytest.mark.parametrize("interaction_order", [1, 2, 3, 6])
def test_combination_batches_preserve_exhaustive_order(interaction_order):
    expected = np.array(list(combinations(range(7), interaction_order)), dtype=np.uint32)
    actual = np.concatenate(list(_combination_batches(7, interaction_order)))
    np.testing.assert_array_equal(actual, expected)


def test_combination_batch_ties_keep_first_model(monkeypatch):
    import importlib

    monkeypatch.setattr(importlib.import_module("fast_select.MDR"), "_COMBINATION_BATCH_SIZE", 2)
    features = np.zeros((12, 6), dtype=np.uint8)
    target = np.array([0, 1] * 6)
    model = MDR(k=2, cv=3, backend="cpu").fit(features, target)
    assert tuple(model.best_interaction_) == (0, 1)
    assert model.best_cvc_ == 3


@pytest.fixture(scope="module")
def simple_dataset():
    """Tiny 2-SNP toy data: high-risk only if both genotypes == 2."""
    X = np.array(
        [
            [2, 2],
            [2, 2],
            [2, 0],
            [0, 2],
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    return X, y


@pytest.fixture(scope="module")
def classifier():
    """Default: k=2, cv=2, backend='auto'."""
    return MDR(k=2, cv=2)


def test_fit_and_attributes(simple_dataset, classifier):
    X, y = simple_dataset
    clf = classifier.fit(X, y)
    assert hasattr(clf, "best_interaction_")
    assert hasattr(clf, "best_model_lookup_table_")
    assert clf.best_interaction_ == (0, 1)
    assert clf.best_cvc_ == clf.cv
    assert math.isclose(clf.best_mean_testing_ba_, 1.0, abs_tol=1e-6)


def test_predict_and_transform(simple_dataset, classifier):
    X, y = simple_dataset
    clf = classifier.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    assert np.array_equal(y_pred, y)

    X_new = clf.transform(X)
    assert X_new.shape == (X.shape[0], 1)
    assert np.array_equal(X_new.ravel(), y_pred)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([0, 0, 1, 2], dtype=np.uint8),
        np.array([0, 1, 2, 3], dtype=np.uint8),
    ],
)
def test_fit_raises_on_invalid_y(bad_y):
    X = np.zeros((len(bad_y), 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        MDR().fit(X, bad_y)


@pytest.mark.parametrize(
    "bad_X",
    [
        np.array([[0, 1, 3]], dtype=np.uint8),  # genotype out of range
        np.array([[-1, 0, 1]], dtype=np.int8),  # negative genotype
    ],
)
def test_fit_raises_on_invalid_X(bad_X):
    y = np.array([0], dtype=np.uint8)
    with pytest.raises(ValueError):
        MDR().fit(bad_X, y)


def test_k_parameter_constraints(simple_dataset):
    X, y = simple_dataset
    n_features = X.shape[1]

    # k == n_features is **allowed** in new version
    MDR(k=n_features, cv=2).fit(X, y)

    # k > n_features still invalid
    with pytest.raises(ValueError):
        MDR(k=n_features + 1, cv=2).fit(X, y)

    # k above kernel hard-limit invalid
    with pytest.raises(ValueError):
        MDR(k=MAX_K_FOR_KERNEL + 1).fit(X, y)


def test_backend_cpu_matches_auto(simple_dataset):
    X, y = simple_dataset
    cpu_clf = MDR(k=2, cv=2, backend="CPU").fit(X, y)
    auto_clf = MDR(k=2, cv=2, backend="auto").fit(X, y)
    np.testing.assert_array_equal(cpu_clf.predict(X), auto_clf.predict(X))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device not available")
def test_backend_gpu_matches_cpu(simple_dataset):
    X, y = simple_dataset
    cpu_clf = MDR(k=2, cv=2, backend="CPU").fit(X, y)
    gpu_clf = MDR(k=2, cv=2, backend="GPU").fit(X, y)
    np.testing.assert_array_equal(cpu_clf.predict(X), gpu_clf.predict(X))


@pytest.mark.skipif(CUDA_AVAILABLE, reason="CUDA available, error case irrelevant")
def test_backend_gpu_raises_without_cuda(simple_dataset):
    X, y = simple_dataset
    with pytest.raises(RuntimeError):
        MDR(k=2, cv=2, backend="GPU").fit(X, y)


if HYPOTHESIS_AVAILABLE:

    @settings(max_examples=50, deadline=None)
    @given(
        n_samples=st.integers(min_value=20, max_value=60),
        n_features=st.integers(min_value=3, max_value=6),
    )
    def test_internal_vs_public_predict_consistency(n_samples, n_features):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 3, size=(n_samples, n_features), dtype=np.uint8)
        y = ((X[:, 0] + X[:, 1]) % 2).astype(np.uint8)

        clf = MDR(k=2, cv=3).fit(X, y)
        lut = clf.best_model_lookup_table_
        y_public = clf.predict(X)
        y_private = clf._internal_predict(X, clf.best_interaction_, lut)
        np.testing.assert_array_equal(y_public, y_private)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device not available")
def test_gpu_kernel_consistency(simple_dataset):
    """GPU kernel BA ≈ reference CPU BA for each combination."""
    from itertools import combinations

    import numpy as np
    from numba import cuda

    from fast_select.MDR import mdr_kernel  # import after CUDA check

    X, y = simple_dataset
    k = 2
    combos = np.array(list(combinations(range(X.shape[1]), k)), dtype=np.uint32)
    n_combos = len(combos)

    # GPU compute
    X_d = cuda.to_device(X)
    y_d = cuda.to_device(y)
    combos_d = cuda.to_device(combos)
    results_d = cuda.device_array(n_combos, dtype=np.float32)

    mdr_kernel[1, 32](X_d, y_d, k, combos_d, results_d)
    gpu_bas = results_d.copy_to_host()

    # CPU reference
    def cpu_ba(combo):
        n_cells = 3**k
        case = np.zeros(n_cells, dtype=int)
        control = np.zeros(n_cells, dtype=int)
        for i in range(X.shape[0]):
            cell = 0
            for idx in combo:
                cell = cell * 3 + X[i, idx]
            (case if y[i] else control)[cell] += 1
        total_case = case.sum()
        total_ctrl = control.sum()
        if total_case == 0 or total_ctrl == 0:
            return 0.0
        thr = total_case / total_ctrl
        tp = tn = 0
        for i in range(n_cells):
            high = (control[i] == 0 and case[i] > 0) or (control[i] > 0 and case[i] / control[i] >= thr)
            tp += case[i] if high else 0
            tn += control[i] if not high else 0
        sens = tp / total_case
        spec = tn / total_ctrl
        return (sens + spec) / 2

    cpu_bas = np.array([cpu_ba(c) for c in combos], dtype=np.float32)
    np.testing.assert_allclose(gpu_bas, cpu_bas, rtol=1e-4, atol=1e-5)


def _balanced_accuracy_oracle(X, y, combination):
    """Ritchie et al. (2001) training balanced accuracy, counted cell by cell in Python."""
    case, control = {}, {}
    for row, label in zip(X[:, list(combination)].tolist(), y.tolist()):
        cell = tuple(row)
        case[cell] = case.get(cell, 0) + (label == 1)
        control[cell] = control.get(cell, 0) + (label != 1)
    total_case, total_control = sum(case.values()), sum(control.values())
    if not total_case or not total_control:
        return 0.0
    true_positive = true_negative = 0
    for cell in case:
        # High risk: case/control ratio >= the dataset ratio (cross-multiplied, exact).
        if case[cell] * total_control >= total_case * control[cell] and (case[cell] or control[cell]):
            true_positive += case[cell]
        else:
            true_negative += control[cell]
    return (true_positive / total_case + true_negative / total_control) / 2


@pytest.mark.parametrize("n_samples", [1, 63, 64, 65, 128, 191, 200])
@pytest.mark.parametrize("interaction_order", [1, 2, 3, 4])
def test_bitwise_and_per_sample_counting_match_the_oracle(n_samples, interaction_order):
    from fast_select.MDR import _batch_balanced_accuracy_bits, _batch_balanced_accuracy_cpu, _genotype_masks

    rng = np.random.default_rng(1000 * n_samples + interaction_order)
    X = rng.integers(0, 3, (n_samples, 6), dtype=np.uint8)
    X[:, 5] = 1  # constant locus: most of its cells are empty
    y = (rng.random(n_samples) < 0.3).astype(np.uint8)
    combos = np.array(list(combinations(range(6), interaction_order)), dtype=np.uint32)
    expected = np.array([_balanced_accuracy_oracle(X, y, c) for c in combos], dtype=np.float32)
    per_sample = _batch_balanced_accuracy_cpu(X, y, combos, interaction_order)
    masks, case_mask = _genotype_masks(X, y)
    bitwise = _batch_balanced_accuracy_bits(masks, case_mask, n_samples, combos, interaction_order)
    np.testing.assert_allclose(per_sample, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(bitwise, per_sample)


def test_bitwise_counting_keeps_exact_ratio_ties_high_risk():
    from fast_select.MDR import _batch_balanced_accuracy_bits, _batch_balanced_accuracy_cpu, _genotype_masks

    # 70 cases and 140 controls (ratio 1:2). Genotype 0 holds 10 cases and 20
    # controls, exactly the dataset ratio, so it is high risk; genotype 2 is
    # empty. Spanning several 64-bit words exercises the word boundaries.
    X = np.concatenate([np.zeros(30), np.ones(180)]).astype(np.uint8).reshape(-1, 1)
    y = np.concatenate([np.ones(10), np.zeros(20), np.ones(60), np.zeros(120)]).astype(np.uint8)
    combos = np.zeros((1, 1), dtype=np.uint32)
    masks, case_mask = _genotype_masks(X, y)
    bitwise = _batch_balanced_accuracy_bits(masks, case_mask, len(y), combos, 1)
    np.testing.assert_array_equal(bitwise, _batch_balanced_accuracy_cpu(X, y, combos, 1))
    assert bitwise[0] == np.float32(_balanced_accuracy_oracle(X, y, (0,))) == np.float32(0.5)


@pytest.mark.parametrize("interaction_order", [1, 2, 3])
def test_fit_is_identical_on_both_counting_paths(monkeypatch, interaction_order):
    import importlib

    module = importlib.import_module("fast_select.MDR")
    rng = np.random.default_rng(interaction_order)
    X = rng.integers(0, 3, (300, 9), dtype=np.uint8)
    y = np.where((X[:, 2] == 2) ^ (rng.random(300) < 0.2), "case", "control")
    models = []
    for bitwise in (True, False):
        monkeypatch.setattr(module, "_use_bitwise_counts", lambda n_samples, k, bitwise=bitwise: bitwise)
        models.append(MDR(k=interaction_order, cv=3, backend="cpu").fit(X, y))
    bitwise_model, per_sample_model = models
    assert bitwise_model.best_interaction_ == per_sample_model.best_interaction_
    assert bitwise_model.best_cvc_ == per_sample_model.best_cvc_
    assert bitwise_model.best_mean_testing_ba_ == per_sample_model.best_mean_testing_ba_
    np.testing.assert_array_equal(bitwise_model.best_model_lookup_table_, per_sample_model.best_model_lookup_table_)
