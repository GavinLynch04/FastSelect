import math
import numpy as np
import pytest

try:
    from hypothesis import given, settings, strategies as st  # type: ignore
    HYPOTHESIS_AVAILABLE = True
except ModuleNotFoundError:
    HYPOTHESIS_AVAILABLE = False

from fast_select.MDR import MDR, MAX_K_FOR_KERNEL

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


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
        np.array([0, 0, 0, 2], dtype=np.uint8),       # non-binary
        np.array([0, 0, 255, 1], dtype=np.uint8),     # >1 value
    ],
)
def test_fit_raises_on_invalid_y(bad_y):
    X = np.zeros((len(bad_y), 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        MDR().fit(X, bad_y)


@pytest.mark.parametrize(
    "bad_X",
    [
        np.array([[0, 1, 3]], dtype=np.uint8),   # genotype out of range
        np.array([[-1, 0, 1]], dtype=np.int8),   # negative genotype
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
    cpu_clf  = MDR(k=2, cv=2, backend="CPU").fit(X, y)
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
        y_public  = clf.predict(X)
        y_private = clf._internal_predict(X, clf.best_interaction_, lut)
        np.testing.assert_array_equal(y_public, y_private)



@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device not available")
def test_gpu_kernel_consistency(simple_dataset):
    """GPU kernel BA ≈ reference CPU BA for each combination."""
    import numpy as np
    from itertools import combinations
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
    cuda.synchronize()
    gpu_bas = results_d.copy_to_host()

    # CPU reference
    def cpu_ba(combo):
        n_cells = 3 ** k
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
            high = (control[i] == 0) or (case[i] / control[i] > thr)
            tp += case[i] if high else 0
            tn += control[i] if not high else 0
        sens = tp / total_case
        spec = tn / total_ctrl
        return (sens + spec) / 2

    cpu_bas = np.array([cpu_ba(c) for c in combos], dtype=np.float32)
    np.testing.assert_allclose(gpu_bas, cpu_bas, rtol=1e-4, atol=1e-5)
