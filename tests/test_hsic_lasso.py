"""HSIC Lasso against Yamada et al. (2014) and Climente-Gonzalez et al. (2019).

The oracle builds the papers' design explicitly: for every permutation and
block, the Gaussian Gram matrix of each standardised feature and the output
Gram matrix (delta kernel ``1/n_c`` or Gaussian), centred as ``H K H``,
optionally divided by its Frobenius norm (2019, Eq. 2), vectorised and stacked
with weight ``sqrt(1 / (n_blocks M))`` (Eqs. 12-13). FastSelect never forms
this matrix, so agreement checks its streamed identities. The solver is checked
against the nonnegative Lasso optimality conditions at the stopping point and,
as corroboration, against scikit-learn's ``lars_path(positive=True)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LogisticRegression, lars_path
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_select import HSICLasso


def _blocks(n, block, n_perm, seed=0):
    order = np.arange(n)
    out = []
    for m in range(n_perm):
        order = np.random.RandomState(seed + m).permutation(order)
        used = (n // block) * block
        out.append(order[:used].reshape(-1, block))
    return out


def _centred(K, normalize):
    b = K.shape[0]
    H = np.eye(b) - np.ones((b, b)) / b
    C = H @ K @ H
    if normalize:
        norm = np.linalg.norm(C, "fro")
        return C / norm if norm > 1e-12 else np.zeros_like(C)
    return C


def hsic_lasso_design(X, y, task, block=20, n_perm=3, sigma_x=1.0, sigma_y=1.0, normalize=True, seed=0):
    """The stacked design ``U`` (rows x d) and target ``v`` of the 2019 paper, Eqs. (12)-(14)."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if block in (0, None):
        block, n_perm = n, 1
    std = X.std(axis=0)
    Z = X / np.where(std > 0, std, 1.0)
    if task == "regression":
        t = np.asarray(y, dtype=np.float64)
        t = t / t.std()
    else:
        t = np.unique(np.asarray(y), return_inverse=True)[1]
    U_rows, v_rows = [], []
    partitions = _blocks(n, block, n_perm, seed)
    weight = np.sqrt(1.0 / (len(partitions) * partitions[0].shape[0]))
    for blocks in partitions:
        for idx in blocks:
            if task == "regression":
                ty = t[idx]
                L = np.exp(-((ty[:, None] - ty[None, :]) ** 2) / (2 * sigma_y**2))
            else:
                ty = t[idx]
                counts = {c: np.sum(ty == c) for c in np.unique(ty)}
                L = np.array([[1.0 / counts[a] if a == b else 0.0 for b in ty] for a in ty])
            v_rows.append(weight * _centred(L, normalize).ravel())
            cols = []
            for k in range(d):
                z = Z[idx, k]
                K = np.exp(-((z[:, None] - z[None, :]) ** 2) / (2 * sigma_x**2))
                cols.append(weight * _centred(K, normalize).ravel())
            U_rows.append(np.column_stack(cols))
    return np.vstack(U_rows), np.concatenate(v_rows)


def _kkt(U, v, beta, active, tol=1e-6):
    """Nonnegative Lasso optimality at the stopping point: active gradients equal lambda, the rest do not exceed it."""
    grad = U.T @ (v - U @ beta)
    lam = grad[active].mean()
    assert np.all(beta[active] > 0)
    assert_allclose(grad[active], lam, rtol=tol, atol=tol * max(1.0, abs(lam)))
    inactive = np.setdiff1d(np.arange(U.shape[1]), active)
    assert np.all(grad[inactive] <= lam + tol * max(1.0, abs(lam)))
    return lam


def _data(n=96, d=8, seed=0, task="classification"):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)) * np.linspace(0.5, 4.0, d) + np.linspace(-3.0, 9.0, d)
    signal = np.sin(2 * X[:, 0] / 0.5) + ((X[:, 1] - X[:, 1].mean()) / X[:, 1].std()) ** 2
    if task == "regression":
        y = signal + 0.2 * rng.standard_normal(n)
    else:
        y = np.where(signal > np.median(signal), "case", "control")
    return X, y


# --------------------------------------------------------------------------- equation oracle


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("block,n_perm", [(20, 3), (16, 2), (0, 1)])
def test_relevance_equals_the_explicit_design(block, n_perm, normalize):
    X, y = _data()
    U, v = hsic_lasso_design(X, y, "classification", block, n_perm, normalize=normalize)
    model = HSICLasso(3, block_size=block, n_permutations=n_perm, normalize=normalize, n_jobs=1).fit(X, y)
    assert_allclose(model.relevance_, U.T @ v, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_coefficients_satisfy_nonnegative_lasso_optimality(task, k):
    X, y = _data(task=task, seed=1)
    U, v = hsic_lasso_design(X, y, task)
    model = HSICLasso(k, n_jobs=1).fit(X, y)
    active = model.top_features_
    assert active.size == k
    assert model.task_ == task
    _kkt(U, v, model.feature_importances_, active)
    assert np.all(model.feature_importances_[np.setdiff1d(np.arange(X.shape[1]), active)] == 0.0)


def test_path_agrees_with_scikit_learn_lars():
    X, y = _data(n=120, d=10, seed=2)
    U, v = hsic_lasso_design(X, y, "classification")
    model = HSICLasso(4, n_jobs=1).fit(X, y)
    _, active_order, coefs = lars_path(U, v, method="lasso", positive=True)
    # FastSelect stops at the knot where a fifth feature would enter: scikit-learn's
    # last knot with four nonzero coefficients.
    counts = (coefs > 0).sum(axis=0)
    knot = int(np.flatnonzero(counts == 4)[-1])
    expected = coefs[:, knot]
    assert set(active_order[:4]) == set(model.top_features_)
    assert_allclose(model.feature_importances_, expected, rtol=1e-6, atol=1e-10)


def test_regression_relevance_and_vanilla_mode():
    X, y = _data(task="regression", seed=3)
    for block in (20, 0):
        U, v = hsic_lasso_design(X, y, "regression", block)
        model = HSICLasso(2, block_size=block, n_jobs=1).fit(X, y)
        assert_allclose(model.relevance_, U.T @ v, rtol=1e-10, atol=1e-13)


# --------------------------------------------------------------------------- edge cases


def test_constant_feature_has_zero_relevance_and_is_never_selected():
    X, y = _data(seed=4)
    X[:, 3] = 0.1
    model = HSICLasso(X.shape[1], n_jobs=1).fit(X, y)
    assert model.relevance_[3] == 0.0
    assert 3 not in model.top_features_
    U, v = hsic_lasso_design(X, y, "classification")
    assert_allclose(model.relevance_, U.T @ v, rtol=1e-10, atol=1e-13)


def test_single_class_blocks_contribute_nothing():
    """With 5 cases in 80 samples and B = 16, some blocks hold one class: their output Gram is zero."""
    X, _ = _data(n=80, seed=5)
    y = np.zeros(80, dtype=int)
    y[[3, 21, 40, 58, 77]] = 1
    single = [set(y[idx]) == {0} for blocks in _blocks(80, 16, 3) for idx in blocks]
    assert any(single) and not all(single)
    model = HSICLasso(2, block_size=16, n_permutations=3, n_jobs=1).fit(X, y)
    U, v = hsic_lasso_design(X, y, "classification", 16, 3)
    assert_allclose(model.relevance_, U.T @ v, rtol=1e-10, atol=1e-13)


def test_leftover_samples_are_dropped_per_permutation():
    X, y = _data(n=97, seed=6)  # 97 = 4 * 20 + 17 left over
    U, v = hsic_lasso_design(X, y, "classification", 20, 3)
    model = HSICLasso(3, block_size=20, n_permutations=3, n_jobs=1).fit(X, y)
    assert_allclose(model.relevance_, U.T @ v, rtol=1e-10, atol=1e-13)


def test_duplicate_of_an_active_feature_never_enters():
    X, y = _data(seed=7)
    X = np.column_stack([X, X[:, 0]])
    model = HSICLasso(X.shape[1], n_jobs=1).fit(X, y)
    assert not {0, X.shape[1] - 1} <= set(model.top_features_)


def test_path_ending_early_returns_fewer_features():
    X, y = _data(n=60, d=4, seed=8)
    X[:, 1:] = 5.0  # three constant features: only feature 0 can ever enter
    model = HSICLasso(3, n_jobs=1).fit(X, y)
    assert_array_equal(model.top_features_, [0])


def test_redundant_nonlinear_copy_is_skipped():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((300, 10))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.4).astype(int)  # no linear signal at all
    X[:, 5] = X[:, 0] + 0.01 * rng.standard_normal(300)
    model = HSICLasso(2, n_jobs=-1).fit(X, y)
    assert len({0, 5} & set(model.top_features_)) == 1
    assert 1 in model.top_features_


# --------------------------------------------------------------------------- invariances


def test_affine_and_label_invariance():
    X, y = _data(seed=10)
    reference = HSICLasso(4, n_jobs=1).fit(X, y)
    moved = HSICLasso(4, n_jobs=1).fit(X * 3.5 - 1e6, np.where(y == "case", 7, -2))
    assert_array_equal(moved.top_features_, reference.top_features_)
    assert_allclose(moved.feature_importances_, reference.feature_importances_, rtol=1e-7, atol=1e-12)


def test_thread_count_does_not_change_the_result():
    X, y = _data(n=200, d=20, seed=11)
    serial = HSICLasso(5, n_jobs=1).fit(X, y)
    parallel = HSICLasso(5, n_jobs=-1).fit(X, y)
    assert_array_equal(parallel.top_features_, serial.top_features_)
    assert_array_equal(parallel.relevance_, serial.relevance_)


def test_random_state_changes_only_the_partitions():
    X, y = _data(seed=12)
    a = HSICLasso(3, random_state=0, n_jobs=1).fit(X, y).relevance_
    b = HSICLasso(3, random_state=5, n_jobs=1).fit(X, y).relevance_
    U, v = hsic_lasso_design(X, y, "classification", seed=5)
    assert_allclose(b, U.T @ v, rtol=1e-10, atol=1e-13)
    assert not np.allclose(a, b)


# --------------------------------------------------------------------------- API


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"task": "ranking"}, ValueError),
        ({"block_size": 1}, ValueError),
        ({"block_size": 500}, ValueError),
        ({"block_size": 2.5}, TypeError),
        ({"n_permutations": 0}, ValueError),
        ({"sigma_x": 0.0}, ValueError),
        ({"sigma_y": -1.0}, ValueError),
        ({"normalize": 1}, TypeError),
        ({"random_state": None}, TypeError),
        ({"backend": "gpu"}, ValueError),
        ({"n_features_to_select": 0}, ValueError),
        ({"n_jobs": 0}, ValueError),
    ],
)
def test_parameter_validation(kwargs, error):
    X, y = _data()
    with pytest.raises(error):
        HSICLasso(**{"n_features_to_select": 2, **kwargs}).fit(X, y)


def test_rejects_constant_regression_target_and_nan():
    X, y = _data(task="regression")
    with pytest.raises(ValueError, match="two distinct values"):
        HSICLasso(2, task="regression").fit(X, np.full(len(y), 3.0))
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        HSICLasso(2).fit(X_nan, y)


def test_fitted_attributes_and_transform():
    X, y = _data()
    model = HSICLasso(3)
    reduced = model.fit_transform(X, y)
    assert model.effective_backend_ == "cpu"
    assert model.block_size_ == 20 and model.n_permutations_ == 3
    assert_array_equal(reduced, X[:, model.top_features_])
    assert list(model.top_features_) == sorted(model.top_features_, key=lambda f: -model.feature_importances_[f])
    vanilla = HSICLasso(3, block_size=0).fit(X, y)
    assert vanilla.block_size_ == X.shape[0] and vanilla.n_permutations_ == 1
    with pytest.raises(ValueError):
        model.transform(X[:, :3])


def test_sklearn_api_compliance():
    check_estimator(HSICLasso(2, block_size=0, n_jobs=1))


def test_in_pipeline_and_grid_search():
    X, y = _data(n=150, seed=13)
    pipe = Pipeline([("select", HSICLasso(2)), ("model", LogisticRegression(max_iter=500))])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 3], "select__normalize": [True, False]}, cv=3)
    search.fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        <= search.best_params_["select__n_features_to_select"]
    )


# --------------------------------------------------------------------------- external cross-check


def _reference_hsic_lasso():
    """pyHSICLasso 1.4.2, the authors' implementation, from the environment or ``PYHSICLASSO_PATH``."""
    import os
    import sys

    path = os.environ.get("PYHSICLASSO_PATH")
    if path and path not in sys.path:
        sys.path.append(path)  # appended, so NumPy/SciPy/pandas still come from the environment
    os.environ.setdefault("MPLBACKEND", "Agg")
    reference = pytest.importorskip("pyHSICLasso")
    import importlib.metadata

    try:
        version = importlib.metadata.version("pyHSICLasso")
    except importlib.metadata.PackageNotFoundError:
        version = next(
            (p.name.split("-")[1] for p in __import__("pathlib").Path(path or ".").glob("pyHSICLasso-*.dist-info")),
            None,
        )
    if version != "1.4.2":
        pytest.skip(f"cross-check pinned to pyHSICLasso 1.4.2, found {version}")
    return reference.HSICLasso


@pytest.mark.parametrize(
    "n,d,block,n_perm,task",
    [
        (120, 12, 20, 3, "classification"),
        (97, 15, 20, 3, "classification"),
        (100, 20, 0, 1, "classification"),
        (150, 25, 20, 3, "regression"),
        (80, 10, 0, 1, "regression"),
    ],
)
def test_matches_pinned_pyhsiclasso(n, d, block, n_perm, task):
    """Corroboration only: same selections in the same order, coefficients to its float32 precision."""
    import contextlib
    import io

    Reference = _reference_hsic_lasso()
    rng = np.random.default_rng(n + d)
    X = rng.standard_normal((n, d)) * rng.uniform(0.5, 5, d)
    signal = np.sin(2 * X[:, 0] / X[:, 0].std()) + (X[:, 1] / X[:, 1].std()) ** 2
    y = (signal > np.median(signal)).astype(int) if task == "classification" else signal + 0.2 * rng.standard_normal(n)
    reference = Reference()
    with contextlib.redirect_stdout(io.StringIO()):
        reference.input(X, y)
        (reference.classification if task == "classification" else reference.regression)(
            num_feat=5, B=block, M=n_perm, n_jobs=1
        )
    model = HSICLasso(5, task=task, block_size=block, n_permutations=n_perm, n_jobs=1).fit(X, y)
    assert_array_equal(model.top_features_, reference.get_index())
    beta = np.asarray(reference.beta, dtype=np.float64).ravel()
    assert_allclose(model.feature_importances_, beta, rtol=1e-5, atol=1e-6 * beta.max())
