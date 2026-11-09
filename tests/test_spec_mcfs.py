"""SPEC and MCFS against their restated definitions (Li et al., ACM CSUR 2017, Sections 2.1.2 and 2.3.4).

SPEC's oracle takes the eigenvector route of the survey's Eq. (4): every
eigenpair of the normalised Laplacian, ``alpha_j = xi_j' f_hat`` and the three
sums. FastSelect evaluates ``phi1``/``phi2`` as quadratic forms without
eigenvectors, so agreement checks that identity. MCFS's oracle solves the
generalised eigenproblem ``L e = lambda D e`` directly (FastSelect uses the
normalised affinity) and checks each sparse regression against the Lasso
optimality conditions. Neither oracle calls FastSelect internals.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from fast_select import MCFS, SPEC, LaplacianScore


def _data(n=50, p=7, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(3), n // 3 + 1)[:n]
    X = rng.random((n, p)) * np.linspace(0.6, 4.0, p)
    X[:, 0] += 1.5 * groups
    X[:, 1] = 0.7 * X[:, 0] + 0.2 * rng.random(n)
    return X, groups


def _rbf(X, gamma=1.0):
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


def spec_oracle(X, S, criterion, m=10):
    degree = S.sum(axis=1)
    inv_root = 1.0 / np.sqrt(degree)
    L = np.eye(len(S)) - inv_root[:, None] * S * inv_root[None, :]
    L = (L + L.T) / 2
    xi1 = np.sqrt(degree) / np.linalg.norm(np.sqrt(degree))  # the trivial eigenvector, eigenvalue 0
    # Remove xi_1 by moving its eigenvalue to 3, above the spectrum (at most 2), so the ascending
    # eigenpairs are the nontrivial ones even when a disconnected graph repeats the eigenvalue 0.
    values, vectors = np.linalg.eigh(L + 3.0 * np.outer(xi1, xi1))
    values, vectors = values[:-1], vectors[:, :-1]
    scores = []
    for f in X.T:
        g = np.sqrt(degree) * f
        norm = np.linalg.norm(g)
        if norm == 0:
            scores.append(np.inf if criterion != "phi3" else -np.inf)
            continue
        f_hat = g / norm
        alpha = vectors.T @ f_hat
        if criterion == "phi1":
            scores.append(f_hat @ L @ f_hat)
        elif criterion == "phi2":
            rest = 1 - (f_hat @ xi1) ** 2
            scores.append(f_hat @ L @ f_hat / rest if rest > 1e-12 else np.inf)
        else:
            scores.append(np.sum((2 - values[:m]) * alpha[:m] ** 2))
    return np.array(scores)


# --------------------------------------------------------------------------- SPEC: equation oracle


@pytest.mark.parametrize("criterion", ["phi1", "phi2", "phi3"])
@pytest.mark.parametrize("affinity", ["rbf", "knn", "supervised"])
def test_spec_matches_the_eigenvector_formulas(criterion, affinity):
    X, y = _data()
    # The supervised graph's spectrum is {0 (once per class), 1}; phi3 is only well defined with
    # m = n_classes - 1, the whole nontrivial null space.
    m = 2 if affinity == "supervised" else 4
    model = SPEC(criterion=criterion, affinity=affinity, gamma=0.7, n_eigenvectors=m).fit(X, y)
    S = model.affinity_.toarray() if hasattr(model.affinity_, "toarray") else model.affinity_
    expected = spec_oracle(X, S, criterion, m=m)
    assert_allclose(model.spec_scores_, expected, rtol=1e-8, atol=1e-11)


def test_spec_graphs_are_the_published_ones():
    X, y = _data(seed=1)
    assert_allclose(SPEC(gamma=0.7).fit(X).affinity_, _rbf(X, 0.7), rtol=1e-12)
    assert_allclose(SPEC().fit(X).affinity_, _rbf(X, 1.0 / X.shape[1]), rtol=1e-12)
    supervised = SPEC(affinity="supervised").fit(X, y).affinity_
    counts = np.bincount(y)
    expected = np.where(y[:, None] == y[None, :], 1.0 / counts[y][:, None], 0.0)
    assert_array_equal(supervised, expected)
    knn = SPEC(affinity="knn", n_neighbors=4).fit(X).affinity_
    assert (knn != LaplacianScore(n_neighbors=4).fit(X).affinity_).nnz == 0


def test_spec_phi2_is_the_laplacian_score():
    X, _ = _data(seed=2)
    spec = SPEC(affinity="knn", criterion="phi2", n_neighbors=5).fit(X)
    laplacian = LaplacianScore(n_neighbors=5).fit(X)
    assert_allclose(spec.spec_scores_, laplacian.laplacian_scores_, rtol=1e-12, atol=1e-14)


def test_spec_precomputed_affinity():
    X, _ = _data(seed=3)
    S = _rbf(X, 0.3)
    model = SPEC(affinity="precomputed", criterion="phi2").fit(X, W=S)
    assert_allclose(model.spec_scores_, spec_oracle(X, S, "phi2"), rtol=1e-8)


def test_spec_ranking_direction():
    """Smaller is better for phi1/phi2, larger for phi3; with the class graph, the class feature is perfectly smooth."""
    X, y = _data(seed=4)
    X = np.column_stack([X, y.astype(float)])
    for criterion in ("phi1", "phi2"):
        model = SPEC(n_features_to_select=1, criterion=criterion, affinity="supervised").fit(X, y)
        assert model.top_features_[0] == X.shape[1] - 1
        assert_array_equal(model.feature_importances_, -model.spec_scores_)
    phi2 = SPEC(criterion="phi2", affinity="supervised").fit(X, y).spec_scores_
    assert phi2[-1] == pytest.approx(0.0, abs=1e-12)
    phi3 = SPEC(n_features_to_select=1, criterion="phi3", affinity="supervised", n_eigenvectors=2).fit(X, y)
    assert phi3.top_features_[0] == X.shape[1] - 1
    assert_array_equal(phi3.feature_importances_, phi3.spec_scores_)


def test_spec_constant_and_zero_features():
    X, _ = _data(seed=5)
    X[:, 2] = 0.1
    X[:, 3] = 0.0
    phi2 = SPEC(criterion="phi2").fit(X)
    assert phi2.spec_scores_[2] == np.inf and phi2.spec_scores_[3] == np.inf
    assert 2 not in phi2.top_features_ and 3 not in phi2.top_features_
    phi1 = SPEC(criterion="phi1").fit(X)
    assert phi1.spec_scores_[2] == pytest.approx(0.0, abs=1e-12)
    assert phi1.spec_scores_[3] == np.inf
    phi3 = SPEC(criterion="phi3", n_eigenvectors=3).fit(X)
    assert phi3.spec_scores_[2] == pytest.approx(0.0, abs=1e-10)
    assert phi3.spec_scores_[3] == -np.inf


def test_spec_phi2_affine_invariance():
    X, _ = _data(seed=6)
    reference = SPEC(criterion="phi2", affinity="knn").fit(X).spec_scores_
    moved = X.copy()
    moved[:, 4] = moved[:, 4] * 9.0 + 1e6  # an affine change of one feature, on the same graph W
    S = SPEC(criterion="phi2", affinity="knn").fit(X).affinity_
    shifted = SPEC(criterion="phi2", affinity="precomputed").fit(moved, W=S).spec_scores_
    assert_allclose(shifted, reference, rtol=1e-8)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"criterion": "phi4"}, ValueError),
        ({"affinity": "cosine"}, ValueError),
        ({"gamma": 0.0}, ValueError),
        ({"gamma": "scale"}, ValueError),
        ({"n_eigenvectors": 0, "criterion": "phi3"}, ValueError),
        ({"n_eigenvectors": 60, "criterion": "phi3"}, ValueError),
        ({"backend": "gpu"}, ValueError),
        ({"n_features_to_select": 0}, ValueError),
        ({"affinity": "precomputed"}, ValueError),
        ({"affinity": "supervised"}, ValueError),
    ],
)
def test_spec_parameter_validation(kwargs, error):
    X, _ = _data()
    with pytest.raises(error):
        SPEC(**kwargs).fit(X)


@pytest.mark.parametrize("criterion", ["phi1", "phi2", "phi3"])
def test_spec_scores_features_in_blocks(monkeypatch, criterion):
    """Feature blocks (and dense symmetry blocks) of two columns/rows must match the oracle."""
    import importlib

    module = importlib.import_module("fast_select.SPEC")
    X, _ = _data(p=9, seed=5)
    X[:, 4] = 2.5  # constant, inside a later block
    X[:, 7] = 0.0  # zero, inside the last full block
    S = _rbf(X, 0.4)
    monkeypatch.setattr(module, "_BLOCK_ELEMENTS", 2 * X.shape[0])
    model = SPEC(criterion=criterion, affinity="precomputed", n_eigenvectors=4).fit(X, W=S)
    assert_allclose(model.spec_scores_, spec_oracle(X, S, criterion, m=4), rtol=1e-8, atol=1e-11)
    late = S.copy()
    late[-1, -3] += 1e-6  # asymmetric only in the last row block
    with pytest.raises(ValueError, match="symmetric"):
        SPEC(affinity="precomputed").fit(X, W=late)


def test_spec_rbf_graph_in_column_blocks(monkeypatch):
    """Wide data builds the RBF graph from column blocks of at least n columns; it must match."""
    import importlib

    module = importlib.import_module("fast_select.SPEC")
    X, _ = _data(n=30, p=71, seed=6)
    X[:, 40] += 1e6  # a large offset inside a later block
    monkeypatch.setattr(module, "_GRAM_BLOCK_ELEMENTS", 1)  # blocks of n = 30 columns: 30 + 30 + 11
    blocked = SPEC(gamma=0.02, criterion="phi2").fit(X)
    assert_allclose(blocked.affinity_, _rbf(X, 0.02), rtol=1e-12)
    monkeypatch.undo()
    whole = SPEC(gamma=0.02, criterion="phi2").fit(X)
    assert_allclose(blocked.spec_scores_, whole.spec_scores_, rtol=1e-10)


def test_spec_rejects_asymmetric_or_negative_affinity():
    X, _ = _data()
    S = _rbf(X)
    S[0, 1] += 0.5
    with pytest.raises(ValueError, match="symmetric"):
        SPEC(affinity="precomputed").fit(X, W=S)
    with pytest.raises(ValueError, match="nonnegative"):
        SPEC(affinity="precomputed").fit(X, W=-_rbf(X))
    for bad in (np.nan, np.inf):
        W = _rbf(X)
        W[2, 3] = W[3, 2] = bad
        with pytest.raises(ValueError, match="finite"):
            SPEC(affinity="precomputed").fit(X, W=W)


def test_spec_sklearn_api_and_pipeline():
    check_estimator(SPEC())
    X, y = _data(n=90, seed=7)
    pipe = Pipeline([("select", SPEC(n_features_to_select=2)), ("model", LogisticRegression())])
    search = GridSearchCV(pipe, {"select__criterion": ["phi1", "phi2", "phi3"]}, cv=3).fit(X, y)
    assert search.best_estimator_.named_steps["select"].top_features_.size == 2


# --------------------------------------------------------------------------- MCFS


def mcfs_embedding_oracle(S, k):
    """Generalised eigenproblem L e = lambda D e: the k smallest nontrivial eigenvectors, D-normalised."""
    degree = S.sum(axis=1)
    D = np.diag(degree)
    values, vectors = scipy.linalg.eigh(D - S, D)
    return values[1 : k + 1], vectors[:, 1 : k + 1]


def _blobs(n=240, informative=4, noise=10, centers=4, seed=0, cluster_std=2.5):
    X, labels = make_blobs(n, informative, centers=centers, cluster_std=cluster_std, random_state=seed)
    rng = np.random.default_rng(seed)
    return np.column_stack([X, rng.normal(0, 3.0, (n, noise))]), labels


def test_mcfs_embedding_solves_the_generalised_eigenproblem():
    X, _ = _blobs()
    model = MCFS(n_features_to_select=3, n_clusters=3).fit(X)
    S = model.affinity_.toarray()
    assert connected_components(model.affinity_)[0] == 1  # distinct eigenvectors to compare
    values, vectors = mcfs_embedding_oracle(S, 3)
    assert_allclose(1.0 - model.eigenvalues_, values, rtol=1e-8, atol=1e-10)
    for i in range(3):  # equal up to sign
        a, b = model.embedding_[:, i], vectors[:, i]
        assert_allclose(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), 1.0, rtol=1e-8)
        assert_allclose(a @ (S.sum(axis=1) * a), 1.0, rtol=1e-8)  # e' D e = 1


def test_mcfs_graph_is_the_laplacian_score_graph():
    X, _ = _blobs(seed=1)
    for mode in ("binary", "heat_kernel"):
        model = MCFS(n_neighbors=6, weight_mode=mode, bandwidth=2.0).fit(X)
        graph = LaplacianScore(n_neighbors=6, weight_mode=mode, bandwidth=2.0).fit(X).affinity_
        assert abs(model.affinity_ - graph).max() < 1e-15


@pytest.mark.parametrize("d", [2, 5])
def test_mcfs_regressions_are_lasso_solutions_with_d_nonzeros(d):
    X, _ = _blobs(seed=2)
    model = MCFS(n_features_to_select=d, n_clusters=4).fit(X)
    Xc = X - X.mean(axis=0)
    for i in range(4):
        w = model.coef_[:, i]
        e = model.embedding_[:, i] - model.embedding_[:, i].mean()
        active = np.flatnonzero(w)
        assert active.size == d
        grad = Xc.T @ (e - Xc @ w)
        lam = np.abs(grad[active]).mean()
        assert_allclose(np.abs(grad[active]), lam, rtol=1e-6)
        assert_allclose(np.sign(grad[active]), np.sign(w[active]))
        assert np.all(np.abs(np.delete(grad, active)) <= lam * (1 + 1e-6))
    assert_allclose(model.feature_importances_, np.abs(model.coef_).max(axis=1))


def test_mcfs_disconnected_graph_uses_component_indicators():
    """Well-separated blobs give one component each: the embedding spans their indicators (minus D^1/2 1)."""
    X, labels = _blobs(n=200, informative=3, noise=2, centers=3, seed=7, cluster_std=0.3)
    model = MCFS(n_features_to_select=2, n_clusters=2).fit(X)
    assert connected_components(model.affinity_)[0] == 3
    assert_allclose(model.eigenvalues_, 1.0, atol=1e-10)
    indicators = np.column_stack([labels == c for c in range(3)]).astype(float)
    residual = model.embedding_ - indicators @ np.linalg.lstsq(indicators, model.embedding_, rcond=None)[0]
    assert np.abs(residual).max() < 1e-8  # constant within each component


def test_mcfs_selects_the_cluster_features():
    X, _ = _blobs(informative=4, noise=12, seed=3, cluster_std=1.0)
    model = MCFS(n_features_to_select=4, n_clusters=4, n_neighbors=8).fit(X)
    assert set(model.top_features_) == {0, 1, 2, 3}


def test_mcfs_translation_invariance_and_precomputed_graph():
    X, _ = _blobs(seed=4)
    reference = MCFS(n_features_to_select=3).fit(X)
    moved = MCFS(n_features_to_select=3).fit(X, W=reference.affinity_)
    assert_allclose(moved.feature_importances_, reference.feature_importances_, rtol=1e-10, atol=1e-15)
    shifted = MCFS(n_features_to_select=3).fit(X + 1e6, W=reference.affinity_)
    assert_array_equal(shifted.top_features_, reference.top_features_)
    assert_allclose(shifted.feature_importances_, reference.feature_importances_, rtol=1e-6, atol=1e-12)


def test_mcfs_large_sample_path_uses_the_sparse_solver():
    X, _ = _blobs(n=1700, informative=3, noise=5, centers=3, seed=5)
    model = MCFS(n_features_to_select=3, n_clusters=2).fit(X)
    assert connected_components(model.affinity_)[0] == 1
    values, _ = mcfs_embedding_oracle(model.affinity_.toarray(), 2)
    assert_allclose(1.0 - model.eigenvalues_, values, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_clusters": 0}, ValueError),
        ({"n_clusters": 300}, ValueError),
        ({"n_neighbors": 0}, ValueError),
        ({"weight_mode": "cosine"}, ValueError),
        ({"bandwidth": -1.0}, ValueError),
        ({"backend": "metal"}, ValueError),
        ({"n_features_to_select": 1.5}, ValueError),
    ],
)
def test_mcfs_parameter_validation(kwargs, error):
    X, _ = _blobs()
    with pytest.raises(error):
        MCFS(**kwargs).fit(X)


def test_mcfs_sklearn_api_and_pipeline():
    check_estimator(MCFS(n_clusters=2))
    X, y = _blobs(n=150, seed=6)
    pipe = Pipeline([("select", MCFS(n_features_to_select=3)), ("model", LogisticRegression(max_iter=500))])
    search = GridSearchCV(pipe, {"select__n_clusters": [2, 4]}, cv=3).fit(X, y)
    assert search.best_estimator_.named_steps["select"].top_features_.size == 3


def test_spec_phi3_sparse_eigensolver_path():
    """Above 300 samples phi3 uses ARPACK on the deflated affinity; it must match the dense oracle."""
    X, _ = _data(n=420, p=6, seed=8)
    for affinity in ("rbf", "knn"):
        model = SPEC(criterion="phi3", affinity=affinity, n_eigenvectors=5).fit(X)
        S = model.affinity_.toarray() if hasattr(model.affinity_, "toarray") else model.affinity_
        assert_allclose(model.spec_scores_, spec_oracle(X, S, "phi3", m=5), rtol=1e-7, atol=1e-10)
