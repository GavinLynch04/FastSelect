"""SPEC: spectral feature selection on a similarity graph.

Source
------
Zhao and Liu, *Spectral Feature Selection for Supervised and Unsupervised
Learning*, ICML 2007, DOI 10.1145/1273496.1273641. The article could not be
consulted (publisher login, and its proceedings copy is gone); this module
follows its restatement in Li et al., *Feature Selection: A Data Perspective*,
ACM Computing Surveys 50(6) (2017), arXiv:1601.07996, Section 2.1.2, Eq. (4).

With a similarity matrix ``S``, degrees ``D``, the normalised Laplacian
``L = D^{-1/2} (D - S) D^{-1/2}`` with eigenpairs ``(lambda_j, xi_j)`` in
ascending order, and ``f_hat = D^{1/2} f / ||D^{1/2} f||``, ``alpha_j =
xi_j' f_hat``, the three criteria with ``gamma(lambda) = lambda`` (the
noise-free case) are::

    phi1(f) = f_hat' L f_hat                          smaller is better
    phi2(f) = f_hat' L f_hat / (1 - (f_hat' xi_1)^2)  smaller is better
    phi3(f) = sum_{j=2}^{m+1} (2 - lambda_j) alpha_j^2  larger is better

``phi2`` equals the Laplacian Score on the same graph. ``phi1`` and ``phi2``
need no eigendecomposition: ``f_hat' L f_hat = f'(D - S) f / f' D f`` and
``xi_1 = D^{1/2} 1 / ||D^{1/2} 1||``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import row_norms
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cpu_backend import resolve_cpu_backend
from .LaplacianScore import _affinity
from .utils import check_integer_param

CRITERIA = ("phi1", "phi2", "phi3")
AFFINITIES = ("rbf", "knn", "supervised", "precomputed")
# Dense eigensolvers are exact and fast enough below this many samples (as in MCFS).
_DENSE_EIGEN_MAX = 300
# Features (and dense symmetry checks) are processed in blocks of about this many
# elements, so temporaries stay near 2 MiB instead of growing with n_features.
_BLOCK_ELEMENTS = 1 << 18
# The RBF graph's centred copy of X is built whole up to this many elements (64 MiB), and in
# column blocks of this size beyond it: small blocks cost BLAS efficiency (about 1.6x on a
# 500 x 5000 graph), so blocking is kept for inputs whose copy would be large.
_GRAM_BLOCK_ELEMENTS = 1 << 23


def _smallest_nontrivial_eigenpairs(S, root, m):
    """The ``m`` smallest eigenpairs of the normalised Laplacian orthogonal to ``xi_1 = D^{1/2} 1``.

    They are the ``m`` largest of the normalised affinity ``A = D^{-1/2} S D^{-1/2}``
    (``lambda_L = 1 - lambda_A``) after deflating ``xi_1`` (eigenvalue 1 of ``A``) to -1, so the
    result is the nontrivial part even when a disconnected graph repeats the eigenvalue. Dense
    LAPACK for small graphs, ARPACK with a deterministic start vector otherwise.
    """
    n = len(root)
    inv_root = 1.0 / root
    trivial = root / np.linalg.norm(root)
    if n <= _DENSE_EIGEN_MAX or m >= n - 2:
        dense = S.toarray() if issparse(S) else np.asarray(S)
        A = inv_root[:, None] * dense * inv_root[None, :]
        A = (A + A.T) / 2 - 2.0 * np.outer(trivial, trivial)
        values, vectors = scipy.linalg.eigh(A, subset_by_index=[n - m, n - 1])
    else:
        if issparse(S):
            A = diags(inv_root) @ csr_matrix(S) @ diags(inv_root)

            def scaled(v):
                return A @ v
        else:
            # A dense A = D^{-1/2} S D^{-1/2} would be a second n-by-n matrix; apply the scaling to vectors.
            def scaled(v):
                return inv_root * (S @ (inv_root * v))

        def matvec(v):
            v = np.ravel(v)
            return scaled(v) - 2.0 * trivial * (trivial @ v)

        operator = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        values, vectors = eigsh(operator, k=m, which="LA", v0=root + np.linspace(0.0, 1.0, n), tol=0.0)
    order = np.argsort(values)[::-1]
    return 1.0 - values[order], vectors[:, order]


def _rbf_similarity(X, gamma):
    """Dense ``exp(-gamma ||x_i - x_j||^2)``, with the self similarity 1 on the diagonal.

    The squared distances are scikit-learn's ``euclidean_distances`` expansion
    (``-2 x_i'x_j + ||x_i||^2 + ||x_j||^2``, clipped at zero, on centred data),
    evaluated in place so that at most two n-by-n matrices are allocated.
    """
    n, p = X.shape
    mean = X.mean(axis=0)
    # Centring avoids cancellation in the expansion. Very wide data is centred and multiplied in
    # column blocks (at least n columns, so a block costs no more memory than the n-by-n result).
    step = max(n, _GRAM_BLOCK_ELEMENTS // max(n, 1))
    norms = np.zeros(n)
    distances = None
    for start in range(0, p, step):
        centred = X[:, start : start + step] - mean[start : start + step]
        norms += row_norms(centred, squared=True)
        product = centred @ centred.T
        if distances is None:
            distances = product
        else:
            distances += product
    distances *= -2.0
    distances += norms[:, None]
    distances += norms[None, :]
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    distances *= -gamma
    return np.exp(distances, out=distances)


def _max_asymmetry(S):
    """``max |S - S'|``; dense matrices are compared in row blocks to avoid n-by-n temporaries."""
    if issparse(S):
        return abs(S - S.T).max() if S.nnz else 0.0
    n = S.shape[0]
    step = max(1, _BLOCK_ELEMENTS // max(n, 1))
    worst = 0.0
    for start in range(0, n, step):
        stop = min(n, start + step)
        worst = max(worst, float(np.abs(S[start:stop] - S[:, start:stop].T).max()))
    return worst


class SPEC(TransformerMixin, BaseEstimator):
    """SPEC: rank features by how smoothly they vary over a similarity graph.

    A feature that takes similar values on similar samples (or within a class,
    in the supervised setting) concentrates on the low-frequency eigenvectors
    of the graph's normalised Laplacian. SPEC (Zhao and Liu, 2007) measures
    this in three ways (``criterion``); ``'phi2'`` is the Laplacian Score.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        The number of top features to select. A float selects that fraction of
        the features; an int selects that many.

    criterion : {'phi1', 'phi2', 'phi3'}, default='phi2'
        ``'phi1'``: ``f_hat' L f_hat``; ``'phi2'``: the same, normalised by the
        part of ``f_hat`` off the trivial eigenvector (the Laplacian Score);
        ``'phi3'``: ``sum (2 - lambda_j) alpha_j^2`` over the
        ``n_eigenvectors`` smallest nontrivial eigenpairs.

    affinity : {'rbf', 'knn', 'supervised', 'precomputed'}, default='rbf'
        The similarity graph. ``'rbf'``: ``exp(-gamma ||x_i - x_j||^2)`` over
        all pairs (unsupervised, as in the paper). ``'knn'``: the symmetric
        Euclidean ``n_neighbors`` graph of :class:`LaplacianScore`.
        ``'supervised'``: ``S_ij = 1 / n_l`` when ``y_i = y_j = l``, else 0;
        needs ``y``. ``'precomputed'``: pass ``W`` to :meth:`fit`.

    gamma : float or None, default=None
        RBF width for ``affinity='rbf'``; None uses ``1 / n_features``, as
        scikit-learn's ``rbf_kernel`` does. (scikit-feature uses 1, which on
        many standardised features drives every off-diagonal similarity to
        about zero.)

    n_neighbors : int, default=5
        Neighbours for ``affinity='knn'``.

    n_eigenvectors : int, default=10
        ``m``, the nontrivial eigenpairs used by ``'phi3'``.

    backend : {'auto', 'cpu'}, default='auto'
        SPEC runs on the CPU only.

    Attributes
    ----------
    spec_scores_ : ndarray of shape (n_features,)
        The raw criterion; lower is better for ``'phi1'``/``'phi2'``, higher
        for ``'phi3'``.

    feature_importances_ : ndarray of shape (n_features,)
        Higher-is-better scores: ``-spec_scores_`` for ``'phi1'``/``'phi2'``,
        ``spec_scores_`` for ``'phi3'``.

    top_features_ : ndarray of shape (n_select,)
        Indices of the selected features, best first (ties: lower index).

    affinity_ : ndarray or sparse matrix of shape (n_samples, n_samples)
        The similarity graph used.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    The original article could not be consulted; this follows the restatement
    in Li et al. (2017), Eq. (4), with ``gamma(lambda) = lambda``. Where that
    restatement and the scikit-feature implementation differ, ``'phi3'`` skips
    the trivial eigenpair (``j = 2..m+1``), as scikit-feature does: including
    it would reward features aligned with ``D^{1/2} 1``. A feature with
    ``||D^{1/2} f|| = 0`` (all zeros) is undefined and ranks last; one parallel
    to the trivial eigenvector (a constant feature) has ``phi2 = 0 / 0``,
    which is treated as undefined (``+inf``, ranked last) as in
    :class:`LaplacianScore`, while ``phi1`` gives it 0. The trivial
    eigenvector is taken as ``xi_1 = D^{1/2} 1 / ||D^{1/2} 1||`` even when a
    disconnected graph (e.g. the supervised one) repeats the eigenvalue 0;
    ``'phi3'`` is still ambiguous when an eigenvalue repeats across its cut
    (the ``m``-th and ``m+1``-th nontrivial eigenvalues are equal). scikit-feature ranks
    ``phi1``/``phi2`` in descending and ``phi3`` in ascending order, the
    opposite of the criteria's meaning; that is not copied. The RBF graph and
    the eigenproblem of ``'phi3'`` take O(n^2) memory. Input containing NaNs
    is rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        criterion: str = "phi2",
        affinity: str = "rbf",
        gamma: float | None = None,
        n_neighbors: int = 5,
        n_eigenvectors: int = 10,
        backend: str = "auto",
    ):
        self.n_features_to_select = n_features_to_select
        self.criterion = criterion
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_eigenvectors = n_eigenvectors
        self.backend = backend

    def _n_select(self, n_features):
        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("If n_features_to_select is a float, it must be in (0, 1].")
            return max(1, int(self.n_features_to_select * n_features))
        if isinstance(self.n_features_to_select, (int, np.integer)) and not isinstance(
            self.n_features_to_select, (bool, np.bool_)
        ):
            if not 0 < self.n_features_to_select <= n_features:
                raise ValueError(
                    f"If n_features_to_select is an int ({self.n_features_to_select}), "
                    f"it must be > 0 and <= n_features (got n_features={n_features})."
                )
            return int(self.n_features_to_select)
        raise TypeError("n_features_to_select must be an int or a float.")

    def _graph(self, X, y, W):
        n = X.shape[0]
        if self.affinity == "precomputed":
            if W is None:
                raise ValueError("affinity='precomputed' needs W passed to fit.")
            S = csr_matrix(W, dtype=np.float64) if issparse(W) else np.asarray(W, dtype=np.float64)
            values = S.data if issparse(S) else S
            # min/max propagate NaN and expose infinities without n-by-n temporaries.
            low, high = (float(values.min()), float(values.max())) if values.size else (0.0, 0.0)
            if S.shape != (n, n) or not (low >= 0.0 and np.isfinite(high)):
                raise ValueError("W must be a finite nonnegative square affinity matching the sample count.")
            if _max_asymmetry(S) > 1e-12 * max(1.0, high):
                raise ValueError("W must be symmetric.")
            return S
        if W is not None:
            raise ValueError("W is only used with affinity='precomputed'.")
        if self.affinity == "rbf":
            gamma = 1.0 / X.shape[1] if self.gamma is None else self.gamma
            if (
                isinstance(gamma, (bool, np.bool_))
                or not isinstance(gamma, (int, float, np.integer, np.floating))
                or not np.isfinite(gamma)
                or gamma <= 0
            ):
                raise ValueError(f"gamma must be None or a positive finite number, got {gamma!r}.")
            return _rbf_similarity(X, float(gamma))
        if self.affinity == "knn":
            k = check_integer_param(self.n_neighbors, "n_neighbors", 1)
            return _affinity(X, k, "binary", 1.0)
        if y is None:
            raise ValueError("affinity='supervised' needs class labels y.")
        y = np.asarray(y)
        check_classification_targets(y)
        if y.shape[0] != n:
            raise ValueError("y must have one label per sample.")
        _, codes = np.unique(y, return_inverse=True)
        counts = np.bincount(codes).astype(np.float64)
        same = codes[:, None] == codes[None, :]
        return np.where(same, 1.0 / counts[codes][:, None], 0.0)

    def fit(self, X, y=None, *, W=None):
        """Score every feature on the similarity graph."""
        X = validate_data(self, X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2)
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {self.criterion!r}.")
        if self.affinity not in AFFINITIES:
            raise ValueError(f"affinity must be one of {AFFINITIES}, got {self.affinity!r}.")
        n_samples, n_features = X.shape
        n_select = self._n_select(n_features)

        S = self._graph(X, y, W)
        degree = np.asarray(S.sum(axis=1)).ravel()
        if np.any(degree <= 0):
            raise ValueError("Every sample needs a positive degree in the similarity graph.")
        root = np.sqrt(degree)
        volume = degree.sum()
        constant = X.max(axis=0) == X.min(axis=0)
        spectral = self.criterion == "phi3"
        if spectral:
            m = check_integer_param(self.n_eigenvectors, "n_eigenvectors", 1)
            if m + 1 > n_samples:
                raise ValueError(f"n_eigenvectors ({m}) must be less than n_samples ({n_samples}).")
            values, vectors = _smallest_nontrivial_eigenpairs(S, root, m)
            # alpha_j = xi_j' D^{1/2} f / ||D^{1/2} f||; the numerator is (D^{1/2} xi_j)' f.
            lifted = vectors * root[:, None]
            alpha = np.empty((m, n_features))
        else:
            energy = np.empty(n_features)
            off = np.empty(n_features)
        mass = np.empty(n_features)  # f'Df

        # Features are independent, so they are scored in column blocks: the
        # temporaries are n-by-block instead of n-by-n_features.
        step = max(1, _BLOCK_ELEMENTS // n_samples)
        for start in range(0, n_features, step):
            block = slice(start, min(n_features, start + step))
            values_b = X[:, block]
            mass[block] = np.einsum("ij,ij->j", values_b, values_b * degree[:, None])
            if spectral:
                alpha[:, block] = lifted.T @ values_b
                continue
            # Centre on the degree-weighted mean: (D - S) annihilates constants, so the
            # energy f'(D - S)f is unchanged, and f~'Df~ = f'Df (1 - (f_hat' xi_1)^2)
            # is computed without cancellation.
            centred = values_b - (degree @ values_b) / volume
            centred[:, constant[block]] = 0.0
            off[block] = np.einsum("ij,ij->j", centred, centred * degree[:, None])  # f~'Df~
            energy[block] = off[block] - np.einsum("ij,ij->j", centred, np.asarray(S @ centred))

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.criterion == "phi1":
                energy = np.maximum(energy, 0.0)
                scores = np.where(mass > 0, energy / mass, np.inf)
            elif self.criterion == "phi2":
                energy = np.maximum(energy, 0.0)
                # A constant feature lies along xi_1 = D^{1/2} 1 / ||D^{1/2} 1||: 0 / 0, undefined.
                scores = np.where((off > 0) & ~constant, energy / off, np.inf)
            else:
                norms = np.sqrt(mass)
                alpha /= np.where(norms > 0, norms, 1.0)
                scores = ((2.0 - values)[:, None] * alpha * alpha).sum(axis=0)
                scores = np.where(norms > 0, scores, -np.inf)

        self.affinity_ = S
        self.spec_scores_ = scores
        if self.criterion == "phi3":
            self.feature_importances_ = scores.copy()
        else:
            self.feature_importances_ = -scores
        self.top_features_ = np.argsort(-self.feature_importances_, kind="stable")[:n_select]
        return self

    def transform(self, X):
        """Reduce ``X`` to the selected features."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X[:, self.top_features_]

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it."""
        return self.fit(X, y, **fit_params).transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        return tags
