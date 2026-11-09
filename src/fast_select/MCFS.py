"""MCFS: multi-cluster feature selection by spectral embedding and sparse regression.

Source
------
Cai, Zhang and He, *Unsupervised Feature Selection for Multi-Cluster Data*,
KDD 2010, DOI 10.1145/1835804.1835848. The article could not be consulted
(publisher block, author host unreachable); this module follows its
restatement in Li et al., *Feature Selection: A Data Perspective*, ACM
Computing Surveys 50(6) (2017), arXiv:1601.07996, Section 2.3.4:

1. build a p-nearest-neighbour graph ``S`` and its Laplacian ``L = D - S``;
2. embed the data with the ``K`` eigenvectors ``e_1..e_K`` of the smallest
   nontrivial eigenvalues of ``L e = lambda D e`` (spectral clustering);
3. for each ``e_i``, solve the L1-regularised regression of Eq. (30),
   ``min_w ||X w - e_i||^2 + alpha ||w||_1``, here along the LARS-Lasso path
   until ``n_features_to_select`` coefficients are nonzero (the cardinality
   constraint of the original, as in scikit-feature);
4. score each feature by ``MCFS(j) = max_i |W(j, i)|``; higher is better.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import lars_path
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cpu_backend import resolve_cpu_backend
from .LaplacianScore import _affinity
from .utils import check_integer_param

# Dense eigensolvers are exact and fast enough below this many samples.
_DENSE_EIGEN_MAX = 300


def _spectral_embedding(S, n_clusters):
    """The ``n_clusters`` generalised eigenvectors of ``L e = lambda D e`` with the smallest nontrivial eigenvalues.

    Solved as the largest eigenvectors of ``D^{-1/2} S D^{-1/2}``, dropping
    the largest (the trivial ``D^{1/2} 1``), then ``e = D^{-1/2} z``. Each
    vector is given a deterministic sign (largest-magnitude entry positive).
    """
    degree = np.asarray(S.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("Every sample needs a positive degree in the affinity graph.")
    inv_root = 1.0 / np.sqrt(degree)
    trivial = np.sqrt(degree) / np.linalg.norm(np.sqrt(degree))
    n = S.shape[0]
    k = n_clusters
    # Deflate the trivial eigenvector D^{1/2} 1 (eigenvalue 1 of the normalised affinity) to -1,
    # below the spectrum, so the k largest eigenpairs are the nontrivial ones even when a
    # disconnected graph repeats the eigenvalue 1.
    if n <= _DENSE_EIGEN_MAX or k >= n - 2:
        dense = S.toarray() if issparse(S) else np.asarray(S)
        A = inv_root[:, None] * dense * inv_root[None, :]
        A = (A + A.T) / 2 - 2.0 * np.outer(trivial, trivial)
        values, vectors = scipy.linalg.eigh(A, subset_by_index=[n - k, n - 1])
    else:
        A = diags(inv_root) @ csr_matrix(S) @ diags(inv_root)
        A = (A + A.T) / 2
        operator = LinearOperator((n, n), matvec=lambda v: A @ v - 2.0 * trivial * (trivial @ v), dtype=np.float64)
        start = np.sqrt(degree) + np.linspace(0.0, 1.0, n)  # deterministic start vector
        values, vectors = eigsh(operator, k=k, which="LA", v0=start, tol=0.0)
    order = np.argsort(values)[::-1]
    embedding = inv_root[:, None] * vectors[:, order]
    signs = np.sign(embedding[np.abs(embedding).argmax(axis=0), np.arange(embedding.shape[1])])
    embedding *= np.where(signs == 0, 1.0, signs)
    return embedding, values[order]


def _lasso_at_cardinality(X, mean, target, n_nonzero):
    """LARS-Lasso coefficients of ``X - mean`` at the first knot with ``n_nonzero`` nonzero coefficients.

    (Or at the path's end, if it never reaches that many.) The path is
    followed only as far as needed: ``n_nonzero`` steps reach the cardinality
    unless the Lasso drops a variable on the way, in which case the step
    budget doubles.
    """
    limit = min(X.shape[0], X.shape[1])
    budget = n_nonzero
    while True:
        # lars_path permutes its design's columns in place, so each call gets a fresh centred
        # copy, already in the Fortran layout lars_path would otherwise copy it into.
        design = np.subtract(X, mean, order="F")
        _, _, coefs = lars_path(design, target, method="lasso", max_iter=budget, copy_X=False)
        counts = np.count_nonzero(coefs, axis=0)
        reached = np.flatnonzero(counts >= n_nonzero)
        if reached.size:
            return coefs[:, reached[0]]
        if coefs.shape[1] <= budget or budget >= 4 * max(limit, n_nonzero):
            return coefs[:, -1]  # the path ended before reaching the cardinality
        budget *= 2


class MCFS(TransformerMixin, BaseEstimator):
    """MCFS: unsupervised selection of features that preserve the data's cluster structure.

    The samples are embedded with the leading nontrivial eigenvectors of a
    nearest-neighbour graph's Laplacian, which unfold its clusters; each
    embedding coordinate is regressed on the features with an L1 penalty, and
    a feature scores by its largest absolute coefficient in any of those
    regressions (Cai, Zhang and He, 2010). Unlike scoring features one at a
    time, the sparse regressions favour features that jointly explain
    different clusters.

    Parameters
    ----------
    n_features_to_select : int or float, default=0.2
        ``d``: the number of features to select, and the number of nonzero
        coefficients each regression stops at. A float is a fraction of the
        features.

    n_clusters : int, default=5
        ``K``, the number of embedding eigenvectors (expected clusters).

    n_neighbors : int, default=5
        ``p`` for the nearest-neighbour graph.

    weight_mode : {'binary', 'heat_kernel'}, default='binary'
        Edge weights of the graph, as in :class:`LaplacianScore`.

    bandwidth : float, default=1.0
        Heat-kernel width ``exp(-||x_i - x_j||^2 / (2 bandwidth^2))``.

    backend : {'auto', 'cpu'}, default='auto'
        MCFS runs on the CPU only.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The MCFS score ``max_i |W(j, i)|``.

    coef_ : ndarray of shape (n_features, n_clusters)
        The sparse regression coefficients ``W``.

    embedding_ : ndarray of shape (n_samples, n_clusters)
        The spectral embedding ``e_1..e_K``.

    eigenvalues_ : ndarray of shape (n_clusters,)
        Eigenvalues of ``D^{-1/2} S D^{-1/2}`` for the embedding vectors
        (``1 - lambda`` of the normalised Laplacian), largest first.

    top_features_ : ndarray of shape (n_select,)
        Indices of the selected features, best first (ties: lower index).

    affinity_ : sparse matrix of shape (n_samples, n_samples)
        The graph used.

    effective_backend_ : str
        Always ``'cpu'``.

    Notes
    -----
    The original article could not be consulted; this follows the
    restatement in Li et al. (2017). Conventions: the graph is the symmetric
    Euclidean kNN graph with self edges of :class:`LaplacianScore` (also
    accepted precomputed through ``fit(..., W=...)``); each regression fits
    an intercept (features and target centred), so scores do not depend on
    feature offsets, but features are not rescaled (Eq. 30 regresses on ``X``
    as given; standardise first for scale-free scores); the regression follows
    the LARS-Lasso path to the first knot with ``d`` nonzero coefficients.
    The trivial eigenvector ``D^{1/2} 1`` is removed explicitly, so a
    disconnected graph keeps its component indicators as embedding vectors.
    Eigenvectors are only defined up to sign (fixed here) and, for repeated
    eigenvalues, up to rotation, which can change scores. scikit-feature
    scores by the largest signed coefficient instead of the largest absolute
    one, and symmetrises the graph after normalising it; neither is copied.
    Input containing NaNs is rejected.
    """

    def __init__(
        self,
        n_features_to_select: int | float = 0.2,
        n_clusters: int = 5,
        n_neighbors: int = 5,
        weight_mode: str = "binary",
        bandwidth: float = 1.0,
        backend: str = "auto",
    ):
        self.n_features_to_select = n_features_to_select
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.weight_mode = weight_mode
        self.bandwidth = bandwidth
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

    def fit(self, X, y=None, *, W=None):
        """Embed the samples, run the sparse regressions and score the features."""
        X = validate_data(self, X, dtype=np.float64, ensure_2d=True, ensure_min_samples=3)
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        n_samples, n_features = X.shape
        n_select = self._n_select(n_features)
        n_clusters = check_integer_param(self.n_clusters, "n_clusters", 1)
        if n_clusters + 1 >= n_samples:
            raise ValueError(f"n_clusters ({n_clusters}) must be less than n_samples - 1 ({n_samples - 1}).")
        if W is None:
            k = check_integer_param(self.n_neighbors, "n_neighbors", 1)
            if self.weight_mode not in ("binary", "heat_kernel"):
                raise ValueError("weight_mode must be 'binary' or 'heat_kernel'.")
            bandwidth = self.bandwidth
            if isinstance(bandwidth, (bool, np.bool_)) or not np.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError(f"bandwidth must be a positive finite number, got {bandwidth!r}.")
            S = _affinity(X, k, self.weight_mode, float(bandwidth))
        else:
            S = csr_matrix(W, dtype=np.float64)
            if S.shape != (n_samples, n_samples) or not np.all(np.isfinite(S.data)) or np.any(S.data < 0):
                raise ValueError("W must be a finite nonnegative square affinity matching the sample count.")
            if abs(S - S.T).max() > 1e-12 * max(1.0, abs(S.data).max() if S.nnz else 1.0):
                raise ValueError("W must be symmetric.")

        embedding, eigenvalues = _spectral_embedding(S, n_clusters)
        mean = X.mean(axis=0)
        coef = np.empty((n_features, n_clusters))
        for i in range(n_clusters):
            target = embedding[:, i] - embedding[:, i].mean()
            coef[:, i] = _lasso_at_cardinality(X, mean, target, n_select)

        self.affinity_ = S
        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.coef_ = coef
        self.feature_importances_ = np.abs(coef).max(axis=1)
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
