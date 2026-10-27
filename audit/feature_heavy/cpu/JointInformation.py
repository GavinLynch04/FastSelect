"""CIFE (Lin & Tang, 2006), DISR (Meyer & Bontempi, 2006), MIFS (Battiti, 1994).

Empirical categorical classification; bits; no automatic discretization.
Equations follow Brown et al. (JMLR 2012). Existing tested Numba entropy
kernels are reused without modifying them. Greedy selection stays sequential.
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, set_num_threads
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data

from fast_select._cpu_backend import resolve_cpu_backend
from fast_select.InformationSelectors import _entropy_one, _first_codes, _InformationSelector, _initial, _step_cpu
from fast_select.mRMR import _encode_categories
from fast_select.utils import check_integer_param, resolve_num_threads


class _JointSelector(_InformationSelector):
    def _fit_joint(self, X, y, criterion):
        if y is None:
            raise ValueError("This selector requires y to be passed, but the target y is None")
        X, y = validate_data(self, X, y, dtype=None, ensure_min_samples=2)
        check_classification_targets(y)
        check_integer_param(self.n_features_to_select, "n_features_to_select", 1)
        if self.n_features_to_select > X.shape[1]:
            raise ValueError("n_features_to_select must not exceed the feature count")
        if criterion == "MIFS":
            if isinstance(self.beta, (bool, np.bool_)) or not isinstance(self.beta, (int, float, np.number)):
                raise ValueError("beta must be a finite nonnegative number")
            if not np.isfinite(self.beta) or self.beta < 0:
                raise ValueError("beta must be a finite nonnegative number")
        self.effective_backend_ = resolve_cpu_backend(self.backend)
        threads = resolve_num_threads(self.n_jobs)
        encoded, labels, counts, sy = _encode_categories(X, y)
        labels = _first_codes(labels)
        previous = get_num_threads()
        try:
            set_num_threads(threads)
            columns, entropy, joint = _initial(np.ascontiguousarray(encoded.T), labels, counts, sy)
            hy = _entropy_one(labels, sy)
            relevance = -joint + entropy + hy
            self.feature_importances_ = relevance.copy()
            self.relevance_scores_ = relevance.copy()
            active = np.ones(len(counts), dtype=bool)
            redundancy = np.zeros(len(counts))
            conditional_sum = np.zeros(len(counts))
            current = relevance.copy()
            selected, objectives = [], []
            for step in range(self.n_features_to_select):
                idx = int(np.argmax(np.where(active, current, -np.inf)))
                selected.append(idx)
                objectives.append(current[idx])
                active[idx] = False
                if step + 1 == self.n_features_to_select:
                    break
                hp, ht = _step_cpu(columns, labels, counts, sy, idx, active, criterion != "MIFS")
                if criterion == "CIFE":
                    redundancy += -hp + entropy[idx] + entropy
                    conditional_sum += joint + joint[idx] - ht - hy
                    current = relevance - redundancy + conditional_sum
                elif criterion == "MIFS":
                    redundancy += -hp + entropy[idx] + entropy
                    current = relevance - self.beta * redundancy
                else:
                    # I((candidate, selected); target) / H(candidate,selected,target).
                    increment = np.divide(hp + hy - ht, ht, out=np.zeros_like(ht), where=ht > 0)
                    conditional_sum += increment
                    current = conditional_sum.copy()
            self.top_features_ = np.asarray(selected, dtype=np.int64)
            self.selection_scores_ = np.asarray(objectives, dtype=np.float64)
        finally:
            set_num_threads(previous)
        return self


class CIFE(_JointSelector):
    """Conditional Informative Feature Extraction, categorical classification.

    Maximize I(candidate;target) - sum I(candidate;selected) +
    sum I(candidate;selected|target). Initialize with maximum marginal MI.
    Unlike JMI, the redundancy/dependence sums are not divided by subset size.
    Importances are marginal MI in bits; selection_scores_ is the CIFE objective.
    CPU/auto only. Constructor, tags and transform follow the existing selectors.
    """

    def fit(self, X, y):
        return self._fit_joint(X, y, "CIFE")


class DISR(_JointSelector):
    """Double Input Symmetrical Relevance, categorical classification.

    Initialize with maximum marginal MI, then maximize the sum of
    I((candidate,selected);target)/H(candidate,selected,target). A zero joint
    entropy contributes zero (explicit constant-variable extension).
    Importances are marginal MI in bits; selection_scores_ is the DISR objective.
    CPU/auto only; fixed requested feature count, with lower-index exact ties.
    """

    def fit(self, X, y):
        return self._fit_joint(X, y, "DISR")


class MIFS(_JointSelector):
    """Battiti (1994) Mutual Information Feature Selection, categorical inputs.

    Maximize marginal MI minus beta times the summed MI with selected features.
    The default beta is 0.5. There is no subset-size averaging or conditional
    redundancy term. Fixed requested count; exact score ties use lower indices.
    Empirical entropy is in bits, with no implicit discretization. CPU/auto only.
    """

    def __init__(self, n_features_to_select=10, beta=0.5, backend="auto", n_jobs=-1):
        self.n_features_to_select = n_features_to_select
        self.beta = beta
        self.backend = backend
        self.n_jobs = n_jobs

    def fit(self, X, y):
        return self._fit_joint(X, y, "MIFS")
