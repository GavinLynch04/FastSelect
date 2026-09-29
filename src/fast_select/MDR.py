from collections import Counter
from itertools import combinations

import numba
import numpy as np
from numba import cuda, njit, prange
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from .utils import check_integer_param, ensure_cuda_context, is_cuda_ready

MAX_K_FOR_KERNEL = 6
MAX_CELLS = 3**MAX_K_FOR_KERNEL


@cuda.jit
def mdr_kernel(X_d, y_d, k, combinations_d, results_d):  # pragma: no cover
    """CUDA kernel computing balanced accuracy for every k-locus model."""
    thread_idx = cuda.grid(1)
    n_combinations = combinations_d.shape[0]

    if thread_idx >= n_combinations:
        return

    case_counts = cuda.local.array(shape=MAX_CELLS, dtype=numba.int32)
    control_counts = cuda.local.array(shape=MAX_CELLS, dtype=numba.int32)

    for i in range(3**k):
        case_counts[i] = 0
        control_counts[i] = 0

    n_samples = X_d.shape[0]
    total_cases = 0

    # Build contingency table
    for i in range(n_samples):
        cell_idx = 0
        for j in range(k):
            feature_index = combinations_d[thread_idx, j]
            genotype = X_d[i, feature_index]
            cell_idx = cell_idx * 3 + genotype

        if y_d[i] == 1:
            case_counts[cell_idx] += 1
        else:
            control_counts[cell_idx] += 1

    for i in range(3**k):
        total_cases += case_counts[i]

    total_controls = n_samples - total_cases

    if total_cases == 0 or total_controls == 0:
        results_d[thread_idx] = 0.0
        return

    threshold_ratio = total_cases / total_controls

    tp = 0
    tn = 0

    for i in range(3**k):
        if control_counts[i] == 0:
            is_high_risk = case_counts[i] > 0
        else:
            is_high_risk = (case_counts[i] / control_counts[i]) >= threshold_ratio

        if is_high_risk:
            tp += case_counts[i]
        else:
            tn += control_counts[i]

    sensitivity = tp / total_cases
    specificity = tn / total_controls
    results_d[thread_idx] = (sensitivity + specificity) / 2.0


@njit(parallel=True, fastmath=True)
def _batch_balanced_accuracy_cpu(X, y, combos, k):  # pragma: no cover
    """
    Compute balanced accuracy for *all* combinations in `combos`
    (shape = (n_combos, k)).  Returns float32 array of length n_combos.
    """
    n_combos = combos.shape[0]
    n_samples = X.shape[0]
    n_cells = 3**k
    bas = np.empty(n_combos, dtype=np.float32)

    for c_idx in prange(n_combos):
        case = np.zeros(n_cells, dtype=np.int32)
        control = np.zeros(n_cells, dtype=np.int32)

        # Fill contingency table for this combination
        for i in range(n_samples):
            cell = 0
            for j in range(k):
                cell = cell * 3 + X[i, combos[c_idx, j]]
            if y[i] == 1:
                case[cell] += 1
            else:
                control[cell] += 1

        total_case = 0
        for i in range(n_cells):
            total_case += case[i]
        total_control = n_samples - total_case

        if total_case == 0 or total_control == 0:
            bas[c_idx] = 0.0
            continue

        thr = total_case / total_control
        tp = 0
        tn = 0
        for i in range(n_cells):
            if (control[i] == 0 and case[i] > 0) or (control[i] > 0 and (case[i] / control[i]) >= thr):
                tp += case[i]
            else:
                tn += control[i]

        sens = tp / total_case
        spec = tn / total_control
        bas[c_idx] = (sens + spec) / 2.0

    return bas


@njit(fastmath=True)
def _predict_lut(X, interaction_indices, lookup_table):  # pragma: no cover
    """Fast MDR prediction using a lookup table (Numba-compiled)."""
    n_samples = X.shape[0]
    k = interaction_indices.shape[0]
    y_pred = np.empty(n_samples, dtype=np.uint8)

    for i in range(n_samples):
        cell_idx = 0
        for j in range(k):
            cell_idx = cell_idx * 3 + X[i, interaction_indices[j]]
        y_pred[i] = lookup_table[cell_idx]

    return y_pred


class MDR(ClassifierMixin, TransformerMixin, BaseEstimator):
    """
    Multifactor Dimensionality Reduction with GPU or CPU backend. This implementation targets the canonical
    use-case of MDR: SNP genotypes coded 0, 1, 2. All features must take exactly three discrete values (0/1/2);
    other data types should be encoded or discretised accordingly before calling fit.

    Parameters
    ----------
    k : int, default=2
        Interaction order to search (e.g. k=2 - pairwise). Max is 6, and this is only feasible with
        powerful hardware (and lots of memory), or with very small datasets.

    cv : int, default=10
        Stratified K-folds for model selection.

    backend : {'auto', 'cpu', 'gpu'}, default='auto'
        Execution backend preference (case-insensitive; stored unchanged and
        normalised when fitting so the estimator can be cloned).

    verbose : bool, default=False
        Print progress information during training.

    Attributes
    ----------
    best_mean_testing_ba_ : float
        Mean testing balanced accuracy over only those cross-validation folds
        in which the selected interaction was the fold winner. It is *not* an
        out-of-fold estimate of that fixed interaction's performance; use an
        outer evaluation to report predictive performance.

    Notes
    -----
    Variant implemented: exhaustive k-way search, training balanced accuracy to
    pick each fold's model, a cell is high risk when its case/control ratio is
    greater than or equal to the overall case/control ratio (Ritchie et al.,
    2001, with balanced accuracy for imbalanced data), and cross-validation
    consistency, then mean testing balanced accuracy, to choose the final model.
    Genotypes must be exactly 0, 1 or 2; other values are rejected, never cast.
    """

    def __init__(self, k: int = 2, cv: int = 10, backend: str = "auto", verbose: bool = False):
        # Parameters are stored exactly as given (scikit-learn contract); they
        # are normalised and validated in ``fit``.
        self.k = k
        self.cv = cv
        self.backend = backend
        self.verbose = verbose

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Genotypes are 0/1/2; negative values are rejected.
        tags.input_tags.positive_only = True
        return tags

    @staticmethod
    def _check_genotypes(X):
        """Validate genotypes *before* any narrowing cast and return them as uint8."""
        if X.dtype.kind not in "biuf":
            raise ValueError("Genotypes must be numeric and coded exactly 0, 1 or 2.")
        if not np.isin(X, (0, 1, 2)).all():
            raise ValueError("Genotypes must be coded exactly 0, 1 or 2 (integral, no other values).")
        return np.ascontiguousarray(X, dtype=np.uint8)

    def _create_lookup_table(self, X, y, interaction_indices):
        """Return 3^k binary lookup table for the given interaction."""
        n_cells = 3**self.k
        # Explicit int64 cell indices: uint8 genotype arithmetic wraps for k >= 5.
        cells = np.zeros(X.shape[0], dtype=np.int64)
        for idx in interaction_indices:
            cells = cells * 3 + X[:, idx].astype(np.int64)
        case_counts = np.bincount(cells[y == 1], minlength=n_cells).astype(np.int64)
        control_counts = np.bincount(cells[y != 1], minlength=n_cells).astype(np.int64)

        total_cases = case_counts.sum()
        total_controls = control_counts.sum()
        threshold = np.inf if total_controls == 0 else total_cases / total_controls
        lookup = np.zeros(n_cells, dtype=np.uint8)
        for cell in range(n_cells):
            if control_counts[cell] == 0:
                lookup[cell] = case_counts[cell] > 0
            else:
                lookup[cell] = (case_counts[cell] / control_counts[cell]) >= threshold
        return lookup

    def _internal_predict(self, X, interaction, lookup_table):
        """Predict labels using Numba-compiled LUT helper."""
        interaction_arr = np.asarray(interaction, dtype=np.uint32)
        return _predict_lut(X, interaction_arr, lookup_table)

    def fit(self, X, y):
        """
        Fits the MDR model to find the best feature subset by evaluating feature
        correlation with the target and inter-feature correlation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be continuous, discrete, or mixed.
        y : array-like of shape (n_samples,)
            Target values. Must be discrete (Classification).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(self, X, y, dtype="numeric")
        check_classification_targets(y)
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("MDR only supports binary classification.")
        y_encoded = (y == self.classes_[1]).astype(np.uint8)
        X = self._check_genotypes(X)
        check_integer_param(self.k, "k", 1)
        check_integer_param(self.cv, "cv", 2)
        if self.k > MAX_K_FOR_KERNEL:
            raise ValueError(f"k={self.k} exceeds MAX_K_FOR_KERNEL={MAX_K_FOR_KERNEL}.")

        n_samples, n_features = X.shape
        if self.k > n_features:
            raise ValueError(f"k must be <= n_features. Got k={self.k}, n_features={n_features}")

        # Decide backend
        backend = self.backend.lower() if isinstance(self.backend, str) else self.backend
        if backend not in ("auto", "cpu", "gpu"):
            raise ValueError("backend must be 'auto', 'cpu', or 'gpu' (case-insensitive).")
        cuda_available = is_cuda_ready()
        if backend == "gpu" and not cuda_available:
            raise RuntimeError("backend='GPU' requested but no CUDA device found.")
        use_gpu = (backend == "gpu") or (backend == "auto" and cuda_available)
        if use_gpu:
            ensure_cuda_context()

        # Pre-compute all k-feature combos
        feature_idx = np.arange(n_features, dtype=np.uint32)
        all_combos = np.array(list(combinations(feature_idx, self.k)), dtype=np.uint32)
        n_combos = len(all_combos)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        fold_best_models = []
        fold_test_bas = []
        if self.verbose:
            print(f"CV with backend={'GPU' if use_gpu else 'CPU'}: " f"{self.k}-way search over {n_combos} combos")

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            if use_gpu:
                X_d = cuda.to_device(X_train)
                y_d = cuda.to_device(y_train)
                combos_d = cuda.to_device(all_combos)
                results_d = cuda.device_array(n_combos, dtype=np.float32)

                threads = 128
                blocks = (n_combos + threads - 1) // threads
                mdr_kernel[blocks, threads](X_d, y_d, self.k, combos_d, results_d)
                train_bas = results_d.copy_to_host()
            else:
                train_bas = _batch_balanced_accuracy_cpu(X_train, y_train, all_combos, self.k)

            best_idx = int(np.argmax(train_bas))
            best_combo = tuple(all_combos[best_idx])
            fold_best_models.append(best_combo)

            lookup = self._create_lookup_table(X_train, y_train, best_combo)
            y_pred_test = self._internal_predict(X_test, best_combo, lookup)

            tp = np.sum((y_test == 1) & (y_pred_test == 1))
            tn = np.sum((y_test == 0) & (y_pred_test == 0))
            n_pos = np.sum(y_test == 1)
            n_neg = np.sum(y_test == 0)
            sens = tp / n_pos if n_pos else 0
            spec = tn / n_neg if n_neg else 0
            test_ba = (sens + spec) / 2.0
            fold_test_bas.append(test_ba)

            if self.verbose:
                print(f"  Fold {fold_i}/{self.cv}: best {best_combo}, " f"Test BA = {test_ba:.4f}")

        counts = Counter(fold_best_models)
        max_cvc = counts.most_common(1)[0][1]
        top_models = [m for m, c in counts.items() if c == max_cvc]

        best_model = None
        best_avg_ba = -1.0
        for model in top_models:
            bas = [fold_test_bas[i] for i, m in enumerate(fold_best_models) if m == model]
            avg_ba = float(np.mean(bas))
            if avg_ba > best_avg_ba:
                best_avg_ba = avg_ba
                best_model = model

        self.best_interaction_ = best_model
        self.best_cvc_ = max_cvc
        self.best_mean_testing_ba_ = best_avg_ba
        if self.verbose:
            print("\nFit Complete")
            print(f"Best interaction: {self.best_interaction_}")
            print(f"CVC: {self.best_cvc_}/{self.cv}")
            print(f"Mean testing BA: {self.best_mean_testing_ba_:.4f}")

        # Train final lookup table on full data
        self.best_model_lookup_table_ = self._create_lookup_table(X, y_encoded, self.best_interaction_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        X = self._check_genotypes(X)
        encoded = self._internal_predict(X, self.best_interaction_, self.best_model_lookup_table_)
        return self.classes_[encoded]

    def transform(self, X):
        return self.predict(X).reshape(-1, 1)

    def predict_proba(self, X):
        """
        Not implemented.

        MDR is fundamentally a hard classifier; this implementation does
        not attempt to derive calibrated probabilities.  If you need risk
        probabilities, consider wrapping MDR in scikit-learn's
        `CalibratedClassifierCV` or implement cell-frequency posteriors.
        """
        check_is_fitted(self)
        raise NotImplementedError("predict_proba is not supported in this MDR implementation.")
