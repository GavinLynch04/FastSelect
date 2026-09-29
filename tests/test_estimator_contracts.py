"""scikit-learn estimator contract for every exported estimator.

Relief-family and TuRF contract checks live beside their algorithms; this module
covers the estimators that previously escaped CI (CFS, mRMR, MDR) and exercises
all of them inside a real ``Pipeline`` / ``GridSearchCV``.
"""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_checks

from fast_select import CFS, MDR, SURF, MultiSURF, ReliefF, TuRF, mRMR

DTYPE_OBJECT_REASON = (
    "categorical (string) values are accepted deliberately, so an unorderable object array is "
    "rejected by NumPy's sorting error rather than by a numeric-conversion message"
)


@pytest.mark.parametrize(
    "estimator",
    [CFS(backend="cpu", n_jobs=1), mRMR(2, backend="cpu")],
    ids=["CFS", "mRMR"],
)
def test_full_sklearn_contract(estimator):
    estimator_checks.check_estimator(estimator, expected_failed_checks={"check_dtype_object": DTYPE_OBJECT_REASON})


# MDR accepts only binary targets and 0/1/2 genotypes, so the checks that feed it
# arbitrary continuous or multiclass data cannot apply.  The rest of the contract
# (construction, cloning, parameter handling, fitted-state checks) is exercised
# through the checks below, which do not depend on the input domain.
MDR_CONTRACT_CHECKS = [
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_get_params_invariance",
    "check_set_params",
    "check_estimator_cloneable",
    "check_estimator_repr",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_estimators_unfitted",
    "check_estimator_tags_renamed",
    "check_valid_tag_types",
    "check_estimator_sparse_tag",
]


def test_mdr_domain_independent_contract():
    generated = estimator_checks.estimator_checks_generator(
        MDR(k=1, cv=2, backend="cpu"), legacy=True, expected_failed_checks=None, mark=None
    )
    ran = []
    for estimator, check in generated:
        name = getattr(check, "func", check).__name__
        if name in MDR_CONTRACT_CHECKS:
            check(estimator)
            ran.append(name)
    # Guard against the selection silently becoming empty on a scikit-learn change.
    assert len(ran) >= 5, ran


def test_mdr_estimator_type_and_transformer_tags():
    assert is_classifier(MDR())
    params = MDR(k=3, cv=4, backend="CPU", verbose=True).get_params()
    assert params == {"k": 3, "cv": 4, "backend": "CPU", "verbose": True}
    assert clone(MDR(backend="GPU")).backend == "GPU"


def _genotype_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, (n, 5))
    y = ((X[:, 0] + X[:, 1]) % 2 == 0).astype(int)
    return X, y


def _relief_kwargs(cls):
    kw = dict(backend="cpu", n_jobs=1)
    if cls is ReliefF:
        kw["n_neighbors"] = 2
    return kw


@pytest.mark.parametrize("cls", [ReliefF, SURF, MultiSURF])
def test_relief_estimators_in_pipeline_and_grid_search(cls):
    X, y = _genotype_data(80)
    pipe = Pipeline([("select", cls(n_features_to_select=2, **_relief_kwargs(cls))), ("clf", LogisticRegression())])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2, 3]}, cv=3).fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )


def test_turf_in_pipeline_and_grid_search():
    X, y = _genotype_data(80)
    turf = TuRF(ReliefF(2, n_neighbors=2, backend="cpu", n_jobs=1), n_features_to_select=2)
    pipe = Pipeline([("select", turf), ("clf", LogisticRegression())])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2]}, cv=3).fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )


def test_mrmr_in_pipeline_and_grid_search():
    X, y = _genotype_data(80)
    pipe = Pipeline([("select", mRMR(2, backend="cpu")), ("clf", LogisticRegression())])
    search = GridSearchCV(pipe, {"select__n_features_to_select": [1, 2, 3], "select__method": ["MID", "MIQ"]}, cv=3)
    search.fit(X, y)
    assert (
        search.best_estimator_.named_steps["select"].top_features_.size
        == search.best_params_["select__n_features_to_select"]
    )


def test_cfs_in_pipeline_and_clone():
    X, y = _genotype_data(80)
    pipe = Pipeline([("select", CFS(backend="cpu", n_jobs=1)), ("clf", LogisticRegression())])
    assert cross_val_score(clone(pipe), X, y, cv=3).shape == (3,)
    GridSearchCV(pipe, {"select__n_bins": [3, 5]}, cv=3).fit(X, y)


def test_mdr_in_grid_search_and_cross_val():
    X, y = _genotype_data(90)
    search = GridSearchCV(MDR(cv=3, backend="cpu"), {"k": [1, 2]}, cv=3).fit(X, y)
    assert search.best_params_["k"] in (1, 2)
    assert cross_val_score(MDR(k=2, cv=3, backend="CPU"), X, y, cv=3).shape == (3,)
