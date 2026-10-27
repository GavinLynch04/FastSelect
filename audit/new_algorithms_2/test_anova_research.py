"""Correct but rejected ANOVA research prototype; outside the runtime package."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import f as f_distribution
from sklearn.feature_selection import f_classif as sklearn_f

from audit.new_algorithms_2.prototypes.Univariate import f_classif


def exact_anova(X, y):
    values = []
    groups = np.unique(y)
    for column in X.T:
        overall = sum(map(Fraction, column)) / len(column)
        between, within = Fraction(), Fraction()
        for label in groups:
            group = list(map(Fraction, column[y == label]))
            mean = sum(group) / len(group)
            between += len(group) * (mean - overall) ** 2
            within += sum((value - mean) ** 2 for value in group)
        values.append(float((between / (len(groups) - 1)) / (within / (len(y) - len(groups)))))
    return np.asarray(values)


def test_anova_exact_equation_and_affine_invariance():
    X = np.array([[1, 3], [2, 5], [4, 3], [7, 4], [8, 9], [11, 8]], dtype=float)
    y = np.array([-7, -7, -7, 19, 19, 19])
    expected = exact_anova(X, y)
    for values in (X, X + 10**8, -X * 8, X * 1e200):
        score, probability = f_classif(values, y)
        np.testing.assert_allclose(score, expected, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(probability, f_distribution.sf(expected, 1, 4), atol=2e-15)


def test_anova_constant_perfect_separation_and_reference():
    y = np.tile([-3, 13], 12)
    X = np.column_stack([np.ones(24), y, np.arange(24)])
    with pytest.warns(UserWarning, match="constant"):
        scores, probability = f_classif(X, y)
    assert np.isnan(scores[0]) and np.isnan(probability[0])
    assert np.isinf(scores[1]) and probability[1] == 0
    rng = np.random.default_rng(31)
    X = rng.normal(size=(173, 19))
    y = rng.integers(0, 4, 173)
    result = f_classif(X, y)
    reference = sklearn_f(X, y)
    np.testing.assert_allclose(result, reference, atol=2e-13, rtol=2e-12)
