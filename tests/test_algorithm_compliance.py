"""Small, equation-driven tests for published algorithm semantics.

These oracles intentionally do not call FastSelect internals.  Their purpose is
to catch a CPU/GPU implementation pair that agrees with itself but has drifted
away from the defining algorithm.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fast_select import MDR, SURF, MultiSURF, ReliefF
from fast_select.CFS import _best_first_search, _cfs_merit
from fast_select.mutual_information import calculate_mi_single_pair
from fast_select.utils import is_cuda_ready, split_discrete_last

FLOAT32_EPS = np.finfo(np.float32).eps


def _distance_matrix(X, is_discrete):
    ranges = np.ptp(X, axis=0)
    ranges[is_discrete] = 1.0
    ranges[ranges == 0.0] = 1.0
    distances = np.zeros((len(X), len(X)), dtype=np.float64)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            diffs = np.where(
                is_discrete,
                X[i] != X[j],
                np.abs(X[i] - X[j]) / ranges,
            )
            distances[i, j] = distances[j, i] = np.sum(diffs)
    return distances, ranges


def _surf_reference(X, y, is_discrete, use_star):
    distances, ranges = _distance_matrix(X, is_discrete)
    threshold = np.mean(distances[np.triu_indices(len(X), k=1)])
    scores = np.zeros(X.shape[1], dtype=np.float64)

    for i in range(len(X)):
        groups = {
            "near_hit": [],
            "near_miss": [],
            "far_hit": [],
            "far_miss": [],
        }
        for j in range(len(X)):
            if i == j:
                continue
            hit = y[i] == y[j]
            if distances[i, j] < threshold:
                groups["near_hit" if hit else "near_miss"].append(j)
            elif use_star and distances[i, j] > threshold:
                groups["far_hit" if hit else "far_miss"].append(j)

        for name, neighbors in groups.items():
            if not neighbors:
                continue
            sign = {
                "near_hit": -1.0,
                "near_miss": 1.0,
                "far_hit": 1.0,
                "far_miss": -1.0,
            }[name]
            for j in neighbors:
                diff = np.where(
                    is_discrete,
                    X[i] != X[j],
                    np.abs(X[i] - X[j]) / ranges,
                )
                scores += sign * diff / (len(X) * len(neighbors))
    return scores


def _multisurf_reference(X, y, is_discrete, use_star):
    distances, ranges = _distance_matrix(X, is_discrete)
    scores = np.zeros(X.shape[1], dtype=np.float64)

    for i in range(len(X)):
        other = np.delete(distances[i], i)
        mean = np.mean(other)
        half_sigma = 0.5 * np.std(other)
        near_threshold = mean - half_sigma
        far_threshold = mean + half_sigma
        groups = {
            "near_hit": [],
            "near_miss": [],
            "far_hit": [],
            "far_miss": [],
        }
        for j in range(len(X)):
            if i == j:
                continue
            hit = y[i] == y[j]
            if distances[i, j] < near_threshold:
                groups["near_hit" if hit else "near_miss"].append(j)
            elif use_star and distances[i, j] > far_threshold:
                groups["far_hit" if hit else "far_miss"].append(j)

        for name, neighbors in groups.items():
            if not neighbors:
                continue
            far = name.startswith("far")
            sign = -1.0 if name.endswith("hit") else 1.0
            for j in neighbors:
                diff = np.where(
                    is_discrete,
                    X[i] != X[j],
                    np.abs(X[i] - X[j]) / ranges,
                ).astype(np.float64)
                contribution = 1.0 - diff if far else diff
                scores += sign * contribution / (len(X) * len(neighbors))
    return scores


def _relieff_reference(X, y, is_discrete, k):
    """ReliefF as defined by Kononenko (1994).

    ``W[A] -= sum_j diff(A, R, H_j) / (m * k)`` for the nearest hits, and
    ``W[A] += P(C)/(1 - P(class(R))) * sum_j diff(A, R, M_j(C)) / (m * k)`` for
    the nearest misses of every other class C.  Groups smaller than ``k`` are
    normalised by the number of neighbours actually found.
    """
    n_samples, n_features = X.shape
    ranges = np.ptp(X, axis=0).astype(np.float64)
    ranges[is_discrete] = 1.0
    ranges[ranges == 0.0] = 1.0

    def diff(a, b):
        return np.where(
            is_discrete,
            (a != b).astype(np.float64),
            np.abs(a.astype(np.float64) - b.astype(np.float64)) / ranges,
        )

    classes, counts = np.unique(y, return_counts=True)
    priors = counts / n_samples

    scores = np.zeros(n_features, dtype=np.float64)
    for i in range(n_samples):
        others = [j for j in range(n_samples) if j != i]
        distances = {j: float(np.sum(diff(X[i], X[j]))) for j in others}
        own_class = int(np.flatnonzero(classes == y[i])[0])

        for c_index, c in enumerate(classes):
            # Ties keep the lower index, matching the kernels' strict-< inserts.
            members = sorted(
                (j for j in others if y[j] == c),
                key=lambda j: (distances[j], j),
            )[:k]
            if not members:
                continue

            if c_index == own_class:
                weight = -1.0 / (n_samples * len(members))
            else:
                weight = (priors[c_index] / (1.0 - priors[own_class])) / (n_samples * len(members))

            for j in members:
                scores += weight * diff(X[i], X[j])

    return scores


RELIEFF_X = np.array(
    [
        [0.0, 10.0, 0, 1],
        [0.3, 11.0, 0, 1],
        [0.7, 13.0, 1, 0],
        [1.1, 16.0, 1, 0],
        [1.6, 20.0, 2, 1],
        [2.2, 25.0, 2, 1],
        [2.9, 31.0, 0, 0],
        [3.7, 38.0, 1, 1],
        [4.6, 46.0, 2, 0],
        [5.6, 55.0, 0, 1],
        [6.7, 65.0, 1, 0],
        [7.9, 76.0, 2, 1],
    ],
    dtype=np.float32,
)
# Deliberately imbalanced and multiclass so the class-prior weights differ.
RELIEFF_Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 7, 7], dtype=np.int32)
RELIEFF_DISCRETE = np.array([False, False, True, True])


@pytest.mark.parametrize("k", [1, 2, 3])
def test_relieff_matches_class_prior_weighted_definition(k):
    """ReliefF must match the paper equation, not merely agree CPU-to-GPU."""
    expected = _relieff_reference(RELIEFF_X, RELIEFF_Y, RELIEFF_DISCRETE, k)

    backends = ("cpu", "gpu") if is_cuda_ready() else ("cpu",)
    for backend in backends:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            actual = (
                ReliefF(
                    n_features_to_select=2,
                    n_neighbors=k,
                    backend=backend,
                    discrete_limit=3,
                )
                .fit(RELIEFF_X, RELIEFF_Y)
                .feature_importances_
            )

        assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"ReliefF {backend} backend drifted from the definition at k={k}",
        )


MIXED_X = np.array(
    [
        [0.0, 0],
        [0.2, 0],
        [0.6, 1],
        [1.4, 1],
        [1.8, 2],
        [2.0, 2],
    ],
    dtype=np.float32,
)
MIXED_Y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
MIXED_DISCRETE = np.array([False, True])


def test_surf_matches_global_radius_and_count_normalization():
    for use_star in (False, True):
        expected = _surf_reference(MIXED_X, MIXED_Y, MIXED_DISCRETE, use_star)
        backends = ("cpu", "gpu") if is_cuda_ready() else ("cpu",)
        for backend in backends:
            actual = (
                SURF(
                    n_features_to_select=2,
                    backend=backend,
                    use_star=use_star,
                    discrete_limit=3,
                    n_jobs=1,
                )
                .fit(MIXED_X, MIXED_Y)
                .feature_importances_
            )
            assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_multisurf_matches_dead_band_and_similarity_far_scoring():
    for use_star in (False, True):
        expected = _multisurf_reference(MIXED_X, MIXED_Y, MIXED_DISCRETE, use_star)
        backends = ("cpu", "gpu") if is_cuda_ready() else ("cpu",)
        for backend in backends:
            actual = (
                MultiSURF(
                    n_features_to_select=2,
                    backend=backend,
                    use_star=use_star,
                    discrete_limit=3,
                    n_jobs=1,
                )
                .fit(MIXED_X, MIXED_Y)
                .feature_importances_
            )
            assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_mutual_information_matches_hand_calculated_units():
    x = np.array([0, 0, 1, 1], dtype=np.int32)
    same = np.array([0, 0, 1, 1], dtype=np.int32)
    independent = np.array([0, 1, 0, 1], dtype=np.int32)

    assert_allclose(calculate_mi_single_pair(x, same, backend="cpu", unit="bit"), 1.0)
    assert_allclose(
        calculate_mi_single_pair(x, same, backend="cpu", unit="nat"),
        math.log(2.0),
    )
    assert_allclose(
        calculate_mi_single_pair(x, independent, backend="cpu", unit="bit"),
        0.0,
        atol=1e-12,
    )


def test_cfs_merit_and_best_first_have_no_unpublished_relevance_cutoff():
    expected = (0.8 + 0.6) / math.sqrt(2.0 + 2.0 * 0.25)
    assert_allclose(_cfs_merit(1.4, 2, 0.25), expected)

    relevance = np.array([0.05, 0.04], dtype=np.float32)
    redundancy = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert _best_first_search(2, relevance, redundancy) == [0]


def _adversarial_distance_dataset():
    """Mixed data whose normalised feature differences span a punishing range.

    Most continuous features are near-constant apart from two outliers, so their
    normalised differences land around 1e-7 while a handful of features
    contribute differences of order 1.  Summing those terms in a different order
    is the worst case for the CPU kernels' continuous/discrete feature split.
    """
    rng = np.random.default_rng(99)
    n_samples, n_features = 40, 96
    X = np.empty((n_samples, n_features), dtype=np.float32)

    for f in range(n_features):
        kind = f % 4
        if kind == 0:
            X[:, f] = rng.integers(0, 3, n_samples)
        elif kind == 1:
            X[:, f] = rng.standard_normal(n_samples)
        else:
            # Near-constant column with two far outliers: after dividing by the
            # range, ordinary pairs differ by ~1e-7 while outlier pairs differ
            # by ~1.
            column = 1.0 + rng.standard_normal(n_samples) * 1e-7
            column[0] = -1e6
            column[-1] = 1e6
            X[:, f] = column

    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int32)
    is_discrete = np.array([f % 4 == 0 for f in range(n_features)])
    return X, y, is_discrete


def test_continuous_discrete_split_keeps_distance_error_bounded():
    """The kernels' feature reordering may only perturb distances by rounding.

    Every term of the distance is non-negative, so reassociating the sum can
    move the float32 result by at most ``(n_terms - 1) * eps`` relative to the
    exact value regardless of order.  This checks the production ordering
    against an exact float64 sum taken in the original feature order, on
    adversarial and on ordinary data.
    """
    for label, (X, _, is_discrete) in (
        ("adversarial", _adversarial_distance_dataset()),
        (
            "ordinary",
            (
                np.random.default_rng(3).standard_normal((40, 96), dtype=np.float32),
                None,
                np.zeros(96, dtype=bool),
            ),
        ),
    ):
        n_features = X.shape[1]
        ranges = np.ptp(X, axis=0).astype(np.float64)
        ranges[is_discrete] = 1.0
        ranges[ranges == 0.0] = 1.0

        columns, n_cont = split_discrete_last(is_discrete)
        scaled = X.astype(np.float32)[:, columns]
        scaled[:, :n_cont] *= (1.0 / ranges[columns[:n_cont]]).astype(np.float32)

        bound = (n_features - 1) * FLOAT32_EPS
        worst = 0.0
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                # Exact float64 sum, original feature order.
                terms = np.where(
                    is_discrete,
                    (X[i] != X[j]).astype(np.float64),
                    np.abs(X[i].astype(np.float64) - X[j].astype(np.float64)) / ranges,
                )
                exact = math.fsum(terms)

                # Production order: continuous block, then discrete block.
                reordered = np.float32(0.0)
                for f in range(n_cont):
                    reordered += abs(scaled[i, f] - scaled[j, f])
                for f in range(n_cont, n_features):
                    if scaled[i, f] != scaled[j, f]:
                        reordered += np.float32(1.0)

                if exact > 0.0:
                    worst = max(worst, abs(float(reordered) - exact) / exact)

        assert worst <= bound, (
            f"{label}: relative distance deviation {worst:.3e} exceeded the "
            f"non-negative-sum reassociation bound {bound:.3e}"
        )


def test_mdr_threshold_ties_are_high_risk():
    X = np.array([[0], [0], [1], [1]], dtype=np.uint8)
    y = np.array([0, 1, 0, 1], dtype=np.uint8)
    lookup = MDR(k=1, cv=2, backend="cpu")._create_lookup_table(X, y, (0,))
    assert_array_equal(lookup[:2], [1, 1])


def test_mdr_round_trips_nonzero_one_labels():
    X = np.array(
        [[2], [2], [2], [1], [1], [0], [0], [0]],
        dtype=np.uint8,
    )
    y = np.array([5, 5, 5, 2, 2, 2, 2, 2], dtype=np.int32)
    model = MDR(k=1, cv=2, backend="cpu").fit(X, y)
    predictions = model.predict(X)
    assert set(np.unique(predictions)) <= {2, 5}
    assert np.mean(predictions == y) >= 0.75
