"""Exact-boundary SURF/SURF* checks against a rational-arithmetic oracle.

Urbanowicz et al. (2018): every pair distance is the sum of range-normalised
feature differences, the SURF radius is the mean over unique pairs, and a pair
is a near neighbour only when its distance is *strictly* below the radius (SURF*
also scores pairs strictly above it, with the updates reversed).

The oracle uses ``fractions.Fraction`` on integer-valued data, so the radius and
every distance are exact rationals and a mathematical tie is a true equality
that no floating-point representation can blur.  Column ranges such as 3, 5, 6,
7 or 9 make the normalised values non-dyadic, which is exactly where float
rounding of ``|a - b| / range`` used to move a tied pair to either side.
"""

from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fast_select import SURF
from fast_select.utils import is_cuda_ready

RANGES = [3, 5, 6, 7, 9]


def surf_exact(X, y, star, discrete):
    """SURF / SURF* scores in exact rational arithmetic; returns (scores, n_tied_pairs)."""
    n, p = X.shape
    Xi = X.astype(np.int64)
    spans = [int(Xi[:, f].max() - Xi[:, f].min()) for f in range(p)]

    def diff(i, j, f):
        if discrete[f]:
            return Fraction(int(Xi[i, f] != Xi[j, f]))
        return Fraction(abs(int(Xi[i, f] - Xi[j, f])), spans[f]) if spans[f] else Fraction(0)

    dist = [[sum(diff(i, j, f) for f in range(p)) for j in range(n)] for i in range(n)]
    pairs = [dist[i][j] for i in range(n) for j in range(i + 1, n)]
    radius = sum(pairs) / len(pairs)
    ties = sum(1 for d in pairs if d == radius)

    scores = [Fraction(0)] * p
    for i in range(n):
        groups = {"near_hit": [], "near_miss": [], "far_hit": [], "far_miss": []}
        for j in range(n):
            if i == j:
                continue
            hit = y[i] == y[j]
            if dist[i][j] < radius:
                groups["near_hit" if hit else "near_miss"].append(j)
            elif star and dist[i][j] > radius:
                groups["far_hit" if hit else "far_miss"].append(j)
        for name, sign in (("near_hit", -1), ("near_miss", 1), ("far_hit", 1), ("far_miss", -1)):
            members = groups[name]
            for j in members:
                for f in range(p):
                    scores[f] += sign * diff(i, j, f) / len(members) / n
    return np.array([float(s) for s in scores]), ties


def make_case(seed, mixed):
    rng = np.random.default_rng(seed)
    n, p = int(rng.integers(3, 9)), int(rng.integers(1, 4))
    X = np.column_stack([rng.integers(0, RANGES[int(rng.integers(len(RANGES)))] + 1, n) for _ in range(p)])
    y = rng.integers(0, 2, n)
    if len(np.unique(y)) < 2:
        y[0], y[1] = 0, 1
    discrete = np.zeros(p, dtype=bool)
    if mixed:
        discrete[::2] = True
    return X.astype(np.float64), y, discrete


def discrete_limit_for(X, discrete):
    """Smallest limit that makes exactly the ``discrete`` columns discrete (or 0 when none are)."""
    if not discrete.any():
        return 0
    return max(len(np.unique(X[:, f])) for f in np.flatnonzero(discrete))


def _case_is_consistent(X, discrete, limit):
    """The limit must not accidentally turn a continuous column discrete."""
    return all((len(np.unique(X[:, f])) <= limit) == discrete[f] for f in range(X.shape[1]))


def _tie_seeds(mixed, wanted=40, search=4000):
    """Seeds whose exact radius equals at least one pair distance (true mathematical ties)."""
    found = []
    for seed in range(search):
        X, y, discrete = make_case(seed, mixed)
        limit = discrete_limit_for(X, discrete)
        if not _case_is_consistent(X, discrete, limit):
            continue
        if surf_exact(X, y, False, discrete)[1] > 0:
            found.append(seed)
        if len(found) == wanted:
            break
    return found


CONTINUOUS_TIE_SEEDS = _tie_seeds(mixed=False)
MIXED_TIE_SEEDS = _tie_seeds(mixed=True)


def test_oracle_cases_exercise_exact_ties():
    assert len(CONTINUOUS_TIE_SEEDS) == 40, "the search must find seeds with exact radius ties"
    assert len(MIXED_TIE_SEEDS) >= 10


@pytest.mark.parametrize("star", [False, True])
@pytest.mark.parametrize(
    "mixed,seed", [(False, s) for s in CONTINUOUS_TIE_SEEDS] + [(True, s) for s in MIXED_TIE_SEEDS]
)
def test_surf_cpu_matches_exact_rational_oracle(seed, mixed, star):
    X, y, discrete = make_case(seed, mixed)
    limit = discrete_limit_for(X, discrete)
    expected, _ = surf_exact(X, y, star, discrete)
    model = SURF(1, backend="cpu", discrete_limit=limit, n_jobs=1, use_star=star).fit(X, y)
    assert_allclose(model.feature_importances_, expected, atol=5e-6)


@pytest.mark.skipif(not is_cuda_ready(), reason="NVIDIA GPU with CUDA (or the CUDA simulator) not available")
@pytest.mark.parametrize("star", [False, True])
@pytest.mark.parametrize("seed", CONTINUOUS_TIE_SEEDS[:12])
def test_surf_gpu_matches_exact_oracle_and_cpu(seed, star):
    X, y, discrete = make_case(seed, mixed=False)
    expected, _ = surf_exact(X, y, star, discrete)
    kw = dict(discrete_limit=0, n_jobs=1, use_star=star)
    cpu = SURF(1, backend="cpu", **kw).fit(X, y).feature_importances_
    gpu = SURF(1, backend="gpu", **kw).fit(X, y).feature_importances_
    assert_allclose(gpu, expected, atol=5e-6)
    assert_allclose(gpu, cpu, atol=5e-6)
