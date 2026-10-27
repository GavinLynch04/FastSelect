"""Persisted second-batch admission evidence; not a timing test in CI."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/new_algorithms_2"
pytestmark = pytest.mark.skipif(
    not (ROOT / "benchmarking/report_more_algorithms.py").exists(),
    reason="Admission provenance is checked in a repository checkout, not the minimal source distribution",
)


@pytest.mark.parametrize("algorithm", ["mutual_info_regression", "GiniIndex", "DISR", "CIFE", "MIFS"])
def test_second_batch_algorithm_passed_all_rounds(algorithm):
    result = json.loads((AUDIT / "sustained.json").read_text())
    cases = [d for d in result["decisions"].values() if d["algorithm"] == algorithm]
    assert len(cases) == 2
    for case in cases:
        assert case["admitted"]
        assert len(case["paired_speedups"]) == 5
        assert min(case["paired_speedups"]) >= 1.03


def test_second_batch_full_measurement_matrix():
    result = json.loads((AUDIT / "sustained.json").read_text())
    assert result["output_parity"]
    rows = result["measurements"]
    assert len(rows) == 100
    for case in {row["case"] for row in rows}:
        for backend in ("reference", "cpu"):
            group = [r for r in rows if r["case"] == case and r["backend"] == backend]
            assert {r["round"] for r in group} == set(range(1, 6))
            assert sum(r["measured_seconds"] for r in group) >= 5
            assert all(r["iterations"] > 0 and len(r["output"]["scores"]) > 0 for r in group)


def test_second_batch_anova_loser_is_not_shipped():
    import fast_select

    assert not hasattr(fast_select, "f_classif")
    assert not (ROOT / "src/fast_select/Univariate.py").exists()
    screen = json.loads((AUDIT / "anova_screen.json").read_text())
    for n, p in {(r["n"], r["p"]) for r in screen}:
        shape = [r for r in screen if r["n"] == n and r["p"] == p]
        reference = next(r for r in shape if r["numba_threads"] == 0)["median"]
        assert all(r["median"] > reference for r in shape if r["numba_threads"])
