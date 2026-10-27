"""Admission evidence is separate from mathematical correctness tests."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/new_algorithms"
pytestmark = pytest.mark.skipif(
    not (ROOT / "benchmarking/report_new_algorithms.py").exists(),
    reason="Admission provenance is checked in a repository checkout, not the minimal source distribution",
)


@pytest.mark.parametrize("algorithm", ["MI", "JMI", "CMIM", "FCBF", "LaplacianScore"])
def test_every_new_algorithm_earned_admission(algorithm):
    decision = json.loads((AUDIT / "admission_decisions.json").read_text())
    cases = decision["algorithms"][algorithm]
    assert len(cases) == 2
    for row in cases.values():
        assert row["admitted"]
        assert len(row["round_speedups"]) == 5
        assert min(row["round_speedups"]) >= 1.03


def test_measurement_duration_and_complete_matrix():
    result = json.loads((AUDIT / "final_admission.json").read_text())
    assert result["implementation"] == "production"
    assert result["output_parity"]
    rows = result["measurements"]
    assert len(rows) == 100  # 10 shapes * 2 versions * 5 rounds
    for case in {r["case"] for r in rows}:
        for backend in ("reference", "cpu"):
            group = [r for r in rows if r["case"] == case and r["backend"] == backend]
            assert {r["round"] for r in group} == set(range(1, 6))
            assert sum(r["measured_seconds"] for r in group) >= 5
            assert all(r["iterations"] >= 1 and r["output"]["scores"] for r in group)


def test_failed_metal_trials_remain_unadmitted():
    decision = json.loads((AUDIT / "admission_decisions.json").read_text())
    assert len(decision["gpu_trials"]) == 8
    for row in decision["gpu_trials"].values():
        assert not row["retained"]
        assert min(row["cpu_over_metal_round_speedups"]) < 1.03
