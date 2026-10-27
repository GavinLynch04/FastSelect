"""The promotion gate must reject numerical drift and unrepeatable speedups."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarking" / "compare_head_vs_current.py"
SPEC = importlib.util.spec_from_file_location("optimization_benchmark", BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _rounds(seconds=1.0, memory=10.0, parity=True):
    return [
        {"round": index, "mean_seconds": seconds, "peak_traced_mb": memory, "full_output_parity": parity}
        for index in range(1, 6)
    ]


def test_gate_accepts_repeatable_speed_or_memory_improvement():
    assert benchmark._improvement_gate(_rounds(), _rounds(seconds=0.8), 0.03, "cpu") == (True, True, False)
    assert benchmark._improvement_gate(_rounds(), _rounds(memory=8.0), 0.03, "cpu") == (True, False, True)


def test_gate_rejects_noise_and_numerical_drift():
    assert benchmark._improvement_gate(_rounds(), _rounds(seconds=0.99), 0.03, "cpu") == (True, False, False)
    candidate = _rounds(seconds=0.8)
    candidate[-1]["mean_seconds"] = 1.1
    assert benchmark._improvement_gate(_rounds(), candidate, 0.03, "cpu") == (True, False, False)
    assert benchmark._improvement_gate(_rounds(), _rounds(seconds=0.8, parity=False), 0.03, "cpu")[0] is False


def test_gate_does_not_treat_missing_memory_as_an_improvement():
    assert benchmark._improvement_gate(_rounds(memory=0), _rounds(memory=0), 0.03, "cpu") == (True, False, False)
    assert benchmark._improvement_gate(_rounds(), _rounds(memory=8), 0.03, "gpu") == (True, False, False)
    with pytest.raises(ValueError, match="matching, nonempty"):
        benchmark._improvement_gate(_rounds(), _rounds()[:-1], 0.03, "cpu")


def test_full_output_check_catches_drift_that_scalar_summaries_miss():
    reference = {"scores": np.array([1.0, 2.0]), "selected": np.array([1, 0])}
    changed_scores = {"scores": np.array([2.0, 1.0]), "selected": np.array([1, 0])}
    assert benchmark._compare_outputs(reference, changed_scores, 1e-6, 1e-7)[0] is False
    changed_selection = {"scores": reference["scores"], "selected": np.array([0, 1])}
    assert benchmark._compare_outputs(reference, changed_selection, 1e-6, 1e-7)[0] is False
    assert benchmark._compare_outputs(reference, reference, 1e-6, 1e-7) == (True, 0.0)


def test_full_output_check_rejects_missing_shapes_and_nan():
    reference = {"scores": np.array([1.0, 2.0])}
    assert benchmark._compare_outputs(reference, {}, 1e-6, 1e-7)[0] is False
    assert benchmark._compare_outputs(reference, {"scores": np.array([1.0])}, 1e-6, 1e-7)[0] is False
    assert benchmark._compare_outputs(reference, {"scores": np.array([np.nan, 2.0])}, 1e-6, 1e-7)[0] is False
    assert benchmark._compare_outputs({}, {}, 1e-6, 1e-7)[0] is False


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_matching_nonfinite_outputs_do_not_pass(value):
    invalid = {"scores": np.array([value])}
    assert benchmark._compare_outputs(invalid, invalid, 1e-6, 1e-7) == (False, None)


@pytest.mark.parametrize("flag", ["--atol", "--rtol", "--min-seconds"])
@pytest.mark.parametrize("value", ["inf", "nan"])
def test_nonfinite_command_line_parameters_are_rejected(monkeypatch, flag, value):
    monkeypatch.setattr("sys.argv", ["benchmark", flag, value])
    with pytest.raises(SystemExit) as raised:
        benchmark._parse_args()
    assert raised.value.code == 2


def test_nonfinite_measurements_cannot_pass_resource_gate():
    assert benchmark._improvement_gate(_rounds(seconds=np.inf), _rounds(), 0.03, "cpu")[1] is False
    assert benchmark._improvement_gate(_rounds(memory=np.inf), _rounds(), 0.03, "cpu")[2] is False
