"""Backend precedence is testable without any GPU or MLX installation."""

import sys

import pytest

from fast_select import mutual_information as mi
from fast_select import utils


@pytest.mark.parametrize(
    "preference,cuda,metal,supported,expected",
    [
        ("auto", True, True, True, "gpu"),
        ("auto", False, True, True, "metal"),
        ("auto", False, False, True, "cpu"),
        ("auto", False, True, False, "cpu"),
        ("cpu", True, True, True, "cpu"),
        ("metal", True, True, True, "metal"),
        ("gpu", True, True, True, "gpu"),
    ],
)
def test_precedence(monkeypatch, preference, cuda, metal, supported, expected):
    monkeypatch.setattr(utils, "is_cuda_ready", lambda: cuda)
    monkeypatch.setattr(utils, "is_metal_ready", lambda: metal)
    assert utils.resolve_compute_backend(preference, metal_supported=supported) == expected


def test_explicit_backend_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(utils, "is_cuda_ready", lambda: False)
    monkeypatch.setattr(utils, "is_metal_ready", lambda: False)
    for preference in ("gpu", "metal"):
        with pytest.raises(RuntimeError):
            utils.resolve_compute_backend(preference)
    with pytest.raises(ValueError, match="limits"):
        utils.resolve_compute_backend("metal", metal_supported=False)


def test_auto_skips_metal_without_sustained_gain(monkeypatch):
    import numpy as np

    from fast_select import CFS, chi2

    monkeypatch.setattr(utils, "is_cuda_ready", lambda: False)
    monkeypatch.setattr(utils, "is_metal_ready", lambda: True)
    assert utils.resolve_compute_backend("auto", auto_metal=False) == "cpu"
    assert utils.resolve_compute_backend("metal", auto_metal=False) == "metal"

    x = np.array([[0, 1], [1, 0], [2, 1], [0, 2]])
    y = np.array([0, 1, 0, 1])
    chi2(x, y)  # must not dispatch to Metal under auto
    # ``fast_select.CFS`` is the class; patch the defining module itself.
    monkeypatch.setattr(sys.modules[CFS.__module__], "is_cuda_ready", lambda: False)
    assert CFS().fit(x, y).effective_backend_ == "cpu"


def test_information_capability_fallback(monkeypatch):
    monkeypatch.setattr(mi, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mi, "is_metal_ready", lambda: True)
    assert mi.resolve_backend("auto", 32) == "gpu"
    assert mi.resolve_backend("auto", 33) == "metal"
    assert mi.resolve_backend("auto", 65) == "cpu"
    with pytest.raises(ValueError, match="64 states"):
        mi.resolve_backend("metal", 65)
