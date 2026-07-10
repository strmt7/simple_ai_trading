from __future__ import annotations

from types import SimpleNamespace

import pytest

from simple_ai_trading import lightgbm_backend


def _backend(kind: str):
    return SimpleNamespace(kind=kind)


def test_lightgbm_cpu_backend_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(lightgbm_backend, "resolve_backend", lambda _value: _backend("cpu"))

    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters("auto", 41)

    assert kind == "cpu"
    assert device == "cpu"
    assert parameters["device_type"] == "cpu"
    assert parameters["seed"] == 41
    assert "gpu_platform_id" not in parameters


def test_lightgbm_opencl_defaults_to_driver_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        lightgbm_backend,
        "resolve_backend",
        lambda _value: _backend("directml"),
    )
    monkeypatch.delenv(lightgbm_backend.OPENCL_PLATFORM_ENV, raising=False)
    monkeypatch.delenv(lightgbm_backend.OPENCL_DEVICE_ENV, raising=False)

    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters("auto", 7)

    assert kind == "opencl"
    assert device == "opencl:auto"
    assert parameters["device_type"] == "gpu"
    assert parameters["gpu_use_dp"] is False
    assert "gpu_platform_id" not in parameters
    assert "gpu_device_id" not in parameters


def test_lightgbm_opencl_honors_explicit_device_pair(monkeypatch) -> None:
    monkeypatch.setenv(lightgbm_backend.OPENCL_PLATFORM_ENV, "2")
    monkeypatch.setenv(lightgbm_backend.OPENCL_DEVICE_ENV, "3")

    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters(
        "directml",
        11,
        resolved_backend=_backend("directml"),
    )

    assert kind == "opencl"
    assert device == "opencl:2:3"
    assert parameters["gpu_platform_id"] == 2
    assert parameters["gpu_device_id"] == 3


@pytest.mark.parametrize(
    ("platform", "device", "message"),
    [
        ("0", "", "must be set together"),
        ("x", "0", "invalid OpenCL"),
        ("-1", "0", "must be non-negative"),
    ],
)
def test_lightgbm_opencl_rejects_ambiguous_overrides(
    monkeypatch,
    platform: str,
    device: str,
    message: str,
) -> None:
    monkeypatch.setenv(lightgbm_backend.OPENCL_PLATFORM_ENV, platform)
    if device:
        monkeypatch.setenv(lightgbm_backend.OPENCL_DEVICE_ENV, device)
    else:
        monkeypatch.delenv(lightgbm_backend.OPENCL_DEVICE_ENV, raising=False)

    with pytest.raises(ValueError, match=message):
        lightgbm_backend.lightgbm_backend_parameters(
            "directml",
            3,
            resolved_backend=_backend("directml"),
        )
