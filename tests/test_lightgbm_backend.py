from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_trading.compute import BackendInfo
from simple_ai_trading import compute, lightgbm_backend


def _backend(requested: str, kind: str) -> BackendInfo:
    return BackendInfo(
        requested=requested,
        kind=kind,  # type: ignore[arg-type]
        device="cpu" if kind == "cpu" else f"{kind}:0",
        vendor="test",
        reason="test fallback" if requested not in {"auto", kind} else "",
    )


def test_lightgbm_cpu_backend_is_seeded_by_default() -> None:
    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters("cpu", 41)

    assert kind == "cpu"
    assert device == "cpu"
    assert parameters["device_type"] == "cpu"
    assert parameters["seed"] == 41
    assert "deterministic" not in parameters
    assert "gpu_platform_id" not in parameters


def test_lightgbm_reproducible_cpu_backend_uses_deterministic_columns() -> None:
    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters(
        "cpu",
        41,
        reproducible=True,
    )

    assert kind == "cpu"
    assert device == "cpu"
    assert parameters["deterministic"] is True
    assert parameters["force_col_wise"] is True


def test_lightgbm_auto_falls_back_only_after_real_probe_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        lightgbm_backend,
        "resolve_backend",
        lambda _value: _backend("auto", "cpu"),
    )
    monkeypatch.setattr(
        lightgbm_backend,
        "_probe_lightgbm_target",
        lambda *_args: (False, "target unavailable"),
    )

    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters("auto", 7)

    assert (kind, device) == ("cpu", "cpu")
    assert parameters["device_type"] == "cpu"


def test_lightgbm_opencl_defaults_to_driver_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        lightgbm_backend,
        "resolve_backend",
        lambda _value: _backend("auto", "directml"),
    )
    monkeypatch.setattr(
        lightgbm_backend,
        "_probe_lightgbm_target",
        lambda target, *_args: (target == "gpu", "probe"),
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
    monkeypatch.setattr(
        lightgbm_backend,
        "_probe_lightgbm_target",
        lambda *_args: (True, "probe"),
    )

    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters(
        "directml",
        11,
        resolved_backend=_backend("directml", "directml"),
    )

    assert kind == "opencl"
    assert device == "opencl:2:3"
    assert parameters["gpu_platform_id"] == 2
    assert parameters["gpu_device_id"] == 3


def test_lightgbm_reproducible_opencl_backend_uses_fp64(monkeypatch) -> None:
    monkeypatch.delenv(lightgbm_backend.OPENCL_PLATFORM_ENV, raising=False)
    monkeypatch.delenv(lightgbm_backend.OPENCL_DEVICE_ENV, raising=False)
    monkeypatch.setattr(
        lightgbm_backend,
        "_probe_lightgbm_target",
        lambda *_args: (True, "probe"),
    )

    parameters, kind, _device = lightgbm_backend.lightgbm_backend_parameters(
        "directml",
        11,
        resolved_backend=_backend("directml", "directml"),
        reproducible=True,
    )

    assert kind == "opencl"
    assert parameters["gpu_use_dp"] is True
    assert "deterministic" not in parameters


def test_lightgbm_can_pin_an_enumerated_opencl_device(monkeypatch) -> None:
    identity = lightgbm_backend.OpenCLDeviceIdentity(
        platform_id=1,
        device_id=2,
        platform_name="Test OpenCL",
        platform_vendor="Test vendor",
        platform_version="OpenCL 3.0",
        device_name="gfx-test",
        board_name="Test discrete GPU",
        device_vendor="Test vendor",
        device_version="OpenCL 3.0",
        driver_version="1.2.3",
        global_memory_bytes=16 * 1024**3,
        maximum_compute_units=32,
    )
    monkeypatch.delenv(lightgbm_backend.OPENCL_PLATFORM_ENV, raising=False)
    monkeypatch.delenv(lightgbm_backend.OPENCL_DEVICE_ENV, raising=False)
    monkeypatch.setattr(
        lightgbm_backend,
        "discover_opencl_gpu_devices",
        lambda: (identity,),
    )
    observed: list[tuple[str, int | None, int | None]] = []

    def probe(
        target: str,
        platform_id: int | None,
        device_id: int | None,
    ) -> tuple[bool, str]:
        observed.append((target, platform_id, device_id))
        return True, "probe"

    monkeypatch.setattr(lightgbm_backend, "_probe_lightgbm_target", probe)

    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters(
        "directml",
        11,
        resolved_backend=_backend("directml", "directml"),
        reproducible=True,
        pin_opencl_device=True,
    )

    assert observed == [("gpu", 1, 2)]
    assert kind == "opencl"
    assert device == "opencl:1:2:Test discrete GPU"
    assert parameters["gpu_platform_id"] == 1
    assert parameters["gpu_device_id"] == 2


def test_lightgbm_auto_prefers_verified_cuda_target(monkeypatch) -> None:
    monkeypatch.setattr(
        lightgbm_backend,
        "resolve_backend",
        lambda _value: _backend("auto", "cuda"),
    )
    calls: list[str] = []

    def probe(target: str, *_args) -> tuple[bool, str]:
        calls.append(target)
        return target == "cuda", "probe"

    monkeypatch.setattr(lightgbm_backend, "_probe_lightgbm_target", probe)

    parameters, kind, device = lightgbm_backend.lightgbm_backend_parameters("auto", 13)

    assert calls == ["cuda"]
    assert (kind, device) == ("cuda", "cuda:0")
    assert parameters["device_type"] == "cuda"


def test_lightgbm_artifact_backend_contract_covers_every_resolver_result() -> None:
    assert lightgbm_backend.SUPPORTED_LIGHTGBM_BACKEND_KINDS == {
        "cpu",
        "opencl",
        "cuda",
    }


def test_model_artifacts_do_not_duplicate_lightgbm_backend_whitelists() -> None:
    package_root = Path(lightgbm_backend.__file__).resolve().parent
    offenders = [
        path.name
        for path in package_root.glob("*.py")
        if "backend_kind not in {" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_lightgbm_pinned_backend_rejects_tensor_runtime_fallback() -> None:
    with pytest.raises(
        compute.BackendUnavailableError, match="requested compute backend"
    ):
        lightgbm_backend.lightgbm_backend_parameters(
            "directml",
            11,
            resolved_backend=_backend("directml", "cpu"),
        )


def test_lightgbm_pinned_backend_rejects_failed_tree_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        lightgbm_backend,
        "_probe_lightgbm_target",
        lambda *_args: (False, "OpenCL target absent"),
    )

    with pytest.raises(compute.BackendUnavailableError, match="OpenCL target absent"):
        lightgbm_backend.lightgbm_backend_parameters(
            "directml",
            11,
            resolved_backend=_backend("directml", "directml"),
        )


def test_lightgbm_backend_rejects_non_boolean_reproducibility() -> None:
    with pytest.raises(ValueError, match="reproducible must be a boolean"):
        lightgbm_backend.lightgbm_backend_parameters(
            "cpu",
            11,
            reproducible=1,  # type: ignore[arg-type]
        )


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
            resolved_backend=_backend("directml", "directml"),
        )
