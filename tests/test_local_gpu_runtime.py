import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import verify_lightgbm_opencl_runtime as runtime


@pytest.fixture
def configured_runtime(tmp_path, monkeypatch):
    library = tmp_path / "test-library"
    library.write_bytes(b"synthetic-library")
    manifest = {
        "lightgbm_version": "4.7.0",
        "device_name": "test-device",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(
        runtime.lightgbm,
        "__version__",
        "4.7.0",
    )
    monkeypatch.setattr(
        runtime.lightgbm.basic, "_LIB", SimpleNamespace(_name=str(library))
    )
    monkeypatch.setattr(
        runtime,
        "lightgbm_backend_parameters",
        lambda *a, **k: (
            {"gpu_platform_id": 0, "gpu_device_id": 0, "gpu_use_dp": True},
            "opencl",
            "test-device",
        ),
    )
    device = SimpleNamespace(
        platform_id=0,
        device_id=0,
        as_dict=lambda: {
            "display_name": "test-device",
            "driver_version": "test-driver",
        },
    )
    monkeypatch.setattr(runtime, "discover_opencl_gpu_devices", lambda: [device])
    return manifest


def test_verified_runtime(configured_runtime):
    assert runtime.verify(configured_runtime)["backend"] == "opencl"


@pytest.mark.parametrize(
    "field,value",
    [
        ("lightgbm_version", "0.0"),
        ("library_sha256", "bad"),
        ("device_name", "other-device"),
    ],
)
def test_wrong_identity_stops_launch(configured_runtime, field, value):
    configured_runtime[field] = value
    with pytest.raises(RuntimeError):
        runtime.verify(configured_runtime)


def test_cpu_fallback_stops_gpu_launcher(configured_runtime, monkeypatch):
    monkeypatch.setattr(
        runtime, "lightgbm_backend_parameters", lambda *a, **k: ({}, "cpu", "cpu")
    )
    with pytest.raises(RuntimeError, match="probe failed"):
        runtime.verify(configured_runtime)


def test_missing_device_stops_launch(configured_runtime, monkeypatch):
    monkeypatch.setattr(runtime, "discover_opencl_gpu_devices", lambda: [])
    with pytest.raises(RuntimeError, match="device differs"):
        runtime.verify(configured_runtime)


@pytest.mark.parametrize("stem", ["gpu-benchmark", "gpu-large-benchmark"])
def test_retained_benchmarks_bound_and_decisions_reconstruct(stem):
    root = Path(__file__).resolve().parents[1]
    folder = root / "docs/review/2026-09-04"
    spec_bytes = (folder / f"{stem}-spec.json").read_bytes()
    spec = json.loads(spec_bytes)
    result = json.loads((folder / f"{stem}-result.json").read_bytes())
    assert result["spec_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
    assert (
        result["implementation_sha256"]
        == hashlib.sha256(
            (root / "tools/benchmark_tree_backends.py").read_bytes()
        ).hexdigest()
    )
    assert result["status"] == "completed"
    assert result["financial_validation"] is False
    for workload in result["workloads"]:
        assert len(workload["trials"]) == 2 * spec["repetitions"]
        passed = all(
            pair["max_prediction_difference"]
            <= spec["max_absolute_prediction_difference"]
            and pair["logloss_difference"] <= spec["max_absolute_logloss_difference"]
            for pair in workload["paired_checks"]
        )
        assert workload["paired_tolerances_passed"] == passed
        assert workload["prefer_gpu_for_this_workload"] == (
            passed
            and workload["cpu_over_gpu_speedup"]
            >= spec["minimum_median_speedup_to_prefer_gpu"]
        )
