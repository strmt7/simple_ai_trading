from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from simple_ai_trading.compute import BackendInfo
from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_CANDIDATES,
)
import simple_ai_trading.round74_device_group_preflight as subject


def _report() -> dict[str, object]:
    measurements = []
    for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES:
        measurements.extend(
            (
                {
                    "candidate_id": candidate_id,
                    "group_size": 1,
                    "rows": 128,
                    "warmup_steps": subject.ROUND74_DEVICE_GROUP_WARMUP_STEPS,
                    "measurement_seconds": [1.28, 1.28, 1.28],
                    "median_seconds": 1.28,
                    "rows_per_second": 100.0,
                    "terminal_loss": 1.0,
                    "warning_count": 0,
                    "cpu_fallback_warning_count": 0,
                },
                {
                    "candidate_id": candidate_id,
                    "group_size": 2,
                    "rows": 256,
                    "warmup_steps": subject.ROUND74_DEVICE_GROUP_WARMUP_STEPS,
                    "measurement_seconds": [2.0, 2.0, 2.0],
                    "median_seconds": 2.0,
                    "rows_per_second": 128.0,
                    "terminal_loss": 1.0,
                    "warning_count": 0,
                    "cpu_fallback_warning_count": 0,
                },
            )
        )
    payload: dict[str, object] = {
        "schema_version": subject.ROUND74_DEVICE_GROUP_PREFLIGHT_SCHEMA_VERSION,
        "created_utc": "2026-08-03T00:00:00+00:00",
        "backend": {
            "requested": "cpu",
            "kind": "cpu",
            "device": "cpu",
            "vendor": "portable CPU reference",
            "selection": "deterministic_cpu_reference",
            "accelerated": False,
            "torch_version": "test",
            "torch_directml_version": "not-installed",
        },
        "source_binding": subject._source_binding(),
        "protocol": {
            "seed": subject.ROUND74_DEVICE_GROUP_SEED,
            "candidate_ids": list(ROUND74_EVENT_MODEL_CANDIDATES),
            "group_sizes": list(subject.ROUND74_DEVICE_GROUP_CANDIDATES),
            "minibatch_rows": 128,
            "warmup_steps": subject.ROUND74_DEVICE_GROUP_WARMUP_STEPS,
            "measurement_steps": subject.ROUND74_DEVICE_GROUP_MEASUREMENT_STEPS,
            "selection_fraction_of_best": (
                subject.ROUND74_DEVICE_GROUP_SELECTION_FRACTION
            ),
            "early_stop_fraction_of_best": (
                subject.ROUND74_DEVICE_GROUP_EARLY_STOP_FRACTION
            ),
            "selection_rule": "smallest_group_within_fraction_of_candidate_best",
            "probe_process": "isolated_child",
            "loss_path": "production_supervised_loss_and_backward",
            "pretraining_group_changed": False,
        },
        "measurements": measurements,
        "failures": [],
        "selected_group_sizes": {
            candidate_id: 2 for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES
        },
        "elapsed_seconds": 1.0,
        "evidence_boundary": {
            "market_data_used": False,
            "database_opened": False,
            "realized_financial_target_used": False,
            "model_selection_output_used": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
    }
    payload["preflight_sha256"] = subject._canonical_sha256(payload)
    return payload


def test_round74_device_group_preflight_is_hash_bound_and_immutable(
    tmp_path: Path,
) -> None:
    report = _report()
    validated = subject.validate_round74_device_group_preflight(
        report,
        expected_backend=BackendInfo(
            requested="cpu",
            kind="cpu",
            device="cpu",
            vendor="portable CPU reference",
            reason="",
            selection="deterministic_cpu_reference",
        ),
    )
    path = subject.write_round74_device_group_preflight(validated, tmp_path)

    assert path.is_file()
    assert path == subject.write_round74_device_group_preflight(validated, tmp_path)
    assert path.name == (
        f"round74-device-group-preflight-{report['preflight_sha256']}.json"
    )


def test_round74_device_group_preflight_rejects_timing_tamper() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["measurements"][0]["rows_per_second"] = 1_000.0  # type: ignore[index]

    with pytest.raises(ValueError, match="preflight differs"):
        subject.validate_round74_device_group_preflight(tampered)


def test_round74_device_group_preflight_rejects_rehashed_selection_tamper() -> None:
    tampered = _report()
    tampered["selected_group_sizes"]["causal_event_attention"] = 1  # type: ignore[index]
    unsigned = dict(tampered)
    unsigned.pop("preflight_sha256")
    tampered["preflight_sha256"] = subject._canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="selection differs"):
        subject.validate_round74_device_group_preflight(tampered)


def test_round74_device_group_selection_prefers_smallest_near_best() -> None:
    measurements = (
        {"group_size": 2, "rows_per_second": 100.0},
        {"group_size": 4, "rows_per_second": 102.0},
        {"group_size": 8, "rows_per_second": 60.0},
    )

    assert subject._selected_group_size(measurements) == 2


def test_round74_device_group_subprocess_drains_pipes_during_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()

    class Process:
        returncode = 0

        def __init__(self) -> None:
            self.calls = 0

        def communicate(self, timeout: float | None = None):
            self.calls += 1
            if self.calls == 1:
                raise subject.subprocess.TimeoutExpired("probe", timeout)
            return subject._canonical_json_bytes(report).decode("ascii"), ""

    process = Process()
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_args, **_kwargs: process)
    clock = iter((0.0, 0.0, 11.0, 11.0))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock))
    progress: list[str] = []

    result = subject.run_round74_device_group_preflight_subprocess(
        BackendInfo(
            requested="cpu",
            kind="cpu",
            device="cpu",
            vendor="portable CPU reference",
            reason="",
            selection="deterministic_cpu_reference",
        ),
        minibatch_rows=128,
        progress=lambda stage, **_values: progress.append(stage),
    )

    assert result["preflight_sha256"] == report["preflight_sha256"]
    assert progress == ["device_group_preflight_heartbeat"]


def test_round74_device_group_subprocess_timeout_kills_only_its_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = None

        def __init__(self) -> None:
            self.killed = False

        def communicate(self, timeout: float | None = None):
            if self.killed:
                return "", ""
            raise subject.subprocess.TimeoutExpired("probe", timeout)

        def kill(self) -> None:
            self.killed = True

    process = Process()
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_args, **_kwargs: process)
    clock = iter((0.0, 0.0, 31.0, 31.0))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="hard timeout"):
        subject.run_round74_device_group_preflight_subprocess(
            BackendInfo(
                requested="cpu",
                kind="cpu",
                device="cpu",
                vendor="portable CPU reference",
                reason="",
                selection="deterministic_cpu_reference",
            ),
            minibatch_rows=128,
            timeout_seconds=30.0,
        )

    assert process.killed is True
