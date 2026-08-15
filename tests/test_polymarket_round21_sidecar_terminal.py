from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round21_sidecar_terminal as terminal_module
from simple_ai_trading.polymarket_round21_sidecar_terminal import (
    build_round21_sidecar_terminal_manifest,
    load_round21_sidecar_terminal_manifest,
    validate_round21_sidecar_terminal_manifest,
    write_round21_sidecar_terminal_manifest,
)


START_MS = 1_800_001_200_000
END_MS = START_MS + 2_592_000_000


def _result(*, index: int, status: str) -> dict[str, object]:
    artifact = str(index + 1) * 64
    details: dict[str, object]
    if status in {"complete", "degraded"}:
        details = {
            "run_id": chr(ord("a") + index) * 32,
            "manifest_sha256": "d" * 64,
            "report_sha256": "e" * 64,
            "started_at_ms": START_MS + index * 1_000_000,
            "ended_at_ms": START_MS + index * 1_000_000 + 900_000,
            "raw_message_count": 30,
            "stream_gap_count": int(status == "degraded"),
            "stream_counts": {"binance_spot": 10, "binance_futures": 20},
            "integrity_errors": [],
            "errors": [],
        }
    else:
        details = {"failure_type": "RuntimeError", "failure": "interrupted"}
    return {
        "artifact_sha256": artifact,
        "segment_index": index,
        "status": status,
        "details": details,
    }


def _patch_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        terminal_module,
        "load_round21_sidecar_campaign_plan",
        lambda _path: SimpleNamespace(
            plan_sha256="f" * 64,
            scheduled_start_ms=START_MS,
            scheduled_end_ms=END_MS,
        ),
    )
    monkeypatch.setattr(
        terminal_module,
        "load_round21_sidecar_segment_results",
        lambda _root, _plan: (
            _result(index=0, status="interrupted"),
            _result(index=1, status="complete"),
        ),
    )


def test_terminal_manifest_excludes_interruption_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch)

    manifest = build_round21_sidecar_terminal_manifest(
        plan_path=tmp_path / "plan.json",
        state_root=tmp_path / "state",
        observed_at_ms=END_MS,
    )

    assert manifest["eligible_run_ids"] == ["b" * 32]
    assert manifest["excluded_segment_indices"] == [0]
    assert manifest["segments"][0]["exclusion_reasons"] == [
        "segment_status_interrupted"
    ]
    assert manifest["outcomes_consulted"] is False
    path = tmp_path / "terminal.json"
    write_round21_sidecar_terminal_manifest(path, manifest)
    assert load_round21_sidecar_terminal_manifest(path) == manifest


def test_terminal_manifest_refuses_early_or_ineligible_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch)
    with pytest.raises(RuntimeError, match="before campaign end"):
        build_round21_sidecar_terminal_manifest(
            plan_path=tmp_path / "plan.json",
            state_root=tmp_path / "state",
            observed_at_ms=END_MS - 1,
        )
    monkeypatch.setattr(
        terminal_module,
        "load_round21_sidecar_segment_results",
        lambda _root, _plan: (_result(index=0, status="interrupted"),),
    )
    with pytest.raises(RuntimeError, match="no eligible segment"):
        build_round21_sidecar_terminal_manifest(
            plan_path=tmp_path / "plan.json",
            state_root=tmp_path / "state",
            observed_at_ms=END_MS,
        )


def test_terminal_manifest_rejects_accounting_chronology_and_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch)
    bad = _result(index=0, status="complete")
    bad["details"]["raw_message_count"] = 31
    monkeypatch.setattr(
        terminal_module,
        "load_round21_sidecar_segment_results",
        lambda _root, _plan: (bad,),
    )
    with pytest.raises(ValueError, match="eligible segment differs"):
        build_round21_sidecar_terminal_manifest(
            plan_path=tmp_path / "plan.json",
            state_root=tmp_path / "state",
            observed_at_ms=END_MS,
        )

    _patch_sources(monkeypatch)
    manifest = build_round21_sidecar_terminal_manifest(
        plan_path=tmp_path / "plan.json",
        state_root=tmp_path / "state",
        observed_at_ms=END_MS,
    )
    changed = json.loads(json.dumps(manifest))
    changed["segments"][1]["raw_message_count"] = 31
    with pytest.raises(ValueError, match="eligible segment differs"):
        validate_round21_sidecar_terminal_manifest(changed)

    path = tmp_path / "bad.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        load_round21_sidecar_terminal_manifest(path)
