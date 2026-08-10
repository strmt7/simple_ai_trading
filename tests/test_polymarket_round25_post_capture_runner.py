from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round25_post_capture_runner as runner
from simple_ai_trading.polymarket_round25_active_campaign import (
    POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256,
    POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION,
    POLYMARKET_ROUND25_END_MS,
    load_round25_active_campaign_plan,
    validate_round25_active_campaign_plan,
)
from simple_ai_trading.polymarket_round25_post_capture_runner import (
    Round25PostCaptureRunnerConfig,
    run_round25_post_capture,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-twap-core-campaign-plan-publication-v2-2026-08-10.json"
)


def _config(tmp_path: Path) -> Round25PostCaptureRunnerConfig:
    state = tmp_path / "state"
    state.mkdir()
    return Round25PostCaptureRunnerConfig(
        repository=ROOT,
        source_database=tmp_path / "capture.duckdb",
        plan_path=PLAN,
        state_root=state,
        output_root=tmp_path / "output",
        source_commit_oid="a" * 40,
    )


def _write_state(config: Round25PostCaptureRunnerConfig, **values: object) -> None:
    body = {
        "schema_version": POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        **values,
    }
    (config.state_root / "campaign-state.json").write_text(
        json.dumps(body, sort_keys=True) + "\n",
        encoding="ascii",
    )


def test_active_plan_is_exactly_hash_bound() -> None:
    plan = load_round25_active_campaign_plan(PLAN)

    assert plan.plan_sha256 == POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256
    changed = json.loads(PLAN.read_text(encoding="ascii"))
    changed["required_rtds_topics"] = ["crypto_prices_chainlink"]
    with pytest.raises(ValueError, match="active campaign plan differs"):
        validate_round25_active_campaign_plan(changed)


def test_runner_waits_without_opening_source_or_creating_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    events: list[str] = []

    status = run_round25_post_capture(
        config,
        observed_at_ms=POLYMARKET_ROUND25_END_MS - 1,
        progress=lambda event, _values: events.append(event),
    )

    assert status["status"] == "waiting_for_terminal_capture"
    assert status["source_database_opened"] is False
    assert status["orders_submitted"] == 0
    assert events == ["waiting_for_terminal_capture"]
    assert not config.output_root.exists()


def test_runner_waits_for_terminal_state_after_schedule(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_state(config, phase="campaign_window")

    status = run_round25_post_capture(
        config,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )

    assert status["status"] == "waiting_for_terminal_state"
    assert status["source_database_opened"] is False
    assert not config.output_root.exists()


def test_runner_creates_terminal_once_and_advances_bounded_coordinator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.source_database.write_bytes(b"database")
    _write_state(config, status="campaign_window_ended")
    terminal = {
        "source_plan_sha256": POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256,
        "manifest_sha256": "b" * 64,
        "eligible_run_ids": ["1" * 32],
    }
    calls: list[str] = []
    events: list[str] = []

    def build(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("build")
        return terminal

    def write(path: Path, value: object) -> None:
        calls.append("write")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="ascii")
        assert value is terminal

    def load(_path: Path) -> dict[str, object]:
        calls.append("load")
        return terminal

    def advance(**values: object) -> dict[str, object]:
        calls.append("advance")
        assert values["source_commit_oid"] == "a" * 40
        assert values["maximum_resolution_conditions"] == 128
        assert values["terminal_transport_manifest"] is terminal
        return {
            "phase": "resolution_collection_pending",
            "state_sha256": "c" * 64,
        }

    monkeypatch.setattr(runner, "build_round25_terminal_transport_manifest", build)
    monkeypatch.setattr(runner, "write_round25_terminal_transport_manifest", write)
    monkeypatch.setattr(runner, "load_round25_terminal_transport_manifest", load)
    monkeypatch.setattr(runner, "advance_round25_post_capture", advance)

    result = run_round25_post_capture(
        config,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
        progress=lambda event, _values: events.append(event),
    )

    assert calls == ["build", "write", "load", "advance"]
    assert result["coordinator_phase"] == "resolution_collection_pending"
    assert result["orders_submitted"] == 0
    assert result["live_trading_authority"] is False
    assert events == ["terminal_transport_ready", "post_capture_pass_complete"]


def test_runner_reuses_existing_terminal_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.source_database.write_bytes(b"database")
    _write_state(config, status="campaign_window_ended")
    terminal_path = config.output_root / "terminal-transport-v2.json"
    terminal_path.parent.mkdir()
    terminal_path.write_text("{}\n", encoding="ascii")
    terminal = {
        "source_plan_sha256": POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256,
        "manifest_sha256": "b" * 64,
        "eligible_run_ids": [],
    }
    monkeypatch.setattr(
        runner,
        "load_round25_terminal_transport_manifest",
        lambda _path: terminal,
    )
    monkeypatch.setattr(
        runner,
        "build_round25_terminal_transport_manifest",
        lambda *_args, **_kwargs: pytest.fail("terminal manifest was rebuilt"),
    )
    monkeypatch.setattr(
        runner,
        "advance_round25_post_capture",
        lambda **_kwargs: {"phase": "feature_materialized", "state_sha256": "c" * 64},
    )

    result = run_round25_post_capture(
        config,
        observed_at_ms=POLYMARKET_ROUND25_END_MS,
    )

    assert result["coordinator_phase"] == "feature_materialized"


def test_runner_rejects_nonterminal_authority_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_state(config, phase="campaign_window", live_trading_authority=True)

    with pytest.raises(ValueError, match="nonterminal campaign state differs"):
        run_round25_post_capture(
            config,
            observed_at_ms=POLYMARKET_ROUND25_END_MS,
        )
