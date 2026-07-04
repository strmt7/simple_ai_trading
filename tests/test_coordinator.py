from __future__ import annotations

from simple_ai_trading.ai_runtime import AICapabilityReport
from simple_ai_trading.coordinator import (
    LoopCoordinator,
    LoopHeartbeat,
    build_runtime_coordinator,
    default_loop_contracts,
    render_coordinator_decision,
)
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


def _ai_ok() -> AICapabilityReport:
    return AICapabilityReport(
        ok=True,
        provider="local-gpu",
        model="qwen2.5:7b",
        gpu_vendor="amd",
        compute_backend_requested="directml",
        compute_backend_kind="directml",
        compute_backend_device="privateuseone:0",
        compute_backend_reason="",
        free_vram_gb=12.0,
        free_ram_gb=32.0,
        model_parameters_b=7.0,
        messages=(),
        warnings=(),
    )


def test_loop_coordinator_blocks_execution_on_stale_risk() -> None:
    coordinator = LoopCoordinator(default_loop_contracts(ai_enabled=True, live_mode=True))
    now = 100_000
    coordinator.ingest(LoopHeartbeat("risk", True, "ok", now - 90_000))
    coordinator.ingest(LoopHeartbeat("execution", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("reconciliation", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("market_data", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("machine_learning", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("ai", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("learning", False, "observe", now))

    decision = coordinator.decide(now_ms=now)

    assert decision.state == "blocked_execution"
    assert decision.allow_execution is False
    assert decision.allow_new_entries is False
    assert decision.stale_loops == ("risk",)
    assert "refresh_risk" in decision.actions


def test_loop_coordinator_waits_when_model_loop_blocks_entries_only() -> None:
    coordinator = LoopCoordinator(default_loop_contracts(ai_enabled=False, live_mode=False))
    now = 100_000
    coordinator.ingest(LoopHeartbeat("risk", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("execution", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("market_data", True, "ok", now))
    coordinator.ingest(LoopHeartbeat("machine_learning", False, "missing_model", now))
    coordinator.ingest(LoopHeartbeat("learning", True, "ok", now))

    decision = coordinator.decide(now_ms=now)

    assert decision.state == "waiting"
    assert decision.allow_execution is True
    assert decision.allow_new_entries is False
    assert decision.blocking_loops == ("machine_learning",)


def test_build_runtime_coordinator_reads_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_trading.coordinator.detect_ai_capabilities", lambda _cfg: _ai_ok())
    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")
    positions_root = tmp_path / "autonomous"

    decision = build_runtime_coordinator(
        RuntimeConfig(dry_run=True, testnet=True, compute_backend="directml"),
        StrategyConfig(),
        model_path=model_path,
        positions_root=positions_root,
        now_ms=1_000,
    )
    rendered = render_coordinator_decision(decision)

    assert decision.state == "ready"
    assert decision.allow_new_entries is True
    assert "[ok] ai" in rendered
    assert "Coordinator" in rendered
