from __future__ import annotations

from simple_ai_trading.ai_runtime import AIRuntimeConfig, detect_ai_capabilities
from simple_ai_trading.command_contract import command_names
from simple_ai_trading.compute import BackendInfo
from simple_ai_trading.windows_app import WINDOWS_APP_COMMANDS
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


def test_ai_runtime_blocks_when_required_gpu_backend_resolves_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_trading.ai_runtime._memory_status_gb", lambda: 32.0)
    monkeypatch.setattr("simple_ai_trading.ai_runtime._nvidia_free_vram_gb", lambda: None)
    monkeypatch.setattr("simple_ai_trading.ai_runtime._amd_free_vram_gb", lambda: None)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime.resolve_backend",
        lambda _requested: BackendInfo("directml", "cpu", "cpu", "Python stdlib", "DirectML unavailable"),
    )

    report = detect_ai_capabilities(AIRuntimeConfig(enabled=True, require_gpu=True))

    assert report.ok is False
    assert any("GPU compute backend" in message for message in report.messages)


def test_ai_runtime_accepts_nvidia_or_amd_headroom(monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_trading.ai_runtime._memory_status_gb", lambda: 32.0)
    monkeypatch.setattr("simple_ai_trading.ai_runtime._nvidia_free_vram_gb", lambda: 10.0)
    monkeypatch.setattr("simple_ai_trading.ai_runtime._amd_free_vram_gb", lambda: None)

    report = detect_ai_capabilities(AIRuntimeConfig(enabled=True, require_gpu=True))

    assert report.ok is True
    assert report.gpu_vendor == "nvidia"


def test_windows_app_commands_match_cli_contract() -> None:
    assert set(WINDOWS_APP_COMMANDS) == set(command_names())
    assert "ai" in WINDOWS_APP_COMMANDS
    assert "backtest" in WINDOWS_APP_COMMANDS


def test_runtime_ai_defaults_enabled_and_strategy_defaults_conservative() -> None:
    runtime = RuntimeConfig()
    strategy = StrategyConfig(leverage=999.0)

    assert runtime.ai_enabled is True
    assert runtime.ai_require_gpu is True
    assert runtime.ai_min_free_vram_gb == 8.0
    assert strategy.risk_level == "conservative"
    assert strategy.reinvest_profits is False
    assert strategy.leverage == 10.0
