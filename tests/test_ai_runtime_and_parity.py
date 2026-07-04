from __future__ import annotations

import sys

from simple_ai_trading.ai_runtime import AIRuntimeConfig, detect_ai_capabilities
from simple_ai_trading.command_contract import command_names, command_specs
from simple_ai_trading.compute import BackendInfo
from simple_ai_trading import windows_app
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
    assert report.model_parameters_b == 7.0


def test_ai_runtime_blocks_sub_multibillion_model(monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_trading.ai_runtime._memory_status_gb", lambda: 32.0)
    monkeypatch.setattr("simple_ai_trading.ai_runtime._nvidia_free_vram_gb", lambda: 10.0)
    monkeypatch.setattr("simple_ai_trading.ai_runtime._amd_free_vram_gb", lambda: None)

    report = detect_ai_capabilities(
        AIRuntimeConfig(enabled=True, require_gpu=True, model="tiny-560m")
    )

    assert report.ok is False
    assert any("below required" in message for message in report.messages)


def test_windows_app_commands_match_cli_contract() -> None:
    assert set(WINDOWS_APP_COMMANDS) == set(command_names())
    assert "ai" in WINDOWS_APP_COMMANDS
    assert "backtest" in WINDOWS_APP_COMMANDS
    assert "model-lab" in WINDOWS_APP_COMMANDS


def test_windows_launcher_reports_missing_native_executable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(windows_app, "native_executable_candidates", lambda: ())
    monkeypatch.setattr(windows_app, "find_native_executable", lambda: None)
    assert windows_app.main() == 2
    assert "Native Windows app executable was not found" in capsys.readouterr().err


def test_windows_launcher_runs_native_executable(monkeypatch, tmp_path) -> None:
    exe = tmp_path / "SimpleAITrading.exe"
    exe.write_text("", encoding="utf-8")
    calls = {}
    monkeypatch.setattr(windows_app, "find_native_executable", lambda: exe)
    def fake_call(args, env):
        calls["args"] = args
        calls["env"] = env
        return 0

    monkeypatch.setattr(windows_app.subprocess, "call", fake_call)
    assert windows_app.main() == 0
    assert calls["args"] == [str(exe)]
    assert calls["env"]["SIMPLE_AI_TRADING_PYTHON"] == sys.executable


def test_windows_launcher_help_exits_cleanly(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["simple-ai-trading-windows", "--help"])
    assert windows_app.main() == 0
    assert "usage: simple-ai-trading-windows" in capsys.readouterr().out


def test_generated_native_contract_matches_cli() -> None:
    header = windows_app._repo_root() / "native" / "windows" / "generated" / "command_contract.hpp"
    text = header.read_text(encoding="utf-8")
    for spec in command_specs():
        option_count = len(spec.options) + len(spec.positionals)
        array_name = "".join(ch if ch.isalnum() else "_" for ch in spec.name).strip("_") or "command"
        options_ptr = f"kOptions_{array_name}" if option_count else "nullptr"
        if option_count:
            assert f"inline constexpr CommandOptionSpec kOptions_{array_name}[]" in text
        assert f'{_wide(spec.name)}, {_wide(spec.help)}, {options_ptr}, {option_count}' in text
        for option in (*spec.options, *spec.positionals):
            flags = ", ".join(option.flags) or option.dest
            assert _wide(flags) in text
            assert _wide(option.dest) in text
            if option.choices:
                assert _wide(", ".join(option.choices)) in text


def test_native_window_initializes_hwnd_during_create() -> None:
    source = (windows_app._repo_root() / "native" / "windows" / "src" / "main.cpp").read_text(encoding="utf-8")
    assert "self->hwnd_ = hwnd;" in source
    assert 'create_control(L"LISTBOX"' in source
    assert 'L"COMBOBOX"' in source
    assert 'L"Stop And Close All"' in source
    assert "repo_root()" in source
    assert "SIMPLE_AI_TRADING_GUI_SMOKE" in source


def _wide(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\r", " ").replace("\n", " ")
    return f'L"{escaped}"'


def test_runtime_ai_defaults_enabled_and_strategy_defaults_conservative() -> None:
    runtime = RuntimeConfig()
    strategy = StrategyConfig(leverage=999.0)

    assert runtime.ai_enabled is True
    assert runtime.ai_model == "qwen2.5:7b"
    assert runtime.ai_require_gpu is True
    assert runtime.ai_min_free_vram_gb == 8.0
    assert runtime.ai_min_model_parameters_b == 2.0
    assert strategy.risk_level == "conservative"
    assert strategy.reinvest_profits is False
    assert strategy.leverage == 20.0
