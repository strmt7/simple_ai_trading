from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest

from simple_ai_trading import cli
from simple_ai_trading.ai_model_benchmark import (
    benchmark_finance_ai_models,
    finance_ai_candidates,
    rescore_finance_ai_benchmark_payload,
)
from simple_ai_trading.ai_runtime import (
    AIRuntimeConfig,
    detect_ai_capabilities,
    estimate_model_parameters_b,
)
from simple_ai_trading.command_contract import command_names, command_specs
from simple_ai_trading.compute import BackendInfo
from simple_ai_trading import windows_app
from simple_ai_trading.windows_app import WINDOWS_APP_COMMANDS
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


def test_ai_runtime_blocks_when_required_gpu_backend_resolves_to_cpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr("simple_ai_trading.ai_runtime._memory_status_gb", lambda: 32.0)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime._nvidia_free_vram_gb", lambda: None
    )
    monkeypatch.setattr("simple_ai_trading.ai_runtime._amd_free_vram_gb", lambda: None)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime.resolve_backend",
        lambda _requested: BackendInfo(
            "directml", "cpu", "cpu", "Python stdlib", "DirectML unavailable"
        ),
    )

    report = detect_ai_capabilities(AIRuntimeConfig(enabled=True, require_gpu=True))

    assert report.ok is False
    assert any("GPU compute backend" in message for message in report.messages)


def test_ai_runtime_accepts_nvidia_or_amd_headroom(monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_trading.ai_runtime._memory_status_gb", lambda: 32.0)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime._nvidia_free_vram_gb", lambda: 10.0
    )
    monkeypatch.setattr("simple_ai_trading.ai_runtime._amd_free_vram_gb", lambda: None)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime.resolve_backend",
        lambda _requested: BackendInfo("directml", "directml", "GPU", "DirectML", ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime._ollama_inventory",
        lambda: {"qwen3:8b": True},
    )

    report = detect_ai_capabilities(AIRuntimeConfig(enabled=True, require_gpu=True))

    assert report.ok is True
    assert report.gpu_vendor == "nvidia"
    assert report.model_parameters_b == 8.0
    assert report.provider_available is True
    assert report.model_available is True
    assert report.model_local is True


def test_ai_runtime_blocks_sub_multibillion_model(monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_trading.ai_runtime._memory_status_gb", lambda: 32.0)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime._nvidia_free_vram_gb", lambda: 10.0
    )
    monkeypatch.setattr("simple_ai_trading.ai_runtime._amd_free_vram_gb", lambda: None)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime.resolve_backend",
        lambda _requested: BackendInfo("directml", "directml", "GPU", "DirectML", ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime._ollama_inventory",
        lambda: {"tiny-560m": True},
    )

    report = detect_ai_capabilities(
        AIRuntimeConfig(enabled=True, require_gpu=True, model="tiny-560m")
    )

    assert report.ok is False
    assert any("below required" in message for message in report.messages)


def test_ai_runtime_blocks_missing_or_cloud_only_ollama_models(monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_trading.ai_runtime._memory_status_gb", lambda: 32.0)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime._nvidia_free_vram_gb", lambda: 10.0
    )
    monkeypatch.setattr("simple_ai_trading.ai_runtime._amd_free_vram_gb", lambda: None)
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime.resolve_backend",
        lambda _requested: BackendInfo("directml", "directml", "GPU", "DirectML", ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.ai_runtime._ollama_inventory",
        lambda: {"deepseek-v3.1:671b-cloud": False},
    )

    missing = detect_ai_capabilities(AIRuntimeConfig(model="qwen3:8b"))
    cloud = detect_ai_capabilities(AIRuntimeConfig(model="deepseek-v3.1:671b-cloud"))

    assert missing.ok is False
    assert any("not installed" in message for message in missing.messages)
    assert cloud.ok is False
    assert cloud.model_available is True
    assert cloud.model_local is False
    assert any("no local weights" in message for message in cloud.messages)


def test_ai_runtime_parses_e_size_model_names() -> None:
    assert estimate_model_parameters_b("gemma4:e4b") == 4.0
    assert estimate_model_parameters_b("qwen3:8b") == 8.0
    assert estimate_model_parameters_b("tiny-560m") == 0.56


def _benchmark_response(
    action: str, risk: float, rationale: str = "risk reviewed"
) -> dict[str, object]:
    return {
        "message": {
            "content": (
                "{"
                f'"action":"{action}",'
                f'"risk_score":{risk},'
                '"confidence":0.82,'
                f'"rationale":"{rationale}",'
                '"concerns":["risk data liquidity uplift drawdown cooldown gap liquidation leverage cost fee ownership position reconcile reconnect correlation concentration untrusted injection provenance human"],'
                '"required_actions":["keep risk controls active"]'
                "}"
            )
        }
    }


def test_finance_ai_benchmark_selects_model_with_correct_structured_actions() -> None:
    actions = {
        "veto_failed_ai_uplift": ("veto", 0.90),
        "cooldown_unpredictable_low_liquidity": ("cooldown", 0.70),
        "approve_clean_positive_uplift": ("approve", 0.20),
        "veto_missing_data_integrity": ("veto", 0.92),
        "veto_liquidation_hidden_by_roi": ("veto", 0.96),
        "veto_negative_after_cost_edge": ("veto", 0.90),
        "veto_position_ownership_mismatch": ("veto", 0.94),
        "cooldown_reconnect_state_uncertain": ("cooldown", 0.78),
        "veto_correlated_diversification_illusion": ("veto", 0.88),
        "veto_untrusted_prompt_injection": ("veto", 0.93),
        "human_review_conflicting_provenance": ("human_review", 0.72),
    }

    def fake_post(_url, payload, _timeout):
        text = payload["messages"][1]["content"]
        for name, (action, risk) in actions.items():
            if name in text:
                return _benchmark_response(action, risk)
        return _benchmark_response("human_review", 0.50)

    report = benchmark_finance_ai_models(
        models=["qwen3:8b"],
        installed_models=["qwen3:8b"],
        post_json=fake_post,
    )

    assert report.passed is True
    assert report.selected_model == "qwen3:8b"
    assert report.results[0].passed is True
    assert report.results[0].action_match_cases == len(report.tests)
    assert report.financial_edge_tested is False
    assert report.trading_authority is False


def test_finance_ai_candidate_registry_includes_local_and_finance_specialists() -> None:
    candidates = {candidate.model: candidate for candidate in finance_ai_candidates()}

    assert candidates["qwen3:8b"].reasoning_or_risk_review is True
    assert candidates["fin-r1:8b"].finance_specialized is True
    assert candidates["fin-r1:8b"].reasoning_or_risk_review is True
    assert candidates["fin-r1:8b"].model_parameters_b == 8.0
    assert candidates["fin-o1:8b"].finance_specialized is True
    assert candidates["fin-o1:8b"].reasoning_or_risk_review is True
    assert candidates["fin-o1:8b"].model_parameters_b == 8.0
    assert candidates["fino1:8b"].finance_specialized is True
    assert candidates["fino1:8b"].model_parameters_b == 8.0
    assert candidates["agentar-fin-r1:8b"].finance_specialized is True
    assert candidates["DragonLLM/Qwen-Open-Finance-R-8B"].finance_specialized is True
    assert candidates["DragonLLM/Qwen-Open-Finance-R-8B"].model_parameters_b == 8.0
    assert candidates["FinGPT/fingpt-mt_llama2-7b_lora"].finance_specialized is True


def test_finance_ai_benchmark_fails_wrong_actions() -> None:
    def fake_post(_url, _payload, _timeout):
        return _benchmark_response("approve", 0.10)

    report = benchmark_finance_ai_models(
        models=["qwen3:8b"],
        installed_models=["qwen3:8b"],
        post_json=fake_post,
    )

    assert report.passed is False
    assert report.selected_model is None
    assert report.results[0].passed is False
    assert report.results[0].action_match_cases < len(report.tests)


def test_finance_ai_rescore_verifies_normalized_response_hashes() -> None:
    actions = {
        "veto_failed_ai_uplift": ("veto", 0.90),
        "cooldown_unpredictable_low_liquidity": ("cooldown", 0.70),
        "approve_clean_positive_uplift": ("approve", 0.20),
        "veto_missing_data_integrity": ("veto", 0.92),
        "veto_liquidation_hidden_by_roi": ("veto", 0.96),
        "veto_negative_after_cost_edge": ("veto", 0.90),
        "veto_position_ownership_mismatch": ("veto", 0.94),
        "cooldown_reconnect_state_uncertain": ("cooldown", 0.78),
        "veto_correlated_diversification_illusion": ("veto", 0.88),
        "veto_untrusted_prompt_injection": ("veto", 0.93),
        "human_review_conflicting_provenance": ("human_review", 0.50),
    }

    def fake_post(_url, payload, _timeout):
        text = payload["messages"][1]["content"]
        for name, (action, risk) in actions.items():
            if name in text:
                return _benchmark_response(action, risk)
        raise AssertionError("unknown benchmark case")

    source = benchmark_finance_ai_models(
        models=["qwen3:8b"],
        installed_models=["qwen3:8b"],
        post_json=fake_post,
        minimum_score=0.90,
    ).asdict()
    source["benchmark_contract"] = "finance-risk-review-adversarial-v4"
    rescored = rescore_finance_ai_benchmark_payload(source)

    assert rescored.passed is True
    assert rescored.selected_model == "qwen3:8b"
    assert rescored.source_evidence[0]["source_contract"].endswith("v4")

    tampered = benchmark_finance_ai_models(
        models=["qwen3:8b"],
        installed_models=["qwen3:8b"],
        post_json=fake_post,
        minimum_score=0.90,
    ).asdict()
    tampered["benchmark_contract"] = "finance-risk-review-adversarial-v4"
    tampered["results"][0]["case_results"][0]["rationale"] = "tampered"
    with pytest.raises(ValueError, match="response hash changed"):
        rescore_finance_ai_benchmark_payload(tampered)


def test_command_ai_benchmark_writes_report(monkeypatch, tmp_path, capsys) -> None:
    class _Report:
        passed = True
        selected_model = "qwen3:8b"
        tests = ({"name": "case"},)
        results = (
            SimpleNamespace(
                model="qwen3:8b",
                passed=True,
                score=0.91,
                action_match_cases=1,
                valid_json_cases=1,
                model_parameters_b=8.0,
                average_latency_seconds=0.5,
                failures=(),
            ),
        )

        def asdict(self):
            return {"passed": True, "selected_model": self.selected_model}

    monkeypatch.setattr(
        "simple_ai_trading.ai_model_benchmark.benchmark_finance_ai_models",
        lambda **_kwargs: _Report(),
    )

    output = tmp_path / "ai_benchmark.json"
    assert (
        cli.command_ai_benchmark(
            argparse.Namespace(
                models="qwen3:8b",
                url="http://127.0.0.1:11434",
                timeout=1.0,
                minimum_score=0.78,
                output=str(output),
                json=False,
            )
        )
        == 0
    )
    assert output.exists()
    assert "selected=qwen3:8b" in capsys.readouterr().out


def test_windows_app_commands_match_cli_contract() -> None:
    assert set(WINDOWS_APP_COMMANDS) == set(command_names())
    assert "ai" in WINDOWS_APP_COMMANDS
    assert "backtest" in WINDOWS_APP_COMMANDS
    assert "model-lab" in WINDOWS_APP_COMMANDS


def test_windows_launcher_reports_missing_native_executable(
    monkeypatch, capsys
) -> None:
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
    assert calls["env"]["SIMPLE_AI_TRADING_REPO_ROOT"] == str(windows_app._repo_root())
    assert str(windows_app._repo_root() / "src") in calls["env"]["PYTHONPATH"].split(
        windows_app.os.pathsep
    )


def test_windows_launcher_help_exits_cleanly(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["simple-ai-trading-windows", "--help"])
    assert windows_app.main() == 0
    assert "usage: simple-ai-trading-windows" in capsys.readouterr().out


def test_generated_native_contract_matches_cli() -> None:
    header = (
        windows_app._repo_root()
        / "native"
        / "windows"
        / "generated"
        / "command_contract.hpp"
    )
    text = header.read_text(encoding="utf-8")
    for spec in command_specs():
        option_count = len(spec.options) + len(spec.positionals)
        array_name = (
            "".join(ch if ch.isalnum() else "_" for ch in spec.name).strip("_")
            or "command"
        )
        options_ptr = f"kOptions_{array_name}" if option_count else "nullptr"
        if option_count:
            assert f"inline constexpr CommandOptionSpec kOptions_{array_name}[]" in text
        assert (
            f"{_wide(spec.name)}, {_wide(spec.help)}, {options_ptr}, {option_count}"
            in text
        )
        for option in (*spec.options, *spec.positionals):
            flags = ", ".join(option.flags) or option.dest
            assert _wide(flags) in text
            assert _wide(option.dest) in text
            if option.choices:
                assert _wide(", ".join(option.choices)) in text

    selection = next(
        spec for spec in command_specs() if spec.name == "tape-depth-select"
    )
    reports = next(option for option in selection.options if option.dest == "report")
    design = next(option for option in selection.options if option.dest == "design")
    assert reports.repeatable is True
    assert reports.value_arity == "1"
    assert design.required is True
    assert (
        '{L"--report", L"report", L"", L"", '
        'L"screening report path; repeat for every declared trial", '
        'L"1", true, true, true}'
    ) in text
    confirmation = next(
        spec for spec in command_specs() if spec.name == "tape-depth-confirm"
    )
    assert {option.dest for option in confirmation.options} >= {
        "selection",
        "report",
        "output",
    }


def test_native_window_initializes_hwnd_during_create() -> None:
    source = (
        windows_app._repo_root() / "native" / "windows" / "src" / "main.cpp"
    ).read_text(encoding="utf-8")
    assert "self->hwnd_ = hwnd;" in source
    assert 'create_control(L"LISTBOX"' in source
    assert 'L"COMBOBOX"' in source
    assert 'L"Overview"' in source
    assert 'L"Trading"' in source
    assert 'L"Research"' in source
    assert 'L"System"' in source
    assert 'L"Settings"' in source
    assert 'L"Stop + Close"' in source
    assert 'L"Paper"' in source
    assert 'L"Testnet live"' in source
    assert 'run_control_sequence({L"autonomous stop"})' in source
    assert 'run_control_sequence({L"autonomous stop", L"close all"})' not in source
    assert "status_bar_" in source
    assert "kStatusBarId = 111" in source
    assert "kModeComboId = 116" in source
    assert "page_summary_" in source
    assert "kApiBudgetRefreshMs = 90000" in source
    assert 'L"api-budget --compact"' in source
    assert "SIMPLE_AI_TRADING_REPO_ROOT" in source
    assert "SIMPLE_AI_TRADING_GUI_DRY_RUN" in source
    assert "SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS" in source
    assert 'root / L".venv" / L"Scripts" / L"python.exe"' in source
    assert "runtime_summary()" in source
    assert "repo_root()" in source
    assert "SIMPLE_AI_TRADING_GUI_SMOKE" in source
    assert 'preview += L" (repeatable)";' in source


def test_native_window_has_repeatable_smoke_and_capture_tools() -> None:
    root = windows_app._repo_root()
    smoke = (root / "tools" / "smoke_native_windows_ui.ps1").read_text(encoding="utf-8")
    capture = (root / "tools" / "capture_native_windows_app.ps1").read_text(
        encoding="utf-8"
    )
    layout = (root / "tools" / "validate_native_windows_layout.ps1").read_text(
        encoding="utf-8"
    )

    assert "SIMPLE_AI_TRADING_GUI_DRY_RUN" in smoke
    assert "Stop + Close" in smoke
    assert "Testnet live" in smoke
    assert "independent pause/stop" in smoke
    assert "unsafe ledger-only close all" in smoke
    assert "SetProcessDPIAware" in capture
    assert "PrintWindow" in capture
    assert "Captured window is too small" in capture
    assert "$StatusId = 111" in layout
    assert "$ModeId = 116" in layout
    assert "overview settings overlap telemetry footer" in layout
    assert "Assert-PixelHealth" in layout
    assert "API budget value" in layout


def _wide(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\r", " ").replace("\n", " ")
    return f'L"{escaped}"'


def test_runtime_ai_defaults_enabled_and_strategy_defaults_conservative() -> None:
    runtime = RuntimeConfig()
    strategy = StrategyConfig(leverage=999.0)

    assert runtime.ai_enabled is True
    assert runtime.ai_model == "qwen3:8b"
    assert runtime.ai_require_gpu is True
    assert runtime.ai_min_free_vram_gb == 8.0
    assert runtime.ai_min_model_parameters_b == 2.0
    assert strategy.risk_level == "conservative"
    assert strategy.reinvest_profits is False
    assert strategy.leverage == 20.0
    assert strategy.max_regime_unpredictability == 0.60
