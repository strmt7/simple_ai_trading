from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pytest

from simple_ai_trading import cli
from simple_ai_trading.ai_benchmark_claim import (
    begin_preregistered_ai_benchmark_claim,
    complete_preregistered_ai_benchmark_claim,
    fail_preregistered_ai_benchmark_claim,
)
from simple_ai_trading.ai_model_benchmark import (
    benchmark_finance_ai_models,
    default_finance_ai_test_cases,
    write_benchmark_report,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "docs" / "ai" / "risk-review" / "qwen3-14b-v8-preregistration.json"
)


class _ClaimStore:
    def __init__(self, report_sha256: str = "a" * 64) -> None:
        self.connection = duckdb.connect(":memory:")
        self.integrity_calls = 0
        self.connection.execute(
            """
            CREATE TABLE polymarket_recorder_run (
                run_id VARCHAR,
                status VARCHAR,
                error VARCHAR,
                report_sha256 VARCHAR
            )
            """
        )
        self.connection.execute(
            "INSERT INTO polymarket_recorder_run VALUES (?, 'complete', '', ?)",
            ["confirmation", report_sha256],
        )

    def connect(self) -> duckdb.DuckDBPyConnection:
        return self.connection

    def integrity_errors(self, run_id: str) -> tuple[str, ...]:
        assert run_id == "confirmation"
        self.integrity_calls += 1
        return ()


def _passing_report():
    cases = default_finance_ai_test_cases()
    responses = []
    for case in cases:
        rationale = "Risk controls remain active. " + " ".join(case.must_mention)
        responses.append(
            {
                "action": case.expected_action,
                "risk_score": (case.min_risk_score + case.max_risk_score) / 2.0,
                "confidence": 0.9,
                "rationale": rationale,
                "concerns": [],
                "required_actions": [],
            }
        )
    index = 0

    def post_json(_url, _payload, _timeout):
        nonlocal index
        response = responses[index]
        index += 1
        return {"message": {"content": json.dumps(response)}}

    return benchmark_finance_ai_models(
        models=["qwen3:14b"],
        installed_models=["qwen3:14b"],
        timeout_seconds=60.0,
        minimum_score=0.78,
        post_json=post_json,
    )


def _begin(store: _ClaimStore, output: Path):
    return begin_preregistered_ai_benchmark_claim(
        store,  # type: ignore[arg-type]
        preregistration_path=PREREGISTRATION,
        confirmation_run_id="confirmation",
        model="qwen3:14b",
        timeout_seconds=60.0,
        minimum_score=0.78,
        output_path=output,
    )


def test_preregistered_ai_benchmark_is_durable_and_exactly_once(tmp_path) -> None:
    store = _ClaimStore()
    output = tmp_path / "qwen3-14b-v8.json"

    claim = _begin(store, output)
    assert claim.status == "claimed"
    assert store.integrity_calls == 1
    report = _passing_report()
    assert report.passed
    write_benchmark_report(report, output)
    completed = complete_preregistered_ai_benchmark_claim(  # type: ignore[arg-type]
        store,
        claim,
    )
    existing = _begin(store, output)

    assert completed.status == "completed"
    assert completed.benchmark_passed is True
    assert len(completed.report_file_sha256) == 64
    assert existing.status == "existing"
    assert existing.report_file_sha256 == completed.report_file_sha256
    assert store.integrity_calls == 1
    output.write_bytes(output.read_bytes() + b"\n")
    try:
        _begin(store, output)
    except ValueError as exc:
        assert "output digest differs" in str(exc)
    else:
        raise AssertionError("a tampered one-shot AI benchmark output was accepted")


def test_failed_preregistered_ai_benchmark_cannot_reopen_cases(tmp_path) -> None:
    store = _ClaimStore("b" * 64)
    output = tmp_path / "failed-qwen3-14b-v8.json"
    claim = _begin(store, output)
    fail_preregistered_ai_benchmark_claim(  # type: ignore[arg-type]
        store,
        claim,
        RuntimeError("provider stopped"),
    )

    try:
        _begin(store, output)
    except ValueError as exc:
        assert "already claimed:state=failed" in str(exc)
    else:
        raise AssertionError("a failed one-shot AI benchmark reopened its cases")


def test_preregistered_ai_benchmark_rejects_semantically_irrelevant_file_drift(
    tmp_path,
) -> None:
    store = _ClaimStore("c" * 64)
    modified = tmp_path / "modified-preregistration.json"
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    payload["candidate"]["selection_reason"] = "Edited after preregistration."
    modified.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from frozen code or run"):
        begin_preregistered_ai_benchmark_claim(
            store,  # type: ignore[arg-type]
            preregistration_path=modified,
            confirmation_run_id="confirmation",
            model="qwen3:14b",
            timeout_seconds=60.0,
            minimum_score=0.78,
            output_path=tmp_path / "forbidden.json",
        )


def test_preregistered_ai_benchmark_is_one_shot_across_confirmation_runs(
    tmp_path,
) -> None:
    store = _ClaimStore("d" * 64)
    first = _begin(store, tmp_path / "first.json")
    assert first.status == "claimed"
    store.connection.execute(
        "INSERT INTO polymarket_recorder_run VALUES (?, 'complete', '', ?)",
        ["second-confirmation", "e" * 64],
    )

    with pytest.raises(ValueError, match="claim identity differs"):
        begin_preregistered_ai_benchmark_claim(
            store,  # type: ignore[arg-type]
            preregistration_path=PREREGISTRATION,
            confirmation_run_id="second-confirmation",
            model="qwen3:14b",
            timeout_seconds=60.0,
            minimum_score=0.78,
            output_path=tmp_path / "second.json",
        )


def test_qwen3_14b_cli_requires_preregistration(monkeypatch, tmp_path, capsys) -> None:
    def unexpected_benchmark(**_kwargs):
        raise AssertionError("Qwen3 14B inference must remain closed")

    monkeypatch.setattr(
        "simple_ai_trading.ai_model_benchmark.benchmark_finance_ai_models",
        unexpected_benchmark,
    )
    status = cli.command_ai_benchmark(
        argparse.Namespace(
            models="qwen3:14b",
            url="http://127.0.0.1:11434",
            timeout=60.0,
            minimum_score=0.78,
            output=str(tmp_path / "forbidden.json"),
            preregistration="",
            confirmation_database="",
            confirmation_run_id="",
            confirmation_memory_limit="512MB",
            confirmation_database_threads=1,
            json=False,
        )
    )

    assert status == 2
    assert "requires its frozen one-shot preregistration" in capsys.readouterr().err
