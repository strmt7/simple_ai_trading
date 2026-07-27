from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import tools.capture_round74_execution_calibration as tool
from tools.capture_round74_execution_calibration import (
    API_KEY_ENV,
    API_SECRET_ENV,
    build_parser,
    command_capture,
    command_recover,
    command_status,
    main,
)


class _FakeTransport:
    instances: list[_FakeTransport] = []

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        timeout_seconds: float,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout_seconds = timeout_seconds
        self.last_rate_limit_headers = {"x-mbx-used-weight-1m": "7"}
        self.instances.append(self)

    def __enter__(self) -> _FakeTransport:
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        return None


class _RecoveryResult:
    def __init__(self, *, complete: bool) -> None:
        self.complete = complete
        self.blocking_round_trip_ids = (
            () if complete else ("round74-prior:BTCUSDT-prior",)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "blocking_round_trip_ids": list(self.blocking_round_trip_ids),
        }


class _Pair:
    pair_sha256 = "a" * 64


class _CaptureResult:
    evidence_admitted = True
    pair = _Pair()

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_admitted": True,
            "pair": {"pair_sha256": self.pair.pair_sha256},
        }


def _network_args(
    tmp_path: Path,
    *,
    command: str,
) -> argparse.Namespace:
    values = [
        "--database",
        str(tmp_path / "evidence.duckdb"),
        "--output-directory",
        str(tmp_path / "artifacts"),
        command,
        "--yes",
        "--acknowledge-non-mainnet-orders",
    ]
    if command == "capture":
        values.extend(
            [
                "--calibration-run-id",
                "round74-test",
                "--round-trip-id",
                "BTCUSDT-0",
                "--symbol",
                "BTCUSDT",
                "--entry-side",
                "BUY",
                "--quantity",
                "1",
                "--reference-quote-notional",
                "100",
            ]
        )
    return build_parser().parse_args(values)


def _install_ephemeral_credentials(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-only-key-material")
    monkeypatch.setenv(API_SECRET_ENV, "test-only-secret-material")


def _read_single_artifact(directory: Path) -> dict[str, object]:
    paths = list(directory.glob("*.json"))
    assert len(paths) == 1
    raw = paths[0].read_bytes()
    assert raw.endswith(b"\n")
    payload = json.loads(raw)
    recorded_hash = payload.pop("artifact_sha256")
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == recorded_hash
    return payload


def test_status_is_local_and_creates_no_blockers(
    tmp_path: Path,
    capsys,
) -> None:
    args = argparse.Namespace(database=str(tmp_path / "evidence.duckdb"))

    assert command_status(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["network_accessed"] is False
    assert payload["orders_submitted"] is False
    assert payload["blocking_round_trip_ids"] == []


def test_network_modes_require_two_confirmations_before_credentials(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(API_SECRET_ENV, raising=False)

    result = main(
        [
            "--database",
            str(tmp_path / "evidence.duckdb"),
            "capture",
            "--calibration-run-id",
            "round74-test",
            "--round-trip-id",
            "BTCUSDT-0",
            "--symbol",
            "BTCUSDT",
            "--entry-side",
            "BUY",
            "--quantity",
            "1",
            "--reference-quote-notional",
            "100",
        ]
    )

    assert result == 2
    assert "requires --yes" in capsys.readouterr().err


def test_capture_never_reads_credentials_without_both_confirmations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "ephemeral-key")
    monkeypatch.setenv(API_SECRET_ENV, "ephemeral-secret")
    args = build_parser().parse_args(
        [
            "--database",
            str(tmp_path / "evidence.duckdb"),
            "capture",
            "--yes",
            "--calibration-run-id",
            "round74-test",
            "--round-trip-id",
            "BTCUSDT-0",
            "--symbol",
            "BTCUSDT",
            "--entry-side",
            "BUY",
            "--quantity",
            "1",
            "--reference-quote-notional",
            "100",
        ]
    )

    try:
        command_capture(args)
    except RuntimeError as exc:
        assert "requires --yes" in str(exc)
    else:
        raise AssertionError("capture unexpectedly passed confirmation gate")


def test_recover_writes_canonical_source_bound_secret_free_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _FakeTransport.instances.clear()
    _install_ephemeral_credentials(monkeypatch)
    monkeypatch.setattr(
        tool,
        "Round74BinanceTestnetExecutionTransport",
        _FakeTransport,
    )
    monkeypatch.setattr(
        tool,
        "recover_round74_execution_calibration",
        lambda **_kwargs: _RecoveryResult(complete=True),
    )

    assert command_recover(_network_args(tmp_path, command="recover")) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["complete"] is True
    payload = _read_single_artifact(tmp_path / "artifacts")
    serialized = json.dumps(payload, sort_keys=True)
    assert "test-only-key-material" not in serialized
    assert "test-only-secret-material" not in serialized
    assert payload["environment"] == "binance_usdm_testnet"
    assert payload["mainnet_orders_submitted"] is False
    assert payload["profitability_claim"] is False
    assert payload["last_rate_limit_headers"] == {"x-mbx-used-weight-1m": "7"}
    assert len(payload["source_sha256"]) == 6
    assert all(len(value) == 64 for value in payload["source_sha256"].values())


def test_capture_blocks_before_submission_when_recovery_is_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_ephemeral_credentials(monkeypatch)
    monkeypatch.setattr(
        tool,
        "Round74BinanceTestnetExecutionTransport",
        _FakeTransport,
    )
    monkeypatch.setattr(
        tool,
        "recover_round74_execution_calibration",
        lambda **_kwargs: _RecoveryResult(complete=False),
    )

    def _unexpected_capture(**_kwargs) -> _CaptureResult:
        raise AssertionError("capture reached after incomplete recovery")

    monkeypatch.setattr(
        tool,
        "capture_round74_execution_calibration_pair",
        _unexpected_capture,
    )

    with pytest.raises(RuntimeError, match="unresolved calibration exposure"):
        command_capture(_network_args(tmp_path, command="capture"))

    assert not (tmp_path / "artifacts").exists()


def test_capture_writes_admitted_pair_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _install_ephemeral_credentials(monkeypatch)
    monkeypatch.setattr(
        tool,
        "Round74BinanceTestnetExecutionTransport",
        _FakeTransport,
    )
    monkeypatch.setattr(
        tool,
        "recover_round74_execution_calibration",
        lambda **_kwargs: _RecoveryResult(complete=True),
    )
    monkeypatch.setattr(
        tool,
        "capture_round74_execution_calibration_pair",
        lambda **_kwargs: _CaptureResult(),
    )

    assert command_capture(_network_args(tmp_path, command="capture")) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["evidence_admitted"] is True
    assert summary["pair_sha256"] == "a" * 64
    payload = _read_single_artifact(tmp_path / "artifacts")
    assert payload["operation"] == "capture"
    assert payload["recovery_before_capture"]["complete"] is True
    assert payload["result"]["evidence_admitted"] is True
