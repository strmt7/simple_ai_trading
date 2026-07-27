from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

import tools.aggregate_round74_execution_calibration as subject


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _write_capture(directory: Path, payload: dict[str, object]) -> Path:
    selected = dict(payload)
    selected["artifact_sha256"] = _canonical_sha256(selected)
    encoded = (
        json.dumps(
            selected,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    target = directory / f"{selected['artifact_sha256']}.json"
    target.write_bytes(encoded)
    return target


@dataclass(frozen=True)
class _Slot:
    ordinal: int
    round_trip_id: str


class _Plan:
    plan_sha256 = "1" * 64
    campaign_id = "round74-aggregate-test"
    target_quote_notional = 100
    slots = (
        _Slot(ordinal=0, round_trip_id="pair-0"),
        _Slot(ordinal=1, round_trip_id="pair-1"),
    )


class _Evidence:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "environment": "binance_usdm_testnet",
            "evidence_sha256": ("a" if self.kind == "entry_exit_latency" else "b") * 64,
        }


class _Bundle:
    reference_quote_notional = 100.0
    entry_exit_latency_evidence = _Evidence("entry_exit_latency")
    residual_slippage_evidence = _Evidence("residual_slippage")

    @staticmethod
    def entry_latency_mapping() -> dict[str, int]:
        return {"BTCUSDT": 1, "ETHUSDT": 2, "SOLUSDT": 3}

    @staticmethod
    def exit_latency_mapping() -> dict[str, int]:
        return {"BTCUSDT": 4, "ETHUSDT": 5, "SOLUSDT": 6}

    @staticmethod
    def slippage_mapping() -> dict[str, float]:
        return {"BTCUSDT": 0.1, "ETHUSDT": 0.2, "SOLUSDT": 0.3}


def _capture_payload(*, ordinal: int, wall_ns: int) -> dict[str, object]:
    round_trip_id = f"pair-{ordinal}"
    pair = {
        "calibration_run_id": _Plan.campaign_id,
        "round_trip_id": round_trip_id,
        "symbol": "BTCUSDT",
        "records": [
            {"path": "entry", "side": "BUY", "ordinal": ordinal},
            {"path": "exit", "side": "SELL", "ordinal": ordinal},
        ],
    }
    pair["pair_sha256"] = _canonical_sha256(pair)
    return {
        "operation": "capture",
        "environment": "binance_usdm_testnet",
        "source_sha256": {},
        "captured_at_wall_ns": wall_ns,
        "campaign_binding": {
            "campaign_plan_artifact_sha256": "2" * 64,
            "campaign_plan_sha256": _Plan.plan_sha256,
            "slot_ordinal": ordinal,
            "round_trip_id": round_trip_id,
        },
        "sizing": {},
        "result": {"evidence_admitted": True, "pair": pair},
    }


def test_complete_campaign_is_aggregated_without_network_or_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="ascii")
    captures = tmp_path / "captures"
    captures.mkdir()
    for ordinal in range(2):
        _write_capture(
            captures,
            _capture_payload(
                ordinal=ordinal,
                wall_ns=100 + ordinal,
            ),
        )

    monkeypatch.setattr(
        subject.capture_tool,
        "_load_campaign_plan_artifact",
        lambda _path: (_Plan(), "2" * 64),
    )
    monkeypatch.setattr(
        subject.capture_tool,
        "_validated_campaign_capture_slots",
        lambda **_kwargs: (0, 1),
    )
    received: dict[str, object] = {}

    def _build(**kwargs: object) -> _Bundle:
        received.update(kwargs)
        return _Bundle()

    monkeypatch.setattr(
        subject,
        "build_round74_execution_calibration_evidence",
        _build,
    )

    result = subject.main(
        [
            "--campaign-plan",
            str(plan_path),
            "--capture-directory",
            str(captures),
            "--output-directory",
            str(tmp_path / "aggregate"),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mainnet_transfer_permitted"] is False
    assert summary["network_accessed"] is False
    target = Path(summary["artifact"])
    payload = json.loads(target.read_text(encoding="ascii"))
    claimed = payload.pop("artifact_sha256")
    assert claimed == _canonical_sha256(payload)
    assert payload["source_capture_artifact_count"] == 2
    assert payload["source_record_count"] == 4
    assert payload["observed_wall_ns"] == 101
    assert payload["authority"] == {
        "testnet_execution_calibration": True,
        "mainnet_execution_equivalence": False,
        "mainnet_transfer_permitted": False,
        "financial_edge_tested": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "testnet_trading_authority": False,
        "live_trading_authority": False,
    }
    assert received["environment"] == "binance_usdm_testnet"
    assert received["reference_quote_notional"] == 100.0
    assert received["observed_wall_ns"] == 101
    assert len(received["records"]) == 4


def test_incomplete_campaign_fails_before_evidence_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="ascii")
    captures = tmp_path / "captures"
    captures.mkdir()
    _write_capture(
        captures,
        _capture_payload(ordinal=0, wall_ns=100),
    )
    monkeypatch.setattr(
        subject.capture_tool,
        "_load_campaign_plan_artifact",
        lambda _path: (_Plan(), "2" * 64),
    )
    monkeypatch.setattr(
        subject.capture_tool,
        "_validated_campaign_capture_slots",
        lambda **_kwargs: (0,),
    )
    monkeypatch.setattr(
        subject,
        "build_round74_execution_calibration_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete campaign reached evidence build")
        ),
    )

    result = subject.main(
        [
            "--campaign-plan",
            str(plan_path),
            "--capture-directory",
            str(captures),
            "--output-directory",
            str(tmp_path / "aggregate"),
        ]
    )

    assert result == 2
    assert "campaign is incomplete" in capsys.readouterr().err
    assert not (tmp_path / "aggregate").exists()


def test_duplicate_slot_identity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = tmp_path / "captures"
    captures.mkdir()
    first = _capture_payload(ordinal=0, wall_ns=100)
    second = _capture_payload(ordinal=1, wall_ns=101)
    second["campaign_binding"]["slot_ordinal"] = 0
    _write_capture(captures, first)
    _write_capture(captures, second)
    monkeypatch.setattr(
        subject.capture_tool,
        "_validated_campaign_capture_slots",
        lambda **_kwargs: (0, 1),
    )

    with pytest.raises(ValueError, match="identity differs"):
        subject._load_complete_campaign_records(
            plan=_Plan(),
            plan_artifact_sha256="2" * 64,
            capture_directory=captures,
        )
