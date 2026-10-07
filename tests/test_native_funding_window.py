"""Native value validity is necessary, but not complete funding cash qualification."""

import hashlib
import json

import pytest

import tools.adjudicate_polymarket_exact_mlb_monotone_prefilter as paths
from tools.capture_public_source_contract import _canonical_hash
from tools.qualify_native_funding_window import adjudicate, qualify_window


def _raw(**changes):
    row = dict(symbol="BTCUSDT", fundingTime=1000, fundingRate=".0001", markPrice="100")
    row.update(changes)
    return json.dumps([row]).encode()


def _qualify(raw, **changes):
    values = dict(
        sha256=hashlib.sha256(raw).hexdigest(),
        symbol="BTCUSDT",
        start_ms=1000,
        end_ms=1999,
        limit=1000,
    )
    values.update(changes)
    return qualify_window(raw, **values)


def test_positive_value_gate_does_not_promote_or_prove_population():
    result = _qualify(_raw())
    assert result["native_rows"] == 1 and result["native_value_gate_passed"]
    assert not result["cash_labels_admitted"]
    assert not result["independent_event_population_qualified"]
    assert not result["owned_entitlement_qualified"]
    assert not result["economic_metrics_computed"]


@pytest.mark.parametrize(
    "changes",
    [
        {"markPrice": ""},
        {"markPrice": "0"},
        {"fundingTime": 999},
        {"fundingTime": 2000},
        {"symbol": "ETHUSDT"},
    ],
)
def test_missing_mark_or_wrong_population_rejects(changes):
    with pytest.raises(ValueError):
        _qualify(_raw(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"start_ms": True},
        {"end_ms": 999},
        {"limit": 1},
        {"limit": 1001},
        {"sha256": "a" * 64},
    ],
)
def test_bounds_limit_or_hash_rejects(changes):
    with pytest.raises(ValueError):
        _qualify(_raw(), **changes)


@pytest.mark.parametrize(
    "raw", [b"[]", json.dumps([json.loads(_raw())[0]] * 2).encode()]
)
def test_empty_or_duplicate_native_rows_reject(raw):
    with pytest.raises(ValueError):
        _qualify(raw)


def test_limit_saturated_page_is_not_assumed_complete():
    row = json.loads(_raw())[0]
    raw = json.dumps([row, row | {"fundingTime": 1001}]).encode()
    with pytest.raises(ValueError, match="saturated"):
        _qualify(raw, limit=2)


def _capture_fixture(tmp_path, monkeypatch, *, raw=None, status=200):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    (tmp_path / "implementation.py").write_bytes(
        b"# Synthetic implementation binding\n"
    )
    plan = {
        "contract_path": "contract.json",
        "request_name": "synthetic-control",
        "frozen_at_utc": "2026-01-01T00:00:00Z",
        "request": {
            "method": "GET",
            "url": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime=1000&endTime=1999&limit=1000",
            "count": 1,
            "body_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "response_byte_ceiling": 1024000,
        "required_utf8_phrases": ["markPrice"],
        "outputs": {
            "raw_path": "native.raw",
            "result_path": "capture.json",
            "journal_path": "capture.jsonl",
        },
        "authority": {
            "account_requests": 0,
            "credentials_used": False,
            "funds_used": False,
            "orders_or_transactions": 0,
            "protected_capture_touched": False,
            "public_unauthenticated_read_only_requests": 1,
            "signed_requests": 0,
            "trading_authority": False,
        },
        "implementations": [
            {
                "path": "implementation.py",
                "sha256": hashlib.sha256(
                    (tmp_path / "implementation.py").read_bytes()
                ).hexdigest(),
            }
        ],
        "funding_window": {
            "symbol": "BTCUSDT",
            "start_ms": 1000,
            "end_ms": 1999,
            "limit": 1000,
        },
        "adjudication_output": "qualified.json",
        "adjudication_journal": "qualified.jsonl",
    }
    plan["contract_sha256"] = _canonical_hash(plan, "contract_sha256")
    (tmp_path / "contract.json").write_text(json.dumps(plan), encoding="ascii")
    body = _raw() if raw is None else raw
    (tmp_path / "native.raw").write_bytes(body)
    receipt = {
        "phase": "completed",
        "status_code": status,
        "error_type": None,
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "response_bytes": len(body),
        "oversize_body_is_truncated": False,
    }
    result = {
        "contract": {"sha256": plan["contract_sha256"]},
        "capture": {"receipt": receipt},
        "source_gate": {"passed": True},
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    (tmp_path / "capture.json").write_text(json.dumps(result), encoding="ascii")
    intent = {
        "phase": "intent",
        "contract_sha256": plan["contract_sha256"],
        "request": plan["request"],
    }
    (tmp_path / "capture.jsonl").write_text(
        json.dumps(intent) + "\n" + json.dumps(receipt) + "\n", encoding="ascii"
    )
    return tmp_path / "contract.json"


@pytest.mark.parametrize("failure", [None, "missing-mark", "http"])
def test_one_use_terminal_adjudication_success_and_failure(
    tmp_path, monkeypatch, failure
):
    contract = _capture_fixture(
        tmp_path,
        monkeypatch,
        raw=_raw(markPrice="") if failure == "missing-mark" else None,
        status=503 if failure == "http" else 200,
    )
    result = adjudicate(contract)
    assert result["passed"] == (failure is None)
    assert not result["profitability_claim"] and not result["accepted_edge"]
    saved = json.loads((tmp_path / "qualified.json").read_text())
    assert (
        saved == result
        and _canonical_hash(saved, "result_sha256") == saved["result_sha256"]
    )
    assert len((tmp_path / "qualified.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        adjudicate(contract)


@pytest.mark.parametrize("failure", ["request-window", "receipt", "raw", "journal"])
def test_adjudication_rejects_tamper_before_consuming_output(
    tmp_path, monkeypatch, failure
):
    contract = _capture_fixture(tmp_path, monkeypatch)
    if failure == "request-window":
        plan = json.loads(contract.read_text())
        plan["funding_window"]["start_ms"] = 999
        plan["contract_sha256"] = _canonical_hash(plan, "contract_sha256")
        contract.write_text(json.dumps(plan))
    elif failure == "receipt":
        result = json.loads((tmp_path / "capture.json").read_text())
        result["result_sha256"] = "a" * 64
        (tmp_path / "capture.json").write_text(json.dumps(result))
    elif failure == "raw":
        (tmp_path / "native.raw").write_bytes(b"[]")
    else:
        (tmp_path / "capture.jsonl").write_text("[]\n")
    with pytest.raises((ValueError, TypeError)):
        adjudicate(contract)
    assert not (tmp_path / "qualified.jsonl").exists()
