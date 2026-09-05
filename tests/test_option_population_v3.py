"""Complete exclusion-union and source-admission checks on synthetic metadata."""

import hashlib
import json

import pytest

from tools import adjudicate_option_population_v3 as gate


def write(path, value, field=None):
    if field:
        value[field] = gate._canonical_hash(value, field)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def metadata(symbols):
    return {
        "optionSymbols": [
            {
                "symbol": symbol,
                "underlying": "BTCUSDT",
                "unit": "1",
                "status": "TRADING",
                "contractType": "CRYPTO_OPTIONS",
                "underlyingType": "CRYPTO",
                "quoteAsset": "USDT",
                "expiryDate": 9999999999999,
            }
            for symbol in symbols
        ]
    }


def fixture(tmp_path, monkeypatch, symbols):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    write(tmp_path / "baseline", metadata(["old"]))
    write(tmp_path / "prior", {"population": {"symbols": ["later"]}}, "result_sha256")
    write(tmp_path / "raw", metadata(symbols))
    source = write(
        tmp_path / "source",
        {
            "contract": {"sha256": "a" * 64},
            "source_gate": {"passed": True},
            "capture": {
                "receipt": {
                    "raw_path": "raw",
                    "completed_at_ms": 1,
                    "response_sha256": hashlib.sha256(
                        (tmp_path / "raw").read_bytes()
                    ).hexdigest(),
                }
            },
        },
        "result_sha256",
    )
    plan = {
        "implementations": [],
        "exclusions": [
            {
                "path": "baseline",
                "kind": "exchange_info",
                "sha256": hashlib.sha256(
                    (tmp_path / "baseline").read_bytes()
                ).hexdigest(),
            },
            {
                "path": "prior",
                "kind": "result",
                "keys": ["population", "symbols"],
                "sha256": hashlib.sha256((tmp_path / "prior").read_bytes()).hexdigest(),
            },
        ],
        "expected_exclusion_count": 2,
        "source_result_path": "source",
        "source_contract_sha256": "a" * 64,
        "output_path": "result",
    }
    write(tmp_path / "plan", plan, "contract_sha256")
    return plan, source


@pytest.mark.parametrize(
    "symbols,expected", [(["old", "later"], []), (["old", "later", "new"], ["new"])]
)
def test_union_excludes_every_previous_population(
    tmp_path, monkeypatch, symbols, expected
):
    fixture(tmp_path, monkeypatch, symbols)
    assert gate.adjudicate(tmp_path / "plan", preflight=True) is None
    assert not (tmp_path / "result").exists()
    result = gate.adjudicate(tmp_path / "plan")
    assert result["distinct_symbols"] == expected
    assert result["new_population_trigger_satisfied"] == bool(expected)
    assert result["price_requests"] == 0
    assert result["accepted_edge"] is False
    with pytest.raises(FileExistsError):
        gate.adjudicate(tmp_path / "plan")


@pytest.mark.parametrize(
    "fault",
    ["raw", "prior", "count", "contract", "source_gate", "expiry", "implementation"],
)
def test_invalid_evidence_stops_without_result(tmp_path, monkeypatch, fault):
    plan, source = fixture(tmp_path, monkeypatch, ["new"])
    if fault in ("raw", "prior"):
        (tmp_path / fault).write_text("{}")
    elif fault == "count":
        plan["expected_exclusion_count"] = 3
        write(tmp_path / "plan", plan, "contract_sha256")
    elif fault == "contract":
        plan["contract_sha256"] = "b" * 64
        write(tmp_path / "plan", plan)
    elif fault == "source_gate":
        source["source_gate"]["passed"] = False
        write(tmp_path / "source", source, "result_sha256")
    elif fault == "implementation":
        plan["implementations"] = [{"path": "baseline", "sha256": "b" * 64}]
        write(tmp_path / "plan", plan, "contract_sha256")
    else:
        value = metadata(["new"])
        value["optionSymbols"][0]["expiryDate"] = 0
        write(tmp_path / "raw", value)
        source["capture"]["receipt"]["response_sha256"] = hashlib.sha256(
            (tmp_path / "raw").read_bytes()
        ).hexdigest()
        write(tmp_path / "source", source, "result_sha256")
    with pytest.raises(ValueError):
        gate.adjudicate(tmp_path / "plan")
    assert not (tmp_path / "result").exists()
