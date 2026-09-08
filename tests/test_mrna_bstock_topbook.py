"""Offline checks of the frozen MRNA gate and its retained-format preflight."""

from decimal import Decimal
import json

import pytest

from tools import screen_mrna_bstock_topbook as study


def book(symbol="MRNABUSDT", **values):
    return json.dumps(
        {
            "symbol": symbol,
            "askPrice": "100",
            "askQty": "2",
            "bidPrice": "99",
            "bidQty": "3",
            **values,
        }
    ).encode()


@pytest.mark.parametrize(
    "value", [None, True, 100, "NaN", "Infinity", "-1", "1e2", "", "1" * 31]
)
def test_invalid_decimal_is_not_economic_evidence(value):
    with pytest.raises(ValueError):
        study.parse_book(book(askPrice=value), "MRNABUSDT")


def test_wrong_symbol_and_nonobject_rejected():
    for raw in (book("ETHUSDT"), b"[]"):
        with pytest.raises(ValueError):
            study.parse_book(raw, "MRNABUSDT")


@pytest.mark.parametrize(
    "bid,skew,quantity,passed",
    [
        ("100", 1, "1", False),
        ("100.5", 1, "1", False),
        ("100.50001", 10000, "1", True),
        ("101", 10001, "1", False),
        ("101", 0, "0", False),
    ],
)
def test_exact_strict_stress_skew_and_quantity_gate(bid, skew, quantity, passed):
    spot = study.parse_book(book(), "MRNABUSDT")
    future = study.parse_book(
        book("MRNAUSDT", bidPrice=bid, bidQty=quantity), "MRNAUSDT"
    )
    result = study.economics(spot, future, skew)
    assert result["passes_fixed_rejection_gate"] is passed
    assert Decimal(result["stressed_headroom_USDT_per_share"]) == Decimal(
        bid
    ) - Decimal("100.5")


def test_zero_ask_does_not_divide_or_pass():
    spot = study.parse_book(book(askPrice="0"), "MRNABUSDT")
    future = study.parse_book(book("MRNAUSDT"), "MRNAUSDT")
    result = study.economics(spot, future, 0)
    assert result["stressed_headroom_bps"] is None
    assert result["passes_fixed_rejection_gate"] is False


def test_production_loader_accepts_hash_bound_retained_actual_books():
    # Retained CRWD is a schema preflight only, never a new MRNA outcome.
    for product, ticker in (("crwdb-spot", "CRWDBUSDT"), ("crwd-futures", "CRWDUSDT")):
        path = (
            study.ROOT
            / f"docs/model-research/action-value/binance-{product}-book-source-result-v1-2026-09-04.json"
        )
        source = study.load_bound(path, "result_sha256")
        receipt = source["capture"]["receipt"]
        raw = (study.ROOT / receipt["raw_path"]).read_bytes()
        assert study.hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
        assert set(study.parse_book(raw, ticker)) == {
            "askPrice",
            "askQty",
            "bidPrice",
            "bidQty",
        }


def test_retained_mrna_result_reconstructs_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("retained reconstruction attempted capture")

    monkeypatch.setattr(study, "capture", forbidden)
    retained = study.load_bound(study.DIRECTORY / "result.json", "result_sha256")
    retained.pop("result_sha256")
    assert study.run(offline=True) == retained
    assert retained["status"] == "exact_observation_rejected"
    assert retained["economics"]["gross_entry_headroom_USDT_per_share"] == "-0.36000000"
    assert retained["accepted_edge"] is False


def test_retained_freeze_precedes_both_complete_journaled_responses():
    contract = study.validate()
    frozen = study.datetime.fromisoformat(
        contract["frozen_at_utc"].replace("Z", "+00:00")
    )
    assert frozen >= study.NOT_BEFORE
    for source in contract["sources"]:
        plan = study.load_bound(
            study.ROOT / source["path"], "contract_sha256", source["sha256"]
        )
        result = study.load_bound(
            study.ROOT / plan["outputs"]["result_path"], "result_sha256"
        )
        rows = [
            json.loads(line)
            for line in (study.ROOT / plan["outputs"]["journal_path"])
            .read_text()
            .splitlines()
        ]
        assert [row["phase"] for row in rows] == ["intent", "completed"]
        assert rows[0]["contract_sha256"] == source["sha256"]
        assert rows[0]["request"] == plan["request"]
        assert (
            frozen.timestamp() * 1000
            <= rows[0]["requested_at_ms"]
            <= rows[1]["completed_at_ms"]
        )
        assert rows[1] == result["capture"]["receipt"]
        assert rows[1]["status_code"] == 200
        assert rows[1]["error_type"] is None


def test_consumed_capture_cannot_repeat(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("consumed study attempted capture")

    monkeypatch.setattr(study, "capture", forbidden)
    with pytest.raises(FileExistsError, match="already consumed"):
        study.run(offline=False)


def test_rank12_and_durability_bind_exact_result_without_global_count_pin():
    registry = study.load_bound(
        study.ROOT / "docs/model-research/structural-edge-priority-registry-v1.json",
        "result_sha256",
    )
    audit = study.load_bound(
        study.ROOT
        / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json",
        "result_sha256",
    )
    result = study.load_bound(study.DIRECTORY / "result.json", "result_sha256")
    row = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 12
    )
    assert {
        "path": study.relative(study.DIRECTORY / "result.json"),
        "result_sha256": result["result_sha256"],
    } in row["canonical_artifacts"]
    terminals = [
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"] == "binance_MRNABUSDT_MRNAUSDT_september8_topbook"
    ]
    assert len(terminals) == 1
    assert terminals[0]["canonical_result_sha256"] == result["result_sha256"]
    assert (
        audit["source_binding"]["registry_result_sha256"] == registry["result_sha256"]
    )
