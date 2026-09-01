from __future__ import annotations

from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.adjudicate_polymarket_release_date_deadline_graph import _package_row


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict[str, object]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_hash(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _market(
    market_id: str,
    question: str,
    *,
    best_bid: str,
    best_ask: str,
    tick: str = "0.01",
) -> dict[str, object]:
    return {
        "id": market_id,
        "question": question,
        "conditionId": "0x" + market_id.zfill(64),
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "bestBid": best_bid,
        "bestAsk": best_ask,
        "outcomes": '["Yes","No"]',
        "clobTokenIds": f'["{market_id}0","{market_id}1"]',
        "orderPriceMinTickSize": tick,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "rate": 0.04,
            "exponent": 1,
            "takerOnly": True,
            "rebateRate": 0.25,
        },
    }


def test_implication_row_uses_no_proxy_direct_yes_and_all_cost_gate() -> None:
    exact = _market("1", "exact", best_bid="0.30", best_ask="0.72")
    deadline = _market("2", "deadline", best_bid="0.20", best_ask="0.20")
    row = _package_row(
        family="implication",
        relation="A_implies_B",
        exact_date=date(2026, 9, 1),
        deadline_date=date(2026, 9, 1),
        first_market=exact,
        first_outcome="No",
        second_market=deadline,
        second_outcome="Yes",
    )

    assert row["metadata_cost_pUSD_per_share"] == "0.90"
    assert row["passes_strict_metadata_gate"] is True
    assert row["passes_fee_and_one_tick_gate"] is True
    assert Decimal(str(row["after_fee_one_tick_profit_floor_pUSD"])) > 0
    assert row["legs"][0]["price_source"] == "one_minus_direct_YES_bestBid"
    assert row["legs"][1]["price_source"] == "direct_YES_bestAsk"


def test_mutual_exclusion_row_rejects_at_floor_before_depth() -> None:
    exact = _market("3", "exact", best_bid="0.10", best_ask="0.91")
    deadline = _market("4", "deadline", best_bid="0.90", best_ask="0.11")
    row = _package_row(
        family="mutual_exclusion",
        relation="not_both",
        exact_date=date(2026, 9, 2),
        deadline_date=date(2026, 9, 1),
        first_market=exact,
        first_outcome="No",
        second_market=deadline,
        second_outcome="No",
    )

    assert row["metadata_cost_pUSD_per_share"] == "1.00"
    assert row["passes_strict_metadata_gate"] is False
    assert row["passes_fee_and_one_tick_gate"] is False


def test_missing_side_specific_price_fails_closed() -> None:
    exact = _market("5", "exact", best_bid="0.20", best_ask="0.81")
    deadline = _market("6", "deadline", best_bid="0.40", best_ask="0.40")
    deadline["bestAsk"] = None
    row = _package_row(
        family="implication",
        relation="A_implies_B",
        exact_date=date(2026, 9, 1),
        deadline_date=date(2026, 9, 1),
        first_market=exact,
        first_outcome="No",
        second_market=deadline,
        second_outcome="Yes",
    )

    assert row["status"] == "missing_side_specific_acquisition_evidence"
    assert row["passes_strict_metadata_gate"] is False
    assert row["passes_fee_and_one_tick_gate"] is False


def test_bound_mythos_graph_is_terminal_and_registry_bound() -> None:
    exact_contract = _load(
        "docs/model-research/action-value/"
        "polymarket-mythos-release-on-metadata-contract-v1-2026-09-01.json"
    )
    exact_capture = _load(
        "docs/model-research/action-value/"
        "polymarket-mythos-release-on-metadata-capture-result-v1-2026-09-01.json"
    )
    deadline_contract = _load(
        "docs/model-research/action-value/"
        "polymarket-mythos-release-by-metadata-contract-v1-2026-09-01.json"
    )
    deadline_capture = _load(
        "docs/model-research/action-value/"
        "polymarket-mythos-release-by-metadata-capture-result-v1-2026-09-01.json"
    )
    contract = _load(
        "docs/model-research/action-value/"
        "polymarket-mythos-release-date-deadline-graph-contract-v1-2026-09-01.json"
    )
    result = _load(
        "docs/model-research/action-value/"
        "polymarket-mythos-release-date-deadline-graph-adjudication-v1-2026-09-01.json"
    )
    registry = _load(
        "docs/model-research/structural-edge-priority-registry-v1.json"
    )
    audit = _load(
        "docs/model-research/action-value/"
        "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
    )

    for metadata_contract, capture in (
        (exact_contract, exact_capture),
        (deadline_contract, deadline_capture),
    ):
        assert (
            _canonical_hash(metadata_contract, "contract_sha256")
            == metadata_contract["contract_sha256"]
        )
        assert _canonical_hash(capture, "result_sha256") == capture["result_sha256"]
        assert capture["contract"]["sha256"] == metadata_contract[  # type: ignore[index]
            "contract_sha256"
        ]
        receipt = capture["capture"]["receipt"]  # type: ignore[index]
        assert receipt["status_code"] == 200
        assert receipt["response_sha256"] == _file_hash(receipt["raw_path"])
        assert capture["source_gate"]["passed"] is True  # type: ignore[index]
        assert capture["authority"]["credentials_used"] is False  # type: ignore[index]

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    assert result["contract"]["sha256"] == contract["contract_sha256"]  # type: ignore[index]
    assert result["scope"]["valid_relation_count"] == 208  # type: ignore[index]
    assert result["summary"]["strict_metadata_subfloor_count"] == 0  # type: ignore[index]
    assert result["summary"]["fee_and_one_tick_positive_count"] == 0  # type: ignore[index]
    assert result["adjudication"]["book_request_justified"] is False  # type: ignore[index]
    assert result["authority"]["network_requests"] == 0  # type: ignore[index]

    family = (
        "polymarket_Mythos_exact_release_date_to_cumulative_deadline_"
        "exhaustive_implication_graph_2026_09_01"
    )
    terminal = {
        row["family"]: row
        for row in registry["terminal_do_not_repeat"]  # type: ignore[index]
    }
    assert len(registry["terminal_do_not_repeat"]) == 140  # type: ignore[arg-type]
    assert terminal[family]["canonical_result_sha256"] == result["result_sha256"]
    assert audit["source_binding"]["registry_result_sha256"] == registry[  # type: ignore[index]
        "result_sha256"
    ]
