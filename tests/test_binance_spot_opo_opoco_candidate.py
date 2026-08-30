from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/model-research/action-value/"
    "binance-spot-opo-opoco-received-quantity-execution-overlay-candidate-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _symbol_projection(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    projection: dict[str, dict[str, object]] = {}
    for symbol in payload["symbols"]:  # type: ignore[index]
        filters = {
            row["filterType"]: row  # type: ignore[index]
            for row in symbol["filters"]  # type: ignore[index]
        }
        projection[symbol["symbol"]] = {  # type: ignore[index]
            "status": symbol["status"],  # type: ignore[index]
            "otoAllowed": symbol["otoAllowed"],  # type: ignore[index]
            "opoAllowed": symbol["opoAllowed"],  # type: ignore[index]
            "ocoAllowed": symbol["ocoAllowed"],  # type: ignore[index]
            "minQty": filters["LOT_SIZE"]["minQty"],
            "stepSize": filters["LOT_SIZE"]["stepSize"],
            "minNotional": filters["NOTIONAL"]["minNotional"],
            "maxNumOrders": filters["MAX_NUM_ORDERS"]["maxNumOrders"],
            "maxNumOrderLists": filters["MAX_NUM_ORDER_LISTS"][
                "maxNumOrderLists"
            ],
        }
    return projection


def test_opo_terms_and_production_testnet_configuration_are_source_bound() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]

    sources = result["sources"]
    for source_name in (
        "official_OPO_FAQ",
        "official_OPO_source_receipt",
        "official_changelog",
        "official_agent_native_index",
        "retained_production_exchangeInfo",
        "frozen_testnet_exchangeInfo_raw",
    ):
        source = sources[source_name]
        assert _sha256(ROOT / source["path"]) == source["sha256"]

    faq = (ROOT / sources["official_OPO_FAQ"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "once it fully fills" in faq
    assert "commission deducted as appropriate" in faq
    assert "Only working orders on the `BUY` side" in faq

    production = json.loads(
        (ROOT / sources["retained_production_exchangeInfo"]["path"]).read_bytes()
    )
    testnet = json.loads(
        (ROOT / sources["frozen_testnet_exchangeInfo_raw"]["path"]).read_bytes()
    )
    production_projection = _symbol_projection(production)
    testnet_projection = _symbol_projection(testnet)
    expected_symbols = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert expected_symbols <= production_projection.keys()
    assert testnet_projection.keys() == expected_symbols
    assert {
        symbol: production_projection[symbol] for symbol in expected_symbols
    } == testnet_projection
    assert all(row["opoAllowed"] is True for row in testnet_projection.values())

    contract_source = sources["frozen_testnet_exchangeInfo_contract"]
    contract = json.loads((ROOT / contract_source["path"]).read_bytes())
    assert _canonical_hash(contract, "contract_sha256") == contract_source[
        "contract_sha256"
    ]
    source_result_source = sources["frozen_testnet_exchangeInfo_source_result"]
    source_result = json.loads((ROOT / source_result_source["path"]).read_bytes())
    assert _canonical_hash(source_result, "result_sha256") == source_result_source[
        "result_sha256"
    ]
    assert source_result["authority"]["credentials_used"] is False


def test_opo_candidate_and_standalone_rejection_are_routed_without_global_pins() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["candidate"] is True
    assert result["adjudication"]["standalone_profit_claim_terminal"] is True
    assert result["mechanism"]["pending_activation_before_full_fill"] is False
    assert result["structural_value"]["standalone_after_cost_profit_floor_USDT"] == "0"

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_5 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert rank_5["opo_execution_retry_trigger"] == result["adjudication"][
        "retry_trigger"
    ]
    assert any(
        row["result_sha256"] == result["result_sha256"]
        for row in rank_5["canonical_artifacts"]
    )
    assert any(
        row["canonical_result_sha256"] == result["result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    )
