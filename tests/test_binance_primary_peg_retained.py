from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/binance/"
    "spot-primary-peg-execution-overlay-contract-v1-2026-08-30.json"
)
CANDIDATE = ROOT / (
    "docs/model-research/binance/"
    "spot-primary-peg-execution-overlay-candidate-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RAW_DIR = ROOT / (
    "docs/model-research/binance/raw/spot-primary-peg-execution-overlay-v1-2026-08-30"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object], field: str = "result_sha256") -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical(body))


def test_contract_sources_and_candidate_reconstruct() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    candidate = json.loads(CANDIDATE.read_text(encoding="ascii"))

    assert _self_hash(contract) == contract["result_sha256"]
    assert _self_hash(candidate) == candidate["result_sha256"]
    assert contract["implementation"]["tool_sha256"] == _sha256(
        (ROOT / contract["implementation"]["tool_path"]).read_bytes()
    )
    for source in contract["official_sources"].values():
        assert _sha256((ROOT / source["path"]).read_bytes()) == source["sha256"]
    for source in contract["windows"]:
        assert _sha256((ROOT / source["path"]).read_bytes()) == source["sha256"]

    assert candidate["sources"]["contract_file_sha256"] == _sha256(
        CONTRACT.read_bytes()
    )
    assert candidate["sources"]["contract_result_sha256"] == contract["result_sha256"]
    assert (
        candidate["sources"]["tool_sha256"] == contract["implementation"]["tool_sha256"]
    )


def test_official_request_intent_and_receipt_are_durable() -> None:
    intent = json.loads(
        (RAW_DIR / "00-request-intent.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (RAW_DIR / "02-request-receipt.json").read_text(encoding="utf-8")
    )
    payload = (RAW_DIR / "01-official-pegged-orders.raw.md").read_bytes()

    assert intent["method"] == receipt["method"] == "GET"
    assert intent["url"] == receipt["url"]
    assert intent["authority"] == "public_unauthenticated_read_only"
    assert intent["body_sha256"] == _sha256(b"")
    assert receipt["status_code"] == 200
    assert receipt["payload_bytes"] == len(payload) == 6410
    assert receipt["payload_sha256"] == _sha256(payload)
    assert receipt["credentials_used"] is False


def test_recurrent_rejection_counterfactual_is_not_profit_evidence() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="ascii"))
    expected = {
        "discovery": {
            "BTCUSDT": (1187, 83),
            "ETHUSDT": (1121, 218),
            "SOLUSDT": (1057, 115),
        },
        "validation": {
            "BTCUSDT": (291, 8),
            "ETHUSDT": (232, 17),
            "SOLUSDT": (206, 6),
        },
    }
    for window in candidate["windows"]:
        for symbol, (comparisons, rejections) in expected[window["role"]].items():
            metrics = window["symbols"][symbol]
            assert metrics["comparison_rows"] == comparisons
            assert metrics["either_side_rejection_count"] == rejections
            assert int(metrics["either_side_rejection_count"]) > 0

    verdict = candidate["verdict"]
    assert verdict["recurrent_in_every_window_and_symbol"] is True
    assert verdict["accepted_edge"] is False
    assert verdict["profitability_claim"] is False
    assert verdict["deployment_ready"] is False
    assert verdict["trading_authority"] is False
    assert candidate["authority"] == {
        "network_requests": 0,
        "credentials_used": False,
        "orders_or_cancellations": 0,
        "account_or_funded_actions": 0,
    }


def test_rank_five_registry_lineage_is_fail_closed() -> None:
    candidate = json.loads(CANDIDATE.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _self_hash(registry) == registry["result_sha256"]
    rank_five = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert {
        "path": CANDIDATE.relative_to(ROOT).as_posix(),
        "result_sha256": candidate["result_sha256"],
    } in rank_five["canonical_artifacts"]
    assert any("PRIMARY_PEG" in item for item in rank_five["blocking_evidence"])
    assert "testnet_or_paper" in rank_five["retry_trigger"]
    assert registry["accepted_edge_count"] == 21
    assert len(registry["prioritized_hypotheses"]) == 44
