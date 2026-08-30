import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
RAW_ROOT = ROOT / (
    "docs/model-research/binance/raw/"
    "spot-block-matching-cost-overlay-v1-2026-08-30"
)
BLOCK_RESULT = ACTION_VALUE / (
    "binance-spot-block-matching-organic-cost-overlay-candidate-v1-2026-08-30.json"
)
DUST_FAILURE = ACTION_VALUE / (
    "binance-dust-convert-source-capture-failure-v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(document: dict, field: str = "result_sha256") -> str:
    body = dict(document)
    claimed = body.pop(field)
    actual = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert actual == claimed
    return actual


def test_spot_block_matching_candidate_is_source_bound_and_fail_closed() -> None:
    result = json.loads(BLOCK_RESULT.read_text(encoding="utf-8"))
    result_hash = _canonical_hash(result)
    assert result_hash == (
        "2d9c4872a6ecd707716cd8d769eb20cb715c1ed7feb61e7e560a5efdb169dc57"
    )

    for source in (
        "request_contract",
        "cms_request_contract",
        "request_journal",
        "official_introduction",
        "official_agent_native_index",
        "official_current_faq_cms",
    ):
        binding = result["source_binding"][source]
        payload = (ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding["file_sha256"]

    cms = json.loads(
        (RAW_ROOT / "05-official-faq-cms.raw.json").read_text(encoding="utf-8")
    )
    assert cms["success"] is True
    assert cms["data"]["code"] == "cc66271ed16f4e24a26a0dedd90e00c8"
    assert cms["data"]["lastUpdateTime"] == 1784533800000
    assert "Maker and Taker fees are both 0.025%." in cms["data"]["body"]
    assert "up to 10% above or below the market price" in cms["data"]["body"]

    terms = result["current_primary_terms"]
    assert terms["maker_fee_bps"] == terms["taker_fee_bps"] == "2.5"
    assert terms["market_maker_rebate_available"] is False
    assert result["rejection_first_economics"]["public_forward_profit_floor_quote"] == (
        "0"
    )
    assert result["verdict"]["accepted_edge"] is False
    assert result["verdict"]["profitability_proved"] is False

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    _canonical_hash(registry)
    rank_five = next(
        item for item in registry["prioritized_hypotheses"] if item["priority_rank"] == 5
    )
    assert any(
        artifact["result_sha256"] == result_hash
        for artifact in rank_five["canonical_artifacts"]
    )


def test_dust_documentation_capture_failure_remains_terminal() -> None:
    failure = json.loads(DUST_FAILURE.read_text(encoding="utf-8"))
    _canonical_hash(failure)
    consumed = failure["consumed_request"]
    assert consumed["http_status"] == 202
    assert consumed["response_bytes"] == 0
    assert failure["adjudication"]["accepted_edge"] is False
    assert failure["adjudication"]["exact_request_retry_allowed"] is False
    assert failure["adjudication"]["endpoint_alias_repair_allowed"] is False
