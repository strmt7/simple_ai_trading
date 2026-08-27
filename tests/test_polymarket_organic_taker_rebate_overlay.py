from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-organic-taker-rebate-overlay-v1-2026-08-26.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RAW = ROOT / ("docs/model-research/action-value/raw/polymarket-organic-taker-rebate-v1")
EXPECTED_HASH = "6a3f907dbebd0c7cc894d95054231540e50cd8e28e6264840a2840be8ac72865"
EXPECTED_REGISTRY_HASH = (
    "aabfdc0750a619b380929c59546d37c86306686bc2144d85c90d770f5bea6d23"
)
PUSD = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
V2_EXCHANGE = "0xe111180000d2663c0091e4f400237545b87b996b"
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def _load(path: Path) -> dict[str, object] | list[dict[str, object]]:
    return json.loads(path.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _decode_abi_string(value: str) -> str:
    encoded = bytes.fromhex(value.removeprefix("0x"))
    offset = int.from_bytes(encoded[:32])
    length = int.from_bytes(encoded[offset : offset + 32])
    return encoded[offset + 32 : offset + 32 + length].decode("ascii")


def test_artifact_and_raw_evidence_are_exactly_hash_bound() -> None:
    artifact = _load(ARTIFACT)
    assert isinstance(artifact, dict)
    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _embedded_hash(artifact) == EXPECTED_HASH
    for receipt in artifact["raw_evidence"]:
        path = ROOT / receipt["path"]
        content = path.read_bytes()
        assert len(content) == receipt["bytes"]
        assert hashlib.sha256(content).hexdigest() == receipt["sha256"]

    common = (RAW / "13-ts-sdk-common.ts").read_text(encoding="utf-8")
    activity = (RAW / "14-ts-sdk-activity.ts").read_text(encoding="utf-8")
    assert "TAKER_REBATE = 'TAKER_REBATE'" in common
    assert "type: z.array(ActivityTypeSchema).optional()" in activity


def test_public_trade_window_reconstructs_the_gold_fee_rebate() -> None:
    artifact = _load(ARTIFACT)
    trades = _load(RAW / "05-top-wallet-taker-trades-2026-08-25.json")
    assert isinstance(artifact, dict)
    assert isinstance(trades, list)
    observed = artifact["economics"]["observed_public_reconciliation"]
    assert len(trades) == observed["observed_trade_rows"] == 1202
    assert len(trades) < 10_000
    assert {row["side"] for row in trades} == {"BUY"}
    assert len({row["conditionId"] for row in trades}) == 355

    with localcontext() as context:
        context.prec = 50
        base = sum(
            Decimal(str(row["size"]))
            * Decimal(str(row["price"]))
            * (Decimal(1) - Decimal(str(row["price"])))
            for row in trades
        )
        notional = sum(
            Decimal(str(row["size"])) * Decimal(str(row["price"])) for row in trades
        )
        fee = base * Decimal("0.07")
        payout = Decimal(observed["observed_payout_pusd"])
        assert base * Decimal("2.3") == Decimal(
            observed["calculated_crypto_weighted_volume"]
        )
        assert notional == Decimal(observed["calculated_taker_trade_value_pusd"])
        assert fee == Decimal(
            observed["calculated_crypto_fee_pusd_before_internal_rounding"]
        )
        assert payout / fee == Decimal(
            observed["observed_payout_divided_by_calculated_fee"]
        )
        assert fee * Decimal("0.18") == Decimal(
            observed["expected_Gold_18_percent_rebate_pusd_before_internal_rounding"]
        )
        assert payout - fee * Decimal("0.18") == Decimal(
            observed["observed_minus_expected_pusd"]
        )
        assert abs(payout - fee * Decimal("0.18")) < Decimal("0.001")


def test_activity_and_onchain_pusd_transfer_match_exactly() -> None:
    artifact = _load(ARTIFACT)
    activity = _load(RAW / "02-top-wallet-taker-rebate-activity.json")
    receipt = _load(RAW / "04-latest-rebate-receipt-response.json")
    assert isinstance(artifact, dict)
    assert isinstance(activity, list)
    assert isinstance(receipt, dict)
    realized = artifact["evidence"]["realized_payout"]
    latest = activity[0]
    assert len(activity) == realized["activity_rows_for_wallet"] == 63
    assert latest["type"] == "TAKER_REBATE"
    assert Decimal(str(latest["usdcSize"])) == Decimal("54.5062")

    result = receipt["result"]
    wallet = artifact["economics"]["observed_public_reconciliation"]["wallet"]
    matches = [
        row
        for row in result["logs"]
        if row["address"].lower() == PUSD
        and row["topics"][0].lower() == TRANSFER_TOPIC
        and ("0x" + row["topics"][2][-40:]).lower() == wallet
    ]
    assert result["status"] == "0x1"
    assert len(matches) == 1
    assert int(matches[0]["data"], 16) == 54_506_200
    assert ("0x" + matches[0]["topics"][1][-40:]).lower() == realized[
        "observed_batch_sender"
    ]


def test_deployed_v2_fee_collateral_is_pusd_without_a_parity_assumption() -> None:
    artifact = _load(ARTIFACT)
    request = _load(RAW / "15-v2-collateral-rpc-request.json")
    response = _load(RAW / "16-v2-collateral-rpc-response.json")
    assert isinstance(artifact, dict)
    assert isinstance(request, list)
    assert isinstance(response, list)
    assert request[0]["params"][0]["to"].lower() == V2_EXCHANGE
    assert request[0]["params"][0]["data"] == "0x5c1548fb"
    assert request[1]["params"][0]["to"].lower() == PUSD
    assert request[1]["params"][0]["data"] == "0x95d89b41"
    assert request[2]["params"][0]["data"] == "0x06fdde03"

    by_id = {row["id"]: row["result"] for row in response}
    assert ("0x" + by_id[1][-40:]).lower() == PUSD
    assert _decode_abi_string(by_id[2]) == "pUSD"
    assert _decode_abi_string(by_id[3]) == "Polymarket USD"
    resolution = artifact["evidence"]["v2_fee_unit_resolution"]
    assert resolution["deployed_exchange"] == V2_EXCHANGE
    assert resolution["deployed_exchange_getCollateral"] == PUSD
    assert resolution["onchain_collateral_symbol"] == "pUSD"
    assert (
        "No USDC-to-pUSD exchange-rate or parity assumption" in resolution["resolution"]
    )


def test_promotion_is_scoped_non_authorizing_and_registered_as_edge_ten() -> None:
    artifact = _load(ARTIFACT)
    registry = _load(REGISTRY)
    assert isinstance(artifact, dict)
    assert isinstance(registry, dict)
    assert artifact["adjudication"]["accepted_edge"] is True
    assert artifact["adjudication"]["deployment_ready"] is False
    assert artifact["adjudication"]["trading_authority"] is False
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_quotes_submitted"] == 0
    assert any("self matching" in row for row in artifact["prohibited_actions"])

    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_HASH
    assert registry["accepted_edge_count"] == 16
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 41))
    candidate = next(
        row
        for row in hypotheses
        if row["mechanism"] == "polymarket_organic_taker_fee_rebate_overlay"
    )
    assert candidate["priority_rank"] == 22
    assert candidate["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-organic-taker-rebate-overlay-v1-2026-08-26.json"
            ),
            "result_sha256": EXPECTED_HASH,
        }
    ]
