from __future__ import annotations

from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-institutional-loan-interest-rebate-overlay-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


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


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return _sha256(_canonical(body))


def _release_text(payload: bytes) -> str:
    parser = _TextParser()
    parser.feed(payload.decode("utf-8"))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def test_issuer_release_and_current_api_sources_reconstruct() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact) == artifact["result_sha256"]
    release_source = artifact["sources"]["issuer_distributed_release"]
    index_source = artifact["sources"]["current_official_api_index"]
    release = (ROOT / release_source["path"]).read_bytes()
    index = (ROOT / index_source["path"]).read_bytes()
    assert _sha256(release) == release_source["sha256"]
    assert _sha256(index) == index_source["sha256"]

    text = _release_text(release)
    assert "News provided by Binance 11 May, 2026" in text
    assert "Effective June 1, 2026" in text
    assert "may also qualify for full monthly interest rebates" in text
    assert "incremental trading volume share, Open Interest, or Net Asset Value" in text
    assert "USDT, USDC, BTC, and $U (United Stables), up to $10 million" in text
    assert "KYB-verified clients at VIP 1 and above are eligible" in text
    assert "All of your Institutional Lending Account balance may be liquidated" in text

    decoded_index = index.decode("utf-8")
    for endpoint in artifact["signed_read_only_contract"][
        "required_endpoints_after_exact_authority"
    ]:
        assert f"`{endpoint}`" in decoded_index
        assert f"`{endpoint}`" in decoded_index and "(USER_DATA)" in next(
            line for line in decoded_index.splitlines() if f"`{endpoint}`" in line
        )


def test_public_access_was_read_only_and_has_no_forward_profit_claim() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    intent = json.loads(
        (ROOT / artifact["sources"]["request_intent"]).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (ROOT / artifact["sources"]["request_receipt"]).read_text(encoding="utf-8")
    )
    assert intent["method"] == receipt["method"] == "GET"
    assert intent["url"] == receipt["url"]
    assert intent["authority"] == "public_unauthenticated_read_only"
    assert receipt["status_code"] == 200
    assert receipt["credentials_used"] is False
    assert (
        receipt["payload_sha256"]
        == artifact["sources"]["issuer_distributed_release"]["sha256"]
    )
    assert artifact["economic_contract"]["public_forward_floor_quote_units"] == "0"
    assert artifact["adjudication"]["profitability_claim"] is False
    assert artifact["adjudication"]["deployment_ready"] is False
    assert artifact["authority"]["account_requests"] == 0
    assert artifact["authority"]["loan_or_collateral_actions"] == 0


def test_narrow_realized_credit_overlay_is_accepted_without_loan_authority() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    adjudication = artifact["adjudication"]
    assert adjudication["accepted_edge"] is True
    assert adjudication["market_direction_forecast_required"] is False
    assert adjudication["trading_authority"] is False
    assert "exact realized monthly" in adjudication["accepted_scope"]
    assert (
        "independently required existing eligible loan"
        in adjudication["accepted_scope"]
    )

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry) == registry["result_sha256"]
    lead = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 26
    )
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in lead["canonical_artifacts"]
