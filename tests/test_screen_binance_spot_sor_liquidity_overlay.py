import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from tools.screen_binance_spot_sor_liquidity_overlay import (
    _evaluate_group,
    _parse_exchange_info,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-spot-sor-liquidity-overlay-contract-v1-2026-08-29.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-spot-sor-liquidity-overlay-result-v1-2026-08-29.json"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
JOURNAL_PATH = ROOT / (
    "data/binance-spot-sor-liquidity-overlay-v1/request-journal.jsonl"
)
RAW_GZIP_PATH = ROOT / (
    "data/binance-spot-sor-liquidity-overlay-v1/raw/exchangeInfo.json.gz"
)


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _symbol(symbol: str, base: str, quote: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "baseAsset": base,
        "quoteAsset": quote,
        "status": "TRADING",
        "isSpotTradingAllowed": True,
        "orderTypes": ["LIMIT", "MARKET"],
    }


def test_exchange_info_selects_only_scoped_valid_sor_groups() -> None:
    payload = {
        "symbols": [
            _symbol("BTCUSDT", "BTC", "USDT"),
            _symbol("BTCUSDC", "BTC", "USDC"),
            _symbol("BNBUSDT", "BNB", "USDT"),
            _symbol("BNBUSDC", "BNB", "USDC"),
        ],
        "sors": [
            {"baseAsset": "BTC", "symbols": ["BTCUSDT", "BTCUSDC"]},
            {"baseAsset": "BNB", "symbols": ["BNBUSDT", "BNBUSDC"]},
        ],
    }
    groups, symbols = _parse_exchange_info(payload, {"BTC", "ETH", "SOL"})
    assert groups == [
        {
            "base_asset": "BTC",
            "symbols": ["BTCUSDT", "BTCUSDC"],
            "quote_assets": ["USDT", "USDC"],
        }
    ]
    assert len(symbols) == 4


def test_sor_top_level_evaluation_weakly_dominates_and_enforces_capacity() -> None:
    group = {"base_asset": "BTC", "symbols": ["BTCUSDT", "BTCUSDC"]}
    books = {
        "BTCUSDT": {
            "bidPrice": Decimal("99"),
            "bidQty": Decimal("20"),
            "askPrice": Decimal("101"),
            "askQty": Decimal("20"),
        },
        "BTCUSDC": {
            "bidPrice": Decimal("100"),
            "bidQty": Decimal("20"),
            "askPrice": Decimal("100"),
            "askQty": Decimal("20"),
        },
    }
    rows = _evaluate_group(group, books, [Decimal("100")], Decimal("1"))
    by_key = {(row["submitted_symbol"], row["side"]): row for row in rows}
    assert Decimal(by_key[("BTCUSDT", "BUY")]["gross_improvement_bips"]) > 1
    assert Decimal(by_key[("BTCUSDT", "SELL")]["gross_improvement_bips"]) > 1
    assert by_key[("BTCUSDC", "BUY")]["gross_improvement_bips"] == "0"
    assert by_key[("BTCUSDC", "SELL")]["gross_improvement_bips"] == "0"

    books["BTCUSDT"]["askQty"] = Decimal("0.1")
    incomplete = _evaluate_group(group, books, [Decimal("100")], Decimal("1"))
    direct_buy = next(
        row
        for row in incomplete
        if row["submitted_symbol"] == "BTCUSDT" and row["side"] == "BUY"
    )
    assert direct_buy["top_level_comparison_complete"] is False
    assert direct_buy["public_candidate"] is False


def test_one_request_terminal_result_reconstructs_retained_evidence() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    journal = [
        json.loads(line)
        for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    for implementation in contract["implementations"]:
        implementation_path = ROOT / implementation["path"]
        assert (
            hashlib.sha256(implementation_path.read_bytes()).hexdigest()
            == (implementation["sha256"])
        )
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["result_sha256"] == (
        "895dc0eba4f72b9b08b19dbba245b20434e4db905fd4ded3ea70779733db6d47"
    )

    receipt = result["capture"]["exchange_info_receipt"]
    with gzip.open(RAW_GZIP_PATH, "rb") as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
    assert len(raw) == receipt["response_bytes"]
    assert hashlib.sha256(RAW_GZIP_PATH.read_bytes()).hexdigest() == (
        "9f3f83fe1efcec7a0230dea646d2038fe5c0a32ae72de5c7e4f5ee7fd850304b"
    )

    assert [entry["phase"] for entry in journal] == ["intent", "completed"]
    assert journal[1] == receipt
    assert result["capture"]["book_ticker_receipt"] is None
    assert result["screen"]["scoped_group_count"] == 0
    assert result["screen"]["public_candidate_count"] == 0
    assert result["adjudication"]["accepted_edge"] is False
    assert result["authority"] == {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }

    assert _canonical_hash(registry, "result_sha256") == (
        "0a34d7289331515f8e7b3f09e856fbc331ecbc3a91130fea20542a39ef211f60"
    )
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert (
        terminal["binance_spot_SOR_liquidity_overlay_current_production_configuration"][
            "canonical_result_sha256"
        ]
        == result["result_sha256"]
    )
    assert len(registry["prioritized_hypotheses"]) == 44
    assert registry["accepted_edge_count"] == 21
