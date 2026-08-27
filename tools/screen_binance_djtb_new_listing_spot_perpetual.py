"""Run the frozen DJTB bStock versus DJT perpetual new-listing screen."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
BSTOCK_LIST_URL = (
    "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/"
    "market/token/rwa/stock/detail/list/ai"
)
SCHEMA_VERSION = "binance-djtb-new-listing-spot-perpetual-v1"
SPOT_SYMBOL = "DJTBUSDT"
FUTURES_SYMBOL = "DJTUSDT"
FIXED_STRESS_BPS = Decimal("50")
FUNDING_EVENTS = 8


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical_json(body).encode("ascii"))


def _filter(symbol: Mapping[str, object], filter_type: str) -> dict[str, object]:
    for value in _list(symbol.get("filters"), name="symbol filters"):
        row = _mapping(value, name="symbol filter")
        if row.get("filterType") == filter_type:
            return row
    raise ValueError(f"{symbol.get('symbol')} lacks {filter_type}")


def _minimum_notional(symbol: Mapping[str, object]) -> Decimal:
    for filter_name in ("MIN_NOTIONAL", "NOTIONAL"):
        try:
            row = _filter(symbol, filter_name)
        except ValueError:
            continue
        value = row.get("minNotional") or row.get("notional")
        if value is not None:
            return Decimal(str(value))
    raise ValueError(f"{symbol.get('symbol')} lacks minimum notional")


def _common_step(left: Decimal, right: Decimal) -> Decimal:
    step = max(left, right)
    smaller = min(left, right)
    if step <= 0 or smaller <= 0 or step % smaller != 0:
        raise ValueError("quantity steps do not have an exact common coarser step")
    return step


def _round_up(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def _levels(raw: object, *, side: str) -> list[tuple[Decimal, Decimal]]:
    rows: list[tuple[Decimal, Decimal]] = []
    for value in _list(raw, name=f"{side} levels"):
        level = _list(value, name=f"{side} level")
        if len(level) != 2:
            raise ValueError(f"{side} level must contain price and quantity")
        price = Decimal(str(level[0]))
        quantity = Decimal(str(level[1]))
        if min(price, quantity) <= 0:
            raise ValueError(f"{side} level values must be positive")
        rows.append((price, quantity))
    if not rows:
        raise ValueError(f"{side} book is empty")
    rows.sort(key=lambda row: row[0], reverse=side == "bids")
    return rows


def _fill(
    levels: list[tuple[Decimal, Decimal]], *, quantity: Decimal
) -> tuple[Decimal, int] | None:
    remaining = quantity
    quote = Decimal(0)
    used = 0
    for price, available in levels:
        filled = min(remaining, available)
        quote += filled * price
        remaining -= filled
        used += 1
        if remaining == 0:
            return quote, used
    return None


class _Client:
    def __init__(self, raw_dir: Path, journal_path: Path) -> None:
        self.raw_dir = raw_dir
        self.journal_path = journal_path
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "simple-ai-trading-public-edge-research/1.0",
            }
        )

    def get(
        self,
        url: str,
        *,
        name: str,
        params: Mapping[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]:
        started_ms = time.time_ns() // 1_000_000
        response = self.session.get(url, params=params, timeout=30)
        finished_ms = time.time_ns() // 1_000_000
        payload = response.content
        raw_path = self.raw_dir / f"{name}.raw"
        if raw_path.exists():
            raise FileExistsError(f"refusing to overwrite retained response: {raw_path}")
        write_bytes_atomic(raw_path, payload)
        receipt = {
            "name": name,
            "method": "GET",
            "url": response.url,
            "status_code": response.status_code,
            "requested_at_ms": started_ms,
            "completed_at_ms": finished_ms,
            "response_bytes": len(payload),
            "response_sha256": _sha256(payload),
            "raw_path": str(raw_path.as_posix()),
        }
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("ab") as stream:
            stream.write((_canonical_json(receipt) + "\n").encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        response.raise_for_status()
        try:
            return response.json(), receipt
        except requests.JSONDecodeError as exc:
            raise ValueError(f"{name} did not return JSON") from exc


def run(*, contract_path: Path, raw_dir: Path, journal_path: Path) -> dict[str, object]:
    contract = _mapping(
        json.loads(contract_path.read_text(encoding="utf-8")), name="contract"
    )
    contract_hash = _canonical_hash(contract, field="contract_sha256")
    if contract_hash != contract.get("contract_sha256"):
        raise ValueError("contract hash does not match canonical contents")
    discovery = _mapping(
        contract.get("discovery_identity_already_observed"), name="discovery identity"
    )
    required_bstock = _mapping(discovery.get("required_row"), name="required bStock")
    economics = _mapping(contract.get("economics"), name="contract economics")
    targets = [
        Decimal(str(value))
        for value in _list(economics.get("target_spot_costs_usdt"), name="targets")
    ]
    if targets != [Decimal("100"), Decimal("1000"), Decimal("5000")]:
        raise ValueError("target sizes differ from the frozen contract")

    client = _Client(raw_dir, journal_path)
    bstock_raw, bstock_source = client.get(
        BSTOCK_LIST_URL, name="bstock-list", params={"type": 3}
    )
    spot_exchange_raw, spot_exchange_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/exchangeInfo",
        name="spot-exchange-info",
        params={"symbol": SPOT_SYMBOL},
    )
    futures_exchange_raw, futures_exchange_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/exchangeInfo",
        name="futures-exchange-info",
    )
    spot_depth_raw, spot_depth_source = client.get(
        f"{SPOT_BASE_URL}/api/v3/depth",
        name="spot-depth",
        params={"symbol": SPOT_SYMBOL, "limit": 1000},
    )
    futures_depth_raw, futures_depth_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/depth",
        name="futures-depth",
        params={"symbol": FUTURES_SYMBOL, "limit": 1000},
    )
    funding_raw, funding_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/fundingRate",
        name="futures-funding",
        params={"symbol": FUTURES_SYMBOL, "limit": 1000},
    )
    premium_raw, premium_source = client.get(
        f"{FUTURES_BASE_URL}/fapi/v1/premiumIndex",
        name="futures-premium-index",
        params={"symbol": FUTURES_SYMBOL},
    )

    bstock_envelope = _mapping(bstock_raw, name="bStock envelope")
    bstock_rows = [
        _mapping(value, name="bStock row")
        for value in _list(bstock_envelope.get("data"), name="bStock rows")
    ]
    matches = [row for row in bstock_rows if row.get("ticker") == "DJT"]
    if len(matches) != 1:
        raise ValueError("expected exactly one DJT bStock identity row")
    bstock = matches[0]
    for key, expected in required_bstock.items():
        if bstock.get(key) != expected:
            raise ValueError(f"DJT bStock identity changed at {key}")

    spot_exchange = _mapping(spot_exchange_raw, name="spot exchange info")
    spot_symbols = [
        _mapping(value, name="spot symbol")
        for value in _list(spot_exchange.get("symbols"), name="spot symbols")
    ]
    if len(spot_symbols) != 1:
        raise ValueError("spot exchange info did not return exactly DJTBUSDT")
    spot_info = spot_symbols[0]
    if (
        spot_info.get("symbol") != SPOT_SYMBOL
        or spot_info.get("status") != "TRADING"
        or spot_info.get("baseAsset") != "DJTB"
        or spot_info.get("quoteAsset") != "USDT"
    ):
        raise ValueError("DJTBUSDT identity or trading state changed")

    futures_exchange = _mapping(futures_exchange_raw, name="futures exchange info")
    future_matches = [
        row
        for row in (
            _mapping(value, name="futures symbol")
            for value in _list(futures_exchange.get("symbols"), name="futures symbols")
        )
        if row.get("symbol") == FUTURES_SYMBOL
    ]
    if len(future_matches) != 1:
        raise ValueError("futures exchange info did not contain exactly DJTUSDT")
    futures_info = future_matches[0]
    if (
        futures_info.get("status") != "TRADING"
        or futures_info.get("contractType") != "TRADIFI_PERPETUAL"
        or futures_info.get("baseAsset") != "DJT"
        or futures_info.get("quoteAsset") != "USDT"
        or futures_info.get("marginAsset") != "USDT"
    ):
        raise ValueError("DJTUSDT identity or trading state changed")

    spot_depth = _mapping(spot_depth_raw, name="spot depth")
    futures_depth = _mapping(futures_depth_raw, name="futures depth")
    spot_asks = _levels(spot_depth.get("asks"), side="asks")
    futures_bids = _levels(futures_depth.get("bids"), side="bids")
    spot_lot = _filter(spot_info, "LOT_SIZE")
    futures_lot = _filter(futures_info, "LOT_SIZE")
    step = _common_step(
        Decimal(str(spot_lot["stepSize"])),
        Decimal(str(futures_lot["stepSize"])),
    )
    minimum_quantity = max(
        Decimal(str(spot_lot["minQty"])),
        Decimal(str(futures_lot["minQty"])),
        _minimum_notional(spot_info) / spot_asks[0][0],
        _minimum_notional(futures_info) / futures_bids[0][0],
    )
    maximum_quantity = min(
        Decimal(str(spot_lot["maxQty"])), Decimal(str(futures_lot["maxQty"]))
    )

    funding_rows = [
        _mapping(value, name="funding row")
        for value in _list(funding_raw, name="funding history")
    ]
    for row in funding_rows:
        if row.get("symbol") != FUTURES_SYMBOL:
            raise ValueError("funding history contains another symbol")
    funding_rows.sort(key=lambda row: int(row["fundingTime"]))
    if len({int(row["fundingTime"]) for row in funding_rows}) != len(funding_rows):
        raise ValueError("funding history contains duplicate timestamps")
    retained_funding = funding_rows[-FUNDING_EVENTS:]
    adverse_funding_rate = sum(
        (
            max(-Decimal(str(row["fundingRate"])), Decimal(0))
            for row in retained_funding
        ),
        Decimal(0),
    )

    rows: list[dict[str, object]] = []
    for target in targets:
        quantity = _round_up(
            max(minimum_quantity, target / spot_asks[0][0]), step
        )
        spot_fill = _fill(spot_asks, quantity=quantity)
        futures_fill = _fill(futures_bids, quantity=quantity)
        capacity_ok = (
            quantity <= maximum_quantity
            and spot_fill is not None
            and futures_fill is not None
        )
        if not capacity_ok:
            rows.append(
                {
                    "target_spot_cost_usdt": _decimal_text(target),
                    "common_quantity": _decimal_text(quantity),
                    "top_level_depth_capacity_passes": False,
                    "failure_reason": "same_quantity_cannot_be_filled_in_both_retained_books",
                }
            )
            continue
        assert spot_fill is not None and futures_fill is not None
        spot_cost, spot_levels_used = spot_fill
        futures_proceeds, futures_levels_used = futures_fill
        gross = futures_proceeds - spot_cost
        fixed_stress = spot_cost * FIXED_STRESS_BPS / Decimal(10_000)
        adverse_funding = spot_cost * adverse_funding_rate
        after_stress = gross - fixed_stress - adverse_funding
        rows.append(
            {
                "target_spot_cost_usdt": _decimal_text(target),
                "common_quantity": _decimal_text(quantity),
                "common_quantity_step": _decimal_text(step),
                "top_level_depth_capacity_passes": True,
                "spot_ask_cost_usdt": _decimal_text(spot_cost),
                "spot_ask_vwap": _decimal_text(spot_cost / quantity),
                "spot_levels_used": spot_levels_used,
                "futures_bid_proceeds_usdt": _decimal_text(futures_proceeds),
                "futures_bid_vwap": _decimal_text(futures_proceeds / quantity),
                "futures_levels_used": futures_levels_used,
                "gross_entry_headroom_usdt": _decimal_text(gross),
                "gross_entry_headroom_bps": _decimal_text(
                    gross / spot_cost * Decimal(10_000)
                ),
                "fixed_stress_bps": _decimal_text(FIXED_STRESS_BPS),
                "fixed_stress_usdt": _decimal_text(fixed_stress),
                "adverse_funding_rate_sum": _decimal_text(adverse_funding_rate),
                "adverse_funding_stress_usdt": _decimal_text(adverse_funding),
                "after_all_frozen_stress_usdt": _decimal_text(after_stress),
                "after_all_frozen_stress_positive": after_stress > 0,
            }
        )

    premium = _mapping(premium_raw, name="premium index")
    if premium.get("symbol") != FUTURES_SYMBOL:
        raise ValueError("premium index identity changed")
    sources = [
        bstock_source,
        spot_exchange_source,
        futures_exchange_source,
        spot_depth_source,
        futures_depth_source,
        funding_source,
        premium_source,
    ]
    if len(sources) != 7:
        raise AssertionError("frozen request plan must contain exactly seven requests")
    started_ms = min(int(source["requested_at_ms"]) for source in sources)
    completed_ms = max(int(source["completed_at_ms"]) for source in sources)
    all_capacity = all(row["top_level_depth_capacity_passes"] is True for row in rows)
    all_positive = all(
        row.get("after_all_frozen_stress_positive") is True for row in rows
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "path": str(contract_path.as_posix()),
            "sha256": contract_hash,
        },
        "authority": {
            "public_unauthenticated_GET_requests": len(sources),
            "authenticated_requests": 0,
            "account_state_accessed": False,
            "orders_transfers_conversions_wallet_or_margin_actions": 0,
            "paper_or_live_trading_authority": False,
        },
        "capture": {
            "started_at_ms": started_ms,
            "completed_at_ms": completed_ms,
            "window_ms": completed_ms - started_ms,
            "request_count": len(sources),
            "all_status_codes_200": all(
                source["status_code"] == 200 for source in sources
            ),
            "raw_response_bytes": sum(
                int(source["response_bytes"]) for source in sources
            ),
            "journal_path": str(journal_path.as_posix()),
            "sources": sources,
        },
        "identity": {
            "bstock": bstock,
            "spot_symbol": SPOT_SYMBOL,
            "futures_symbol": FUTURES_SYMBOL,
            "futures_contract_type": futures_info["contractType"],
            "futures_underlying_type": futures_info.get("underlyingType"),
            "exact_multiplier": str(bstock["multiplier"]),
        },
        "funding": {
            "history_event_count": len(funding_rows),
            "retained_adverse_event_count": len(retained_funding),
            "eight_event_persistence_available": len(retained_funding) == FUNDING_EVENTS,
            "first_funding_time_ms": (
                None if not funding_rows else int(funding_rows[0]["fundingTime"])
            ),
            "last_funding_time_ms": (
                None if not funding_rows else int(funding_rows[-1]["fundingTime"])
            ),
            "adverse_negative_rate_sum": _decimal_text(adverse_funding_rate),
            "positive_funding_credited": False,
            "current_last_funding_rate_not_credited": str(premium["lastFundingRate"]),
            "next_funding_time_ms": premium["nextFundingTime"],
        },
        "economics": {
            "fixed_stress_bps": _decimal_text(FIXED_STRESS_BPS),
            "row_count": len(rows),
            "all_targets_capacity_valid": all_capacity,
            "after_all_frozen_stress_positive_count": sum(
                row.get("after_all_frozen_stress_positive") is True for row in rows
            ),
            "all_targets_after_stress_positive": all_positive,
            "rows": rows,
        },
        "adjudication": {
            "status": (
                "unaccepted_new_listing_candidate_survives_public_stress"
                if all_capacity and all_positive
                else "first_DJTB_DJT_listing_snapshot_fails_frozen_public_stress_or_capacity"
            ),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "trading_authority": False,
            "retry_trigger": (
                "exact_read_only_account_cost_and_eligibility_evidence_then_separately_frozen_multi_regime_recurrence"
                if all_capacity and all_positive
                else "material_fee_basis_funding_book_product_access_or_exact_account_conversion_change"
            ),
        },
        "limitations": [
            "the_two_entry_legs_are_non_atomic_and_displayed_depth_is_not_an_owned_fill",
            "the_perpetual_exit_basis_is_stressed_not_locked",
            "new_listing_funding_history_cannot_establish_cross_regime_persistence",
            "exact_account_fees_eligibility_stock_settlement_taxes_margin_and_liquidation_are_unbound",
            "no_reference_index_mark_last_trade_dividend_or_favorable_funding_is_credited",
        ],
        "implementation": {
            "path": "tools/screen_binance_djtb_new_listing_spot_perpetual.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.journal.exists():
        raise FileExistsError(f"refusing to append to prior journal: {args.journal}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prior result: {args.output}")
    if args.raw_dir.exists() and any(args.raw_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty raw directory: {args.raw_dir}")
    result = run(
        contract_path=args.contract,
        raw_dir=args.raw_dir,
        journal_path=args.journal,
    )
    write_bytes_atomic(args.output, (_canonical_json(result) + "\n").encode("ascii"))
    print(json.dumps(result["adjudication"], indent=2))
    print(json.dumps(result["funding"], indent=2))
    print(json.dumps(result["economics"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
