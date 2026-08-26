"""Analyze a finished public oracle-to-CLOB close monitor."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.storage import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data/polymarket-post-observation-monitor-v2.duckdb"
DEFAULT_REPORT = ROOT / "data/polymarket-post-observation-monitor-v2-report.json"
DEFAULT_OUTPUT = (
    ROOT
    / "docs/model-research/action-value/"
    "polymarket-post-observation-prospective-v2-2026-08-26.json"
)
HIGH_BID_PRICES = {Decimal("0.99"), Decimal("0.999")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _price(value: object) -> Decimal:
    return Decimal(str(value))


def _analyze(database: Path, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_bytes())
    if report.get("status") != "complete" or report.get("stream_gap_count") != 0:
        raise ValueError("capture is not complete and gap-free")
    report_body = dict(report)
    claimed_report_hash = str(report_body.pop("report_sha256", ""))
    if _canonical_sha256(report_body) != claimed_report_hash:
        raise ValueError("capture report self-hash differs")

    with PolymarketEvidenceStore(database, read_only=True) as store:
        connection = store.connect()
        run_rows = connection.execute(
            """
            SELECT run_id, status, report_json, report_sha256
            FROM polymarket_recorder_run ORDER BY started_at_ms DESC
            """
        ).fetchall()
        if (
            len(run_rows) != 1
            or str(run_rows[0][0]) != str(report.get("run_id"))
            or str(run_rows[0][1]) != "complete"
            or json.loads(str(run_rows[0][2])) != report
            or str(run_rows[0][3]) != claimed_report_hash
        ):
            raise ValueError("capture run lineage differs from report")
        run_id = str(run_rows[0][0])
        market_rows = connection.execute(
            """
            SELECT asset, condition_id, slug, event_start_ms, end_ms,
                   up_token_id, down_token_id
            FROM polymarket_market_snapshot
            WHERE run_id = ?
            QUALIFY row_number() OVER (
                PARTITION BY condition_id ORDER BY observed_wall_ms
            ) = 1
            ORDER BY end_ms, asset
            """,
            [run_id],
        ).fetchall()
        event_rows = [
            (
                event.stream,
                event.event_type,
                event.symbol,
                event.condition_id,
                event.asset_id,
                event.source_time_ms,
                event.received_wall_ms,
                dict(event.event),
            )
            for event in store.iter_public_events(
                run_id,
                streams=("clob_market", "polymarket_rtds"),
                ordered=True,
            )
            if event.event_type
            in {
                "book",
                "price_change",
                "last_trade_price",
                "crypto_prices_twap_sixty:update",
            }
        ]

    oracle: dict[tuple[str, int], tuple[Decimal, int]] = {}
    clob_by_condition: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for row in event_rows:
        stream, event_type, symbol, condition_id = map(str, row[:4])
        source_time_ms = row[5]
        if (
            stream == "polymarket_rtds"
            and event_type == "crypto_prices_twap_sixty:update"
            and source_time_ms is not None
        ):
            payload = row[7].get("payload")
            if isinstance(payload, dict) and payload.get("value") is not None:
                oracle[(symbol, int(source_time_ms))] = (
                    _price(payload["value"]),
                    int(row[6]),
                )
        elif stream == "clob_market" and condition_id:
            clob_by_condition[condition_id.lower()].append(row)

    results: list[dict[str, Any]] = []
    for market in market_rows:
        asset, condition, slug = map(str, market[:3])
        start_ms, end_ms = int(market[3]), int(market[4])
        opening = oracle.get((asset, start_ms))
        closing = oracle.get((asset, end_ms))
        if opening is None or closing is None:
            continue
        outcome = "Up" if closing[0] >= opening[0] else "Down"
        winning_token = str(market[5] if outcome == "Up" else market[6])
        close_receipt = closing[1]
        high_bid_sizes: dict[tuple[str, Decimal], Decimal] = {}
        first_growth_ms: int | None = None
        first_fill_ms: int | None = None
        observed_gross = Decimal("0")
        fill_count = 0

        for row in clob_by_condition.get(condition.lower(), []):
            event_type = str(row[1])
            source_ms = int(row[5]) if row[5] is not None else None
            received_ms = int(row[6])
            if received_ms > end_ms + 120_000:
                continue
            event = row[7]
            if event_type == "book" and str(row[4]) == winning_token:
                for bid in event.get("bids", []):
                    if not isinstance(bid, dict):
                        continue
                    price = _price(bid.get("price"))
                    if price in HIGH_BID_PRICES:
                        high_bid_sizes[(winning_token, price)] = _price(
                            bid.get("size")
                        )
            elif event_type == "price_change":
                for change in event.get("price_changes", []):
                    if (
                        not isinstance(change, dict)
                        or str(change.get("asset_id")) != winning_token
                        or str(change.get("side")) != "BUY"
                    ):
                        continue
                    price = _price(change.get("price"))
                    if price not in HIGH_BID_PRICES:
                        continue
                    size = _price(change.get("size"))
                    key = (winning_token, price)
                    previous = high_bid_sizes.get(key, Decimal("0"))
                    high_bid_sizes[key] = size
                    if (
                        received_ms >= close_receipt
                        and size > previous
                        and first_growth_ms is None
                    ):
                        first_growth_ms = received_ms
            elif (
                event_type == "last_trade_price"
                and str(row[4]) == winning_token
                and str(event.get("side")) == "SELL"
                and source_ms is not None
                and source_ms >= close_receipt
            ):
                price = _price(event.get("price"))
                size = _price(event.get("size"))
                fill_count += 1
                observed_gross += size * (Decimal("1") - price)
                if first_fill_ms is None:
                    first_fill_ms = received_ms

        results.append(
            {
                "asset": asset,
                "slug": slug,
                "outcome": outcome,
                "opening_twap": str(opening[0]),
                "closing_twap": str(closing[0]),
                "oracle_receipt_delay_ms": close_receipt - end_ms,
                "post_close_observation_ms": int(report["ended_at_ms"]) - end_ms,
                "first_winner_bid_growth_delay_ms": (
                    None if first_growth_ms is None else first_growth_ms - end_ms
                ),
                "first_later_winner_sell_fill_delay_ms": (
                    None if first_fill_ms is None else first_fill_ms - end_ms
                ),
                "later_winner_sell_fill_count": fill_count,
                "observed_gross_pusd": str(observed_gross),
            }
        )

    complete = len(results)
    growth_count = sum(
        row["first_winner_bid_growth_delay_ms"] is not None for row in results
    )
    fill_count = sum(
        row["first_later_winner_sell_fill_delay_ms"] is not None for row in results
    )
    gross = sum((_price(row["observed_gross_pusd"]) for row in results), Decimal())
    created_at = datetime.fromtimestamp(
        int(report["ended_at_ms"]) / 1_000,
        tz=timezone.utc,
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": "polymarket-post-observation-prospective-v2",
        "created_at_utc": created_at,
        "authority": {
            "credentials_used": False,
            "funds_used": False,
            "orders_submitted": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "accepted_edge": False,
            "profitability_claim": False,
        },
        "source": {
            "database_path": database.relative_to(ROOT).as_posix(),
            "database_sha256": _sha256(database),
            "report_path": report_path.relative_to(ROOT).as_posix(),
            "report_sha256": _sha256(report_path),
            "run_id": run_id,
            "stream_gap_count": report["stream_gap_count"],
        },
        "economic_summary": {
            "complete_conditions": complete,
            "conditions_with_post_observation_winner_bid_growth": growth_count,
            "conditions_with_later_winner_sell_fills": fill_count,
            "public_observed_gross_pusd": str(gross),
        },
        "evidence_rows": results,
        "limitations": [
            "Only one fully observed five-minute interval was available for each asset, and all three resolved Up; this is neither a regime-persistence nor direction-balance result.",
            "Winning-side bid growth recurred in all three conditions, but qualifying later seller-aggressed winner fills appeared only for BTC.",
            "The capture ended before a full 120-second post-close window elapsed, so the absence of later ETH and SOL fills is bounded by the retained observation window.",
            "Public events cannot prove a fresh account order was accepted after the oracle receipt or establish owned fill lineage, queue position, achievable size, fees, or after-cost profit.",
        ],
        "verdict": {
            "accepted_edge": False,
            "deployment_ready": False,
            "cross_asset_public_recurrence": (
                complete > 0
                and growth_count == complete
                and fill_count == complete
                and {str(row["asset"]) for row in results} == {"BTC", "ETH", "SOL"}
            ),
            "reason": (
                "Public recurrence cannot prove fresh account order acceptance, "
                "owned fills, queue position, achievable size, or after-cost profit."
            ),
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = _analyze(args.database.resolve(), args.report.resolve())
    write_json_atomic(args.output.resolve(), result, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
