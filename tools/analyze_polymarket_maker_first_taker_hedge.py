#!/usr/bin/env python3
"""Diagnose historical Polymarket maker-first, taker-hedge sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

from simple_ai_trading.storage import write_json_atomic


DEFAULT_MARKET_PATTERN = r"^(btc|eth|sol)-updown-(5m|15m|1h)-"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-hedge-delay-seconds", type=int, default=60)
    parser.add_argument("--minimum-close-buffer-seconds", type=int, default=10)
    parser.add_argument("--taker-fee-rate", type=float, default=0.07)
    parser.add_argument("--market-pattern", default=DEFAULT_MARKET_PATTERN)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_as_dicts(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: list[object] | None = None,
) -> list[dict[str, Any]]:
    cursor = connection.execute(query, parameters or [])
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _one_row(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: list[object] | None = None,
) -> dict[str, Any]:
    cursor = connection.execute(query, parameters or [])
    names = [column[0] for column in cursor.description]
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("aggregate query returned no row")
    return dict(zip(names, row, strict=True))


def _candidate_sql() -> str:
    return """
        CREATE TEMP TABLE candidate_sequences AS
        WITH raw_deduplicated AS (
            SELECT DISTINCT *
            FROM read_parquet(?)
        ),
        scoped_fills AS (
            SELECT
                lower(maker) AS maker,
                lower(taker) AS taker,
                upper(taker_direction) AS taker_direction,
                condition_id,
                asset_id,
                outcome_seq,
                block_timestamp,
                epoch(close_at) AS close_epoch,
                market_slug,
                upper(regexp_extract(market_slug, '^([a-z]+)-', 1)) AS asset,
                regexp_extract(
                    market_slug,
                    '^(?:btc|eth|sol)-updown-(5m|15m|1h)-',
                    1
                ) AS interval,
                price,
                usdc_amount / price AS shares
            FROM raw_deduplicated
            WHERE regexp_matches(market_slug, ?)
              AND price > 0
              AND usdc_amount > 0
              AND maker IS NOT NULL
              AND taker IS NOT NULL
              AND lower(maker) <> lower(taker)
        ),
        participations AS (
            SELECT
                maker AS actor,
                'maker' AS role,
                CASE
                    WHEN taker_direction = 'SELL' THEN 'BUY'
                    ELSE 'SELL'
                END AS actor_direction,
                * EXCLUDE (maker, taker)
            FROM scoped_fills
            UNION ALL
            SELECT
                taker AS actor,
                'taker' AS role,
                taker_direction AS actor_direction,
                * EXCLUDE (maker, taker)
            FROM scoped_fills
        ),
        exactly_two AS (
            SELECT actor, condition_id
            FROM participations
            GROUP BY actor, condition_id
            HAVING count(*) = 2
        ),
        eligible AS (
            SELECT p.*
            FROM participations AS p
            INNER JOIN exactly_two AS e USING (actor, condition_id)
        )
        SELECT
            first.actor,
            first.condition_id,
            first.market_slug,
            first.asset,
            first.interval,
            first.outcome_seq AS maker_outcome_seq,
            hedge.outcome_seq AS hedge_outcome_seq,
            first.block_timestamp AS maker_fill_timestamp,
            hedge.block_timestamp AS hedge_fill_timestamp,
            hedge.block_timestamp - first.block_timestamp AS hedge_delay_seconds,
            CAST(floor((first.block_timestamp % 86400) / 3600) AS INTEGER)
                AS maker_fill_hour_utc,
            first.shares,
            first.price AS maker_price,
            hedge.price AS hedge_price,
            ? * hedge.price * (1 - hedge.price) AS hedge_fee_per_share,
            first.price + hedge.price AS gross_cost_per_share,
            first.price + hedge.price
                + (? * hedge.price * (1 - hedge.price))
                AS current_fee_sensitive_cost_per_share,
            first.shares * (
                1 - first.price - hedge.price
                - (? * hedge.price * (1 - hedge.price))
            ) AS current_fee_sensitive_pnl
        FROM eligible AS first
        INNER JOIN eligible AS hedge
            ON first.actor = hedge.actor
           AND first.condition_id = hedge.condition_id
           AND first.role = 'maker'
           AND first.actor_direction = 'BUY'
           AND hedge.role = 'taker'
           AND hedge.actor_direction = 'BUY'
           AND first.outcome_seq <> hedge.outcome_seq
           AND hedge.block_timestamp - first.block_timestamp BETWEEN 1 AND ?
           AND hedge.block_timestamp <= hedge.close_epoch - ?
           AND abs(first.shares - hedge.shares)
               <= greatest(
                   1e-9,
                   1e-9 * greatest(first.shares, hedge.shares)
               )
    """


def _summary_query(group_columns: str) -> str:
    select_prefix = f"{group_columns}," if group_columns else ""
    group_clause = f"GROUP BY {group_columns}" if group_columns else ""
    order_clause = f"ORDER BY {group_columns}" if group_columns else ""
    return f"""
        SELECT
            {select_prefix}
            count(*) AS sequence_count,
            count(DISTINCT actor) AS actor_count,
            count(DISTINCT condition_id) AS condition_count,
            count(*) FILTER (
                WHERE current_fee_sensitive_cost_per_share < 1
            ) AS current_fee_sensitive_positive_count,
            avg(
                (current_fee_sensitive_cost_per_share < 1)::INTEGER
            ) AS current_fee_sensitive_positive_fraction,
            min(current_fee_sensitive_cost_per_share)
                AS minimum_current_fee_sensitive_cost_per_share,
            median(current_fee_sensitive_cost_per_share)
                AS median_current_fee_sensitive_cost_per_share,
            max(current_fee_sensitive_cost_per_share)
                AS maximum_current_fee_sensitive_cost_per_share,
            min(current_fee_sensitive_pnl)
                AS worst_sequence_current_fee_sensitive_pnl,
            max(current_fee_sensitive_pnl)
                AS best_sequence_current_fee_sensitive_pnl,
            sum(current_fee_sensitive_pnl)
                AS aggregate_current_fee_sensitive_pnl,
            sum(shares) AS matched_shares,
            median(hedge_delay_seconds) AS median_hedge_delay_seconds
        FROM candidate_sequences
        {group_clause}
        {order_clause}
    """


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf") or value != value:
            raise ValueError("analysis produced a non-finite float")
        return round(value, 12)
    return value


def analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    input_path = arguments.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if arguments.maximum_hedge_delay_seconds < 1:
        raise ValueError("maximum hedge delay must be positive")
    if arguments.minimum_close_buffer_seconds < 0:
        raise ValueError("minimum close buffer cannot be negative")
    if not 0 <= arguments.taker_fee_rate <= 1:
        raise ValueError("taker fee rate must be between zero and one")

    connection = duckdb.connect(":memory:")
    try:
        source_scope = _one_row(
            connection,
            """
                SELECT
                    count(*) AS source_row_count,
                    count(DISTINCT condition_id) AS source_condition_count,
                    count(*) FILTER (WHERE regexp_matches(market_slug, ?))
                        AS scoped_row_count,
                    count(DISTINCT condition_id) FILTER (
                        WHERE regexp_matches(market_slug, ?)
                    ) AS scoped_condition_count
                FROM read_parquet(?)
            """,
            [
                arguments.market_pattern,
                arguments.market_pattern,
                str(input_path),
            ],
        )
        deduplication = _one_row(
            connection,
            """
                SELECT count(*) AS deduplicated_source_row_count
                FROM (SELECT DISTINCT * FROM read_parquet(?))
            """,
            [str(input_path)],
        )
        scoped_rows_by_asset = _rows_as_dicts(
            connection,
            """
                SELECT
                    upper(regexp_extract(market_slug, '^([a-z]+)-', 1))
                        AS asset,
                    count(*) AS row_count,
                    count(DISTINCT condition_id) AS condition_count
                FROM read_parquet(?)
                WHERE regexp_matches(market_slug, ?)
                GROUP BY asset
                ORDER BY asset
            """,
            [str(input_path), arguments.market_pattern],
        )
        connection.execute(
            _candidate_sql(),
            [
                str(input_path),
                arguments.market_pattern,
                arguments.taker_fee_rate,
                arguments.taker_fee_rate,
                arguments.taker_fee_rate,
                arguments.maximum_hedge_delay_seconds,
                arguments.minimum_close_buffer_seconds,
            ],
        )
        overall = _one_row(connection, _summary_query(""))
        by_asset = _rows_as_dicts(connection, _summary_query("asset"))
        by_asset_interval = _rows_as_dicts(
            connection,
            _summary_query("asset, interval"),
        )
        by_hour_utc = _rows_as_dicts(
            connection,
            _summary_query("maker_fill_hour_utc"),
        )
    finally:
        connection.close()

    source_scope.update(deduplication)
    source_scope["exact_duplicate_rows_removed"] = (
        source_scope["source_row_count"]
        - source_scope["deduplicated_source_row_count"]
    )
    source_scope["scoped_rows_by_asset"] = scoped_rows_by_asset
    candidate_assets = {row["asset"] for row in by_asset}
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "polymarket_maker_first_taker_hedge_historical_diagnostic",
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "dataset_layer": "TimeSeventeen/Polymarket-v1 daily_aligned",
            "dataset_partition": input_path.stem,
        },
        "frozen_contract": {
            "market_slug_regex": arguments.market_pattern,
            "maker_leg": "actor is maker buying one outcome",
            "hedge_leg": "same actor later is taker buying the opposite outcome",
            "actor_condition_participation_count": 2,
            "quantity_match": "absolute difference <= max(1e-9, 1e-9 * maximum quantity)",
            "minimum_hedge_delay_seconds": 1,
            "maximum_hedge_delay_seconds": (
                arguments.maximum_hedge_delay_seconds
            ),
            "minimum_close_buffer_seconds": (
                arguments.minimum_close_buffer_seconds
            ),
            "exact_cleaned_row_deduplication": True,
            "maker_fee_per_share": 0,
            "current_crypto_taker_fee_sensitivity": (
                "shares * rate * price * (1 - price)"
            ),
            "taker_fee_rate": arguments.taker_fee_rate,
            "dataset_fee_usdc_used": False,
        },
        "source_scope": source_scope,
        "results": {
            "overall": overall,
            "by_asset": by_asset,
            "zero_sequence_assets": sorted(
                {"BTC", "ETH", "SOL"} - candidate_assets
            ),
            "by_asset_interval": by_asset_interval,
            "by_maker_fill_hour_utc": by_hour_utc,
        },
        "evidence_limits": [
            "one UTC daily partition only",
            "historical v1 fills do not establish current v2 execution",
            "cleaned layer omits on-chain event identifiers",
            "exact-row deduplication can collapse indistinguishable real fills",
            "no order placements cancellations quote updates queue positions or books",
            "pair attribution is a conservative wallet-condition heuristic",
            "current fee formula is a sensitivity and not the historical fee field",
            "positive delayed sequences can be ex-post favorable price movement rather than lockable arbitrage",
            "no authenticated ownership inventory merge redemption rebate or complete costs",
            "no bull bear sideways choppy volatility liquidity or latency acceptance slices",
        ],
        "verdict": {
            "accepted_edge": False,
            "market_direction_independent_edge_proved": False,
            "public_after_cost_profit_floor": 0,
            "reason": (
                "fills can reveal historical sequence recurrence, but absent creation-time "
                "opposite executable quotes and queue state cannot prove a causal lock; "
                "delayed positive outcomes remain exposed to intervening price direction"
            ),
        },
    }
    safe_result = _json_safe(result)
    canonical = json.dumps(
        safe_result,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    safe_result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    return safe_result


def main() -> int:
    arguments = _parser().parse_args()
    result = analyze(arguments)
    output_path = arguments.output.resolve()
    write_json_atomic(output_path, result, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
