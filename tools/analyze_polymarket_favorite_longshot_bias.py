#!/usr/bin/env python3
"""Evaluate a frozen Polymarket favorite-longshot bias preflight."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any, Callable

import duckdb

from simple_ai_trading.storage import write_json_atomic


SECONDS_PER_YEAR = 31_557_600
PRICE_BANDS = (
    ("favorite_90_to_below_95", 0.90, 0.95, False),
    ("favorite_95_to_99", 0.95, 0.99, True),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--pwi-history", type=Path, required=True)
    parser.add_argument("--price-gap-history", type=Path, required=True)
    parser.add_argument("--calibration-snapshot", type=Path, required=True)
    parser.add_argument("--methodology", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(payload.pop(hash_field)).lower()
    actual = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    if claimed != actual:
        raise ValueError(f"{path.name} {hash_field} mismatch: {claimed} != {actual}")
    payload[hash_field] = claimed
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: object, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _role_slices(rows: list[Any]) -> dict[str, list[Any]]:
    train_end = math.floor(len(rows) * 0.60)
    validation_end = math.floor(len(rows) * 0.80)
    return {
        "training": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end:],
    }


def _fraction(rows: list[float], predicate: Callable[[float], bool]) -> float:
    return sum(1 for value in rows if predicate(value)) / len(rows) if rows else 0.0


def _series_role_metrics(
    rows: list[dict[str, object]],
    *,
    value_key: str,
    median_limit: float,
    fraction_limit: float,
) -> dict[str, object]:
    values = [float(row[value_key]) for row in rows]
    if not values:
        return {
            "row_count": 0,
            "date_start": None,
            "date_end": None,
            "median": None,
            "fraction_below_threshold": None,
            "passes": False,
        }
    value_median = median(values)
    fraction_below = _fraction(values, lambda value: value < (1.0 if value_key == "pwi_alpha" else 0.0))
    return {
        "row_count": len(rows),
        "date_start": str(rows[0]["date"]),
        "date_end": str(rows[-1]["date"]),
        "median": round(value_median, 12),
        "fraction_below_threshold": round(fraction_below, 12),
        "passes": value_median < median_limit and fraction_below >= fraction_limit,
    }


def _external_persistence(
    pwi_path: Path,
    price_gap_path: Path,
    calibration_path: Path,
) -> dict[str, object]:
    raw_pwi = _read_csv(pwi_path)
    excluded_below_floor = sum(
        1
        for row in raw_pwi
        if row.get("pwi_alpha", "")
        and _finite_float(row["n_trades_nonbot"], field="n_trades_nonbot") < 1000
    )
    pwi = sorted(
        (
            {
                "date": row["date"],
                "pwi_alpha": _finite_float(row["pwi_alpha"], field="pwi_alpha"),
            }
            for row in raw_pwi
            if row.get("pwi_alpha", "")
            and _finite_float(row["n_trades_nonbot"], field="n_trades_nonbot") >= 1000
        ),
        key=lambda row: str(row["date"]),
    )
    price_gap = sorted(
        (
            {
                "date": row["date"],
                "longshot_gap": _finite_float(row["longshot_gap"], field="longshot_gap"),
            }
            for row in _read_csv(price_gap_path)
            if row.get("longshot_gap", "")
        ),
        key=lambda row: str(row["date"]),
    )
    pwi_roles = {
        role: _series_role_metrics(
            rows,
            value_key="pwi_alpha",
            median_limit=0.90,
            fraction_limit=0.65,
        )
        for role, rows in _role_slices(pwi).items()
    }
    gap_roles = {
        role: _series_role_metrics(
            rows,
            value_key="longshot_gap",
            median_limit=0.0,
            fraction_limit=0.60,
        )
        for role, rows in _role_slices(price_gap).items()
    }
    calibration = []
    for row in _read_csv(calibration_path):
        center = _finite_float(row["price_bin_center"], field="price_bin_center")
        if center >= 0.90:
            calibration.append(
                {
                    "price_bin_center": center,
                    "realized_win_rate": _finite_float(
                        row["realized_win_rate"], field="realized_win_rate"
                    ),
                    "n_trades": int(float(row["n_trades"])),
                    "calibration_gap": _finite_float(
                        row["calibration_gap"], field="calibration_gap"
                    ),
                }
            )
    passes = (
        len(pwi) >= 104
        and len(price_gap) >= 104
        and all(bool(metrics["passes"]) for metrics in pwi_roles.values())
        and all(bool(metrics["passes"]) for metrics in gap_roles.values())
    )
    return {
        "pwi_rows_in_source": len(raw_pwi),
        "pwi_rows_excluded_below_current_1000_trade_floor": excluded_below_floor,
        "eligible_pwi_week_count": len(pwi),
        "eligible_longshot_gap_week_count": len(price_gap),
        "pwi_roles": pwi_roles,
        "longshot_gap_roles": gap_roles,
        "pooled_high_price_calibration_bins": calibration,
        "passes_persistence_lead_gate": passes,
        "execution_or_profit_proved": False,
    }


def _query_entries(parquet_path: Path, market_pattern: str) -> list[dict[str, object]]:
    query = """
        WITH raw_deduplicated AS (
            SELECT DISTINCT *
            FROM read_parquet(?)
        ),
        eligible AS (
            SELECT
                condition_id,
                asset_id,
                market_slug,
                upper(regexp_extract(market_slug, '^([a-z]+)-', 1)) AS asset,
                block_timestamp,
                epoch(close_at) AS close_epoch,
                epoch(resolved_at) AS resolved_epoch,
                price,
                usdc_amount / price AS available_fill_shares,
                outcome_label,
                winning_outcome_label,
                CASE
                    WHEN lower(outcome_label) = lower(winning_outcome_label) THEN 1
                    ELSE 0
                END AS won,
                CASE
                    WHEN price >= 0.90 AND price < 0.95
                        THEN 'favorite_90_to_below_95'
                    WHEN price >= 0.95 AND price <= 0.99
                        THEN 'favorite_95_to_99'
                END AS price_band
            FROM raw_deduplicated
            WHERE regexp_matches(market_slug, ?)
              AND lower(resolution_status) = 'resolved'
              AND upper(taker_direction) = 'BUY'
              AND price >= 0.90
              AND price <= 0.99
              AND price > 0
              AND usdc_amount / price >= 5
              AND close_at IS NOT NULL
              AND resolved_at IS NOT NULL
              AND outcome_label IS NOT NULL
              AND winning_outcome_label IS NOT NULL
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY condition_id, price_band
                    ORDER BY block_timestamp, asset_id
                ) AS candidate_rank
            FROM eligible
        )
        SELECT * EXCLUDE (candidate_rank)
        FROM ranked
        WHERE candidate_rank = 1
        ORDER BY price_band, close_epoch, condition_id
    """
    connection = duckdb.connect(":memory:")
    try:
        cursor = connection.execute(query, [str(parquet_path), market_pattern])
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()


def _entry_economics(entry: dict[str, object]) -> dict[str, object]:
    price = _finite_float(entry["price"], field="price")
    block_timestamp = int(entry["block_timestamp"])
    resolved_epoch = int(_finite_float(entry["resolved_epoch"], field="resolved_epoch"))
    close_epoch = int(_finite_float(entry["close_epoch"], field="close_epoch"))
    holding_seconds = max(0, resolved_epoch - block_timestamp)
    fee_per_share = 0.07 * price * (1.0 - price)
    capital_cost_per_share = price * 0.10 * holding_seconds / SECONDS_PER_YEAR
    net_per_share = (
        int(entry["won"])
        - price
        - fee_per_share
        - 0.001
        - capital_cost_per_share
    )
    return {
        **entry,
        "holding_seconds": holding_seconds,
        "seconds_to_close": close_epoch - block_timestamp,
        "current_taker_fee_per_share_pUSD": fee_per_share,
        "capital_hurdle_per_share_pUSD": capital_cost_per_share,
        "net_pnl_five_shares_pUSD": 5.0 * net_per_share,
    }


def _bootstrap_lower_bound(
    values: list[float], *, seed: int, repetitions: int = 2000, block_length: int = 4
) -> float | None:
    if not values:
        return None
    generator = random.Random(seed)
    sample_means: list[float] = []
    block = min(block_length, len(values))
    blocks_needed = math.ceil(len(values) / block)
    for _ in range(repetitions):
        sample: list[float] = []
        for _ in range(blocks_needed):
            start = generator.randrange(len(values))
            sample.extend(values[(start + offset) % len(values)] for offset in range(block))
        sample_means.append(sum(sample[: len(values)]) / len(values))
    sample_means.sort()
    index = math.floor(0.025 * (len(sample_means) - 1))
    return sample_means[index]


def _entry_group_metrics(
    rows: list[dict[str, object]], *, seed: int, require_bootstrap: bool
) -> dict[str, object]:
    values = [float(row["net_pnl_five_shares_pUSD"]) for row in rows]
    lower = (
        _bootstrap_lower_bound(values, seed=seed)
        if require_bootstrap and values
        else None
    )
    mean_value = sum(values) / len(values) if values else None
    return {
        "condition_count": len(rows),
        "mean_net_pnl_five_shares_pUSD": (
            round(mean_value, 12) if mean_value is not None else None
        ),
        "aggregate_net_pnl_five_shares_pUSD": round(sum(values), 12),
        "win_fraction": (
            round(sum(int(row["won"]) for row in rows) / len(rows), 12)
            if rows
            else None
        ),
        "mean_entry_price": (
            round(sum(float(row["price"]) for row in rows) / len(rows), 12)
            if rows
            else None
        ),
        "family_adjusted_bootstrap_mean_lower_bound_pUSD": (
            round(lower, 12) if lower is not None else None
        ),
        "positive_mean": bool(mean_value is not None and mean_value > 0),
        "positive_bootstrap_lower_bound": bool(lower is not None and lower > 0),
    }


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"20260827:{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _local_execution_translation(
    parquet_path: Path, market_pattern: str
) -> dict[str, object]:
    entries = [_entry_economics(row) for row in _query_entries(parquet_path, market_pattern)]
    band_results: list[dict[str, object]] = []
    for band_id, lower, upper, upper_inclusive in PRICE_BANDS:
        band_rows = [row for row in entries if row["price_band"] == band_id]
        role_rows = _role_slices(band_rows)
        roles = {
            role: _entry_group_metrics(
                rows,
                seed=_stable_seed(f"{band_id}:{role}"),
                require_bootstrap=True,
            )
            for role, rows in role_rows.items()
        }
        slices: dict[str, dict[str, object]] = {}
        slice_definitions: dict[str, Callable[[dict[str, object]], bool]] = {
            "BTC": lambda row: row["asset"] == "BTC",
            "ETH": lambda row: row["asset"] == "ETH",
            "SOL": lambda row: row["asset"] == "SOL",
            "winning_outcome_Up": lambda row: str(row["winning_outcome_label"]).lower() == "up",
            "winning_outcome_Down": lambda row: str(row["winning_outcome_label"]).lower() == "down",
            "entry_at_least_60_seconds_before_close": lambda row: int(row["seconds_to_close"]) >= 60,
            "entry_below_60_seconds_before_close": lambda row: int(row["seconds_to_close"]) < 60,
        }
        supported_slice_passes = []
        for name, predicate in slice_definitions.items():
            selected = [row for row in band_rows if predicate(row)]
            metrics = _entry_group_metrics(
                selected,
                seed=_stable_seed(f"{band_id}:slice:{name}"),
                require_bootstrap=False,
            )
            supported = len(selected) >= 5
            metrics["supported"] = supported
            metrics["passes_when_supported"] = bool(
                not supported or metrics["positive_mean"]
            )
            if supported:
                supported_slice_passes.append(bool(metrics["positive_mean"]))
            slices[name] = metrics
        enough_conditions = len(band_rows) >= 30
        roles_pass = all(
            int(metrics["condition_count"]) >= 6
            and bool(metrics["positive_mean"])
            and bool(metrics["positive_bootstrap_lower_bound"])
            for metrics in roles.values()
        )
        band_results.append(
            {
                "price_band": band_id,
                "minimum_inclusive": lower,
                "maximum": upper,
                "maximum_inclusive": upper_inclusive,
                "overall": _entry_group_metrics(
                    band_rows,
                    seed=_stable_seed(f"{band_id}:overall"),
                    require_bootstrap=True,
                ),
                "roles": roles,
                "required_slices": slices,
                "passes_local_economics_gate": (
                    enough_conditions
                    and roles_pass
                    and all(supported_slice_passes)
                ),
            }
        )
    return {
        "eligible_first_fill_count": len(entries),
        "distinct_condition_count": len({str(row["condition_id"]) for row in entries}),
        "distinct_partition_dates": 1,
        "bands": band_results,
        "passes_local_economics_gate": all(
            bool(row["passes_local_economics_gate"]) for row in band_results
        ),
        "passes_minimum_12_date_promotion_gate": False,
    }


def analyze(arguments: argparse.Namespace) -> dict[str, object]:
    repository_root = Path.cwd().resolve()
    contract_path = arguments.contract.resolve()
    amendment_path = arguments.amendment.resolve()
    contract = _load_hashed_json(contract_path, "contract_sha256")
    amendment = _load_hashed_json(amendment_path, "contract_sha256")
    parquet_path = arguments.parquet.resolve()
    if _sha256(parquet_path) != contract["retained_local_action_value_source"]["sha256"]:
        raise ValueError("retained parquet hash does not match the frozen contract")

    source_paths = {
        "methodology": arguments.methodology.resolve(),
        "pwi_history": arguments.pwi_history.resolve(),
        "price_gap_history": arguments.price_gap_history.resolve(),
        "calibration_snapshot": arguments.calibration_snapshot.resolve(),
    }
    external = _external_persistence(
        source_paths["pwi_history"],
        source_paths["price_gap_history"],
        source_paths["calibration_snapshot"],
    )
    local = _local_execution_translation(
        parquet_path,
        str(contract["local_execution_translation"]["market_slug_regex"]),
    )
    materially_reopened = bool(
        external["passes_persistence_lead_gate"]
        and local["passes_local_economics_gate"]
    )
    if not external["passes_persistence_lead_gate"]:
        disposition = "external_behavioral_persistence_gate_failed"
    elif not local["passes_local_economics_gate"]:
        disposition = "persistent_behavioral_fact_did_not_translate_to_local_after_cost_execution"
    else:
        disposition = "materially_reopened_only_for_a_separate_frozen_twelve_date_preflight"

    result: dict[str, object] = {
        "schema_version": "polymarket-favorite-longshot-bias-preflight-v1-2026-08-27",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": _portable_path(contract_path, repository_root),
            "contract_sha256": contract["contract_sha256"],
            "amendment_path": _portable_path(amendment_path, repository_root),
            "amendment_sha256": amendment["contract_sha256"],
        },
        "implementation": {
            "path": _portable_path(Path(__file__).resolve(), repository_root),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "raw_sources": {
            name: {
                "path": _portable_path(path, repository_root),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "retained_parquet": {
            "path": parquet_path.as_posix(),
            "bytes": parquet_path.stat().st_size,
            "sha256": _sha256(parquet_path),
        },
        "external_persistence": external,
        "local_execution_translation": local,
        "verdict": {
            "disposition": disposition,
            "materially_reopened_for_twelve_date_preflight": materially_reopened,
            "accepted_edge": False,
            "profitability_claim": False,
            "paper_or_live_authority": False,
            "why_not_promoted": [
                "the local execution translation contains only one UTC date versus the frozen twelve-date minimum",
                "the local source is v1 historical fills and cannot prove current v2 order acceptance depth latency or settlement",
                "the all-category aggregate calibration series is persistence evidence rather than BTC ETH SOL execution evidence",
            ],
        },
    }
    result["result_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("ascii")
    ).hexdigest()
    return result


def main() -> int:
    arguments = _parser().parse_args()
    result = analyze(arguments)
    write_json_atomic(arguments.output, result, sort_keys=False)
    print(
        _canonical_json(
            {
                "accepted_edge": result["verdict"]["accepted_edge"],
                "disposition": result["verdict"]["disposition"],
                "materially_reopened": result["verdict"][
                    "materially_reopened_for_twelve_date_preflight"
                ],
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
