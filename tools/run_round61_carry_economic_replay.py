"""Run the frozen Round 61 matched spot-perpetual carry replay."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
import sqlite3
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.derivatives_archive import (  # noqa: E402
    _canonical_row_digest_update,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.ingest_round61_carry_event_sources import (  # noqa: E402
    DESIGN_SHA256,
    FUTURES_SOURCE,
    MANIFEST_FILE_SHA256,
    MANIFEST_SHA256,
    MARK_SOURCE,
    SPOT_SOURCE,
    _git,
    _validate_design,
    _validate_manifest,
)
from tools.run_round59_funding_persistence_feasibility import (  # noqa: E402
    SYMBOLS,
    _canonical_json,
    _canonical_sha256,
    _connect_read_only,
    _file_sha256,
    _load_verified_funding,
    _read_object,
    _stationary_bootstrap_means,
    _validate_finite,
)
from tools.run_round60_full_history_funding_replication import (  # noqa: E402
    _validate_certificate as _validate_round60_certificate,
    _validate_design as _validate_round60_design,
)


ROUND = 61
REPORT_SCHEMA = "round-061-carry-economic-replay-report-v1"
SOURCE_CERTIFICATE_SCHEMA = "round-061-carry-event-source-certificate-v1"
SOURCE_CERTIFICATE_FILE_SHA256 = (
    "a419767b6e04ab6b97aa26f9b526a3c5ac80e37303237af3cd39348f78ca912a"
)
SOURCE_CERTIFICATE_CANONICAL_SHA256 = (
    "579c27e3575b46f07231bf510787195048eeedf2b4ca00413b55271ad14a2d30"
)
SOURCE_CERTIFICATE_IMPLEMENTATION_COMMIT = "3f7d1f4dfe9de4c7af4236720a5cf671963bfc37"
FUNDING_CERTIFICATE_FILE_SHA256 = (
    "45262aea2ca244c9b1323370c220cbf7ddc8c2e4956f33eac4278ed5b2d6b373"
)
FUNDING_CERTIFICATE_CANONICAL_SHA256 = (
    "e3fe53ba87d728eed85efdaa93350b3c76ab7adcb7216cac690c5029818a9736"
)
MINUTE_MS = 60_000


def _minute(timestamp_ms: int) -> int:
    return int(timestamp_ms) // MINUTE_MS * MINUTE_MS


def _validate_source_certificate(
    path: Path,
    *,
    design: Mapping[str, object],
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], str, str]:
    file_sha = _file_sha256(path)
    certificate = _read_object(path, "Round 61 source certificate")
    canonical = dict(certificate)
    claimed = str(canonical.pop("source_certificate_sha256", ""))
    archives = certificate.get("filtered_archive_evidence", [])
    futures = certificate.get("existing_futures_archive_evidence", [])
    series = certificate.get("series_evidence", [])
    if (
        file_sha != SOURCE_CERTIFICATE_FILE_SHA256
        or claimed != SOURCE_CERTIFICATE_CANONICAL_SHA256
        or claimed != _canonical_sha256(canonical)
        or certificate.get("schema_version") != SOURCE_CERTIFICATE_SCHEMA
        or certificate.get("round") != ROUND
        or certificate.get("design_sha256") != design["design_sha256"]
        or certificate.get("manifest_sha256") != manifest["manifest_sha256"]
        or certificate.get("implementation_commit")
        != SOURCE_CERTIFICATE_IMPLEMENTATION_COMMIT
        or certificate.get("complete") is not True
        or certificate.get("persistent_zip_archive_created") is not False
        or tuple(certificate.get("symbols", ())) != SYMBOLS
        or not isinstance(archives, list)
        or not isinstance(futures, list)
        or not isinstance(series, list)
        or len(archives) != 190
        or len(futures) != 95
        or len(series) != 9
        or any(
            row.get("checksum_status") != "verified"
            or row.get("archive_sha256") != row.get("checksum_sha256")
            or int(row.get("missing_required_rows", -1))
            != len(row.get("missing_required_open_times_ms", ()))
            for row in archives
        )
        or any(
            row.get("checksum_status") != "verified"
            or row.get("sha256") != row.get("checksum_sha256")
            for row in futures
        )
    ):
        raise ValueError("Round 61 source certificate identity drifted")
    expected_series = {
        (symbol, kind)
        for symbol in SYMBOLS
        for kind in ("spot_execution", "futures_execution", "funding_mark_price")
    }
    if {(row.get("symbol"), row.get("kind")) for row in series} != expected_series:
        raise ValueError("Round 61 source certificate series drifted")
    _validate_finite(certificate, "source certificate")
    return certificate, file_sha, claimed


def _validate_funding_certificate(
    path: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    design_path = (
        ROOT
        / "docs/model-research/action-value/round-060-full-history-funding-replication-design.json"
    )
    design, _ = _validate_round60_design(design_path)
    certificate, file_sha, canonical_sha = _validate_round60_certificate(path, design)
    if (
        file_sha != FUNDING_CERTIFICATE_FILE_SHA256
        or canonical_sha != FUNDING_CERTIFICATE_CANONICAL_SHA256
    ):
        raise ValueError("Round 60 funding source certificate drifted")
    return design, certificate, file_sha, canonical_sha


def _series_contracts(
    certificate: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    output: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in certificate["series_evidence"]:
        key = (str(row["symbol"]), str(row["kind"]))
        if key in output:
            raise ValueError("Round 61 source certificate contains duplicate series")
        output[key] = row
    return output


def _time_chunks(values: Sequence[int]) -> list[list[int]]:
    return [list(values[start : start + 400]) for start in range(0, len(values), 400)]


def _load_execution_rows(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    market_type: str,
    source: str,
    required_times: Sequence[int],
) -> tuple[dict[int, dict[str, object]], str]:
    output: dict[int, dict[str, object]] = {}
    digest = hashlib.sha256()
    for times in _time_chunks(required_times):
        placeholders = ",".join("?" for _ in times)
        rows = connection.execute(
            f"""
            SELECT open_time, open, high, low, close, volume, close_time,
                   quote_volume, trade_count, taker_buy_base_volume,
                   taker_buy_quote_volume
            FROM candles
            WHERE symbol=? AND market_type=? AND interval='1m' AND source=?
              AND open_time IN ({placeholders})
            ORDER BY open_time
            """,
            (symbol, market_type, source, *times),
        ).fetchall()
        for raw_row in rows:
            row = dict(raw_row)
            open_time = int(row["open_time"])
            if open_time in output:
                raise ValueError(f"{symbol} {market_type} execution row is duplicated")
            _canonical_row_digest_update(
                digest,
                (
                    open_time,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    int(row["close_time"]),
                    float(row["quote_volume"]),
                    int(row["trade_count"]),
                    float(row["taker_buy_base_volume"]),
                    float(row["taker_buy_quote_volume"]),
                ),
            )
            output[open_time] = row
    return output, digest.hexdigest()


def _load_mark_rows(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    required_times: Sequence[int],
) -> tuple[dict[int, dict[str, object]], str]:
    output: dict[int, dict[str, object]] = {}
    digest = hashlib.sha256()
    for times in _time_chunks(required_times):
        placeholders = ",".join("?" for _ in times)
        rows = connection.execute(
            f"""
            SELECT open_time, open, high, low, close, close_time
            FROM futures_reference_bars
            WHERE symbol=? AND market_type='futures' AND kind='mark_price'
              AND interval='1m' AND source=? AND open_time IN ({placeholders})
            ORDER BY open_time
            """,
            (symbol, MARK_SOURCE, *times),
        ).fetchall()
        for raw_row in rows:
            row = dict(raw_row)
            open_time = int(row["open_time"])
            if open_time in output:
                raise ValueError(f"{symbol} mark-price row is duplicated")
            _canonical_row_digest_update(
                digest,
                (
                    open_time,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row["close_time"]),
                ),
            )
            output[open_time] = row
    return output, digest.hexdigest()


def _reconcile_series(
    rows: Mapping[int, Mapping[str, object]],
    digest: str,
    *,
    required_times: Sequence[int],
    contract: Mapping[str, object],
) -> None:
    required = set(int(value) for value in required_times)
    missing = required - set(rows)
    if (
        int(contract["required_rows"]) != len(required)
        or int(contract["stored_rows"]) != len(rows)
        or int(contract.get("missing_required_rows", len(missing))) != len(missing)
        or contract["selected_row_stream_sha256"] != digest
    ):
        raise ValueError(
            f"{contract['symbol']} {contract['kind']} database rows drifted"
        )


def _validate_manifest_funding(
    episodes: Sequence[Mapping[str, object]],
    funding_rows: Sequence[sqlite3.Row],
) -> dict[int, dict[str, object]]:
    times = [int(row["calc_time"]) for row in funding_rows]
    by_time = {int(row["calc_time"]): dict(row) for row in funding_rows}
    for episode in episodes:
        decision = int(episode["decision_time_ms"])
        end = int(episode["end_time_ms"])
        current = by_time.get(decision)
        if current is None or not math.isclose(
            10_000.0 * float(current["funding_rate"]),
            float(episode["current_funding_bps"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Round 61 current funding event drifted")
        start_index = bisect_right(times, decision)
        end_index = bisect_right(times, end)
        expected = times[start_index:end_index]
        observed = [int(value) for value in episode["future_funding_calc_times_ms"]]
        if observed != expected:
            raise ValueError("Round 61 future funding event set drifted")
    return by_time


def _source_gaps(
    episode: Mapping[str, object],
    *,
    spot: Mapping[int, Mapping[str, object]],
    futures: Mapping[int, Mapping[str, object]],
    marks: Mapping[int, Mapping[str, object]],
    funding: Mapping[int, Mapping[str, object]],
) -> list[str]:
    decision = _minute(int(episode["decision_time_ms"]))
    end = _minute(int(episode["end_time_ms"]))
    required = (
        ("spot_entry", decision, spot),
        ("spot_exit", end, spot),
        ("perpetual_entry", decision, futures),
        ("perpetual_exit", end, futures),
    )
    gaps = [
        f"{name}:{timestamp}"
        for name, timestamp, rows in required
        if timestamp not in rows
    ]
    for raw_time in episode["future_funding_calc_times_ms"]:
        timestamp = int(raw_time)
        if timestamp not in funding:
            gaps.append(f"settled_funding:{timestamp}")
        mark_time = _minute(timestamp)
        if mark_time not in marks:
            gaps.append(f"funding_mark_price:{mark_time}")
    return gaps


def _capacity_check(
    *,
    name: str,
    fill_notional_usdt: float,
    available_quote_usdt: float,
    maximum_participation: float,
) -> dict[str, object]:
    if (
        not math.isfinite(fill_notional_usdt)
        or not math.isfinite(available_quote_usdt)
        or fill_notional_usdt <= 0.0
        or available_quote_usdt < 0.0
    ):
        raise ValueError(f"{name} capacity inputs are invalid")
    participation = (
        fill_notional_usdt / available_quote_usdt
        if available_quote_usdt > 0.0
        else None
    )
    return {
        "fill": name,
        "fill_notional_usdt": fill_notional_usdt,
        "available_same_side_taker_quote_usdt": available_quote_usdt,
        "participation_fraction": participation,
        "maximum_participation_fraction": maximum_participation,
        "passed": participation is not None and participation <= maximum_participation,
    }


def _score_episode(
    episode: Mapping[str, object],
    *,
    spot: Mapping[int, Mapping[str, object]],
    futures: Mapping[int, Mapping[str, object]],
    marks: Mapping[int, Mapping[str, object]],
    funding: Mapping[int, Mapping[str, object]],
    position: Mapping[str, object],
    capacity: Mapping[str, object],
    costs: Mapping[str, object],
) -> dict[str, object]:
    decision = _minute(int(episode["decision_time_ms"]))
    end = _minute(int(episode["end_time_ms"]))
    spot_entry = spot[decision]
    spot_exit = spot[end]
    futures_entry = futures[decision]
    futures_exit = futures[end]
    target = float(position["target_spot_entry_notional_usdt"])
    committed = float(position["committed_capital_usdt"])
    base_quantity = target / float(spot_entry["high"])
    prices = {
        "spot_entry_buy": float(spot_entry["high"]),
        "spot_exit_sell": float(spot_exit["low"]),
        "perpetual_entry_sell": float(futures_entry["low"]),
        "perpetual_exit_buy": float(futures_exit["high"]),
    }
    center_prices = {
        "spot_entry_close": float(spot_entry["close"]),
        "spot_exit_close": float(spot_exit["close"]),
        "perpetual_entry_close": float(futures_entry["close"]),
        "perpetual_exit_close": float(futures_exit["close"]),
    }
    notionals = {
        "spot_entry": base_quantity * prices["spot_entry_buy"],
        "spot_exit": base_quantity * prices["spot_exit_sell"],
        "perpetual_entry": base_quantity * prices["perpetual_entry_sell"],
        "perpetual_exit": base_quantity * prices["perpetual_exit_buy"],
    }
    maximum = float(capacity["maximum_same_side_one_minute_taker_participation"])
    fills = [
        _capacity_check(
            name="spot_entry",
            fill_notional_usdt=notionals["spot_entry"],
            available_quote_usdt=float(spot_entry["taker_buy_quote_volume"]),
            maximum_participation=maximum,
        ),
        _capacity_check(
            name="spot_exit",
            fill_notional_usdt=notionals["spot_exit"],
            available_quote_usdt=float(spot_exit["quote_volume"])
            - float(spot_exit["taker_buy_quote_volume"]),
            maximum_participation=maximum,
        ),
        _capacity_check(
            name="perpetual_entry",
            fill_notional_usdt=notionals["perpetual_entry"],
            available_quote_usdt=float(futures_entry["quote_volume"])
            - float(futures_entry["taker_buy_quote_volume"]),
            maximum_participation=maximum,
        ),
        _capacity_check(
            name="perpetual_exit",
            fill_notional_usdt=notionals["perpetual_exit"],
            available_quote_usdt=float(futures_exit["taker_buy_quote_volume"]),
            maximum_participation=maximum,
        ),
    ]
    capacity_eligible = all(bool(row["passed"]) for row in fills)
    result: dict[str, object] = {
        "base_quantity": base_quantity,
        "adverse_execution_prices": prices,
        "center_reference_prices": center_prices,
        "actual_fill_notionals_usdt": notionals,
        "fill_capacity": fills,
        "capacity_eligible": capacity_eligible,
        "economically_scored": capacity_eligible,
    }
    if not capacity_eligible:
        return result

    spot_pnl = base_quantity * (prices["spot_exit_sell"] - prices["spot_entry_buy"])
    perpetual_pnl = base_quantity * (
        prices["perpetual_entry_sell"] - prices["perpetual_exit_buy"]
    )
    funding_settlements: list[dict[str, object]] = []
    funding_pnl = 0.0
    for raw_time in episode["future_funding_calc_times_ms"]:
        timestamp = int(raw_time)
        rate = float(funding[timestamp]["funding_rate"])
        mark = marks[_minute(timestamp)]
        adverse_mark = float(mark["low"] if rate >= 0.0 else mark["high"])
        payment = base_quantity * adverse_mark * rate
        funding_pnl += payment
        funding_settlements.append(
            {
                "calc_time_ms": timestamp,
                "mark_open_time_ms": _minute(timestamp),
                "funding_rate": rate,
                "adverse_mark_price": adverse_mark,
                "center_mark_price": float(mark["close"]),
                "short_funding_pnl_usdt": payment,
            }
        )
    basis_pnl = spot_pnl + perpetual_pnl
    gross_pnl = basis_pnl + funding_pnl
    spot_fee_rate = float(costs["spot_taker_fee_bps_per_fill"]) / 10_000.0
    futures_fee_rate = float(costs["futures_taker_fee_bps_per_fill"]) / 10_000.0
    operational_rate = (
        float(costs["additional_operational_slippage_bps_per_fill"]) / 10_000.0
    )
    exchange_fees = spot_fee_rate * (
        notionals["spot_entry"] + notionals["spot_exit"]
    ) + futures_fee_rate * (notionals["perpetual_entry"] + notionals["perpetual_exit"])
    operational_slippage = operational_rate * sum(notionals.values())
    stress_net = gross_pnl - exchange_fees - operational_slippage
    result.update(
        {
            "spot_pnl_usdt": spot_pnl,
            "perpetual_pnl_usdt": perpetual_pnl,
            "basis_pnl_usdt": basis_pnl,
            "short_funding_pnl_usdt": funding_pnl,
            "gross_pnl_usdt": gross_pnl,
            "exchange_taker_fees_usdt": exchange_fees,
            "additional_operational_slippage_usdt": operational_slippage,
            "stress_net_pnl_usdt": stress_net,
            "stress_net_leg_notional_bps": 10_000.0 * stress_net / target,
            "stress_net_committed_capital_bps": 10_000.0 * stress_net / committed,
            "funding_settlements": funding_settlements,
        }
    )
    return result


def _risk_metrics(
    episodes: Sequence[Mapping[str, object]],
    *,
    uncertainty: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    if not episodes:
        return {
            "episodes": 0,
            "first_decision_time_ms": None,
            "last_decision_time_ms": None,
            "mean_stress_net_committed_capital_bps": None,
            "median_stress_net_committed_capital_bps": None,
            "positive_stress_net_fraction": None,
            "p10_stress_net_committed_capital_bps": None,
            "p90_stress_net_committed_capital_bps": None,
            "worst_episode_committed_capital_bps": None,
            "expected_shortfall_10pct_committed_capital_bps": None,
            "maximum_sequential_drawdown_committed_capital_bps": None,
            "bootstrap_lower_95_mean_stress_net_committed_capital_bps": None,
            "bootstrap_upper_95_mean_stress_net_committed_capital_bps": None,
            "distinct_calendar_years": 0,
            "positive_calendar_year_fraction": None,
            "maximum_single_year_episode_fraction": None,
            "maximum_single_episode_share_of_positive_pnl": None,
            "yearly_results": [],
        }
    returns = np.asarray(
        [float(row["stress_net_committed_capital_bps"]) for row in episodes],
        dtype=np.float64,
    )
    pnl = np.asarray(
        [float(row["stress_net_pnl_usdt"]) for row in episodes],
        dtype=np.float64,
    )
    bootstrap = _stationary_bootstrap_means(
        returns,
        samples=int(uncertainty["bootstrap_samples"]),
        mean_block_length=float(uncertainty["mean_block_length_episodes"]),
        seed=seed,
    )
    equity = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(returns)))
    drawdown = np.maximum.accumulate(equity) - equity
    tail_count = max(1, math.ceil(0.10 * returns.size))
    years = [
        datetime.fromtimestamp(int(row["decision_time_ms"]) / 1000, tz=UTC).year
        for row in episodes
    ]
    year_counts = Counter(years)
    yearly_results: list[dict[str, object]] = []
    for year in sorted(year_counts):
        indexes = [index for index, value in enumerate(years) if value == year]
        year_pnl = float(np.sum(pnl[indexes]))
        year_return = float(np.sum(returns[indexes]))
        yearly_results.append(
            {
                "year": year,
                "episodes": len(indexes),
                "stress_net_pnl_usdt": year_pnl,
                "stress_net_committed_capital_bps": year_return,
                "positive": year_return > 0.0,
            }
        )
    positive_pnl = pnl[pnl > 0.0]
    positive_share = (
        float(np.max(positive_pnl) / np.sum(positive_pnl))
        if positive_pnl.size
        else None
    )
    lower = float(uncertainty["confidence_lower_quantile"])
    upper = float(uncertainty["confidence_upper_quantile"])
    return {
        "episodes": int(returns.size),
        "first_decision_time_ms": int(episodes[0]["decision_time_ms"]),
        "last_decision_time_ms": int(episodes[-1]["decision_time_ms"]),
        "mean_stress_net_committed_capital_bps": float(np.mean(returns)),
        "median_stress_net_committed_capital_bps": float(np.median(returns)),
        "positive_stress_net_fraction": float(np.mean(returns > 0.0)),
        "p10_stress_net_committed_capital_bps": float(np.quantile(returns, 0.10)),
        "p90_stress_net_committed_capital_bps": float(np.quantile(returns, 0.90)),
        "worst_episode_committed_capital_bps": float(np.min(returns)),
        "expected_shortfall_10pct_committed_capital_bps": float(
            np.mean(np.sort(returns)[:tail_count])
        ),
        "maximum_sequential_drawdown_committed_capital_bps": float(np.max(drawdown)),
        "bootstrap_lower_95_mean_stress_net_committed_capital_bps": float(
            np.quantile(bootstrap, lower)
        ),
        "bootstrap_upper_95_mean_stress_net_committed_capital_bps": float(
            np.quantile(bootstrap, upper)
        ),
        "total_stress_net_pnl_usdt": float(np.sum(pnl)),
        "mean_basis_pnl_usdt": float(
            np.mean([float(row["basis_pnl_usdt"]) for row in episodes])
        ),
        "mean_short_funding_pnl_usdt": float(
            np.mean([float(row["short_funding_pnl_usdt"]) for row in episodes])
        ),
        "mean_exchange_taker_fees_usdt": float(
            np.mean([float(row["exchange_taker_fees_usdt"]) for row in episodes])
        ),
        "mean_additional_operational_slippage_usdt": float(
            np.mean(
                [float(row["additional_operational_slippage_usdt"]) for row in episodes]
            )
        ),
        "distinct_calendar_years": len(yearly_results),
        "positive_calendar_year_fraction": float(
            np.mean([bool(row["positive"]) for row in yearly_results])
        ),
        "maximum_single_year_episode_fraction": max(year_counts.values())
        / returns.size,
        "maximum_single_episode_share_of_positive_pnl": positive_share,
        "yearly_results": yearly_results,
    }


def _gate_check(
    check_id: str,
    *,
    observed: object,
    operator: str,
    threshold: object,
    passed: bool,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _symbol_gate(
    summary: Mapping[str, object],
    metrics: Mapping[str, object],
    gate: Mapping[str, object],
) -> dict[str, object]:
    source_count = int(summary["source_eligible_episodes"])
    source_fraction = float(summary["source_eligible_fraction"])
    capacity_count = int(summary["capacity_eligible_episodes"])
    capacity_fraction = float(summary["capacity_eligible_fraction"])
    checks = [
        _gate_check(
            "source_rows_reconciled",
            observed=summary["source_rows_reconciled"],
            operator="is",
            threshold=True,
            passed=summary["source_rows_reconciled"] is True,
        ),
        _gate_check(
            "minimum_source_eligible_episodes",
            observed=source_count,
            operator=">=",
            threshold=gate["minimum_source_eligible_episodes_per_symbol"],
            passed=source_count
            >= int(gate["minimum_source_eligible_episodes_per_symbol"]),
        ),
        _gate_check(
            "minimum_source_eligible_fraction",
            observed=source_fraction,
            operator=">=",
            threshold=gate["minimum_source_eligible_fraction_per_symbol"],
            passed=source_fraction
            >= float(gate["minimum_source_eligible_fraction_per_symbol"]),
        ),
        _gate_check(
            "minimum_capacity_eligible_episodes",
            observed=capacity_count,
            operator=">=",
            threshold=gate["minimum_capacity_eligible_episodes_per_symbol"],
            passed=capacity_count
            >= int(gate["minimum_capacity_eligible_episodes_per_symbol"]),
        ),
        _gate_check(
            "minimum_capacity_eligible_fraction",
            observed=capacity_fraction,
            operator=">=",
            threshold=gate["minimum_capacity_eligible_fraction"],
            passed=capacity_fraction
            >= float(gate["minimum_capacity_eligible_fraction"]),
        ),
    ]
    metric_specs = (
        (
            "minimum_stress_net_positive_fraction",
            "positive_stress_net_fraction",
            ">=",
            "minimum_stress_net_positive_fraction",
        ),
        (
            "positive_median_stress_net",
            "median_stress_net_committed_capital_bps",
            ">",
            "median_stress_net_committed_capital_bps_strictly_above",
        ),
        (
            "positive_bootstrap_lower_mean",
            "bootstrap_lower_95_mean_stress_net_committed_capital_bps",
            ">",
            "bootstrap_lower_95_mean_stress_net_committed_capital_bps_strictly_above",
        ),
        (
            "maximum_sequential_drawdown",
            "maximum_sequential_drawdown_committed_capital_bps",
            "<=",
            "maximum_sequential_drawdown_committed_capital_bps",
        ),
        (
            "minimum_worst_episode",
            "worst_episode_committed_capital_bps",
            ">=",
            "minimum_worst_episode_committed_capital_bps",
        ),
        (
            "minimum_expected_shortfall_10pct",
            "expected_shortfall_10pct_committed_capital_bps",
            ">=",
            "minimum_expected_shortfall_10pct_committed_capital_bps",
        ),
        (
            "minimum_distinct_calendar_years",
            "distinct_calendar_years",
            ">=",
            "minimum_distinct_calendar_years",
        ),
        (
            "minimum_positive_calendar_year_fraction",
            "positive_calendar_year_fraction",
            ">=",
            "minimum_positive_calendar_year_fraction",
        ),
        (
            "maximum_single_year_episode_fraction",
            "maximum_single_year_episode_fraction",
            "<=",
            "maximum_single_year_episode_fraction",
        ),
        (
            "maximum_single_episode_share_of_positive_pnl",
            "maximum_single_episode_share_of_positive_pnl",
            "<=",
            "maximum_single_episode_share_of_positive_pnl",
        ),
    )
    for check_id, metric_name, operator, threshold_name in metric_specs:
        observed = metrics.get(metric_name)
        threshold = gate[threshold_name]
        if observed is None:
            passed = False
        elif operator == ">":
            passed = float(observed) > float(threshold)
        elif operator == ">=":
            passed = float(observed) >= float(threshold)
        else:
            passed = float(observed) <= float(threshold)
        checks.append(
            _gate_check(
                check_id,
                observed=observed,
                operator=operator,
                threshold=threshold,
                passed=passed,
            )
        )
    return {
        "checks": checks,
        "passed": all(bool(row["passed"]) for row in checks),
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design = _validate_design(arguments.design.resolve())
    manifest = _validate_manifest(arguments.manifest.resolve())
    source_certificate, source_file_sha, source_sha = _validate_source_certificate(
        arguments.source_certificate.resolve(), design=design, manifest=manifest
    )
    funding_design, funding_certificate, funding_file_sha, funding_sha = (
        _validate_funding_certificate(arguments.funding_certificate.resolve())
    )
    if _git("status", "--porcelain"):
        raise ValueError("Round 61 economic runner requires a clean worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    series = _series_contracts(source_certificate)
    archive_by_symbol = {
        row["symbol"]: row for row in funding_certificate["archive_evidence"]
    }
    position = design["position_contract"]
    capacity = design["capacity_contract"]
    costs = design["cost_contract"]
    uncertainty = design["uncertainty_contract"]
    gate = design["risk_and_authorization_gate"]
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "design_sha256": DESIGN_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "implementation_commit": implementation_commit,
        "status": "running",
        "source": {
            "database_file": arguments.database.resolve().name,
            "event_source_certificate_file_sha256": source_file_sha,
            "event_source_certificate_canonical_sha256": source_sha,
            "funding_source_certificate_file_sha256": funding_file_sha,
            "funding_source_certificate_canonical_sha256": funding_sha,
            "symbols": list(SYMBOLS),
        },
        "position_contract": position,
        "execution_contract": design["execution_contract"],
        "cost_contract": costs,
        "capacity_contract": capacity,
        "risk_metric_contract": design["risk_metric_contract"],
        "symbol_results": [],
        "selection_contaminated": True,
        "tick_execution_replay_authorized": False,
        "model_training_authorized": False,
        "ai_evaluation_authorized": False,
        "trading_authority": False,
        "testnet_or_live_authority": False,
        "profitability_claim": False,
        "leverage_applied": False,
        "synthetic_or_filled_source_rows": False,
    }
    with _connect_read_only(arguments.database.resolve()) as connection:
        for symbol_index, item in enumerate(manifest["symbol_manifests"]):
            symbol = str(item["symbol"])
            episodes = list(item["episodes"])
            spot_times = list(item["required_spot_open_times_ms"])
            futures_times = list(item["required_futures_execution_open_times_ms"])
            mark_times = list(item["required_mark_open_times_ms"])
            print(
                _canonical_json({"phase": "source-audit-start", "symbol": symbol}),
                flush=True,
            )
            spot, spot_digest = _load_execution_rows(
                connection,
                symbol=symbol,
                market_type="spot",
                source=SPOT_SOURCE,
                required_times=spot_times,
            )
            futures, futures_digest = _load_execution_rows(
                connection,
                symbol=symbol,
                market_type="futures",
                source=FUTURES_SOURCE,
                required_times=futures_times,
            )
            marks, marks_digest = _load_mark_rows(
                connection, symbol=symbol, required_times=mark_times
            )
            _reconcile_series(
                spot,
                spot_digest,
                required_times=spot_times,
                contract=series[(symbol, "spot_execution")],
            )
            _reconcile_series(
                futures,
                futures_digest,
                required_times=futures_times,
                contract=series[(symbol, "futures_execution")],
            )
            _reconcile_series(
                marks,
                marks_digest,
                required_times=mark_times,
                contract=series[(symbol, "funding_mark_price")],
            )
            funding_range = funding_design["source_contract"]["ranges_by_symbol"][
                symbol
            ]
            funding_view = {
                "source_contract": {
                    "start_period": funding_range["start_period"],
                    "end_period": funding_range["end_period"],
                    "periods_per_symbol": funding_range["period_count"],
                    "expected_rows": {symbol: archive_by_symbol[symbol]["rows_read"]},
                }
            }
            funding_rows, funding_evidence = _load_verified_funding(
                connection,
                symbol=symbol,
                design=funding_view,
                certificate=funding_certificate,
            )
            funding = _validate_manifest_funding(episodes, funding_rows)
            episode_results: list[dict[str, object]] = []
            for episode in episodes:
                gaps = _source_gaps(
                    episode,
                    spot=spot,
                    futures=futures,
                    marks=marks,
                    funding=funding,
                )
                result: dict[str, object] = {
                    "episode_id": episode["episode_id"],
                    "decision_time_ms": episode["decision_time_ms"],
                    "end_time_ms": episode["end_time_ms"],
                    "decision_open_time_ms": _minute(int(episode["decision_time_ms"])),
                    "end_open_time_ms": _minute(int(episode["end_time_ms"])),
                    "current_funding_bps": episode["current_funding_bps"],
                    "future_funding_settlements": len(
                        episode["future_funding_calc_times_ms"]
                    ),
                    "source_eligible": not gaps,
                    "source_ineligible_reasons": gaps,
                    "capacity_eligible": False,
                    "economically_scored": False,
                }
                if not gaps:
                    result.update(
                        _score_episode(
                            episode,
                            spot=spot,
                            futures=futures,
                            marks=marks,
                            funding=funding,
                            position=position,
                            capacity=capacity,
                            costs=costs,
                        )
                    )
                episode_results.append(result)
            source_count = sum(bool(row["source_eligible"]) for row in episode_results)
            capacity_count = sum(
                bool(row["capacity_eligible"]) for row in episode_results
            )
            scored = [
                row for row in episode_results if bool(row["economically_scored"])
            ]
            summary = {
                "manifest_episodes": len(episode_results),
                "source_rows_reconciled": True,
                "source_eligible_episodes": source_count,
                "source_ineligible_episodes": len(episode_results) - source_count,
                "source_eligible_fraction": source_count / len(episode_results),
                "capacity_eligible_episodes": capacity_count,
                "capacity_ineligible_source_eligible_episodes": source_count
                - capacity_count,
                "capacity_eligible_fraction": capacity_count / source_count
                if source_count
                else 0.0,
                "economically_scored_episodes": len(scored),
            }
            metrics = _risk_metrics(
                scored,
                uncertainty=uncertainty,
                seed=int(uncertainty["seed"]) + symbol_index,
            )
            symbol_gate = _symbol_gate(summary, metrics, gate)
            symbol_result = {
                "symbol": symbol,
                "source": {
                    "spot_required_rows": len(spot_times),
                    "spot_stored_rows": len(spot),
                    "futures_required_rows": len(futures_times),
                    "futures_stored_rows": len(futures),
                    "mark_required_rows": len(mark_times),
                    "mark_stored_rows": len(marks),
                    "funding": funding_evidence,
                },
                "summary": summary,
                "metrics": metrics,
                "gate": symbol_gate,
                "episodes": episode_results,
            }
            symbol_result["result_sha256"] = _canonical_sha256(symbol_result)
            report["symbol_results"].append(symbol_result)
            print(
                _canonical_json(
                    {
                        "phase": "symbol-complete",
                        "symbol": symbol,
                        "source_eligible": source_count,
                        "capacity_eligible": capacity_count,
                        "gate_passed": symbol_gate["passed"],
                    }
                ),
                flush=True,
            )
    passed_symbols = [
        row["symbol"] for row in report["symbol_results"] if row["gate"]["passed"]
    ]
    authorized = tuple(passed_symbols) == SYMBOLS
    report["tick_execution_replay_authorized"] = authorized
    report["status"] = (
        "minute_stress_gate_passed_tick_replay_only"
        if authorized
        else "rejected_elevated_funding_carry"
    )
    report["result"] = {
        "passed_symbols": passed_symbols,
        "required_symbols": list(SYMBOLS),
        "all_symbols_passed": authorized,
        "authorized_next_step": (
            "separately frozen tick-level execution replay"
            if authorized
            else "none; reject this elevated-funding carry family"
        ),
    }
    _validate_finite(report)
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(arguments.output.resolve(), report, indent=2)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-061-carry-economic-replay-design.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-061-carry-event-manifest.json",
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--funding-certificate", type=Path, required=True)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(
        _canonical_json(
            {
                "round": report["round"],
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "tick_execution_replay_authorized": report[
                    "tick_execution_replay_authorized"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
