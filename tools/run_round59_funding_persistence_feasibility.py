"""Run the frozen Round 59 funding-persistence feasibility screen."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from simple_ai_trading.types import (  # noqa: E402
    DEFAULT_FUTURES_TAKER_FEE_BPS,
    DEFAULT_SPOT_TAKER_FEE_BPS,
)


ROUND = 59
DESIGN_SCHEMA = "round-059-funding-persistence-feasibility-design-v1"
REPORT_SCHEMA = "round-059-funding-persistence-feasibility-report-v1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SOURCE_CERTIFICATE_FILE_SHA256 = (
    "e2fe434d7c290f09913160506c52fce30849a6bd319465390c4b4d22dad482a7"
)
SOURCE_CERTIFICATE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
ROUND57_COST_CONTRACT_SHA256 = (
    "ef42dcd1fcf003838a34c78a3d87a49b45d78f16b7be47b596fc9eece9841dd6"
)
TYPES_BLOB_OID = "5f222890236011c94705a1839498c0320023f3d3"
ROUND57_COST_BLOB_OID = "4634f73355c0ecc7ca72f5072da4ac9074b9ae34"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _canonical_value(value: Mapping[str, object], digest_key: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(digest_key, ""))
    actual = _canonical_sha256(canonical)
    if claimed != actual:
        raise ValueError(f"{digest_key} does not match canonical content")
    return actual


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _validate_finite(value: object, label: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{label}[{index}]")


def _validate_design(path: Path) -> dict[str, object]:
    design = _read_object(path, "Round 59 design")
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or _canonical_value(design, "design_sha256") == ""
        or design.get("governance", {}).get("source_data_previously_consumed")
        is not True
        or design.get("governance", {}).get("selection_contaminated") is not True
        or any(
            design.get("governance", {}).get(name) is not False
            for name in (
                "promotion_permitted",
                "profitability_claim_permitted",
                "trading_authority_permitted",
                "testnet_or_live_authority_permitted",
                "leverage_permitted",
                "model_training_permitted",
                "ai_evaluation_permitted",
            )
        )
        or tuple(design.get("source_contract", {}).get("symbols", ())) != SYMBOLS
        or design.get("source_contract", {}).get("price_rows_permitted") is not False
        or design.get("source_contract", {}).get("spot_rows_permitted") is not False
        or design.get("authorization_gate", {}).get(
            "same_trigger_and_horizon_must_pass_all_symbols"
        )
        is not True
    ):
        raise ValueError("Round 59 frozen design drifted")
    _validate_finite(design, "design")
    return design


def _validate_certificate(
    path: Path, design: Mapping[str, object]
) -> dict[str, object]:
    if _file_sha256(path) != SOURCE_CERTIFICATE_FILE_SHA256:
        raise ValueError("Round 38 source certificate file hash drifted")
    certificate = _read_object(path, "Round 38 source certificate")
    if (
        _canonical_value(certificate, "source_certificate_sha256")
        != SOURCE_CERTIFICATE_CANONICAL_SHA256
        or certificate.get("schema_version")
        != "round-038-derivatives-source-certificate-v1"
        or tuple(certificate.get("symbols", ())) != SYMBOLS
        or certificate.get("start_period") != design["source_contract"]["start_period"]
        or certificate.get("end_period") != design["source_contract"]["end_period"]
    ):
        raise ValueError("Round 38 source certificate identity drifted")
    return certificate


def _validate_cost_references(design: Mapping[str, object]) -> None:
    costs = design["cost_reference_contract"]
    contract_path = ROOT / str(costs["round57_cost_contract_path"])
    contract = _read_object(contract_path, "Round 57 cost contract")
    if (
        _canonical_value(contract, "contract_sha256") != ROUND57_COST_CONTRACT_SHA256
        or _git("rev-parse", "HEAD:src/simple_ai_trading/types.py") != TYPES_BLOB_OID
        or _git("rev-parse", f"HEAD:{costs['round57_cost_contract_path']}")
        != ROUND57_COST_BLOB_OID
        or DEFAULT_SPOT_TAKER_FEE_BPS != 10.0
        or DEFAULT_FUTURES_TAKER_FEE_BPS != 4.0
        or float(costs["repo_offline_four_leg_taker_bps"])
        != 2.0 * (DEFAULT_SPOT_TAKER_FEE_BPS + DEFAULT_FUTURES_TAKER_FEE_BPS)
        or float(costs["stress_four_leg_bps"])
        != float(costs["repo_offline_four_leg_taker_bps"])
        + 4.0 * float(costs["stress_additional_slippage_bps_per_fill"])
        or float(contract["feature_spec"]["maker_entry_fee_bps"]) != 2.0
        or float(costs["optimistic_futures_maker_only_bps"])
        != 2.0 * float(contract["feature_spec"]["maker_entry_fee_bps"])
    ):
        raise ValueError("Round 59 cost reference identity drifted")


def _periods(start: str, end: str) -> list[str]:
    start_year, start_month = (int(value) for value in start.split("-"))
    end_year, end_month = (int(value) for value in end.split("-"))
    periods: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        periods.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return periods


def _period_bounds_ms(period: str) -> tuple[int, int]:
    year, month = (int(value) for value in period.split("-"))
    start = datetime(year, month, 1, tzinfo=UTC)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = datetime(year, month + 1, 1, tzinfo=UTC)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _row_stream_sha256(rows: Sequence[sqlite3.Row]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            [
                int(row["calc_time"]),
                int(row["funding_interval_hours"]),
                float(row["funding_rate"]),
            ],
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = path.resolve().as_uri().replace("file:///", "file:/") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _archive_identity(rows: Sequence[sqlite3.Row]) -> str:
    return _canonical_sha256(
        [
            {
                "url": row["url"],
                "period": row["period"],
                "rows_read": int(row["rows_read"]),
                "sha256": row["sha256"],
                "checksum_sha256": row["checksum_sha256"],
                "row_stream_sha256": row["row_stream_sha256"],
            }
            for row in rows
        ]
    )


def _load_verified_funding(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    design: Mapping[str, object],
    certificate: Mapping[str, object],
) -> tuple[list[sqlite3.Row], dict[str, object]]:
    source = design["source_contract"]
    periods = _periods(str(source["start_period"]), str(source["end_period"]))
    metadata = connection.execute(
        """
        SELECT url, period, status, rows_read, sha256, checksum_sha256,
               checksum_status, row_stream_sha256
        FROM derivatives_archive_files
        WHERE symbol=? AND market_type='futures' AND data_type='fundingRate'
          AND period BETWEEN ? AND ?
        ORDER BY period
        """,
        (symbol, periods[0], periods[-1]),
    ).fetchall()
    expected_archive = next(
        row
        for row in certificate["archive_evidence"]
        if row["symbol"] == symbol and row["data_type"] == "fundingRate"
    )
    if (
        [row["period"] for row in metadata] != periods
        or len(metadata) != int(source["periods_per_symbol"])
        or any(row["status"] != "complete" for row in metadata)
        or any(row["checksum_status"] != "verified" for row in metadata)
        or any(row["sha256"] != row["checksum_sha256"] for row in metadata)
        or _archive_identity(metadata) != expected_archive["archive_identity_sha256"]
    ):
        raise ValueError(f"{symbol} funding archive metadata drifted")

    all_rows: list[sqlite3.Row] = []
    period_evidence: list[dict[str, object]] = []
    for item in metadata:
        start_ms, end_ms = _period_bounds_ms(str(item["period"]))
        rows = connection.execute(
            """
            SELECT calc_time, funding_interval_hours, funding_rate
            FROM funding_rates
            WHERE symbol=? AND market_type='futures'
              AND calc_time>=? AND calc_time<?
            ORDER BY calc_time
            """,
            (symbol, start_ms, end_ms),
        ).fetchall()
        digest = _row_stream_sha256(rows)
        if len(rows) != int(item["rows_read"]) or digest != item["row_stream_sha256"]:
            raise ValueError(f"{symbol} {item['period']} funding rows drifted")
        if any(
            not 1 <= int(row["funding_interval_hours"]) <= 8
            or not math.isfinite(float(row["funding_rate"]))
            or abs(float(row["funding_rate"])) > 0.1
            for row in rows
        ):
            raise ValueError(f"{symbol} {item['period']} funding values are invalid")
        all_rows.extend(rows)
        period_evidence.append(
            {
                "period": item["period"],
                "rows": len(rows),
                "row_stream_sha256": digest,
            }
        )
    if len(all_rows) != int(source["expected_rows"][symbol]):
        raise ValueError(f"{symbol} total funding rows drifted")
    times = [int(row["calc_time"]) for row in all_rows]
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError(f"{symbol} funding timestamps are not strictly increasing")
    evidence = {
        "symbol": symbol,
        "periods": len(period_evidence),
        "rows": len(all_rows),
        "first_calc_time_ms": times[0],
        "last_calc_time_ms": times[-1],
        "minimum_interval_hours": min(
            int(row["funding_interval_hours"]) for row in all_rows
        ),
        "maximum_interval_hours": max(
            int(row["funding_interval_hours"]) for row in all_rows
        ),
        "archive_identity_sha256": _archive_identity(metadata),
        "funding_row_stream_sha256": _row_stream_sha256(all_rows),
        "period_evidence_sha256": _canonical_sha256(period_evidence),
    }
    return all_rows, evidence


def _sign_transition(rows: Sequence[sqlite3.Row]) -> dict[str, object]:
    rates = np.asarray([float(row["funding_rate"]) for row in rows], dtype=np.float64)
    current_positive = rates[:-1] > 0.0
    next_positive = rates[1:] > 0.0
    pp = int(np.count_nonzero(current_positive & next_positive))
    pn = int(np.count_nonzero(current_positive & ~next_positive))
    np_count = int(np.count_nonzero(~current_positive & next_positive))
    nn = int(np.count_nonzero(~current_positive & ~next_positive))
    positive_total = pp + pn
    nonpositive_total = np_count + nn
    return {
        "positive_to_positive": pp,
        "positive_to_nonpositive": pn,
        "nonpositive_to_positive": np_count,
        "nonpositive_to_nonpositive": nn,
        "next_positive_given_positive": (
            pp / positive_total if positive_total else None
        ),
        "next_positive_given_nonpositive": (
            np_count / nonpositive_total if nonpositive_total else None
        ),
    }


def _triggered(current_bps: float, trigger: Mapping[str, object]) -> bool:
    threshold = float(trigger["value"])
    if trigger["operator"] == "strictly_greater":
        return current_bps > threshold
    if trigger["operator"] == "greater_or_equal":
        return current_bps >= threshold
    raise ValueError(f"unsupported trigger operator: {trigger['operator']}")


def _episodes(
    rows: Sequence[sqlite3.Row],
    *,
    trigger: Mapping[str, object],
    horizon_hours: int,
) -> list[dict[str, object]]:
    times = [int(row["calc_time"]) for row in rows]
    rates = np.asarray([float(row["funding_rate"]) for row in rows], dtype=np.float64)
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(rates)))
    horizon_ms = horizon_hours * 60 * 60 * 1000
    last_source_time = times[-1]
    next_eligible_ms = times[0]
    episodes: list[dict[str, object]] = []
    for index, decision_ms in enumerate(times):
        end_ms = decision_ms + horizon_ms
        if end_ms > last_source_time:
            break
        if decision_ms < next_eligible_ms:
            continue
        current_bps = 10_000.0 * rates[index]
        if not _triggered(current_bps, trigger):
            continue
        future_start = bisect_right(times, decision_ms)
        future_end = bisect_right(times, end_ms)
        gross_bps = 10_000.0 * float(prefix[future_end] - prefix[future_start])
        episodes.append(
            {
                "decision_time_ms": decision_ms,
                "end_time_ms": end_ms,
                "current_funding_bps": current_bps,
                "future_settlements": future_end - future_start,
                "gross_future_funding_bps": gross_bps,
            }
        )
        next_eligible_ms = end_ms
    return episodes


def _stationary_bootstrap_means(
    values: np.ndarray,
    *,
    samples: int,
    mean_block_length: float,
    seed: int,
) -> np.ndarray:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap requires a nonempty one-dimensional sample")
    rng = np.random.default_rng(seed)
    n = values.size
    restart_probability = 1.0 / mean_block_length
    output = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        index = int(rng.integers(0, n))
        total = 0.0
        for position in range(n):
            total += float(values[index])
            if position + 1 < n:
                if rng.random() < restart_probability:
                    index = int(rng.integers(0, n))
                else:
                    index = (index + 1) % n
        output[sample] = total / n
    return output


def _episode_metrics(
    episodes: Sequence[Mapping[str, object]],
    *,
    costs: Mapping[str, float],
    uncertainty: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    gross = np.asarray(
        [float(row["gross_future_funding_bps"]) for row in episodes],
        dtype=np.float64,
    )
    if gross.size == 0:
        return {
            "episodes": 0,
            "first_decision_time_ms": None,
            "last_decision_time_ms": None,
            "mean_gross_funding_bps": None,
            "median_gross_funding_bps": None,
            "p10_gross_funding_bps": None,
            "p90_gross_funding_bps": None,
            "bootstrap_lower_95_mean_gross_bps": None,
            "bootstrap_upper_95_mean_gross_bps": None,
            "mean_future_settlements": None,
            "cost_comparisons": {
                name: {
                    "cost_reference_bps": cost,
                    "mean_net_reference_bps": None,
                    "median_net_reference_bps": None,
                    "positive_net_reference_fraction": None,
                    "bootstrap_lower_95_mean_net_reference_bps": None,
                    "bootstrap_upper_95_mean_net_reference_bps": None,
                }
                for name, cost in costs.items()
            },
        }
    means = _stationary_bootstrap_means(
        gross,
        samples=int(uncertainty["bootstrap_samples"]),
        mean_block_length=float(uncertainty["mean_block_length_episodes"]),
        seed=seed,
    )
    lower_q = float(uncertainty["confidence_lower_quantile"])
    upper_q = float(uncertainty["confidence_upper_quantile"])
    metrics: dict[str, object] = {
        "episodes": int(gross.size),
        "first_decision_time_ms": int(episodes[0]["decision_time_ms"]),
        "last_decision_time_ms": int(episodes[-1]["decision_time_ms"]),
        "mean_gross_funding_bps": float(np.mean(gross)),
        "median_gross_funding_bps": float(np.median(gross)),
        "p10_gross_funding_bps": float(np.quantile(gross, 0.10)),
        "p90_gross_funding_bps": float(np.quantile(gross, 0.90)),
        "bootstrap_lower_95_mean_gross_bps": float(np.quantile(means, lower_q)),
        "bootstrap_upper_95_mean_gross_bps": float(np.quantile(means, upper_q)),
        "mean_future_settlements": float(
            np.mean([int(row["future_settlements"]) for row in episodes])
        ),
    }
    comparisons: dict[str, object] = {}
    for name, cost in costs.items():
        net = gross - cost
        comparisons[name] = {
            "cost_reference_bps": cost,
            "mean_net_reference_bps": float(np.mean(net)),
            "median_net_reference_bps": float(np.median(net)),
            "positive_net_reference_fraction": float(np.mean(net > 0.0)),
            "bootstrap_lower_95_mean_net_reference_bps": float(
                np.quantile(means - cost, lower_q)
            ),
            "bootstrap_upper_95_mean_net_reference_bps": float(
                np.quantile(means - cost, upper_q)
            ),
        }
    metrics["cost_comparisons"] = comparisons
    return metrics


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design = _validate_design(arguments.design.resolve())
    certificate = _validate_certificate(arguments.certificate.resolve(), design)
    _validate_cost_references(design)
    implementation_commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain"):
        raise ValueError("Round 59 runner requires a clean worktree")

    costs_contract = design["cost_reference_contract"]
    costs = {
        "optimistic_futures_maker_only": float(
            costs_contract["optimistic_futures_maker_only_bps"]
        ),
        "repo_offline_four_leg_taker": float(
            costs_contract["repo_offline_four_leg_taker_bps"]
        ),
        "stress_four_leg": float(costs_contract["stress_four_leg_bps"]),
    }
    uncertainty = design["uncertainty_contract"]
    gate_contract = design["authorization_gate"]
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "design_sha256": design["design_sha256"],
        "implementation_commit": implementation_commit,
        "status": "running",
        "source": {
            "certificate_file_sha256": SOURCE_CERTIFICATE_FILE_SHA256,
            "certificate_canonical_sha256": SOURCE_CERTIFICATE_CANONICAL_SHA256,
            "database_file": arguments.database.resolve().name,
            "symbols": list(SYMBOLS),
            "start_period": design["source_contract"]["start_period"],
            "end_period": design["source_contract"]["end_period"],
        },
        "cost_references_bps": costs,
        "symbol_results": [],
        "breadth_gates": [],
        "spot_history_ingestion_authorized": False,
        "selection_contaminated": True,
        "price_rows_read": False,
        "premium_index_rows_read": False,
        "spot_rows_read": False,
        "model_trained": False,
        "ai_evaluated": False,
        "profitability_claim": False,
        "trading_authority": False,
        "testnet_or_live_authority": False,
        "leverage_applied": False,
    }
    cell_lookup: dict[tuple[str, int, str], dict[str, object]] = {}
    with _connect_read_only(arguments.database.resolve()) as connection:
        for symbol_index, symbol in enumerate(SYMBOLS):
            print(
                _canonical_json({"phase": "source-audit-start", "symbol": symbol}),
                flush=True,
            )
            funding_rows, source_evidence = _load_verified_funding(
                connection,
                symbol=symbol,
                design=design,
                certificate=certificate,
            )
            cells: list[dict[str, object]] = []
            for trigger_index, trigger in enumerate(
                design["causal_trigger_contract"]["triggers_bps"]
            ):
                for horizon_index, horizon in enumerate(
                    design["episode_contract"]["holding_horizons_hours"]
                ):
                    episodes = _episodes(
                        funding_rows,
                        trigger=trigger,
                        horizon_hours=int(horizon),
                    )
                    seed = (
                        int(uncertainty["seed"])
                        + 100 * symbol_index
                        + 10 * trigger_index
                        + horizon_index
                    )
                    metrics = _episode_metrics(
                        episodes,
                        costs=costs,
                        uncertainty=uncertainty,
                        seed=seed,
                    )
                    stress = metrics["cost_comparisons"]["stress_four_leg"]
                    passed = (
                        int(metrics["episodes"])
                        >= int(
                            gate_contract["minimum_nonoverlapping_episodes_per_symbol"]
                        )
                        and float(stress["positive_net_reference_fraction"])
                        >= float(gate_contract["minimum_stress_net_positive_fraction"])
                        and float(stress["median_net_reference_bps"])
                        > float(gate_contract["median_stress_net_bps_strictly_above"])
                        and float(stress["bootstrap_lower_95_mean_net_reference_bps"])
                        > float(
                            gate_contract[
                                "bootstrap_lower_95_mean_stress_net_bps_strictly_above"
                            ]
                        )
                    )
                    cell = {
                        "symbol": symbol,
                        "trigger_id": trigger["id"],
                        "trigger_operator": trigger["operator"],
                        "trigger_value_bps": trigger["value"],
                        "horizon_hours": horizon,
                        "bootstrap_seed": seed,
                        **metrics,
                        "symbol_gate_passed": passed,
                    }
                    cells.append(cell)
                    cell_lookup[(str(trigger["id"]), int(horizon), symbol)] = cell
            result = {
                "symbol": symbol,
                "source": source_evidence,
                "sign_transition": _sign_transition(funding_rows),
                "cells": cells,
            }
            result["result_sha256"] = _canonical_sha256(result)
            report["symbol_results"].append(result)
            print(
                _canonical_json(
                    {
                        "phase": "symbol-complete",
                        "symbol": symbol,
                        "source_rows": source_evidence["rows"],
                        "cells": len(cells),
                        "passed_cells": sum(
                            bool(cell["symbol_gate_passed"]) for cell in cells
                        ),
                    }
                ),
                flush=True,
            )

    breadth_gates: list[dict[str, object]] = []
    for trigger in design["causal_trigger_contract"]["triggers_bps"]:
        for horizon in design["episode_contract"]["holding_horizons_hours"]:
            passed_symbols = [
                symbol
                for symbol in SYMBOLS
                if cell_lookup[(str(trigger["id"]), int(horizon), symbol)][
                    "symbol_gate_passed"
                ]
            ]
            breadth_gates.append(
                {
                    "trigger_id": trigger["id"],
                    "horizon_hours": horizon,
                    "passed_symbols": passed_symbols,
                    "required_symbols": list(SYMBOLS),
                    "passed": tuple(passed_symbols) == SYMBOLS,
                }
            )
    report["breadth_gates"] = breadth_gates
    passing_count = sum(bool(row["passed"]) for row in breadth_gates)
    authorized = passing_count >= int(gate_contract["passing_cell_count_required"])
    report["spot_history_ingestion_authorized"] = authorized
    report["status"] = (
        "support_passed_spot_ingestion_authorized"
        if authorized
        else "rejected_funding_persistence_support"
    )
    report["result"] = {
        "symbol_cells": len(SYMBOLS)
        * len(design["causal_trigger_contract"]["triggers_bps"])
        * len(design["episode_contract"]["holding_horizons_hours"]),
        "breadth_cells": len(breadth_gates),
        "passing_breadth_cells": passing_count,
        "spot_history_ingestion_authorized": authorized,
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
        / "docs/model-research/action-value/round-059-funding-persistence-feasibility-design.json",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round38-derivatives-source-20260712-v2\certificate.json"
        ),
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run(arguments)
    print(
        _canonical_json(
            {
                "round": report["round"],
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "spot_history_ingestion_authorized": report[
                    "spot_history_ingestion_authorized"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
