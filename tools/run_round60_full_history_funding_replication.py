"""Run Round 60's unchanged full-history funding-persistence replication."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round59_funding_persistence_feasibility import (  # noqa: E402
    SYMBOLS,
    _canonical_json,
    _canonical_sha256,
    _connect_read_only,
    _episode_metrics,
    _episodes,
    _file_sha256,
    _load_verified_funding,
    _read_object,
    _sign_transition,
    _validate_cost_references,
    _validate_design as _validate_round59_design,
    _validate_finite,
)


ROUND = 60
DESIGN_SCHEMA = "round-060-full-history-funding-replication-design-v1"
CERTIFICATE_SCHEMA = "round-060-full-history-funding-source-certificate-v1"
REPORT_SCHEMA = "round-060-full-history-funding-replication-report-v1"
ROUND59_REPORT_FILE_SHA256 = (
    "f99843b7998a9bc473f7c8d8c80c52a8e718e7729e5d9896263bdfe01538d14e"
)
ROUND59_REPORT_CANONICAL_SHA256 = (
    "268e0a4734ae10ad2f413ca77e75de3cdc55ae98a0ae07bd5c3a944499be03d0"
)


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


def _validate_design(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    design = _read_object(path, "Round 60 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    reuse = design.get("protocol_reuse_contract", {})
    source_path = ROOT / str(reuse.get("source_design_path", ""))
    prior = _validate_round59_design(source_path)
    section_hashes = reuse.get("reused_sections", {})
    expected_ranges = {
        "BTCUSDT": ("2020-01", "2026-06", 78),
        "ETHUSDT": ("2020-01", "2026-06", 78),
        "SOLUSDT": ("2020-09", "2026-06", 70),
    }
    ranges = design.get("source_contract", {}).get("ranges_by_symbol", {})
    observed_ranges = {
        symbol: (
            ranges.get(symbol, {}).get("start_period"),
            ranges.get(symbol, {}).get("end_period"),
            ranges.get(symbol, {}).get("period_count"),
        )
        for symbol in SYMBOLS
    }
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or claimed != _canonical_sha256(canonical)
        or _file_sha256(source_path) != reuse.get("source_design_file_sha256")
        or prior["design_sha256"] != reuse.get("source_design_canonical_sha256")
        or any(
            _canonical_sha256(prior[name]) != section_hashes.get(name)
            for name in (
                "causal_trigger_contract",
                "episode_contract",
                "cost_reference_contract",
                "uncertainty_contract",
                "authorization_gate",
            )
        )
        or observed_ranges != expected_ranges
        or tuple(design.get("source_contract", {}).get("symbols", ())) != SYMBOLS
        or any(
            design.get("governance", {}).get(name) is not False
            for name in (
                "protocol_change_permitted",
                "promotion_permitted",
                "profitability_claim_permitted",
                "trading_authority_permitted",
                "testnet_or_live_authority_permitted",
                "leverage_permitted",
                "model_training_permitted",
                "ai_evaluation_permitted",
            )
        )
    ):
        raise ValueError("Round 60 frozen replication design drifted")
    _validate_finite(design, "design")
    return design, prior


def _validate_certificate(
    path: Path, design: Mapping[str, object]
) -> tuple[dict[str, object], str, str]:
    certificate = _read_object(path, "Round 60 source certificate")
    canonical = dict(certificate)
    claimed = str(canonical.pop("source_certificate_sha256", ""))
    archives = certificate.get("archive_evidence", [])
    ranges = design["source_contract"]["ranges_by_symbol"]
    if (
        certificate.get("schema_version") != CERTIFICATE_SCHEMA
        or certificate.get("round") != ROUND
        or certificate.get("design_sha256") != design["design_sha256"]
        or claimed != _canonical_sha256(canonical)
        or tuple(certificate.get("symbols", ())) != SYMBOLS
        or certificate.get("persistent_zip_archive_created") is not False
        or len(archives) != len(SYMBOLS)
    ):
        raise ValueError("Round 60 source certificate identity drifted")
    by_symbol = {row.get("symbol"): row for row in archives}
    for symbol in SYMBOLS:
        contract = ranges[symbol]
        row = by_symbol.get(symbol, {})
        if (
            row.get("data_type") != "fundingRate"
            or row.get("first_period") != contract["start_period"]
            or row.get("last_period") != contract["end_period"]
            or row.get("period_count") != contract["period_count"]
            or not int(row.get("rows_read", 0)) > 0
            or not 1 <= int(row.get("minimum_interval_hours", 0))
            or not int(row.get("maximum_interval_hours", 9)) <= 8
            or any(
                len(str(row.get(key, ""))) != 64
                for key in (
                    "archive_identity_sha256",
                    "funding_row_stream_sha256",
                    "period_evidence_sha256",
                )
            )
        ):
            raise ValueError(f"Round 60 {symbol} source certificate drifted")
    _validate_finite(certificate, "certificate")
    return certificate, _file_sha256(path), claimed


def _gate_passed(metrics: Mapping[str, object], gate: Mapping[str, object]) -> bool:
    if int(metrics["episodes"]) < int(
        gate["minimum_nonoverlapping_episodes_per_symbol"]
    ):
        return False
    stress = metrics["cost_comparisons"]["stress_four_leg"]
    return (
        float(stress["positive_net_reference_fraction"])
        >= float(gate["minimum_stress_net_positive_fraction"])
        and float(stress["median_net_reference_bps"])
        > float(gate["median_stress_net_bps_strictly_above"])
        and float(stress["bootstrap_lower_95_mean_net_reference_bps"])
        > float(gate["bootstrap_lower_95_mean_stress_net_bps_strictly_above"])
    )


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design, protocol = _validate_design(arguments.design.resolve())
    certificate, certificate_file_sha, certificate_sha = _validate_certificate(
        arguments.certificate.resolve(), design
    )
    _validate_cost_references(protocol)
    if _git("status", "--porcelain"):
        raise ValueError("Round 60 runner requires a clean worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    cost_contract = protocol["cost_reference_contract"]
    costs = {
        "optimistic_futures_maker_only": float(
            cost_contract["optimistic_futures_maker_only_bps"]
        ),
        "repo_offline_four_leg_taker": float(
            cost_contract["repo_offline_four_leg_taker_bps"]
        ),
        "stress_four_leg": float(cost_contract["stress_four_leg_bps"]),
    }
    uncertainty = protocol["uncertainty_contract"]
    gate = protocol["authorization_gate"]
    ranges = design["source_contract"]["ranges_by_symbol"]
    archive_by_symbol = {row["symbol"]: row for row in certificate["archive_evidence"]}
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "design_sha256": design["design_sha256"],
        "implementation_commit": implementation_commit,
        "status": "running",
        "source": {
            "certificate_file_sha256": certificate_file_sha,
            "certificate_canonical_sha256": certificate_sha,
            "database_file": arguments.database.resolve().name,
            "symbols": list(SYMBOLS),
            "ranges_by_symbol": ranges,
        },
        "protocol_source": {
            "round": 59,
            "design_sha256": protocol["design_sha256"],
            "report_file_sha256": ROUND59_REPORT_FILE_SHA256,
            "report_canonical_sha256": ROUND59_REPORT_CANONICAL_SHA256,
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
    lookup: dict[tuple[str, int, str], dict[str, object]] = {}
    with _connect_read_only(arguments.database.resolve()) as connection:
        for symbol_index, symbol in enumerate(SYMBOLS):
            contract = ranges[symbol]
            source_view = {
                "source_contract": {
                    "start_period": contract["start_period"],
                    "end_period": contract["end_period"],
                    "periods_per_symbol": contract["period_count"],
                    "expected_rows": {symbol: archive_by_symbol[symbol]["rows_read"]},
                }
            }
            print(
                _canonical_json({"phase": "source-audit-start", "symbol": symbol}),
                flush=True,
            )
            rows, source_evidence = _load_verified_funding(
                connection,
                symbol=symbol,
                design=source_view,
                certificate=certificate,
            )
            cells: list[dict[str, object]] = []
            for trigger_index, trigger in enumerate(
                protocol["causal_trigger_contract"]["triggers_bps"]
            ):
                for horizon_index, horizon in enumerate(
                    protocol["episode_contract"]["holding_horizons_hours"]
                ):
                    episodes = _episodes(
                        rows, trigger=trigger, horizon_hours=int(horizon)
                    )
                    seed = (
                        int(uncertainty["seed"])
                        + 100 * symbol_index
                        + 10 * trigger_index
                        + horizon_index
                    )
                    metrics = _episode_metrics(
                        episodes, costs=costs, uncertainty=uncertainty, seed=seed
                    )
                    cell = {
                        "symbol": symbol,
                        "trigger_id": trigger["id"],
                        "trigger_operator": trigger["operator"],
                        "trigger_value_bps": trigger["value"],
                        "horizon_hours": horizon,
                        "bootstrap_seed": seed,
                        **metrics,
                        "symbol_gate_passed": _gate_passed(metrics, gate),
                    }
                    cells.append(cell)
                    lookup[(str(trigger["id"]), int(horizon), symbol)] = cell
            result = {
                "symbol": symbol,
                "source": source_evidence,
                "sign_transition": _sign_transition(rows),
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
    breadth: list[dict[str, object]] = []
    for trigger in protocol["causal_trigger_contract"]["triggers_bps"]:
        for horizon in protocol["episode_contract"]["holding_horizons_hours"]:
            passed_symbols = [
                symbol
                for symbol in SYMBOLS
                if lookup[(str(trigger["id"]), int(horizon), symbol)][
                    "symbol_gate_passed"
                ]
            ]
            breadth.append(
                {
                    "trigger_id": trigger["id"],
                    "horizon_hours": horizon,
                    "passed_symbols": passed_symbols,
                    "required_symbols": list(SYMBOLS),
                    "passed": tuple(passed_symbols) == SYMBOLS,
                }
            )
    passing_count = sum(bool(row["passed"]) for row in breadth)
    authorized = passing_count >= int(gate["passing_cell_count_required"])
    report["breadth_gates"] = breadth
    report["spot_history_ingestion_authorized"] = authorized
    report["status"] = (
        "full_history_support_passed_spot_ingestion_authorized"
        if authorized
        else "rejected_full_history_funding_persistence_support"
    )
    report["result"] = {
        "symbol_cells": len(lookup),
        "breadth_cells": len(breadth),
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
        / "docs/model-research/action-value/round-060-full-history-funding-replication-design.json",
    )
    parser.add_argument("--certificate", type=Path, required=True)
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
                "spot_history_ingestion_authorized": report[
                    "spot_history_ingestion_authorized"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
