from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_round13 import polymarket_round13_scenarios
from simple_ai_trading import polymarket_round13_evaluation as evaluation
from simple_ai_trading.polymarket_round13_publication import (
    publish_round13_evaluation,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _bootstrap() -> dict[str, object]:
    return {
        "samples": 10_000,
        "block_length_groups": 12,
        "lower_95_mean_group_utility_quote": 0.01,
        "median_mean_group_utility_quote": 0.02,
        "upper_95_mean_group_utility_quote": 0.03,
        "bootstrap_samples_sha256": "a" * 64,
        "prng": "splitmix64",
        "seed": 13_013,
    }


def _metric(scenario: str, policy: str) -> dict[str, object]:
    equity = [
        {
            "event_start_ms": 1_700_000_000_000 + index * 300_000,
            "group_utility_quote": 0.1,
            "cumulative_utility_quote": (index + 1) * 0.1,
            "drawdown_quote": 0.0,
        }
        for index in range(160)
    ]
    conditions = [
        {
            "condition_id": f"{scenario}-{policy}-{index}",
            "asset": ("BTC", "ETH", "SOL")[index % 3],
            "event_start_ms": 1_700_000_000_000 + index * 300_000,
            "utility_quote": 0.1,
        }
        for index in range(160)
    ]
    return {
        "scenario": scenario,
        "policy": policy,
        "condition_count": 480,
        "attempt_count": 60,
        "simulated_filled_conditions": 45,
        "simulated_fills_per_asset": {"BTC": 15, "ETH": 15, "SOL": 15},
        "simulated_no_fill_attempts": 10,
        "not_submitted_attempts": 5,
        "unknown_after_submit_conditions": 0,
        "attempt_states": {"simulated_fill": 45, "simulated_no_fill": 10},
        "abstentions_by_reason": {"no_positive_conservative_edge": 100},
        "wins": 32,
        "losses": 13,
        "total_utility_quote": 16.0,
        "mean_condition_utility_quote": 16 / 480,
        "median_condition_utility_quote": 0.01,
        "median_simulated_filled_condition_utility_quote": 0.2,
        "per_asset_utility_quote": {"BTC": 5.0, "ETH": 5.5, "SOL": 5.5},
        "allocated_capital_quote": 1000.0,
        "maximum_group_entry_exposure_quote": 12.0,
        "capital_deployed_quote": 120.0,
        "maximum_group_exposure_fraction": 0.012,
        "market_horizon_capital_time_quote_seconds": 10_000.0,
        "turnover_quote": 120.0,
        "maximum_drawdown_quote": 1.0,
        "median_positive_group_profit_quote": 0.1,
        "drawdown_limit_quote": 0.2,
        "bootstrap": _bootstrap(),
        "equity": equity,
        "per_condition_utility": conditions,
        "gate_reasons_without_control": [],
        "gate_without_control_passed": True,
    }


def _score() -> dict[str, object]:
    bins = [
        {
            "bin": index,
            "lower_probability": index / 10,
            "upper_probability": (index + 1) / 10,
            "count": 16,
            "mean_probability": (index + 0.5) / 10,
            "observed_frequency": (index + 0.45) / 10,
        }
        for index in range(10)
    ]
    return {
        "count": 160,
        "log_loss": 0.6,
        "brier_score": 0.2,
        "calibration": {
            "available": True,
            "intercept": 0.0,
            "slope": 1.0,
            "iterations": 3,
        },
        "reliability_bins": bins,
    }


def _report() -> dict[str, object]:
    scenarios: dict[str, object] = {}
    for frozen in polymarket_round13_scenarios():
        treatment = _metric(frozen.name, "calibrated")
        control = _metric(frozen.name, "raw_market_prior")
        paired = [
            {
                "condition_id": f"condition-{index}",
                "asset": ("BTC", "ETH", "SOL")[index % 3],
                "event_start_ms": 1_700_000_000_000 + index * 300_000,
                "treatment_utility_quote": 0.1,
                "control_utility_quote": 0.05,
                "difference_quote": 0.05,
            }
            for index in range(160)
        ]
        treatment.update(
            {
                "control_comparison": {
                    "treatment_minus_control_total_utility_quote": 8.0,
                    "non_tied_condition_count": 160,
                    "treatment_minus_control_bootstrap": _bootstrap(),
                    "per_condition": paired,
                },
                "gate_reasons": [],
                "gate_passed": True,
            }
        )
        scenarios[frozen.name] = {
            "calibrated_treatment": treatment,
            "raw_market_prior_control": control,
            "no_trade_control_total_utility_quote": 0.0,
        }
    score = _score()
    report_without_hash: dict[str, object] = {
        "schema_version": evaluation.POLYMARKET_ROUND13_EVALUATION_SCHEMA_VERSION,
        "round": 13,
        "contract_sha256": "b" * 64,
        "run_id": "run",
        "run_report_sha256": "c" * 64,
        "capture_manifest_sha256": "d" * 64,
        "pipeline_report_sha256": "e" * 64,
        "scenario_dataset_sha256": ["f" * 64],
        "resolution_evidence_sha256": ["1" * 64],
        "resolution_finalization": {
            "run_id": "run",
            "status": "complete",
            "market_count": 480,
            "finalized_count": 480,
            "newly_finalized_count": 480,
            "pending_condition_ids": [],
            "resolution_ids": ["2" * 64],
            "poll_count": 1,
            "integrity_prevalidated": True,
        },
        "opened_before_resolution_query_at_ms": 1_700_000_000_001,
        "created_at_ms": 1_700_000_000_002,
        "utc_span_ms": {
            "start": 1_700_000_000_000,
            "end": 1_700_048_000_000,
        },
        "data": {
            "independent_synchronized_groups": 160,
            "resolved_conditions": 480,
            "resolved_markets_per_asset": {"BTC": 160, "ETH": 160, "SOL": 160},
            "outcome_classes_per_asset": {
                "BTC": ["Down", "Up"],
                "ETH": ["Down", "Up"],
                "SOL": ["Down", "Up"],
            },
            "real_data_only": True,
            "labels_opened_after_claim": True,
        },
        "allocated_confirmation_capital_quote": "1000",
        "execution_scenarios": [
            item.asdict() for item in polymarket_round13_scenarios()
        ],
        "executable_evaluation_gates": {"minimum_synchronized_event_groups": 160},
        "proper_scores": {
            "calibrated": {
                "pooled": score,
                "per_asset": {asset: score for asset in ("BTC", "ETH", "SOL")},
            },
            "raw_market_prior": {
                "pooled": score,
                "per_asset": {asset: score for asset in ("BTC", "ETH", "SOL")},
            },
            "pooled_difference": {"log_loss": 0.0, "brier_score": 0.0},
        },
        "scenarios": scenarios,
        "primary_gate_passed": True,
        "all_stress_gates_passed": True,
        "confirmation_passed": True,
        "after_cost_edge_confirmed": True,
        "settlement_overhead_measured": False,
        "authenticated_lifecycle_proven": False,
        "annualized_roi_available": False,
        "profitability_claim": False,
        "paper_authority": False,
        "live_trading_authority": False,
        "ai_edge_claim": False,
    }
    return {**report_without_hash, "report_sha256": _sha(report_without_hash)}


def _store_report(store: PolymarketEvidenceStore, report: dict[str, object]) -> None:
    evaluation._ensure_evaluation_tables(store)
    store.connect().execute(
        "INSERT INTO polymarket_round13_evaluation_report VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            report["report_sha256"],
            report["schema_version"],
            report["contract_sha256"],
            report["run_id"],
            report["pipeline_report_sha256"],
            report["created_at_ms"],
            _canonical(report),
        ],
    )


def test_round13_publication_is_deterministic_and_table_backed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "publication.duckdb"
    root = tmp_path / "research"
    report = _report()
    with PolymarketEvidenceStore(database) as store:
        _store_report(store, report)
        first = publish_round13_evaluation(
            store,
            report_sha256=str(report["report_sha256"]),
            research_root=root,
        )
        second = publish_round13_evaluation(
            store,
            report_sha256=str(report["report_sha256"]),
            research_root=root,
        )

    assert first.manifest_sha256 == second.manifest_sha256
    assert (root / "latest/charts/round13-treatment-control.svg").is_file()
    assert (root / "latest/charts/round13-equity-drawdown.svg").is_file()
    with (root / "latest/tables/round13-scenario-summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert rows[0]["assets"] == "BTC/ETH/SOL"
    assert rows[0]["total_utility_quote"] == "16"
    manifest = json.loads(
        (root / "latest/publication-integrity.json").read_text(encoding="utf-8")
    )
    claimed = manifest.pop("manifest_sha256")
    assert claimed == _sha(manifest)
    assert manifest["manual_chart_edits_permitted"] is False


def test_round13_publication_rejects_tampered_history(tmp_path: Path) -> None:
    database = tmp_path / "tamper.duckdb"
    root = tmp_path / "research"
    report = _report()
    with PolymarketEvidenceStore(database) as store:
        _store_report(store, report)
        publish_round13_evaluation(
            store,
            report_sha256=str(report["report_sha256"]),
            research_root=root,
        )
        progress = root / "latest/tables/optimization-progress.csv"
        progress.write_text(progress.read_text(encoding="utf-8") + "tampered\n")

        with pytest.raises(ValueError, match="history table hash differs"):
            publish_round13_evaluation(
                store,
                report_sha256=str(report["report_sha256"]),
                research_root=root,
            )
