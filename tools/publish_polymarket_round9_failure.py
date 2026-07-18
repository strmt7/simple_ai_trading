"""Publish deterministic Round 9 Polymarket admission-failure evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from html import escape
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = (
    ROOT / "data" / "polymarket-round9-confirmation-v4-20260716-152838Z.duckdb"
)
DEFAULT_PIPELINE_REPORT_SHA256 = (
    "1d3b1e0df05dbb4a7f5b9be9fe7b40fd03ba8f6f06bf90115851adb10efb4d8b"
)
CURRENT_DIR = ROOT / "docs" / "model-research" / "polymarket"
LATEST_DIR = CURRENT_DIR / "latest"
REPORT_PATH = CURRENT_DIR / "round-009-causal-action-value-failure-report.json"
INTEGRITY_PATH = LATEST_DIR / "publication-integrity.json"
TABLE_DIR = LATEST_DIR / "tables"
CHART_DIR = LATEST_DIR / "charts"
REPORT_SCHEMA_VERSION = "polymarket-round9-admission-failure-report-v1"
PUBLICATION_SCHEMA_VERSION = "polymarket-round9-failure-publication-v1"
EXPECTED_ACTION_CONTRACT_SHA256 = (
    "c8988fd548cff295800b977d6e6c92c39e9f2867b6c6e4b5f7e3d0b2b96f9800"
)
EXPECTED_RIDGE_CONTRACT_SHA256 = (
    "4b192e7f30af3e3d6e7dfb1b2b3342518e23de6d750b6b1cfd2334d87f2f5a12"
)
EXPECTED_FAILURE_REASONS = (
    "entry_confirmation_enters_excluded_close_window",
    "missing_entry_execution_book",
)
COLORS = ("#0F766E", "#2563EB", "#D97706", "#B42318", "#64748B")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _iso_utc(epoch_ms: int) -> str:
    return (
        datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _percentage(count: int, total: int) -> str:
    if total <= 0:
        raise ValueError("percentage denominator must be positive")
    return format(Decimal(count) * Decimal(100) / Decimal(total), ".6f")


def _bar_chart(
    *,
    title: str,
    subtitle: str,
    rows: Sequence[Mapping[str, object]],
    value_key: str,
    label_key: str,
    footer: str,
) -> str:
    width = 1280
    longest_label = max(len(str(row[label_key]).replace("_", " ")) for row in rows)
    left = 500 if longest_label > 36 else 360
    top = 150
    right = 1180
    row_height = 74
    height = top + row_height * len(rows) + 120
    maximum = max(int(row[value_key]) for row in rows)
    if maximum <= 0:
        raise ValueError("chart requires at least one positive value")
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#F8FAFC"/>',
        f'<text x="48" y="56" font-family="Segoe UI,Arial,sans-serif" font-size="28" font-weight="700" fill="#0F172A">{escape(title)}</text>',
        f'<text x="48" y="91" font-family="Segoe UI,Arial,sans-serif" font-size="16" fill="#475569">{escape(subtitle)}</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = int(row[value_key])
        bar_width = int((right - left) * value / maximum)
        label = str(row[label_key]).replace("_", " ")
        color = COLORS[index % len(COLORS)]
        body.extend(
            [
                f'<text x="48" y="{y + 27}" font-family="Segoe UI,Arial,sans-serif" font-size="16" fill="#1E293B">{escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{right - left}" height="38" rx="4" fill="#E2E8F0"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="38" rx="4" fill="{color}"/>',
                f'<text x="{left + 12}" y="{y + 26}" font-family="Segoe UI,Arial,sans-serif" font-size="15" font-weight="600" fill="#FFFFFF">{value:,}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="48" y="{height - 44}" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="#64748B">{escape(footer)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(body) + "\n"


def _query_rows(
    connection: duckdb.DuckDBPyConnection,
    pipeline_report_sha256: str,
) -> dict[str, object]:
    pipeline_row = connection.execute(
        """
        SELECT schema_version, contract_sha256, run_id, run_report_sha256,
               config_json, eligibility_sha256, batch_ids_json,
               action_dataset_sha256_json, action_count,
               classifier_eligible_count, positive_complete_count,
               category_counts_json, terminal_reason_counts_json,
               report_json, implementation_sha256,
               excluded_after_event_scope_count
        FROM polymarket_action_value_pipeline
        WHERE report_sha256 = ?
        """,
        [pipeline_report_sha256],
    ).fetchone()
    if pipeline_row is None:
        raise ValueError("pipeline report does not exist")
    (
        pipeline_schema,
        action_contract,
        run_id,
        run_report_sha256,
        config_json,
        eligibility_sha256,
        batch_ids_json,
        dataset_ids_json,
        action_count,
        classifier_eligible_count,
        positive_complete_count,
        category_counts_json,
        terminal_reason_counts_json,
        report_json,
        implementation_sha256,
        excluded_after_event_scope_count,
    ) = pipeline_row
    if action_contract != EXPECTED_ACTION_CONTRACT_SHA256:
        raise ValueError("unexpected Round 9 action contract")
    report = json.loads(str(report_json))
    claimed_report_sha256 = report.pop("report_sha256", None)
    if (
        claimed_report_sha256 != pipeline_report_sha256
        or _canonical_sha256(report) != pipeline_report_sha256
    ):
        raise ValueError("pipeline report content hash does not match")
    batch_ids = json.loads(str(batch_ids_json))
    dataset_ids = json.loads(str(dataset_ids_json))
    if len(batch_ids) != 47 or len(dataset_ids) != 47:
        raise ValueError("Round 9 pipeline does not contain 47 batches")

    terminal_counts = {
        str(key): int(value)
        for key, value in json.loads(str(terminal_reason_counts_json)).items()
    }
    category_counts = {
        str(key): int(value)
        for key, value in json.loads(str(category_counts_json)).items()
    }
    blocking_counts = {
        reason: terminal_counts.get(reason, 0)
        for reason in EXPECTED_FAILURE_REASONS
        if terminal_counts.get(reason, 0) > 0
    }
    if blocking_counts != {
        "entry_confirmation_enters_excluded_close_window": 12,
        "missing_entry_execution_book": 388,
    }:
        raise ValueError("Round 9 admission failure counts differ")

    claim = connection.execute(
        """
        SELECT schema_version, contract_sha256, dataset_sha256, state,
               report_sha256, failure_sha256, started_at_ms, completed_at_ms
        FROM polymarket_model_fit_claim
        WHERE experiment = 'round9_ridge' AND parent_sha256 = ?
        """,
        [pipeline_report_sha256],
    ).fetchone()
    if claim is None:
        raise ValueError("Round 9 ridge failure claim is missing")
    error_message = "unproven post-submission entry state:" + ",".join(
        f"{reason}:{blocking_counts[reason]}" for reason in sorted(blocking_counts)
    )
    expected_failure_sha256 = _canonical_sha256(
        {"error_type": "ValueError", "error_message": error_message}
    )
    if (
        claim[1] != EXPECTED_RIDGE_CONTRACT_SHA256
        or claim[3] != "failed"
        or claim[4] is not None
        or claim[5] != expected_failure_sha256
    ):
        raise ValueError("Round 9 ridge claim does not match the admission failure")

    connection.execute(
        "CREATE OR REPLACE TEMP TABLE selected_round9_dataset(dataset_sha256 VARCHAR PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT INTO selected_round9_dataset VALUES (?)",
        [(str(value),) for value in dataset_ids],
    )
    action_summary = connection.execute(
        """
        SELECT count(*), count(DISTINCT action_feature_sha256),
               sum(classifier_eligible::INTEGER), sum(positive_complete::INTEGER),
               sum(condition_blocked::INTEGER), sum(entry_filled::INTEGER),
               sum(exit_filled::INTEGER), min(decision_received_wall_ms),
               max(decision_received_wall_ms)
        FROM polymarket_action_value_row
        JOIN selected_round9_dataset USING (dataset_sha256)
        """
    ).fetchone()
    if action_summary is None:
        raise ValueError("Round 9 action summary is missing")
    if tuple(int(value) for value in action_summary[:7]) != (
        251892,
        251892,
        250587,
        13752,
        28438,
        222527,
        194089,
    ):
        raise ValueError("Round 9 action population differs")

    asset_rows = connection.execute(
        """
        SELECT asset, count(*), sum(classifier_eligible::INTEGER),
               sum(positive_complete::INTEGER),
               sum(CAST(net_quote AS DECIMAL(38, 18)))
        FROM polymarket_action_value_row
        JOIN selected_round9_dataset USING (dataset_sha256)
        GROUP BY asset ORDER BY asset
        """
    ).fetchall()
    outcome_rows = connection.execute(
        """
        SELECT outcome, count(*), sum(classifier_eligible::INTEGER),
               sum(positive_complete::INTEGER)
        FROM polymarket_action_value_row
        JOIN selected_round9_dataset USING (dataset_sha256)
        GROUP BY outcome ORDER BY outcome
        """
    ).fetchall()
    category_rows = connection.execute(
        """
        SELECT category, count(*)
        FROM polymarket_action_value_row
        JOIN selected_round9_dataset USING (dataset_sha256)
        GROUP BY category ORDER BY count(*) DESC, category
        """
    ).fetchall()
    terminal_rows = connection.execute(
        """
        SELECT terminal_reason, count(*)
        FROM polymarket_action_value_row
        JOIN selected_round9_dataset USING (dataset_sha256)
        GROUP BY terminal_reason ORDER BY count(*) DESC, terminal_reason
        """
    ).fetchall()
    blocking_rows = connection.execute(
        """
        SELECT terminal_reason, count(*), count(DISTINCT condition_id),
               count(DISTINCT decision_received_wall_ms),
               count(DISTINCT asset), count(DISTINCT outcome)
        FROM polymarket_action_value_row
        JOIN selected_round9_dataset USING (dataset_sha256)
        WHERE terminal_reason IN (
            'missing_entry_execution_book',
            'entry_confirmation_enters_excluded_close_window'
        )
        GROUP BY terminal_reason ORDER BY terminal_reason
        """
    ).fetchall()
    batch_rows = connection.execute(
        """
        SELECT batch_id, batch_sha256, action_dataset_sha256,
               feature_dataset_sha256, group_starts_json,
               condition_ids_json
        FROM polymarket_action_value_batch
        WHERE implementation_sha256 = ?
        ORDER BY group_starts_json
        """,
        [implementation_sha256],
    ).fetchall()
    if len(batch_rows) != 47 or {str(row[0]) for row in batch_rows} != set(batch_ids):
        raise ValueError("Round 9 batch identity differs from pipeline")
    condition_ids = {
        str(value) for row in batch_rows for value in json.loads(str(row[5]))
    }
    if len(condition_ids) != 141:
        raise ValueError("Round 9 condition breadth differs")
    return {
        "pipeline_schema_version": str(pipeline_schema),
        "action_contract_sha256": str(action_contract),
        "run_id": str(run_id),
        "run_report_sha256": str(run_report_sha256),
        "config": json.loads(str(config_json)),
        "eligibility_sha256": str(eligibility_sha256),
        "batch_ids": tuple(str(value) for value in batch_ids),
        "dataset_ids": tuple(str(value) for value in dataset_ids),
        "implementation_sha256": str(implementation_sha256),
        "excluded_after_event_scope_count": int(excluded_after_event_scope_count),
        "action_count": int(action_count),
        "classifier_eligible_count": int(classifier_eligible_count),
        "positive_complete_count": int(positive_complete_count),
        "category_counts": category_counts,
        "terminal_counts": terminal_counts,
        "blocking_counts": blocking_counts,
        "claim": claim,
        "error_message": error_message,
        "action_summary": action_summary,
        "asset_rows": asset_rows,
        "outcome_rows": outcome_rows,
        "category_rows": category_rows,
        "terminal_rows": terminal_rows,
        "blocking_rows": blocking_rows,
        "batch_rows": batch_rows,
        "condition_count": len(condition_ids),
    }


def publish(database: Path, pipeline_report_sha256: str) -> str:
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        evidence = _query_rows(connection, pipeline_report_sha256)
    finally:
        connection.close()

    total = int(evidence["action_count"])
    category_rows = [
        {
            "category": str(name),
            "action_count": int(count),
            "percentage_of_actions": _percentage(int(count), total),
        }
        for name, count in evidence["category_rows"]
    ]
    terminal_rows = [
        {
            "terminal_reason": str(name),
            "action_count": int(count),
            "percentage_of_actions": _percentage(int(count), total),
        }
        for name, count in evidence["terminal_rows"]
    ]
    asset_rows = [
        {
            "asset": str(asset),
            "action_count": int(count),
            "classifier_eligible_count": int(eligible),
            "positive_complete_count": int(positive),
            "positive_complete_rate_of_eligible": _percentage(
                int(positive), int(eligible)
            ),
            "all_candidate_net_quote": format(Decimal(net), "f"),
        }
        for asset, count, eligible, positive, net in evidence["asset_rows"]
    ]
    outcome_rows = [
        {
            "outcome": str(outcome),
            "action_count": int(count),
            "classifier_eligible_count": int(eligible),
            "positive_complete_count": int(positive),
        }
        for outcome, count, eligible, positive in evidence["outcome_rows"]
    ]
    blocking_rows = [
        {
            "terminal_reason": str(reason),
            "action_count": int(count),
            "condition_count": int(conditions),
            "decision_timestamp_count": int(timestamps),
            "asset_count": int(assets),
            "outcome_count": int(outcomes),
        }
        for reason, count, conditions, timestamps, assets, outcomes in evidence[
            "blocking_rows"
        ]
    ]

    disposition_path = TABLE_DIR / "round9-action-disposition.csv"
    terminal_path = TABLE_DIR / "round9-terminal-reasons.csv"
    asset_path = TABLE_DIR / "round9-asset-population.csv"
    outcome_path = TABLE_DIR / "round9-outcome-population.csv"
    blocking_path = TABLE_DIR / "round9-admission-blockers.csv"
    _write_csv(
        disposition_path,
        ("category", "action_count", "percentage_of_actions"),
        category_rows,
    )
    _write_csv(
        terminal_path,
        ("terminal_reason", "action_count", "percentage_of_actions"),
        terminal_rows,
    )
    _write_csv(
        asset_path,
        (
            "asset",
            "action_count",
            "classifier_eligible_count",
            "positive_complete_count",
            "positive_complete_rate_of_eligible",
            "all_candidate_net_quote",
        ),
        asset_rows,
    )
    _write_csv(
        outcome_path,
        (
            "outcome",
            "action_count",
            "classifier_eligible_count",
            "positive_complete_count",
        ),
        outcome_rows,
    )
    _write_csv(
        blocking_path,
        (
            "terminal_reason",
            "action_count",
            "condition_count",
            "decision_timestamp_count",
            "asset_count",
            "outcome_count",
        ),
        blocking_rows,
    )

    action_summary = evidence["action_summary"]
    start_utc = _iso_utc(int(action_summary[7]))
    end_utc = _iso_utc(int(action_summary[8]))
    subtitle = (
        f"Real BTC/ETH/SOL evidence | {start_utc} to {end_utc} | "
        "250 ms decisions | displayed depth, fees, 500 ms local latency"
    )
    disposition_chart = CHART_DIR / "round9-action-disposition.svg"
    blocker_chart = CHART_DIR / "round9-admission-failure.svg"
    _atomic_text(
        disposition_chart,
        _bar_chart(
            title="Round 9 action disposition",
            subtitle=subtitle,
            rows=category_rows,
            value_key="action_count",
            label_key="category",
            footer=(
                "Counts cover all hypothetical Up/Down candidates. They are not "
                "executed strategy trades or ROI evidence."
            ),
        ),
    )
    _atomic_text(
        blocker_chart,
        _bar_chart(
            title="Round 9 pre-fit admission failure",
            subtitle=(
                "Any nonzero post-submission unknown state rejects the complete "
                "frozen Ridge experiment"
            ),
            rows=blocking_rows,
            value_key="action_count",
            label_key="terminal_reason",
            footer=(
                "No rows were censored or relabeled. Ridge, MLP, AI uplift, ROI, "
                "and drawdown metrics do not exist for this round."
            ),
        ),
    )

    claim = evidence["claim"]
    batch_root = _canonical_sha256(
        [
            {
                "batch_id": str(row[0]),
                "batch_sha256": str(row[1]),
                "action_dataset_sha256": str(row[2]),
                "feature_dataset_sha256": str(row[3]),
                "group_starts": json.loads(str(row[4])),
            }
            for row in evidence["batch_rows"]
        ]
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "round": 9,
        "status": "failed_before_model_fit",
        "failure_stage": "clear_label_admission_after_opaque_claim",
        "source": {
            "database_file": database.name,
            "run_id": evidence["run_id"],
            "run_report_sha256": evidence["run_report_sha256"],
            "pipeline_report_sha256": pipeline_report_sha256,
            "pipeline_schema_version": evidence["pipeline_schema_version"],
            "continuity_eligibility_sha256": evidence["eligibility_sha256"],
            "action_contract_sha256": evidence["action_contract_sha256"],
            "ridge_contract_sha256": EXPECTED_RIDGE_CONTRACT_SHA256,
            "action_implementation_sha256": evidence["implementation_sha256"],
            "ordered_action_batch_root_sha256": batch_root,
        },
        "scope": {
            "assets": ["BTC", "ETH", "SOL"],
            "market_group_count": len(evidence["batch_rows"]),
            "condition_count": evidence["condition_count"],
            "decision_start_utc": start_utc,
            "decision_end_utc": end_utc,
            "decision_cadence_ms": evidence["config"]["action"]["decision_cadence_ms"],
            "per_leg_submission_latency_ms": evidence["config"]["action"][
                "per_leg_submission_latency_ms"
            ],
            "holding_period_ms": evidence["config"]["action"]["holding_period_ms"],
            "maximum_post_target_execution_observation_delay_ms": evidence["config"][
                "action"
            ]["maximum_post_target_execution_observation_delay_ms"],
            "minimum_remaining_market_time_ms": evidence["config"]["action"][
                "minimum_remaining_market_time_ms"
            ],
        },
        "population": {
            "feature_row_count": total // 2,
            "action_count": total,
            "distinct_action_feature_sha256_count": int(action_summary[1]),
            "classifier_eligible_count": evidence["classifier_eligible_count"],
            "positive_complete_count": evidence["positive_complete_count"],
            "condition_blocked_count": int(action_summary[4]),
            "entry_filled_count": int(action_summary[5]),
            "exit_filled_count": int(action_summary[6]),
            "excluded_after_event_scope_count": evidence[
                "excluded_after_event_scope_count"
            ],
            "categories": category_rows,
            "terminal_reasons": terminal_rows,
            "assets": asset_rows,
            "outcomes": outcome_rows,
        },
        "admission_failure": {
            "error_type": "ValueError",
            "error_message": evidence["error_message"],
            "blocking_reasons": blocking_rows,
            "claim_schema_version": str(claim[0]),
            "claim_dataset_sha256": str(claim[2]),
            "claim_state": str(claim[3]),
            "failure_sha256": str(claim[5]),
            "started_at_utc": _iso_utc(int(claim[6])),
            "completed_at_utc": _iso_utc(int(claim[7])),
            "retry_permitted": False,
        },
        "model_evidence": {
            "ridge_fit_reached": False,
            "ridge_report_sha256": None,
            "ridge_score": None,
            "mlp_authorized": False,
            "mlp_score": None,
            "ai_uplift_authorized": False,
            "ai_score": None,
            "profitability_claim": False,
            "roi_claim": False,
            "drawdown_claim": False,
            "paper_authority": False,
            "trading_authority": False,
        },
        "interpretation": {
            "candidate_net_warning": (
                "all_candidate_net_quote sums every hypothetical candidate and is "
                "not a sequential strategy return"
            ),
            "failure_meaning": (
                "The public-feed replay could not prove every post-submission entry "
                "state within the frozen confirmation window. Censoring or treating "
                "those rows as no-fill would invent evidence."
            ),
            "required_next_step": (
                "Use a separately frozen development round with explicit unknown-state "
                "worst-case capital loss and authenticated lifecycle telemetry before "
                "any independent confirmation."
            ),
        },
    }
    report["report_canonical_sha256"] = _canonical_sha256(report)
    _write_json(REPORT_PATH, report)

    artifact_paths = (
        REPORT_PATH,
        disposition_path,
        terminal_path,
        asset_path,
        outcome_path,
        blocking_path,
        disposition_chart,
        blocker_chart,
    )
    integrity: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "round": 9,
        "status": "failed_before_model_fit",
        "pipeline_report_sha256": pipeline_report_sha256,
        "artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in artifact_paths
        ],
        "profitability_claim": False,
        "trading_authority": False,
    }
    integrity["publication_sha256"] = _canonical_sha256(integrity)
    _write_json(INTEGRITY_PATH, integrity)
    return str(integrity["publication_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--pipeline-report-sha256", default=DEFAULT_PIPELINE_REPORT_SHA256
    )
    args = parser.parse_args()
    publication_sha256 = publish(
        args.database.resolve(), str(args.pipeline_report_sha256)
    )
    print(f"Round 9 failure publication: {publication_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
