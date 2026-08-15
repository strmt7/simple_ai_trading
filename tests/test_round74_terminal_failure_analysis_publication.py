from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/model-research/action-value"
ARTIFACT = RESEARCH / "round-074-terminal-failure-analysis-2026-08-10.json"
TERMINAL = RESEARCH / "round-074-terminal-campaign-outcome-v2-2026-08-10.json"
CSV = RESEARCH / "round-074-terminal-slot-coverage-2026-08-10.csv"
CHART = RESEARCH / "round-074-terminal-slot-coverage-2026-08-10.svg"
FILE_SHA256 = "86522ac242035254e63c97820e90093a864a3c90cd0a62210ebcb2457b1b694b"
ARTIFACT_SHA256 = "f03d8b885318c8a783b251f6f38368808480219a2fe0c96147d208f0dd671da1"
CSV_SHA256 = "ec060e0bf787426cb5a68921a11cf42073eee2f967a676914ffc87528664b69c"
CHART_SHA256 = "fec5ee5a718dfcf001b2253574499fc9483100cd6f76d6ba7e820bfd47f7c937"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return [value] if isinstance(value, str) else []


def test_terminal_failure_analysis_is_source_bound_and_non_authorizing() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    claimed = value.pop("artifact_sha256")
    assert claimed == ARTIFACT_SHA256
    assert claimed == _canonical_sha256(value)
    assert value["schema_version"] == "round-074-terminal-failure-analysis-v1"
    sources = value["source_bindings"]
    assert (
        sources["terminal_outcome_file_sha256"]
        == hashlib.sha256(TERMINAL.read_bytes()).hexdigest()
    )
    assert sources["slot_csv_file_sha256"] == CSV_SHA256
    assert sources["chart_file_sha256"] == CHART_SHA256
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == CSV_SHA256
    assert hashlib.sha256(CHART.read_bytes()).hexdigest() == CHART_SHA256
    assert value["diagnosis"] == {
        "basis": (
            "Most never-started slots are concentrated in two long preregistered "
            "scheduling gaps; terminal evidence does not identify the exact "
            "host-level cause."
        ),
        "classification": "campaign_continuity_failure_indicated",
        "exact_host_cause_established": False,
        "market_feed_failure_established_as_primary_cause": False,
        "model_architecture_evaluated": False,
        "predictive_edge_evaluated": False,
        "profitability_evaluated": False,
    }
    assert value["scope"] == {
        "credentials_used": False,
        "live_trading_authority": False,
        "model_training_performed": False,
        "orders_submitted": False,
        "paper_trading_authority": False,
        "source_database_access": "not_opened",
        "target_data_accessed": False,
    }
    assert not any(Path(text).is_absolute() for text in _strings(value))


def test_terminal_failure_analysis_reconciles_all_720_slot_rows() -> None:
    with CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 720
    assert [int(row["slot_ordinal"]) for row in rows] == list(range(720))
    assert Counter(row["status"] for row in rows) == {
        "admitted": 460,
        "missed": 238,
        "transport_excluded": 22,
    }
    assert Counter(row["failure_classification"] for row in rows) == {
        "admitted": 460,
        "captured_fresh_audit_failed": 1,
        "captured_stored_report_missing": 8,
        "captured_unsupported_terminal_class": 12,
        "empty_supervisor_output": 1,
        "never_started_no_slot_directory": 215,
        "prior_audit_terminal_class_failed": 1,
        "transport_excluded": 22,
    }
    assert rows[13]["status"] == "admitted"
    assert rows[13]["evidence_kind"] == "supplemental_adjudication"
    assert rows[486]["failure_classification"] == "empty_supervisor_output"
    assert int(rows[513]["cumulative_training_eligible_anchor_ns"]) == (
        233_306_813_789_700
    )


def test_terminal_failure_analysis_quantifies_without_overclaiming_cause() -> None:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    analysis = value["terminal_slot_analysis"]
    assert analysis["slot_directory_count"] == 505
    assert analysis["attempted_missed_count"] == 23
    assert analysis["major_gap_slot_count"] == 212
    assert math.isclose(
        analysis["major_gap_fraction_of_never_started_slots"],
        212 / 215,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    major = [
        (row["first_slot_ordinal"], row["last_slot_ordinal"], row["slot_count"])
        for row in analysis["never_started_contiguous_blocks"]
        if row["major_gap"]
    ]
    assert major == [(58, 82, 25), (151, 337, 187)]
    quota = value["training_quota"]
    assert quota["observed_eligible_anchor_ns"] == 233_306_813_789_700
    assert quota["required_eligible_anchor_ns"] == 394_740_000_000_000
    assert quota["deficit_eligible_anchor_ns"] == 161_433_186_210_300
    assert quota["quota_passed"] is False
