from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "logical-implication-parity-snapshot-v1-2026-08-25.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_logical_parity_artifact_reconstructs_claim_and_source_binding() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    claim = report["result_claim"]
    canonical = json.dumps(
        claim,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == report["result_claim_sha256"]

    implementation = report["implementation"]
    for prefix in ("tool", "module", "source_helper"):
        assert (
            _sha256(ROOT / implementation[f"{prefix}_path"])
            == implementation[f"{prefix}_sha256"]
        )


def test_logical_parity_artifact_reconstructs_aggregate_counts() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    groups = report["groups"]
    claim = report["result_claim"]
    assert claim["threshold_group_count"] == sum(
        group["group_type"] == "threshold" for group in groups
    )
    assert claim["deadline_group_count"] == sum(
        group["group_type"] == "deadline" for group in groups
    )
    assert claim["eligible_event_count"] == len({group["event_id"] for group in groups})
    for field in (
        "evaluated_pair_count",
        "executable_pair_count",
        "gross_positive_pair_count",
        "after_cost_positive_pair_count",
    ):
        assert claim[field] == sum(group[field] for group in groups)
    assert claim["gross_positive_pair_count"] == 0
    assert claim["after_cost_positive_pair_count"] == 0
    assert claim["accepted_edge"] is False
    assert claim["promotion_status"] == "rejected_unpromoted"


def test_logical_parity_artifact_records_every_fail_closed_exclusion() -> None:
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    evidence = report["source_evidence"]
    for singular, plural in (
        ("missing_market_end_date", "missing_market_end_dates"),
        ("missing_fee_schedule", "missing_fee_schedules"),
        ("clob_identity_mismatch", "clob_identity_mismatches"),
    ):
        assert evidence[f"excluded_{singular}_count"] == len(
            evidence[f"excluded_{plural}"]
        )
        assert evidence[f"excluded_{singular}_count"] > 0
    assert report["safety"] == {
        "credentials_used": False,
        "orders_placed": False,
        "positive_gross_snapshot_would_require_sequential_confirmation": True,
        "public_books_prove_fills": False,
    }
