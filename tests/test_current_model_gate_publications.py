from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research"
ROUND29 = (
    RESEARCH / "polymarket" / "round-029-stage1-readiness-adjudication-2026-08-23.json"
)
ROUND76 = (
    RESEARCH / "action-value" / "round-076-round75-source-gate-adjudication-v1.json"
)
MODEL_DEV = RESEARCH / "model-dev-three-way-audit-2026-08-23.json"
LATEST_STATUS = RESEARCH / "action-value" / "latest-status"


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


def _bound(path: Path, digest_field: str = "artifact_sha256") -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    claimed = value.pop(digest_field)
    assert claimed == _canonical_sha256(value)
    value[digest_field] = claimed
    return value


def test_round29_readiness_adjudication_blocks_incomplete_stage1_without_targets() -> (
    None
):
    report = _bound(ROUND29)

    assert report["decision"] == {
        "new_exposure_authority": False,
        "recovery_of_slot_b_wal_can_repair_campaign": False,
        "round29_selection_permitted": False,
        "status": "blocked_incomplete_stage1_campaign",
        "target_or_model_access_permitted": False,
    }
    assert all(value is False for value in report["authority"].values())
    assert all(value is False for value in report["gates"].values())
    assert report["audit_method"]["database_files_opened"] is False
    assert report["audit_method"]["financial_outcomes_accessed"] is False
    assert report["observed_primary_slots"]["stage1-a"]["terminal_result_present"]
    assert not report["observed_primary_slots"]["stage1-b"]["terminal_result_present"]
    assert report["observed_primary_slots"]["stage1-b"]["wal_present"]
    assert not report["observed_primary_slots"]["stage1-c"]["database_present"]
    assert (
        report["cross_regime_boundary"]["profit_in_every_future_regime_guaranteed"]
        is False
    )


def test_round76_adjudication_blocks_candidate_before_implementation() -> None:
    report = _bound(ROUND76)

    assert report["decision"] == {
        "candidate_hypothesis_rejected_by_results": False,
        "candidate_implementation_permitted": False,
        "candidate_training_permitted": False,
        "round75_data_permitted_for_round76": False,
        "status": "blocked_before_implementation_by_round75_source_gate",
    }
    assert all(value is False for value in report["authority"].values())
    assert all(value is False for value in report["evidence_boundary"].values())
    assert report["round75_terminal_facts"]["tuning_admitted_epochs"] == 0
    assert report["round75_terminal_facts"]["test_admitted_epochs"] == 0
    assert report["round75_terminal_facts"]["wal_bytes"] == 9_431_058


def test_model_dev_three_way_audit_forbids_bulk_integration() -> None:
    report = _bound(MODEL_DEV)

    assert report["baseline"]["main_descends_from_frozen_commit"] is True
    assert report["three_way_counts"] == {
        "already_integrated_exact": 214,
        "deletion_integrated": 2,
        "diverged_three_way": 25,
        "unpublished_tracked_only": 9,
        "unpublished_unique": 20,
        "untracked_collision": 47,
    }
    assert report["content_review"]["frozen_only_top_level_source_symbols"] == 0
    assert report["decision"]["bulk_integration_permitted"] is False
    assert report["decision"]["frozen_worktree_modification_permitted"] is False
    assert all(value is False for value in report["authority"].values())


def test_latest_status_is_source_bound_and_contains_no_invented_later_metrics() -> None:
    status = _bound(LATEST_STATUS / "status.json", "status_sha256")
    assert status["current_round"] == 76
    assert status["latest_completed_model_evaluation"] == {
        "accepted_market_edge": False,
        "round": 72,
        "status": "rejected",
    }
    assert status["later_round_status"] == {
        "73": "invalidated_before_model",
        "74": "terminal_campaign_quota_failed_no_model",
        "75": "rejected_incomplete_campaign",
        "76": "blocked_by_round75_source_gate",
    }
    assert all(value is False for value in status["claims"].values())
    assert status["cross_regime_policy"]["future_profit_or_no_loss_guaranteed"] is False
    publisher = status["publisher"]
    assert (
        hashlib.sha256((ROOT / publisher["path"]).read_bytes()).hexdigest()
        == (publisher["file_sha256"])
    )

    with (LATEST_STATUS / "progress.csv").open(
        "r", encoding="ascii", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["round"]) for row in rows] == list(range(1, 77))
    for row in rows[72:]:
        assert row["selected_signals"] == "0"
        assert row["executable_trades"] == "0"
        assert row["direction_auc"] == ""
        assert row["spearman_ic"] == ""
        assert row["mean_gross_bps"] == ""
        assert row["mean_net_bps"] == ""

    graph = (LATEST_STATUS / "research-progress.svg").read_text(encoding="ascii")
    assert "Rounds 73 through 76 produced no model or profitability result" in graph
    assert "candidate blocked before implementation" in graph
    readme = (LATEST_STATUS / "README.md").read_text(encoding="ascii")
    assert "No accepted market edge or trading authority" in readme
    assert status["status_sha256"] in readme
