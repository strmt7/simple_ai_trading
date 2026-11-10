"""A valid file hash alone must not admit stale or contradictory ledger state."""

import copy
import json
import sys

import pytest

from tools.verify_research_registry_binding import _canonical_hash, main, verify_binding


def seal(registry, audit):
    registry["result_sha256"] = _canonical_hash(registry)
    audit["source_binding"]["registry_result_sha256"] = registry["result_sha256"]
    audit["result_sha256"] = _canonical_hash(audit)


def ledgers():
    registry = {
        "accepted_edge_count": 2,
        "prioritized_hypotheses": [{"priority_rank": 1}, {"priority_rank": 2}],
        "terminal_do_not_repeat": [{"family": "consumed-window"}],
    }
    audit = {
        "source_binding": {"accepted_edge_count": 2},
        "decision": {
            "accepted_scope_count": 2,
            "stable_current_account_qualified_after_all_cost_edge_count": 0,
        },
        "classification": [{"count": 2, "edge_ordinals": [1, 2]}],
    }
    seal(registry, audit)
    return registry, audit


def test_valid_binding_reports_census_without_promoting_financial_claims():
    registry, audit = ledgers()
    before = copy.deepcopy((registry, audit))
    result = verify_binding(registry, audit)
    assert (
        result["observations"],
        result["hypotheses"],
        result["scoped_mechanisms"],
    ) == (1, 2, 2)
    assert result["recorded_qualified_stable_edges"] == 0
    assert not result["profitability_qualification_supplied_by_verifier"]
    assert (registry, audit) == before


def test_separately_valid_hashes_cannot_admit_stale_cross_binding():
    registry, audit = ledgers()
    registry["terminal_do_not_repeat"].append({"family": "new-window"})
    registry["result_sha256"] = _canonical_hash(registry)
    with pytest.raises(ValueError, match="different registry revision"):
        verify_binding(registry, audit)


@pytest.mark.parametrize("stale", [False, True])
def test_cli_reads_selected_ledgers_and_reports_binding_failure(
    tmp_path, monkeypatch, capsys, stale
):
    registry, audit = ledgers()
    if stale:
        registry["terminal_do_not_repeat"].append({"family": "new-window"})
        registry["result_sha256"] = _canonical_hash(registry)
    registry_path, audit_path = tmp_path / "registry.json", tmp_path / "audit.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--registry", str(registry_path), "--audit", str(audit_path)],
    )
    assert main() == int(stale)
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is not stale
    if stale:
        assert report["reason"] == "audit binds a different registry revision"


@pytest.mark.parametrize("target", ["registry", "audit"])
def test_content_change_without_new_hash_rejects(target):
    registry, audit = ledgers()
    (registry if target == "registry" else audit)["injected"] = "changed"
    with pytest.raises(ValueError, match="self-hash differs"):
        verify_binding(registry, audit)


@pytest.mark.parametrize("count", [True, -1, "2"])
def test_rehashed_invalid_scope_count_rejects(count):
    registry, audit = ledgers()
    registry["accepted_edge_count"] = count
    seal(registry, audit)
    with pytest.raises(ValueError, match="nonnegative integer"):
        verify_binding(registry, audit)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_observation",
        "duplicate_rank",
        "wrong_scope_count",
        "overlapping_classification",
        "wrong_group_count",
        "excess_qualification",
    ],
)
def test_rehashed_contradictory_census_still_rejects(mutation):
    registry, audit = ledgers()
    if mutation == "duplicate_observation":
        registry["terminal_do_not_repeat"].append({"family": "consumed-window"})
    elif mutation == "duplicate_rank":
        registry["prioritized_hypotheses"][1]["priority_rank"] = 1
    elif mutation == "wrong_scope_count":
        audit["decision"]["accepted_scope_count"] = 1
    elif mutation == "overlapping_classification":
        audit["classification"][0]["edge_ordinals"] = [1, 1]
    elif mutation == "wrong_group_count":
        audit["classification"][0]["count"] = 1
    else:
        audit["decision"][
            "stable_current_account_qualified_after_all_cost_edge_count"
        ] = 3
    seal(registry, audit)
    with pytest.raises(ValueError):
        verify_binding(registry, audit)


@pytest.mark.parametrize(
    "mutation",
    [
        "bound_scope",
        "rank_boolean",
        "classification_boolean",
        "missing_terminal_list",
        "missing_classification_list",
    ],
)
def test_rehashed_schema_and_identity_failures_reject(mutation):
    registry, audit = ledgers()
    if mutation == "bound_scope":
        audit["source_binding"]["accepted_edge_count"] = 1
    elif mutation == "rank_boolean":
        registry["prioritized_hypotheses"][0]["priority_rank"] = True
    elif mutation == "classification_boolean":
        audit["classification"][0]["edge_ordinals"] = [True, 2]
    elif mutation == "missing_terminal_list":
        registry["terminal_do_not_repeat"] = {}
    else:
        audit["classification"] = {}
    seal(registry, audit)
    with pytest.raises(ValueError):
        verify_binding(registry, audit)


def test_cli_missing_file_fails_without_printing_private_path(
    tmp_path, monkeypatch, capsys
):
    missing = tmp_path / "private-ledger.json"
    monkeypatch.setattr(sys, "argv", ["verify", "--registry", str(missing)])
    assert main() == 1
    text = capsys.readouterr().out
    assert str(missing) not in text
    assert json.loads(text) == {"valid": False, "error_type": "FileNotFoundError"}
