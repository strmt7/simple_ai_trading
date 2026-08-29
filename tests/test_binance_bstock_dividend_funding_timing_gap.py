from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "binance-bstock-dividend-perp-funding-timing-gap-candidate-v1-2026-08-27.json"
)
TRIGGER_RESULT = ROOT / "docs/model-research/action-value" / (
    "binance-glw-special-funding-trigger-result-v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "c073b61271886a5add71c2578caa889dfb97b1245327ae746bd517a91e52530d"
TRIGGER_RESULT_HASH = "823448f115ecf7fe3e7fe8862855f40dfd351ed041fce2aa94196d069c8d585a"
REGISTRY_HASH = "0a34d7289331515f8e7b3f09e856fbc331ecbc3a91130fea20542a39ef211f60"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_candidate_is_hash_bound_unaccepted_and_public_only() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    adjudication = artifact["adjudication"]
    assert adjudication["accepted_edge"] is False
    assert adjudication["deployment_ready"] is False
    assert adjudication["market_direction_forecast_required"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["trading_authority"] is False
    authority = artifact["authority"]
    assert authority["authenticated_requests"] == 0
    assert authority["account_state_accessed"] is False
    assert authority["orders_or_transfers_submitted"] == 0
    assert authority["funded_actions"] == 0
    assert authority["retained_public_http_responses"] == 15


def test_historical_special_debits_match_gross_dividends() -> None:
    artifact = _load(ARTIFACT)
    expected = {
        "AMAT": Decimal("0.53"),
        "MSFT": Decimal("0.91"),
    }

    for row in artifact["historical_mechanism_evidence"]:
        gross = expected[row["symbol"]]
        debit = abs(Decimal(row["funding_rate"])) * Decimal(row["funding_mark_price"])
        precision = Decimal("0.000000000001")
        assert debit.quantize(precision) == Decimal(
            row["funding_debit_per_short_contract_unit"]
        )
        assert abs(gross - debit).quantize(precision) == Decimal(
            row["funding_debit_absolute_difference_from_gross_dividend"]
        )
        assert abs(gross - debit) <= Decimal("0.000003")
        assert row["funding_time_utc"] == "2026-08-20T00:00:01.002Z"


def test_direct_capture_is_negative_before_other_costs() -> None:
    artifact = _load(ARTIFACT)
    contract = artifact["economic_contract"]
    gross = Decimal("0.53")
    deductions = Decimal("0.10")
    net = gross - deductions

    assert net - gross == -deductions
    assert "N-D=-F<0" in contract["direct_pre_adjustment_pair"]
    assert contract["public_net_distribution_floor"] == "0"
    assert "cannot credit the dividend twice" in contract["why_close_and_reopen_is_not_a_free_shortcut"]


def test_only_glw_has_the_preregistered_weekend_gap() -> None:
    artifact = _load(ARTIFACT)
    cases = {row["symbol"]: row for row in artifact["current_prospective_cases"]}

    assert cases["GLW"]["ex_date_day"] == "Friday"
    assert cases["GLW"]["record_date_day"] == "Monday"
    assert cases["GLW"]["bStock_record_snapshot_utc"] == "2026-08-31T00:00:00Z"
    assert "weekend_timing_gap_candidate" in cases["GLW"]["status"]
    assert "same_day_record_structure" in cases["GS"]["status"]
    gate = artifact["prospective_gate"]
    assert "do_not_assume_the_GLW_adjustment_time" in gate["first_observation"]
    assert "one synchronized depth_20" in gate["second_observation"]
    assert "net distribution floor remains zero" in gate["stop_conditions"]


def test_glw_one_use_observation_blocks_conditional_books() -> None:
    result = _load(TRIGGER_RESULT)

    assert result["result_sha256"] == TRIGGER_RESULT_HASH
    assert _canonical_hash(result) == TRIGGER_RESULT_HASH
    assert result["history_row_count"] == 8
    assert result["history_observed_rate_types"] == ["Regular"]
    assert result["history_special_row_count"] == 0
    assert result["conditional_batch_executed"] is False
    assert result["request_count"] == 1
    assert result["accepted_edge"] is False
    assert result["profitability_claim"] is False
    assert "terminal_history_reconciliation" in result["next_retry_trigger"]


def test_registry_adds_candidate_and_closes_only_direct_family() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 21
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 45))
    candidate = next(
        row
        for row in hypotheses
        if row["mechanism"]
        == "binance_bstock_dividend_perpetual_funding_timing_gap"
    )
    assert candidate["priority_rank"] == 34
    assert candidate["market_direction_forecast_required"] is False
    assert "after_2026_08_31" in candidate["retry_trigger"]
    assert candidate["canonical_artifacts"][-1]["result_sha256"] == TRIGGER_RESULT_HASH
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_bstock_direct_same_day_dividend_capture_long_bstock_short_tradfi_perpetual"
    )
    assert terminal["canonical_result_sha256"] == EXPECTED_HASH
    assert "gross_dividends" in terminal["reason"]
