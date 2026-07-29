from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round14_campaign as campaign_module
from simple_ai_trading.polymarket_round14_campaign import (
    POLYMARKET_ROUND14_TOTAL_SLOTS,
    PolymarketRound14CampaignConfig,
    create_round14_campaign_plan,
    inspect_round14_campaign,
    validate_round14_campaign_plan,
)


CREATED_MS = 1_800_000_000_000
START_MS = CREATED_MS + 1_800_000
CONTRACT_PATH = (
    "docs/model-research/polymarket/"
    "round-014-btc-5m-prospective-contract-v1.json"
)


def _plan() -> dict[str, object]:
    return create_round14_campaign_plan(
        created_at_ms=CREATED_MS,
        scheduled_start_ms=START_MS,
        contract_repository_path=CONTRACT_PATH,
        contract_sha256="a" * 64,
    )


def _rehash(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("plan_sha256", None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    body["plan_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def test_campaign_plan_is_exactly_thirty_days_and_has_no_authority() -> None:
    raw = _plan()
    plan = validate_round14_campaign_plan(raw)

    assert plan.total_slots == POLYMARKET_ROUND14_TOTAL_SLOTS == 1_440
    assert plan.scheduled_slot_ms(1) - plan.scheduled_slot_ms(0) == 1_800_000
    assert plan.scheduled_end_ms - plan.scheduled_start_ms == 2_592_000_000
    assert raw["required_assets"] == ["BTC"]
    assert raw["profitability_claim"] is False
    assert raw["paper_trading_authority"] is False
    assert raw["live_trading_authority"] is False


def test_campaign_plan_rejects_schedule_resource_or_authority_drift() -> None:
    for key, value in (
        ("scheduled_start_ms", START_MS + 1),
        ("queue_capacity", 99_999),
        ("live_trading_authority", True),
    ):
        tampered = _plan()
        tampered[key] = value
        with pytest.raises(ValueError, match="campaign plan is invalid"):
            validate_round14_campaign_plan(_rehash(tampered))


def test_campaign_plan_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate keys"):
        campaign_module.load_round14_campaign_plan(path)


def test_campaign_inspection_is_truthful_and_resource_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    plan_path = repository / "plan.json"
    contract_path = repository / CONTRACT_PATH
    database = repository / "data" / "campaign.duckdb"
    state = repository / "data" / "campaign-state"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text("{}\n", encoding="utf-8")
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    config = PolymarketRound14CampaignConfig(
        repository=repository,
        plan_path=plan_path,
        database_path=database,
        state_root=state,
    )
    program = type("Program", (), {"contract_sha256": "a" * 64})()
    monkeypatch.setattr(campaign_module, "load_round14_contract", lambda _path: program)
    monkeypatch.setattr(
        campaign_module,
        "_resource_reason",
        lambda _config: "minimum_free_space_not_met",
    )

    result = inspect_round14_campaign(config, now_ms=START_MS)

    assert result["relation"] == "open"
    assert result["current_slot_index"] == 0
    assert result["resource_block"] == "minimum_free_space_not_met"
    assert result["profitability_claim"] is False
    assert result["live_trading_authority"] is False


def test_transport_completion_cannot_claim_model_eligibility(
    tmp_path: Path,
) -> None:
    plan = validate_round14_campaign_plan(_plan())

    result = campaign_module._write_slot_result(
        tmp_path,
        plan=plan,
        slot_index=0,
        scheduled_start_ms=START_MS,
        status="complete",
        observed_at_ms=START_MS + 1,
        details={"recorder_status": "complete"},
    )

    assert result["condition_level_admission_required"] is True
    assert result["model_data_eligible"] is False
    assert result["profitability_claim"] is False
    assert result["live_trading_authority"] is False


def test_live_qualification_artifact_is_hash_bound_and_non_promotional() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-014-btc-5m-live-qualification-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")

    assert hashlib.sha256(encoded).hexdigest() == claimed
    assert payload["status"] == "qualification_passed"
    assert payload["capture"]["stream_gap_count"] == 0
    assert payload["causal_replay"]["resolution_count"] == 0
    assert payload["storage"]["source_database_published"] is False
    assert payload["storage"]["reproducible_from_repository_alone"] is False
    assert payload["scope"]["model_data_eligible"] is False
    assert payload["scope"]["profitability_claim"] is False
    assert payload["scope"]["paper_trading_authority"] is False
    assert payload["scope"]["live_trading_authority"] is False
