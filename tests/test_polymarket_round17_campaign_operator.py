from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round17_campaign_operator as operator
from simple_ai_trading import polymarket_round17_development_operator as development
from simple_ai_trading.polymarket_round17_campaign_operator import (
    Round17CampaignOperatorConfig,
    inspect_round17_campaign_readiness,
    iter_round17_campaign_development_conditions,
    materialize_round17_campaign_development_index,
)
from simple_ai_trading.polymarket_round17_development_operator import (
    Round17DevelopmentOperatorConfig,
    run_round17_development,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PLAN = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-campaign-plan-v1.json"
)
COHORT_PLAN = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-017-btc-5m-cohort-plan-v1.json"
)
ADMISSION_SPEC = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-admission-spec-v1.json"
)
RISK_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-prospective-contract-v1.json"
)
ECONOMIC_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-017-btc-5m-economic-pretest-v1.json"
)
PLAN_SHA256 = "c19ef7733efb86202742f045a45b3d92e8e17bb922c3c5f780240243889609b5"
START_MS = 1_785_344_400_000


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _write_hashed(path: Path, payload: dict[str, object]) -> None:
    selected = {**payload, "artifact_sha256": _sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")


def _partial_config(tmp_path: Path) -> Round17CampaignOperatorConfig:
    database = tmp_path / "evidence.duckdb"
    database.write_bytes(b"x")
    state_root = tmp_path / "state"
    _write_hashed(
        state_root / "slots" / "slot-0000.json",
        {
            "schema_version": "polymarket-round14-campaign-slot-result-v1",
            "plan_sha256": PLAN_SHA256,
            "slot_index": 0,
            "scheduled_start_ms": START_MS,
            "observed_at_ms": START_MS + 1_800_000,
            "status": "degraded",
            "condition_level_admission_required": True,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "details": {
                "run_id": "run-slot-0000",
                "report_sha256": "a" * 64,
                "errors": [],
                "integrity_errors": [],
            },
        },
    )
    _write_hashed(
        state_root / "campaign-state.json",
        {
            "schema_version": "polymarket-round14-campaign-state-v1",
            "plan_sha256": PLAN_SHA256,
            "terminal_slot_count": 1,
            "active_slot_indexes": [1],
            "next_slot_index": 1,
            "database_bytes": 1,
            "observed_at_ms": START_MS + 1_900_000,
            "recovered_interrupted_run_count": 0,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
    )
    return Round17CampaignOperatorConfig(
        campaign_plan_path=CAMPAIGN_PLAN,
        cohort_plan_path=COHORT_PLAN,
        admission_spec_path=ADMISSION_SPEC,
        database_path=database,
        state_root=state_root,
    )


def test_round17_campaign_readiness_never_opens_partial_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _partial_config(tmp_path)

    def forbidden_store(*_args, **_kwargs):
        raise AssertionError("partial campaign database was opened")

    monkeypatch.setattr(operator, "PolymarketEvidenceStore", forbidden_store)
    readiness = inspect_round17_campaign_readiness(config)

    assert readiness.ready is False
    assert readiness.terminal_slot_count == 1
    assert readiness.active_slot_indexes == (1,)
    assert readiness.asdict()["database_opened"] is False
    assert readiness.asdict()["test_features_accessed"] is False
    assert readiness.gates["all_slots_terminal"] is False
    assert readiness.gates["wal_absent"] is True
    with pytest.raises(RuntimeError, match="not terminal"):
        tuple(iter_round17_campaign_development_conditions(config))
    with pytest.raises(RuntimeError, match="not terminal"):
        materialize_round17_campaign_development_index(config)

    output = tmp_path / "round17-development.json"
    with pytest.raises(RuntimeError, match="not terminal"):
        run_round17_development(
            Round17DevelopmentOperatorConfig(
                campaign=config,
                risk_contract_path=RISK_CONTRACT,
                economic_contract_path=ECONOMIC_CONTRACT,
                output_path=output,
            )
        )
    assert not output.exists()


def test_round17_campaign_readiness_is_hash_bound(tmp_path: Path) -> None:
    readiness = inspect_round17_campaign_readiness(_partial_config(tmp_path))

    with pytest.raises(ValueError, match="integrity differs"):
        replace(readiness, readiness_sha256="f" * 64).validated()


def test_round17_development_operator_has_no_binance_execution_dependency() -> None:
    source_path = (
        ROOT
        / "src"
        / "simple_ai_trading"
        / "polymarket_round17_development_operator.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any("binance" in module.lower() for module in imported_modules)


def test_round17_development_stops_after_rejected_model_and_closes_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _partial_config(tmp_path)
    condition_ids = {
        role: "0x" + hashlib.sha256(role.encode("ascii")).hexdigest()
        for role in (
            "train",
            "tune_calibration",
            "tune_selection",
            "tune_uncertainty",
            "tune_economic",
        )
    }
    references = tuple(
        SimpleNamespace(condition_id=condition_id, role=role)
        for role, condition_id in condition_ids.items()
    )
    markets = {
        item.condition_id: SimpleNamespace(
            event_start_ms=START_MS + index * 300_000,
            asdict=lambda condition_id=item.condition_id: {
                "condition_id": condition_id
            },
        )
        for index, item in enumerate(references)
    }
    index = SimpleNamespace(
        cohort_manifest=SimpleNamespace(conditions=references),
        markets=tuple(markets.values()),
        market_mapping=lambda: markets,
    )
    acquisition = SimpleNamespace(
        complete=True,
        observations=(),
        labels=lambda *_args: (),
    )
    target = SimpleNamespace(
        development_dataset_sha256="d" * 64,
        target_manifest_sha256="e" * 64,
    )
    appended: list[str] = []
    reader_closed = False

    class FakeAssembler:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def append(self, dataset: object) -> None:
            appended.append(str(dataset.condition_id))  # type: ignore[attr-defined]

        def finish(self) -> dict[str, object]:
            return {
                "train": object(),
                "tune_calibration": object(),
                "tune_selection": object(),
                "tune_uncertainty": object(),
            }

    def conditions():
        nonlocal reader_closed
        try:
            for reference in references:
                market = markets[reference.condition_id]
                yield SimpleNamespace(
                    dataset=SimpleNamespace(
                        condition_id=reference.condition_id,
                        rows=(),
                    ),
                    market=market,
                    books=(),
                    cohort_condition=reference,
                )
        finally:
            reader_closed = True

    def rejected_model(*_args, **_kwargs) -> dict[str, object]:
        return {
            "development_accepted": False,
            "selected_candidate_id": "rejected",
            "pretest_sha256": "f" * 64,
        }

    def publish(_config, **kwargs):
        assert reader_closed is True
        assert kwargs["status"] == "model_rejected"
        assert kwargs["calibration"] is None
        assert kwargs["economic_pretest"] is None
        return {"status": kwargs["status"]}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("rejected model reached a downstream economic stage")

    monkeypatch.setattr(
        development,
        "materialize_round17_campaign_development_index",
        lambda *_args, **_kwargs: index,
    )
    monkeypatch.setattr(
        development,
        "acquire_round17_development_resolutions",
        lambda *_args, **_kwargs: acquisition,
    )
    monkeypatch.setattr(
        development,
        "build_round17_development_target_manifest",
        lambda *_args, **_kwargs: target,
    )
    monkeypatch.setattr(development, "Round17DevelopmentPanelAssembler", FakeAssembler)
    monkeypatch.setattr(
        development,
        "iter_round17_campaign_development_conditions",
        lambda *_args, **_kwargs: conditions(),
    )
    monkeypatch.setattr(development, "fit_round17_development_pretest", rejected_model)
    monkeypatch.setattr(development, "validate_round17_pretest_artifact", dict)
    monkeypatch.setattr(development, "fit_round17_probability_calibration", forbidden)
    monkeypatch.setattr(
        development,
        "materialize_round17_condition_economic_outcomes",
        forbidden,
    )
    monkeypatch.setattr(development, "_publish", publish)

    result = run_round17_development(
        Round17DevelopmentOperatorConfig(
            campaign=campaign,
            risk_contract_path=RISK_CONTRACT,
            economic_contract_path=ECONOMIC_CONTRACT,
            output_path=tmp_path / "round17-development.json",
        )
    )

    assert result == {"status": "model_rejected"}
    assert appended == [
        condition_ids["train"],
        condition_ids["tune_calibration"],
        condition_ids["tune_selection"],
        condition_ids["tune_uncertainty"],
    ]


def test_round17_campaign_readiness_rejects_noncontiguous_slots(
    tmp_path: Path,
) -> None:
    config = _partial_config(tmp_path)
    first = config.state_root / "slots" / "slot-0000.json"
    first.rename(config.state_root / "slots" / "slot-0001.json")

    with pytest.raises(ValueError, match="not contiguous"):
        inspect_round17_campaign_readiness(config)
