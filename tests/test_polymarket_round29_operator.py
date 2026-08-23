from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round29_operator as operator
from simple_ai_trading.polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round29_economics import (
    POLYMARKET_ROUND29_ECONOMIC_SCHEMA_VERSION,
    paired_round29_economic_scenario,
)
from simple_ai_trading.polymarket_round29_model import Round29ModelSample
from simple_ai_trading.polymarket_round29_operator import (
    build_round29_selection_input_manifest,
    validate_round29_economic_report,
    validate_round29_selection_input_manifest,
    verify_round29_implementation_sources,
)
from simple_ai_trading.polymarket_round29_settlement_features import (
    POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "docs/model-research/polymarket/"
    "round-029-settlement-state-matched-ablation-preregistration-v1.json"
)
AMENDMENT = (
    ROOT / "docs/model-research/polymarket/"
    "round-029-model-economic-operator-implementation-amendment-v1.json"
)


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


def _claim(hash_field: str, marker: str) -> dict[str, object]:
    body: dict[str, object] = {"schema_version": marker, "target_accessed": False}
    body[hash_field] = _canonical_sha256(body)
    return body


def _sample(index: int, *, role: str) -> Round29ModelSample:
    target = index % 2
    diagnostic_base = (0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES)
    settlement = (1.0 if target else -1.0,) + (0.0,) * (
        len(POLYMARKET_ROUND29_SETTLEMENT_FEATURE_NAMES) - 1
    )
    bbo = (0.0,) * len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES)
    primary_base = (*diagnostic_base, *bbo)
    event_start_ms = 1_800_000 + index * 1_200_000
    digest = hashlib.sha256(str(index).encode("ascii")).hexdigest()
    return Round29ModelSample(
        slot_id=f"stage1-{role}",
        role=role,
        condition_id="0x" + f"{index + 1:064x}",
        event_start_ms=event_start_ms,
        decision_time_ms=event_start_ms + 30_000,
        market_prior_probability=0.6,
        diagnostic_base_values=diagnostic_base,
        diagnostic_augmented_values=(*diagnostic_base, *settlement),
        primary_base_values=primary_base,
        primary_augmented_values=(*primary_base, *settlement),
        target_up=target,
        condition_weight=1.0,
        diagnostic_feature_row_sha256=digest,
        primary_feature_row_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
    ).validated()


def _inputs() -> tuple[
    tuple[Round29ModelSample, ...],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    samples = tuple(
        _sample(index, role=role)
        for index, role in enumerate(
            ("train", "train", "calibration", "calibration", "selection", "selection")
        )
    )
    return (
        samples,
        load_round27_model_contract(ROOT),
        json.loads(PREREGISTRATION.read_text(encoding="ascii")),
        json.loads(AMENDMENT.read_text(encoding="ascii")),
    )


def _manifest() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    samples, contract, preregistration, amendment = _inputs()
    manifest = build_round29_selection_input_manifest(
        samples=samples,
        feature_store_audit=_claim("audit_sha256", "feature-audit"),
        bbo_overlay_report=_claim("report_sha256", "bbo-overlay"),
        settlement_overlay_report=_claim("report_sha256", "settlement-overlay"),
        target_store_audit=_claim("audit_sha256", "target-audit"),
        contract=contract,
        preregistration=preregistration,
        implementation_amendment=amendment,
    )
    return manifest, contract, preregistration


def test_round29_implementation_and_manifest_are_exact_source_bound() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="ascii"))

    verified = verify_round29_implementation_sources(amendment, root=ROOT)
    manifest, _contract, _preregistration = _manifest()

    assert verified["knowledge_at_freeze"]["official_outcomes_accessed"] is False
    assert validate_round29_selection_input_manifest(manifest) == manifest
    assert [item["role"] for item in manifest["roles"]] == [
        "train",
        "calibration",
        "selection",
    ]
    assert manifest["sealed_partition_accessed"] is False
    tampered = json.loads(json.dumps(manifest))
    tampered["roles"][0]["row_count"] = 0
    tampered.pop("manifest_sha256")
    tampered["manifest_sha256"] = _canonical_sha256(tampered)
    with pytest.raises(ValueError, match="selection input manifest differs"):
        validate_round29_selection_input_manifest(tampered)


def _scenario(
    conditions: tuple[str, ...],
    *,
    delay_ms: int,
    pnl: str,
    drawdown: str,
) -> dict[str, object]:
    trades = [
        {
            "condition_id": condition_id,
            "execution_state": "FILLED",
            "net_pnl_quote": pnl,
        }
        for condition_id in conditions
    ]
    body: dict[str, object] = {
        "delay_ms": delay_ms,
        "evaluated_condition_count": len(conditions),
        "net_pnl_quote": format(Decimal(pnl) * len(conditions), "f"),
        "maximum_drawdown_fraction": drawdown,
        "scenario_edge_gate_passed": True,
        "trades": trades,
    }
    body["scenario_sha256"] = _canonical_sha256(body)
    return body


def _economic_arm(
    *,
    conditions: tuple[str, ...],
    source_sha256: str,
    resolution_sha256: str,
    pnl: str,
    drawdown: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
        "source_audit_sha256": source_sha256,
        "resolution_evidence_sha256": resolution_sha256,
        "config": {"fixture": "same"},
        "scenarios": [
            _scenario(
                conditions,
                delay_ms=delay,
                pnl=pnl,
                drawdown=drawdown,
            )
            for delay in (250, 500, 1_000, 2_000)
        ],
        "economic_edge_gate_passed": True,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def _economic_report(
    manifest: dict[str, object],
    pair: SimpleNamespace,
    selection_claim: dict[str, object],
    resolution_sha256: str,
) -> dict[str, object]:
    conditions = tuple("0x" + f"{index + 1:064x}" for index in range(20))
    base = _economic_arm(
        conditions=conditions,
        source_sha256=str(manifest["manifest_sha256"]),
        resolution_sha256=resolution_sha256,
        pnl="0.10",
        drawdown="0.02",
    )
    augmented = _economic_arm(
        conditions=conditions,
        source_sha256=str(manifest["manifest_sha256"]),
        resolution_sha256=resolution_sha256,
        pnl="0.20",
        drawdown="0.01",
    )
    paired = [
        paired_round29_economic_scenario(
            base=base["scenarios"][index],
            augmented=augmented["scenarios"][index],
            ordered_conditions=conditions,
            bootstrap_draws=1_000,
            bootstrap_seed=29_029,
        )
        for index in range(4)
    ]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_ECONOMIC_SCHEMA_VERSION,
        "partition_role": "selection",
        "round27_model_contract_sha256": manifest["round27_model_contract_sha256"],
        "round29_preregistration_sha256": manifest["round29_preregistration_sha256"],
        "round29_implementation_amendment_sha256": manifest[
            "round29_implementation_amendment_sha256"
        ],
        "round29_selection_claim_sha256": selection_claim["claim_sha256"],
        "selected_model_family": pair.model_family,
        "base_model_sha256": pair.base_model.model_sha256,
        "augmented_model_sha256": pair.augmented_model.model_sha256,
        "source_audit_sha256": manifest["manifest_sha256"],
        "resolution_evidence_sha256": resolution_sha256,
        "condition_population_sha256": _canonical_sha256(list(conditions)),
        "base_economic_report": base,
        "augmented_economic_report": augmented,
        "paired_scenarios": paired,
        "economic_uplift_gate_passed": True,
        "sealed_partition_accessed": False,
        "economic_metrics_computed": True,
        "ai_assist_evaluated": False,
        "credentials_used": False,
        "edge_claim": False,
        "execution_connected": False,
        "live_trading_authority": False,
        "orders_submitted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def test_round29_restart_validation_rejects_rehashed_nested_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, contract, preregistration = _manifest()
    pair = SimpleNamespace(
        model_family="l2_offset_logistic",
        base_model=SimpleNamespace(model_sha256="1" * 64),
        augmented_model=SimpleNamespace(model_sha256="2" * 64),
    )
    monkeypatch.setattr(
        operator,
        "load_round29_selected_pair",
        lambda *_args, **_kwargs: pair,
    )
    selection_claim = {"claim_sha256": "3" * 64}
    resolution_sha256 = "4" * 64
    report = _economic_report(
        manifest,
        pair,
        selection_claim,
        resolution_sha256,
    )

    assert (
        validate_round29_economic_report(
            report,
            input_manifest=manifest,
            selection_claim=selection_claim,
            contract=contract,
            preregistration=preregistration,
            resolution_evidence_sha256=resolution_sha256,
        )
        == report
    )

    tampered = json.loads(json.dumps(report))
    inherited = tampered["paired_scenarios"][0]["inherited_round28_matched_scenario"]
    inherited["base_scenario_sha256"] = "9" * 64
    inherited.pop("paired_scenario_sha256")
    inherited["paired_scenario_sha256"] = _canonical_sha256(inherited)
    wrapper = tampered["paired_scenarios"][0]
    wrapper.pop("paired_scenario_sha256")
    wrapper["paired_scenario_sha256"] = _canonical_sha256(wrapper)
    tampered.pop("report_sha256")
    tampered["report_sha256"] = _canonical_sha256(tampered)
    with pytest.raises(ValueError, match="matched economic report differs"):
        validate_round29_economic_report(
            tampered,
            input_manifest=manifest,
            selection_claim=selection_claim,
            contract=contract,
            preregistration=preregistration,
            resolution_evidence_sha256=resolution_sha256,
        )
