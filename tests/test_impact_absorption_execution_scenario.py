from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_latency_evidence_claims,
    round74_slippage_evidence_claims,
)
from simple_ai_trading.impact_absorption_execution_evidence import (
    Round74ExecutionEvidenceBundle,
)
from simple_ai_trading.impact_absorption_execution_scenario import (
    ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256,
    ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID,
    ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID,
    Round74ExecutionAggregateSource,
    Round74PublicTransportSource,
    build_round74_public_execution_scenario,
    load_round74_execution_aggregate_source,
    load_round74_public_execution_scenario_artifact,
)
from simple_ai_trading.round74_target_assembly_manifest import (
    _audit_execution_scenario_artifact,
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


def _bundle(
    *,
    latency_ns: int = 100_000_000,
) -> Round74ExecutionEvidenceBundle:
    entries = {symbol: latency_ns for symbol in ROUND74_EVENT_TARGET_SYMBOLS}
    exits = {symbol: latency_ns + 10_000_000 for symbol in ROUND74_EVENT_TARGET_SYMBOLS}
    slippage = {
        symbol: float(index + 1)
        for index, symbol in enumerate(ROUND74_EVENT_TARGET_SYMBOLS)
    }
    latency_claims = round74_latency_evidence_claims(
        decision_to_entry_latency_ns_by_symbol=entries,
        decision_to_exit_latency_ns_by_symbol=exits,
    )
    slippage_claims = round74_slippage_evidence_claims(
        reference_quote_notional=100.0,
        additional_slippage_bps_per_side_by_symbol=slippage,
    )
    return Round74ExecutionEvidenceBundle(
        reference_quote_notional=100.0,
        decision_to_entry_latency_ns_by_symbol=tuple(entries.items()),
        decision_to_exit_latency_ns_by_symbol=tuple(exits.items()),
        additional_slippage_bps_per_side_by_symbol=tuple(slippage.items()),
        entry_exit_latency_evidence=Round74EventTargetEvidence.create(
            kind="entry_exit_latency",
            environment="binance_usdm_testnet",
            observed_wall_ns=2_000_000_000,
            record_count=1800,
            source_query_or_protocol_sha256="1" * 64,
            source_payload_sha256="2" * 64,
            claims=latency_claims,
        ),
        residual_slippage_evidence=Round74EventTargetEvidence.create(
            kind="residual_slippage",
            environment="binance_usdm_testnet",
            observed_wall_ns=2_000_000_000,
            record_count=1800,
            source_query_or_protocol_sha256="1" * 64,
            source_payload_sha256="2" * 64,
            claims=slippage_claims,
        ),
    )


def _aggregate(
    *,
    latency_ns: int = 100_000_000,
) -> Round74ExecutionAggregateSource:
    return Round74ExecutionAggregateSource(
        bundle=_bundle(latency_ns=latency_ns),
        artifact_sha256="3" * 64,
        artifact_file_sha256="4" * 64,
        observed_wall_ns=2_000_000_000,
        source_record_count=1800,
    )


def _transport(
    *,
    latency_ns: int = 20_000_000,
) -> Round74PublicTransportSource:
    return Round74PublicTransportSource(
        run_id="a" * 32,
        cohort_binding_sha256="5" * 64,
        capture_report_sha256="6" * 64,
        observed_wall_ns=3_000_000_000,
        source_payload_sha256="7" * 64,
        transport_latency_ns_by_symbol=tuple(
            (symbol, (latency_ns,) * 300) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
    )


def _write_hashed(path: Path, payload: dict[str, object]) -> None:
    value = dict(payload)
    value["artifact_sha256"] = _canonical_sha256(value)
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def test_scenario_adds_observed_transport_tail_without_transfer_claim() -> None:
    scenario = build_round74_public_execution_scenario(
        transport_source=_transport(),
        execution_aggregate=_aggregate(),
    )

    assert scenario.scenario_contract_sha256 == (
        ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256
    )
    assert scenario.entry_latency_mapping() == {
        symbol: 120_000_000 for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    }
    assert scenario.exit_latency_mapping() == {
        symbol: 130_000_000 for symbol in ROUND74_EVENT_TARGET_SYMBOLS
    }
    assert scenario.entry_exit_latency_evidence.source_id == (
        ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID
    )
    assert scenario.residual_slippage_evidence.source_id == (
        ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID
    )
    payload = scenario.as_dict()
    assert payload["authority"]["mainnet_fill_evidence"] is False
    assert (
        payload["upstream_testnet_calibration"]["mainnet_transfer_permitted"] is False
    )
    assert (
        payload["scenario_panel"][1]["exact_future_public_l2_replay_required"] is True
    )


def test_scenario_artifact_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    scenario = build_round74_public_execution_scenario(
        transport_source=_transport(),
        execution_aggregate=_aggregate(),
    )
    path = tmp_path / "scenario.json"
    _write_hashed(path, scenario.as_dict())

    restored = load_round74_public_execution_scenario_artifact(path)

    assert restored.bundle.scenario_sha256 == scenario.scenario_sha256
    assert len(restored.artifact_sha256) == 64
    tampered = json.loads(path.read_text(encoding="ascii"))
    tampered["authority"]["mainnet_fill_evidence"] = True
    path.write_text(
        json.dumps(tampered, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="artifact digest differs"):
        load_round74_public_execution_scenario_artifact(path)


def test_target_manifest_audits_scenario_semantics() -> None:
    scenario = build_round74_public_execution_scenario(
        transport_source=_transport(),
        execution_aggregate=_aggregate(),
    )
    payload = scenario.as_dict()

    _audit_execution_scenario_artifact(
        payload,
        run_id=scenario.run_id,
    )

    payload["authority"]["mainnet_fill_evidence"] = True
    with pytest.raises(ValueError, match="scenario artifact differs"):
        _audit_execution_scenario_artifact(
            payload,
            run_id=scenario.run_id,
        )


def test_execution_aggregate_loader_preserves_testnet_identity(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    captures = [
        {
            "slot_ordinal": ordinal,
            "round_trip_id": f"round-trip-{ordinal:03d}",
            "capture_artifact_sha256": "8" * 64,
            "capture_artifact_file_sha256": "9" * 64,
            "pair_sha256": "a" * 64,
        }
        for ordinal in range(900)
    ]
    payload: dict[str, object] = {
        "schema_version": "round-074-execution-calibration-aggregate-v1",
        "operation": "aggregate",
        "environment": "binance_usdm_testnet",
        "campaign_plan": {
            "plan_sha256": "b" * 64,
            "plan_artifact_sha256": "c" * 64,
            "plan_artifact_file_sha256": "d" * 64,
            "campaign_id": "test-campaign",
            "slot_count": 900,
        },
        "source_capture_artifacts": captures,
        "source_capture_artifact_count": 900,
        "source_record_count": 1800,
        "observed_wall_ns": 2_000_000_000,
        "execution_evidence": {
            "reference_quote_notional": bundle.reference_quote_notional,
            "decision_to_entry_latency_ns_by_symbol": (bundle.entry_latency_mapping()),
            "decision_to_exit_latency_ns_by_symbol": (bundle.exit_latency_mapping()),
            "additional_slippage_bps_per_side_by_symbol": (bundle.slippage_mapping()),
            "entry_exit_latency_evidence": (
                bundle.entry_exit_latency_evidence.as_dict()
            ),
            "residual_slippage_evidence": (bundle.residual_slippage_evidence.as_dict()),
        },
        "aggregator_source": {
            "path": "tools/aggregate_round74_execution_calibration.py",
            "sha256": "e" * 64,
        },
        "network_accessed": False,
        "orders_submitted": False,
        "credential_material_read": False,
        "authority": {
            "testnet_execution_calibration": True,
            "mainnet_execution_equivalence": False,
            "mainnet_transfer_permitted": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    path = tmp_path / "aggregate.json"
    _write_hashed(path, payload)

    restored = load_round74_execution_aggregate_source(path)

    assert restored.bundle.entry_latency_mapping() == (bundle.entry_latency_mapping())
    assert restored.bundle.entry_exit_latency_evidence.environment == (
        "binance_usdm_testnet"
    )


def test_scenario_fails_closed_when_selected_latency_is_unrepresentable() -> None:
    with pytest.raises(ValueError, match="exceeds target bound"):
        build_round74_public_execution_scenario(
            transport_source=_transport(latency_ns=200_000_000),
            execution_aggregate=_aggregate(latency_ns=4_900_000_000),
        )


def test_scenario_rejects_insufficient_public_transport_sample() -> None:
    source = _transport()
    insufficient = Round74PublicTransportSource(
        run_id=source.run_id,
        cohort_binding_sha256=source.cohort_binding_sha256,
        capture_report_sha256=source.capture_report_sha256,
        observed_wall_ns=source.observed_wall_ns,
        source_payload_sha256=source.source_payload_sha256,
        transport_latency_ns_by_symbol=tuple(
            (symbol, values[:299])
            for symbol, values in source.transport_latency_ns_by_symbol
        ),
    )

    with pytest.raises(ValueError, match="transport sample differs"):
        build_round74_public_execution_scenario(
            transport_source=insufficient,
            execution_aggregate=_aggregate(),
        )
