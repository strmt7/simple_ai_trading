from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest

from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    Round74SegmentedCohortRunBinding,
)
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
    Round74PublicTransportSource,
    build_round74_public_execution_scenario,
    load_round74_execution_aggregate_source,
)
from simple_ai_trading.round74_commission_capture import (
    capture_round74_mainnet_commission,
)
from simple_ai_trading.round74_public_target_operator import (
    build_round74_public_target_manifest,
    write_round74_public_target_manifest,
)
from simple_ai_trading.round74_public_target_sources import (
    build_round74_cohort_capture_source_payload,
    load_round74_canonical_source_artifact,
)
from simple_ai_trading.round74_target_assembly_manifest import (
    load_and_audit_round74_target_assembly_manifest,
)
from tools.build_round74_public_execution_scenario import (
    _load_binding as load_execution_scenario_binding,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_EXCHANGE_INFO = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-exchange-info-evidence-2026-07-27-v2.json"
)
REAL_FUNDING = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-funding-evidence-quiet-run-2026-07-27.json"
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


def _write_hashed(path: Path, payload: Mapping[str, object]) -> None:
    value = dict(payload)
    value["artifact_sha256"] = _canonical_sha256(value)
    path.write_bytes(
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _binding() -> Round74SegmentedCohortRunBinding:
    start = 2_000_000_000_000_000_000
    binding = Round74SegmentedCohortRunBinding(
        plan_sha256="1" * 64,
        slot_ordinal=0,
        role="training",
        run_id="a" * 32,
        report_sha256="2" * 64,
        supervisor_sha256="3" * 64,
        fresh_frame_audit_sha256="4" * 64,
        fresh_epoch_audit_sha256="5" * 64,
        terminal_status="completed",
        terminal_error="",
        capture_start_wall_ns=start,
        capture_end_wall_ns=start + 1_210_000_000_000,
        feature_ready_wall_ns=start + 5_000_000_000,
        usable_end_wall_ns=start + 1_205_000_000_000,
        message_count=1_000,
        frame_count=10,
        compressed_payload_bytes=10_000,
    )
    binding.validate()
    return binding


def _execution_bundle() -> Round74ExecutionEvidenceBundle:
    entries = {
        symbol: 100_000_000 + index
        for index, symbol in enumerate(ROUND74_EVENT_TARGET_SYMBOLS)
    }
    exits = {
        symbol: 120_000_000 + index
        for index, symbol in enumerate(ROUND74_EVENT_TARGET_SYMBOLS)
    }
    slippage = {
        symbol: float(index + 1)
        for index, symbol in enumerate(ROUND74_EVENT_TARGET_SYMBOLS)
    }
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
            source_query_or_protocol_sha256="6" * 64,
            source_payload_sha256="7" * 64,
            claims=round74_latency_evidence_claims(
                decision_to_entry_latency_ns_by_symbol=entries,
                decision_to_exit_latency_ns_by_symbol=exits,
            ),
        ),
        residual_slippage_evidence=Round74EventTargetEvidence.create(
            kind="residual_slippage",
            environment="binance_usdm_testnet",
            observed_wall_ns=2_000_000_000,
            record_count=1800,
            source_query_or_protocol_sha256="6" * 64,
            source_payload_sha256="7" * 64,
            claims=round74_slippage_evidence_claims(
                reference_quote_notional=100.0,
                additional_slippage_bps_per_side_by_symbol=slippage,
            ),
        ),
    )


def _aggregate_payload() -> dict[str, object]:
    bundle = _execution_bundle()
    return {
        "schema_version": "round-074-execution-calibration-aggregate-v1",
        "operation": "aggregate",
        "environment": "binance_usdm_testnet",
        "campaign_plan": {
            "plan_sha256": "8" * 64,
            "plan_artifact_sha256": "9" * 64,
            "plan_artifact_file_sha256": "b" * 64,
            "campaign_id": "operator-contract-test",
            "slot_count": 900,
        },
        "source_capture_artifacts": [
            {
                "slot_ordinal": ordinal,
                "round_trip_id": f"round-trip-{ordinal:03d}",
                "capture_artifact_sha256": "c" * 64,
                "capture_artifact_file_sha256": "d" * 64,
                "pair_sha256": "e" * 64,
            }
            for ordinal in range(900)
        ],
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
            "sha256": "f" * 64,
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


@dataclass(frozen=True)
class _Response:
    payload: object
    headers: Mapping[str, object]
    status_code: int = 200

    @property
    def content(self) -> bytes:
        return json.dumps(self.payload, separators=(",", ":")).encode("ascii")

    def json(self) -> object:
        return self.payload


def _commission_artifact() -> dict[str, object]:
    def request(
        method: str,
        url: str,
        **kwargs: object,
    ) -> _Response:
        assert method == "GET"
        if url.endswith("/fapi/v1/time"):
            return _Response(
                {"serverTime": 1_800_000_000_000},
                {"X-MBX-USED-WEIGHT-1M": "1"},
            )
        symbol = str(dict(kwargs["params"])["symbol"])
        return _Response(
            {
                "symbol": symbol,
                "makerCommissionRate": "0.0002",
                "takerCommissionRate": "0.0004",
                "rpiCommissionRate": "0.00005",
            },
            {"X-MBX-USED-WEIGHT-1M": "21"},
        )

    capture = capture_round74_mainnet_commission(
        api_key="test-key",
        api_secret="test-secret",
        timeout_seconds=5.0,
        request=request,
    )
    return {
        "schema_version": "round-074-commission-artifact-v1",
        "capture": capture.as_dict(),
        "execution_git_commit": "1" * 40,
        "source": {
            "capture_module": ("src/simple_ai_trading/round74_commission_capture.py"),
            "capture_module_sha256": "2" * 64,
            "capture_tool": "tools/capture_round74_commission_evidence.py",
            "capture_tool_sha256": "3" * 64,
        },
        "credential_transport": {
            "source": "process_environment_only",
            "api_key_variable": "TEST_KEY",
            "api_secret_variable": "TEST_SECRET",
            "credential_values_persisted": False,
        },
    }


def _source_panel(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    binding = _binding()
    cohort = build_round74_cohort_capture_source_payload(
        binding=binding,
        database_relative_path="data/test.duckdb",
    )
    (source_root / "cohort.json").write_bytes(_canonical_bytes(cohort))
    (source_root / "exchange.json").write_bytes(REAL_EXCHANGE_INFO.read_bytes())
    commission = _commission_artifact()
    _write_hashed(source_root / "commission.json", commission)
    funding = json.loads(REAL_FUNDING.read_text(encoding="ascii"))
    funding.pop("artifact_sha256")
    funding["capture_binding"]["run_id"] = binding.run_id
    _write_hashed(source_root / "funding.json", funding)
    _write_hashed(source_root / "aggregate.json", _aggregate_payload())
    aggregate = load_round74_execution_aggregate_source(source_root / "aggregate.json")
    scenario = build_round74_public_execution_scenario(
        transport_source=Round74PublicTransportSource(
            run_id=binding.run_id,
            cohort_binding_sha256=binding.binding_sha256,
            capture_report_sha256=binding.report_sha256,
            observed_wall_ns=3_000_000_000,
            source_payload_sha256="4" * 64,
            transport_latency_ns_by_symbol=tuple(
                (symbol, (20_000_000,) * 300) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            ),
        ),
        execution_aggregate=aggregate,
    )
    _write_hashed(source_root / "scenario.json", scenario.as_dict())
    return source_root, {
        "cohort_capture": "cohort.json",
        "exchange_info": "exchange.json",
        "commission": "commission.json",
        "funding": "funding.json",
        "execution_calibration": "aggregate.json",
        "execution_scenario": "scenario.json",
    }


def test_public_target_operator_reopens_six_sources_and_round_trips(
    tmp_path: Path,
) -> None:
    source_root, sources = _source_panel(tmp_path)

    manifest = build_round74_public_target_manifest(
        source_artifact_root=source_root,
        source_relative_paths=sources,
    )
    target = write_round74_public_target_manifest(
        manifest=manifest,
        output_directory=tmp_path / "manifests",
    )
    restored = load_and_audit_round74_target_assembly_manifest(
        manifest_path=target,
        source_artifact_root=source_root,
    )

    assert restored.manifest_sha256 == manifest.manifest_sha256
    assert restored.run_id == _binding().run_id
    assert len(restored.source_artifacts) == 6
    assert restored.assembly.spec.execution_environment == ("binance_usdm_mainnet")
    assert (
        write_round74_public_target_manifest(
            manifest=manifest,
            output_directory=tmp_path / "manifests",
        )
        == target
    )


def test_public_target_operator_rejects_tamper_and_path_escape(
    tmp_path: Path,
) -> None:
    source_root, sources = _source_panel(tmp_path)
    commission = source_root / sources["commission"]
    commission.write_bytes(commission.read_bytes() + b" ")

    with pytest.raises(ValueError, match="encoding differs"):
        build_round74_public_target_manifest(
            source_artifact_root=source_root,
            source_relative_paths=sources,
        )

    sources["commission"] = "../commission.json"
    with pytest.raises(ValueError, match="relative path differs"):
        build_round74_public_target_manifest(
            source_artifact_root=source_root,
            source_relative_paths=sources,
        )


def test_cohort_source_loader_rejects_changed_binding() -> None:
    binding = _binding()
    payload = build_round74_cohort_capture_source_payload(
        binding=binding,
        database_relative_path="data/test.duckdb",
    )
    payload["cohort_binding"]["message_count"] = 999
    path = ROOT / "not-written.json"

    assert not path.exists()
    with pytest.raises(ValueError, match="binding.*differs"):
        from simple_ai_trading.round74_public_target_sources import (
            audit_round74_cohort_capture_source_payload,
        )

        audit_round74_cohort_capture_source_payload(
            {key: value for key, value in payload.items() if key != "artifact_sha256"},
            run_id=binding.run_id,
            cohort_binding_sha256=binding.binding_sha256,
        )


def test_canonical_source_loader_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"x","schema_version":"y","artifact_sha256":"'
        + "0" * 64
        + '"}\n',
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="JSON differs"):
        load_round74_canonical_source_artifact(path, label="duplicate")


def test_execution_scenario_loader_accepts_canonical_cohort_source(
    tmp_path: Path,
) -> None:
    binding = _binding()
    payload = build_round74_cohort_capture_source_payload(
        binding=binding,
        database_relative_path="data/test.duckdb",
    )
    path = tmp_path / "cohort.json"
    path.write_bytes(_canonical_bytes(payload))

    restored = load_execution_scenario_binding(path)

    assert restored.as_dict() == binding.as_dict()
