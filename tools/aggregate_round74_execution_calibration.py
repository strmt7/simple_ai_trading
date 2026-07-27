"""Aggregate a complete Round 74 testnet execution-calibration campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from simple_ai_trading.impact_absorption_execution_evidence import (
    Round74ExecutionEvidenceBundle,
    build_round74_execution_calibration_evidence,
)

if __package__:
    from tools import capture_round74_execution_calibration as capture_tool
else:
    import capture_round74_execution_calibration as capture_tool


ROUND74_EXECUTION_AGGREGATE_SCHEMA_VERSION = (
    "round-074-execution-calibration-aggregate-v1"
)
_MAXIMUM_CAPTURE_ARTIFACT_BYTES = 4 * 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(
    *,
    output_directory: Path,
    payload: Mapping[str, object],
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["artifact_sha256"] = _canonical_sha256(value)
    encoded = (_canonical_json(value) + "\n").encode("ascii")
    target = output_directory / f"{value['artifact_sha256']}.json"
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError("existing execution aggregate bytes differ")
        return target
    temporary = target.with_suffix(".json.tmp")
    if temporary.exists():
        raise RuntimeError("execution aggregate temporary path already exists")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return target


def _evidence_payload(
    bundle: Round74ExecutionEvidenceBundle,
) -> dict[str, object]:
    return {
        "reference_quote_notional": bundle.reference_quote_notional,
        "decision_to_entry_latency_ns_by_symbol": (bundle.entry_latency_mapping()),
        "decision_to_exit_latency_ns_by_symbol": (bundle.exit_latency_mapping()),
        "additional_slippage_bps_per_side_by_symbol": (bundle.slippage_mapping()),
        "entry_exit_latency_evidence": (bundle.entry_exit_latency_evidence.as_dict()),
        "residual_slippage_evidence": (bundle.residual_slippage_evidence.as_dict()),
    }


def _load_complete_campaign_records(
    *,
    plan: object,
    plan_artifact_sha256: str,
    capture_directory: Path,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[dict[str, object], ...],
    int,
]:
    slots = tuple(getattr(plan, "slots", ()))
    if not slots or not capture_directory.is_dir():
        raise ValueError("execution calibration campaign captures differ")
    completed = capture_tool._validated_campaign_capture_slots(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        capture_directory=capture_directory,
    )
    expected_ordinals = tuple(range(len(slots)))
    if completed != expected_ordinals:
        raise ValueError("execution calibration campaign is incomplete")

    by_round_trip_id = {str(getattr(slot, "round_trip_id")): slot for slot in slots}
    records_by_ordinal: dict[int, tuple[Mapping[str, object], ...]] = {}
    sources_by_ordinal: dict[int, dict[str, object]] = {}
    observed_wall_ns = 0
    paths = tuple(sorted(capture_directory.glob("*.json")))
    if len(paths) != len(slots):
        raise ValueError("execution calibration campaign file count differs")
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > _MAXIMUM_CAPTURE_ARTIFACT_BYTES
        ):
            raise ValueError("execution calibration capture file differs")
        payload, artifact_sha256 = capture_tool._load_canonical_artifact(path)
        binding = payload.get("campaign_binding")
        result = payload.get("result")
        if not isinstance(binding, Mapping) or not isinstance(result, Mapping):
            raise ValueError("execution calibration capture binding differs")
        pair = result.get("pair")
        if not isinstance(pair, Mapping):
            raise ValueError("execution calibration capture pair differs")
        ordinal = binding.get("slot_ordinal")
        round_trip_id = str(binding.get("round_trip_id", ""))
        slot = by_round_trip_id.get(round_trip_id)
        raw_records = pair.get("records")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or slot is None
            or ordinal != int(getattr(slot, "ordinal"))
            or ordinal in records_by_ordinal
            or not isinstance(raw_records, list)
            or len(raw_records) != 2
            or any(not isinstance(record, Mapping) for record in raw_records)
        ):
            raise ValueError("execution calibration capture identity differs")
        captured_at_wall_ns = payload.get("captured_at_wall_ns")
        if (
            isinstance(captured_at_wall_ns, bool)
            or not isinstance(captured_at_wall_ns, int)
            or captured_at_wall_ns <= 0
        ):
            raise ValueError("execution calibration capture time differs")
        pair_payload = dict(pair)
        pair_sha256 = str(pair_payload.pop("pair_sha256", ""))
        records_by_ordinal[ordinal] = tuple(raw_records)
        sources_by_ordinal[ordinal] = {
            "slot_ordinal": ordinal,
            "round_trip_id": round_trip_id,
            "capture_artifact_sha256": artifact_sha256,
            "capture_artifact_file_sha256": _file_sha256(path),
            "pair_sha256": pair_sha256,
        }
        observed_wall_ns = max(observed_wall_ns, captured_at_wall_ns)

    return (
        tuple(
            record
            for ordinal in expected_ordinals
            for record in records_by_ordinal[ordinal]
        ),
        tuple(sources_by_ordinal[ordinal] for ordinal in expected_ordinals),
        observed_wall_ns,
    )


def aggregate_round74_execution_calibration(
    *,
    campaign_plan_path: Path,
    capture_directory: Path,
    output_directory: Path,
) -> Path:
    """Validate every campaign source and publish testnet-only tail evidence."""

    selected_plan_path = campaign_plan_path.resolve()
    selected_capture_directory = capture_directory.resolve()
    if (
        campaign_plan_path.is_symlink()
        or not selected_plan_path.is_file()
        or capture_directory.is_symlink()
    ):
        raise ValueError("execution calibration aggregate input differs")
    plan, plan_artifact_sha256 = capture_tool._load_campaign_plan_artifact(
        selected_plan_path
    )
    records, source_artifacts, observed_wall_ns = _load_complete_campaign_records(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        capture_directory=selected_capture_directory,
    )
    bundle = build_round74_execution_calibration_evidence(
        records=records,
        environment="binance_usdm_testnet",
        observed_wall_ns=observed_wall_ns,
        reference_quote_notional=float(plan.target_quote_notional),
    )
    payload: dict[str, object] = {
        "schema_version": ROUND74_EXECUTION_AGGREGATE_SCHEMA_VERSION,
        "operation": "aggregate",
        "environment": "binance_usdm_testnet",
        "campaign_plan": {
            "plan_sha256": plan.plan_sha256,
            "plan_artifact_sha256": plan_artifact_sha256,
            "plan_artifact_file_sha256": _file_sha256(selected_plan_path),
            "campaign_id": plan.campaign_id,
            "slot_count": len(plan.slots),
        },
        "source_capture_artifacts": list(source_artifacts),
        "source_capture_artifact_count": len(source_artifacts),
        "source_record_count": len(records),
        "observed_wall_ns": observed_wall_ns,
        "execution_evidence": _evidence_payload(bundle),
        "aggregator_source": {
            "path": "tools/aggregate_round74_execution_calibration.py",
            "sha256": _file_sha256(Path(__file__).resolve()),
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
    return _write_artifact(
        output_directory=output_directory.resolve(),
        payload=payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and aggregate one complete Round 74 Binance testnet "
            "execution-calibration campaign without network or database access."
        )
    )
    parser.add_argument("--campaign-plan", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        target = aggregate_round74_execution_calibration(
            campaign_plan_path=args.campaign_plan,
            capture_directory=args.capture_directory,
            output_directory=args.output_directory,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        _canonical_json(
            {
                "artifact": str(target),
                "artifact_sha256": target.stem,
                "environment": "binance_usdm_testnet",
                "mainnet_transfer_permitted": False,
                "network_accessed": False,
                "orders_submitted": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
