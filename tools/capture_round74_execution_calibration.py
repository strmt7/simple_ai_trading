"""Capture or recover one Round 74 USD-M testnet execution pair."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import duckdb

import simple_ai_trading.impact_absorption_execution_evidence as evidence_module
import simple_ai_trading.round74_execution_calibration_campaign as campaign_module
import simple_ai_trading.round74_execution_calibration_capture as capture_module
import simple_ai_trading.round74_execution_calibration_coordinator as coordinator_module
import simple_ai_trading.round74_execution_calibration_journal as journal_module
import simple_ai_trading.round74_execution_calibration_sizing as sizing_module
import simple_ai_trading.round74_execution_calibration_transport as transport_module
from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SYMBOLS,
)
from simple_ai_trading.round74_execution_calibration_campaign import (
    Round74ExecutionCampaignPlan,
    build_round74_execution_campaign_plan,
)
from simple_ai_trading.round74_execution_calibration_coordinator import (
    capture_round74_execution_calibration_pair,
    recover_round74_execution_calibration,
)
from simple_ai_trading.round74_execution_calibration_journal import (
    Round74ExecutionCalibrationJournal,
)
from simple_ai_trading.round74_execution_calibration_sizing import (
    prepare_round74_execution_sizing,
)
from simple_ai_trading.round74_execution_calibration_transport import (
    Round74BinanceTestnetExecutionTransport,
)


ROUND74_EXECUTION_TOOL_SCHEMA_VERSION = "round-074-execution-calibration-tool-v1"
API_KEY_ENV = "SIMPLE_AI_TRADING_BINANCE_TESTNET_API_KEY"
API_SECRET_ENV = "SIMPLE_AI_TRADING_BINANCE_TESTNET_API_SECRET"
_SOURCE_MODULES = (
    evidence_module,
    campaign_module,
    capture_module,
    coordinator_module,
    journal_module,
    sizing_module,
    transport_module,
)


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


def _decimal(value: object, *, label: str) -> Decimal:
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive decimal") from exc
    if not selected.is_finite() or selected <= 0:
        raise ValueError(f"{label} must be a positive decimal")
    return selected


def _source_hashes() -> dict[str, str]:
    selected: dict[str, str] = {}
    tool_path = Path(__file__).resolve()
    selected[tool_path.name] = hashlib.sha256(tool_path.read_bytes()).hexdigest()
    for module in _SOURCE_MODULES:
        module_path = Path(str(module.__file__)).resolve()
        selected[module_path.name] = hashlib.sha256(
            module_path.read_bytes()
        ).hexdigest()
    return dict(sorted(selected.items()))


def _write_artifact(
    *,
    output_directory: Path,
    payload: dict[str, object],
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["artifact_sha256"] = _canonical_sha256(value)
    encoded = (_canonical_json(value) + "\n").encode("ascii")
    target = output_directory / f"{value['artifact_sha256']}.json"
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError("existing execution artifact bytes differ")
        return target
    temporary = target.with_suffix(".json.tmp")
    if temporary.exists():
        raise RuntimeError("execution artifact temporary path already exists")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return target


def _load_canonical_artifact(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("execution artifact cannot be read") from exc
    if not raw or len(raw) > 4 * 1024 * 1024:
        raise ValueError("execution artifact size differs")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("execution artifact JSON differs") from exc
    if not isinstance(payload, dict):
        raise ValueError("execution artifact root differs")
    try:
        canonical = (_canonical_json(payload) + "\n").encode("ascii")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("execution artifact JSON differs") from exc
    if raw != canonical:
        raise ValueError("execution artifact is not canonical")
    artifact_sha256 = str(payload.pop("artifact_sha256", ""))
    if artifact_sha256 != _canonical_sha256(payload):
        raise ValueError("execution artifact digest differs")
    return payload, artifact_sha256


def _load_campaign_plan_artifact(
    path: Path,
) -> tuple[Round74ExecutionCampaignPlan, str]:
    payload, artifact_sha256 = _load_canonical_artifact(path)
    if (
        payload.get("operation") != "plan"
        or payload.get("environment") != "binance_usdm_testnet"
        or payload.get("network_accessed") is not False
        or payload.get("orders_submitted") is not False
        or payload.get("source_sha256") != _source_hashes()
        or not isinstance(payload.get("plan"), dict)
    ):
        raise ValueError("campaign plan artifact authority differs")
    plan = Round74ExecutionCampaignPlan.from_dict(payload["plan"])
    return plan, artifact_sha256


def _credentials() -> tuple[str, str]:
    api_key = os.environ.get(API_KEY_ENV, "")
    api_secret = os.environ.get(API_SECRET_ENV, "")
    if not api_key or not api_secret:
        raise RuntimeError(
            f"set {API_KEY_ENV} and {API_SECRET_ENV}; values are never persisted"
        )
    return api_key, api_secret


def _require_order_confirmation(args: argparse.Namespace) -> None:
    if not args.yes or not args.acknowledge_non_mainnet_orders:
        raise RuntimeError(
            "network mode requires --yes and --acknowledge-non-mainnet-orders"
        )


def _base_payload(*, operation: str) -> dict[str, object]:
    return {
        "schema_version": ROUND74_EXECUTION_TOOL_SCHEMA_VERSION,
        "operation": operation,
        "captured_at_wall_ns": time.time_ns(),
        "environment": "binance_usdm_testnet",
        "rest_base_url": (transport_module.ROUND74_EXECUTION_TRANSPORT_REST_BASE_URL),
        "websocket_base_url": (
            transport_module.ROUND74_EXECUTION_TRANSPORT_WS_BASE_URL
        ),
        "source_sha256": _source_hashes(),
        "credential_material_persisted": False,
        "signed_query_persisted": False,
        "mainnet_orders_submitted": False,
        "mainnet_trading_authority": False,
        "profitability_claim": False,
    }


def command_status(args: argparse.Namespace) -> int:
    database = Path(args.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as connection:
        journal = Round74ExecutionCalibrationJournal(connection)
        journal.verify()
        blockers = journal.blocking_round_trip_ids()
        snapshots = journal.current_snapshots()
    payload = {
        **_base_payload(operation="status"),
        "database": str(database),
        "blocking_round_trip_ids": list(blockers),
        "intent_count": len(snapshots),
        "network_accessed": False,
        "orders_submitted": False,
    }
    print(_canonical_json(payload))
    return 1 if blockers else 0


def command_plan(args: argparse.Namespace) -> int:
    plan = build_round74_execution_campaign_plan(
        campaign_id=args.campaign_id,
        target_quote_notional=_decimal(
            args.target_quote_notional,
            label="target quote notional",
        ),
        pairs_per_symbol=args.pairs_per_symbol,
    )
    payload = {
        **_base_payload(operation="plan"),
        "plan": plan.as_dict(),
        "network_accessed": False,
        "orders_submitted": False,
    }
    target = _write_artifact(
        output_directory=Path(args.output_directory),
        payload=payload,
    )
    print(
        _canonical_json(
            {
                "artifact": str(target),
                "plan_sha256": plan.plan_sha256,
                "slot_count": len(plan.slots),
                "pairs_per_symbol": plan.pairs_per_symbol,
            }
        )
    )
    return 0


def _validated_campaign_capture_slots(
    *,
    plan: Round74ExecutionCampaignPlan,
    plan_artifact_sha256: str,
    capture_directory: Path,
) -> tuple[int, ...]:
    if not capture_directory.exists():
        return ()
    if not capture_directory.is_dir():
        raise ValueError("campaign capture directory differs")
    by_ordinal = {slot.ordinal: slot for slot in plan.slots}
    completed: list[int] = []
    source_hashes = _source_hashes()
    for path in sorted(capture_directory.glob("*.json")):
        payload, artifact_sha256 = _load_canonical_artifact(path)
        if path.stem != artifact_sha256:
            raise ValueError("campaign capture artifact filename differs")
        binding = payload.get("campaign_binding")
        result = payload.get("result")
        sizing = payload.get("sizing")
        if (
            payload.get("operation") != "capture"
            or payload.get("environment") != "binance_usdm_testnet"
            or payload.get("source_sha256") != source_hashes
            or not isinstance(binding, dict)
            or not isinstance(result, dict)
            or not isinstance(sizing, dict)
            or binding.get("campaign_plan_artifact_sha256") != plan_artifact_sha256
            or binding.get("campaign_plan_sha256") != plan.plan_sha256
            or result.get("evidence_admitted") is not True
            or not isinstance(result.get("pair"), dict)
        ):
            raise ValueError("campaign capture artifact authority differs")
        ordinal = binding.get("slot_ordinal")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal not in by_ordinal
            or ordinal in completed
        ):
            raise ValueError("campaign capture slot identity differs")
        slot = by_ordinal[ordinal]
        pair = dict(result["pair"])
        pair_sha256 = str(pair.pop("pair_sha256", ""))
        records = pair.get("records")
        if (
            pair_sha256 != _canonical_sha256(pair)
            or binding.get("round_trip_id") != slot.round_trip_id
            or pair.get("calibration_run_id") != plan.campaign_id
            or pair.get("round_trip_id") != slot.round_trip_id
            or pair.get("symbol") != slot.symbol
            or sizing.get("symbol") != slot.symbol
            or sizing.get("entry_side") != slot.entry_side
            or sizing.get("target_quote_notional")
            != format(plan.target_quote_notional, "f")
            or not isinstance(records, list)
            or len(records) != 2
            or not any(
                isinstance(record, dict)
                and record.get("path") == "entry"
                and record.get("side") == slot.entry_side
                for record in records
            )
        ):
            raise ValueError("campaign capture pair identity differs")
        completed.append(ordinal)
    return tuple(sorted(completed))


def command_campaign_status(args: argparse.Namespace) -> int:
    plan, artifact_sha256 = _load_campaign_plan_artifact(Path(args.campaign_plan))
    completed_ordinals = _validated_campaign_capture_slots(
        plan=plan,
        plan_artifact_sha256=artifact_sha256,
        capture_directory=Path(args.capture_directory),
    )
    completed_ids = tuple(
        plan.slots[ordinal].round_trip_id for ordinal in completed_ordinals
    )
    next_slot = plan.next_slot(
        completed_round_trip_ids=completed_ids,
    )
    print(
        _canonical_json(
            {
                "campaign_plan_sha256": plan.plan_sha256,
                "completed_slot_count": len(completed_ordinals),
                "total_slot_count": len(plan.slots),
                "complete": next_slot is None,
                "next_slot": (next_slot.as_dict() if next_slot is not None else None),
                "network_accessed": False,
                "orders_submitted": False,
            }
        )
    )
    return 0


def command_recover(args: argparse.Namespace) -> int:
    _require_order_confirmation(args)
    api_key, api_secret = _credentials()
    database = Path(args.database)
    output_directory = Path(args.output_directory)
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as connection:
        journal = Round74ExecutionCalibrationJournal(connection)
        with Round74BinanceTestnetExecutionTransport(
            api_key=api_key,
            api_secret=api_secret,
            timeout_seconds=args.timeout_seconds,
        ) as transport:
            result = recover_round74_execution_calibration(
                transport=transport,
                journal=journal,
            )
            rate_limits = dict(transport.last_rate_limit_headers)
        payload = {
            **_base_payload(operation="recover"),
            "database": str(database),
            "result": result.as_dict(),
            "last_rate_limit_headers": rate_limits,
            "orders_may_have_been_submitted": True,
        }
        target = _write_artifact(
            output_directory=output_directory,
            payload=payload,
        )
    print(
        _canonical_json(
            {
                "artifact": str(target),
                "complete": result.complete,
                "blocking_round_trip_ids": list(result.blocking_round_trip_ids),
            }
        )
    )
    return 0 if result.complete else 1


def command_capture(args: argparse.Namespace) -> int:
    _require_order_confirmation(args)
    api_key, api_secret = _credentials()
    database = Path(args.database)
    output_directory = Path(args.output_directory)
    target_quote_notional = _decimal(
        args.target_quote_notional,
        label="target quote notional",
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database)) as connection:
        journal = Round74ExecutionCalibrationJournal(connection)
        with Round74BinanceTestnetExecutionTransport(
            api_key=api_key,
            api_secret=api_secret,
            timeout_seconds=args.timeout_seconds,
        ) as transport:
            recovery = recover_round74_execution_calibration(
                transport=transport,
                journal=journal,
            )
            if not recovery.complete:
                raise RuntimeError("unresolved calibration exposure blocks a new pair")
            sizing = prepare_round74_execution_sizing(
                symbol=args.symbol,
                entry_side=args.entry_side,
                target_quote_notional=target_quote_notional,
                exchange_information=transport.exchange_information(args.symbol),
                mark_price=transport.mark_price(args.symbol),
                book=transport.book(args.symbol),
            )
            result = capture_round74_execution_calibration_pair(
                transport=transport,
                journal=journal,
                calibration_run_id=args.calibration_run_id,
                round_trip_id=args.round_trip_id,
                symbol=args.symbol,
                entry_side=args.entry_side,
                quantity=sizing.quantity,
                reference_quote_notional=(sizing.reference_quote_notional),
            )
            rate_limits = dict(transport.last_rate_limit_headers)
        payload = {
            **_base_payload(operation="capture"),
            "database": str(database),
            "campaign_binding": getattr(args, "campaign_binding", None),
            "recovery_before_capture": recovery.as_dict(),
            "sizing": sizing.as_dict(),
            "result": result.as_dict(),
            "last_rate_limit_headers": rate_limits,
            "orders_may_have_been_submitted": True,
        }
        target = _write_artifact(
            output_directory=output_directory,
            payload=payload,
        )
    print(
        _canonical_json(
            {
                "artifact": str(target),
                "evidence_admitted": result.evidence_admitted,
                "pair_sha256": (
                    result.pair.pair_sha256 if result.pair is not None else None
                ),
            }
        )
    )
    return 0 if result.evidence_admitted else 1


def command_capture_slot(args: argparse.Namespace) -> int:
    _require_order_confirmation(args)
    plan, artifact_sha256 = _load_campaign_plan_artifact(Path(args.campaign_plan))
    if (
        isinstance(args.slot, bool)
        or not isinstance(args.slot, int)
        or args.slot < 0
        or args.slot >= len(plan.slots)
    ):
        raise ValueError("campaign slot ordinal differs")
    selected = plan.slots[args.slot]
    args.calibration_run_id = plan.campaign_id
    args.round_trip_id = selected.round_trip_id
    args.symbol = selected.symbol
    args.entry_side = selected.entry_side
    args.target_quote_notional = format(
        plan.target_quote_notional,
        "f",
    )
    args.campaign_binding = {
        "campaign_plan_artifact_sha256": artifact_sha256,
        "campaign_plan_sha256": plan.plan_sha256,
        "slot_ordinal": selected.ordinal,
        "round_trip_id": selected.round_trip_id,
    }
    return command_capture(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture bot-owned, flat-to-flat Binance USD-M test-environment "
            "execution evidence. This tool never targets mainnet."
        )
    )
    parser.add_argument(
        "--database",
        default="data/microstructure.duckdb",
        help="shared DuckDB evidence database",
    )
    parser.add_argument(
        "--output-directory",
        default="data/round74-execution-calibration",
        help="source-bound artifact directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status", help="inspect local blockers only")
    status.set_defaults(func=command_status)
    plan = subparsers.add_parser(
        "plan",
        help="write a deterministic balanced testnet campaign plan",
    )
    plan.add_argument("--campaign-id", required=True)
    plan.add_argument("--target-quote-notional", required=True)
    plan.add_argument(
        "--pairs-per-symbol",
        type=int,
        default=(
            evidence_module.ROUND74_EXECUTION_CALIBRATION_MINIMUM_PAIRS_PER_SYMBOL
        ),
    )
    plan.set_defaults(func=command_plan)
    campaign_status = subparsers.add_parser(
        "campaign-status",
        help="validate local captures and report the exact next slot",
    )
    campaign_status.add_argument("--campaign-plan", required=True)
    campaign_status.add_argument("--capture-directory", required=True)
    campaign_status.set_defaults(func=command_campaign_status)
    for name, handler in (
        ("recover", command_recover),
        ("capture", command_capture),
        ("capture-slot", command_capture_slot),
    ):
        selected = subparsers.add_parser(
            name,
            help=(
                "reconcile and reduce bot-owned testnet exposure"
                if name == "recover"
                else (
                    "recover first, then capture one plan-bound testnet pair"
                    if name == "capture-slot"
                    else "recover first, then capture one testnet pair"
                )
            ),
        )
        selected.add_argument("--yes", action="store_true")
        selected.add_argument(
            "--acknowledge-non-mainnet-orders",
            action="store_true",
            help="confirm that this mode can submit Binance testnet orders",
        )
        selected.add_argument(
            "--timeout-seconds",
            type=float,
            default=10.0,
        )
        selected.set_defaults(func=handler)
    capture = subparsers.choices["capture"]
    capture.add_argument("--calibration-run-id", required=True)
    capture.add_argument("--round-trip-id", required=True)
    capture.add_argument(
        "--symbol",
        choices=ROUND74_EVENT_TARGET_SYMBOLS,
        required=True,
    )
    capture.add_argument(
        "--entry-side",
        choices=("BUY", "SELL"),
        required=True,
    )
    capture.add_argument(
        "--target-quote-notional",
        required=True,
        help=(
            "maximum testnet quote notional; legal quantity is derived from "
            "live exchange filters, mark price, and captured depth"
        ),
    )
    capture_slot = subparsers.choices["capture-slot"]
    capture_slot.add_argument(
        "--campaign-plan",
        required=True,
        help="canonical source-bound campaign plan artifact",
    )
    capture_slot.add_argument("--slot", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
