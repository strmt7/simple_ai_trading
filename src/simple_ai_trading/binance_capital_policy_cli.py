"""Offline operator inspection, activation and exact prepared-policy recovery."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

from .binance_capital_policy import BinanceCapitalPolicy, BinanceCapitalPolicyStore
from .binance_runtime_control import BinanceRuntimeControl, RuntimeControlError
from .positions import PositionsStore


def register_binance_capital_policy_command(subparsers):
    parser = subparsers.add_parser(
        "capital-policy",
        help="inspect or explicitly activate an enrolled Binance loss policy offline",
    )
    parser.add_argument(
        "--action", choices=["inspect", "activate", "recover"], default="inspect"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        help="activate only: private strict policy JSON, at most 8192 bytes",
    )
    parser.add_argument(
        "--expected-head-sha256", help="activate only: exact inspected policy head"
    )
    parser.add_argument(
        "--expected-store-id", help="recover only: prepared policy storage ID"
    )
    parser.add_argument(
        "--expected-record-sha256",
        help="recover only: exact prepared enrollment digest",
    )
    parser.add_argument("--runtime-control", type=Path)
    parser.add_argument("--runtime-enrollment", type=Path)
    parser.add_argument("--execution-enrollment", type=Path)
    parser.add_argument("--policy-enrollment", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=command_capital_policy)


def command_capital_policy(args):
    payload = {
        "schema_version": "binance-capital-policy-operator-v1",
        "action": args.action,
        "account_queried": False,
        "order_submitted": False,
        "rearmed": False,
        "activation_outcome": "not_started"
        if args.action != "inspect"
        else "not_requested",
    }
    status = 2
    try:
        activation = args.action == "activate"
        recovery = args.action == "recover"
        if (
            (activation and (args.policy is None or args.expected_head_sha256 is None))
            or (
                not activation
                and (args.policy is not None or args.expected_head_sha256 is not None)
            )
            or (
                recovery
                and (
                    args.expected_store_id is None
                    or args.expected_record_sha256 is None
                )
            )
            or (
                not recovery
                and (
                    args.expected_store_id is not None
                    or args.expected_record_sha256 is not None
                )
            )
        ):
            raise ValueError("capital policy action options differ")
        store = PositionsStore(args.data_dir)
        control = BinanceRuntimeControl(
            args.runtime_control or store.root / "binance_runtime_control.sqlite3",
            enrollment_path=args.runtime_enrollment,
        )
        policies = BinanceCapitalPolicyStore(
            store,
            control=control,
            execution_enrollment_path=args.execution_enrollment,
            enrollment_path=args.policy_enrollment,
        )
        if activation:
            with args.policy.open("rb") as source:
                policy = BinanceCapitalPolicy.parse(source.read(8193))
            payload["activation_outcome"] = "unknown"
            result = policies.activate(
                policy, expected_head_sha256=args.expected_head_sha256
            )
            payload["activation_outcome"] = "active"
        elif recovery:
            payload["activation_outcome"] = "unknown"
            result = policies.recover(
                expected_store_id=args.expected_store_id,
                expected_record_sha256=args.expected_record_sha256,
            )
            payload["activation_outcome"] = "active"
        else:
            result = policies.inspect(allow_prepared=True)
        payload.update(
            status="inspected" if args.action == "inspect" else "active", result=result
        )
        status = 0
    except (
        OSError,
        sqlite3.Error,
        ValueError,
        TypeError,
        KeyError,
        RuntimeControlError,
    ):
        payload.update(
            status="blocked",
            message="Capital policy storage, scope, document or action is unavailable, changed or incompatible. Preserve originals; inspect before retrying. Stop was not cleared.",
        )
    output = sys.stdout if status == 0 else sys.stderr
    print(
        json.dumps(payload, indent=2)
        if args.json
        else f"capital-policy: {payload['status']} {json.dumps(payload.get('result', {}), sort_keys=True) if status == 0 else payload['message']}",
        file=output,
    )
    return status
