"""Adjudicate the frozen CRWDB/CRWD top-book parity prefilter offline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "binance-crwdb-bstock-perpetual-topbook-contract-v1"
RESULT_SCHEMA = "binance-crwdb-bstock-perpetual-topbook-result-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _verify_self_hash(value: Mapping[str, object], field: str, name: str) -> None:
    if _canonical_hash(value, field) != value.get(field):
        raise ValueError(f"{name} canonical hash mismatch")


def _load_source(binding: Mapping[str, object]) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = ROOT / str(binding["result_path"])
    source = _load_object(result_path)
    _verify_self_hash(source, "result_sha256", result_path.name)
    if source.get("contract") != {
        "path": binding.get("contract_path"),
        "sha256": binding.get("contract_sha256"),
    }:
        raise ValueError(f"source contract binding mismatch: {result_path.name}")
    if source.get("source_gate", {}).get("passed") is not True:
        raise ValueError(f"source gate failed: {result_path.name}")
    receipt = source.get("capture", {}).get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"source receipt missing: {result_path.name}")
    raw_path = ROOT / str(receipt["raw_path"])
    raw = raw_path.read_bytes()
    if _sha256(raw) != receipt.get("response_sha256"):
        raise ValueError(f"raw source hash mismatch: {raw_path.name}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"book payload is not an object: {raw_path.name}")
    return source, payload


def _decimal(value: object) -> Decimal:
    return Decimal(str(value or "0"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite CRWDB prefilter evidence")

    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unexpected contract schema")
    _verify_self_hash(contract, "contract_sha256", "CRWDB contract")
    implementation = contract.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("implementation binding missing")
    if _sha256(Path(__file__).read_bytes()) != implementation.get("sha256"):
        raise ValueError("implementation hash mismatch")

    inventory_binding = contract.get("inventory_result")
    if not isinstance(inventory_binding, Mapping):
        raise ValueError("inventory result binding missing")
    inventory = _load_object(ROOT / str(inventory_binding["path"]))
    if inventory.get("result_sha256") != inventory_binding.get("result_sha256"):
        raise ValueError("inventory result declared hash mismatch")
    inventory_hash_body = dict(inventory)
    inventory_claim = str(inventory_hash_body.pop("result_sha256"))
    if _sha256(_canonical_json(inventory_hash_body).encode()) != inventory_claim:
        raise ValueError("inventory result canonical hash mismatch")
    if inventory.get("next_selected_ticker") != "CRWD":
        raise ValueError("frozen deterministic selector did not choose CRWD")
    expected_match = {
        "bstock_multiplier": "1",
        "bstock_spot_symbol": "CRWDBUSDT",
        "futures_contract_type": "TRADIFI_PERPETUAL",
        "futures_status": "TRADING",
        "futures_symbol": "CRWDUSDT",
        "futures_underlying_type": "EQUITY",
        "ticker": "CRWD",
    }
    if expected_match not in inventory.get("matching_unscreened_pairs", []):
        raise ValueError("CRWD exact-one matching pair is absent")

    sources = contract.get("source_results")
    if not isinstance(sources, Mapping):
        raise ValueError("source bindings missing")
    spot_source, spot = _load_source(sources["spot_book"])
    futures_source, future = _load_source(sources["futures_book"])
    if spot.get("symbol") != "CRWDBUSDT" or future.get("symbol") != "CRWDUSDT":
        raise ValueError("book symbol identity mismatch")
    spot_ask = _decimal(spot.get("askPrice"))
    spot_ask_qty = _decimal(spot.get("askQty"))
    futures_bid = _decimal(future.get("bidPrice"))
    futures_bid_qty = _decimal(future.get("bidQty"))
    positive_entry = (
        spot_ask > 0
        and spot_ask_qty > 0
        and futures_bid > 0
        and futures_bid_qty > 0
    )
    fixed_bps = _decimal(contract["economic_gate"]["fixed_stress_bps"])
    if fixed_bps != Decimal("50"):
        raise ValueError("unexpected fixed stress")
    gross = futures_bid - spot_ask
    fixed_cost = spot_ask * fixed_bps / Decimal("10000")
    after_fixed = gross - fixed_cost
    after_fixed_bps = (
        after_fixed / spot_ask * Decimal("10000")
        if spot_ask > 0
        else Decimal("-Infinity")
    )
    passes = positive_entry and after_fixed > 0
    skew_ms = abs(
        spot_source["capture"]["receipt"]["requested_at_ms"]
        - futures_source["capture"]["receipt"]["requested_at_ms"]
    )
    if skew_ms > contract["capture"]["maximum_request_start_skew_ms"]:
        raise ValueError("capture start skew exceeded")
    common_top_quantity = min(spot_ask_qty, futures_bid_qty)

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": {
            "path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": contract["contract_sha256"],
        },
        "inventory_result": inventory_binding,
        "capture": {
            "request_count": 2,
            "capture_skew_ms": skew_ms,
            "spot_source_result_sha256": spot_source["result_sha256"],
            "futures_source_result_sha256": futures_source["result_sha256"],
        },
        "identity": expected_match,
        "economics": {
            "spot_ask_USDT_per_share": format(spot_ask, "f"),
            "spot_ask_quantity_shares": format(spot_ask_qty, "f"),
            "futures_bid_USDT_per_share": format(futures_bid, "f"),
            "futures_bid_quantity_shares": format(futures_bid_qty, "f"),
            "common_top_quantity_shares_diagnostic_only": format(
                common_top_quantity, "f"
            ),
            "positive_entry_sides": positive_entry,
            "gross_entry_headroom_USDT_per_share": format(gross, "f"),
            "fixed_stress_bps": format(fixed_bps, "f"),
            "fixed_stress_USDT_per_share": format(fixed_cost, "f"),
            "after_fixed_stress_USDT_per_share": format(after_fixed, "f"),
            "after_fixed_stress_bps": format(after_fixed_bps, "f"),
            "passes_fixed_rejection_gate": passes,
            "favorable_funding_credited": False,
            "full_depth_verified": False,
            "account_conversion_verified": False,
        },
        "adjudication": {
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "depth_requests": 0,
            "funding_requests": 0,
            "next_action": (
                "freeze_a_separate_full_depth_adverse_funding_account_cost_and_exit_basis_stress"
                if passes
                else "terminalize_the_exact_CRWDBUSDT_CRWDUSDT_topbook_snapshot_without_depth_funding_account_credential_order_or_fund_access"
            ),
        },
        "authority": {
            "account_state_accessed": False,
            "authenticated_requests": 0,
            "credentials_used": False,
            "funds_used": False,
            "orders_quotes_transfers_or_wallet_actions": 0,
            "paper_or_live_trading_authority": False,
            "protected_capture_touched": False,
            "public_unauthenticated_GET_requests": 2,
        },
        "implementation": implementation,
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    write_bytes_atomic(
        args.output, (_canonical_json(result) + "\n").encode("ascii")
    )
    print(
        _canonical_json(
            {
                "gross_bps": format(
                    gross / spot_ask * Decimal("10000")
                    if spot_ask > 0
                    else Decimal("-Infinity"),
                    "f",
                ),
                "after_fixed_bps": format(after_fixed_bps, "f"),
                "passes": passes,
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
