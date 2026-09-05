"""Offline reconstruction of the consumed testnet campaign; no venue access."""

from __future__ import annotations

from collections import Counter
import base64
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from simple_ai_trading.spot_testnet_coverage import lifecycle_coverage
from simple_ai_trading.spot_testnet_evidence import owned_trade_cash
from tools.spot_testnet_campaign_transport import Journal, SYMBOLS

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("docs/review/2026-09-05/spot-testnet-execution")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(rows: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    """Rebuild each terminal cash row from owned observations and exact trades."""
    intents = {}
    identities = {}
    orders = {}
    trades = {}
    cash = {}
    for row in rows:
        kind = row["kind"]
        if kind == "order_intent":
            params = row["params"]
            cid = params["newClientOrderId"]
            if cid in intents:
                raise ValueError("duplicate intent")
            intents[cid] = params
        elif kind == "order_identity":
            cid = row["original_client_id"]
            identity = (row["symbol"], row["orderId"])
            if cid not in intents or intents[cid]["symbol"] != identity[0]:
                raise ValueError("foreign identity")
            if cid in identities and identities[cid] != identity:
                raise ValueError("changed numeric identity")
            if identity in identities.values() and identities.get(cid) != identity:
                raise ValueError("shared numeric identity")
            identities[cid] = identity
        elif kind == "order_observation":
            order = row["order"]
            identity = (order["symbol"], order["orderId"])
            if identity not in identities.values():
                raise ValueError("unowned observation")
            cid = next(k for k, v in identities.items() if v == identity)
            intent = intents[cid]
            if (
                order["clientOrderId"] not in {cid, cid + "c"}
                or order["side"] != intent["side"]
                or Decimal(order["origQty"]) != Decimal(intent["quantity"])
            ):
                raise ValueError("observation intent mismatch")
            orders[identity] = order
        elif kind == "owned_trades":
            # The frozen journal omits symbol on the envelope; reject ID collisions.
            matches = [k for k in orders if k[1] == row["orderId"]]
            if len(matches) != 1:
                raise ValueError("ambiguous trade envelope")
            trades[matches[0]] = row["trades"]
        elif kind == "order_cash":
            cid = row["client_id"]
            identity = identities[cid]
            symbol = identity[0]
            value = {
                "symbol": symbol,
                **owned_trade_cash(orders[identity], trades[identity], symbol[:-4]),
            }
            if any(row[k] != v for k, v in value.items()) or cid in cash:
                raise ValueError("cash mismatch or duplicate accounting")
            cash[cid] = value
    terminal = [r for r in rows if r["kind"] == "run_finished"]
    if len(terminal) != 1 or rows[-1] != terminal[0]:
        raise ValueError("missing final terminal record")
    recorded = {
        k: v
        for k, v in terminal[0].items()
        if k not in {"kind", "utc", "previous", "sha256"}
    }
    if recorded != result or result["order_cash"] != cash or set(cash) != set(intents):
        raise ValueError("result differs from complete reconstructed ledger")
    residual = {
        s: str(
            sum(
                (Decimal(v["base_delta"]) for v in cash.values() if v["symbol"] == s),
                Decimal(0),
            )
        )
        for s in SYMBOLS
    }
    quote = str(sum((Decimal(v["quote_delta"]) for v in cash.values()), Decimal(0)))
    if residual != result["residual_base"] or quote != result["quote_cash_delta"]:
        raise ValueError("aggregate accounting mismatch")
    coverage = lifecycle_coverage(rows, cash, SYMBOLS)
    requests = Counter(r["method"] for r in rows if r["kind"] == "http_intent")
    return {
        "schema_version": "spot-testnet-execution-offline-review-v1",
        "original_reporting_gate_passed": result["required_live_cases_passed"],
        "corrected_required_live_cases_passed": all(
            all(v.values()) for v in coverage.values()
        ),
        "live_case_coverage": coverage,
        "request_counts": dict(requests),
        "owned_order_intents": len(intents),
        "owned_trade_count": sum(v["trade_count"] for v in cash.values()),
        "quote_cash_delta_virtual_USDT": quote,
        "residual_base": residual,
        "partial_fills_observed": sum(v["partial_fill"] for v in cash.values()),
        "cold_process_restart_tested": result["cold_process_restart_tested"],
        "profitability_evidence": False,
        "network_requests_for_this_review": 0,
        "scope": "Normalized owned evidence replay, not origin HTTP bytes or independent economic validation",
    }


def review(root: Path = ROOT) -> dict[str, Any]:
    """Validate immutable source disposition before reading the retained journal."""
    base = root / BASE
    plan = json.loads((base / "plan.json").read_bytes())
    disposition = json.loads((base / "source-disposition.json").read_bytes())
    for path, expected in plan["implementation_sha256"].items():
        retained = disposition["archived_sources"].get(path, path)
        encoded = disposition.get("base64_archived_sources", {}).get(path)
        data = (
            (root / retained).read_bytes()
            if encoded is None
            else base64.b64decode((root / encoded).read_bytes())
        )
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("frozen source mismatch")
    result = reconstruct(
        Journal(base / "journal.jsonl").rows,
        json.loads((base / "result.json").read_bytes()),
    )
    paths = [
        (BASE / name).as_posix()
        for name in (
            "plan.json",
            "journal.jsonl",
            "result.json",
            "source-disposition.json",
            "frozen-runner.py",
            "frozen-api.py.base64",
            "official-source-extraction.json",
        )
    ]
    paths += [
        "tools/review_spot_testnet_execution.py",
        "src/simple_ai_trading/spot_testnet_coverage.py",
    ]
    result["input_sha256"] = {p: sha(root / p) for p in paths}
    return result


if __name__ == "__main__":
    print(json.dumps(review(), indent=2, sort_keys=True, allow_nan=False))
