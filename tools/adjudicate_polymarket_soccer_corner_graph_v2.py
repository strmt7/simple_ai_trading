"""Run the corner graph with the exact retained 24-or-48-hour fallback variants."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools import adjudicate_polymarket_soccer_corner_graph as base


ROOT = Path(__file__).resolve().parents[1]


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


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if contract.get("schema_version") != "polymarket-soccer-corner-graph-contract-v2":
        raise ValueError("unexpected contract schema")
    if contract_path != _root_path(str(contract["contract_path"])):
        raise ValueError("contract path mismatch")
    if _canonical_hash(contract, field="contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("contract timestamp is invalid or future")
    implementation = _root_path(str(contract["implementation"]["path"]))
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    for dependency in contract["implementation"]["dependencies"]:
        dependency_path = _root_path(str(dependency["path"]))
        if _sha256(dependency_path.read_bytes()) != dependency["sha256"]:
            raise ValueError(f"dependency hash mismatch: {dependency['path']}")


def _description(market: Mapping[str, Any]) -> str:
    return " ".join(str(market.get("description") or "").split())


def _require_corner_rules(market: Mapping[str, Any]) -> None:
    market_type = str(market.get("sportsMarketType"))
    description = _description(market)
    common = (
        "corners taken",
        "not corners awarded",
        "canceled entirely, with no make-up game, this market will resolve 50",
        "official statistics published by the governing body or event organizers",
        "this market will resolve 50-50",
    )
    exact_fallback = (
        "If no acceptable data is available within 24 hours" in description
        or "If no acceptable data is available within 48 hours" in description
    )
    if market_type in base.TOTAL_TYPES:
        outcomes = json.loads(str(market.get("outcomes")))
        if (
            outcomes != ["Over", "Under"]
            or not exact_fallback
            or not all(phrase in description for phrase in common)
            or Decimal(str(market.get("line"))) % 1 != Decimal("0.5")
        ):
            raise ValueError(f"corner-total rule changed: {market.get('id')}")
        return
    if market_type == "soccer_game_corners_odd_even":
        outcomes = json.loads(str(market.get("outcomes")))
        parity = (
            'resolve to "Odd"',
            'resolve to "Even"',
            "Zero corners is considered even",
        )
        if (
            outcomes != ["Odd", "Even"]
            or not exact_fallback
            or not all(phrase in description for phrase in (*common, *parity))
        ):
            raise ValueError(f"corner-parity rule changed: {market.get('id')}")
        return
    raise ValueError(f"unexpected corner market type: {market_type}")


def main() -> int:
    base._validate_contract = _validate_contract
    base._require_corner_rules = _require_corner_rules
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
