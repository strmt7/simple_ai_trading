"""Run the frozen corner graph with corrected adjacent-pair enumeration."""

from __future__ import annotations

import builtins
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import adjudicate_polymarket_soccer_corner_graph as base
from tools import adjudicate_polymarket_soccer_corner_graph_v2 as v2


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if contract.get("schema_version") != "polymarket-soccer-corner-graph-contract-v3":
        raise ValueError("unexpected contract schema")
    if contract_path != v2._root_path(str(contract["contract_path"])):
        raise ValueError("contract path mismatch")
    if v2._canonical_hash(contract, field="contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    frozen = datetime.fromisoformat(
        str(contract["frozen_at_utc"]).replace("Z", "+00:00")
    )
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise ValueError("contract timestamp is invalid or future")
    implementation = v2._root_path(str(contract["implementation"]["path"]))
    if v2._sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise ValueError("implementation hash mismatch")
    for dependency in contract["implementation"]["dependencies"]:
        dependency_path = v2._root_path(str(dependency["path"]))
        if v2._sha256(dependency_path.read_bytes()) != dependency["sha256"]:
            raise ValueError(f"dependency hash mismatch: {dependency['path']}")


def _adjacent_zip(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> zip:
    if strict and len(first) != len(second) + 1:
        raise ValueError("adjacent ladder inputs must differ by exactly one")
    return builtins.zip(first, second)


def main() -> int:
    base._validate_contract = _validate_contract
    base._require_corner_rules = v2._require_corner_rules
    base.zip = _adjacent_zip
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
