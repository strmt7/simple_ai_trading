"""Verify research-ledger hashes, cross-binding and census without promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def verify_binding(registry: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Validate recorded identities/counts, not financial truth or profitability."""
    for label, value in (("registry", registry), ("audit", audit)):
        if not isinstance(value, dict) or value.get("result_sha256") != _canonical_hash(
            value
        ):
            raise ValueError(f"{label} self-hash differs")
    binding = audit["source_binding"]
    if binding["registry_result_sha256"] != registry["result_sha256"]:
        raise ValueError("audit binds a different registry revision")
    scoped = _count(registry["accepted_edge_count"], "scoped mechanism count")
    decision = audit["decision"]
    if (
        _count(binding["accepted_edge_count"], "bound scope count") != scoped
        or _count(decision["accepted_scope_count"], "audited scope count") != scoped
    ):
        raise ValueError("scoped mechanism counts differ")
    qualified = _count(
        decision["stable_current_account_qualified_after_all_cost_edge_count"],
        "recorded qualified-edge count",
    )
    if qualified > scoped:
        raise ValueError("recorded qualified-edge count exceeds scoped mechanisms")
    for key, identity in (
        ("prioritized_hypotheses", "priority_rank"),
        ("terminal_do_not_repeat", "family"),
    ):
        rows = registry[key]
        if not isinstance(rows, list):
            raise ValueError(f"{key} must be a list")
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{key} contains an invalid record")
            value = row.get(identity)
            if identity == "priority_rank":
                valid = type(value) is int and value > 0
            else:
                valid = isinstance(value, str) and bool(value)
            if not valid or value in seen:
                raise ValueError(f"{key} contains an invalid or duplicate identity")
            seen.add(value)
    classification = audit["classification"]
    if not isinstance(classification, list):
        raise ValueError("scope classification must be a list")
    ordinals = []
    for group in classification:
        members = group["edge_ordinals"]
        if (
            not isinstance(members, list)
            or any(type(value) is not int or value <= 0 for value in members)
            or _count(group["count"], "classified scope count") != len(members)
        ):
            raise ValueError("scope classification membership differs from its count")
        ordinals.extend(members)
    if sorted(ordinals) != list(range(1, scoped + 1)):
        raise ValueError("scope classification is not an exact partition")
    return {
        "validation_kind": "registry_binding_and_recorded_census_only",
        "observations": len(registry["terminal_do_not_repeat"]),
        "hypotheses": len(registry["prioritized_hypotheses"]),
        "scoped_mechanisms": scoped,
        "recorded_qualified_stable_edges": qualified,
        "registry_sha256": registry["result_sha256"],
        "audit_sha256": audit["result_sha256"],
        "profitability_qualification_supplied_by_verifier": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        result = verify_binding(registry, audit)
    except (KeyError, TypeError, ValueError, OSError) as error:
        result = {"valid": False, "error_type": type(error).__name__}
        if isinstance(error, ValueError):
            result["reason"] = str(error)
        print(json.dumps(result))
        return 1
    print(json.dumps({"valid": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
