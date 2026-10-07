"""Reconcile a frozen completed-period CMS table from retained public bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from tools.capture_public_source_contract import (
    _canonical_hash,
    _load_object,
    _root_path,
    _validate_contract,
)


def _nodes(root: dict) -> list[dict]:
    stack, nodes = [(root, 0)], []
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, dict) or depth > 50 or len(nodes) >= 10_000:
            raise ValueError("CMS node structure exceeds its frozen bound")
        kind, children = node.get("node"), node.get("child", [])
        if kind not in {"root", "element", "text"} or not isinstance(children, list):
            raise ValueError("CMS node representation is unsupported")
        if kind == "text" and not isinstance(node.get("text"), str):
            raise ValueError("CMS text node is invalid")
        nodes.append(node)
        stack.extend((child, depth + 1) for child in reversed(children))
    return nodes


def _text(node: dict) -> str:
    return " ".join(
        " ".join(
            item["text"] for item in _nodes(node) if item["node"] == "text"
        ).split()
    )


def period_table(article: dict, specification: dict) -> list[list[str]]:
    """Select one exact header/population, never examples or a favorable sibling."""
    if article.get("success") is not True or article.get("code") != "000000":
        raise ValueError("CMS response did not prove success")
    data = article.get("data")
    if not isinstance(data, dict) or any(
        data.get(key) != specification[key] for key in ("id", "code", "title")
    ):
        raise ValueError("CMS article identity differs from the frozen population")
    root = json.loads(data["body"])
    tables = [node for node in _nodes(root) if node.get("tag") == "table"]
    matches = []
    for table in tables:
        rows = [
            [
                _text(cell)
                for cell in row.get("child", [])
                if cell.get("tag") in {"td", "th"}
            ]
            for row in _nodes(table)
            if row.get("tag") == "tr"
        ]
        if rows and rows[0] == specification["headers"]:
            matches.append(rows)
    if len(matches) != 1:
        raise ValueError("completed-period table is absent or ambiguous")
    rows = matches[0][1:]
    if (
        any(len(row) != len(specification["headers"]) for row in rows)
        or [row[0] for row in rows] != specification["period_labels"]
    ):
        raise ValueError("completed-period population differs from the frozen table")
    return rows


def _cell_state(value: str, *, percent: bool) -> str:
    if value.startswith("To be updated on "):
        return "unpublished_placeholder"
    grammar = r"\d+(?:\.\d+)?%" if percent else r"\$\d+(?:\.\d+)?"
    return (
        "published_numeric"
        if re.fullmatch(grammar, value, flags=re.ASCII)
        else "unparsed_no_numeric_claim"
    )


def reconcile(contract_path: Path, *, preflight: bool = False) -> dict | None:
    contract_path = contract_path.resolve()
    plan = _load_object(contract_path)
    _validate_contract(plan, contract_path)
    baseline = _root_path(plan["baseline_raw"]["path"])
    if (
        hashlib.sha256(baseline.read_bytes()).hexdigest()
        != plan["baseline_raw"]["sha256"]
    ):
        raise ValueError("immutable baseline source hash differs")
    period_table(_load_object(baseline), plan["article"])
    if preflight:
        return None
    destination = _root_path(plan["adjudication_result_path"])
    if destination.exists() or not destination.parent.is_dir():
        raise FileExistsError("adjudication is one-use and requires an existing parent")
    source = _load_object(_root_path(plan["outputs"]["result_path"]))
    if (
        _canonical_hash(source, "result_sha256") != source.get("result_sha256")
        or source["contract"]["sha256"] != plan["contract_sha256"]
    ):
        raise ValueError("fresh source lineage is invalid")
    raw_path = _root_path(plan["outputs"]["raw_path"])
    raw = raw_path.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != source["capture"]["receipt"]["response_sha256"]
    ):
        raise ValueError("fresh raw response hash differs")
    rows, error, metadata = [], None, None
    if source["source_gate"]["passed"]:
        try:
            article = _load_object(raw_path)
            table = period_table(article, plan["article"])
            metadata = {
                key: article["data"].get(key)
                for key in (
                    "id",
                    "code",
                    "title",
                    "version",
                    "publishDate",
                    "lastUpdateTime",
                )
            }
            rows = [
                {
                    "period": row[0],
                    "base_apr": row[1],
                    "base_state": _cell_state(row[1], percent=True),
                    "boosted_apr": row[2],
                    "boosted_state": _cell_state(row[2], percent=True),
                    "token_value": row[3],
                    "token_value_state": _cell_state(row[3], percent=False),
                }
                for row in table
            ]
        except (
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            RuntimeError,
        ) as failure:
            error = str(failure)[:200]
    else:
        error = "frozen public source gate failed"
    observed = datetime.fromtimestamp(
        source["capture"]["receipt"]["completed_at_ms"] / 1000, timezone.utc
    )
    end = datetime.fromisoformat(plan["campaign_end_utc"].replace("Z", "+00:00"))
    result = {
        "schema_version": "completed-yield-distribution-reconciliation-v1",
        "contract_sha256": plan["contract_sha256"],
        "source_result_sha256": source["result_sha256"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "observed_at_utc": observed.isoformat(),
        "article": metadata,
        "period_rows": rows,
        "error": error,
        "source_and_table_passed": error is None,
        "first_period_numeric_base_published": bool(rows)
        and rows[0]["base_state"] == "published_numeric",
        "first_period_numeric_reference_published": bool(rows)
        and rows[0]["token_value_state"] == "published_numeric",
        "campaign_ended_at_observation": observed >= end,
        "guaranteed_forward_reward_usd": "0",
        "owned_account_qualified": False,
        "executable_after_all_cost_profit_proved": False,
        "accepted_edge": False,
        "market_or_account_escalation_authorized": False,
        "retry": "Only independently observed material official program/terms change; this deadline trigger is consumed",
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    with destination.open("x", encoding="ascii", newline="\n") as output:
        output.write(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        )
        output.flush()
        os.fsync(output.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = reconcile(args.contract, preflight=args.preflight)
    print(json.dumps({"preflight": args.preflight, "result": result}))


if __name__ == "__main__":
    main()
