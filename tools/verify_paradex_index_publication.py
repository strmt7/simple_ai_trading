"""Reconstruct the frozen index study from committed bytes, without network."""

from decimal import Decimal, localcontext
import json
from pathlib import Path
import re

from tools.screen_paradex_index_boundaries import (
    ASSETS,
    ROOT,
    binance_rows,
    canonical,
    digest,
    economic_rows,
    index_row,
    requests_for,
    role_result,
)


def documentation_dispositions(root: Path = ROOT) -> dict:
    base = root / "docs/review/2026-09-04/paradex-index-source"
    disposition = json.loads((base / "disposition.json").read_bytes())
    expected = disposition.pop("result_sha256")
    assert (
        expected == "d73c091f3d484f5585202d84102d7e5ce9bbcd73b91d29ec29b4e35a6f233b4a"
    )
    assert digest(canonical(disposition)) == expected
    assert (
        digest((root / disposition["extractor_path"]).read_bytes())
        == disposition["extractor_sha256"]
    )
    journal_raw = (base / "requests.jsonl").read_bytes()
    assert digest(journal_raw) == disposition["request_journal_sha256"]
    journal = [json.loads(line) for line in journal_raw.decode().splitlines()]
    resolved = {}
    for item in disposition["sources"]:
        assert not (root / item["original_path"]).exists()
        section = (root / item["section_path"]).read_bytes()
        assert len(section) == item["section_bytes"]
        assert digest(section) == item["section_sha256"]
        assert not re.search(rb"(?:AKIA|ASIA)[A-Z0-9]{16}|PRIVATE KEY-----", section)
        assert any(
            r.get("raw_sha256") == item["original_sha256"]
            and r.get("bytes") == item["original_bytes"]
            and r.get("passed") is True
            for r in journal
        )
        resolved[item["original_path"]] = item["original_sha256"]
    assert len(resolved) == 2
    return resolved


def verify(root: Path = ROOT) -> dict:
    contract_path = root / "docs/review/2026-09-04/paradex-index-contract.json"
    contract = json.loads(contract_path.read_bytes())
    expected = contract.pop("contract_sha256")
    assert digest(canonical(contract)) == expected
    removed_sources = documentation_dispositions(root)
    for path, source_hash in contract["source_sha256"].items():
        if path in removed_sources:
            # Original response identity is preserved, not reconstructed. The
            # reviewed safe section has its own explicit hash and disposition.
            assert removed_sources[path] == source_hash
        else:
            assert digest((root / path).read_bytes()) == source_hash, path
    base = root / contract["output_root"]
    result = json.loads((base / "result.json").read_bytes())
    result_hash = result.pop("result_sha256")
    assert digest(canonical(result)) == result_hash
    assert result["contract_sha256"] == expected
    assert result["source_sha256"] == contract["source_sha256"]
    assert result["status"] == "complete_funding_only_prefilter"
    requests = requests_for(contract["start_boundary_ms"], contract["end_boundary_ms"])
    assert requests == contract["requests"] and len(requests) == 84
    journal = [
        json.loads(line) for line in (base / "requests.jsonl").read_text().splitlines()
    ]
    assert len(journal) == 2 * len(requests) + 1
    assert len(result["receipts"]) == len(requests)
    selected = {}
    offsets = []
    for i, (request, receipt) in enumerate(
        zip(requests, result["receipts"], strict=True)
    ):
        began, ended = journal[2 * i], journal[2 * i + 1]
        assert began["phase"] == "request_started" and began["method"] == "GET"
        assert all(began[key] == value for key, value in request.items())
        assert ended["phase"] == "request_completed" and ended["status"] == 200
        assert all(
            ended[key] == value for key, value in receipt.items() if key != "raw_path"
        )
        raw = (root / receipt["raw_path"]).read_bytes()
        assert len(raw) == receipt["bytes"] <= 1048576
        assert digest(raw) == receipt["raw_sha256"]
        payload = json.loads(raw, parse_float=Decimal)
        if request["kind"] == "index":
            selected[request["name"]] = index_row(payload, request)
            offsets.append(selected[request["name"]]["time"] - request["boundary"])
        else:
            selected[request["name"]] = binance_rows(
                payload,
                asset=request["asset"],
                start=contract["start_boundary_ms"],
                end=contract["end_boundary_ms"],
            )
    rebuilt = {}
    headroom = []
    whole_window = []
    with localcontext() as context:
        context.prec = 50
        for asset in ASSETS:
            indices = [selected[f"{asset.lower()}-index-{i:02d}"] for i in range(27)]
            rows = economic_rows(indices, selected[f"{asset.lower()}-binance"])
            train = sum(
                Decimal(r["paradex_short_cash_usdc_per_base"])
                + Decimal(r["binance_long_cash_usdt_per_base"])
                for r in rows[:13]
            )
            sign = 1 if train >= 0 else -1
            roles = {
                name: role_result(rows[left:right], sign=sign)
                for name, left, right in (
                    ("training", 0, 13),
                    ("validation", 13, 19),
                    ("test", 19, 26),
                )
            }
            rebuilt[asset] = {
                "sign": sign,
                "rows": rows,
                "roles": roles,
                "passes_prefilter": all(r["passes_prefilter"] for r in roles.values()),
            }
            headroom.extend(
                {
                    "asset": asset,
                    "role": name,
                    "after_execution_only_bips": str(
                        Decimal(role["par_valuation_gross_bips"]) - 20
                    ),
                }
                for name, role in roles.items()
            )
            # Rejection-only comparator: charge execution once across all roles.
            # Includes training data and is not an independent policy evaluation.
            whole = role_result(rows, sign=sign)
            whole_window.append(
                {
                    "asset": asset,
                    "par_valuation_gross_bips": whole["par_valuation_gross_bips"],
                    "after_one_execution_allowance_bips": str(
                        Decimal(whole["par_valuation_gross_bips"]) - 20
                    ),
                    "independent_validation": False,
                }
            )
    assert rebuilt == result["assets"]
    survivors = [a for a, v in rebuilt.items() if v["passes_prefilter"]]
    assert result["survivors"] == journal[-1]["survivors"] == survivors == []
    assert journal[-1]["phase"] == "study_complete"
    assert not any(
        result[key]
        for key in ("accepted_edge", "profitability_claim", "trading_authority")
    )
    assert all(Decimal(row["after_execution_only_bips"]) < 0 for row in headroom)
    assert all(
        Decimal(row["after_one_execution_allowance_bips"]) < 0 for row in whole_window
    )
    return {
        "verified_raw_responses": len(requests),
        "documentation_bodies_removed_with_explicit_section_dispositions": 2,
        "verified_intervals": len(ASSETS) * 26,
        "raw_bytes": sum(r["bytes"] for r in result["receipts"]),
        "selected_offset_ms_min": min(offsets),
        "selected_offset_ms_max": max(offsets),
        "source_request_first_at": journal[0]["at_utc"],
        "source_request_last_at": journal[-1]["at_utc"],
        "execution_only_headroom": headroom,
        "whole_window_rejection_only": whole_window,
        "result_sha256": result_hash,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
