from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/action-value/binance-perpetual-clamp-bounds-paper-source-contract-v1-2026-08-31.json"
)
RESULT = (
    ROOT
    / "docs/model-research/action-value/binance-perpetual-clamp-bounds-paper-source-failure-v1-2026-08-31.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_ssrn_source_failure_is_hash_bound_and_consumed_once() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    binding = result["source_binding"]
    raw = ROOT / binding["raw_path"]  # type: ignore[index]
    journal = ROOT / binding["journal_path"]  # type: ignore[index]

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _file_hash(raw) == binding["raw_sha256"]  # type: ignore[index]
    assert _file_hash(journal) == binding["journal_sha256"]  # type: ignore[index]
    assert raw.stat().st_size == 5625
    assert raw.read_bytes().startswith(b"<!DOCTYPE html>")
    assert b"Enable JavaScript and cookies to continue" in raw.read_bytes()
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [entry["phase"] for entry in entries] == ["intent", "completed"]
    assert entries[1]["status_code"] == 403
    assert entries[1]["response_bytes"] == 5625


def test_abstract_does_not_authorize_market_research_or_profit_claim() -> None:
    result = _load(RESULT)

    assert result["capture"]["required_pdf_signature_present"] is False  # type: ignore[index]
    assert result["capture"]["complete_methodology_available"] is False  # type: ignore[index]
    assert result["adjudication"]["accepted_edge"] is False  # type: ignore[index]
    assert result["adjudication"]["market_data_collector_authorized"] is False  # type: ignore[index]
    assert result["adjudication"]["profitability_claim"] is False  # type: ignore[index]
    assert result["adjudication"]["public_forward_profit_floor"] == "0"  # type: ignore[index]
    assert result["authority"]["literature_requests"] == 1  # type: ignore[index]
    assert result["authority"]["venue_market_data_requests"] == 0  # type: ignore[index]
    assert result["authority"]["credentials_used"] is False  # type: ignore[index]


def test_registry_terminalizes_only_the_failed_primary_source_lead() -> None:
    result = _load(RESULT)
    registry = _load(REGISTRY)

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    row = next(
        item
        for item in registry["terminal_do_not_repeat"]  # type: ignore[union-attr]
        if item["family"]
        == "binance_perpetual_clamp_model_free_bounds_SSRN_5262988_primary_source_failure"
    )
    assert row["canonical_result_sha256"] == result["result_sha256"]
    assert "do_not_retry" in row["reason"]
    assert registry["accepted_edge_count"] == 29
    assert len(registry["prioritized_hypotheses"]) == 47  # type: ignore[arg-type]
    assert len(registry["terminal_do_not_repeat"]) == 121  # type: ignore[arg-type]
