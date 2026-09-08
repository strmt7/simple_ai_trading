"""Offline publication checks for the exact retained September 8 research sources."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from tools.capture_public_source_contract import _canonical_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("docs/review/2026-09-08")


def _load(path: Path, field: str = "result_sha256") -> dict:
    value = json.loads((ROOT / path).read_bytes())
    assert _canonical_hash(value, field) == value[field]
    return value


@pytest.mark.parametrize(
    "relative",
    [
        "usdt-flexible/source-contract.json",
        "hk-listing-match/announcement-contract.json",
        "hk-listing-match/polymarket-instruments-contract.json",
    ],
)
def test_source_bytes_chronology_and_journals_reconstruct(relative):
    contract = _load(BASE / relative, "contract_sha256")
    output = contract["outputs"]
    source = _load(Path(output["result_path"]))
    raw = (ROOT / output["raw_path"]).read_bytes()
    receipt = source["capture"]["receipt"]
    journal = [
        json.loads(line)
        for line in (ROOT / output["journal_path"]).read_bytes().splitlines()
    ]
    assert len(journal) == 2
    assert journal[0]["request"] == contract["request"]
    assert journal[0]["contract_sha256"] == contract["contract_sha256"]
    assert journal[1] == receipt
    assert journal[0]["requested_at_ms"] == receipt["requested_at_ms"]
    assert datetime.fromisoformat(contract["frozen_at_utc"]) <= datetime.fromtimestamp(
        receipt["requested_at_ms"] / 1000, UTC
    )
    assert receipt["requested_at_ms"] <= receipt["completed_at_ms"]
    assert hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
    assert len(raw) == receipt["response_bytes"] <= contract["response_byte_ceiling"]
    assert receipt["status_code"] == 200 and receipt["error_type"] is None
    assert source["source_gate"]["passed"] and not source["accepted_edge"]
    for binding in contract["implementations"]:
        assert (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            == binding["sha256"]
        )


def test_usdt_bonus_is_capped_conditional_and_not_promoted():
    result = _load(BASE / "usdt-flexible/result.json")
    source = _load(BASE / "usdt-flexible/source-result.json")
    assert result["source_result_sha256"] == source["result_sha256"]
    assert not result["accepted_edge"] and not result["account_qualified"]
    assert not result["deployment_ready"] and result["accepted_scope_count_delta"] == 0
    terms, example = result["terms"], result["economic_sensitivity"]
    start = datetime.fromisoformat(terms["start_utc"])
    end = datetime.fromisoformat(terms["end_utc"])
    days = (end.date() - start.date()).days
    assert (
        days
        == example[
            "complete_eligible_days_if_subscribed_September8_held_through_September22"
        ]
        == 14
    )
    with localcontext() as context:
        context.prec = 60
        daily = (
            Decimal(terms["bonus_principal_cap_USDT"])
            * Decimal(terms["bonus_apr_percent"])
            / 100
            / example["day_count_basis"]
        )
        assert daily == Decimal(example["bonus_USDT_per_complete_eligible_day_at_cap"])
        assert daily * days == Decimal(example["fourteen_day_bonus_USDT_at_cap"])
    assert not example["day_count_basis_source_proved"]
    assert not example["variable_base_APR_credited"]
    assert example["future_guaranteed_bonus_floor_USDT"] == "0"
    source_bytes = (
        (ROOT / "tools/review_usdt_flexible_september8.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    )
    assert hashlib.sha256(source_bytes).hexdigest() == result["reviewer_sha256"]


def test_listing_no_match_projection_does_not_authorize_economic_access():
    result = _load(BASE / "hk-listing-match/result.json")
    rows = json.loads(
        (ROOT / BASE / "hk-listing-match/polymarket-instruments-raw.json").read_bytes()
    )
    fields = ("instrument_id", "symbol", "base_asset", "quote_asset")
    assert result["identity_projection"] == [
        {key: row[key] for key in fields} for row in rows
    ]
    assert result["inventory_count"] == len(rows) == 67
    assert not [row for row in rows if row["base_asset"] in {"BYD", "HK0992"}]
    assert result["exact_base_matches"] == []
    assert result["status"] == "terminal_no_exact_counterpart"
    assert (
        result["economic_requests"]
        == result["account_requests"]
        == result["orders_or_funds"]
        == 0
    )
    assert not result["accepted_edge"] and not result["deployment_ready"]
    for name, embedded in result["source_results"].items():
        assert embedded == _load(BASE / f"hk-listing-match/{name}-source-result.json")
    source_bytes = (
        (ROOT / "tools/review_hk_listing_match_september8.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    )
    assert hashlib.sha256(source_bytes).hexdigest() == result["reviewer_sha256"]
