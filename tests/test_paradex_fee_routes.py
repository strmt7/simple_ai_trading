"""Source facts and ambiguity checks; no account, token or order requests."""

import json
from datetime import datetime

import pytest

from tools.review_paradex_fee_routes import BASE, build_review, parse_terms, sha


def texts():
    return [
        (BASE / (name + ".md")).read_text(encoding="utf-8")
        for name in ("fees", "profiles")
    ]


def test_retained_review_reconstructs_and_qualification_stays_unknown():
    result = json.loads((BASE / "review.json").read_bytes())
    expected = result.pop("result_sha256")
    assert (
        sha(
            json.dumps(
                result, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        == expected
    )
    result.pop("created_at_utc")
    assert result == build_review()
    terms = result["terms"]
    assert terms["standard_api_default"] == "Pro"
    assert terms["retail_perps_spot_taker_fee_bps"] == "0"
    assert terms["pro_perps_spot_maker_fee_bps"] == "0.3"
    assert terms["pro_perps_spot_minimum_taker_fee_bps"] == "1.75"
    assert terms["retail_documented_speed_bump_ms"]["cancellation"] == 300
    assert terms["account_qualification"] is None
    assert not result["accepted_edge"] and not result["profitability_claim"]


def test_duplicate_tier_is_ambiguous_not_last_wins():
    fees, profiles = texts()
    duplicate = next(line for line in fees.splitlines() if line.startswith("| Pro 0 "))
    with pytest.raises(ValueError, match="ambiguous"):
        parse_terms(fees + "\n" + duplicate, profiles)


def test_missing_retail_reclassification_clause_rejects():
    fees, profiles = texts()
    with pytest.raises(ValueError, match="classification"):
        parse_terms(
            fees,
            profiles.replace(
                "account automatically switches to Pro Order behavior", "unknown"
            ),
        )


def test_inconsistent_fee_units_rejects():
    fees, profiles = texts()
    with pytest.raises(ValueError, match="disagree"):
        parse_terms(fees.replace("(0.3 bps) fee", "(3 bps) fee"), profiles)


def test_html_does_not_silently_satisfy_markdown_contract():
    fees, profiles = texts()
    with pytest.raises(ValueError, match="HTML"):
        parse_terms("<!DOCTYPE html>" + fees, profiles)


def test_exact_request_receipts_match_frozen_source_only_plan():
    plan = json.loads((BASE.parent / "paradex-fee-source-plan.json").read_bytes())
    records = [
        json.loads(line) for line in (BASE / "requests.jsonl").read_bytes().splitlines()
    ]
    assert len(records) == 2 * len(plan["requests"]) == 4
    last = datetime.fromisoformat(plan["frozen_at_utc"])
    for request, started, completed in zip(
        plan["requests"], records[::2], records[1::2]
    ):
        assert started["phase"] == "started"
        assert started["method"] == "GET" and started["redirects"] is False
        assert started["url"] == request["Url"]
        assert started["name"] == completed["name"] == request["Name"]
        assert started["script_sha256"] == plan["implementation_sha256"]
        assert started["max_bytes"] == plan["byte_ceiling_per_request"]
        assert completed["content_type"] == "text/markdown; charset=utf-8"
        assert completed["bytes"] <= started["max_bytes"]
        begin, end = (
            datetime.fromisoformat(x["time_utc"]) for x in (started, completed)
        )
        assert last <= begin <= end
        assert (end - begin).total_seconds() <= plan["deadline_seconds_per_request"]
        last = end
