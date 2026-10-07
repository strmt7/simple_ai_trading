from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode

import pytest

from tools import screen_future_nfl_payoff_floor as module


ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT / "data/polymarket-nfl-september13-early-monotone-catalog-v1/raw/events.json"
)


def _event() -> dict:
    raw = CONTROL.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "e086bd126636f0d7726d3ce56b58d238d39c11d17c25b7147d63566239cc4ee9"
    )
    event = json.loads(raw)["events"][0]
    moneyline = next(
        m for m in event["markets"] if m["sportsMarketType"] == "moneyline"
    )
    totals = sorted(
        (m for m in event["markets"] if m["sportsMarketType"] == "totals"),
        key=lambda m: Decimal(str(m["line"])),
    )[:2]
    event["markets"] = [moneyline, *totals]
    for market in totals:
        market["bestAsk"] = 0.51
        market["bestBid"] = 0.49
    return event


def _screen(payload: object) -> dict:
    return module.screen(
        payload, minimum="2026-09-13T00:00:00Z", maximum="2026-09-13T23:59:59Z"
    )


def test_complete_control_and_symbolic_payoff_floor() -> None:
    event = json.loads(CONTROL.read_bytes())["events"][0]
    result = module._event_ladder(event)
    assert result["relation_count"] == 253
    assert result["price_complete_count"] == 253
    assert result["price_incomplete_count"] == 0
    assert result["strict_sub_floor_count"] == 0
    assert result["fees_depth_clock_and_recurrence_qualified"] is False
    for lower, upper in ((1, 2), (43, 47), (127, 128)):
        for score in (0, lower - 1, lower, upper - 1, upper, 10**9):
            assert int(score >= lower) + int(score < upper) >= 1
        assert Decimal("0.5") + Decimal("0.5") == 1


def test_midpoints_are_never_read_and_selected_quotes_only() -> None:
    event = _event()
    lower, upper = event["markets"][1:]
    lower.pop("outcomePrices")
    upper["outcomePrices"] = "invalid"
    lower.pop("bestBid")
    upper.pop("bestAsk")
    result = module._event_ladder(event)
    assert result["strongest_package"]["side_specific_cost_pusd"] == Decimal("1.02")
    assert result["strict_sub_floor_count"] == 0
    lower["bestAsk"] = 0.49
    assert module._event_ladder(event)["strict_sub_floor_count"] == 0
    lower["bestAsk"] = 0.30
    upper["bestBid"] = 0.80
    result = module._event_ladder(event)
    assert result["strongest_package"]["gross_surplus_pusd"] == Decimal("0.5")
    assert result["gross_candidate_only"] is True
    assert result["fees_depth_clock_and_recurrence_qualified"] is False


@pytest.mark.parametrize("bad", [None, "NaN", "Infinity", -0.1, 1.1])
def test_missing_or_invalid_selected_quote_is_not_zero(bad: object) -> None:
    event = _event()
    event["markets"][1]["bestAsk"] = bad
    result = module._event_ladder(event)
    assert result["price_incomplete_count"] == 1
    assert result["price_complete_count"] == 0
    assert result["strongest_package"] is None
    assert result["gross_candidate_only"] is False


@pytest.mark.parametrize(
    "change", ["postpone", "cancel", "schedule", "team", "threshold", "extra"]
)
def test_complete_resolver_identity_not_one_phrase(change: str) -> None:
    event = _event()
    market = event["markets"][2]
    replacements = {
        "postpone": ("until the game has been completed", "for 24 hours"),
        "cancel": ("resolve 50-50", "resolve to Under"),
        "schedule": ("September 13", "September 14"),
        "team": ("Bears and Panthers", "Other and Panthers"),
        "threshold": ("or more points", "or fewer points"),
    }
    if change == "extra":
        market["description"] += " Otherwise this market resolves differently."
    else:
        market["description"] = market["description"].replace(*replacements[change])
    with pytest.raises(ValueError, match="identity|rules"):
        module._event_ladder(event)


@pytest.mark.parametrize("line", [None, "NaN", "Infinity", -1.5, 2, 128.5])
def test_threshold_resource_and_half_point_boundaries(line: object) -> None:
    event = _event()
    event["markets"][1]["line"] = line
    with pytest.raises((ValueError, ArithmeticError)):
        module._event_ladder(event)


def test_no_outcome_adaptive_fallback_event() -> None:
    first = _event()
    later = deepcopy(first)
    later["id"], later["slug"], later["startTime"] = (
        "later",
        "later",
        "2026-09-13T18:00:00Z",
    )
    later["markets"][1]["bestAsk"] = 0.1
    later["markets"][2]["bestBid"] = 0.9
    result = _screen({"events": [later, first]})
    assert result["event_slug"] == first["slug"]
    assert result["strict_sub_floor_count"] == 0
    first["markets"][1]["description"] += " invalid extra rule"
    with pytest.raises(ValueError, match="rules"):
        _screen({"events": [later, first]})


@pytest.mark.parametrize(
    "payload", [None, [], {"events": None}, {"events": [], "next_cursor": None}]
)
def test_no_pagination_or_malformed_population(payload: object) -> None:
    with pytest.raises(ValueError):
        _screen(payload)
    assert _screen({"events": []})["status"] == "no_deployed_event"


def test_population_and_market_identity_and_resource_rejections() -> None:
    event = _event()
    for field in ("id", "slug"):
        other = deepcopy(event)
        other[field] = None
        with pytest.raises(ValueError, match="identity"):
            _screen({"events": [other]})
    for field, value in (
        ("active", False),
        ("series", []),
        ("startTime", "2026-09-14T00:00:00Z"),
    ):
        other = deepcopy(event)
        other[field] = value
        with pytest.raises(ValueError):
            _screen({"events": [other]})
    with pytest.raises(ValueError, match="duplicated"):
        _screen({"events": [event, deepcopy(event)]})
    with pytest.raises(ValueError, match="oversized"):
        _screen({"events": [event] * 51})
    other = deepcopy(event)
    other["padding"] = "x" * 1024 * 1024
    with pytest.raises(ValueError, match="per-row"):
        _screen({"events": [other]})
    other = deepcopy(event)
    other["markets"][1]["id"] = other["markets"][2]["id"]
    with pytest.raises(ValueError, match="ambiguous"):
        module._event_ladder(other)
    other = deepcopy(event)
    other["markets"] = other["markets"] * 100
    with pytest.raises(ValueError, match="resource"):
        module._event_ladder(other)


def test_strict_native_json_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        json.loads(
            '{"events": [], "events": []}', object_pairs_hook=module._unique_object
        )
    with pytest.raises(ValueError, match="nonfinite"):
        json.loads('{"events": NaN}', parse_constant=module._reject_constant)


@pytest.mark.parametrize("change", ["condition", "token", "oversize", "noncanonical"])
def test_native_asset_aliases_are_not_distinct_payoff_legs(change: str) -> None:
    event = _event()
    lower, upper = event["markets"][1:]
    if change == "condition":
        upper["conditionId"] = lower["conditionId"]
    elif change == "token":
        upper["clobTokenIds"] = lower["clobTokenIds"]
    elif change == "oversize":
        upper["clobTokenIds"] = json.dumps([str(2**256), "1"])
    else:
        upper["clobTokenIds"] = json.dumps(["01", "2"])
    with pytest.raises(ValueError, match="identity"):
        module._event_ladder(event)


def _future_plan() -> dict:
    now = datetime.now(timezone.utc)
    window = {
        "minimum": (now + timedelta(days=1)).isoformat(),
        "maximum": (now + timedelta(days=2)).isoformat(),
    }
    return {
        "frozen_at_utc": now.isoformat(),
        "screen": dict(module.LIMITS),
        "window": window,
        "response_byte_ceiling": 50 * 1024 * 1024,
        "request": {
            "url": "https://gamma-api.polymarket.com/events/keyset?"
            + urlencode(
                {
                    "limit": 50,
                    "order": "startTime",
                    "ascending": "true",
                    "closed": "false",
                    "series_id": 12185,
                    "start_time_min": window["minimum"],
                    "start_time_max": window["maximum"],
                }
            )
        },
        "outputs": {
            "raw_path": "native.raw",
            "journal_path": "capture.jsonl",
            "result_path": "capture.json",
        },
        "screen_outputs": {
            "journal_path": "screen.jsonl",
            "result_path": "screen.json",
        },
    }


def test_screen_plan_and_deadline_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "capture", lambda _path, *, preflight: None)
    monkeypatch.setattr(module, "_root_path", lambda p: tmp_path / p)
    path = tmp_path / "contract.json"
    plan = _future_plan()
    module._validate(plan, path)
    other = deepcopy(plan)
    other["screen"]["event_limit"] = 51
    with pytest.raises(ValueError, match="policy"):
        module._validate(other, path)
    other = deepcopy(plan)
    other["window"]["minimum"] = other["frozen_at_utc"]
    with pytest.raises(ValueError, match="strictly future"):
        module._validate(other, path)
    other = deepcopy(plan)
    now = datetime.now(timezone.utc)
    other["frozen_at_utc"] = (now - timedelta(days=2)).isoformat()
    other["window"]["minimum"] = (now - timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="deadline"):
        module._validate(other, path)
    other = deepcopy(plan)
    other["request"]["url"] += "&unexpected=1"
    with pytest.raises(ValueError, match="query"):
        module._validate(other, path)
    other = deepcopy(plan)
    other["screen_outputs"]["journal_path"] = other["screen_outputs"]["result_path"]
    with pytest.raises(ValueError, match="outputs differ"):
        module._validate(other, path)
    other = deepcopy(plan)
    other["screen_outputs"]["journal_path"] = "native.raw"
    with pytest.raises(ValueError, match="distinct"):
        module._validate(other, path)
    (tmp_path / "screen.json").write_text("occupied")
    with pytest.raises(ValueError, match="exist"):
        module._validate(plan, path)


def test_cli_routes_to_same_forward_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        sys, "argv", ["screen", "--contract", "frozen.json", "--preflight"]
    )
    monkeypatch.setattr(
        module, "run", lambda p, *, preflight: calls.append((p, preflight))
    )
    module.main()
    assert calls == [(Path("frozen.json"), True)]


def test_durable_forward_run_and_preflight_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.dumps({"events": [_event()]}).encode()
    (tmp_path / "native.raw").write_bytes(raw)
    plan = {
        "contract_sha256": "a" * 64,
        "window": {
            "minimum": "2026-09-13T00:00:00Z",
            "maximum": "2026-09-13T23:59:59Z",
        },
        "outputs": {"raw_path": "native.raw"},
        "screen_outputs": {
            "journal_path": "journal.jsonl",
            "result_path": "result.json",
        },
    }
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(plan))
    monkeypatch.setattr(module, "_root_path", lambda p: tmp_path / p)
    monkeypatch.setattr(module, "_validate", lambda *_args: None)
    calls = []
    capture = {
        "source_gate": {"passed": True},
        "capture": {"receipt": {"response_sha256": hashlib.sha256(raw).hexdigest()}},
        "result_sha256": "b" * 64,
    }
    monkeypatch.setattr(module, "capture", lambda p: calls.append(p) or capture)
    module.run(contract, preflight=True)
    assert not calls and not (tmp_path / "journal.jsonl").exists()
    module.run(contract)
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["screen"]["status"] == "screened"
    assert result["book_or_fee_request_authorized"] is False
    assert result["accepted_edge"] is False
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        module.run(contract)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["transport", "hash", "duplicate", "constant"])
def test_terminal_capture_failures_are_durable_without_retry(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.dumps({"events": [_event()]}).encode()
    if failure == "duplicate":
        raw = b'{"events": [], "events": []}'
    elif failure == "constant":
        raw = b'{"events": NaN}'
    (tmp_path / "native.raw").write_bytes(raw)
    plan = {
        "contract_sha256": "a" * 64,
        "window": {
            "minimum": "2026-09-13T00:00:00Z",
            "maximum": "2026-09-13T23:59:59Z",
        },
        "outputs": {"raw_path": "native.raw"},
        "screen_outputs": {
            "journal_path": "journal.jsonl",
            "result_path": "result.json",
        },
    }
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(plan))
    monkeypatch.setattr(module, "_root_path", lambda p: tmp_path / p)
    monkeypatch.setattr(module, "_validate", lambda *_args: None)
    calls = []
    capture = {
        "source_gate": {"passed": failure != "transport"},
        "capture": {
            "receipt": {
                "response_sha256": "c" * 64
                if failure == "hash"
                else hashlib.sha256(raw).hexdigest()
            }
        },
        "result_sha256": "b" * 64,
    }
    monkeypatch.setattr(module, "capture", lambda p: calls.append(p) or capture)
    module.run(contract)
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["screen"]["status"] == "terminal_unqualified"
    assert result["accepted_edge"] is False
    assert result["book_or_fee_request_authorized"] is False
    assert len(calls) == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
    ]
    assert records[0]["phase"] == "intent"
    assert records[1]["phase"] == "completed"
    assert records[1]["status"] == "terminal_unqualified"
