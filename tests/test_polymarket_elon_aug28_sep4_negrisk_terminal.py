from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.screen_polymarket_exact_negrisk_event import _screen_compatible_event


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / (
    "polymarket-elon-aug28-sep4-exact-negrisk-prefilter-"
    "contract-v1-2026-08-30.json"
)
TERMINAL = ACTION_VALUE / (
    "polymarket-elon-aug28-sep4-exact-negrisk-prefilter-"
    "terminal-v1-2026-08-30.json"
)
DATA = ROOT / "data/polymarket-elon-aug28-sep4-exact-negrisk-prefilter-v1"
RAW = DATA / "raw/event.json"
JOURNAL = DATA / "request-journal.jsonl"
PREFLIGHT_JOURNAL = DATA / "raw/preflight-journal.jsonl"
CONSUMED_RUNNER = DATA / "raw/screen-polymarket-exact-negrisk-event-consumed.py"
CURRENT_RUNNER = ROOT / "tools/screen_polymarket_exact_negrisk_event.py"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical(body))


def test_consumed_exact_request_and_local_preflight_are_fully_retained() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    terminal = json.loads(TERMINAL.read_text(encoding="ascii"))
    journal = [
        json.loads(line) for line in JOURNAL.read_text(encoding="ascii").splitlines()
    ]
    preflight = json.loads(PREFLIGHT_JOURNAL.read_text(encoding="ascii"))

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(terminal, "result_sha256") == terminal["result_sha256"]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[0]["requested_at_ms"] <= journal[1]["completed_at_ms"]
    assert journal[1]["status_code"] == 200
    assert journal[1]["response_bytes"] == len(RAW.read_bytes())
    assert journal[1]["response_sha256"] == _sha256(RAW.read_bytes())
    assert preflight["network_requests"] == 0
    assert preflight["outcome"] == "local_pre_main_import_failure_unconsumed"


def test_retained_partial_event_rejects_before_any_book_request() -> None:
    event = json.loads(RAW.read_text(encoding="utf-8"))
    terminal = json.loads(TERMINAL.read_text(encoding="ascii"))
    fixed, screened, conversions, failure = _screen_compatible_event(event, 26)

    assert fixed is True
    assert screened is None
    assert conversions == []
    assert failure == "event contains an unavailable or incompatible market"
    closed = [market for market in event["markets"] if market["closed"] is True]
    active = [
        market
        for market in event["markets"]
        if market["closed"] is False and market["acceptingOrders"] is True
    ]
    assert [market["groupItemTitle"] for market in closed] == ["<20", "20-39"]
    assert all(json.loads(market["outcomePrices"]) == ["0", "1"] for market in closed)
    active_yes_sum = sum(
        Decimal(json.loads(market["outcomePrices"])[0]) for market in active
    )
    assert len(active) == 24
    assert active_yes_sum == Decimal("1.0055")
    assert terminal["retained_offline_rejection"]["strict_gross_headroom_pUSD"] == (
        "-0.0055"
    )
    assert terminal["authority"]["book_or_fee_requests"] == 0
    assert terminal["adjudication"]["accepted_edge"] is False


def test_consumed_runner_lineage_and_rank_31_terminal_are_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    terminal = json.loads(TERMINAL.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    recovery = terminal["runner_recovery"]

    assert _sha256(CONSUMED_RUNNER.read_bytes()) == contract["implementations"][0][
        "sha256"
    ]
    assert _sha256(CURRENT_RUNNER.read_bytes()) == recovery[
        "current_reusable_runner_sha256"
    ]
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    assert {
        "path": TERMINAL.relative_to(ROOT).as_posix(),
        "result_sha256": terminal["result_sha256"],
    } in family["canonical_artifacts"]
    assert any(
        row["canonical_result_sha256"] == terminal["result_sha256"]
        and "Elon_August_28_to_September_4" in row["family"]
        for row in registry["terminal_do_not_repeat"]
    )
