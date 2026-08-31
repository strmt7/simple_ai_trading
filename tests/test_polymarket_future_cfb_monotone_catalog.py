from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.screen_polymarket_cfb_monotone_catalog import (
    _screen_event,
    _validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / ("polymarket-future-cfb-monotone-catalog-contract-v1-2026-08-30.json")
RESULT = BASE / ("polymarket-future-cfb-monotone-catalog-result-v1-2026-08-30.json")
RAW_DIR = ROOT / "data/polymarket-future-cfb-monotone-catalog-v1"
RAW = RAW_DIR / "raw/events.json"
JOURNAL = RAW_DIR / "request-journal.jsonl"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
SEP8_12_CONTRACT = BASE / (
    "polymarket-cfb-september-8-12-complete-monotone-catalog-contract-v1-2026-08-31.json"
)
SEP8_12_RESULT = BASE / (
    "polymarket-cfb-september-8-12-complete-monotone-catalog-result-v1-2026-08-31.json"
)
SEP8_12_RAW_DIR = ROOT / (
    "data/polymarket-cfb-september-8-12-complete-monotone-catalog-v1"
)
RETAINED_PARITY = BASE / (
    "polymarket-holding-yield-retained-multi-event-parity-v1-2026-08-31.json"
)
HISTORICAL_TWO_LEG_TOOL = BASE / (
    "source-bound-implementations/screen-polymarket-exact-two-leg-package-ba7ebaa1.py.raw"
)
HISTORICAL_TWO_LEG_LINEAGE = (
    BASE / "historical-two-leg-tool-immutable-lineage-v1-2026-08-31.json"
)
HISTORICAL_CFB_RUNNER = BASE / (
    "source-bound-implementations/screen-polymarket-cfb-monotone-catalog-d26ef14f.py.raw"
)
HISTORICAL_CFB_LINEAGE = (
    BASE / "historical-cfb-catalog-runner-immutable-lineage-v1-2026-08-31.json"
)
SEP8_12_METHOD_ADJUDICATION = BASE / (
    "polymarket-cfb-september-8-12-empty-method-adjudication-v1-2026-08-31.json"
)
CFB_SIDE_SPECIFIC_CORRECTION = (
    BASE / "polymarket-cfb-retained-side-specific-correction-v1-2026-08-31.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_path(implementation: dict[str, str]) -> Path:
    historical = {
        "ba7ebaa1fb5235ba026f7cf4cb21ff73ba19430fee4a4f678fbdea37635eb32f": (
            HISTORICAL_TWO_LEG_TOOL
        ),
        "d26ef14fe6178e209ceb1ec98c37d7797e054108d623de3ba16e8fb3e2fd2467": (
            HISTORICAL_CFB_RUNNER
        ),
    }
    return historical.get(implementation["sha256"], ROOT / implementation["path"])


def test_complete_one_request_catalog_excludes_consumed_events() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    raw = RAW.read_bytes()
    payload = json.loads(raw)
    journal = [json.loads(line) for line in JOURNAL.read_text().splitlines()]

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    for implementation in contract["implementations"]:
        assert (
            hashlib.sha256(
                _implementation_path(implementation).read_bytes()
            ).hexdigest()
            == (implementation["sha256"])
        )
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["capture"]["returned_event_count"] == len(payload["events"]) == 18
    assert result["capture"]["population_complete_under_frozen_filter"] is True
    assert "next_cursor" not in payload
    slugs = {event["slug"] for event in payload["events"]}
    assert "cfb-ballst-ohiost-2026-09-05" not in slugs
    assert "cfb-clmsn-lsu-2026-09-05" not in slugs


def test_all_relations_are_retained_and_reject_before_books() -> None:
    result = _load(RESULT)
    screen = result["screen"]

    assert screen["included_event_count"] == 13
    assert screen["excluded_event_count"] == 5
    assert screen["complete_relation_count"] == len(screen["relations"]) == 19
    assert screen["candidate_count_strictly_below_payout_floor"] == 0
    best = screen["best_relation"]
    assert best["event_slug"] == "cfb-sjst-emich-2026-09-04"
    assert best["family"] == "full_game_total"
    assert Decimal(best["displayed_price_sum_per_share_pUSD"]) == Decimal("1.02")
    assert screen["depth_candidate"] is None
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["fee_requests"] == 0
    assert result["adjudication"]["accepted_edge"] is False


def test_population_contract_fails_closed_on_series_drift() -> None:
    contract = _load(CONTRACT)
    drifted = deepcopy(contract)
    drifted["capture"]["series_id"] = "different"
    drifted["contract_sha256"] = _self_hash(drifted, "contract_sha256")

    with pytest.raises(RuntimeError, match="population contract changed"):
        _validate_contract(drifted, CONTRACT.resolve())


def test_registry_terminalizes_only_the_exact_catalog_window() -> None:
    result = _load(RESULT)
    registry = _load(REGISTRY)

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 30
    )
    assert {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in family["canonical_artifacts"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["canonical_result_sha256"] == result["result_sha256"]
    )
    assert "do_not_paginate_narrow_repeat" in terminal["reason"]
    assert "2026_09_03T00_00_00Z" in family["retry_trigger"]


def test_sep8_12_catalog_is_a_complete_empty_terminal_population() -> None:
    contract = _load(SEP8_12_CONTRACT)
    result = _load(SEP8_12_RESULT)
    adjudication = _load(SEP8_12_METHOD_ADJUDICATION)
    raw_path = SEP8_12_RAW_DIR / "raw/events.json"
    journal_path = SEP8_12_RAW_DIR / "request-journal.jsonl"
    raw = raw_path.read_bytes()
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]

    assert contract["contract_sha256"] == (
        "dc73c6b56d2ae89a4eaab93666f7b5b6bf5856daed1663ef5b5fef4e76f8c178"
    )
    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert result["result_sha256"] == (
        "5511d2446585d02577c91249f454e458b1b7551e656cd4f68abd8dae0ac526be"
    )
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    for implementation in contract["implementations"]:
        assert (
            hashlib.sha256(
                _implementation_path(implementation).read_bytes()
            ).hexdigest()
            == implementation["sha256"]
        )
    assert hashlib.sha256(raw).hexdigest() == (
        "1caf48c002786edb458302862e4e33fbff3b2afecb9ec796169d7a3881ad7e9e"
    )
    assert len(raw) == 97
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["capture"]["returned_event_count"] == 0
    assert result["capture"]["next_cursor_present"] is False
    assert result["capture"]["population_complete_under_frozen_filter"] is True
    assert result["screen"]["complete_relation_count"] == 0
    assert result["screen"]["depth_candidate"] is None
    assert result["authority"]["book_requests"] == 0
    assert adjudication["result_sha256"] == (
        "29d8342af4060c5e75d9e4cfca88db38e59e67f32a332847b046d0b1f900d5a0"
    )
    assert _self_hash(adjudication, "result_sha256") == adjudication["result_sha256"]
    assert adjudication["decision_invariance"]["price_rows_evaluated"] == 0
    assert adjudication["finding"]["representation_mismatch"] is True

    registry = _load(REGISTRY)
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 30
    )
    assert {
        "path": SEP8_12_RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in family["canonical_artifacts"]
    assert {
        "path": SEP8_12_METHOD_ADJUDICATION.relative_to(ROOT).as_posix(),
        "result_sha256": adjudication["result_sha256"],
    } in family["canonical_artifacts"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["canonical_result_sha256"] == adjudication["result_sha256"]
    )
    assert (
        "zero_price_rows_made_the_empty_population_decision_invariant"
        in terminal["reason"]
    )


def test_retained_multi_event_parity_rejects_every_exact_relation() -> None:
    parity = _load(RETAINED_PARITY)

    assert parity["result_sha256"] == (
        "b3f5480fea15bb73990a278dba93356cb34c49f91d82e2624c79fef8191e09f4"
    )
    assert _self_hash(parity, "result_sha256") == parity["result_sha256"]
    markets: dict[str, dict[str, object]] = {}
    for source in parity["sources"]["retained_market_files"]:
        path = ROOT / source["path"]
        raw = path.read_bytes()
        assert len(raw) == source["bytes"]
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]
        payload = json.loads(raw)
        market = payload[0] if isinstance(payload, list) else payload
        markets[market["conditionId"]] = market

    for relation in parity["relations"]:
        total = Decimal("0")
        for leg in relation["legs"]:
            market = markets[leg["condition_id"]]
            expected = (
                Decimal(str(market["bestAsk"]))
                if leg["side"] == "YES"
                else Decimal("1") - Decimal(str(market["bestBid"]))
            )
            assert Decimal(leg["conservative_price_pUSD"]) == expected
            total += expected
        floor = Decimal(relation["optimistic_rule_consistent_payout_floor_pUSD"])
        assert total == Decimal(relation["displayed_rejection_price_sum_pUSD"])
        assert total - floor == Decimal(relation["price_minus_floor_pUSD"])
        assert relation["strictly_sub_floor"] is (total < floor)

    assert parity["summary"]["complete_relation_count"] == 7
    assert parity["summary"]["strictly_sub_floor_candidate_count"] == 0
    assert parity["summary"]["best_price_minus_floor_pUSD"] == "0.01"
    assert parity["authority"]["network_requests"] == 0
    assert parity["verdict"]["accepted_edge"] is False

    registry = _load(REGISTRY)
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    assert {
        "path": RETAINED_PARITY.relative_to(ROOT).as_posix(),
        "result_sha256": parity["result_sha256"],
    } in family["canonical_artifacts"]


def test_historical_two_leg_tool_uses_immutable_lineage_sidecar() -> None:
    lineage = _load(HISTORICAL_TWO_LEG_LINEAGE)
    sidecar = HISTORICAL_TWO_LEG_TOOL.read_bytes()

    assert lineage["result_sha256"] == (
        "a584e3665300bb405183370a66f47e2fc83ddc43764251ec236c2341f7a4d1ef"
    )
    assert _self_hash(lineage, "result_sha256") == lineage["result_sha256"]
    assert len(sidecar) == lineage["immutable_sidecar"]["bytes"] == 17547
    assert lineage["immutable_sidecar"]["sha256"] == (
        "ba7ebaa1fb5235ba026f7cf4cb21ff73ba19430fee4a4f678fbdea37635eb32f"
    )
    assert hashlib.sha256(sidecar).hexdigest() == lineage["immutable_sidecar"]["sha256"]
    assert (
        lineage["affected_bindings"]["reference_count"]
        == len(lineage["affected_bindings"]["paths"])
        == 32
    )
    assert lineage["repair"]["consumed_contracts_changed"] is False


def test_current_cfb_runner_uses_side_specific_rejection_prices() -> None:
    retained = json.loads(
        (
            ROOT
            / "data/polymarket-cfb-september-7-complete-monotone-catalog-v1/raw/events.json"
        ).read_bytes()
    )
    relations, _summary = _screen_event(retained["events"][0])

    assert relations
    for relation in relations:
        assert relation["price_semantics"] == "bestAsk_or_1-bestBid_rejection_only"
        assert relation["superset_positive_price_source"] in {"bestAsk", "1-bestBid"}
        assert relation["subset_complement_price_source"] in {
            "bestAsk",
            "1-bestBid",
        }
        assert Decimal(relation["displayed_price_sum_per_share_pUSD"]) == (
            Decimal(relation["superset_positive_price_pUSD"])
            + Decimal(relation["subset_complement_price_pUSD"])
        )


def test_all_retained_cfb_relations_are_corrected_to_side_specific_prices() -> None:
    correction = _load(CFB_SIDE_SPECIFIC_CORRECTION)

    assert correction["result_sha256"] == (
        "cc247e76896eea386ae252474099af3c6872393e77ea81b9d4262671f17ac9e0"
    )
    assert _self_hash(correction, "result_sha256") == correction["result_sha256"]
    assert (
        correction["aggregate"]["complete_relation_count"]
        == len(correction["relations"])
        == 139
    )
    assert correction["aggregate"]["source_outcome_price_candidate_count"] == 8
    assert correction["aggregate"]["side_specific_candidate_count"] == 0

    markets_by_window: dict[str, dict[str, dict[str, object]]] = {}
    for source in correction["sources"]:
        result_path = ROOT / source["result_path"]
        raw_path = ROOT / source["raw_path"]
        assert (
            hashlib.sha256(result_path.read_bytes()).hexdigest()
            == source["result_file_sha256"]
        )
        raw = raw_path.read_bytes()
        assert len(raw) == source["raw_bytes"]
        assert hashlib.sha256(raw).hexdigest() == source["raw_sha256"]
        payload = json.loads(raw)
        markets_by_window[source["window"]] = {
            str(market["id"]): market
            for event in payload["events"]
            for market in event.get("markets", [])
        }

    for relation in correction["relations"]:
        markets = markets_by_window[relation["window"]]
        superset = markets[relation["superset_market_id"]]
        subset = markets[relation["subset_market_id"]]
        superset_price = (
            Decimal(str(superset["bestAsk"]))
            if relation["superset_price_source"] == "bestAsk"
            else Decimal("1") - Decimal(str(superset["bestBid"]))
        )
        subset_price = (
            Decimal(str(subset["bestAsk"]))
            if relation["subset_complement_price_source"] == "bestAsk"
            else Decimal("1") - Decimal(str(subset["bestBid"]))
        )
        total = superset_price + subset_price
        floor = Decimal(relation["minimum_payout_pUSD"])
        assert superset_price == Decimal(relation["superset_price_pUSD"])
        assert subset_price == Decimal(relation["subset_complement_price_pUSD"])
        assert total == Decimal(relation["side_specific_sum_pUSD"])
        assert relation["strictly_sub_floor"] is (total < floor)

    registry = _load(REGISTRY)
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 30
    )
    assert {
        "path": CFB_SIDE_SPECIFIC_CORRECTION.relative_to(ROOT).as_posix(),
        "result_sha256": correction["result_sha256"],
    } in family["canonical_artifacts"]


def test_historical_cfb_runner_uses_immutable_lineage_sidecar() -> None:
    lineage = _load(HISTORICAL_CFB_LINEAGE)
    sidecar = HISTORICAL_CFB_RUNNER.read_bytes()

    assert lineage["result_sha256"] == (
        "e55df3c23a28a49cc346f3d382a04b0fa9410612ae519be6e59547a85c3b4627"
    )
    assert _self_hash(lineage, "result_sha256") == lineage["result_sha256"]
    assert len(sidecar) == lineage["immutable_sidecar"]["bytes"] == 11535
    assert hashlib.sha256(sidecar).hexdigest() == lineage["immutable_sidecar"]["sha256"]
    assert (
        lineage["affected_bindings"]["reference_count"]
        == len(lineage["affected_bindings"]["paths"])
        == 9
    )
    assert lineage["prospective_correction"]["current_path_sha256"] == (
        hashlib.sha256(
            (ROOT / "tools/screen_polymarket_cfb_monotone_catalog.py").read_bytes()
        ).hexdigest()
    )
