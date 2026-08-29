import hashlib
import json
from pathlib import Path

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import _canonical_hash
from tools.screen_polymarket_source_selected_crypto_negrisk_series import (
    _event_series_matches,
    _select_series,
    _series_is_eligible,
)


ROOT = Path(__file__).resolve().parents[1]


def test_series_scope_uses_asset_tokens_not_substrings() -> None:
    base = {
        "id": "1",
        "active": True,
        "closed": False,
        "archived": False,
        "recurrence": "weekly",
        "slug": "resolution-calendar",
        "title": "Resolution calendar",
    }
    assert not _series_is_eligible(base)
    assert _series_is_eligible({**base, "title": "Bitcoin weekly price range"})
    assert _series_is_eligible({**base, "slug": "eth-weekly-range"})
    assert _series_is_eligible({**base, "ticker": "SOL"})


def test_selection_is_first_eligible_server_order_and_rejects_embedded_events() -> None:
    ineligible = {
        "id": "1",
        "active": True,
        "closed": False,
        "recurrence": "weekly",
        "title": "Election tracker",
    }
    first = {**ineligible, "id": "2", "title": "ETH weekly range"}
    second = {**ineligible, "id": "3", "title": "Bitcoin weekly range"}
    selected, classifications = _select_series([ineligible, first, second])
    assert selected == first
    assert [row["eligible"] for row in classifications] == [False, True, True]

    with_events = {**first, "events": []}
    try:
        _select_series([with_events])
    except RuntimeError as exc:
        assert "exclude_events=true" in str(exc)
    else:
        raise AssertionError("embedded events must fail closed")


def test_event_must_bind_selected_series() -> None:
    event = {"series": [{"id": "17"}, {"id": "18"}]}
    assert _event_series_matches(event, "18")
    assert not _event_series_matches(event, "19")


def test_consumed_series_artifacts_are_hash_bound_and_empty() -> None:
    pairs = (
        (
            "docs/model-research/action-value/polymarket-source-selected-crypto-negrisk-series-contract-v1-2026-08-29.json",
            "contract_sha256",
            "a13f554d2b6b40dce3ea83236bff5cd99993f41aa1c988fe38eb99dc28d43f5e",
        ),
        (
            "docs/model-research/action-value/polymarket-source-selected-crypto-negrisk-series-result-v1-2026-08-29.json",
            "result_sha256",
            "73ae75ccf391a30ef592f649f63ef535e29865e1234bbff03021c806fd75268b",
        ),
        (
            "docs/model-research/action-value/polymarket-btc-multi-strikes-weekly-negrisk-contract-v1-2026-08-29.json",
            "contract_sha256",
            "ca2f7b556207a0d2d04c006ed502bf35a673a87642e0e5dd443227af8aba82d2",
        ),
        (
            "docs/model-research/action-value/polymarket-btc-multi-strikes-weekly-negrisk-result-v1-2026-08-29.json",
            "result_sha256",
            "f032753b45c82b2e0945d1a8c0e0d5fc01f8fb1727cdad34e73064c7590417ba",
        ),
    )
    for relative, key, expected in pairs:
        payload = json.loads((ROOT / relative).read_text(encoding="ascii"))
        assert payload[key] == expected
        assert _canonical_hash(payload, key) == expected

    source_result = json.loads((ROOT / pairs[1][0]).read_text(encoding="ascii"))
    assert source_result["selection"]["selected"]["series_id"] == "10684"
    assert (
        source_result["capture"]["event_population_complete_under_frozen_filter"]
        is True
    )
    assert source_result["screen"]["fixed_negrisk_event_count"] == 0
    multi_result = json.loads((ROOT / pairs[3][0]).read_text(encoding="ascii"))
    assert multi_result["capture"]["returned_event_count"] == 0
    assert multi_result["capture"]["population_complete_under_frozen_filter"] is True

    for relative, expected in (
        (
            "data/polymarket-source-selected-crypto-negrisk-series-v1/raw/series.json",
            "f28b1e8d97d4f918fe0640faf7d8dc7594344675892a379db6df820e0cd37a34",
        ),
        (
            "data/polymarket-source-selected-crypto-negrisk-series-v1/raw/events.json",
            "1caf48c002786edb458302862e4e33fbff3b2afecb9ec796169d7a3881ad7e9e",
        ),
        (
            "data/polymarket-btc-multi-strikes-weekly-negrisk-v1/raw/events.json",
            "1caf48c002786edb458302862e4e33fbff3b2afecb9ec796169d7a3881ad7e9e",
        ),
    ):
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
