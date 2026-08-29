from tools.screen_polymarket_source_selected_crypto_negrisk_series import (
    _event_series_matches,
    _select_series,
    _series_is_eligible,
)


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
