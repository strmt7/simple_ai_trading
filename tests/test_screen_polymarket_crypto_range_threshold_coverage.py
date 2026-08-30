from __future__ import annotations

from tools.screen_polymarket_crypto_range_threshold_coverage import _screen_events


RANGE_LABELS = [
    "<60",
    "60-70",
    "70-80",
    "80-90",
    "90-100",
    "100-110",
    "110-120",
    "120-130",
    "130-140",
    "140-150",
    ">150",
]
THRESHOLDS = [str(value) for value in range(60, 170, 10)]


def _market(label: str, index: int, *, yes: str, no: str) -> dict[str, object]:
    return {
        "id": f"market-{index}",
        "conditionId": f"condition-{index}",
        "question": f"question {label}",
        "groupItemTitle": label,
        "description": "",
        "outcomes": '["Yes","No"]',
        "outcomePrices": f'["{yes}","{no}"]',
        "clobTokenIds": f'["yes-{index}","no-{index}"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "feesEnabled": True,
        "feeSchedule": None,
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.001,
    }


def test_complete_cross_event_screen_enumerates_both_coverage_directions() -> None:
    range_event = {
        "slug": "range",
        "title": "range title",
        "markets": [
            _market(label, index, yes="0.05", no="0.95")
            for index, label in enumerate(reversed(RANGE_LABELS))
        ],
    }
    threshold_event = {
        "slug": "threshold",
        "title": "threshold title",
        "markets": [
            _market(label, index + 20, yes="0.4", no="0.3")
            for index, label in enumerate(reversed(THRESHOLDS))
        ],
    }
    contract = {
        "range_event": {
            "slug": "range",
            "title": "range title",
            "expected_labels": RANGE_LABELS,
            "required_rule_fragments": [],
        },
        "threshold_event": {
            "slug": "threshold",
            "title": "threshold title",
            "expected_thresholds": THRESHOLDS,
            "required_rule_fragments": [],
        },
        "shared_boundaries": [
            {"threshold_label": value, "range_start_label": label}
            for value, label in zip(THRESHOLDS[:-1], RANGE_LABELS[1:], strict=True)
        ],
    }

    legs, packages = _screen_events(range_event, threshold_event, contract)

    assert len(legs) == 22
    assert len(packages) == 20
    assert {row["direction"] for row in packages} == {
        "lower_coverage",
        "upper_coverage",
    }
    upper_60 = next(
        row
        for row in packages
        if row["direction"] == "upper_coverage" and row["boundary"] == "60"
    )
    assert upper_60["range_labels"] == RANGE_LABELS[1:]
    assert upper_60["displayed_price_sum_pUSD"] == "0.80"
    assert upper_60["passes_strict_displayed_gross_gate"] is True
    lower_60 = next(
        row
        for row in packages
        if row["direction"] == "lower_coverage" and row["boundary"] == "60"
    )
    assert lower_60["range_labels"] == RANGE_LABELS[:2]
    assert lower_60["displayed_price_sum_pUSD"] == "0.50"
