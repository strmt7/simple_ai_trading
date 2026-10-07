"""Frozen CMS table selection must not infer yields from unrelated content."""

import copy
import json

import pytest

from tools.reconcile_binance_yield_distribution import _cell_state, period_table


def _fixture():
    spec = {
        "id": 7,
        "code": "exact-campaign",
        "title": "Exact campaign",
        "headers": ["Period", "Base", "Boosted", "Token"],
        "period_labels": ["First", "Second"],
    }
    rows = [
        spec["headers"],
        ["First", "4.5%", "5.4%", "$0.10"],
        ["Second", "To be updated on 2026-09-18", "unknown", "unknown"],
    ]
    table = {
        "node": "element",
        "tag": "table",
        "child": [
            {
                "node": "element",
                "tag": "tr",
                "child": [
                    {
                        "node": "element",
                        "tag": "td",
                        "child": [{"node": "text", "text": cell}],
                    }
                    for cell in row
                ],
            }
            for row in rows
        ],
    }
    root = {"node": "root", "child": [table]}
    article = {
        "success": True,
        "code": "000000",
        "data": {
            **{key: spec[key] for key in ("id", "code", "title")},
            "body": json.dumps(root),
        },
    }
    return article, spec, root, rows


def test_exact_population_and_cells_are_retained():
    article, spec, _, rows = _fixture()
    assert period_table(article, spec) == rows[1:]


@pytest.mark.parametrize(
    "change", ["identity", "duplicate", "population", "width", "kind"]
)
def test_ambiguous_or_changed_source_fails_closed(change):
    article, spec, root, _ = _fixture()
    if change == "identity":
        article["data"]["id"] = 8
    elif change == "duplicate":
        root["child"].append(copy.deepcopy(root["child"][0]))
    elif change == "population":
        spec["period_labels"].reverse()
    elif change == "width":
        root["child"][0]["child"][1]["child"].pop()
    else:
        root["child"][0]["node"] = "unsupported"
    article["data"]["body"] = json.dumps(root)
    with pytest.raises(ValueError):
        period_table(article, spec)


@pytest.mark.parametrize(
    "value,percent,expected",
    [
        ("4.5%", True, "published_numeric"),
        ("$0.10", False, "published_numeric"),
        ("To be updated on 2026-09-18", True, "unpublished_placeholder"),
        ("NaN%", True, "unparsed_no_numeric_claim"),
        ("-5%", True, "unparsed_no_numeric_claim"),
        ("$0.10 (example)", False, "unparsed_no_numeric_claim"),
    ],
)
def test_cells_never_coerce_uncertain_economics(value, percent, expected):
    assert _cell_state(value, percent=percent) == expected
