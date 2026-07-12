"""Tests for the repository financial-terminology contract."""

from __future__ import annotations

from tools.audit_financial_terminology import audit_entries, audit_repository


def test_repository_authored_surfaces_use_financial_terminology() -> None:
    assert audit_repository() == []


def test_audit_accepts_established_financial_terms() -> None:
    findings = audit_entries(
        [
            (
                "docs/example.md",
                "Probability of profit and expected net return are evaluated net of costs.",
            ),
            (
                "docs/charts/signal-selection.svg",
                "<title>Signals passing pre-trade risk controls</title>",
            ),
        ]
    )

    assert findings == []


def test_audit_rejects_superseded_text_and_artifact_names() -> None:
    informal_phrase = "action" + " funnel"
    findings = audit_entries(
        [
            ("docs/example.md", f"The {informal_phrase} accepted one row."),
            ("docs/charts/action" + "-funnel.svg", "<title>Selection</title>"),
        ]
    )

    assert [(item.path, item.line, item.replacement) for item in findings] == [
        ("docs/charts/" + "action" + "-funnel.svg", None, "signal-selection"),
        ("docs/example.md", 1, "signal selection"),
    ]


def test_audit_covers_source_workflows_and_stale_scope_language() -> None:
    model_set = "model" + "-zoo"
    stale_scope = "multi-asset by" + " design"
    findings = audit_entries(
        [
            ("src/simple_ai_trading/example.py", f'"""A {model_set}."""'),
            (".github/workflows/example.yml", f"name: {stale_scope}"),
        ]
    )

    assert [(item.path, item.line, item.replacement) for item in findings] == [
        (
            ".github/workflows/example.yml",
            1,
            "BTC/ETH/SOL-only by design",
        ),
        (
            "src/simple_ai_trading/example.py",
            1,
            "candidate-model set",
        ),
    ]


def test_audit_rejects_legacy_branding_and_superseded_evidence_copy() -> None:
    legacy_brand = "simple" + "_bitcoin_trading"
    stale_latest = "current retained per-iteration" + " evidence is"
    stale_chart = "positive calibration" + " traces"
    findings = audit_entries(
        [
            (f"docs/{legacy_brand}/README.md", "Legacy package."),
            ("docs/evidence.md", f"The {stale_latest} Round 8."),
            ("docs/chart.svg", f"<title>{stale_chart}</title>"),
        ]
    )

    assert [(item.path, item.line, item.replacement) for item in findings] == [
        (
            "docs/chart.svg",
            1,
            "positive threshold-selection simulations",
        ),
        (
            "docs/evidence.md",
            1,
            "explicitly named latest-only evidence tracks",
        ),
        (
            f"docs/{legacy_brand}/README.md",
            None,
            "simple_ai_trading",
        ),
    ]


def test_audit_rejects_superseded_active_microstructure_contract() -> None:
    stale_feature = "Feature contract `l1-tape-causal-" + "v7`"
    stale_pair = "current v16/" + "v7"
    findings = audit_entries(
        [
            ("README.md", stale_feature),
            ("docs/model.md", stale_pair),
        ]
    )

    assert [(item.path, item.line, item.replacement) for item in findings] == [
        ("docs/model.md", 1, "current v16/v8"),
        ("README.md", 1, "Feature contract `l1-tape-causal-v8`"),
    ]


def test_audit_rejects_execution_language_for_simulated_statistics() -> None:
    stale_statistic = "executed" + " mean"

    findings = audit_entries([("docs/chart.svg", f"<text>{stale_statistic}</text>")])

    assert [(item.path, item.line, item.replacement) for item in findings] == [
        ("docs/chart.svg", 1, "simulated-trade mean")
    ]
