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
