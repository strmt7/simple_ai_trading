"""Tests for the repository financial-terminology contract."""

from __future__ import annotations

from tools.audit_financial_terminology import audit_entries, audit_repository


def test_audit_does_not_read_retained_raw_evidence(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tools import audit_financial_terminology as audit

    relative = "docs/model-research/polymarket/raw/example/source.raw.py"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\xff")
    monkeypatch.setattr(audit, "git_visible_files", lambda _root: [relative])

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("raw evidence must not be read by terminology lint")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    assert audit.audit_repository(tmp_path) == []


def test_raw_exclusion_does_not_hide_authored_research() -> None:
    term = "policy" + " replay"
    findings = audit_entries(
        [
            ("docs/model-research/polymarket/raw/example/source.raw.md", term),
            ("docs/model-research/polymarket/review.md", term),
        ]
    )
    assert [item.path for item in findings] == [
        "docs/model-research/polymarket/review.md"
    ]


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


def test_audit_preserves_exact_frozen_evidence_bytes() -> None:
    frozen_term = "policy" + " replay"
    frozen_overlay_term = "ai" + " reviewer"
    findings = audit_entries(
        [
            (
                "docs/model-research/action-value/"
                "round-049-cost-aware-action-hurdle-tcn-design.json",
                f'"uses": ["fixed-{frozen_term}"]',
            ),
            (
                "docs/model-research/action-value/"
                "round-074-event-sequence-model-design-v116.json",
                f'"role": "{frozen_overlay_term}"',
            ),
            (
                "src/simple_ai_trading/impact_absorption_ai_contract_screen.py",
                f'"""Frozen {frozen_overlay_term} contract."""',
            ),
            ("docs/current.json", f'"uses": ["fixed-{frozen_term}"]'),
            (
                "docs/model-research/action-value/"
                "polymarket-us-combo-rfq-overview-adjudication-v1-2026-09-01.json",
                "executed" + " mean",
            ),
        ]
    )

    assert [(item.path, item.line) for item in findings] == [("docs/current.json", 1)]


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
