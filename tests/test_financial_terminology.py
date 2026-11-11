"""Tests for the repository financial-terminology contract."""

from __future__ import annotations

import pytest

from tools.audit_financial_terminology import (
    IMMUTABLE_EVIDENCE_TEXT_SHA256,
    REPO_ROOT,
    audit_entries,
    audit_repository,
)


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
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
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


@pytest.mark.parametrize("relative", sorted(IMMUTABLE_EVIDENCE_TEXT_SHA256))
def test_audit_preserves_only_exact_frozen_content(relative: str) -> None:
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert audit_entries([(relative, text)]) == []
    # A Windows checkout may materialize CRLF without changing historical text.
    assert audit_entries([(relative, text.replace("\n", "\r\n"))]) == []
    assert audit_entries([(relative, text.replace("\n", "\r"))])

    extra = "magic" + " alpha"
    changed = audit_entries([(relative, text + "\n" + extra)])
    assert any(item.term == extra for item in changed)
    assert audit_entries([(relative, extra)])[0].term == extra
    # A matching body at another authored path does not inherit the exemption.
    copied = audit_entries([("docs/copied-evidence.md", text)])
    assert copied


def test_audit_checks_modified_historical_content_on_disk(
    tmp_path, monkeypatch
) -> None:
    from tools import audit_financial_terminology as audit

    relative = next(iter(IMMUTABLE_EVIDENCE_TEXT_SHA256))
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes((REPO_ROOT / relative).read_bytes())
    monkeypatch.setattr(audit, "git_visible_files", lambda _root: [relative])
    assert audit_repository(tmp_path) == []
    with target.open("a", encoding="utf-8") as stream:
        stream.write("\n" + "magic" + " alpha")
    assert any(item.term == "magic" + " alpha" for item in audit_repository(tmp_path))


def test_frozen_content_does_not_exempt_a_superseded_filename(monkeypatch) -> None:
    from tools import audit_financial_terminology as audit

    relative = next(iter(IMMUTABLE_EVIDENCE_TEXT_SHA256))
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    name = "docs/action" + "-funnel.md"
    monkeypatch.setitem(
        audit.IMMUTABLE_EVIDENCE_TEXT_SHA256,
        name,
        audit.IMMUTABLE_EVIDENCE_TEXT_SHA256[relative],
    )
    findings = audit_entries([(name, text)])
    assert len(findings) == 1
    assert findings[0].line is None


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


@pytest.mark.parametrize(
    "text",
    [
        "Quote executed means the orders were submitted.",
        "The unexecuted mean is not an executed metric.",
        "The model zoology example is unrelated.",
    ],
)
def test_phrase_matching_does_not_flag_word_substrings(text):
    assert audit_entries([("docs/example.md", text)]) == []


@pytest.mark.parametrize("wrapped", ["{}", "({})", '"{}"', "{}:", "{}\n"])
def test_metric_phrase_is_rejected_as_a_complete_token_sequence(wrapped):
    term = "executed" + " mean"
    findings = audit_entries([("docs/example.md", wrapped.format(term))])
    assert len(findings) == 1 and findings[0].term == term


@pytest.mark.parametrize(
    "relative",
    [
        "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json",
        "docs/model-research/action-value/polymarket-us-combo-rfq-overview-adjudication-v1-2026-09-01.json",
        "docs/review/2026-09-04/nyc-sep5-frontier/durability-before.json",
        "docs/review/2026-09-04/paradex-index-publication/durability-before.json",
        "docs/review/2026-10-07/model-integration/audit-before.json",
    ],
)
def test_reclassified_evidence_is_checked_without_whole_file_exemption(relative):
    assert relative not in IMMUTABLE_EVIDENCE_TEXT_SHA256
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert audit_entries([(relative, text)]) == []
    term = "executed" + " mean"
    assert any(f.term == term for f in audit_entries([(relative, text + "\n" + term)]))


@pytest.mark.parametrize(
    "term", ["calibration threshold" + " traces", "ai" + " reviewers"]
)
def test_explicit_plural_terms_remain_enforced(term):
    assert any(f.term == term for f in audit_entries([("docs/example.md", term)]))
