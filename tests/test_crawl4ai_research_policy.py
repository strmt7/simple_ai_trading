"""Keep the user's crawler requirement separate from native financial captures."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_caveman_and_cocoindex_routes_are_mandatory_and_present() -> None:
    root = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Caveman and CocoIndex Code are mandatory for every AI agent" in root
    for skill in ("caveman", "cocoindex-code-search"):
        assert f".agents/skills/{skill}/SKILL.md" in root
        assert (ROOT / ".agents/skills" / skill / "SKILL.md").is_file()
    caveman = " ".join(
        (ROOT / ".agents/skills/caveman/SKILL.md").read_text(encoding="utf-8").split()
    )
    assert "internal AI communication only" in caveman
    assert "Never lose negation, numbers, units" in caveman
    assert "remain normal clear prose" in caveman
    assert "Trading/capture and verification rules do not change" in caveman


def test_crawl4ai_is_required_by_the_root_and_workflow() -> None:
    root = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    workflow = (ROOT / "docs/AGENT_WORKFLOWS.md").read_text(encoding="utf-8")
    assert "[Crawl4AI](docs/CRAWL4AI_RESEARCH.md)" in root
    assert "explicit user requirement" in root
    assert "Crawl4AI is mandatory for web-page source reading" in workflow
    assert (ROOT / "docs/CRAWL4AI_RESEARCH.md").is_file()


def test_native_evidence_and_failed_crawls_keep_their_boundaries() -> None:
    text = " ".join(
        (ROOT / "docs/CRAWL4AI_RESEARCH.md").read_text(encoding="utf-8").split()
    )
    for boundary in (
        "not exactly one HTTP GET",
        "Rendered HTML is not the original wire body",
        "Native `success=true` alone is insufficient",
        "require HTTP 200",
        "existing bounded native transport",
        "not permission to claim that Crawl4AI was used",
        "alter consumed contracts",
        "No provider files were copied or modified",
    ):
        assert boundary in text
    assert "-o all" in text
    assert "check_robots_txt=true" in text
    assert "no new runtime dependency" in text


def test_crawler_probe_does_not_qualify_native_success_without_content() -> None:
    review = json.loads(
        (ROOT / "docs/review/2026-10-08/crawl4ai-research-result.json").read_bytes()
    )
    assert review["native_result"]["success"] is True
    assert review["native_result"]["http_status"] == 202
    assert review["native_result"]["markdown_utf8_bytes"] == 1
    assert review["source_admitted"] is False
    assert review["retry_or_alternate_url_used"] is False
    assert review["financial_claims_qualified"] is False
    assert review["financial_counts_unchanged"]["qualified_stable_edges"] == 0
