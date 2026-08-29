from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "recent-structural-edge-literature-delta-contract-v1.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "recent-structural-edge-literature-delta-v1-2026-08-29.json"
)
DATA_ROOT = ROOT / "data/recent-structural-edge-literature-delta-v1"
RAW_PATH = DATA_ROOT / "raw/arxiv-query.atom"
JOURNAL_PATH = DATA_ROOT / "request-journal.jsonl"

ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("timestamp must include an offset")
    return parsed.astimezone(UTC)


def _journal(payload: dict[str, Any]) -> None:
    with JOURNAL_PATH.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(url: str, timeout_seconds: int) -> tuple[bytes, int]:
    requested_at_ms = time.time_ns() // 1_000_000
    intent = {
        "method": "GET",
        "name": "arxiv-recent-structural-edge-literature-query",
        "phase": "intent",
        "request_body_sha256": _sha256(b""),
        "requested_at_ms": requested_at_ms,
        "url": url,
    }
    _journal(intent)
    request = Request(
        url,
        method="GET",
        headers={"User-Agent": "simple-ai-trading-public-research/1"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
            status_code = response.status
    except HTTPError as exc:
        response_bytes = exc.read()
        status_code = exc.code
        RAW_PATH.write_bytes(response_bytes)
        _journal(
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
                "response_bytes": len(response_bytes),
                "response_sha256": _sha256(response_bytes),
                "status_code": status_code,
            }
        )
        raise
    RAW_PATH.write_bytes(response_bytes)
    _journal(
        {
            **intent,
            "completed_at_ms": time.time_ns() // 1_000_000,
            "phase": "completed",
            "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
            "response_bytes": len(response_bytes),
            "response_sha256": _sha256(response_bytes),
            "status_code": status_code,
        }
    )
    return response_bytes, status_code


def _text(node: ElementTree.Element, name: str) -> str:
    child = node.find(f"{{{ATOM}}}{name}")
    return "" if child is None or child.text is None else " ".join(child.text.split())


def _paper(entry: ElementTree.Element) -> dict[str, Any]:
    categories = sorted(
        category.attrib["term"]
        for category in entry.findall(f"{{{ATOM}}}category")
        if "term" in category.attrib
    )
    authors = [
        _text(author, "name") for author in entry.findall(f"{{{ATOM}}}author")
    ]
    doi_node = entry.find(f"{{{ARXIV}}}doi")
    return {
        "arxiv_id_url": _text(entry, "id"),
        "authors": authors,
        "categories": categories,
        "doi": "" if doi_node is None or doi_node.text is None else doi_node.text.strip(),
        "published_utc": _text(entry, "published"),
        "summary": _text(entry, "summary"),
        "title": _text(entry, "title"),
        "updated_utc": _text(entry, "updated"),
    }


def _score(paper: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    haystack = f"{paper['title']} {paper['summary']}".casefold()
    hits = sorted(
        keyword
        for keyword in contract["classification"]["structural_keywords"]
        if keyword.casefold() in haystack
    )
    venue_hits = sorted(
        keyword
        for keyword in contract["classification"]["venue_keywords"]
        if keyword.casefold() in haystack
    )
    paper["structural_keyword_hits"] = hits
    paper["venue_keyword_hits"] = venue_hits
    paper["candidate_lead"] = bool(hits and venue_hits)
    return paper


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = contract["contract_sha256"]
    if _canonical_hash(contract, "contract_sha256") != contract_hash:
        raise RuntimeError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    frozen_at = _parse_utc(contract["frozen_at_utc"])
    now = datetime.now(UTC)
    if frozen_at > now:
        raise RuntimeError("frozen_at_utc is in the future")
    if DATA_ROOT.exists() or RESULT_PATH.exists():
        raise RuntimeError("one-use output already exists")
    RAW_PATH.parent.mkdir(parents=True)

    response_bytes, status_code = _capture(
        contract["request"]["url"], contract["request"]["timeout_seconds"]
    )
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")
    root = ElementTree.fromstring(response_bytes)
    total_node = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    total_results = int(total_node.text) if total_node is not None else None
    papers = [
        _score(_paper(entry), contract)
        for entry in root.findall(f"{{{ATOM}}}entry")
    ]
    cutoff = _parse_utc(contract["classification"]["published_not_before_utc"])
    recent = [paper for paper in papers if _parse_utc(paper["published_utc"]) >= cutoff]
    known_ids = set(contract["classification"]["known_arxiv_ids"])
    novel = [
        paper
        for paper in recent
        if not any(known_id in paper["arxiv_id_url"] for known_id in known_ids)
    ]
    candidates = [paper for paper in novel if paper["candidate_lead"]]
    cutoff_population_complete = len(papers) < contract["request"]["max_results"] or (
        bool(papers) and _parse_utc(papers[-1]["published_utc"]) < cutoff
    )

    result: dict[str, Any] = {
        "schema_version": "recent-structural-edge-literature-delta-v1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract_hash,
        },
        "authority": contract["authority"],
        "capture": {
            "request_count": 1,
            "status_code": status_code,
            "raw_path": RAW_PATH.relative_to(ROOT).as_posix(),
            "raw_sha256": _sha256(response_bytes),
            "raw_bytes": len(response_bytes),
        },
        "population": {
            "api_total_results": total_results,
            "returned_entries": len(papers),
            "published_since_cutoff": len(recent),
            "novel_since_cutoff": len(novel),
            "candidate_leads": len(candidates),
            "population_complete": total_results is not None
            and total_results <= contract["request"]["max_results"],
            "published_since_cutoff_population_complete": cutoff_population_complete,
        },
        "candidate_leads": candidates,
        "all_returned_papers": papers,
        "decision": {
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "next_action": (
                "source_validate_the_highest_scored_novel_mechanism_before_any_venue_data"
                if candidates
                else "no_recent_primary_literature_trigger_reopen_only_on_a_new_paper_or_venue_change"
            ),
        },
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps({"population": result["population"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
