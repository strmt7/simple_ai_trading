"""One-use secret-free documentation extraction; never prints discarded text."""

from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04/paradex-index-source"
KEY_PATTERN = re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def extract(raw: bytes, title: str) -> bytes:
    parser = VisibleText()
    parser.feed(raw.decode("utf-8"))
    starts = [
        i
        for i in range(len(parser.parts) - 1)
        if parser.parts[i : i + 2] == [title, "Copy page"]
    ]
    if len(starts) != 1:
        raise ValueError("documentation main-section marker is ambiguous")
    start = starts[0]
    stop = parser.parts.index("Was this page helpful?", start)
    text = "\n".join(parser.parts[start:stop]) + "\n"
    if KEY_PATTERN.search(text) or "PRIVATE KEY-----" in text:
        raise ValueError("sensitive pattern inside selected source; stop")
    return text.encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    records = []
    if (BASE / "disposition.json").exists():
        raise FileExistsError("source extraction already recorded")
    for name, title in (
        ("mechanism", "Funding Mechanism"),
        ("history-api", "Funding data history"),
    ):
        original = BASE / (name + ".html")
        raw = original.read_bytes()
        safe = extract(raw, title)
        output = BASE / (name + "-section.txt")
        with output.open("xb") as stream:
            stream.write(safe)
            stream.flush()
            os.fsync(stream.fileno())
        records.append(
            {
                "original_path": original.relative_to(ROOT).as_posix(),
                "original_bytes": len(raw),
                "original_sha256": sha(raw),
                "removed_aws_identifier_pattern_occurrences": len(
                    KEY_PATTERN.findall(raw.decode("utf-8"))
                ),
                "section_path": output.relative_to(ROOT).as_posix(),
                "section_bytes": len(safe),
                "section_sha256": sha(safe),
                "title_marker": title,
                "disposition": "original HTML removed from working tree and publishable commit; only receipt hash and exact selected visible text retained",
            }
        )
    result = {
        "schema_version": "public-source-secret-free-section-disposition-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "GitHub push protection identified embedded AWS access-key patterns in site documentation outside the relevant visible page sections. No bypass and no source refetch.",
        "extractor_path": Path(__file__).relative_to(ROOT).as_posix(),
        "extractor_sha256": sha(Path(__file__).read_bytes()),
        "request_journal_sha256": sha((BASE / "requests.jsonl").read_bytes()),
        "algorithm": "HTMLParser visible text excluding script/style, unique title followed by Copy page, through but excluding Was this page helpful; whitespace-stripped text segments joined by LF",
        "sources": records,
        "original_bytes_reconstructible_from_published_section": False,
        "economic_market_responses_changed": False,
    }
    result["result_sha256"] = sha(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    with (BASE / "disposition.json").open("xb") as stream:
        stream.write(json.dumps(result, indent=2).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "section_bytes": [r["section_bytes"] for r in records],
                "removed_pattern_counts": [
                    r["removed_aws_identifier_pattern_occurrences"] for r in records
                ],
                "result_sha256": result["result_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
