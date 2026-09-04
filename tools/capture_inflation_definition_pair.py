"""One fixed public definition pair; never inspect market prices or outcomes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools.screen_polymarket_exact_negrisk_long_only_frontier_v2 import bounded_request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04/inflation-definition-pair"
SLUGS = (
    "august-inflation-us-annual-1786474662954",
    "august-inflation-us-monthly-1786474662954",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    plan_path = BASE / "plan.json"
    plan = json.loads(plan_path.read_bytes())
    assert plan["slugs"] == list(SLUGS)
    for p, digest in plan["implementation_sha256"].items():
        if sha((ROOT / p).read_bytes()) != digest:
            raise ValueError("frozen implementation differs")
    outputs = [
        BASE / f"{i}-{suffix}"
        for i in range(2)
        for suffix in ("raw.json", "journal.jsonl")
    ]
    if any(p.exists() for p in outputs) or (BASE / "definitions.json").exists():
        raise FileExistsError("definition pair consumed; no retry")
    definitions = []
    for i, slug in enumerate(SLUGS):
        raw_path, journal_path = outputs[2 * i : 2 * i + 2]
        raw, receipt = bounded_request(
            method="GET",
            url="https://gamma-api.polymarket.com/events/slug/" + slug,
            body=b"",
            name=slug,
            raw_path=raw_path,
            raw_relative_path=raw_path.relative_to(ROOT).as_posix(),
            journal_path=journal_path,
        )
        event = json.loads(raw)
        if not isinstance(event, dict) or event.get("slug") != slug:
            raise ValueError("unexpected exact event identity; source retained")
        markets = event.get("markets")
        if not isinstance(markets, list) or len(markets) != plan["expected_counts"][i]:
            raise ValueError("complete declared market count differs; no size repair")
        definitions.append(
            {
                "slug": slug,
                "id": event.get("id"),
                "receipt": receipt,
                "event_fields": {
                    k: event.get(k)
                    for k in (
                        "title",
                        "description",
                        "active",
                        "closed",
                        "negRisk",
                        "negRiskAugmented",
                        "endDate",
                    )
                },
                "markets": [
                    {
                        k: m.get(k)
                        for k in (
                            "id",
                            "conditionId",
                            "question",
                            "description",
                            "groupItemTitle",
                            "active",
                            "closed",
                            "acceptingOrders",
                            "resolutionSource",
                            "endDate",
                        )
                    }
                    for m in markets
                ],
            }
        )
    result = {
        "schema_version": "inflation-definition-pair-source-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(plan_path.read_bytes()),
        "definitions": definitions,
        "source_only": True,
        "prices_inspected": False,
        "accepted_edge": False,
        "account_requests": 0,
        "book_requests": 0,
        "orders": 0,
        "protected_access": False,
    }
    result["result_sha256"] = sha(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    with (BASE / "definitions.json").open("xb") as stream:
        stream.write(json.dumps(result, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "event_counts": [len(x["markets"]) for x in definitions],
                "prices_inspected": False,
            }
        )
    )


if __name__ == "__main__":
    main()
