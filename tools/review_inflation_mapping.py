"""Source-bound definition audit and synthetic aggregation counterexample."""

from datetime import datetime, timezone
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04/inflation-definition-pair"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def aggregation(values, factors):
    """Toy common-weight positive components, not an implementation of BLS."""
    if not values or len(values) != len(factors):
        raise ValueError("nonempty matched components required")
    if any(not x.is_finite() or x <= 0 for x in (*values, *factors)):
        raise ValueError("finite positive values and factors required")
    unadjusted = sum(values, D(0))
    adjusted = sum((x / f for x, f in zip(values, factors, strict=True)), D(0))
    return unadjusted, adjusted


def build_review():
    definitions = json.loads((BASE / "definitions.json").read_bytes())
    body = {k: v for k, v in definitions.items() if k != "result_sha256"}
    if (
        sha(
            json.dumps(
                body, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        != definitions["result_sha256"]
    ):
        raise ValueError("definition artifact differs")
    plan = json.loads((BASE / "plan.json").read_bytes())
    if sha((BASE / "plan.json").read_bytes()) != definitions["plan_sha256"]:
        raise ValueError("plan binding differs")
    for i, e in enumerate(definitions["definitions"]):
        raw = (BASE / f"{i}-raw.json").read_bytes()
        receipt = e["receipt"]
        journal = [
            json.loads(line)
            for line in (BASE / f"{i}-journal.jsonl").read_bytes().splitlines()
        ]
        if (
            len(journal) != 2
            or journal[1] != receipt
            or sha(raw) != receipt["response_sha256"]
        ):
            raise ValueError("raw definition receipt differs")
        event = json.loads(raw)
        if (
            event["slug"] != plan["slugs"][i]
            or len(event["markets"]) != plan["expected_counts"][i]
        ):
            raise ValueError("exact population differs")
        for source, saved in zip(event["markets"], e["markets"], strict=True):
            if {k: source.get(k) for k in saved} != saved:
                raise ValueError("definition projection differs")
    annual, monthly = definitions["definitions"]
    for e, marker in (
        (annual, "before seasonal adjustment"),
        (monthly, "seasonally adjusted Consumer Price Index for All Urban Consumers"),
    ):
        descriptions = {m["description"] for m in e["markets"]}
        if len(descriptions) != 1 or marker not in next(iter(descriptions)):
            raise ValueError("series definitions absent or inconsistent")
        text = next(iter(descriptions))
        for phrase in (
            "August 2026",
            "one decimal point",
            "most recent previous month with available data",
        ):
            if phrase not in text:
                raise ValueError("precision or fallback semantics differ")
    usage = json.dumps(json.loads((BASE / "bls-tool-extraction.json").read_bytes()))
    methods = json.dumps(
        json.loads((BASE / "bls-aggregation-tool-extraction.json").read_bytes())
    )
    if (
        "45 of the 81 components" not in usage
        or "adjusted detail is aggregated" not in methods
    ):
        raise ValueError("retained method evidence absent")
    scenarios = []
    for values in ((D(90), D(110)), (D(110), D(90))):
        nsa, sa = aggregation(values, (D("0.9"), D("1.1")))
        scenarios.append(
            {
                "components": list(map(str, values)),
                "unadjusted": str(nsa),
                "adjusted": str(sa),
                "annual_pct_with_base_200": str(100 * (nsa / 200 - 1)),
                "monthly_pct_with_previous_sa_200": str(100 * (sa / 200 - 1)),
            }
        )
    sources = [p for p in BASE.iterdir() if p.is_file() and p.name != "review.json"]
    sources += [Path(__file__)] + [ROOT / p for p in plan["implementation_sha256"]]
    for path, expected in plan["implementation_sha256"].items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise ValueError("capture implementation differs")
    return {
        "schema_version": "inflation-mapping-method-review-v1",
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p.read_bytes())
            for p in sorted(set(sources))
        },
        "event_ids": [e["id"] for e in definitions["definitions"]],
        "market_counts": [len(e["markets"]) for e in definitions["definitions"]],
        "annual_adjustment": "unadjusted",
        "monthly_adjustment": "seasonally adjusted",
        "rounding": "one decimal percentage point in the official release; exact half-boundary handling not proved",
        "fallback": "most recent available previous month if release missing by next scheduled CPI release",
        "bls_provenance": "Retained web-tool extractions, not original HTTP bodies. Usage page modified February 13 2026; aggregation page modified February 11 2019 and cites 2015 methods. No current full replication contract or factor matrix was obtained.",
        "toy_only_counterexamples": scenarios,
        "conditional_bound": "If NSA=sum(v_i) and SA=sum(v_i/f_i), with common positive component contributions and f_i>0, SA/NSA is a convex combination of 1/f_i. Its range lies between min(1/f_i) and max(1/f_i). This is not yet a source-qualified BLS aggregation adapter.",
        "decision": "Reject a single-known-factor or same-scalar exact-hedge claim for this pair. No prices, fees or books inspected. Do not infer negative expected value or impossibility of a future bounded component mapping.",
        "next_evidence": "Source-bound first-release component aggregation, exact current factors, relevant denominator indexes, rounding intervals and every fallback vintage; prove an exhaustive joint outcome support before a separately frozen retained-price study. No refetch or sibling substitution on unchanged evidence.",
        "prices_inspected": False,
        "account_requests": 0,
        "orders": 0,
        "accepted_edge": False,
        "profitability_claim": False,
    }


def main():
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **build_review(),
    }
    result["result_sha256"] = sha(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    with (BASE / "review.json").open("xb") as stream:
        stream.write(json.dumps(result, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "toy_only_counterexamples": result["toy_only_counterexamples"],
            }
        )
    )


if __name__ == "__main__":
    main()
