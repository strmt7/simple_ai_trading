#!/usr/bin/env python3
"""Rescore retained Round 27 complete-set episodes under current taker tiers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Mapping

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round27_mechanics import (
    ROUND27_BOOK_SAMPLE_INTERVAL_MS,
    _canonical_sha256,
    _latency_benchmarks,
    _load_claim,
    _paired_quotes,
    _validate_stage0_lineage,
)
from simple_ai_trading.storage import write_json_atomic


SCHEMA_VERSION = "polymarket-round27-complete-set-taker-rebate-overlay-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _minimum(rows: list[dict[str, object]], key: str) -> str | None:
    values = [Decimal(str(row[key])) for row in rows if row[key] is not None]
    return None if not values else format(min(values), "f")


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "same_state_episode_count": sum(
            int(row["same_state_episode_count"]) for row in rows
        ),
        "venue_delay_survivor_count": sum(
            int(row["venue_delay_survivor_count"]) for row in rows
        ),
        "minimum_sequential_survivor_count": sum(
            int(row["minimum_sequential_survivor_count"]) for row in rows
        ),
        "up_then_down_survivor_count": sum(
            int(row["up_then_down_survivor_count"]) for row in rows
        ),
        "down_then_up_survivor_count": sum(
            int(row["down_then_up_survivor_count"]) for row in rows
        ),
        "lower_source_cost_first_survivor_count": sum(
            int(row["lower_source_cost_first_survivor_count"]) for row in rows
        ),
        "both_orders_survivor_count": sum(
            int(row["both_orders_survivor_count"]) for row in rows
        ),
        "best_same_state_cost_pusd_per_complete_set": _minimum(
            rows, "best_same_state_cost"
        ),
        "best_venue_delay_cost_pusd_per_complete_set": _minimum(
            rows, "best_venue_delay_cost"
        ),
        "best_minimum_sequential_cost_pusd_per_complete_set": _minimum(
            rows, "best_minimum_sequential_cost"
        ),
        "best_up_then_down_cost_pusd_per_complete_set": _minimum(
            rows, "best_up_then_down_cost"
        ),
        "best_down_then_up_cost_pusd_per_complete_set": _minimum(
            rows, "best_down_then_up_cost"
        ),
        "best_lower_source_cost_first_cost_pusd_per_complete_set": _minimum(
            rows, "best_lower_source_cost_first_cost"
        ),
        "best_worst_order_cost_pusd_per_complete_set": _minimum(
            rows, "best_worst_order_cost"
        ),
    }


def _load_inputs(root: Path) -> dict[str, dict[str, object]]:
    polymarket = root / "docs/model-research/polymarket"
    action_value = root / "docs/model-research/action-value"
    return {
        "audit": _load_claim(
            polymarket
            / "latest/round-027-stage0-condition-audit/condition-replay-audit.json",
            claim="audit_sha256",
            label="Round 27 condition audit",
        ),
        "preregistration": _load_claim(
            polymarket / "round-027-execution-hypothesis-preregistration-v3.json",
            claim="preregistration_sha256",
            label="Round 27 preregistration",
        ),
        "capture_contract": _load_claim(
            polymarket / "round-027-stage0-mechanics-capture-v1.json",
            claim="contract_sha256",
            label="Round 27 capture contract",
        ),
        "capture_result": _load_claim(
            polymarket / "round-027-stage0-mechanics-capture-result-v1-2026-08-15.json",
            claim="result_sha256",
            label="Round 27 capture result",
        ),
        "mechanics": _load_claim(
            polymarket
            / "latest/round-027-mechanics-diagnostic/mechanics-diagnostic.json",
            claim="mechanics_sha256",
            label="Round 27 mechanics diagnostic",
        ),
        "rebate": _load_claim(
            action_value / "polymarket-organic-taker-rebate-overlay-v1-2026-08-26.json",
            claim="result_sha256",
            label="Polymarket taker rebate overlay",
        ),
    }


def _tier_fractions(rebate: Mapping[str, object]) -> dict[str, Decimal]:
    evidence = rebate.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("rebate evidence is missing")
    terms = evidence.get("official_terms")
    if not isinstance(terms, Mapping):
        raise ValueError("rebate official terms are missing")
    fractions = terms.get("tier_rebate_fractions")
    if not isinstance(fractions, Mapping):
        raise ValueError("rebate tier fractions are missing")
    result = {"None": Decimal("0")}
    result.update({str(name): Decimal(str(value)) for name, value in fractions.items()})
    if any(value < 0 or value > 1 for value in result.values()):
        raise ValueError("rebate tier fraction is outside zero to one")
    return dict(sorted(result.items(), key=lambda item: (item[1], item[0])))


def _assert_baseline(
    mechanics: Mapping[str, object],
    baseline_rows: list[dict[str, object]],
) -> None:
    published = mechanics.get("complete_set_latency")
    if not isinstance(published, Mapping):
        raise ValueError("published complete-set latency is missing")
    expected_rows = published.get("segment_benchmarks")
    if not isinstance(expected_rows, list):
        raise ValueError("published segment benchmarks are missing")
    expected = {
        (str(row["condition_id"]), str(row["segment_id"])): row for row in expected_rows
    }
    actual = {
        (str(row["condition_id"]), str(row["segment_id"])): row for row in baseline_rows
    }
    if set(expected) != set(actual):
        raise ValueError("baseline segment identity differs from published mechanics")
    fields = (
        "same_state_episode_count",
        "venue_delay_survivor_count",
        "minimum_sequential_survivor_count",
        "best_same_state_cost",
        "best_venue_delay_cost",
        "best_minimum_sequential_cost",
    )
    for key in expected:
        if any(expected[key].get(field) != actual[key].get(field) for field in fields):
            raise ValueError(f"baseline metrics differ for {key[0]} {key[1]}")


def run(
    root: Path,
    *,
    database: Path,
    output: Path,
    progress_path: Path | None,
) -> dict[str, object]:
    inputs = _load_inputs(root)
    lineage = _validate_stage0_lineage(
        audit=inputs["audit"],
        preregistration=inputs["preregistration"],
        capture_contract=inputs["capture_contract"],
        capture_result=inputs["capture_result"],
    )
    audit = inputs["audit"]
    conditions = [
        item
        for item in audit.get("conditions", [])
        if isinstance(item, dict) and item.get("eligible") is True
    ]
    if not conditions:
        raise ValueError("Round 27 overlay has no eligible conditions")
    tiers = _tier_fractions(inputs["rebate"])
    rows_by_tier: dict[str, list[dict[str, object]]] = {tier: [] for tier in tiers}

    with PolymarketEvidenceStore(
        database,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        for index, item in enumerate(conditions, start=1):
            condition_id = str(item["condition_id"])
            intervals = {
                (condition_id, str(segment["segment_id"])): (
                    int(segment["interval_start_ms"]),
                    int(segment["interval_end_ms"]),
                )
                for segment in item.get("segments", [])
                if isinstance(segment, dict) and segment.get("eligible") is True
            }
            replay = PolymarketEvidenceReplay.load(
                store,
                run_id=str(audit["run_id"]),
                allow_segmented_gaps=True,
                include_resolutions=False,
                book_sample_interval_ms=ROUND27_BOOK_SAMPLE_INTERVAL_MS,
                condition_ids=(condition_id,),
                maximum_received_wall_ms_by_condition={
                    condition_id: int(item["end_ms"]) - 1
                },
                materialized_minimum_depth_levels=1,
                cap_materialized_depth_to_minimum_order_size=True,
            )
            quotes = _paired_quotes(replay, intervals=intervals)
            grouped: dict[tuple[str, str], list[object]] = {}
            for quote in quotes:
                grouped.setdefault((quote.condition_id, quote.segment_id), []).append(
                    quote
                )
            for tier, fraction in tiers.items():
                for key in sorted(intervals):
                    rows_by_tier[tier].append(
                        {
                            "condition_id": key[0],
                            "segment_id": key[1],
                            **_latency_benchmarks(
                                grouped.get(key, ()),
                                taker_rebate_fraction=fraction,
                                include_ordering_details=True,
                            ),
                        }
                    )
            store.recycle_analytical_connections()
            if progress_path is not None:
                write_json_atomic(
                    progress_path,
                    {
                        "completed_condition_count": index,
                        "condition_count": len(conditions),
                        "condition_id": condition_id,
                    },
                    indent=2,
                    sort_keys=True,
                )
            print(
                json.dumps(
                    {
                        "completed_condition_count": index,
                        "condition_count": len(conditions),
                        "condition_id": condition_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    _assert_baseline(inputs["mechanics"], rows_by_tier["None"])
    scenarios = []
    for tier, fraction in tiers.items():
        rows = rows_by_tier[tier]
        aggregate = _aggregate(rows)
        qualifying = [
            row
            for row in rows
            if int(row["lower_source_cost_first_survivor_count"]) > 0
            or int(row["both_orders_survivor_count"]) > 0
        ]
        scenarios.append(
            {
                "tier": tier,
                "rebate_fraction": format(fraction, "f"),
                **aggregate,
                "qualifying_segments": qualifying,
            }
        )
    causal = [
        row for row in scenarios if row["lower_source_cost_first_survivor_count"] > 0
    ]
    lowest = causal[0]["tier"] if causal else None
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "retained_target_free_complete_set_latency_rescore_under_current_taker_rebate_tiers",
        "predecessors": {
            "round27_mechanics_sha256": inputs["mechanics"]["mechanics_sha256"],
            "round27_capture_result_sha256": inputs["capture_result"]["result_sha256"],
            "taker_rebate_result_sha256": inputs["rebate"]["result_sha256"],
        },
        "lineage": lineage,
        "method": {
            "historical_capture_started_at_ms": inputs["capture_result"][
                "capture_report"
            ]["started_at_ms"],
            "historical_capture_ended_at_ms": inputs["capture_result"][
                "capture_report"
            ]["ended_at_ms"],
            "quantity_shares": "5",
            "rebate_application": "subtract exact tier fraction times each retained rounded taker-fee component; gross prices depth and timing are unchanged",
            "episode_selection": "recomputed independently at every tier over all retained paired quote states; not limited to the six no-rebate episodes",
            "latency": "same recorded venue delay and optimistic two-delay sequential floor as the predecessor; network and order-response latency remain excluded",
            "causal_ordering_gate": "at the episode source state choose the lower rebated-cost leg first, then use the first-delay cost for that leg and the second-delay cost for its complement; ties choose Up first",
            "robust_ordering_diagnostic": "both_orders_survivor_count requires both Up-then-Down and Down-then-Up to remain below one",
            "account_tier_assumed": False,
        },
        "scenarios": scenarios,
        "adjudication": {
            "accepted_edge": False,
            "candidate_edge": bool(causal),
            "deployment_ready": False,
            "lowest_historical_causal_ordering_survivor_tier": lowest,
            "market_direction_forecast_required": False,
            "profitability_claim": False,
            "reason": (
                "At least one historical retained episode survives the source-time lower-cost-leg-first sequential rule at a documented tier, but current recurrence, exact account tier, network and order-response latency, atomic fill, merge cost, and after-cost profit remain unproved."
                if causal
                else "The optimistic ex-post minimum sequential cost can fall below one, but no documented tier rescues a retained episode under the source-time lower-cost-leg-first rule."
            ),
            "trading_authority": False,
        },
        "authority": {
            "credentials_used": False,
            "execution_connected": False,
            "new_market_requests": 0,
            "orders_or_mutations_submitted": 0,
            "public_retained_data_only": True,
        },
        "limitations": [
            "The retained cohort is a five-hour BTC five-minute-market capture from 2026-08-15, not current recurrence evidence.",
            "Tier fractions are account-hypothetical until a direct wallet shows the completed effective tier.",
            "The predecessor sequential floor excludes network and order-response latency, merge cost, gas or relayer cost, adverse selection, and fill risk.",
            "The predecessor minimum sequential cost chooses the cheaper of both leg orderings after both later books are known. It is retained only as an optimistic lower bound and is not the candidate gate.",
            "No public wallet tier is used as evidence for the users account.",
        ],
    }
    body["result_sha256"] = _canonical_sha256(body)
    write_json_atomic(output, body, indent=2, sort_keys=True)
    return body


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.repository.resolve()
    result = run(
        root,
        database=_resolve(root, arguments.database),
        output=_resolve(root, arguments.output),
        progress_path=(
            None if arguments.progress is None else _resolve(root, arguments.progress)
        ),
    )
    print(json.dumps(result["adjudication"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
