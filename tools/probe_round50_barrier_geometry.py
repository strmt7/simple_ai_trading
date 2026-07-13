"""Probe training-role geometry for predeclared path-bounded barrier targets."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.action_hurdle_tcn_model import (  # noqa: E402
    build_action_hurdle_temporal_dataset,
)
from simple_ai_trading.barrier_payoff_data import (  # noqa: E402
    EVENT_NAMES,
    SIDE_NAMES,
    BarrierSpecification,
    build_barrier_payoff_dataset,
)
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    SYMBOLS,
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


SCHEMA = "round-050-barrier-target-geometry-probe-v1"
EXPECTED_PREDECESSOR_DATASET_SHA256 = (
    "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
)
ANCHOR_ID = "h60-v1-r2"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _event(stage: str, **details: object) -> None:
    print(
        json.dumps(
            {"stage": stage, **details},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _progress(stage: str, payload: Mapping[str, object]) -> None:
    _event(stage, **dict(payload))


def _specifications() -> tuple[tuple[str, BarrierSpecification], ...]:
    rows: list[tuple[str, BarrierSpecification]] = []
    for horizon in (30, 60):
        for volatility_multiple in (0.75, 1.0):
            for reward_risk in (1.5, 2.0):
                identifier = (
                    f"h{horizon}-v{str(volatility_multiple).replace('.', '')}"
                    f"-r{str(reward_risk).replace('.', '')}"
                )
                if horizon == 60 and volatility_multiple == 1.0 and reward_risk == 2.0:
                    identifier = ANCHOR_ID
                rows.append(
                    (
                        identifier,
                        BarrierSpecification(
                            horizon_minutes=horizon,
                            stop_volatility_multiple=volatility_multiple,
                            take_profit_to_stop_ratio=reward_risk,
                            minimum_stop_bps=24.0,
                            maximum_stop_bps=80.0,
                            round_trip_execution_charge_bps=12.0,
                        ),
                    )
                )
    return tuple(rows)


def _geometry_rows(
    candidate_id: str,
    dataset: object,
) -> list[dict[str, object]]:
    training = dataset.role_masks["training"]
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        for side_index, side_name in enumerate(SIDE_NAMES):
            events = dataset.event_code[training, symbol_index, side_index]
            duration = dataset.event_minute[training, symbol_index, side_index]
            payoff = dataset.net_payoff_bps[training, symbol_index, side_index].astype(
                np.float64
            )
            price_return = dataset.price_return_bps[
                training, symbol_index, side_index
            ].astype(np.float64)
            funding = dataset.funding_cash_flow_bps[
                training, symbol_index, side_index
            ].astype(np.float64)
            gap = dataset.gap_through_slippage_bps[
                training, symbol_index, side_index
            ].astype(np.float64)
            ambiguous = dataset.ambiguous_stop_first[training, symbol_index, side_index]
            stop = dataset.stop_bps[training, symbol_index].astype(np.float64)
            take = dataset.take_profit_bps[training, symbol_index].astype(np.float64)
            event_rates = {
                name: float(np.mean(events == event_index))
                for event_index, name in enumerate(EVENT_NAMES)
            }
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "symbol": symbol,
                    "side": side_name,
                    "rows": int(payoff.size),
                    "event_rates": event_rates,
                    "mean_holding_minutes": float(np.mean(duration)),
                    "median_holding_minutes": float(np.median(duration)),
                    "stop_bps_p10": float(np.quantile(stop, 0.10)),
                    "stop_bps_median": float(np.median(stop)),
                    "stop_bps_p90": float(np.quantile(stop, 0.90)),
                    "take_profit_bps_median": float(np.median(take)),
                    "net_payoff_mean_bps": float(np.mean(payoff)),
                    "net_payoff_standard_deviation_bps": float(np.std(payoff)),
                    "net_payoff_p01_bps": float(np.quantile(payoff, 0.01)),
                    "net_payoff_p05_bps": float(np.quantile(payoff, 0.05)),
                    "net_payoff_median_bps": float(np.median(payoff)),
                    "net_payoff_p95_bps": float(np.quantile(payoff, 0.95)),
                    "profit_rate": float(np.mean(payoff > 0.0)),
                    "price_return_mean_bps": float(np.mean(price_return)),
                    "funding_mean_bps": float(np.mean(funding)),
                    "ambiguous_stop_first_fraction": float(np.mean(ambiguous)),
                    "gap_through_fraction": float(np.mean(gap > 0.0)),
                    "gap_through_mean_when_positive_bps": (
                        float(np.mean(gap[gap > 0.0])) if np.any(gap > 0.0) else 0.0
                    ),
                    "gap_through_p99_bps": float(np.quantile(gap, 0.99)),
                }
            )
    return rows


def _structural_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    reasons: list[str] = []
    minimum_stop_event_rate = min(
        float(row["event_rates"]["stop_loss"]) for row in rows
    )
    minimum_take_event_rate = min(
        float(row["event_rates"]["take_profit"]) for row in rows
    )
    maximum_timeout_event_rate = max(
        float(row["event_rates"]["timeout"]) for row in rows
    )
    maximum_ambiguity = max(float(row["ambiguous_stop_first_fraction"]) for row in rows)
    if minimum_stop_event_rate < 0.05:
        reasons.append("a_symbol_side_has_fewer_than_five_percent_stop_events")
    if minimum_take_event_rate < 0.03:
        reasons.append("a_symbol_side_has_fewer_than_three_percent_take_events")
    if maximum_timeout_event_rate > 0.90:
        reasons.append("a_symbol_side_has_more_than_ninety_percent_timeouts")
    if maximum_ambiguity > 0.02:
        reasons.append("minute_bar_stop_take_ambiguity_exceeds_two_percent")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "minimum_stop_event_rate": minimum_stop_event_rate,
        "minimum_take_profit_event_rate": minimum_take_event_rate,
        "maximum_timeout_event_rate": maximum_timeout_event_rate,
        "maximum_ambiguous_stop_first_fraction": maximum_ambiguity,
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    panel, price_source = load_verified_minute_panel(
        arguments.database.resolve(), progress=_progress
    )
    premium, funding, derivatives_source = load_derivatives_state(
        arguments.database.resolve(),
        panel,
        price_source,
        source_certificate_path=arguments.source_certificate.resolve(),
        progress=_progress,
    )
    source = build_derivatives_hurdle_dataset(
        panel,
        premium,
        funding,
        derivatives_source,
        progress=_progress,
    )
    temporal = build_action_hurdle_temporal_dataset(source)
    if temporal.dataset_sha256 != EXPECTED_PREDECESSOR_DATASET_SHA256:
        raise ValueError("Round 50 barrier probe reconstructed a different corpus")
    _event(
        "predecessor_validated",
        dataset_sha256=temporal.dataset_sha256,
        timestamps=temporal.timestamps,
    )

    candidates: list[dict[str, object]] = []
    geometry: list[dict[str, object]] = []
    for candidate_id, specification in _specifications():
        _event("barrier_candidate_started", candidate_id=candidate_id)
        dataset = build_barrier_payoff_dataset(
            panel, funding, source, temporal, specification
        )
        candidate_rows = _geometry_rows(candidate_id, dataset)
        gate = _structural_gate(candidate_rows)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "specification": specification.asdict(),
                "dataset_sha256": dataset.dataset_sha256,
                "training_geometry_gate": gate,
                "is_predeclared_anchor": candidate_id == ANCHOR_ID,
            }
        )
        geometry.extend(candidate_rows)
        _event(
            "barrier_candidate_complete",
            candidate_id=candidate_id,
            gate_passed=gate["passed"],
            minimum_take_profit_event_rate=gate["minimum_take_profit_event_rate"],
            maximum_timeout_event_rate=gate["maximum_timeout_event_rate"],
        )

    anchor = next(item for item in candidates if item["candidate_id"] == ANCHOR_ID)
    output: dict[str, object] = {
        "schema_version": SCHEMA,
        "round": 50,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "probe_sha256": "PENDING",
        "purpose": (
            "Training-role target geometry only. The matrix is a structural sensitivity "
            "check, not model selection or economic optimization."
        ),
        "predecessor_dataset_sha256": temporal.dataset_sha256,
        "predeclared_anchor_candidate_id": ANCHOR_ID,
        "anchor_permitted_for_model_experiment": bool(
            anchor["training_geometry_gate"]["passed"]
        ),
        "candidate_selection_from_return_performance_permitted": False,
        "calibration_or_viability_metrics_reported": False,
        "selection_contaminated": True,
        "profitability_claim": False,
        "trading_authority": False,
        "candidates": candidates,
        "training_geometry": geometry,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "persistent_feature_or_target_copy_created": False,
        },
    }
    canonical = dict(output)
    canonical.pop("probe_sha256")
    output["probe_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(arguments.output.resolve(), output, indent=2, sort_keys=True)
    _event(
        "barrier_probe_complete",
        output=str(arguments.output.resolve()),
        probe_sha256=output["probe_sha256"],
        anchor_permitted=output["anchor_permitted_for_model_experiment"],
        elapsed_seconds=output["runtime"]["elapsed_seconds"],
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
