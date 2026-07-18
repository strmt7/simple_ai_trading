"""Freeze the complete one-use Polymarket Round 13 confirmation contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_action_pipeline import (
    polymarket_action_pipeline_implementation_sha256,
)
from simple_ai_trading.polymarket_round12_reference import (
    load_round12_reference_from_round11_artifact,
    polymarket_round12_primary_policy,
    polymarket_round12_reference_implementation_sha256,
)
from simple_ai_trading.polymarket_round13_capture import (
    POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS,
)
from simple_ai_trading.polymarket_round13 import (
    POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE,
    POLYMARKET_ROUND13_CONTRACT_SCHEMA_VERSION,
    POLYMARKET_ROUND13_CTF_EXCHANGE_V2_COMMIT,
    POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD,
    POLYMARKET_ROUND13_V2_CLIENT_COMMIT,
    polymarket_round13_evaluation_gates,
    polymarket_round13_program_implementation_sha256,
    polymarket_round13_scenarios,
    polymarket_round13_upstream_order_semantics,
)
from simple_ai_trading.storage import write_json_atomic


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_hashed_json(path: Path, hash_field: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read frozen predecessor: {path.name}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"frozen predecessor is not an object: {path.name}")
    payload = dict(decoded)
    claimed = payload.pop(hash_field, None)
    if claimed != _sha256(payload):
        raise ValueError(f"frozen predecessor hash differs: {path.name}")
    return decoded


def freeze(
    predecessor_path: Path,
    round12_invalidation_path: Path,
    output_path: Path,
) -> str:
    if output_path.exists():
        raise ValueError("Round 13 contract already exists and cannot be overwritten")
    predecessor = _read_hashed_json(predecessor_path, "artifact_sha256")
    invalidation = _read_hashed_json(round12_invalidation_path, "artifact_sha256")
    authority = invalidation.get("authority")
    outcome_access = invalidation.get("outcome_access_evidence")
    persisted_counts = invalidation.get("persisted_table_counts")
    recorder_report = invalidation.get("recorder_report")
    if (
        invalidation.get("round") != 12
        or invalidation.get("status") != "invalidated_before_outcome_access"
        or not isinstance(authority, Mapping)
        or authority.get("model_selection") is not False
        or authority.get("profitability_claim") is not False
        or authority.get("paper_trading") is not False
        or authority.get("live_trading") is not False
        or not isinstance(outcome_access, Mapping)
        or outcome_access.get("operator_invalidated_before_outcome_access") is not True
        or outcome_access.get("performance_labels_opened") is not False
        or outcome_access.get("persisted_resolution_evidence_rows") != 0
        or not isinstance(persisted_counts, Mapping)
        or persisted_counts.get("polymarket_resolution_evidence") != 0
        or not isinstance(recorder_report, Mapping)
        or recorder_report.get("run_id") != invalidation.get("run_id")
        or recorder_report.get("status") != "failed"
        or not _is_sha256(recorder_report.get("report_sha256"))
    ):
        raise ValueError("Round 12 invalidation cannot authorize Round 13 selection")
    model = load_round12_reference_from_round11_artifact(predecessor_path)
    policy = polymarket_round12_primary_policy()
    gates = polymarket_round13_evaluation_gates()
    contract_without_hash: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND13_CONTRACT_SCHEMA_VERSION,
        "round": 13,
        "status": "frozen_before_fresh_capture",
        "scope": {
            "venue": "Polymarket public five-minute crypto Up/Down markets",
            "assets": ["BTC", "ETH", "SOL"],
            "strategy": (
                "single dual-minimum-guarded quote-denominated FOK buy held to "
                "official resolution"
            ),
            "profile": "conservative research confirmation",
            "synthetic_data": False,
            "leverage_applicable": False,
            "forced_activity": False,
        },
        "predecessor_evidence": {
            "artifact_filename": predecessor_path.name,
            "artifact_sha256": predecessor["artifact_sha256"],
            "artifact_file_sha256": _file_sha256(predecessor_path),
            "round11_result": "development_rejected",
            "round12_invalidation_filename": round12_invalidation_path.name,
            "round12_invalidation_artifact_sha256": invalidation["artifact_sha256"],
            "round12_invalidation_file_sha256": _file_sha256(round12_invalidation_path),
            "round12_run_id": invalidation["run_id"],
            "round12_run_report_sha256": recorder_report["report_sha256"],
            "round12_result": "invalidated_before_outcome_access",
            "selection_authority_from_round12": False,
        },
        "hypothesis": {
            "primary": (
                "The already frozen monotone calibration of the contemporaneous "
                "market prior has positive after-cost utility on one untouched "
                "prospective capture under the exact primary and stress executions."
            ),
            "falsification": (
                "Any data, activity, state, utility, control, drawdown, capital, "
                "or stress gate failure rejects this exact candidate."
            ),
            "complexity_rule": (
                "No additional feature, fit, threshold, AI model, or post-label "
                "selection is permitted in this confirmation."
            ),
        },
        "model_contract": {
            "model_sha256": model.model_sha256,
            "formula": (
                "sigmoid(frozen_intercept + frozen_slope * "
                "(logit(clip(market_prior_up,1e-9,1-1e-9)) + frozen_shift))"
            ),
            "coefficients": {
                "calibration_intercept": model.calibration_intercept,
                "calibration_slope": model.calibration_slope,
                "residual_intercept": model.residual_intercept,
                "probability_clip": model.probability_clip,
            },
            "orientation": "Down probability is exactly one minus Up probability.",
            "cpu_reference": "python_scalar_binary64",
            "external_feature_coefficients_applied": False,
            "training_or_refit_performed": False,
            "online_adaptation": False,
            "accelerator_authority": False,
        },
        "primary_policy": {
            "policy_sha256": policy.policy_sha256,
            "profile": policy.profile,
            "minimum_direction_probability": policy.minimum_direction_probability,
            "minimum_expected_edge_quote": policy.minimum_expected_edge_quote,
            "numeric_decision_guard": POLYMARKET_ROUND13_NUMERIC_DECISION_GUARD,
            "minimum_remaining_seconds": policy.minimum_remaining_seconds,
            "submission_latency_ms": policy.submission_latency_ms,
            "maximum_execution_observation_delay_ms": (
                policy.maximum_execution_observation_delay_ms
            ),
            "retry_interval_ms": policy.retry_interval_ms,
            "forced_activity": False,
            "selection": (
                "At each causal 250 ms decision, choose the unique outcome with "
                "probability at least 0.80 and quote-denominated, fee-inclusive "
                "full-depth expected edge above 0.02 quote plus a 1e-12 numerical "
                "guard; derive a cent-exact market BUY amount and tick-aligned FOK "
                "worst-price limit whose quote amount and signed minimum shares each "
                "meet the recorded numeric minimum and preserve that edge under a "
                "conservative fee and rounding bound; otherwise abstain."
            ),
            "lifecycle": (
                "Retry only a definitely observed FOK non-fill after 1000 ms. The "
                "first simulated fill or unknown post-submit state locks the condition."
            ),
            "position": (
                "Never hedge, average down, martingale, or open the opposite outcome."
            ),
        },
        "capture_contract": {
            "one_shot": True,
            "fresh_prospective_duration_seconds": (
                POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS
            ),
            "decision_cadence_ms": 250,
            "minimum_synchronized_event_groups": (
                gates.minimum_synchronized_event_groups
            ),
            "minimum_resolved_markets_per_asset": (
                gates.minimum_resolved_markets_per_asset
            ),
            "capture_sources": [
                "Polymarket public Gamma market discovery",
                "Polymarket public CLOB protocol-version endpoint",
                "Polymarket public CLOB metadata, fee, and REST book endpoints",
                "Polymarket public CLOB market WebSocket",
                "Polymarket RTDS Chainlink stream",
                "Binance spot bookTicker WebSocket",
                "Binance spot aggregate-trade WebSocket",
            ],
            "post_claim_resolution_sources": [
                "Polymarket CLOB and Gamma terminal resolution endpoints",
            ],
            "clocks": ["local monotonic receipt", "UTC wall receipt", "source time"],
            "storage": (
                "One bounded DuckDB evidence database with compressed immutable raw "
                "chunks and compact hash-linked derived tables."
            ),
            "start_gate": (
                "A clean committed Git tree containing this contract, every critical "
                "implementation module, and every Round 13 test must be attested "
                "before the recorder run row and before the first source message."
            ),
            "failure_policy": (
                "Interruption, recorder error, integrity error, CLOB protocol drift, "
                "source identity drift, missing manifest, or undersized scope fails "
                "the capture. After the one-use claim opens, network interruption or "
                "pending official resolution preserves that exact claim for an "
                "identity-bound resume."
            ),
        },
        "label_free_action_contract": {
            "independent_unit": "one synchronized BTC/ETH/SOL five-minute event start",
            "derived_feature_labels": (
                "official_up must be null and resolution_event_id empty in every "
                "Round 13 feature row"
            ),
            "calibration_snapshot": (
                "One latest same-segment admissible snapshot per condition while at "
                "least 120 seconds remain, persisted before outcomes are requested."
            ),
            "treatment": "frozen calibrated market-prior policy",
            "control": "same execution with the raw normalized market prior",
            "unknown_state": (
                "A missing same-segment post-submit observation or post-submit tick "
                "drift is unknown, locks the condition, and is never a no-fill."
            ),
            "simulated_no_fill": (
                "Only an observed same-segment full-depth shortfall within the frozen "
                "FOK worst-price limit is a simulated no-fill with zero modeled "
                "inventory and zero utility."
            ),
            "full_depth": True,
            "displayed_depth_is_not_queue_position": True,
            "slippage_protection": "per-attempt tick-aligned FOK worst-price limit",
        },
        "execution_scenarios": {
            "scenario_schema_version": "polymarket-round13-label-free-scenario-v1",
            "scenarios": [item.asdict() for item in polymarket_round13_scenarios()],
            "primary_observation_window_ms": 500,
            "fee_source": "recorded per-market fee schedule",
            "tick_source": "recorded per-market tick-size events",
            "depth_source": "recorded full displayed ask ladder",
        },
        "order_semantics": polymarket_round13_upstream_order_semantics(),
        "risk_contract": {
            "confirmation_capital_quote": format(
                POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE, "f"
            ),
            "reinvestment": False,
            "leverage": False,
            "maximum_group_exposure_quote": format(
                POLYMARKET_ROUND13_CONFIRMATION_CAPITAL_QUOTE, "f"
            ),
            "unknown_after_submit_charge": (
                "maximum fee-inclusive loss permitted by the frozen FOK limit"
            ),
            "drawdown_denominator": "explicit confirmation capital only",
            "no_inferred_capital": True,
        },
        "executable_evaluation_gates": gates.asdict(),
        "evaluation_contract": {
            "claim_order": (
                "Persist and commit one-use contract/run/pipeline/scenario identities "
                "before calling any official resolution loader."
            ),
            "resolution_endpoint_gate": (
                "The internal dual-source finalizer refuses a preregistered Round 13 "
                "run unless its exact open one-use claim revalidates."
            ),
            "label": (
                "Known simulated-fill utility uses only the signed minimum share "
                "quantity guaranteed by the FOK limit, never modeled price-improvement "
                "shares: payout minus exact quote amount and modeled fee when correct, "
                "or negative fee-inclusive cost when wrong. Simulated no-fill is "
                "zero; unknown is charged the frozen maximum loss."
            ),
            "bootstrap": {
                "unit": "chronological synchronized event group",
                "method": "circular moving-block bootstrap of the mean",
                "prng": "SplitMix64",
                "sample_serialization": "little-endian IEEE-754 binary64",
                "samples": gates.bootstrap_samples,
                "seed": gates.bootstrap_seed,
                "block_groups": gates.bootstrap_block_groups,
                "confidence_interval": "linear 2.5% and 97.5% quantiles",
            },
            "proper_scores": [
                "pooled and per-asset log loss",
                "pooled and per-asset Brier score",
                "logistic calibration intercept and slope",
                "decile reliability bins",
                "exact differences from raw market prior",
            ],
            "conjunctive_gates": [
                "minimum independent groups and resolved markets",
                "both outcomes represented in every asset",
                "minimum simulated fills overall and per asset",
                "minimum non-tied treatment/control conditions",
                "zero selected unknown-after-submit conditions",
                "positive total, mean-condition, median-filled-condition, and per-asset utility",
                "strictly positive lower bootstrap mean-group utility",
                "treatment total and lower paired bootstrap strictly beat raw prior",
                "drawdown within both frozen limits",
                "maximum event-group entry exposure within explicit allocation",
                "every primary and stress scenario passes",
            ],
            "one_use_failure": (
                "Deterministic integrity, identity, or scoring failure after claim "
                "opening marks the contract failed. Process interruption, network "
                "failure, and pending official settlement leave the exact claim open "
                "and resumable without changing code, data, policy, or scenarios. A "
                "complete report is idempotently readable."
            ),
        },
        "publication_contract": {
            "latest_only": True,
            "tables": [
                "round13-scenario-summary.csv",
                "round13-equity.csv",
                "round13-per-asset.csv",
                "round13-reliability.csv",
                "round13-treatment-control.csv",
                "round13-admission-states.csv",
                "optimization-progress.csv",
            ],
            "charts": [
                "round13-equity-drawdown.svg",
                "round13-per-asset-utility.svg",
                "round13-reliability.svg",
                "round13-treatment-control.svg",
                "round13-stress.svg",
                "round13-admission.svg",
                "optimization-progress.svg",
            ],
            "source_rule": (
                "Every chart is deterministic SVG generated from exact integrity-hashed "
                "CSV rows derived from the immutable evaluation report."
            ),
            "staged_crash_recoverable_replacement": True,
            "single_directory_atomic_exchange": False,
            "manual_chart_edits_permitted": False,
        },
        "implementation": {
            "reference_implementation_sha256": (
                polymarket_round12_reference_implementation_sha256()
            ),
            "action_pipeline_implementation_sha256": (
                polymarket_action_pipeline_implementation_sha256()
            ),
            "round13_program_implementation_sha256": (
                polymarket_round13_program_implementation_sha256()
            ),
            "source_hash_normalization": "utf8_lf_normalized",
            "reference_numeric_semantics": "finite IEEE-754 binary64",
            "economic_decimal_semantics": (
                "local decimal precision 50 with ROUND_HALF_EVEN; caller context ignored"
            ),
            "financial_gate_numeric_semantics": (
                "local decimal precision 50 with ROUND_HALF_EVEN through gate "
                "decisions; finite IEEE-754 binary64 only for the frozen bootstrap "
                "and report serialization"
            ),
            "runtime_paths_in_model_semantics": False,
            "environment_variables_in_model_semantics": False,
            "operating_system_branches_in_model_semantics": False,
        },
        "portability_contract": {
            "reference_decision": (
                "Identical finite inputs must pass the scalar conformance tests on a "
                "supported host. Economic arithmetic uses a local precision-50 decimal "
                "context independent of caller state. A 1e-12 guard around probability, "
                "edge, and tie gates prevents platform-level floating-point noise from "
                "granting action."
            ),
            "capability_discovery": (
                "Database, network, clock, filesystem, memory, and optional accelerator "
                "capabilities are probed at runtime; no OS implies a GPU vendor."
            ),
            "accelerators": (
                "Accelerators are optional performance providers and never alter the "
                "reference decision or admission semantics."
            ),
            "installation": (
                "Inputs and outputs use explicit paths or package resources. Absolute "
                "database paths may remain as inert recorder provenance, but drive "
                "letters, usernames, locale, shell, and working directory never alter "
                "model decisions, gates, or execution economics. Relocation preserves "
                "the original evidence identity instead of recomputing it."
            ),
            "resource_bounds": (
                "Queues, memory, database threads, batches, retries, and deadlines are "
                "bounded configuration; resource exhaustion fails closed."
            ),
            "support_claim": (
                "Portability is claimed only for published capability contracts and "
                "tested environments; missing capabilities fail before capture/action."
            ),
        },
        "freshness": {
            "capture_started": False,
            "confirmation_consumed": False,
            "outcome_labels_consulted": False,
            "thresholds_changed_after_freeze": False,
            "one_shot": True,
        },
        "authority": {
            "paper_trading": False,
            "live_trading": False,
            "profitability_claim": False,
            "roi_claim": False,
            "drawdown_claim": False,
            "ai_edge_claim": False,
            "authenticated_order_lifecycle_proven": False,
            "owned_balance_reconciliation_proven": False,
            "settlement_overhead_measured": False,
        },
        "unavailable_metrics": [
            "annualized ROI",
            "Sharpe and deflated Sharpe",
            "probability of backtest overfitting",
            "live capacity",
            "authenticated lifecycle success",
            "settlement-overhead-adjusted return",
        ],
        "research_basis": [
            "https://docs.polymarket.com/market-data/websocket/market-channel",
            "https://docs.polymarket.com/market-data/websocket/overview",
            "https://docs.polymarket.com/api-reference/rate-limits",
            "https://docs.polymarket.com/api-reference/authentication",
            "https://docs.polymarket.com/trading/orders/create",
            "https://docs.polymarket.com/trading/orderbook",
            "https://docs.polymarket.com/trading/fees",
            "https://github.com/Polymarket/py-clob-client-v2/tree/"
            f"{POLYMARKET_ROUND13_V2_CLIENT_COMMIT}",
            "https://github.com/Polymarket/ctf-exchange-v2/tree/"
            f"{POLYMARKET_ROUND13_CTF_EXCHANGE_V2_COMMIT}",
            "https://doi.org/10.1093/biomet/82.3.561",
            "https://doi.org/10.1029/96WR00928",
        ],
    }
    contract = {
        **contract_without_hash,
        "contract_sha256": _sha256(contract_without_hash),
    }
    write_json_atomic(output_path, contract, indent=2, sort_keys=False)
    return str(contract["contract_sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor", required=True, type=Path)
    parser.add_argument("--round12-invalidation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        freeze(
            args.predecessor.resolve(),
            args.round12_invalidation.resolve(),
            args.output.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
