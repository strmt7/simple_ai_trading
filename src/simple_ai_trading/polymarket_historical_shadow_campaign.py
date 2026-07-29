"""Prospectively frozen evaluation of a bounded Polymarket shadow run."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .polymarket_historical_shadow_settlement import (
    settle_shadow_opportunity,
    validate_shadow_official_resolution,
)


SHADOW_HOUR_POLICY_SCHEMA_VERSION = "polymarket-btc-shadow-hour-policy-v1"
SHADOW_HOUR_EVALUATION_SCHEMA_VERSION = (
    "polymarket-btc-shadow-hour-evaluation-v1"
)
_MAXIMUM_POLICY_BYTES = 64 * 1024
_MAXIMUM_LOG_BYTES = 8 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Polymarket shadow evidence contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket shadow evidence contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def _source_commit(value: str) -> str:
    commit = str(value or "").strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("Polymarket shadow campaign source commit is invalid")
    return commit


def _strict_json(path: Path, *, maximum_bytes: int) -> object:
    if path.is_symlink():
        raise ValueError("Polymarket shadow evidence cannot be a symlink")
    raw = path.read_bytes()
    if not raw or len(raw) > maximum_bytes:
        raise ValueError("Polymarket shadow evidence size is invalid")
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Polymarket shadow evidence is not strict JSON") from exc


def load_shadow_hour_policy(path: str | Path) -> Mapping[str, object]:
    policy_path = Path(path)
    value = _strict_json(policy_path, maximum_bytes=_MAXIMUM_POLICY_BYTES)
    if not isinstance(value, Mapping):
        raise ValueError("Polymarket shadow policy is not an object")
    policy = dict(value)
    claimed = _sha(policy.pop("policy_sha256", ""), name="shadow policy")
    if _canonical_sha256(policy) != claimed:
        raise ValueError("Polymarket shadow policy integrity failed")
    scope = policy.get("scope")
    identity = policy.get("model_identity")
    selection = policy.get("selection_policy")
    settlement = policy.get("settlement_policy")
    reporting = policy.get("reporting_policy")
    if (
        policy.get("schema_version") != SHADOW_HOUR_POLICY_SCHEMA_VERSION
        or not isinstance(scope, Mapping)
        or scope.get("venue") != "polymarket"
        or scope.get("asset") != "BTC"
        or scope.get("market_variant") != "fiveminute"
        or scope.get("public_data_only") is not True
        or scope.get("credentials_used") is not False
        or scope.get("orders_submitted") is not False
        or not isinstance(identity, Mapping)
        or not isinstance(selection, Mapping)
        or selection.get("maximum_selected_entries_per_event") != 1
        or selection.get("selection")
        != "first_chronological_status_candidate"
        or selection.get("threshold_changes_after_freeze_allowed") is not False
        or selection.get("outcomes_or_future_books_consulted") is not False
        or not isinstance(settlement, Mapping)
        or settlement.get("gamma_and_clob_terminal_identity_required") is not True
        or settlement.get("gamma_and_clob_winner_agreement_required") is not True
        or settlement.get("displayed_depth_fill_is_counterfactual_only") is not True
        or settlement.get("real_fill_claim_allowed") is not False
        or not isinstance(reporting, Mapping)
        or int(reporting.get("minimum_events_for_profitability_claim", 0)) < 50
        or reporting.get("profitability_claim_for_this_run_allowed") is not False
        or policy.get("trading_authority") is not False
        or policy.get("profitability_claim") is not False
        or int(policy.get("created_at_ms", 0))
        >= int(policy.get("eligible_event_start_ms", 0))
        or int(policy.get("eligible_event_start_ms", 0))
        >= int(policy.get("expected_run_end_ms", 0))
    ):
        raise ValueError("Polymarket shadow policy semantics differ")
    for name in (
        "dataset_sha256",
        "pretest_artifact_sha256",
        "evaluation_artifact_sha256",
        "support_profile_sha256",
    ):
        _sha(identity.get(name), name=f"shadow policy {name}")
    return {**policy, "policy_sha256": claimed}


def load_shadow_hour_log(
    path: str | Path,
    *,
    policy: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], str]:
    log_path = Path(path)
    if log_path.is_symlink():
        raise ValueError("Polymarket shadow log cannot be a symlink")
    raw = log_path.read_bytes()
    if not raw or len(raw) > _MAXIMUM_LOG_BYTES:
        raise ValueError("Polymarket shadow log size is invalid")
    scope = policy["scope"]
    if not isinstance(scope, Mapping) or log_path.name != scope.get(
        "source_log_basename"
    ):
        raise ValueError("Polymarket shadow log identity differs")
    records: list[Mapping[str, object]] = []
    for line in raw.splitlines():
        if not line:
            continue
        try:
            record = json.loads(
                line.decode("utf-8", errors="strict"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Polymarket shadow log is not strict JSONL") from exc
        if not isinstance(record, Mapping):
            raise ValueError("Polymarket shadow log record is not an object")
        if (
            record.get("trading_authority") is not False
            or record.get("profitability_claim") is not False
            or not isinstance(record.get("payload"), Mapping)
        ):
            raise ValueError("Polymarket shadow log contains invalid authority")
        records.append(record)
    started = [item for item in records if item.get("event") == "started"]
    stopped = [item for item in records if item.get("event") == "stopped"]
    if len(started) != 1 or len(stopped) != 1:
        raise ValueError("Polymarket shadow log is not terminal")
    identity = policy["model_identity"]
    started_payload = started[0]["payload"]
    if not isinstance(identity, Mapping) or any(
        started_payload.get(log_name) != identity.get(policy_name)
        for log_name, policy_name in (
            ("candidate_id", "candidate_id"),
            ("dataset_sha256", "dataset_sha256"),
            ("pretest_artifact_sha256", "pretest_artifact_sha256"),
            ("evaluation_artifact_sha256", "evaluation_artifact_sha256"),
            ("support_profile_sha256", "support_profile_sha256"),
        )
    ):
        raise ValueError("Polymarket shadow log model identity differs")
    if int(stopped[0].get("observed_at_ms", 0)) < int(
        policy["expected_run_end_ms"]
    ):
        raise ValueError("Polymarket shadow log ended before its frozen horizon")
    return tuple(records), hashlib.sha256(raw).hexdigest()


def _prediction_metrics(
    predictions: Sequence[Mapping[str, object]],
    winner: str,
) -> Mapping[str, float | int]:
    target = 1.0 if winner == "Up" else 0.0
    losses = []
    briers = []
    correct = 0
    for prediction in predictions:
        probability = float(prediction["probability_up"])
        if not 0.0 < probability < 1.0 or not math.isfinite(probability):
            raise ValueError("Polymarket shadow prediction probability differs")
        winner_probability = probability if target == 1.0 else 1.0 - probability
        losses.append(-math.log(winner_probability))
        briers.append((probability - target) ** 2)
        correct += int((probability >= 0.5) == (target == 1.0))
    return {
        "rows": len(predictions),
        "mean_log_loss": float(sum(losses) / len(losses)),
        "mean_brier_score": float(sum(briers) / len(briers)),
        "directional_accuracy": float(correct / len(predictions)),
    }


def build_shadow_hour_evaluation(
    *,
    policy: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    source_log_sha256: str,
    terminal_sources: Mapping[
        int,
        tuple[Mapping[str, object], Mapping[str, object], int],
    ],
    source_commit: str,
) -> tuple[Mapping[str, object], str]:
    """Evaluate all eligible rows and one preselected entry per event."""

    eligible_start = int(policy["eligible_event_start_ms"])
    expected_end = int(policy["expected_run_end_ms"])
    predictions_by_event: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    opportunities_by_event: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    opportunity_error_count = 0
    stopped_health: Mapping[str, object] | None = None
    for record in records:
        event = record.get("event")
        payload = record["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("Polymarket shadow record payload differs")
        if event == "prediction" and payload.get("status") == "observed":
            event_start = int(payload["event_start_ms"])
            if eligible_start <= event_start < expected_end:
                predictions_by_event[event_start].append(payload)
        elif event == "opportunity":
            event_start = int(payload["event_start_ms"])
            if eligible_start <= event_start < expected_end:
                opportunities_by_event[event_start].append(payload)
        elif event == "opportunity_error":
            opportunity_error_count += 1
        elif event == "stopped":
            stopped_health = payload
    if not predictions_by_event or stopped_health is None:
        raise ValueError("Polymarket shadow evaluation population is empty")
    if set(predictions_by_event) != set(opportunities_by_event):
        raise ValueError("Polymarket shadow event identity coverage differs")
    if set(predictions_by_event) != set(terminal_sources):
        raise ValueError("Polymarket shadow terminal source coverage differs")

    event_reports = []
    selected_pnls: list[Decimal] = []
    selected_wins = 0
    total_prediction_rows = 0
    for event_start in sorted(predictions_by_event):
        predictions = sorted(
            predictions_by_event[event_start],
            key=lambda item: int(item["decision_time_ms"]),
        )
        opportunities = sorted(
            opportunities_by_event[event_start],
            key=lambda item: (
                int(item["decision_time_ms"]),
                int(item["observed_at_ms"]),
            ),
        )
        representative = opportunities[0]
        gamma, clob, resolution_observed_at = terminal_sources[event_start]
        resolution = validate_shadow_official_resolution(
            representative,
            gamma_market=gamma,
            clob_market=clob,
            resolution_observed_at_ms=resolution_observed_at,
        )
        winner = str(resolution["winner"])
        metrics = _prediction_metrics(predictions, winner)
        total_prediction_rows += int(metrics["rows"])
        candidates = [
            item for item in opportunities if item.get("status") == "candidate"
        ]
        selected_report: Mapping[str, object] | None = None
        if candidates:
            settlement, _ = settle_shadow_opportunity(
                candidates[0],
                gamma_market=gamma,
                clob_market=clob,
                resolution_observed_at_ms=resolution_observed_at,
                source_log_sha256=source_log_sha256,
                source_commit=source_commit,
            )
            counterfactual = settlement["displayed_depth_counterfactual"]
            prediction = settlement["prediction"]
            if not isinstance(counterfactual, Mapping) or not isinstance(
                prediction,
                Mapping,
            ):
                raise ValueError("Polymarket shadow settlement payload differs")
            pnl = Decimal(str(counterfactual["net_pnl_quote"]))
            selected_pnls.append(pnl)
            selected_wins += int(prediction["selected_outcome_won"] is True)
            selected_report = {
                "opportunity_artifact_sha256": candidates[0]["artifact_sha256"],
                "decision_time_ms": int(candidates[0]["decision_time_ms"]),
                "selected_outcome": prediction["selected_outcome"],
                "selected_outcome_won": prediction["selected_outcome_won"],
                "counterfactual_net_pnl_quote": format(pnl, "f"),
            }
        event_reports.append(
            {
                "event_start_ms": event_start,
                "event_end_ms": int(representative["event_end_ms"]),
                "condition_id": representative["condition_id"],
                "gamma_market_id": representative["gamma_market_id"],
                "winner": winner,
                "prediction_metrics": metrics,
                "observed_prediction_rows": len(predictions),
                "opportunity_count": len(opportunities),
                "candidate_opportunity_count": len(candidates),
                "selected_entry": selected_report,
                "gamma_payload_sha256": resolution["gamma_payload_sha256"],
                "clob_payload_sha256": resolution["clob_payload_sha256"],
            }
        )

    event_count = len(event_reports)
    condition_balanced = {
        name: float(
            sum(float(item["prediction_metrics"][name]) for item in event_reports)
            / event_count
        )
        for name in (
            "mean_log_loss",
            "mean_brier_score",
            "directional_accuracy",
        )
    }
    equity = Decimal("0")
    peak = Decimal("0")
    maximum_drawdown = Decimal("0")
    for pnl in selected_pnls:
        equity += pnl
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    gains = sum((item for item in selected_pnls if item > 0), Decimal("0"))
    losses = -sum((item for item in selected_pnls if item < 0), Decimal("0"))
    minimum_claim_events = int(
        policy["reporting_policy"]["minimum_events_for_profitability_claim"]
    )
    source_root = Path(__file__).parent
    artifact: dict[str, object] = {
        "schema_version": SHADOW_HOUR_EVALUATION_SCHEMA_VERSION,
        "policy_sha256": policy["policy_sha256"],
        "source_log_sha256": _sha(
            source_log_sha256,
            name="shadow source log",
        ),
        "source_commit": _source_commit(source_commit),
        "implementation_sha256": {
            "campaign_evaluation": _file_sha256(Path(__file__)),
            "settlement": _file_sha256(
                source_root
                / "polymarket_historical_shadow_settlement.py"
            ),
        },
        "population": {
            "eligible_event_start_ms": eligible_start,
            "expected_run_end_ms": expected_end,
            "resolved_events": event_count,
            "observed_prediction_rows": total_prediction_rows,
            "opportunity_errors": opportunity_error_count,
        },
        "condition_balanced_prediction_metrics": condition_balanced,
        "first_candidate_counterfactual": {
            "selected_events": len(selected_pnls),
            "winning_events": selected_wins,
            "win_rate": (
                float(selected_wins / len(selected_pnls))
                if selected_pnls
                else None
            ),
            "net_pnl_quote": format(sum(selected_pnls, Decimal("0")), "f"),
            "gross_profit_quote": format(gains, "f"),
            "gross_loss_quote": format(losses, "f"),
            "profit_factor": (
                float(gains / losses) if losses > 0 else None
            ),
            "maximum_drawdown_quote": format(maximum_drawdown, "f"),
            "real_orders_submitted": 0,
            "real_fills_observed": 0,
            "displayed_depth_fill_counterfactual_only": True,
        },
        "events": event_reports,
        "terminal_feed_health": dict(stopped_health),
        "claim_gate": {
            "minimum_events": minimum_claim_events,
            "minimum_events_passed": event_count >= minimum_claim_events,
            "profitability_claim": False,
            "trading_authority": False,
        },
        "trading_authority": False,
        "profitability_claim": False,
    }
    artifact_sha = _canonical_sha256(artifact)
    return {**artifact, "artifact_sha256": artifact_sha}, artifact_sha


__all__ = [
    "SHADOW_HOUR_EVALUATION_SCHEMA_VERSION",
    "SHADOW_HOUR_POLICY_SCHEMA_VERSION",
    "build_shadow_hour_evaluation",
    "load_shadow_hour_log",
    "load_shadow_hour_policy",
]
