"""Causal materialization of live AI shadow decisions against ML trades."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .ai_uplift import AIUpliftPolicy, assess_ai_uplift
from .live_ai_assist import (
    load_live_ai_entry_audit,
    validate_live_ai_entry_audit_records,
)
from .positions import BOT_OWNER, ClosedTrade, PositionsStore


LIVE_AI_UPLIFT_SCHEMA_VERSION = "live-ai-shadow-uplift-v1"
_DAY_MS = 86_400_000
_ACTION_STATUS = {
    "approve": "shadow_approve",
    "veto": "shadow_veto",
    "cooldown": "shadow_cooldown",
}


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _sha256(value: object, *, label: str) -> str:
    parsed = str(value or "").lower()
    if len(parsed) != 64 or any(
        character not in "0123456789abcdef" for character in parsed
    ):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return parsed


def _trade_metrics(
    trade_pnls: Sequence[float],
    daily_pnls: Sequence[float],
    *,
    initial_capital: float,
    liquidation_events: int,
    dataset_fingerprint: str,
    role: str,
) -> dict[str, object]:
    executed = tuple(float(value) for value in trade_pnls)
    realized_pnl = math.fsum(executed)
    gross_profit = math.fsum(value for value in executed if value > 0.0)
    gross_loss = abs(math.fsum(value for value in executed if value < 0.0))
    equity = initial_capital
    peak = initial_capital
    maximum_drawdown = 0.0
    daily_returns: list[float] = []
    for pnl in daily_pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        if peak > 0.0:
            maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
        daily_returns.append(float(pnl) / initial_capital)
    losses = [value for value in daily_returns if value < 0.0]
    downside_deviation = (
        math.sqrt(math.fsum(value * value for value in losses) / len(daily_returns))
        if daily_returns and losses
        else 0.0
    )
    downside_ratio = (
        math.fsum(daily_returns) / len(daily_returns) / downside_deviation * math.sqrt(365.0)
        if downside_deviation > 0.0
        else 0.0
    )
    maximum_loss_streak = 0
    current_loss_streak = 0
    for value in executed:
        if value < 0.0:
            current_loss_streak += 1
            maximum_loss_streak = max(maximum_loss_streak, current_loss_streak)
        else:
            current_loss_streak = 0
    metrics: dict[str, object] = {
        "realized_pnl": realized_pnl,
        "roi_pct": realized_pnl / initial_capital * 100.0,
        "max_drawdown": maximum_drawdown,
        "expectancy": realized_pnl / len(executed) if executed else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else 0.0,
        "closed_trades": len(executed),
        "win_rate": (
            sum(value > 0.0 for value in executed) / len(executed) if executed else 0.0
        ),
        "liquidation_events": max(0, int(liquidation_events)),
        "max_consecutive_losses": maximum_loss_streak,
        "downside_return_risk_ratio": downside_ratio,
        "dataset_fingerprint": dataset_fingerprint,
    }
    metrics["evidence_sha256"] = _canonical_sha256(
        {
            "schema_version": LIVE_AI_UPLIFT_SCHEMA_VERSION,
            "role": role,
            "metrics": metrics,
            "trade_pnls": list(executed),
            "daily_pnls": [float(value) for value in daily_pnls],
        }
    )
    return metrics


def assess_live_ai_shadow_uplift(
    trades: Sequence[ClosedTrade],
    audit_records: Sequence[Mapping[str, object]],
    *,
    initial_capital: float,
    model_name: str,
    model_parameters_b: float | None = None,
    policy: AIUpliftPolicy | None = None,
    minimum_causal_coverage: float = 0.90,
    maximum_review_age_seconds: int = 300,
) -> dict[str, object]:
    """Build same-trade daily returns using only pre-entry AI decisions."""

    capital = _finite(initial_capital, label="initial_capital")
    coverage_floor = _finite(
        minimum_causal_coverage,
        label="minimum_causal_coverage",
    )
    maximum_age_ms = int(maximum_review_age_seconds) * 1_000
    if (
        capital <= 0.0
        or not 0.90 <= coverage_floor <= 1.0
        or not 1_000 <= maximum_age_ms <= 300_000
        or not str(model_name).strip()
    ):
        raise ValueError("live AI uplift policy is invalid")
    verified_records = validate_live_ai_entry_audit_records(audit_records)
    records_by_case: dict[str, Mapping[str, object]] = {}
    for record in verified_records:
        raw_case = record.get("case") if isinstance(record, Mapping) else None
        if not isinstance(raw_case, Mapping):
            raise ValueError("live AI audit record has no case")
        case_id = _sha256(raw_case.get("case_id"), label="AI case_id")
        if case_id in records_by_case:
            raise ValueError("live AI audit repeats a case")
        records_by_case[case_id] = record
    candidates = [
        trade
        for trade in trades
        if trade.owner == BOT_OWNER and trade.ai_review_mode == "shadow_only"
    ]
    candidates.sort(key=lambda trade: (trade.opened_at_ms, trade.closed_at_ms, trade.id))
    rejection_counts: Counter[str] = Counter()
    used_case_ids: set[str] = set()
    eligible: list[dict[str, object]] = []
    model_digests: set[str] = set()
    terminal_fingerprints: set[str] = set()
    for trade in candidates:
        case_id = str(trade.ai_review_case_id or "").lower()
        record = records_by_case.get(case_id)
        if record is None:
            rejection_counts["audit_record_missing"] += 1
            continue
        if case_id in used_case_ids:
            rejection_counts["audit_case_reused"] += 1
            continue
        raw_case = record.get("case")
        raw_decision = record.get("decision")
        if not isinstance(raw_case, Mapping) or not isinstance(raw_decision, Mapping):
            rejection_counts["audit_evidence_missing"] += 1
            continue
        try:
            completed_at_ms = int(record["completed_at_ms"])
            observed_at_ms = int(raw_case["observed_at_ms"])
            maximum_risk_multiplier = float(raw_case["maximum_risk_multiplier"])
            decision_risk_multiplier = float(raw_decision["risk_multiplier"])
        except (KeyError, TypeError, ValueError, OverflowError):
            rejection_counts["audit_timing_or_risk_invalid"] += 1
            continue
        if (
            observed_at_ms > completed_at_ms
            or completed_at_ms > int(trade.opened_at_ms)
            or int(trade.opened_at_ms) - completed_at_ms > maximum_age_ms
        ):
            rejection_counts["review_not_causally_available"] += 1
            continue
        if (
            str(raw_case.get("symbol")) != trade.symbol
            or str(raw_case.get("market_type")) != trade.market_type
            or str(raw_case.get("proposed_side")) != trade.side
        ):
            rejection_counts["case_trade_identity_mismatch"] += 1
            continue
        action = str(raw_decision.get("action"))
        if (
            raw_decision.get("valid") is not True
            or action not in _ACTION_STATUS
            or trade.ai_review_status != _ACTION_STATUS[action]
            or maximum_risk_multiplier <= 0.0
            or not 0.0 <= decision_risk_multiplier <= maximum_risk_multiplier
        ):
            rejection_counts["decision_contract_mismatch"] += 1
            continue
        model_digest = str(raw_case.get("model_digest") or "").lower()
        observed_model_digest = str(
            raw_decision.get("observed_model_digest") or ""
        ).lower()
        try:
            model_digest = _sha256(model_digest, label="AI model digest")
            terminal_fingerprint = _sha256(
                raw_case.get("terminal_model_fingerprint"),
                label="terminal model fingerprint",
            )
        except ValueError:
            rejection_counts["model_identity_invalid"] += 1
            continue
        if (
            observed_model_digest != model_digest
            or raw_decision.get("model_residency_status") != "gpu_resident"
        ):
            rejection_counts["approved_gpu_model_not_observed"] += 1
            continue
        scale = (
            decision_risk_multiplier / maximum_risk_multiplier
            if action == "approve"
            else 0.0
        )
        baseline_pnl = _finite(trade.realized_pnl, label="trade realized_pnl")
        eligible.append(
            {
                "trade": trade,
                "case_id": case_id,
                "record_sha256": _sha256(
                    record.get("record_sha256"),
                    label="AI audit record",
                ),
                "action": action,
                "scale": scale,
                "baseline_pnl": baseline_pnl,
                "ai_pnl": baseline_pnl * scale,
            }
        )
        used_case_ids.add(case_id)
        model_digests.add(model_digest)
        terminal_fingerprints.add(terminal_fingerprint)
    causal_coverage = len(eligible) / len(candidates) if candidates else 0.0
    materialization_reasons: list[str] = []
    if not candidates:
        materialization_reasons.append("ai_shadow_trades_missing")
    if causal_coverage < coverage_floor:
        materialization_reasons.append(
            f"causal_ai_trade_coverage<{coverage_floor:.2f}"
        )
    if len(model_digests) != 1:
        materialization_reasons.append("ai_model_digest_not_unique")
    if len(terminal_fingerprints) != 1:
        materialization_reasons.append("terminal_model_fingerprint_not_unique")
    dataset_rows = [
        {
            "trade_id": item["trade"].id,
            "opened_at_ms": item["trade"].opened_at_ms,
            "closed_at_ms": item["trade"].closed_at_ms,
            "case_id": item["case_id"],
            "record_sha256": item["record_sha256"],
            "baseline_pnl": item["baseline_pnl"],
            "ai_pnl": item["ai_pnl"],
            "scale": item["scale"],
        }
        for item in eligible
    ]
    dataset_fingerprint = _canonical_sha256(
        {
            "schema_version": LIVE_AI_UPLIFT_SCHEMA_VERSION,
            "initial_capital": capital,
            "rows": dataset_rows,
        }
    )
    if eligible:
        first_day = min(int(item["trade"].opened_at_ms) // _DAY_MS for item in eligible)
        last_day = max(int(item["trade"].closed_at_ms) // _DAY_MS for item in eligible)
    else:
        first_day = 0
        last_day = -1
    baseline_by_day: dict[int, float] = {}
    ai_by_day: dict[int, float] = {}
    for item in eligible:
        day = int(item["trade"].closed_at_ms) // _DAY_MS
        baseline_by_day[day] = baseline_by_day.get(day, 0.0) + float(
            item["baseline_pnl"]
        )
        ai_by_day[day] = ai_by_day.get(day, 0.0) + float(item["ai_pnl"])
    days = list(range(first_day, last_day + 1))
    baseline_daily = [baseline_by_day.get(day, 0.0) for day in days]
    ai_daily = [ai_by_day.get(day, 0.0) for day in days]
    matched_periods = [
        {
            "scope": "bot_owned_live_ai_shadow_daily_return",
            "period_start_ms": day * _DAY_MS,
            "period_end_ms": (day + 1) * _DAY_MS,
            "baseline_return": baseline / capital,
            "ai_return": ai / capital,
        }
        for day, baseline, ai in zip(days, baseline_daily, ai_daily, strict=True)
    ]
    baseline_trade_pnls = [float(item["baseline_pnl"]) for item in eligible]
    ai_trade_pnls = [
        float(item["ai_pnl"]) for item in eligible if float(item["scale"]) > 0.0
    ]
    baseline_liquidations = sum(
        "liquidat" in str(item["trade"].reason).lower() for item in eligible
    )
    ai_liquidations = sum(
        "liquidat" in str(item["trade"].reason).lower() and float(item["scale"]) > 0.0
        for item in eligible
    )
    baseline_metrics = _trade_metrics(
        baseline_trade_pnls,
        baseline_daily,
        initial_capital=capital,
        liquidation_events=baseline_liquidations,
        dataset_fingerprint=dataset_fingerprint,
        role="baseline_ml",
    )
    ai_metrics = _trade_metrics(
        ai_trade_pnls,
        ai_daily,
        initial_capital=capital,
        liquidation_events=ai_liquidations,
        dataset_fingerprint=dataset_fingerprint,
        role="ai_shadow_counterfactual",
    )
    model_digest = next(iter(model_digests)) if len(model_digests) == 1 else ""
    uplift = assess_ai_uplift(
        baseline_metrics,
        ai_metrics,
        model_name=str(model_name),
        model_parameters_b=model_parameters_b,
        model_artifact_sha256=model_digest,
        matched_periods=matched_periods,
        policy=policy,
    )
    combined_reasons = tuple(
        dict.fromkeys([*materialization_reasons, *uplift.reasons])
    )
    accepted = uplift.accepted and not materialization_reasons
    report = {
        "schema_version": LIVE_AI_UPLIFT_SCHEMA_VERSION,
        "accepted": accepted,
        "advisory_only": not accepted,
        "model_name": str(model_name),
        "model_digest": model_digest,
        "terminal_model_fingerprint": (
            next(iter(terminal_fingerprints))
            if len(terminal_fingerprints) == 1
            else ""
        ),
        "candidate_trades": len(candidates),
        "causally_eligible_trades": len(eligible),
        "causal_coverage": causal_coverage,
        "minimum_causal_coverage": coverage_floor,
        "maximum_review_age_seconds": maximum_age_ms // 1_000,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "dataset_fingerprint": dataset_fingerprint,
        "matched_periods": matched_periods,
        "uplift": uplift.asdict(),
        "reasons": list(combined_reasons),
        "trading_authority": False,
        "profitability_claim": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def assess_live_ai_shadow_uplift_paths(
    *,
    positions_root: Path,
    audit_path: Path,
    initial_capital: float,
    model_name: str,
    model_parameters_b: float | None = None,
    policy: AIUpliftPolicy | None = None,
) -> dict[str, object]:
    store = PositionsStore(Path(positions_root))
    return assess_live_ai_shadow_uplift(
        store.load_ledger(),
        load_live_ai_entry_audit(Path(audit_path)),
        initial_capital=initial_capital,
        model_name=model_name,
        model_parameters_b=model_parameters_b,
        policy=policy,
    )


__all__ = [
    "LIVE_AI_UPLIFT_SCHEMA_VERSION",
    "assess_live_ai_shadow_uplift",
    "assess_live_ai_shadow_uplift_paths",
]
