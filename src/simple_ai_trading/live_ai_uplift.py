"""Causal materialization of live AI shadow decisions against ML trades."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence

from .ai_uplift import AIUpliftPolicy, assess_ai_uplift
from .live_ai_assist import (
    load_live_ai_entry_audit,
    validate_live_ai_entry_audit_records,
)
from .positions import BOT_OWNER, ClosedTrade, PositionsStore


LIVE_AI_UPLIFT_SCHEMA_VERSION = "live-ai-shadow-uplift-v2"
_DAY_MS = 86_400_000
_ACTION_STATUS = {
    "approve": "shadow_approve",
    "veto": "shadow_veto",
    "cooldown": "shadow_cooldown",
}
_SECOND_MS = 1_000


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


def load_one_second_trade_paths(
    market_db: Path,
    trades: Sequence[ClosedTrade],
) -> dict[str, tuple[dict[str, object], ...]]:
    """Read exact one-second high/low paths without mutating the market store."""

    database = Path(market_db).resolve()
    if not database.is_file():
        raise ValueError(f"one-second market database is missing: {database}")
    uri = f"{database.as_uri()}?mode=ro"
    paths: dict[str, tuple[dict[str, object], ...]] = {}
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        for trade in trades:
            start_ms = int(trade.opened_at_ms) // _SECOND_MS * _SECOND_MS
            end_ms = int(trade.closed_at_ms) // _SECOND_MS * _SECOND_MS
            rows = connection.execute(
                """
                SELECT open_time, high, low, source
                FROM candles
                WHERE symbol = ? AND market_type = ? AND interval = '1s'
                  AND open_time >= ? AND open_time <= ?
                ORDER BY open_time
                """,
                (trade.symbol.upper(), trade.market_type, start_ms, end_ms),
            ).fetchall()
            paths[trade.id] = tuple(
                {
                    "timestamp_ms": int(row["open_time"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "source": str(row["source"]),
                }
                for row in rows
            )
    except sqlite3.Error as exc:
        raise ValueError(f"one-second path query failed: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    return paths


def _maximum_drawdown_from_events(
    events: Mapping[int, float],
    *,
    initial_capital: float,
) -> float:
    equity = initial_capital
    peak = initial_capital
    maximum_drawdown = 0.0
    for timestamp_ms in sorted(events):
        equity += float(events[timestamp_ms])
        peak = max(peak, equity)
        if peak > 0.0:
            maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
    return maximum_drawdown


def _intratrade_path_risk(
    eligible: Sequence[Mapping[str, object]],
    paths: Mapping[str, Sequence[Mapping[str, object]]] | None,
    *,
    initial_capital: float,
) -> tuple[float, float, dict[str, object], tuple[str, ...]]:
    baseline_events: dict[int, float] = {}
    ai_events: dict[int, float] = {}
    path_rows: list[dict[str, object]] = []
    sources: Counter[str] = Counter()
    reasons: list[str] = []
    complete_trade_count = 0
    for item in eligible:
        trade = item["trade"]
        if not isinstance(trade, ClosedTrade):
            raise ValueError("eligible AI uplift trade is invalid")
        raw_path = paths.get(trade.id) if paths is not None else None
        if not raw_path:
            reasons.append("intratrade_path_missing")
            continue
        start_ms = int(trade.opened_at_ms) // _SECOND_MS * _SECOND_MS
        end_ms = int(trade.closed_at_ms) // _SECOND_MS * _SECOND_MS
        expected_count = (end_ms - start_ms) // _SECOND_MS + 1
        normalized: list[dict[str, object]] = []
        path_sources: Counter[str] = Counter()
        path_valid = len(raw_path) == expected_count
        for index, row in enumerate(raw_path):
            try:
                timestamp_ms = int(row["timestamp_ms"])
                high = _finite(row["high"], label="intratrade high")
                low = _finite(row["low"], label="intratrade low")
                source = str(row["source"])
            except (KeyError, TypeError, ValueError, OverflowError):
                path_valid = False
                break
            if (
                timestamp_ms != start_ms + index * _SECOND_MS
                or low <= 0.0
                or high < low
                or not source
            ):
                path_valid = False
                break
            normalized.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "high": high,
                    "low": low,
                    "source": source,
                }
            )
            path_sources[source] += 1
        if not path_valid:
            reasons.append("intratrade_path_incomplete_or_invalid")
            continue
        qty = _finite(trade.qty, label="trade quantity")
        entry_price = _finite(trade.entry_price, label="trade entry price")
        exit_price = _finite(trade.exit_price, label="trade exit price")
        fees = _finite(trade.fees, label="trade fees")
        realized_pnl = _finite(trade.realized_pnl, label="trade realized_pnl")
        if (
            trade.side not in {"LONG", "SHORT"}
            or qty <= 0.0
            or entry_price <= 0.0
            or exit_price <= 0.0
            or fees < 0.0
        ):
            reasons.append("intratrade_trade_accounting_invalid")
            continue
        direction = 1.0 if trade.side == "LONG" else -1.0
        expected_realized = direction * (exit_price - entry_price) * qty - fees
        tolerance = max(1e-8, abs(expected_realized) * 1e-9)
        if abs(expected_realized - realized_pnl) > tolerance:
            reasons.append("intratrade_trade_accounting_mismatch")
            continue
        scale = _finite(item["scale"], label="AI trade scale")
        previous_baseline = 0.0
        previous_ai = 0.0
        for row in normalized:
            adverse_price = float(row["low"] if trade.side == "LONG" else row["high"])
            baseline_value = direction * (adverse_price - entry_price) * qty - fees
            ai_value = baseline_value * scale
            timestamp_ms = int(row["timestamp_ms"])
            baseline_events[timestamp_ms] = baseline_events.get(timestamp_ms, 0.0) + (
                baseline_value - previous_baseline
            )
            ai_events[timestamp_ms] = ai_events.get(timestamp_ms, 0.0) + (
                ai_value - previous_ai
            )
            previous_baseline = baseline_value
            previous_ai = ai_value
        settlement_ms = end_ms + _SECOND_MS
        baseline_events[settlement_ms] = baseline_events.get(settlement_ms, 0.0) + (
            realized_pnl - previous_baseline
        )
        ai_realized = realized_pnl * scale
        ai_events[settlement_ms] = ai_events.get(settlement_ms, 0.0) + (
            ai_realized - previous_ai
        )
        path_rows.append(
            {
                "trade_id": trade.id,
                "row_count": len(normalized),
                "first_timestamp_ms": start_ms,
                "last_timestamp_ms": end_ms,
                "rows_sha256": _canonical_sha256(normalized),
            }
        )
        sources.update(path_sources)
        complete_trade_count += 1
    unique_reasons = tuple(dict.fromkeys(reasons))
    verified = bool(eligible) and complete_trade_count == len(eligible) and not unique_reasons
    evidence = {
        "verified": verified,
        "interval": "1s",
        "conservative_mark": "low_for_long_high_for_short",
        "fee_timing": "full_recorded_fees_at_entry",
        "eligible_trade_count": len(eligible),
        "complete_trade_count": complete_trade_count,
        "row_count": sum(int(row["row_count"]) for row in path_rows),
        "source_counts": dict(sorted(sources.items())),
        "trade_paths": path_rows,
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    return (
        _maximum_drawdown_from_events(
            baseline_events,
            initial_capital=initial_capital,
        ),
        _maximum_drawdown_from_events(
            ai_events,
            initial_capital=initial_capital,
        ),
        evidence,
        unique_reasons,
    )


def _trade_metrics(
    trade_pnls: Sequence[float],
    daily_pnls: Sequence[float],
    *,
    initial_capital: float,
    liquidation_events: int,
    dataset_fingerprint: str,
    role: str,
    path_maximum_drawdown: float | None = None,
    path_evidence_sha256: str = "",
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
    if path_maximum_drawdown is not None:
        maximum_drawdown = _finite(
            path_maximum_drawdown,
            label="path maximum drawdown",
        )
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
        "intratrade_path_evidence_sha256": path_evidence_sha256,
    }
    metrics["evidence_sha256"] = _canonical_sha256(
        {
            "schema_version": LIVE_AI_UPLIFT_SCHEMA_VERSION,
            "role": role,
            "metrics": metrics,
            "trade_pnls": list(executed),
            "daily_pnls": [float(value) for value in daily_pnls],
            "intratrade_path_evidence_sha256": path_evidence_sha256,
        }
    )
    return metrics


def assess_live_ai_shadow_uplift(
    trades: Sequence[ClosedTrade],
    audit_records: Sequence[Mapping[str, object]],
    *,
    initial_capital: float,
    model_name: str,
    intratrade_paths: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ]
    | None = None,
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
    candidate_case_ids = {
        str(trade.ai_review_case_id or "").lower()
        for trade in candidates
    }
    audited_case_ids = set(records_by_case)
    matched_proposal_case_ids = audited_case_ids.intersection(candidate_case_ids)
    unmatched_proposal_case_ids = audited_case_ids.difference(candidate_case_ids)
    proposal_outcome_coverage = (
        len(matched_proposal_case_ids) / len(audited_case_ids)
        if audited_case_ids
        else 0.0
    )
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
    if unmatched_proposal_case_ids:
        rejection_counts["counterfactual_outcome_missing"] = len(
            unmatched_proposal_case_ids
        )
        materialization_reasons.append(
            "ai_shadow_proposal_outcomes_incomplete"
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
    (
        baseline_path_drawdown,
        ai_path_drawdown,
        intratrade_path_evidence,
        intratrade_path_reasons,
    ) = _intratrade_path_risk(
        eligible,
        intratrade_paths,
        initial_capital=capital,
    )
    materialization_reasons.extend(intratrade_path_reasons)
    if eligible and not intratrade_path_evidence["verified"]:
        materialization_reasons.append("intratrade_path_risk_not_verified")
    path_evidence_sha256 = str(intratrade_path_evidence["evidence_sha256"])
    baseline_metrics = _trade_metrics(
        baseline_trade_pnls,
        baseline_daily,
        initial_capital=capital,
        liquidation_events=baseline_liquidations,
        dataset_fingerprint=dataset_fingerprint,
        role="baseline_ml",
        path_maximum_drawdown=baseline_path_drawdown,
        path_evidence_sha256=path_evidence_sha256,
    )
    ai_metrics = _trade_metrics(
        ai_trade_pnls,
        ai_daily,
        initial_capital=capital,
        liquidation_events=ai_liquidations,
        dataset_fingerprint=dataset_fingerprint,
        role="ai_shadow_counterfactual",
        path_maximum_drawdown=ai_path_drawdown,
        path_evidence_sha256=path_evidence_sha256,
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
        "audited_proposals": len(audited_case_ids),
        "matched_proposal_outcomes": len(matched_proposal_case_ids),
        "unmatched_proposal_outcomes": len(unmatched_proposal_case_ids),
        "proposal_outcome_coverage": proposal_outcome_coverage,
        "required_proposal_outcome_coverage": 1.0,
        "unmatched_proposal_case_ids_sha256": _canonical_sha256(
            sorted(unmatched_proposal_case_ids)
        ),
        "maximum_review_age_seconds": maximum_age_ms // 1_000,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "dataset_fingerprint": dataset_fingerprint,
        "intratrade_path_evidence": intratrade_path_evidence,
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
    market_db: Path,
    initial_capital: float,
    model_name: str,
    model_parameters_b: float | None = None,
    policy: AIUpliftPolicy | None = None,
) -> dict[str, object]:
    store = PositionsStore(Path(positions_root))
    trades = store.load_ledger()
    candidates = tuple(
        trade
        for trade in trades
        if trade.owner == BOT_OWNER and trade.ai_review_mode == "shadow_only"
    )
    intratrade_paths = load_one_second_trade_paths(Path(market_db), candidates)
    return assess_live_ai_shadow_uplift(
        trades,
        load_live_ai_entry_audit(Path(audit_path)),
        initial_capital=initial_capital,
        model_name=model_name,
        intratrade_paths=intratrade_paths,
        model_parameters_b=model_parameters_b,
        policy=policy,
    )


__all__ = [
    "LIVE_AI_UPLIFT_SCHEMA_VERSION",
    "assess_live_ai_shadow_uplift",
    "assess_live_ai_shadow_uplift_paths",
    "load_one_second_trade_paths",
]
