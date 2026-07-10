"""No-order public-feed shadow evaluation for deployment-refit candidates."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

from .microstructure_data import MicrostructureCaptureResult
from .microstructure_data import (
    BINANCE_FUTURES_MARKET_STREAM_URL,
    BINANCE_FUTURES_PUBLIC_STREAM_URL,
    MICROSTRUCTURE_SCHEMA_VERSION,
)
from .microstructure_live import (
    LiveMicrostructurePrediction,
    LiveTopOfBook,
    MicrostructureFeedIntegrityError,
    StreamingMicrostructureCoordinator,
)
from .microstructure_model import (
    MICROSTRUCTURE_SHADOW_EVIDENCE_VERSION,
    MicrostructureActionScorer,
    MicrostructureModelArtifact,
    ShadowValidationEvidence,
    TradingMetrics,
    _candidate_payload_sha256,
    _risk_utility,
    _trading_metrics,
    _validated_shadow_binding,
    load_microstructure_action_scorer,
    load_microstructure_model_artifact,
)
from .storage import write_json_atomic


SHADOW_EVIDENCE_VERSION = MICROSTRUCTURE_SHADOW_EVIDENCE_VERSION
_PROVIDER = "binance_public_usdm_websocket"


@dataclass(frozen=True)
class ShadowConfig:
    minimum_duration_seconds: float = 21_600.0
    minimum_decisions: int = 100
    minimum_virtual_trades: int = 20
    settlement_delay_ms: int = 100
    maximum_capture_age_seconds: int = 900

    def validated(self) -> "ShadowConfig":
        if self.minimum_duration_seconds < 60.0:
            raise ValueError("shadow minimum duration must be at least 60 seconds")
        if self.minimum_decisions < 1 or self.minimum_virtual_trades < 1:
            raise ValueError("shadow decision and trade minimums must be positive")
        if self.settlement_delay_ms < 0 or self.settlement_delay_ms > 5_000:
            raise ValueError("shadow settlement delay must lie in [0, 5000] ms")
        if self.maximum_capture_age_seconds < 60:
            raise ValueError("shadow maximum capture age must be at least 60 seconds")
        return self

    def asdict(self) -> dict[str, object]:
        return asdict(self)


PROMOTION_SHADOW_CONFIG = ShadowConfig()


@dataclass(frozen=True)
class VirtualShadowTrade:
    side: str
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    predicted_net_bps: float
    realized_net_bps: float
    exit_reason: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowReplayResult:
    started_at_ms: int
    completed_at_ms: int
    decisions: int
    actionable_decisions: int
    rejected_while_open: int
    execution_liquidity_rejections: int
    expired_entries: int
    pending_entries_at_end: int
    end_censored_signals: int
    trades: tuple[VirtualShadowTrade, ...]
    metrics: TradingMetrics
    feed_sequence_gaps: int
    invalid_events: int
    late_event_resets: int
    feature_gap_resets: int
    deadline_misses: int
    inference_failures: int
    forced_closes: int
    orders_submitted: int

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.completed_at_ms - self.started_at_ms) / 1_000.0)


@dataclass(frozen=True)
class ShadowEvaluationReport:
    version: str
    generated_at_ms: int
    status: str
    trading_authority: bool
    reasons: tuple[str, ...]
    candidate_sha256: str
    deployment_model_sha256: str
    symbol: str
    provider: str
    config: Mapping[str, object]
    capture: Mapping[str, object]
    replay: Mapping[str, object]
    metrics: Mapping[str, object]
    trades_path: str
    trades_sha256: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass
class _OpenVirtualPosition:
    side: str
    entry_time_ms: int
    entry_price: float
    deadline_ms: int
    predicted_net_bps: float
    stop_price: float
    take_price: float


@dataclass
class _PendingVirtualEntry:
    side: str
    execute_at_ms: int
    predicted_net_bps: float


class _VirtualShadowLedger:
    def __init__(
        self,
        scorer: MicrostructureActionScorer,
        *,
        entry_cutoff_ms: int | None,
    ) -> None:
        self.scorer = scorer
        self.entry_cutoff_ms = entry_cutoff_ms
        self.pending_entry: _PendingVirtualEntry | None = None
        self.open_position: _OpenVirtualPosition | None = None
        self.last_quote: LiveTopOfBook | None = None
        self._last_quote_key: tuple[int, int, int] | None = None
        self.decisions = 0
        self.actionable_decisions = 0
        self.rejected_while_open = 0
        self.execution_liquidity_rejections = 0
        self.expired_entries = 0
        self.pending_entries_at_end = 0
        self.end_censored_signals = 0
        self.forced_closes = 0
        self.trades: list[VirtualShadowTrade] = []

    def observe_quote(self, quote: LiveTopOfBook) -> None:
        if str(quote.symbol).upper() != self.scorer.symbol:
            raise ValueError("shadow quote symbol does not match the scorer")
        quote_values = (
            quote.bid,
            quote.ask,
            quote.bid_qty,
            quote.ask_qty,
        )
        if (
            not all(math.isfinite(value) and value > 0.0 for value in quote_values)
            or quote.bid >= quote.ask
        ):
            raise ValueError("shadow quote is invalid or crossed")
        quote_key = (
            int(quote.event_time_ms),
            int(quote.transaction_time_ms),
            int(quote.update_id),
        )
        if self._last_quote_key is not None and quote_key <= self._last_quote_key:
            return
        self._last_quote_key = quote_key
        self.last_quote = quote
        pending = self.pending_entry
        if pending is not None and quote.event_time_ms >= pending.execute_at_ms:
            if quote.event_time_ms - pending.execute_at_ms > self.scorer.max_quote_age_ms:
                self.expired_entries += 1
                self.pending_entry = None
            else:
                if pending.side == "LONG":
                    participation = (
                        self.scorer.reference_order_notional_quote / quote.ask
                    ) / quote.ask_qty
                    entry_price = quote.ask
                else:
                    participation = (
                        self.scorer.reference_order_notional_quote / quote.bid
                    ) / quote.bid_qty
                    entry_price = quote.bid
                if participation > self.scorer.max_l1_participation:
                    self.execution_liquidity_rejections += 1
                    self.pending_entry = None
                else:
                    if pending.side == "LONG":
                        stop_price = entry_price * (
                            1.0 - self.scorer.stop_loss_bps / 10_000.0
                        )
                        take_price = entry_price * (
                            1.0 + self.scorer.take_profit_bps / 10_000.0
                        )
                    else:
                        stop_price = entry_price * (
                            1.0 + self.scorer.stop_loss_bps / 10_000.0
                        )
                        take_price = entry_price * (
                            1.0 - self.scorer.take_profit_bps / 10_000.0
                        )
                    self.open_position = _OpenVirtualPosition(
                        side=pending.side,
                        entry_time_ms=quote.event_time_ms,
                        entry_price=float(entry_price),
                        deadline_ms=(
                            quote.event_time_ms
                            + int(self.scorer.horizon_seconds) * 1_000
                        ),
                        predicted_net_bps=pending.predicted_net_bps,
                        stop_price=float(stop_price),
                        take_price=float(take_price),
                    )
                    self.pending_entry = None
        position = self.open_position
        if position is None or quote.event_time_ms <= position.entry_time_ms:
            return
        reason = ""
        if position.side == "LONG":
            if quote.bid <= position.stop_price:
                reason = "stop"
            elif quote.bid >= position.take_price:
                reason = "take"
            elif quote.event_time_ms >= position.deadline_ms:
                reason = "horizon"
            exit_price = quote.bid
        else:
            if quote.ask >= position.stop_price:
                reason = "stop"
            elif quote.ask <= position.take_price:
                reason = "take"
            elif quote.event_time_ms >= position.deadline_ms:
                reason = "horizon"
            exit_price = quote.ask
        if reason:
            self._close(position, quote.event_time_ms, exit_price, reason)

    def observe_prediction(self, value: LiveMicrostructurePrediction) -> None:
        self.decisions += 1
        side = str(value.prediction.side).upper()
        if side not in {"LONG", "SHORT"}:
            return
        if (
            self.entry_cutoff_ms is not None
            and value.signal_deadline_ms > self.entry_cutoff_ms
        ):
            self.end_censored_signals += 1
            return
        self.actionable_decisions += 1
        if self.open_position is not None or self.pending_entry is not None:
            self.rejected_while_open += 1
            return
        if value.signal_deadline_ms <= value.observed_exchange_time_ms:
            self.expired_entries += 1
            return
        predicted = (
            float(value.prediction.long_expected_net_bps)
            if side == "LONG"
            else float(value.prediction.short_expected_net_bps)
        )
        self.pending_entry = _PendingVirtualEntry(
            side=side,
            predicted_net_bps=predicted,
            execute_at_ms=int(value.signal_deadline_ms),
        )

    def finish_capture(self) -> None:
        if self.pending_entry is not None:
            self.pending_entries_at_end += 1
            self.pending_entry = None
        if self.open_position is None or self.last_quote is None:
            return
        position = self.open_position
        exit_price = self.last_quote.bid if position.side == "LONG" else self.last_quote.ask
        self.forced_closes += 1
        self._close(
            position,
            max(self.last_quote.event_time_ms, position.entry_time_ms + 1),
            exit_price,
            "capture_end_forced",
        )

    def _close(
        self,
        position: _OpenVirtualPosition,
        exit_time_ms: int,
        exit_price: float,
        reason: str,
    ) -> None:
        execution_price = float(exit_price)
        if reason in {"stop", "take"}:
            slippage_fraction = self.scorer.trigger_execution_slippage_bps / 10_000.0
            execution_price *= (
                1.0 - slippage_fraction
                if position.side == "LONG"
                else 1.0 + slippage_fraction
            )
            if execution_price <= 0.0:
                raise ValueError("shadow trigger slippage produced a non-positive price")
        if position.side == "LONG":
            gross_bps = (execution_price / position.entry_price - 1.0) * 10_000.0
        else:
            gross_bps = (
                (position.entry_price - execution_price) / position.entry_price * 10_000.0
            )
        realized = gross_bps - 2.0 * self.scorer.taker_fee_bps
        self.trades.append(
            VirtualShadowTrade(
                side=position.side,
                entry_time_ms=position.entry_time_ms,
                exit_time_ms=int(exit_time_ms),
                entry_price=position.entry_price,
                exit_price=execution_price,
                predicted_net_bps=position.predicted_net_bps,
                realized_net_bps=float(realized),
                exit_reason=reason,
            )
        )
        self.open_position = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_trades(path: Path, trades: Sequence[VirtualShadowTrade]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    fields = tuple(VirtualShadowTrade.__dataclass_fields__)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.asdict())
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(temporary)
        os.replace(temporary, path)
        return digest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _event_payload(raw: object) -> Mapping[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    data = raw.get("data", raw)
    if not isinstance(data, Mapping):
        return None
    event_type = str(data.get("e") or "")
    if event_type not in {"bookTicker", "trade", "aggTrade"}:
        return None
    return data


def replay_shadow_capture(
    scorer: MicrostructureActionScorer,
    synchronized_raw_path: str | Path,
    *,
    settlement_delay_ms: int = 100,
    clock_offset_ms: float,
    entry_cutoff_ms: int | None,
) -> ShadowReplayResult:
    """Replay a just-captured public feed through the exact no-order live engine."""

    coordinator = StreamingMicrostructureCoordinator(
        scorer,
        settlement_delay_ms=int(settlement_delay_ms),
    )
    offset_ms = float(clock_offset_ms)
    if not math.isfinite(offset_ms) or abs(offset_ms) > 60_000.0:
        raise ValueError("shadow clock offset is invalid")
    cutoff_ms = None if entry_cutoff_ms is None else int(entry_cutoff_ms)
    if cutoff_ms is not None and cutoff_ms <= 0:
        raise ValueError("shadow entry cutoff must be positive")
    ledger = _VirtualShadowLedger(scorer, entry_cutoff_ms=cutoff_ms)
    first_event_ms = 0
    last_event_ms = 0
    last_received_at_ns = 0
    last_observed_exchange_ms = 0
    invalid_events = 0
    feed_sequence_gaps = 0
    target = Path(synchronized_raw_path)
    try:
        source = gzip.open(target, "rt", encoding="utf-8")
    except OSError as exc:
        raise ValueError("shadow capture cannot be opened") from exc
    with source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                received_at, raw_json = stripped.split(" ", 1)
                received_at_ns = int(received_at)
                if received_at_ns <= 0 or received_at_ns < last_received_at_ns:
                    raise ValueError("capture receive timestamps are not monotonic")
                raw = json.loads(raw_json)
                payload = _event_payload(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                invalid_events += 1
                raise ValueError(
                    f"shadow capture line {line_number} is invalid"
                ) from exc
            if payload is None:
                continue
            last_received_at_ns = received_at_ns
            observed_exchange_ms = int(received_at_ns / 1_000_000.0 + offset_ms)
            last_observed_exchange_ms = max(
                last_observed_exchange_ms,
                observed_exchange_ms,
            )
            event_time_ms = int(payload.get("E", 0) or 0)
            if event_time_ms <= 0:
                invalid_events += 1
                continue
            if first_event_ms <= 0:
                first_event_ms = event_time_ms
            last_event_ms = max(last_event_ms, event_time_ms)
            try:
                coordinator.ingest(payload)
                if str(payload.get("e")) == "bookTicker":
                    current_quote = coordinator.aggregator.current_quote()
                    if current_quote is None:
                        raise ValueError("shadow live aggregator lost its current quote")
                    ledger.observe_quote(current_quote)
                predictions = coordinator.evaluate_ready(
                    exchange_now_ms=last_observed_exchange_ms,
                    order_notional_quote=scorer.reference_order_notional_quote,
                )
            except MicrostructureFeedIntegrityError:
                feed_sequence_gaps += 1
                continue
            except (KeyError, TypeError, ValueError):
                invalid_events += 1
                continue
            for prediction in predictions:
                ledger.observe_prediction(prediction)
    if first_event_ms <= 0 or last_event_ms <= first_event_ms:
        raise ValueError("shadow capture contains no usable market interval")
    ledger.finish_capture()
    pnls = [trade.realized_net_bps for trade in ledger.trades]
    sides = [1 if trade.side == "LONG" else -1 for trade in ledger.trades]
    timestamps = [trade.entry_time_ms for trade in ledger.trades]
    return ShadowReplayResult(
        started_at_ms=first_event_ms,
        completed_at_ms=last_event_ms,
        decisions=ledger.decisions,
        actionable_decisions=ledger.actionable_decisions,
        rejected_while_open=ledger.rejected_while_open,
        execution_liquidity_rejections=ledger.execution_liquidity_rejections,
        expired_entries=ledger.expired_entries,
        pending_entries_at_end=ledger.pending_entries_at_end,
        end_censored_signals=ledger.end_censored_signals,
        trades=tuple(ledger.trades),
        metrics=_trading_metrics(pnls, sides, timestamps),
        feed_sequence_gaps=feed_sequence_gaps,
        invalid_events=invalid_events + coordinator.aggregator.invalid_event_count,
        late_event_resets=coordinator.late_event_resets,
        feature_gap_resets=coordinator.engine.gap_resets,
        deadline_misses=coordinator.deadline_misses,
        inference_failures=coordinator.inference_failures,
        forced_closes=ledger.forced_closes,
        orders_submitted=0,
    )


def evaluate_shadow_capture(
    artifact: MicrostructureModelArtifact,
    artifact_path: str | Path,
    capture: MicrostructureCaptureResult,
    *,
    report_path: str | Path,
    trades_path: str | Path,
    config: ShadowConfig | None = None,
) -> tuple[ShadowEvaluationReport, MicrostructureModelArtifact | None]:
    """Evaluate one immutable capture and return an accepted artifact only on pass."""

    cfg = (config or ShadowConfig()).validated()
    if artifact.status != "shadow_candidate" or artifact.rejection_reasons:
        raise ValueError("shadow evaluation requires an unrejected shadow_candidate")
    if artifact.deployment_refit is None or artifact.deployment_model_strings is None:
        raise ValueError("shadow candidate is missing its deployment refit")
    if artifact.shadow_validation is not None:
        raise ValueError("shadow evidence is already attached")
    if (
        capture.status != "pass"
        or capture.provider != "binance"
        or capture.schema_version != MICROSTRUCTURE_SCHEMA_VERSION
        or capture.market_type != "futures"
        or capture.errors
    ):
        raise ValueError("shadow capture did not pass the futures feed contract")
    if tuple(capture.symbols) != (artifact.symbol,) or len(capture.evidence) != 1:
        raise ValueError("shadow capture does not belong exclusively to the model symbol")
    capture_evidence = capture.evidence[0]
    if capture_evidence.error:
        raise ValueError("shadow capture symbol evidence contains an error")
    lower_symbol = artifact.symbol.lower()
    expected_stream_urls = (
        f"{BINANCE_FUTURES_PUBLIC_STREAM_URL}?streams="
        f"{lower_symbol}@depth@100ms/{lower_symbol}@bookTicker",
        f"{BINANCE_FUTURES_MARKET_STREAM_URL}?streams={lower_symbol}@aggTrade",
    )
    if tuple(capture.stream_urls) != expected_stream_urls:
        raise ValueError("shadow capture did not use the locked Binance public streams")
    raw_path = Path(capture_evidence.synchronized_raw_path)
    original_raw_path = Path(capture_evidence.raw_path)
    snapshot_path = Path(capture_evidence.snapshot_json_path)
    manifest_path = Path(capture.manifest_path)
    if not all(
        path.is_file()
        for path in (raw_path, original_raw_path, snapshot_path, manifest_path)
    ):
        raise ValueError("shadow capture artifacts are missing")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("shadow capture manifest is unreadable") from exc
    if manifest_payload != capture.asdict():
        raise ValueError("shadow capture manifest does not match the supplied evidence")
    if (
        _sha256(original_raw_path) != capture_evidence.raw_sha256
        or _sha256(raw_path) != capture_evidence.synchronized_raw_sha256
        or _sha256(snapshot_path) != capture_evidence.snapshot_json_sha256
    ):
        raise ValueError("shadow capture artifact hash verification failed")
    if (
        capture.started_at_ms <= 0
        or capture.completed_at_ms <= capture.started_at_ms
        or capture.clock_sync.samples < 3
        or not all(
            math.isfinite(value)
            for value in (
                capture.clock_sync.offset_ms,
                capture.clock_sync.median_rtt_ms,
                capture.clock_sync.minimum_rtt_ms,
            )
        )
        or capture.clock_sync.minimum_rtt_ms < 0.0
        or capture.clock_sync.median_rtt_ms < capture.clock_sync.minimum_rtt_ms
    ):
        raise ValueError("shadow capture timing evidence is invalid")
    now_ms = int(time.time() * 1_000)
    if capture.completed_at_ms > now_ms + 60_000:
        raise ValueError("shadow capture completion time lies in the future")
    if now_ms - capture.completed_at_ms > cfg.maximum_capture_age_seconds * 1_000:
        raise ValueError("shadow capture is too old for promotion evaluation")
    try:
        fitted_at = datetime.fromisoformat(artifact.deployment_refit.fitted_at)
    except ValueError as exc:
        raise ValueError("shadow deployment refit timestamp is invalid") from exc
    if fitted_at.tzinfo is None:
        raise ValueError("shadow deployment refit timestamp has no timezone")
    if capture.started_at_ms < int(fitted_at.timestamp() * 1_000):
        raise ValueError("shadow capture predates the deployment refit")
    persisted = load_microstructure_model_artifact(artifact_path)
    if persisted != artifact:
        raise ValueError("shadow artifact path does not match the supplied candidate")
    scorer = load_microstructure_action_scorer(
        artifact_path,
        require_accepted=False,
        as_of_ms=capture.completed_at_ms,
    )
    if capture_evidence.last_exchange_time_ms is None:
        raise ValueError("shadow capture has no terminal exchange timestamp")
    entry_cutoff_ms = (
        int(capture_evidence.last_exchange_time_ms)
        - int(scorer.horizon_seconds) * 1_000
        - max(2_000, int(scorer.total_latency_ms))
    )
    replay = replay_shadow_capture(
        scorer,
        raw_path,
        settlement_delay_ms=cfg.settlement_delay_ms,
        clock_offset_ms=capture.clock_sync.offset_ms,
        entry_cutoff_ms=entry_cutoff_ms,
    )
    reasons: list[str] = []
    if cfg.asdict() != PROMOTION_SHADOW_CONFIG.asdict():
        reasons.append("shadow_protocol_deviates_from_locked_promotion_config")
    if replay.duration_seconds < cfg.minimum_duration_seconds:
        reasons.append("shadow_duration_below_minimum")
    if replay.decisions < cfg.minimum_decisions:
        reasons.append("shadow_decisions_below_minimum")
    if len(replay.trades) < cfg.minimum_virtual_trades:
        reasons.append("shadow_virtual_trades_below_minimum")
    if replay.metrics.total_net_bps <= 0.0 or _risk_utility(
        replay.metrics, artifact.risk_level
    ) <= 0.0:
        reasons.append("shadow_not_profitable_after_drawdown_penalty")
    if replay.metrics.profit_factor is None or replay.metrics.profit_factor <= 1.0:
        reasons.append("shadow_profit_factor_not_above_one")
    if any(
        value != 0
        for value in (
            capture_evidence.sequence_gap_count,
            capture_evidence.crossed_book_count,
            capture_evidence.invalid_event_count,
            replay.feed_sequence_gaps,
            replay.invalid_events,
            replay.late_event_resets,
            replay.feature_gap_resets,
            replay.deadline_misses,
            replay.inference_failures,
            replay.forced_closes,
            replay.expired_entries,
            replay.pending_entries_at_end,
            replay.orders_submitted,
        )
    ):
        reasons.append("shadow_feed_inference_or_position_integrity_failed")
    trade_target = Path(trades_path)
    trades_sha = _write_trades(trade_target, replay.trades)
    candidate_sha = _candidate_payload_sha256(artifact)
    report = ShadowEvaluationReport(
        version=SHADOW_EVIDENCE_VERSION,
        generated_at_ms=int(time.time() * 1_000),
        status="passed" if not reasons else "rejected",
        trading_authority=not reasons,
        reasons=tuple(reasons),
        candidate_sha256=candidate_sha,
        deployment_model_sha256=artifact.deployment_refit.deployment_model_sha256,
        symbol=artifact.symbol,
        provider=_PROVIDER,
        config=cfg.asdict(),
        capture={
            "capture_id": capture.capture_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "raw_path": str(raw_path),
            "raw_sha256": _sha256(raw_path),
            "started_at_ms": capture.started_at_ms,
            "completed_at_ms": capture.completed_at_ms,
            "requested_duration_seconds": capture.requested_duration_seconds,
            "clock_sync": capture.clock_sync.asdict(),
            "symbol_evidence": capture_evidence.asdict(),
        },
        replay={
            "started_at_ms": replay.started_at_ms,
            "completed_at_ms": replay.completed_at_ms,
            "duration_seconds": replay.duration_seconds,
            "decisions": replay.decisions,
            "actionable_decisions": replay.actionable_decisions,
            "rejected_while_open": replay.rejected_while_open,
            "execution_liquidity_rejections": replay.execution_liquidity_rejections,
            "expired_entries": replay.expired_entries,
            "pending_entries_at_end": replay.pending_entries_at_end,
            "end_censored_signals": replay.end_censored_signals,
            "feed_sequence_gaps": replay.feed_sequence_gaps,
            "invalid_events": replay.invalid_events,
            "late_event_resets": replay.late_event_resets,
            "feature_gap_resets": replay.feature_gap_resets,
            "deadline_misses": replay.deadline_misses,
            "inference_failures": replay.inference_failures,
            "forced_closes": replay.forced_closes,
            "orders_submitted": replay.orders_submitted,
        },
        metrics=asdict(replay.metrics),
        trades_path=str(trade_target),
        trades_sha256=trades_sha,
    )
    report_target = Path(report_path)
    write_json_atomic(report_target, report.asdict(), sort_keys=True)
    if not report.passed:
        return report, None
    report_sha = _sha256(report_target)
    profit_factor = replay.metrics.profit_factor
    if profit_factor is None:
        raise RuntimeError("passed shadow report has no finite profit factor")
    evidence = ShadowValidationEvidence(
        version=SHADOW_EVIDENCE_VERSION,
        report_sha256=report_sha,
        trades_sha256=trades_sha,
        capture_manifest_sha256=str(report.capture["manifest_sha256"]),
        raw_capture_sha256=str(report.capture["raw_sha256"]),
        candidate_sha256=candidate_sha,
        deployment_model_sha256=artifact.deployment_refit.deployment_model_sha256,
        symbol=artifact.symbol,
        provider=_PROVIDER,
        clock_offset_ms=float(capture.clock_sync.offset_ms),
        started_at_ms=replay.started_at_ms,
        completed_at_ms=replay.completed_at_ms,
        duration_seconds=replay.duration_seconds,
        decisions=replay.decisions,
        actionable_decisions=replay.actionable_decisions,
        virtual_trades=len(replay.trades),
        long_trades=replay.metrics.long_trades,
        short_trades=replay.metrics.short_trades,
        execution_liquidity_rejections=replay.execution_liquidity_rejections,
        expired_entries=replay.expired_entries,
        pending_entries_at_end=replay.pending_entries_at_end,
        end_censored_signals=replay.end_censored_signals,
        total_net_bps=replay.metrics.total_net_bps,
        profit_factor=float(profit_factor),
        max_drawdown_bps=replay.metrics.max_drawdown_bps,
        feed_sequence_gaps=(
            int(capture_evidence.sequence_gap_count) + replay.feed_sequence_gaps
        ),
        invalid_events=(
            int(capture_evidence.invalid_event_count)
            + int(capture_evidence.crossed_book_count)
            + replay.invalid_events
        ),
        late_event_resets=replay.late_event_resets,
        feature_gap_resets=replay.feature_gap_resets,
        deadline_misses=replay.deadline_misses,
        inference_failures=replay.inference_failures,
        forced_closes=replay.forced_closes,
        orders_submitted=0,
        attached_at=datetime.now(tz=UTC).isoformat(),
    )
    accepted = replace(artifact, status="accepted", shadow_validation=evidence)
    _validated_shadow_binding(accepted)
    return report, accepted


__all__ = [
    "PROMOTION_SHADOW_CONFIG",
    "SHADOW_EVIDENCE_VERSION",
    "ShadowConfig",
    "ShadowEvaluationReport",
    "ShadowReplayResult",
    "VirtualShadowTrade",
    "evaluate_shadow_capture",
    "replay_shadow_capture",
]
