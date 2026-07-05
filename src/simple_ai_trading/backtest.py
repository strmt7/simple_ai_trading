"""Backtesting engine for autonomous day-trading strategies."""

from __future__ import annotations
import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Sequence

from .assets import MAX_AUTONOMOUS_LEVERAGE
from .compute import BackendInfo, resolve_backend
from .execution_simulation import SymbolExecutionProfile, simulate_market_fill
from .features import ModelRow
from .liquidity_session import apply_liquidity_session_meta, liquidity_session_adjustment
from .meta_label import MetaLabelDecision, apply_meta_label_policy
from .model import (
    TrainedModel,
    confidence_adjusted_probability,
    effective_training_backend_name,
    model_decision_threshold,
)
from .regime import classify_market_regime
from .risk_controls import market_regime_unpredictability, stop_loss_sized_notional_pct
from .types import StrategyConfig


@dataclass
class BacktestResult:
    starting_cash: float
    ending_cash: float
    realized_pnl: float
    win_rate: float
    trades: int
    max_drawdown: float
    closed_trades: int
    gross_exposure: float
    total_fees: float
    stopped_by_drawdown: bool
    max_exposure: float
    trades_per_day_cap_hit: int
    buy_hold_pnl: float = 0.0
    edge_vs_buy_hold: float = 0.0
    scoring_backend_requested: str = "cpu"
    scoring_backend_kind: str = "cpu"
    scoring_backend_device: str = "cpu"
    scoring_backend_reason: str = ""
    equity_curve: tuple[dict[str, float | int], ...] = ()
    trade_pnls: tuple[float, ...] = ()
    trade_returns: tuple[float, ...] = ()
    trade_log: tuple[dict[str, object], ...] = ()
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_trade_return: float = 0.0
    trade_return_stdev: float = 0.0
    max_consecutive_losses: int = 0
    meta_label_skips: int = 0
    meta_label_downsizes: int = 0
    regime_entry_skips: int = 0


@dataclass(frozen=True)
class ThresholdBacktestCalibration:
    threshold: float
    accepted: bool
    score: float
    realized_pnl: float
    total_fees: float
    max_drawdown: float
    win_rate: float
    closed_trades: int
    edge_vs_buy_hold: float
    baseline_threshold: float
    baseline_score: float
    baseline_realized_pnl: float
    baseline_closed_trades: int
    best_threshold: float
    best_score: float
    evaluated_thresholds: int
    rows: int

    def asdict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _finite_float(value: object, default: float = 0.0) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def risk_adjusted_backtest_score(result: object, *, starting_cash: float = 1000.0) -> float:
    realized = _finite_float(getattr(result, "realized_pnl", 0.0))
    total_fees = max(0.0, _finite_float(getattr(result, "total_fees", 0.0)))
    max_drawdown = max(0.0, _finite_float(getattr(result, "max_drawdown", 0.0)))
    closed_trades = int(_finite_float(getattr(result, "closed_trades", 0), 0.0))
    stopped_by_drawdown = bool(getattr(result, "stopped_by_drawdown", False))
    cash = max(1.0, _finite_float(starting_cash, 1000.0))

    score = realized - total_fees - (max_drawdown * cash)
    if stopped_by_drawdown:
        score -= cash * 0.5
    if closed_trades <= 0:
        score -= cash * 0.05
    return float(score)


def _bps_to_rate(bps: float) -> float:
    return max(0.0, bps) / 10_000.0


def _fill_price(
    price: float,
    side_sign: int,
    cfg: StrategyConfig,
    *,
    notional: float = 0.0,
    volume: float = 0.0,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> float:
    return simulate_market_fill(
        price,
        side_sign,
        notional,
        cfg,
        bar_volume_notional=volume,
        symbol_profile=symbol_profile,
    ).fill_price


def _normalize_market_direction(signal_score: float, threshold: float, market_type: str) -> int:
    if market_type == "futures":
        threshold = max(0.5, float(threshold))
        if signal_score >= threshold:
            return 1
        if signal_score <= (1.0 - threshold):
            return -1
        return 0
    return 1 if signal_score >= threshold else 0


def _close_position(
    position_side: int,
    price: float,
    entry_price: float,
    qty: float,
    notional: float,
    margin_used: float,
    cfg: StrategyConfig,
    *,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> tuple[float, float, float]:
    fee_rate = _bps_to_rate(cfg.taker_fee_bps)
    exit_price = _fill_price(
        price,
        -position_side,
        cfg,
        notional=abs(notional),
        volume=abs(notional) * 20.0,
        symbol_profile=symbol_profile,
    )
    realized = position_side * (exit_price - entry_price) * qty
    exit_fee = abs(exit_price * qty) * fee_rate
    return margin_used + realized - exit_fee, realized, exit_fee


def _safe_day(ts_ms: int) -> int:
    return int(ts_ms // (24 * 60 * 60 * 1000))


def _equity_point(timestamp: int, equity: float, drawdown: float, position_side: int) -> dict[str, float | int]:
    return {
        "timestamp": int(timestamp),
        "equity": float(equity),
        "drawdown": float(max(0.0, drawdown)),
        "position_side": int(position_side),
    }


def _trade_return(net_pnl: float, equity_reference: float) -> float:
    reference = max(1.0, abs(_finite_float(equity_reference, 1.0)))
    return float(net_pnl) / reference


def _max_consecutive_losses(trade_pnls: list[float]) -> int:
    longest = 0
    current = 0
    for pnl in trade_pnls:
        if pnl < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _path_quality_metrics(
    trade_pnls: list[float],
    trade_returns: list[float],
) -> dict[str, float | int]:
    clean_pnls = [float(value) for value in trade_pnls if math.isfinite(float(value))]
    clean_returns = [float(value) for value in trade_returns if math.isfinite(float(value))]
    gross_profit = sum(value for value in clean_pnls if value > 0.0)
    gross_loss = abs(sum(value for value in clean_pnls if value < 0.0))
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    expectancy = sum(clean_pnls) / len(clean_pnls) if clean_pnls else 0.0
    average_return = sum(clean_returns) / len(clean_returns) if clean_returns else 0.0
    if len(clean_returns) >= 2:
        variance = sum((value - average_return) ** 2 for value in clean_returns) / (len(clean_returns) - 1)
        return_stdev = math.sqrt(max(0.0, variance))
    else:
        return_stdev = 0.0
    return {
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(min(999.0, max(0.0, profit_factor))),
        "expectancy": float(expectancy),
        "average_trade_return": float(average_return),
        "trade_return_stdev": float(return_stdev),
        "max_consecutive_losses": int(_max_consecutive_losses(clean_pnls)),
    }


def _buy_hold_pnl(
    rows: List[ModelRow],
    starting_cash: float,
    cfg: StrategyConfig,
    *,
    market_type: str,
    leverage: float,
    symbol_profile: SymbolExecutionProfile | None = None,
) -> float:
    """Return fee/slippage-aware same-notional buy-and-hold baseline P&L."""

    if not rows or starting_cash <= 0:
        return 0.0
    first = rows[0].close
    last = rows[-1].close
    if first <= 0 or last <= 0:
        return 0.0
    baseline_notional = starting_cash * stop_loss_sized_notional_pct(cfg, market_type, leverage=leverage)
    if baseline_notional <= 0.0:
        return 0.0
    fee_rate = _bps_to_rate(cfg.taker_fee_bps)
    entry = _fill_price(
        first,
        1,
        cfg,
        notional=baseline_notional,
        volume=baseline_notional * 20.0,
        symbol_profile=symbol_profile,
    )
    exit_price = _fill_price(
        last,
        -1,
        cfg,
        notional=baseline_notional,
        volume=baseline_notional * 20.0,
        symbol_profile=symbol_profile,
    )
    if entry <= 0 or exit_price <= 0:
        return 0.0
    qty = baseline_notional / entry
    entry_fee = baseline_notional * fee_rate
    exit_notional = qty * exit_price
    exit_fee = exit_notional * fee_rate
    return exit_notional - exit_fee - baseline_notional - entry_fee


def _clamp_threshold(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return max(0.0, min(1.0, value))


def _score_backend_payload(backend: BackendInfo) -> dict[str, str]:
    return {
        "scoring_backend_requested": backend.requested,
        "scoring_backend_kind": backend.kind,
        "scoring_backend_device": backend.device,
        "scoring_backend_reason": backend.reason,
    }


def _fallback_score_backend(requested: BackendInfo, reason: str) -> BackendInfo:
    return BackendInfo(
        requested=requested.requested,
        kind="cpu",
        device="cpu",
        vendor="Python stdlib",
        reason=reason[:240],
    )


def _position_size_from_risk(
    cash: float,
    cfg: StrategyConfig,
    *,
    market_type: str,
    leverage: float,
) -> tuple[float, float]:
    """Return gross notional and margin sized from stop-loss risk budget."""

    if cash <= 0.0:
        return 0.0, 0.0
    notional_pct = stop_loss_sized_notional_pct(cfg, market_type, leverage=leverage)
    gross = cash * notional_pct
    if market_type == "spot":
        return max(0.0, gross), max(0.0, gross)

    effective_leverage = max(1.0, min(MAX_AUTONOMOUS_LEVERAGE, _finite_float(leverage, 1.0)))
    margin = gross / effective_leverage if effective_leverage > 0.0 else gross
    return max(0.0, gross), max(0.0, margin)


def _torch_device_for_backend(backend: BackendInfo):  # pragma: no cover - optional GPU runtime
    if backend.kind == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return backend.device


def _batch_probabilities_torch(  # pragma: no cover - exercised by host GPU smoke verification
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    backend: BackendInfo,
    batch_size: int,
) -> list[float]:
    import torch  # type: ignore

    device = _torch_device_for_backend(backend)
    batch = max(1, int(batch_size or 8192))
    probabilities: list[float] = []
    temperature = max(1e-6, float(getattr(model, "probability_temperature", 1.0) or 1.0))

    members = list(model.ensemble_members)
    model_specs = []
    if members:
        for member in members:
            model_specs.append((member.weights, member.bias, member.feature_means, member.feature_stds))
    else:
        model_specs.append((model.weights, model.bias, model.feature_means, model.feature_stds))
    model_tensors = []
    for weights, bias, means, stds in model_specs:
        w = torch.tensor(list(weights), dtype=torch.float32, device=device)
        b = torch.tensor(float(bias), dtype=torch.float32, device=device)
        mean_t = torch.tensor(list(means), dtype=torch.float32, device=device)
        std_t = torch.tensor(list(stds), dtype=torch.float32, device=device)
        std_t = torch.where(torch.abs(std_t) > 0.0, std_t, torch.ones_like(std_t))
        model_tensors.append((w, b, mean_t, std_t))

    base_mean_t = torch.tensor(list(model.feature_means), dtype=torch.float32, device=device)
    base_std_t = torch.tensor(list(model.feature_stds), dtype=torch.float32, device=device)
    base_std_t = torch.where(torch.abs(base_std_t) > 0.0, base_std_t, torch.ones_like(base_std_t))
    hybrid_specs = []
    for expert in getattr(model, "hybrid_experts", []) or []:
        expert_weight = max(0.0, float(expert.weight))
        if expert_weight <= 0.0:
            continue
        if expert.kind in {"lorentzian_knn", "rational_quadratic_kernel"}:
            prototypes = [
                prototype
                for prototype in expert.prototypes
                if len(prototype.features) == model.feature_dim
            ]
            if not prototypes:
                continue
            proto_features = torch.tensor(
                [prototype.features for prototype in prototypes],
                dtype=torch.float32,
                device=device,
            )
            proto_labels = torch.tensor(
                [float(1 if prototype.label else 0) for prototype in prototypes],
                dtype=torch.float32,
                device=device,
            )
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                proto_features,
                proto_labels,
                max(1, min(int(expert.k), len(prototypes))),
                max(1e-6, float(expert.bandwidth)),
                max(1e-6, float(expert.alpha)),
                max(1, int(expert.feature_count)),
            ))
        elif expert.kind == "technical_confluence":
            hybrid_specs.append((
                expert.kind,
                expert_weight,
                None,
                None,
                0,
                1.0,
                1.0,
                max(1, int(expert.feature_count)),
            ))

    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        features = torch.tensor([row.features for row in chunk], dtype=torch.float32, device=device)
        chunk_probs = None
        for w, b, mean_t, std_t in model_tensors:
            normalized = (features - mean_t) / std_t
            logits = normalized.matmul(w.reshape(-1, 1)).reshape(-1) + b
            logits = torch.clamp(logits / temperature, min=-50.0, max=50.0)
            probs = torch.sigmoid(logits)
            chunk_probs = probs if chunk_probs is None else chunk_probs + probs
        if chunk_probs is None:  # pragma: no cover - model_specs is always populated above
            raise RuntimeError("No model probabilities were produced.")
        if len(model_specs) > 1:
            chunk_probs = chunk_probs / float(len(model_specs))
        if hybrid_specs:
            base_weight = max(0.0, float(getattr(model, "hybrid_base_weight", 1.0)))
            weighted = chunk_probs * base_weight
            total = torch.full_like(chunk_probs, base_weight)
            expert_features = (features - base_mean_t) / base_std_t
            for kind, expert_weight, proto_features, proto_labels, k, bandwidth, alpha, feature_count in hybrid_specs:
                expert_probs = None
                if kind == "lorentzian_knn" and proto_features is not None and proto_labels is not None:
                    distances = torch.log1p(torch.abs(expert_features[:, None, :] - proto_features[None, :, :]))
                    distances = distances.sum(dim=2) / float(max(1, model.feature_dim))
                    nearest_distance, nearest_index = torch.topk(distances, k=int(k), dim=1, largest=False)
                    label_matrix = proto_labels.reshape(1, -1).expand(features.shape[0], -1)
                    nearest_labels = torch.gather(label_matrix, 1, nearest_index)
                    neighbor_weights = 1.0 / torch.clamp(nearest_distance + 1e-6, min=1e-9)
                    expert_probs = (neighbor_weights * nearest_labels).sum(dim=1) / neighbor_weights.sum(dim=1)
                elif kind == "rational_quadratic_kernel" and proto_features is not None and proto_labels is not None:
                    deltas = expert_features[:, None, :] - proto_features[None, :, :]
                    scaled = torch.sum(deltas * deltas, dim=2) / float(max(1, model.feature_dim))
                    kernel_weights = torch.pow(
                        1.0 + scaled / float(2.0 * alpha * bandwidth * bandwidth),
                        float(-alpha),
                    )
                    expert_probs = (kernel_weights * proto_labels.reshape(1, -1)).sum(dim=1) / kernel_weights.sum(dim=1)
                elif kind == "technical_confluence":
                    count = max(1, min(int(feature_count), int(features.shape[1]), 13))
                    values = features[:, :count]
                    if count < 13:
                        padding = torch.zeros((features.shape[0], 13 - count), dtype=torch.float32, device=device)
                        values = torch.cat((values, padding), dim=1)
                    else:
                        values = values[:, :13]
                    momentum_1 = values[:, 0]
                    momentum_3 = values[:, 1]
                    momentum_10 = values[:, 2]
                    momentum_20 = values[:, 3]
                    ema_spread = values[:, 4]
                    rsi = torch.clamp(values[:, 5], min=0.0, max=1.0)
                    ema_gap = values[:, 6]
                    relative_atr = torch.abs(values[:, 7])
                    volatility_20 = torch.abs(values[:, 8])
                    volume_ratio = values[:, 9]
                    trend_acceleration = values[:, 10]
                    gap_to_vwap = values[:, 11]
                    volume_trend = values[:, 12]
                    trend = (
                        0.24 * torch.tanh(momentum_20 * 80.0)
                        + 0.20 * torch.tanh(momentum_10 * 100.0)
                        + 0.14 * torch.tanh(momentum_3 * 140.0)
                        - 0.16 * torch.tanh(ema_spread * 90.0)
                        + 0.10 * torch.tanh(trend_acceleration * 240.0)
                        + 0.06 * torch.tanh(volume_trend * 4.0)
                    )
                    mean_reversion = (
                        0.18 * torch.tanh((0.38 - rsi) * 5.0)
                        - 0.14 * torch.tanh(gap_to_vwap * 150.0)
                        - 0.08 * torch.tanh(momentum_1 * 180.0)
                    )
                    breakout = (
                        0.10 * torch.tanh(volume_ratio * 2.5)
                        + 0.10 * torch.tanh((relative_atr + volatility_20) * 80.0)
                        + 0.08 * torch.tanh((momentum_10 + momentum_20) * 80.0)
                        - 0.04 * torch.tanh(torch.abs(ema_gap) * 150.0)
                    )
                    expert_probs = torch.clamp(torch.sigmoid((trend + mean_reversion + breakout) * 2.2), min=0.0, max=1.0)
                if expert_probs is None:
                    continue
                weighted = weighted + float(expert_weight) * expert_probs
                total = total + float(expert_weight)
            chunk_probs = weighted / torch.clamp(total, min=1e-12)
        if bool(getattr(model, "probability_inverted", False)):
            chunk_probs = 1.0 - chunk_probs
        chunk_probs = torch.clamp(chunk_probs, min=0.0, max=1.0)
        probabilities.extend(float(value) for value in chunk_probs.detach().cpu().tolist())
    return probabilities


def _backtest_probabilities(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    compute_backend: str | None,
    batch_size: int,
) -> tuple[list[float], BackendInfo]:
    backend = resolve_backend(effective_training_backend_name(compute_backend))
    if backend.kind == "cpu":
        return [model.predict_proba(row.features) for row in rows], backend
    try:
        return _batch_probabilities_torch(rows, model, backend=backend, batch_size=batch_size), backend
    except Exception as exc:
        fallback = _fallback_score_backend(
            backend,
            f"{backend.kind} backtest scoring failed ({exc.__class__.__name__}); fell back to CPU",
        )
        return [model.predict_proba(row.features) for row in rows], fallback


def _threshold_grid(start: float, end: float, steps: int, baseline: float) -> list[float]:
    if steps <= 1:
        return [_clamp_threshold(baseline)]
    start = _clamp_threshold(float(start))
    end = _clamp_threshold(float(end))
    if end <= start:
        end = min(1.0, start + 0.01)
    values = [start + (end - start) * i / (steps - 1) for i in range(steps)]
    values.append(_clamp_threshold(baseline))
    return sorted(set(round(_clamp_threshold(value), 12) for value in values))


def _result_payload(result: BacktestResult) -> dict[str, float | int]:
    return {
        "realized_pnl": float(result.realized_pnl),
        "total_fees": float(result.total_fees),
        "max_drawdown": float(result.max_drawdown),
        "win_rate": float(result.win_rate),
        "closed_trades": int(result.closed_trades),
        "edge_vs_buy_hold": float(result.edge_vs_buy_hold),
        "profit_factor": float(getattr(result, "profit_factor", 0.0)),
        "expectancy": float(getattr(result, "expectancy", 0.0)),
        "average_trade_return": float(getattr(result, "average_trade_return", 0.0)),
        "max_consecutive_losses": int(getattr(result, "max_consecutive_losses", 0)),
        "regime_entry_skips": int(getattr(result, "regime_entry_skips", 0)),
    }


def calibrate_threshold_for_backtest(
    rows: List[ModelRow],
    model: TrainedModel,
    cfg: StrategyConfig,
    *,
    starting_cash: float = 1000.0,
    market_type: str = "spot",
    baseline_threshold: float | None = None,
    start: float = 0.05,
    end: float = 0.95,
    steps: int = 31,
    min_score_delta: float = 0.0,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
) -> ThresholdBacktestCalibration:
    baseline_threshold = _clamp_threshold(
        _finite_float(baseline_threshold, model_decision_threshold(model, cfg.signal_threshold))
        if baseline_threshold is not None
        else model_decision_threshold(model, cfg.signal_threshold)
    )
    if market_type == "futures":
        baseline_threshold = max(0.5, baseline_threshold)
        start = max(0.5, float(start))
    probabilities, score_backend = _backtest_probabilities(
        rows,
        model,
        compute_backend=compute_backend,
        batch_size=score_batch_size,
    )
    baseline_model = replace(model, decision_threshold=baseline_threshold)
    baseline_result = run_backtest(
        rows,
        baseline_model,
        cfg,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
        precomputed_probabilities=probabilities,
        precomputed_score_backend=score_backend,
    )
    baseline_score = risk_adjusted_backtest_score(baseline_result, starting_cash=starting_cash)
    best_threshold = baseline_threshold
    best_score = baseline_score
    best_result = baseline_result
    thresholds = _threshold_grid(start, end, steps, baseline_threshold)

    for threshold in thresholds:
        candidate_model = replace(model, decision_threshold=threshold)
        result = run_backtest(
            rows,
            candidate_model,
            cfg,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
            precomputed_probabilities=probabilities,
            precomputed_score_backend=score_backend,
        )
        score = risk_adjusted_backtest_score(result, starting_cash=starting_cash)
        if (
            score > best_score + 1e-12
            or (
                abs(score - best_score) <= 1e-12
                and result.realized_pnl > best_result.realized_pnl
            )
        ):
            best_threshold = threshold
            best_score = score
            best_result = result

    has_profit_backed_result = best_result.closed_trades > 0 and best_result.realized_pnl > 0.0
    accepted = (
        has_profit_backed_result
        and best_score > baseline_score + max(0.0, _finite_float(min_score_delta, 0.0))
    )
    selected_threshold = best_threshold if accepted else baseline_threshold
    selected_score = best_score if accepted else baseline_score
    selected_result = best_result if accepted else baseline_result
    payload = _result_payload(selected_result)
    return ThresholdBacktestCalibration(
        threshold=float(selected_threshold),
        accepted=bool(accepted),
        score=float(selected_score),
        realized_pnl=float(payload["realized_pnl"]),
        total_fees=float(payload["total_fees"]),
        max_drawdown=float(payload["max_drawdown"]),
        win_rate=float(payload["win_rate"]),
        closed_trades=int(payload["closed_trades"]),
        edge_vs_buy_hold=float(payload["edge_vs_buy_hold"]),
        baseline_threshold=float(baseline_threshold),
        baseline_score=float(baseline_score),
        baseline_realized_pnl=float(baseline_result.realized_pnl),
        baseline_closed_trades=int(baseline_result.closed_trades),
        best_threshold=float(best_threshold),
        best_score=float(best_score),
        evaluated_thresholds=len(thresholds),
        rows=len(rows),
    )


def run_backtest(
    rows: List[ModelRow],
    model: TrainedModel,
    cfg: StrategyConfig,
    *,
    starting_cash: float = 1000.0,
    market_type: str = "spot",
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
    symbol_profile: SymbolExecutionProfile | None = None,
    precomputed_probabilities: Sequence[float] | None = None,
    precomputed_score_backend: BackendInfo | None = None,
) -> BacktestResult:
    score_backend = resolve_backend(effective_training_backend_name(compute_backend))
    if not rows:
        return BacktestResult(
            starting_cash=starting_cash,
            ending_cash=starting_cash,
            realized_pnl=0.0,
            win_rate=0.0,
            trades=0,
            max_drawdown=0.0,
            stopped_by_drawdown=False,
            closed_trades=0,
            gross_exposure=0.0,
            total_fees=0.0,
            max_exposure=0.0,
            trades_per_day_cap_hit=0,
            buy_hold_pnl=0.0,
            edge_vs_buy_hold=0.0,
            **_score_backend_payload(score_backend),
        )

    cash = float(starting_cash)
    equity_peak = cash
    max_drawdown = 0.0
    stopped_by_drawdown = False
    wins = 0
    closed_trades = 0
    total_fees = 0.0
    max_exposure = 0.0
    cap_hits = 0
    equity_curve: list[dict[str, float | int]] = []
    trade_pnls: list[float] = []
    trade_returns: list[float] = []
    trade_log: list[dict[str, object]] = []
    meta_label_skips = 0
    meta_label_downsizes = 0
    regime_entry_skips = 0

    position_side = 0
    notional = 0.0
    qty = 0.0
    entry_price = 0.0
    margin_used = 0.0
    entry_fee_paid = 0.0
    entry_equity_reference = cash
    entry_timestamp = int(rows[0].timestamp)
    entry_meta = MetaLabelDecision(False, "take", 1.0, 0.0, "initial")
    pending_signal = 0
    pending_meta = MetaLabelDecision(False, "no_signal", 0.0, 0.0, "initial")
    last_close_timestamp: int | None = None
    cooldown_ms = max(0, int(cfg.cooldown_minutes)) * 60 * 1000
    unpredictability_cooldown_ms = max(0, int(cfg.unpredictability_cooldown_minutes)) * 60 * 1000
    regime_cooldown_until: int | None = None
    final_mark_price = rows[-1].close

    fee_rate = _bps_to_rate(cfg.taker_fee_bps)
    leverage = 1.0 if market_type == "spot" else cfg.leverage
    if leverage < 1:
        leverage = 1.0
    if market_type == "futures" and leverage > MAX_AUTONOMOUS_LEVERAGE:
        leverage = MAX_AUTONOMOUS_LEVERAGE
    decision_threshold = model_decision_threshold(model, cfg.signal_threshold)

    daily_trade_count: Dict[int, int] = {}
    max_daily: int | None = int(cfg.max_trades_per_day)
    if max_daily <= 0:
        max_daily = None

    max_open_positions = int(cfg.max_open_positions)
    regime_gate_min_rows = max(8, min(len(rows), int(cfg.liquidity_lookback_bars)))
    if precomputed_probabilities is not None:
        probabilities = precomputed_probabilities
        if len(probabilities) != len(rows):
            raise ValueError(
                f"precomputed_probabilities length mismatch: {len(probabilities)}/{len(rows)}"
            )
        if precomputed_score_backend is not None:
            score_backend = precomputed_score_backend
    else:
        probabilities, score_backend = _backtest_probabilities(
            rows,
            model,
            compute_backend=compute_backend,
            batch_size=score_batch_size,
        )

    for row_index, (row, raw_score) in enumerate(zip(rows, probabilities, strict=True)):
        execution_signal = pending_signal
        execution_meta = pending_meta
        score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        liquidity_adjustment = liquidity_session_adjustment(
            rows,
            row_index,
            cfg,
            decision_threshold,
        )
        pending_signal = _normalize_market_direction(score, liquidity_adjustment.threshold, market_type)
        base_pending_meta = apply_meta_label_policy(
            getattr(model, "meta_label_policy", {}),
            adjusted_probability=score,
            threshold=liquidity_adjustment.threshold,
            side=pending_signal,
            market_type=market_type,
        )
        pending_meta = apply_liquidity_session_meta(base_pending_meta, liquidity_adjustment) if pending_signal != 0 else base_pending_meta
        price = row.close
        final_mark_price = price
        regime_gate_ready = row_index + 1 >= regime_gate_min_rows
        if regime_gate_ready:
            regime_window_start = max(0, row_index + 1 - max(8, int(cfg.liquidity_lookback_bars)))
            regime_evidence = classify_market_regime(rows[regime_window_start:row_index + 1])
            regime_score = market_regime_unpredictability(
                regime_evidence.dominant_regime,
                regime_evidence.confidence,
                regime_evidence.notes,
            )
        else:
            regime_score = 0.0
        regime_limit = float(cfg.max_regime_unpredictability)
        regime_score_over_limit = regime_score > regime_limit
        if regime_score_over_limit and unpredictability_cooldown_ms > 0:
            regime_cooldown_until = max(
                int(regime_cooldown_until or row.timestamp),
                int(row.timestamp) + unpredictability_cooldown_ms,
            )
        regime_cooldown_active = regime_cooldown_until is not None and int(row.timestamp) < regime_cooldown_until
        day = _safe_day(row.timestamp)
        if day not in daily_trade_count:
            daily_trade_count[day] = 0
        trade_cap_reached = max_daily is not None and daily_trade_count[day] >= max_daily
        cooldown_active = (
            last_close_timestamp is not None
            and cooldown_ms > 0
            and row.timestamp - last_close_timestamp < cooldown_ms
        )
        if position_side == 0 and execution_signal != 0:
            if execution_meta.size_multiplier <= 0.0:
                if execution_meta.enabled:
                    meta_label_skips += 1
                continue
            if regime_score_over_limit or regime_cooldown_active:
                regime_entry_skips += 1
                continue
            if cooldown_active:
                continue
            if trade_cap_reached:
                cap_hits += 1
                continue
            gross, effective_margin = _position_size_from_risk(
                cash,
                cfg,
                market_type=market_type,
                leverage=leverage,
            )
            if execution_meta.enabled and execution_meta.action == "downsize":
                meta_label_downsizes += 1
                multiplier = max(0.0, min(1.0, float(execution_meta.size_multiplier)))
                gross *= multiplier
                effective_margin *= multiplier

            if gross <= 0 or effective_margin >= cash:
                continue

            if max_open_positions <= 0:
                cap_hits += 1
                continue

            side_sign = 1 if execution_signal > 0 else -1
            row_volume_notional = max(0.0, float(getattr(row, "volume", 0.0) or 0.0) * price)
            entry = _fill_price(
                price,
                side_sign,
                cfg,
                notional=gross,
                volume=row_volume_notional,
                symbol_profile=symbol_profile,
            )
            if entry <= 0:
                continue

            fee = gross * fee_rate
            total_cost = effective_margin + fee
            if cash < total_cost:
                continue

            entry_equity_reference = cash
            entry_timestamp = int(row.timestamp)
            entry_fee_paid = fee
            cash -= total_cost
            total_fees += fee
            position_side = side_sign
            notional = side_sign * gross
            qty = abs(gross / entry)
            entry_price = entry
            margin_used = effective_margin
            daily_trade_count[day] = daily_trade_count.get(day, 0) + 1
            entry_meta = execution_meta

            max_exposure = max(max_exposure, abs(notional))

        elif position_side != 0:
            current_pnl_pct = (price - entry_price) / entry_price if position_side > 0 else (entry_price - price) / entry_price
            should_close = (
                current_pnl_pct >= cfg.take_profit_pct
                or current_pnl_pct <= -cfg.stop_loss_pct
                or execution_signal == 0
                or execution_signal == (-position_side)
            )

            if should_close:
                closed_side = position_side
                closed_notional = abs(notional)
                closed_entry_price = entry_price
                cash_delta, realized, exit_fee = _close_position(
                    position_side=position_side,
                    price=price,
                    entry_price=entry_price,
                    qty=qty,
                    notional=notional,
                    margin_used=margin_used,
                    cfg=cfg,
                    symbol_profile=symbol_profile,
                )
                cash += cash_delta
                total_fees += exit_fee
                closed_trades += 1
                net_pnl = realized - entry_fee_paid - exit_fee
                return_pct = _trade_return(net_pnl, entry_equity_reference)
                trade_pnls.append(float(net_pnl))
                trade_returns.append(float(return_pct))
                trade_log.append({
                    "opened_at": int(entry_timestamp),
                    "closed_at": int(row.timestamp),
                    "side": int(closed_side),
                    "gross_notional": float(closed_notional),
                    "entry_price": float(closed_entry_price),
                    "exit_mark_price": float(price),
                    "realized_pnl": float(realized),
                    "net_pnl": float(net_pnl),
                    "return_pct": float(return_pct),
                    "entry_fee": float(entry_fee_paid),
                    "exit_fee": float(exit_fee),
                    "meta_label_action": str(entry_meta.action),
                    "meta_label_size_multiplier": float(entry_meta.size_multiplier),
                    "meta_label_signal_strength": float(entry_meta.signal_strength),
                    "meta_label_reason": str(entry_meta.reason),
                })
                if realized > 0:
                    wins += 1

                position_side = 0
                notional = 0.0
                qty = 0.0
                entry_price = 0.0
                margin_used = 0.0
                entry_fee_paid = 0.0
                entry_equity_reference = cash
                entry_meta = MetaLabelDecision(False, "take", 1.0, 0.0, "reset")
                last_close_timestamp = row.timestamp

        # mark-to-market drawdown control with unrealized exposure
        if position_side != 0:
            unrealized = position_side * (price - entry_price) * qty
            equity = cash + margin_used + unrealized
        else:
            equity = cash

        if equity > equity_peak:
            equity_peak = equity
        dd = 1.0 if equity <= 0.0 and equity_peak > 0.0 else ((equity_peak - equity) / equity_peak if equity_peak else 0.0)
        if dd > max_drawdown:
            max_drawdown = dd
        equity_curve.append(_equity_point(int(row.timestamp), equity, dd, position_side))

        if cfg.max_drawdown_limit > 0.0 and dd >= cfg.max_drawdown_limit:
            stopped_by_drawdown = True
            break

    # force close residual position at final mark
    if position_side != 0:
        closed_side = position_side
        closed_notional = abs(notional)
        closed_entry_price = entry_price
        final_delta, final_realized, final_fee = _close_position(
            position_side=position_side,
            price=final_mark_price,
            entry_price=entry_price,
            qty=qty,
            notional=notional,
            margin_used=margin_used,
            cfg=cfg,
            symbol_profile=symbol_profile,
        )
        cash += final_delta
        total_fees += final_fee
        closed_trades += 1
        net_pnl = final_realized - entry_fee_paid - final_fee
        return_pct = _trade_return(net_pnl, entry_equity_reference)
        trade_pnls.append(float(net_pnl))
        trade_returns.append(float(return_pct))
        trade_log.append({
            "opened_at": int(entry_timestamp),
            "closed_at": int(rows[-1].timestamp),
            "side": int(closed_side),
            "gross_notional": float(closed_notional),
            "entry_price": float(closed_entry_price),
            "exit_mark_price": float(final_mark_price),
            "realized_pnl": float(final_realized),
            "net_pnl": float(net_pnl),
            "return_pct": float(return_pct),
            "entry_fee": float(entry_fee_paid),
            "exit_fee": float(final_fee),
            "meta_label_action": str(entry_meta.action),
            "meta_label_size_multiplier": float(entry_meta.size_multiplier),
            "meta_label_signal_strength": float(entry_meta.signal_strength),
            "meta_label_reason": str(entry_meta.reason),
        })
        if final_realized > 0:
            wins += 1
        position_side = 0
        notional = 0.0
        qty = 0.0
        entry_price = 0.0
        margin_used = 0.0
        if cash > equity_peak:
            equity_peak = cash
        final_dd = 1.0 if cash <= 0.0 and equity_peak > 0.0 else ((equity_peak - cash) / equity_peak if equity_peak else 0.0)
        if final_dd > max_drawdown:
            max_drawdown = final_dd
        equity_curve.append(_equity_point(int(rows[-1].timestamp), cash, final_dd, position_side))

    realized_pnl = cash - starting_cash
    win_rate = wins / closed_trades if closed_trades else 0.0

    trades = closed_trades

    buy_hold_pnl = _buy_hold_pnl(
        rows,
        starting_cash,
        cfg,
        market_type=market_type,
        leverage=leverage,
        symbol_profile=symbol_profile,
    )
    path_quality = _path_quality_metrics(trade_pnls, trade_returns)

    return BacktestResult(
        starting_cash=starting_cash,
        ending_cash=cash,
        realized_pnl=realized_pnl,
        win_rate=win_rate,
        trades=trades,
        max_drawdown=max_drawdown,
        stopped_by_drawdown=stopped_by_drawdown,
        closed_trades=closed_trades,
        gross_exposure=max_exposure,
        total_fees=total_fees,
        max_exposure=max_exposure,
        trades_per_day_cap_hit=cap_hits,
        buy_hold_pnl=buy_hold_pnl,
        edge_vs_buy_hold=realized_pnl - buy_hold_pnl,
        equity_curve=tuple(equity_curve),
        trade_pnls=tuple(trade_pnls),
        trade_returns=tuple(trade_returns),
        trade_log=tuple(trade_log),
        gross_profit=float(path_quality["gross_profit"]),
        gross_loss=float(path_quality["gross_loss"]),
        profit_factor=float(path_quality["profit_factor"]),
        expectancy=float(path_quality["expectancy"]),
        average_trade_return=float(path_quality["average_trade_return"]),
        trade_return_stdev=float(path_quality["trade_return_stdev"]),
        max_consecutive_losses=int(path_quality["max_consecutive_losses"]),
        meta_label_skips=int(meta_label_skips),
        meta_label_downsizes=int(meta_label_downsizes),
        regime_entry_skips=int(regime_entry_skips),
        **_score_backend_payload(score_backend),
    )
