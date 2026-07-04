# Optimization Round 004 - Regime Entry Gate

Date: 2026-07-04

## Scope

This round improves risk control, not profitability reporting. It adds a
point-in-time live-entry gate that can force the bot to wait when the current
market regime is noisy, under-separated, or data-poor.

## Research Basis

- Binance Spot diff-depth streams publish order-book updates at 1000 ms or 100
  ms, which supports live spread/depth monitoring without high REST polling.
- Binance USD-M futures docs recommend websocket/user-data streams for volatile
  markets and note that depth streams can update as fast as 100 ms.
- Regime-filtered trading literature commonly treats market-regime detection as
  a risk-management filter: disallow or downsize trades in high-volatility
  regimes instead of forcing every model signal into the market.
- Existing repo model-lab work already uses triple-barrier labels,
  meta-labeling, purged walk-forward checks, and temporal robustness. This
  round extends the same "do not trade noisy states" principle into live
  entry-risk control.

## Implemented

- Added `StrategyConfig.max_regime_unpredictability`, defaulting to the
  conservative threshold.
- Added profile-specific thresholds:
  conservative `0.60`, regular `0.72`, aggressive `0.85`.
- Added `strategy --max-regime-unpredictability` so the CLI and Windows app
  contract expose the setting.
- Added `market_regime_unpredictability()` in risk controls. It scores
  volatile chop, mixed/low-separation regimes, insufficient data, short windows,
  and flat/noisy returns on a 0-1 scale.
- Extended `assess_entry_risk()` with `unpredictable_regime` and
  `regime_cooldown` blocks.
- Wired `live` and autonomous decisions to classify rolling point-in-time model
  rows before order sizing. `live` also logs `regime_unpredictability_gate`
  events and activates the configured cooldown.

## Performance Evidence

No ROI, P&L, drawdown, win-rate, or chart claim is made for this round.

This round is a safety implementation. Future profitability rounds must use
exchange-sourced artifacts and the provenance required by
[Data Provenance Policy](../DATA_PROVENANCE_POLICY.md).

## Validation

The focused tests added in this round verify:

- noisy regimes score above safer trending regimes,
- risk reports expose the regime gate,
- over-limit regime scores block entries,
- active regime cooldowns block entries,
- `strategy --max-regime-unpredictability` persists the setting.
