# Optimization Round 003 - Data Coverage and API Efficiency

Date: 2026-07-04

## Scope

This round prevents short, incomplete, or recent-limit datasets from being
misrepresented as full-history financial evidence. It also makes Binance API
usage auditable.

## Implemented

- Added `DataCoverageReport` and `describe_candle_coverage()`.
- `backtest`, `backtest-chart`, `backtest-panel`, and `model-lab` emit data
  coverage evidence with source scope, UTC span, candle counts, model-row
  counts, gap count, coverage ratio, full-history flag, and truth basis.
- `model-lab --full-history` pages backward through Binance klines using the
  venue maximum page size and labels outcomes `binance_full_history`.
- Default model-lab smoke/research fetches are labeled `binance_recent_limit`.
- Data-coverage hard failures block model-lab promotion and AI review.
- Serialized model execution stamps include data coverage.
- `data-sync --full-history` ignores the recent-row target and pages backward
  until no older rows are returned.
- Binance request metadata records used-weight/order-count headers and
  `Retry-After` seconds when provided by the exchange.
- `backtest-chart` labels timestamped equity paths with UTC start/end dates and
  simulated duration in days/years.

## Performance Evidence

No ROI, P&L, drawdown, or chart claim is made in this repo for this round.

This round is a data-truth control. Future performance reports must include the
provenance required by
[Data Provenance Policy](../DATA_PROVENANCE_POLICY.md).
