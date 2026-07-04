# Data Provenance Policy

Simple AI Trading must not publish invented market data, invented financial
results, or generated performance charts as optimization evidence.

## Rules

- No tracked files under `data/`.
- No committed optimization result JSON/SVG unless it is generated from
  exchange-sourced candles or signed exchange/account evidence and includes
  machine-readable provenance.
- No docs may present test-double output, unit-test candles, or generated
  regression scenarios as financial performance.
- Backtest and model-lab reports must identify symbol, market, interval, UTC
  span, row count, source scope, gap count, coverage ratio, and whether fills
  are simulated or exchange-confirmed.
- Real local run artifacts belong under ignored `data/` and must be regenerated
  from source APIs, not committed as authoritative repo data.
- Tests may use deterministic test doubles to exercise edge cases, but those
  values are not market evidence and must stay inside `tests/`.
- Financial model artifacts must pass sanity checks for finite parameters,
  coherent dimensions, bounded probability settings, nonzero accepted rows,
  valid coverage, and bounded risk metrics before they can be presented as
  live-ready or AI-approved.

## Required Evidence For Performance Claims

Any future optimization report that claims ROI, P&L, drawdown, win rate, or
profit factor must include:

- public or signed data source name,
- symbol list,
- market type,
- interval,
- exact UTC start and end,
- candle or fill row count,
- request/source scope,
- coverage ratio and gap count,
- fees, spread, latency, liquidity, and slippage assumptions,
- generated artifact path,
- command used to regenerate it.

If that evidence is missing, the report must say that no performance claim is
made.
