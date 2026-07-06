# Data Provenance Policy

Simple AI Trading must not publish invented market data, invented financial
results, or generated performance charts as optimization evidence.

## Rules

- No tracked files under `data/`.
- No committed optimization result JSON/SVG/CSV unless it is generated from
  exchange-sourced candles or signed exchange/account evidence and includes
  machine-readable provenance. The round `report.json` must declare
  `artifact_class: exchange_sourced_backtest_graph_data`,
  `tracked_repo_artifact: true`, list every tracked graph/data artifact, and
  include an `artifact_integrity` manifest with SHA-256 hashes, byte counts, and
  CSV row/column counts for every tracked artifact other than the report itself.
- Per-round optimization result graphs are latest-only on GitHub. Historical
  round CSV/JSON evidence may remain only when it is still valid and manifested,
  but SVG/PNG result charts from older iterations must be removed. Cross-round
  movement belongs in the single generated `docs/optimization/iteration-progress`
  CSV/SVG artifact, which is rebuilt from machine-readable round reports.
- No docs may present test-double output, unit-test candles, or generated
  regression scenarios as financial performance.
- Backtest and model-lab reports must identify symbol, market, interval, UTC
  span, row count, source scope, gap count, coverage ratio, and whether fills
  are simulated or exchange-confirmed.
- Futures `1s` candle evidence must identify the Binance `aggTrades` archive
  source used to aggregate real trades into one-second OHLCV candles; it must
  not be described as USD-M futures kline evidence. No-trade seconds may only
  be represented as carry-forward candles with zero volume and zero trade count.
- Promotion-grade day-trading optimization evidence must be generated with
  `tools/optimization_round.py --promotion-grade`. That contract is limited to
  the exact BTC/ETH/SOL trio for the selected quote asset, `1s` data, prefilled
  SQLite candles, verified archive checksums, zero missing-second gaps, and the
  configured minimum stored history span. The generated `report.json` must
  include `promotion_grade_contract.status: pass`; otherwise any ROI, P&L,
  drawdown, win-rate, profit-factor, or chart output is research-only.
- Real local run artifacts belong under ignored `data/` and must be regenerated
  from source APIs, not committed as authoritative repo data.
- Tests may use deterministic test doubles to exercise edge cases, but those
  values are not market evidence and must stay inside `tests/`.
- Financial model artifacts must pass sanity checks for finite parameters,
  coherent dimensions, bounded probability settings, nonzero accepted rows,
  valid coverage, and bounded risk metrics before they can be presented as
  live-ready or AI-approved.
- Signed live startup and `risk --live --model` additionally require
  `execution_validation.data_coverage` in the selected model artifact. That
  coverage must match the runtime symbol and market, use SQLite-backed `1s`
  candles, use full available history, span at least one year, show at least
  `99.5%` coverage, show zero missing-second gaps, and contain positive
  candle/model-row counts. A runtime interval other than `1s` blocks signed
  startup even if the model otherwise passes promotion checks.

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
- committed CSV/table source for each graph,
- SHA-256/byte/row-count manifest entry for each committed graph/table artifact,
- command used to regenerate it.

If that evidence is missing, the report must say that no performance claim is
made.
