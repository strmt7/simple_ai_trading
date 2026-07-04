# Optimization Evidence

This directory only accepts real-data optimization evidence.

Earlier local engineering benchmark artifacts have been removed. Implementation
rounds may describe code and safety gates, but ROI, P&L, drawdown, and chart
claims must come from exchange-sourced backtests or signed testnet/paper
artifacts with the provenance required by
[Data Provenance Policy](../DATA_PROVENANCE_POLICY.md).

Current implementation notes:

- [Round 001 - Market-Quality Regime Features](round-001-market-quality.md)
- [Round 002 - Learning Feedback Promotion Gate](round-002-learning-feedback-gate.md)
- [Round 003 - Data Coverage and API Efficiency](round-003-data-coverage-api-efficiency.md)
- [Round 004 - Regime Entry Gate](round-004-regime-entry-gate.md)

Real-data graph checkpoints:

| Round | Objective | Configured futures default | Market data used | Symbols | Validation span UTC | Accepted symbols | Mean ROI | Worst drawdown | Tables and charts |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| [005](round-005-conservative-5x-spot-1s-realdata/data/report.json) | conservative | 5x | Binance spot 1s | BTCUSDT, ETHUSDT | 2017-08-17 19:00:13 to 23:59:51 | 0/2 | -0.0238% | 0.0476% | [metrics](round-005-conservative-5x-spot-1s-realdata/data/backtest-metrics.csv), [portfolio](round-005-conservative-5x-spot-1s-realdata/data/portfolio-timeline.csv), [BTC chart](round-005-conservative-5x-spot-1s-realdata/charts/01-BTCUSDT-conservative.svg), [ETH chart](round-005-conservative-5x-spot-1s-realdata/charts/02-ETHUSDT-conservative.svg) |
| [006](round-006-regular-10x-spot-1s-realdata/data/report.json) | regular | 10x | Binance spot 1s | BTCUSDT, ETHUSDT | 2017-08-17 19:00:15 to 23:59:55 | 0/2 | -0.0476% | 0.0953% | [metrics](round-006-regular-10x-spot-1s-realdata/data/backtest-metrics.csv), [portfolio](round-006-regular-10x-spot-1s-realdata/data/portfolio-timeline.csv), [BTC chart](round-006-regular-10x-spot-1s-realdata/charts/01-BTCUSDT-regular.svg), [ETH chart](round-006-regular-10x-spot-1s-realdata/charts/02-ETHUSDT-regular.svg) |
| [007](round-007-aggressive-15x-spot-1s-realdata/data/report.json) | aggressive | 15x | Binance spot 1s | BTCUSDT, ETHUSDT | 2017-08-17 19:00:16 to 23:59:57 | 0/2 | -0.1191% | 0.1191% | [metrics](round-007-aggressive-15x-spot-1s-realdata/data/backtest-metrics.csv), [portfolio](round-007-aggressive-15x-spot-1s-realdata/data/portfolio-timeline.csv), [BTC chart](round-007-aggressive-15x-spot-1s-realdata/charts/01-BTCUSDT-aggressive.svg), [ETH chart](round-007-aggressive-15x-spot-1s-realdata/charts/02-ETHUSDT-aggressive.svg) |
| [008](round-008-futures-conservative-5x-data-health/data/report.json) | conservative | 5x effective futures | Binance USD-M futures 1m | BTCUSDT, ETHUSDT | no prefilled futures rows | 0/2 | 0.0000% | 0.0000% | [metrics](round-008-futures-conservative-5x-data-health/data/backtest-metrics.csv), [data health](round-008-futures-conservative-5x-data-health/data/data-health.json) |
| [009](round-009-futures-conservative-5x-realdata/data/report.json) | conservative | 5x effective futures | Binance USD-M futures 1m | BTCUSDT, ETHUSDT | 2020-10-01 12:06:59 to 2020-12-31 23:51:59 | 0/2 | -10.0584% | 10.0859% | [metrics](round-009-futures-conservative-5x-realdata/data/backtest-metrics.csv), [portfolio](round-009-futures-conservative-5x-realdata/data/portfolio-timeline.csv), [BTC chart](round-009-futures-conservative-5x-realdata/charts/01-BTCUSDT-conservative.svg), [ETH chart](round-009-futures-conservative-5x-realdata/charts/02-ETHUSDT-conservative.svg) |
| [010](round-010-futures-selection-guard-conservative-5x-realdata/data/report.json) | conservative | 5x effective futures | Binance USD-M futures 1m | BTCUSDT, ETHUSDT | 2020-10-01 12:06:59 to 2020-12-31 23:51:59 | 0/2 | -5.4485% | 10.1227% | [metrics](round-010-futures-selection-guard-conservative-5x-realdata/data/backtest-metrics.csv), [portfolio](round-010-futures-selection-guard-conservative-5x-realdata/data/portfolio-timeline.csv), [BTC chart](round-010-futures-selection-guard-conservative-5x-realdata/charts/01-BTCUSDT-conservative.svg), [ETH chart](round-010-futures-selection-guard-conservative-5x-realdata/charts/02-ETHUSDT-conservative.svg) |

Rounds 005-007 are not promotion-grade successes. They are deliberately
committed as failed real-data checkpoints: the local SQLite database currently
contains only `BTCUSDT` and `ETHUSDT` spot `1s` archives for one UTC day in
August 2017, so these runs cannot satisfy the 50-pair, many-year, futures
leverage benchmark. Spot execution still resolves to 1x effective leverage;
the `5x`/`10x`/`15x` values above are the configured futures defaults recorded
in the strategy payloads, not proof that futures leverage improved returns.
The data-health reports show contiguous 1s coverage with zero gaps for both
symbols; the ETHUSDT archive has a verified checksum, while the BTCUSDT archive
is recorded as unverified and must be replaced before a promotion claim.
Round 008 is a futures-specific fail-closed checkpoint: it proves the evidence
tool now records effective futures leverage, but the local SQLite database has
no prefilled `BTCUSDT` or `ETHUSDT` futures `1m` rows, so training and
backtesting are blocked before any result can be promoted.
Round 009 replaces that blocked futures checkpoint with verified Binance USD-M
monthly archives for all of 2020 on `BTCUSDT` and `ETHUSDT`. The evidence still
fails promotion: both symbols hit the conservative drawdown stop near 10%, had
negative realized P&L, negative market edge versus buy-and-hold, zero win rate,
and rejected profit factor/expectancy gates. The scoring backend recorded in
the metrics is DirectML on `privateuseone:0`, so this is a real GPU-backed
failure result, not a CPU-only placeholder.
Round 010 adds a selection slice before the final holdout, uses that selection
slice for threshold calibration and guarded probability inversion, and keeps the
same final holdout span for comparison. It improves the two-symbol mean ROI
from Round 009, mostly by reducing ETHUSDT losses, but it is still rejected:
BTCUSDT remains a drawdown-stop failure and the combined market edge remains
strongly negative versus the passive baseline.

Promotion-grade optimization rounds generated by `tools/optimization_round.py`
must write both graph images and the CSV/JSON data used to render them under a
numbered round directory. The provenance audit only allows tracked
`docs/optimization/<round>/data/*.csv`, `report.json`, `data-health.json`, and
chart SVG artifacts when `report.json` declares `artifact_class:
exchange_sourced_backtest_graph_data`, `tracked_repo_artifact: true`, and lists
the exact tracked artifacts. Future agents must replot from the committed CSV
tables, not visually edit SVGs.

The evidence tool supports `--market spot|futures`. Spot `1s` evidence uses the
Binance Spot kline interval; standard USD-M futures optimization must use a
futures-supported interval such as `1m`. Reports and `backtest-metrics.csv`
record both configured profile leverage and effective leverage so a spot round
with an aggressive profile cannot be mistaken for a 15x futures test.

For promotion claims, run `tools/optimization_round.py` with
`--require-prefilled-data`, `--min-data-rows`, `--min-coverage-ratio`,
`--max-gap-count`, and `--require-verified-checksum` after `archive-sync` has
filled the SQLite market database. That mode records per-symbol database health
and blocks training/backtesting when rows, coverage, gaps, archive status, or
checksum evidence fail instead of silently paging missing candles from the
network during optimization.

Round datasets must preserve liquidity-session columns such as rolling quote
volume, same-period liquidity flags, UTC timestamp, and gap evidence. Graphs
that compare optimization rounds should be regenerated from those tables so
session/holiday effects remain auditable instead of being visually inferred.
