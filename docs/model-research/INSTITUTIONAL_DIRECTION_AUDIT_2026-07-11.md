# Institutional Direction Audit - 2026-07-11

Status: active architecture decision. This is not profitability, execution, or
trading-authority evidence.

## Verdict

The BTC/ETH/SOL platform direction is viable. The current exact-BBO action-value
work remains useful as a taker-directional baseline, but it is not a sufficient
platform architecture and must not consume a full-history build before bounded
viability is demonstrated.

Round 11 showed the distinction that matters: selection AUC reached
`0.56649-0.79777`, yet every candidate was rejected and no non-overlapping trade
was executable. Some top-score tails were profitable only in the selection
slice while the corresponding policy tail was negative. This is evidence of
forecast discrimination without stable economic actionability, not evidence
that leverage or a lower threshold would create alpha.

## Decisions

1. Optimize executable after-cost return under hard drawdown constraints. Raw
   AUC, trade count, leverage, and gross return are diagnostics, not objectives.
2. Keep abstention first-class. Multiple trades per active day are expected,
   but no daily trade quota may override liquidity, uncertainty, risk, or
   cooldown gates.
3. Train a common predictive contract. Conservative, regular, and aggressive
   may use different barrier policies and action-value heads, but predictor
   regularization must not change merely because the operator selected more
   risk. Profile differences belong in thresholds, sizing, leverage ceilings,
   stops, cooldowns, and loss budgets.
4. Establish unlevered edge first. Leverage scales exposure and losses; it does
   not improve prediction. Profile leverage is a ceiling after stop-loss sizing,
   liquidity capacity, liquidation distance, and portfolio limits.
5. Keep the current taker path as one execution family. Add passive maker
   research only after L2 queue reconstruction and latency-aware fill simulation
   exist; an L1 backtest cannot claim maker fills.
6. Use a dual-rate model system. Fast causal ML/deep time-series inference owns
   candidate scoring. A slower multibillion local model may veto, downsize, or
   extend cooldowns from normalized structured context, but never blocks closes
   or overrides deterministic risk.
7. Require paired AI uplift. The AI path uses the same fixed market periods as
   the baseline, masks symbol/date identity in historical prompts, caches exact
   inputs/outputs, and is rejected unless it improves after-cost returns without
   worsening drawdown or tail loss.

## Multi-Fidelity Evaluation Sequence

### Stage 0 - Contract and Feasibility

- Real, checksummed Binance USD-M `bookTicker` and `trades` only.
- Verify causality, quote age, latency, spread crossing, fees, slippage, stop/take
  path handling, and L1 capacity.
- Measure the ex-post opportunity ceiling and cost decomposition. If even a
  non-causal oracle cannot clear modeled costs at useful frequency, stop that
  execution family instead of tuning a classifier.
- No terminal data and no AI calls.

### Stage 1 - Bounded Model Viability

- One bounded BTC window with separate train, early-stop, calibration, policy,
  and untouched selection roles.
- Shared regularized predictor parameters across risk profiles.
- Direct-mean, upper-quantile, and distributional action scores at 300 and 900
  seconds.
- At least 20 non-overlapping trades is a statistical evidence floor, not a
  command to trade. A low-activity candidate may continue only with explicit
  abstention evidence and positive risk utility.
- Selection is opened once. Failed components receive a predeclared diagnostic
  budget on development roles; selection is never repeatedly tuned.

### Stage 2 - Architecture Screen

Only if Stage 1 exposes reproducible positive tails:

- compare engineered-feature LightGBM with a causal temporal convolution model;
- add multi-task future-return, adverse-excursion, favorable-excursion, and
  holding-time heads;
- test calibrated stacking and regime mixture-of-experts;
- use successive halving on development roles so weak models do not receive full
  epochs or full data;
- run component and feature-group ablations before adding complexity.

### Stage 3 - BTC/ETH/SOL Transfer

- Repeat the fixed architecture on bounded, non-overlapping real windows for
  BTCUSDT, ETHUSDT, and SOLUSDT.
- Add point-in-time cross-asset context and evaluate both pooled and
  symbol-specific heads.
- Reject a pooled model if gains come from one symbol or one regime.
- Evaluate portfolio overlap, correlation clusters, concurrent loss, and cash
  reserve with profile-specific capital limits.

### Stage 4 - AI Uplift

- Run the local multibillion reviewer only on baseline candidate decisions or
  slower regime snapshots, never every tick.
- Compare baseline and AI paths on identical contiguous periods.
- Require positive paired-delta and moving-block-bootstrap lower bounds, plus no
  deterioration in drawdown, CVaR, loss streak, liquidation, or close safety.

### Stage 5 - Full History and Promotion

Only surviving fixed procedures earn multi-year/full-available-history data,
purged walk-forward validation, multiple-testing controls, stress scenarios,
one-use terminal evaluation, shadow operation, and testnet forward evidence.

## Data and Compute Contract

- Raw research data lives in one DuckDB warehouse on operator-selected storage.
  Paths are runtime arguments and never committed.
- Official inventories and SHA-256 sidecars identify immutable source objects.
  Complete partitions are reused; merges are transactional and re-certified.
- Archive ZIPs and extracted CSVs are transient unless retention is explicitly
  requested. Parallel warehouses are not the default because they amplify
  writes and require a second full copy during consolidation.
- A bounded screen may use a bounded inventory certificate. Promotion still
  requires a true full-history inventory and cannot relabel bounded metadata.
- DuckDB scans, decompression, parsing, and ACID writes remain CPU/I/O work.
  LightGBM OpenCL and neural tensor training use the detected accelerator.
  GPU utilization is measured, not forced for workloads where data movement
  would increase elapsed time.
- CPU, memory, VRAM, and disk budgets are explicit. OOM fallback may reduce
  batch size, never silently change labels, model family, precision contract, or
  evaluation periods.

## Required Metrics

Every model comparison must report exact UTC periods, symbols, source hashes,
rows, decisions, trades, active-day ratio, trades per active day, gross and net
return, fees, spread, slippage, turnover, exposure, profit factor, expectancy,
win rate, maximum drawdown, daily return distribution, CVaR, loss streak,
single-trade concentration, calibration, directional baselines, and compute
backend. Capital-level claims additionally require an equity ledger and
profile-specific stop-loss sizing; a sum of per-trade basis points is not ROI.

## Architecture Sources

- Binance public-data format and checksums:
  <https://github.com/binance/binance-public-data>
- HftBacktest queue, latency, and L2/L3 replay:
  <https://github.com/nkaz001/hftbacktest>
- NautilusTrader deterministic backtest/live architecture and reconciliation:
  <https://nautilustrader.io/docs/latest/concepts/architecture/>
- LOBFrame finding that forecast quality does not imply actionable trades:
  <https://arxiv.org/abs/2403.09267>
- TLOB transaction-cost sensitivity:
  <https://arxiv.org/abs/2502.15757>
- Adaptive conformal uncertainty for dependent time series:
  <https://arxiv.org/abs/2202.07282>
- Regime-weighted conformal tail-risk calibration:
  <https://arxiv.org/abs/2602.03903>
- Time-series foundation-model return benchmark showing sparse, asset-specific
  gains over random walk: <https://arxiv.org/abs/2606.27100>
- Deep event-based LOB simulation as research inspiration, not fill evidence:
  <https://arxiv.org/abs/2502.17417>

Forums and issue trackers may supply hypotheses and operational failure modes.
They cannot establish model validity, exchange behavior, or profitability
without primary-source corroboration and local reproduction.
