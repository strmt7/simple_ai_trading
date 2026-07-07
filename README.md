# Simple AI Trading

<!-- BEGIN GENERATED BADGES -->
[![andrej-karpathy-skills](https://img.shields.io/static/v1?label=&message=andrej-karpathy-skills&color=555&logo=github&logoColor=white)](https://github.com/forrestchang/andrej-karpathy-skills)
<!-- END GENERATED BADGES -->

Simple AI Trading is a Windows-first, testnet-first autonomous day-trading CLI and desktop app focused only on the major BTC, ETH, and SOL Binance spot/futures markets. It has been expanded from the original single-pair prototype into a tightly scoped major-asset runtime that measures per-symbol liquidity automatically, trains/retrains models, runs realistic backtests with execution frictions, and exposes the same workflows through both the CLI and Windows app.

## Beta Warning

`0.1.0-beta.1` is a beta research release. Many advanced AI, live-trading, high-frequency archive, optimization, and Windows-app workflows are still incomplete or non-functional in production terms. Signed execution remains testnet/demo-first and real-money mainnet execution is intentionally disabled. Do not rely on this software to protect capital or produce profit.

This software is experimental trading infrastructure. It does not guarantee profit, 1-2% daily returns, or positive ROI. The goal is to make risk, liquidity, execution, and model checks explicit before any non-mainnet order path is used.

## Current Scope

- Major-asset day trading on Binance testnet or Demo Trading endpoints, limited to BTC, ETH, and SOL.
- Default symbols: `BTCUSDC`, `ETHUSDC`, `SOLUSDC`; USD-M futures workflows use the matching `BTCUSDT`, `ETHUSDT`, and `SOLUSDT` contracts.
- Unsupported bases, low-liquidity assets, leveraged-token patterns, and lookalike symbols are rejected before data sync, archive ingestion, universe ranking, or optimization.
- Conservative risk profile by default, with `conservative`, `regular`, and `aggressive` profiles.
- Mandatory diversification controls: minimum eligible assets, single-asset allocation cap, portfolio risk cap, and max open positions.
- Hard loss-budget controls: daily loss, session loss, consecutive-loss lockout, network-interruption halt, and post-reconnect observation cooldown.
- Futures leverage is risk-profile driven when futures mode is enabled: `5x` conservative, `10x` regular, `15x` aggressive, with a hard app-level ceiling of `20x`. Spot trading still resolves to `1x`.
- Profit reinvestment is disabled by default. Enabling it prints a warning because compounding amplifies losses as well as gains.
- CPU-only mode is allowed for wider installability, but AI is disabled there and training/backtesting warns that it will be slower.
- Windows GPU acceleration defaults to DirectML via `torch-directml`, which works across AMD, NVIDIA, and Intel DirectX 12 GPUs.
- Training, retraining, tuning, model-lab feature generation/scoring, external-signal scoring, probability-temperature calibration, threshold calibration, backtest replay, and backtest-panel feature generation use GPU-first `auto` whenever the caller does not explicitly select a backend. Training, scoring, and calibration artifacts record requested/resolved backend evidence; feature builders fall back to the original CPU path if a GPU tensor operation is unavailable. Hybrid model-zoo backtest scoring also uses the tensor backend for Lorentzian, rational-quadratic, and technical-confluence experts when supported.
- AI defaults to a local multibillion model identifier (`qwen3:8b`) and a minimum 2B-parameter preflight. AI-assisted signal approval requires explicit holdout uplift over the non-AI ML baseline, paired trade/window return-delta evidence with a positive-delta-rate and sign-test gate, and no worse drawdown, liquidation, loss-streak, profit-factor, win-rate, or downside return/risk evidence when those metrics are available; otherwise AI remains advisory/review-only. Use `simple-ai-trading ai-benchmark` to compare installed local AI reviewers against structured finance-risk cases before relying on AI review.
- Binance API budget telemetry is captured from exchange response headers when available. Authenticated live startup is blocked when a current budget sample shows any known request-weight or order-count window is 80% or more consumed, or when Binance returns `Retry-After`.

## Install

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -e .[gpu]
```

For a CPU-only install:

```powershell
.\.venv311\Scripts\python.exe -m pip install -e .
```

CPU-only mode can run non-AI workflows, but AI features are disabled and training/backtesting will be much slower.

## Verify Hardware

```powershell
.\.venv311\Scripts\python.exe -m simple_ai_trading compute
.\.venv311\Scripts\python.exe -m simple_ai_trading ai
```

On Windows, a healthy AMD/NVIDIA/Intel GPU install should resolve to `compute=directml`. This host was verified with `torch-directml` on an AMD Radeon GPU using a real tensor operation on `privateuseone:0`.

`simple-ai-trading ai` also reports inferred model size. Use `--model` and `--min-model-parameters-b` if you install a different local LLM. A sub-multibillion model or a CPU-only backend blocks AI approval.

DirectML references:

- https://microsoft.github.io/DirectML/
- https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html

## Configure

```powershell
.\.venv311\Scripts\python.exe -m simple_ai_trading configure
.\.venv311\Scripts\python.exe -m simple_ai_trading connect
.\.venv311\Scripts\python.exe -m simple_ai_trading strategy --profile conservative
```

Set the supported major symbols in the runtime config or pass them to `universe`:

```powershell
.\.venv311\Scripts\python.exe -m simple_ai_trading universe --symbols BTCUSDC,ETHUSDC,SOLUSDC
```

The universe gate first enforces the hard BTC/ETH/SOL product scope, then measures exchange status, quote asset, structural leveraged-token patterns, 24h quote volume, trade count, bid/ask spread, likely pegged-pair behavior, and a combined liquidity score. Automatic ranking can adapt the volume/trade floors to the current leaders inside the selected major-asset quote market, but only above hard absolute liquidity floors and without relaxing spread, leveraged-token, pegged-pair, or major-scope filters. If fewer than the configured minimum assets qualify, the command exits nonzero.

Out-of-market and low-liquidity handling is also data-probed, not a fixed clock rule. Backtests, `live`, and `autonomous start` compare each bar's volume against trailing per-symbol history and same UTC weekday/hour/minute-bucket history from prior bars. That means holidays, partial sessions, unusually thin periods, and changing market participation are detected from the actual exchange data available for that symbol and timestamp, then translated into stricter signal thresholds and smaller position sizes.

Live entry risk also includes a deterministic market-regime unpredictability gate. Conservative, regular, and aggressive profiles set different `max_regime_unpredictability` thresholds and cooldown durations; volatile chop, mixed/low-separation regimes, short windows, insufficient data, or malformed/non-normalized regime scores can block fresh entries and force the bot to wait instead of trading into noise.

## Windows App

Build the native Win32 desktop app:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_native_windows.ps1
```

Launch the desktop operator app:

```powershell
.\.venv311\Scripts\simple-ai-trading-windows.exe
```

or:

```powershell
.\run-gui.cmd
```

The Windows app is a native C++20 Win32 operator workstation inspired by the SuperZip app structure: PowerShell/CMake build, DPI-aware/resizable Win32 layout, DWM dark window chrome, real listbox/combobox/edit/button controls, grouped operator workflows, and generated workflow metadata. The current shell uses a cleaner operator dashboard with Home, Run Bot, Research, Risk Center, Data Center, Settings, and Command Browser pages. Recommended workflow cards never duplicate the pinned safety strip; Stop + Close, Pause, Reconcile, Positions, and Risk Review stay in one always-visible row. The command picker is still generated from the same argparse command contract as the CLI. The parity test `tests/test_ai_runtime_and_parity.py` fails if a CLI command, option flag, positional argument, or choice is not present in the native contract.

The interface is intentionally simple: the app groups workflows by operator intent instead of exposing every internal model check as a separate task. Complex safeguards run in the background and surface as clear states such as blocked, waiting, review required, or stop-and-close.

Startup behavior:

- The app resolves the repo-local `.venv311` Python and sets `PYTHONPATH` before launching CLI commands, so dev builds do not depend on a globally installed package.
- If DirectML/GPU is available, the Compute workflow reports the active backend in the output console.
- If only CPU is available, the app remains usable, shows a warning, and disables AI.
- The app has direct buttons for Stop + Close, Pause, Reconcile, Positions, and Risk Review. Normal workflow cards are distinct paper, research, graph, data, and settings tasks so the UI does not present multiple similar buttons for the same action.
- The bottom status bar shows the shared CLI API-budget summary. It refreshes opportunistically rather than constantly: automatic refresh is capped to the 60-120 second band and defaults to 90 seconds, while command-completion updates use cached status.
- `tools\smoke_native_windows_ui.ps1` launches the app in dry-run mode, walks every workflow page, clicks the dashboard cards and safety buttons, and then performs a real Compute smoke unless `-SkipRealCompute` is passed.
- `tools\capture_native_windows_app.ps1` creates a DPI-aware PNG artifact of the dashboard and fails if the captured app is smaller than the configured minimum window size.
- `tools\validate_native_windows_layout.ps1` launches the real Win32 app, checks stable control geometry, footer API-budget presence, hidden/visible workflow cards, and screenshot pixel health so small-window or overlapping UI regressions fail before release.

## Core Workflows

```powershell
simple-ai-trading fetch --symbol ETHUSDC --limit 1000
simple-ai-trading train --preset thorough --compute-backend directml
simple-ai-trading evaluate
simple-ai-trading backtest --compute-backend directml --execution-db data/market_data.sqlite
simple-ai-trading backtest-chart --output data/backtest_performance.svg --execution-db data/market_data.sqlite
simple-ai-trading backtest-panel --interval 1m --compute-backend directml --execution-db data/market_data.sqlite
simple-ai-trading risk --paper
simple-ai-trading coordinator
simple-ai-trading universe
simple-ai-trading data-sync --symbol BTCUSDC --interval 1m --full-history
simple-ai-trading api-budget --compact
simple-ai-trading archive-sync --symbol BTCUSDC --interval 1s --cadence monthly
simple-ai-trading archive-sync --symbols BTCUSDT,ETHUSDT,SOLUSDT --market futures --interval 1s --cadence daily --start-period 2024-01-01 --end-period 2024-01-31 --plan-only --require-checksum --json
simple-ai-trading archive-sync --symbols BTCUSDT,ETHUSDT,SOLUSDT --market futures --interval 1s --cadence daily --start-period 2024-01-01 --end-period 2024-01-31 --require-checksum
simple-ai-trading data-health --interval 1s --market spot --min-rows 1000000 --require-verified-checksum --json
python tools\optimization_round.py --round-id round-004 --market spot --symbols ETHUSDT --interval 1s --require-prefilled-data --min-data-rows 70000 --require-verified-checksum
python tools\optimization_round.py --round-id round-btc-eth-sol-futures-1s --market futures --quote-asset USDT --objective conservative --model-candidates 3 --promotion-grade --min-promotion-data-years 2 --require-gpu
simple-ai-trading model-blueprint --risk-level conservative
simple-ai-trading model-lab --market futures --quote-asset USDT --interval 1s --market-db data/market_data.sqlite --require-db-data --objective conservative --objective regular --objective aggressive --max-symbols 3
simple-ai-trading ai-review --report data/model_lab/model_lab_report.json
```

`data-sync` writes closed candles, raw exchange snapshots, typed top-of-book spread/depth rows, sync runs, archive-file ingestion records, and API-budget snapshots to the single SQLite market database. With `--full-history`, it pages backward through the venue's maximum allowed kline page size until the exchange returns no older data; normal bounded syncs remain efficient for incremental refreshes. `archive-sync` ingests official Binance public archive ZIPs directly into the same SQLite database and records source URL, rows, byte count, SHA-256, Binance sidecar checksum SHA-256 when available, checksum status, and ingestion status. Spot `1s` uses official kline archives. USD-M futures `1s` uses official `aggTrades` archives and deterministically aggregates real trades into one-second OHLCV candles because Binance does not publish USD-M futures `1s` kline archives. Seconds without trades are represented as carry-forward no-trade candles with zero volume and zero trade count, so coverage is continuous without inventing traded volume. Long second-level backfills should be staged with `--start-period`, `--end-period`, and `--plan-only`; the plan reports listed, filtered, selected, first, and last archive periods plus official S3 byte estimates when available before any ZIP is downloaded. Non-plan archive downloads are blocked above `--max-planned-gb` (`50` by default) so an accidental multi-year command cannot start a huge ingest without an explicit bounded window or raised cap. It accepts a single `--symbol`, comma-separated `--symbols`, or `--top-symbols N`, but all paths reject anything outside BTC, ETH, and SOL quoted in USDC or USDT. `--top-symbols` still ranks the supported majors from current exchange metadata before ingestion so bulk research uses current liquidity rather than stale assumptions. Use `--require-checksum` for promotion-grade archive ingestion; checksum mismatches fail before candle rows are written. Sync artifacts include kline request counts, rows received, coverage ratio, gap count, and Binance rate-limit telemetry captured from response headers when the exchange provides it. `api-budget` reads or refreshes that status; cached samples auto-refresh only when stale, defaulting to 90 seconds. `backtest`, `backtest-chart`, and `backtest-panel` can opt into SQLite evidence with `--execution-db data/market_data.sqlite`, which converts the latest per-symbol bid/ask spread and top-level depth into pessimistic fill assumptions and stores the execution-profile evidence in backtest artifacts.

`data-health` audits the SQLite market database before training or optimization. It reports row counts, UTC spans, span-years, expected rows, gap count, coverage ratio, archive-file status counts, and checksum-status counts for every stored symbol/market/interval or for an explicit symbol batch. It exits nonzero when minimum rows, span, coverage, gap, archive-error, checksum-mismatch, or `--require-verified-checksum` gates fail; bounded windows count missing bars at the requested UTC start and end boundaries, not only gaps between stored rows. Promotion-grade optimization now uses the single fail-closed contract `tools/optimization_round.py --promotion-grade`, which forces the exact BTC/ETH/SOL trio for the selected quote asset, forces `1s`, disables network backfill, requires verified archive checksums, requires zero missing-second gaps, enforces the configured minimum stored history span, and writes `promotion_grade_contract` to `report.json`. If that contract or the critical-analysis layer fails, the tool exits nonzero and the artifacts cannot be treated as performance evidence. Futures `1s` optimization is allowed only with prefilled `aggTrades`-derived candles; otherwise the tool fails instead of calling a nonexistent futures `1s` kline endpoint. The tool defaults to `--model-candidates 3`, evaluating a bounded set of label horizon/target/model regularization candidates per symbol before the final holdout; `report.json` and `backtest-metrics.csv` record `model_candidate_count`, `model_selected_candidate`, and `model_selection_score`. Signed live startup and `risk --live --model` now require promoted `TrainedModel` artifacts to carry multi-candidate evidence; when the resolved runtime backend is DirectML/CUDA/ROCm/MPS, they also require accelerator evidence for training and probability calibration. The round writes `round-status.json` during long runs, including per-symbol phases for data health, SQLite load, feature generation, DirectML training, probability-temperature calibration, threshold calibration, holdout scoring, and artifact streaming, plus `data-health.json` beside the graph tables. When hard data gates are active and symbols are not supplied explicitly, optimization now over-samples the live liquidity-ranked universe and keeps the first requested count that also pass local data-health; skipped candidates are recorded as `selection_health_rejections` in the round report. SVG charts are deterministic visual summaries while the full-resolution timeline CSVs remain the source of truth, avoiding multi-megabyte chart bloat without weakening evidence. Each round report records both configured leverage and effective leverage, so spot evidence cannot be misread as a futures-leverage backtest.

Liquidity-session controls do not assume that "day trading hours" are fixed forever. The active defaults use observed bar volume by symbol and historical clock bucket in backtests, the `live` command, and the autonomous decision function, so low-liquidity holidays, partial days, overnight crypto liquidity drops, and future stock-market schedule changes can be reflected by the data instead of a static UTC start/end hour.

`backtest-chart` writes an SVG performance chart from the actual mark-to-market equity path produced by the day-trading simulation. When timestamps are present, the chart labels the UTC start/end dates and the simulated duration in days/years instead of presenting an unlabeled sample index. The same command appears in the Windows app.

`model-blueprint` exposes the research-backed model and training roadmap as the same CLI/Windows-app parity command. It separates implemented, evidence-only, research, blocked, sandbox, and advisory model families so future model work cannot silently promote AI forecasts, RL policies, or order-book research into executable trading authority without updating tests and docs.

`model-lab` is the cross-symbol optimization workflow for BTC, ETH, and SOL. It automatically ranks the supported majors from exchange ticker/book data, trains the base GPU model across multiple label target/horizon profiles, serializes meta-label take/downsize/skip policy evidence, requires real accepted purged chronological walk-forward folds for selected candidates, applies a selection-risk gate that deflates the selected score by the number of tried model variants, evaluates Lorentzian-neighbor, rational-quadratic-kernel, and technical-confluence hybrid experts, records hybrid ablation scores showing what happens when each expert family is removed, records feature-group ablation scores for the selected advanced feature vector, then replays every accepted objective under symbol-specific execution stress and final-model temporal robustness windows. The current v9 advanced feature signature keeps the v8 information-event labels and expands the aggTrade-derived order-flow block from 9 to 13 fields per window with flow strength, persistence, acceleration, and price/flow divergence; legacy v8 signatures still rebuild the original 9-field layout. The bounded promotion-count search now starts with `default`, `day_trade_frequency_probe_forward`, and `day_trade_frequency_probe_downside`, so the default three-candidate smoke path tests baseline plus explicit long/short intraday activity before wider volatility/session/event probes. Downside-positive labels are oriented to short-side futures execution after probability calibration. Failed selections preserve diagnostic thresholds and trade evidence, but executable models are parked in a no-entry state (`decision_threshold=1.0`, long threshold `1.0`, short side disabled) until a selection replay passes objective gates. The adaptive hybrid model-zoo now searches rejected base selections from their diagnostic thresholds and includes conservative low-base rescue profiles (`technical_rescue_core`, `neighbor_kernel_rescue`, and `balanced_rescue_committee`), but it can replace the base model only when its chronological selection replay passes the same profitability, activity, drawdown, liquidation, and path-quality gates. The optimizer now also runs a stratified rule-alpha template zoo after classifier/hybrid selection, covering momentum breakout, VWAP/RSI mean reversion, trend-pullback, volatility breakout, volume-flow proxy, order-flow momentum, flow-reversion, flow-consensus breakout, liquidity-absorption reversal, micro-flow scalp, VWAP snapback scalp, liquidity-sweep reversal, compression breakout scalp, volume-synchronized flow, and adaptive tape-regime templates with normal and inverted orientation. The default 135-candidate prefix covers the full base family/profile matrix before spending slots on nearby threshold/sensitivity/deadband variants, so short-hold scalp, longer-hold, and flow-state families cannot be starved by nested-loop ordering. Order-flow alpha experts serialize offsets into the advanced aggTrade-derived order-flow feature block plus their feature count so CPU, DirectML, CLI, app, and live scoring stay in parity; CPU/live rule-alpha scoring preserves the full serialized feature vector, and DirectML batch scoring has parity coverage. Rejected alpha evidence records best profile, family, score, P&L, closed trades, win rate, profit factor, max drawdown, exit-reason counts, side counts, reject reason, orientation, evaluated candidate count, active/profitable/accepted candidate counts, forward-event signal count, after-cost forward-edge count, best raw event candidate, most-active candidate, best-PnL candidate, and active family/profile coverage; promoted alpha models serialize as `rule_alpha` experts for CLI/app/live parity. Backtests and live/autonomous entry paths apply enabled meta-label policies as pre-entry skip/downsize gates only. Use `--market futures` to research long/short futures behavior without changing saved runtime defaults. Use `--interval 1s --market-db data/market_data.sqlite --require-db-data` to force training on the locally archived second-level SQLite candles; that path fails instead of falling back to API klines when the DB has no matching rows. Use `--full-history` only for API-based promotion research; recent-limit runs are explicitly labeled `binance_recent_limit`, and DB-backed runs are labeled `sqlite_market_data`, so source scope cannot be mistaken. Every outcome includes a `data_coverage` object with symbol, market, interval, source scope, UTC date span, row count, gap count, coverage ratio, full-history flag, and truth basis. By default, model-lab also consumes `data/autonomous/learning_feedback.json` when present; `--learning-feedback PATH` can point at a specific artifact. A symbol is rejected if any required objective fails profitability, drawdown, trade-count, spread, latency, fee, liquidity-crunch, purged walk-forward, temporal robustness, statistical edge, selection-risk, data-coverage integrity, or unresolved repeated-loss learning-feedback gates. After individual symbols pass, model-lab also writes `portfolio_risk.json` and rejects the accepted set if combined correlation clusters, plain effective symbol count, correlation-adjusted effective symbol count, portfolio CVaR, or portfolio drawdown break the risk-level policy. Portfolio CVaR and drawdown use actual allocation-capped equity weights; undeployed allocation stays cash reserve instead of being normalized into risky exposure. Rejection reports include explicit per-window and portfolio reasons. See [docs/MODEL_RESEARCH_AND_OPTIMIZATION.md](docs/MODEL_RESEARCH_AND_OPTIMIZATION.md), [docs/MODEL_TRAINING_INSPIRATION.md](docs/MODEL_TRAINING_INSPIRATION.md), and [docs/OPEN_SOURCE_GAP_ANALYSIS_AND_RULESET_2026-07-05.md](docs/OPEN_SOURCE_GAP_ANALYSIS_AND_RULESET_2026-07-05.md).

`ai-review` sends a compact, redacted model-lab report to a local structured-output Ollama model and writes `ai_risk_review.json`. It is an advisory risk review with fail-closed output: deterministic model-lab/portfolio failures, missing or failed data-coverage evidence for accepted symbols, failed selection-risk deflation, unresolved learning-feedback promotion blocks, missing or failed AI-vs-ML uplift evidence when AI is enabled, positive hybrid or feature ablation deltas that show a selected component is hurting the accepted score, missing GPU AI capability, sub-multibillion local model evidence, unavailable providers, or invalid model JSON all produce a veto/review-required result rather than an approval.

`coordinator` reads independent loop heartbeats/status for risk, execution, reconciliation, market data, machine learning, AI, and learning feedback, then emits one operator state: `ready`, `waiting`, `review_required`, or `blocked_execution`. Risk/execution/reconciliation can block execution; market data, ML, and AI block new entries; learning feedback does not mutate live positions, but it can block future model promotion when repeated symbol losses have not recovered in stress and temporal validation.

For quick host checks, `model-lab` and `train-suite` accept `--max-candidates N`. This is a smoke/research limiter only; omit it for a full optimization run.

Financial-sanity checks now run on promoted model artifacts, generated backtest results, model-lab reports, and AI-review prechecks. They block impossible model dimensions, non-finite parameters, nonsensical probability settings, missing or poor promoted-model probability calibration evidence, impossible coverage/drawdown values, accepted reports with zero rows, missing or failed accepted-outcome data coverage, incomplete Binance source/provenance evidence, missing truth-basis evidence, nonpositive model-row/candle counts, accepted outcomes without real accepted purged walk-forward folds, accepted outcomes without passed selection-risk evidence, accepted stress/temporal reports without measured scenario/window/statistical-edge evidence, accepted outcomes without an accepted portfolio-risk report, accepted portfolio reports whose symbol evidence is missing/duplicated/mismatched, accepted AI uplift without complete baseline/AI/delta metrics, model-size evidence, and paired holdout statistical evidence, raw backtest liquidation flags/events/losses/exits, and internally inconsistent backtest accounting identities for cash, fees, trade P&L, trade counts, exposure, exit reasons, win rate, path-quality metrics, trade-level timestamps/prices/returns, and equity-curve drawdown. Objective acceptance also consumes the generated-backtest financial-sanity report directly, so no optimizer, model-lab, stress, temporal, or ranking path can promote an incoherent backtest by relying on positive P&L alone. Promoted models must carry calibrated probability evidence with Brier score `<=0.35` and expected calibration error `<=0.20`. These checks are intentionally conservative and do not turn a backtest into an investment recommendation.

## Autonomous Control

```powershell
simple-ai-trading autonomous start --paper
simple-ai-trading autonomous pause
simple-ai-trading autonomous resume
simple-ai-trading autonomous stop
simple-ai-trading autonomous status
simple-ai-trading reconcile
simple-ai-trading positions --stats --learning
```

`stop` is fail-closed for the local autonomous ledger: it writes `STOPPING` and closes locally tracked open positions at the latest available mark price, falling back to entry price if no quote is available, only when live ownership evidence is complete. `reconcile` reads the signed spot/futures account state, compares exchange exposure against verified bot-owned non-paper local open positions, writes `data/autonomous/reconciliation.json`, and exits nonzero on exchange-only, local-only, quantity-mismatched, unverified local live exposure, malformed local open-position ledger files, or malformed account payloads. Futures reconciliation requires a signed account payload with a `positions` list; spot reconciliation requires `balances`. Signed `live --live` startup also uses that reconciliation gate; it refuses to manage pre-existing exchange exposure unless it matches a bot-owned ledger position with a bot client order id and exchange fill/acknowledgement evidence. Spot closes require filled or partially filled exchange status; futures reduce-only closes may also use a bot-owned exchange-acknowledged order ID. Signed CLI opens now use deterministic `sait-o-*` client order ids and signed closes use `sait-c-*`, so a restarted session can prove which exchange exposure is bot-owned before touching it.

Autonomous signed execution now has a shared execution-lifecycle preflight in `simple_ai_trading.execution_lifecycle`. It is separate from model entry risk: a risk-policy block can prevent new entries while still allowing verified bot-owned emergency closes, but ledger corruption, missing signed reconciliation, external exchange exposure, quantity mismatch, or unverified bot ownership block both opens and closes. Authenticated autonomous startup, operator stop, risk-close, auto-close-threshold closes, and every live open call use the same lifecycle vocabulary before order submission. `autonomous start --live` is no longer blocked by the old "not wired" guard, but it remains non-mainnet only and still requires credentials, promoted model readiness, API-budget headroom for startup/open, reconciliation, and bot-owned order evidence.
Autonomous stop, risk-close, and auto-close-threshold paths preserve partially
filled exchange closes: the filled quantity is appended to the closed-trade
ledger, the unfilled quantity remains in `open_positions.json` with
`PARTIALLY_FILLED` evidence, and the close/run report is marked incomplete until
the remainder is closed or reconciled.

Network interruptions are treated as a recovery state, not as a normal trading iteration. The `live` loop keeps retrying market-data reads and records `market_error_retry` events instead of entering on stale data. After connectivity returns, it records a clean recovery observation, waits through `recovery_cooldown_seconds` when configured, and skips fresh entries for that observation step. The autonomous loop adds signed reconciliation before resume: it records a heartbeat that says reconciliation is required, reconciles exchange exposure, checks daily/session loss budgets, checks loss streaks, writes an observation heartbeat, and skips that iteration before allowing any new entry. If reconciliation finds exchange-only exposure, local-only exposure, unverified local live exposure, malformed account payloads, malformed local ledger files, or a quantity mismatch, the autonomous loop exits fail-closed and does not touch positions that are not represented in a verified bot-owned ledger row.

Closed trades automatically refresh `data/autonomous/learning_feedback.json`. This is the bounded self-improvement loop: it summarizes recurring loss reasons, symbols, sides, loss streaks, and retraining/cooldown review hints. It never edits a live model, loosens risk settings, or changes open positions. The next model-lab run consumes it automatically and blocks a symbol with repeated losses unless the new candidate shows positive stress and temporal recovery evidence.

Signed live-style startup requires a promoted model artifact. A model must carry passing `selection_risk` evidence from the model-lab/training-suite promotion path before `live --live` will use it; stale or hand-written model JSON is rejected before order logic starts. That evidence now includes both a multiple-trials deflated selected score and a two-panel CSCV/PBO-style overfit diagnostic that compares selection ranking with validation ranking. Signed startup also requires `execution_validation` stamped by `model-lab`: symbol-specific liquidity evidence, data-coverage integrity, accepted stress validation, accepted temporal robustness, accepted portfolio risk, and any applicable learning-feedback recovery evidence. Plain `train-suite` artifacts are research/paper artifacts until model-lab stamps them after the final portfolio gate. Paper mode can regenerate a bad model for experimentation, but it will not silently trade with a stale artifact. Authenticated live mode also refuses in-loop retraining; run `model-lab` again and promote a fresh artifact instead.

## Risk Levels

`conservative` is the default:

- `5x` default futures leverage, while spot mode still resolves to `1x`.
- Lower stop-loss capital-at-risk budgets and position caps.
- Longer cooldowns.
- Stricter liquidity/spread thresholds.
- Lower drawdown tolerance.

`regular` and `aggressive` default to `10x` and `15x` futures leverage respectively, but still keep leverage capped at `20x`, require diversification, and preserve exchange/testnet safeguards. Leverage is not treated as an ROI target; it only scales permitted futures notional after stop-loss sizing, position caps, loss budgets, exchange brackets, liquidation-buffer checks, and reconciliation gates pass.

Signed futures opens also clamp leverage to the active Binance notional bracket
for the intended gross order notional before local margin is reserved and before
submitting the market order, so a small-order leverage ceiling is not applied
blindly to larger orders. The same bracket-capped value is persisted on the
bot-owned position ledger and reused for reduce-only close accounting. The live
loop does not change futures leverage at startup, and reduce-only closes do not
change leverage; exchange-side leverage is mutated only as part of a fresh
bot-owned futures open.

Position sizing treats `risk_per_trade` as the maximum estimated equity budget
intended to be lost if the configured stop-loss is hit, including taker fees and
the adverse exit-fill buffer from the execution simulator, then caps gross
notional by max position size, per-asset allocation, leverage, exchange
constraints, and available cash. Futures leverage may reduce required margin,
but it cannot push gross exposure above `max_asset_allocation_pct`. Core risk
percentages and execution-cost inputs are normalized in `StrategyConfig` before
any CLI, app, live, or backtest path consumes them; explicit zero remains
operator-visible, while invalid negative or non-finite values fall back to
conservative defaults instead of becoming optimistic assumptions. The CLI, live
loop, risk report, backtester, optimization evidence generator, and Windows app
command surface all use the same stop-loss-sized notional calculation. Signed
non-dry operation is blocked when stop-loss protection is disabled or when one
estimated stop-loss would exceed the tightest active daily, session, or
portfolio risk budget. It is also blocked when the configured stop-loss is
`>=100%`, because that cannot produce a positive protective stop for long
exposure. Signed
futures operation is also blocked when `liquidation_buffer_pct` is disabled, so
leverage cannot run without the maintenance-plus-buffer liquidation proxy.
Backtests preserve candle high/low bounds in model rows; stop-loss, take-profit,
drawdown, and liquidation checks use those intrabar bounds when available, and
an ambiguous bar that touches both stop and take exits at the stop. When the
intrabar adverse mark breaches the drawdown limit, the backtest closes the
position at that same adverse mark instead of a recovered candle close. Entries
filled at a candle close do not reuse that candle's earlier high/low range for
post-entry risk, avoiding impossible same-bar stops from prices that occurred
before the simulated position existed. Futures backtests apply an
isolated-margin liquidation proxy: if margin balance falls below the configured
`liquidation_buffer_pct` maintenance-plus-buffer requirement, the isolated
margin is treated as lost, the position is cleared, and the run is rejected from
promotion.
Backtest win rate is classified by net trade P&L after entry and exit fees, not
by gross price movement, so a fee-eroded trade is not reported as a win.

Exchange-backed trading caps follow the active symbol's quote and base assets. The persisted runtime field names remain backward-compatible (`managed_usdc` for quote capacity and `managed_btc` for base-asset capacity), but the CLI and app render and enforce them as USDC/USDT plus BTC/ETH/SOL according to the configured pair.

Day-trading objectives require enough activity to prove a repeatable edge, but the closed-trade minimum and trades/day target are evidence targets, not quotas that force entries. Sparse historical activity fails model or threshold promotion when there is no risk explanation. If regime or meta-label risk gates explicitly skipped entries during unpredictable markets, low activity can still pass the activity gate so the bot is allowed to wait rather than trade into noise. Path-quality gates also reject superficially profitable candidates when one outlier trade contributes too much of gross positive P&L.

Hard capital controls are separate from ROI goals. The conservative profile defaults to a `0.60%` daily loss budget, `1.20%` session loss budget, two-loss streak lockout, three consecutive network errors before recovery-halt messaging, and a 60 second post-reconnect observation cooldown. Regular and aggressive raise those limits gradually, but risk reporting blocks live operation when these controls are disabled or dangerously loose.

## Optimization Evidence

Round-level implementation notes live under `docs/optimization/`. This repo no longer publishes ROI, P&L, drawdown, or chart claims unless they come from exchange-sourced backtests or signed testnet/paper artifacts with the provenance required by [docs/DATA_PROVENANCE_POLICY.md](docs/DATA_PROVENANCE_POLICY.md). Optimization reports now include `critical_analysis`; zero-trade abstention, no accepted symbols, all nonpositive strategy ROI, no profitable symbols, or any liquidation event are failed evidence even when the passive baseline lost money. Rejected threshold searches preserve `best_*` and `threshold_diagnostic_best_*` trade diagnostics, but rejected executable models are forced into no-entry thresholds so failure diagnostics cannot become live or final-holdout trades. The manual optimization tool exits nonzero for failed verdicts and refreshes `docs/optimization/iteration-progress/` with a single progress CSV/SVG derived from tracked round reports. The provenance audit rejects result charts from more than one historical round, so GitHub keeps only the latest per-iteration graph set plus the rolling progress graph. The current retained smoke checkpoint is `round-alpha-event-study-1d-smoke`: a one-candidate default-model run that exercised the v9 171-field conservative feature vector, seven conservative hybrid profiles, and 270 normal/inverted order-flow-aware rule-alpha replays on verified BTCUSDT/ETHUSDT/SOLUSDT futures 1s data for 2024-06-01. Rule-alpha stop and take-profit targets are floored by modeled one-way execution cost, entry/exit taker fees, and explicit profit/stop buffers so invalid scalp targets cannot look tradable before costs. The run failed profitability gates with zero accepted symbols, zero closed holdout trades, mean ROI `0.0%`, mean buy-and-hold ROI `-0.0990046820488913%`, DirectML training/scoring evidence, and no liquidations. The event-study layer found signals for all 270 rule-alpha variants per symbol but zero positive after-cost forward-edge candidates; the best raw event candidates were still negative net edge (`-33.9167073462523`, `-34.14643238432157`, and `-33.786158043817835` bps for BTCUSDT, ETHUSDT, and SOLUSDT). The expanded alpha search did generate internal lifecycle activity before the final no-entry model: BTCUSDT had 234 active rule-alpha candidates, ETHUSDT had 252, SOLUSDT had 234, and each symbol had a 24-closed-trade most-active candidate. Volume-synchronized flow remained the most-active family on all three symbols, but it was still deeply negative after costs. None were profitable after costs, so the optimizer correctly refused promotion. Performance claims must be regenerated from real source data before being documented.

## Live-Market Simulation

The backtester no longer assumes frictionless fills. It models:

- per-symbol spread,
- latency buffers,
- liquidity haircuts for testnet-to-mainnet differences,
- market impact from candle-volume participation,
- taker fees,
- isolated-margin liquidation-buffer checks,
- same-notional buy-and-hold comparison and risk-adjusted scoring.

See [docs/LIVE_MARKET_SIMULATION.md](docs/LIVE_MARKET_SIMULATION.md).

## Safety Invariants

- Mainnet signed calls are disabled by default.
- Testnet or Demo Trading must be enabled for signed execution.
- Runtime credentials are redacted in artifacts.
- `20x` leverage is the hard app cap even if Binance reports a larger exchange bracket.
- `universe` must prove liquidity for the configured diversified symbols.
- CPU-only mode disables AI.
- CLI and Windows app command parity is tested.
- Native Windows metadata is generated from the Python CLI parser and includes command options, positionals, choices, defaults, and help text.
- Backtests include pessimistic execution assumptions.
- Backtest and model-lab artifacts include data-coverage evidence with UTC date spans, interval, source scope, coverage ratio, gap count, and a truth basis that distinguishes simulated fills from exchange fills.
- Recent-limit research artifacts are labeled and cannot silently claim full available history.
- Model artifacts, generated backtests, and accepted model-lab reports must pass financial-sanity checks for finite parameters, coherent probabilities, valid row counts, valid coverage, internally consistent accounting/path metrics, and bounded risk metrics.
- Objective acceptance is fail-closed on generated-backtest financial sanity before profitability, drawdown, activity, or edge gates are treated as promotable evidence.
- Market-edge reports cannot be accepted unless the underlying generated backtest also passes financial-sanity checks.
- Repo-facing performance claims must come from real source data with the provenance required by `docs/DATA_PROVENANCE_POLICY.md`.
- Authenticated live/testnet order loops do not trust requested quantity as filled quantity; they require execution fields or a signed order-status reconciliation. `spot-roundtrip` also refuses to size its second signed leg from an ACK-only first leg. `origQty`, local requested size, and local fallback price are not fill evidence for live ledgers.
- Authenticated `live --live` startup refuses unverified exchange exposure; it will only resume an existing position if the local bot ledger proves ownership with bot client-order and exchange fill/acknowledgement evidence.
- Autonomous stop closes only live positions with complete bot ownership evidence and preserves uncertain ledger entries with explicit rejection reasons to avoid touching external exposure.
- `reconcile` must be clean before treating the local autonomous ledger as flat or aligned with exchange state; corrupt, non-list, unknown-field, or financially invalid open-position ledger rows are hard failures.
- `live` post-outage recovery requires a clean market observation and cooldown before any fresh entry; autonomous recovery additionally requires reconciliation and hard loss-budget checks.
- Exchange exposure that is not represented in the bot ledger is reported as a mismatch and is not closed by the bot.

## Test

```powershell
.\.venv311\Scripts\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File tools\build_native_windows.ps1
```

Focused checks used during this revamp:

```powershell
.\.venv311\Scripts\python.exe -m pytest -q tests/test_compute.py tests/test_ai_runtime_and_parity.py tests/test_autonomous.py tests/test_market_universe.py tests/test_backtest.py tests/test_backtest_coverage.py
```

## Release

The beta release tag is `v0.1.0-beta.1`. Python packaging uses the PEP 440-compatible version `0.1.0b1`.

The manual GitHub Actions workflow `beta-release` builds the native Windows app, runs tests and coverage, packages a portable beta ZIP, attaches checksums, and publishes a GitHub prerelease. See [docs/release.md](docs/release.md).
