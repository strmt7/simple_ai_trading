# Simple AI Trading

<!-- BEGIN GENERATED BADGES -->
[![andrej-karpathy-skills](https://img.shields.io/static/v1?label=&message=andrej-karpathy-skills&color=555&logo=github&logoColor=white)](https://github.com/forrestchang/andrej-karpathy-skills)
<!-- END GENERATED BADGES -->

Simple AI Trading is a Windows-first, testnet-first autonomous day-trading CLI and desktop app for liquid Binance spot and futures markets. It has been expanded from the original single-pair prototype into a diversified runtime that can manage multiple symbols, measure per-symbol liquidity automatically, train/retrain models, run realistic backtests with execution frictions, and expose the same workflows through both the CLI and Windows app.

## Beta Warning

`0.1.0-beta.1` is a beta research release. Many advanced AI, live-trading, high-frequency archive, optimization, and Windows-app workflows are still incomplete or non-functional in production terms. Signed execution remains testnet/demo-first and real-money mainnet execution is intentionally disabled. Do not rely on this software to protect capital or produce profit.

This software is experimental trading infrastructure. It does not guarantee profit, 1-2% daily returns, or positive ROI. The goal is to make risk, liquidity, execution, and model checks explicit before any non-mainnet order path is used.

## Current Scope

- Multi-asset day trading on Binance testnet or Demo Trading endpoints.
- Default symbols: `BTCUSDC`, `ETHUSDC`, `BNBUSDC`; users can configure any Binance symbol, then `universe` must prove liquidity before use.
- Conservative risk profile by default, with `conservative`, `regular`, and `aggressive` profiles.
- Mandatory diversification controls: minimum eligible assets, single-asset allocation cap, portfolio risk cap, and max open positions.
- Hard loss-budget controls: daily loss, session loss, consecutive-loss lockout, network-interruption halt, and post-reconnect observation cooldown.
- Futures leverage is risk-profile driven when futures mode is enabled: `5x` conservative, `10x` regular, `15x` aggressive, with a hard app-level ceiling of `20x`. Spot trading still resolves to `1x`.
- Profit reinvestment is disabled by default. Enabling it prints a warning because compounding amplifies losses as well as gains.
- CPU-only mode is allowed for wider installability, but AI is disabled there and training/backtesting warns that it will be slower.
- Windows GPU acceleration defaults to DirectML via `torch-directml`, which works across AMD, NVIDIA, and Intel DirectX 12 GPUs.
- Training, retraining, tuning, model-lab scoring, external-signal scoring, threshold calibration, and backtest replay use GPU-first `auto` whenever the caller does not explicitly select a backend. CPU use is explicit or recorded as a fallback reason in model/backtest artifacts. Hybrid model-zoo backtest scoring also uses the tensor backend for Lorentzian, rational-quadratic, and technical-confluence experts when supported.
- AI defaults to a local multibillion model identifier (`qwen2.5:7b`) and a minimum 2B-parameter preflight. AI-assisted signal approval requires explicit holdout uplift over the non-AI ML baseline; otherwise AI remains advisory/review-only.
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

Set multiple symbols in the runtime config or pass them to `universe`:

```powershell
.\.venv311\Scripts\python.exe -m simple_ai_trading universe --symbols BTCUSDC,ETHUSDC,BNBUSDC
```

The universe gate does not use a static allowlist. It measures exchange status, quote asset, structural leveraged-token patterns, 24h quote volume, trade count, bid/ask spread, likely pegged-pair behavior, and a combined liquidity score. Automatic ranking can adapt the volume/trade floors to the current leaders in the selected quote-asset market, but only above hard absolute liquidity floors and without relaxing spread, leveraged-token, or pegged-pair filters. If fewer than the configured minimum assets qualify, the command exits nonzero.

Out-of-market and low-liquidity handling is also data-probed, not a fixed clock rule. Backtests, `live`, and `autonomous start` compare each bar's volume against trailing per-symbol history and same UTC weekday/hour/minute-bucket history from prior bars. That means holidays, partial sessions, unusually thin periods, and changing market participation are detected from the actual exchange data available for that symbol and timestamp, then translated into stricter signal thresholds and smaller position sizes.

Live entry risk also includes a deterministic market-regime unpredictability gate. Conservative, regular, and aggressive profiles set different `max_regime_unpredictability` thresholds and cooldown durations; volatile chop, mixed/low-separation regimes, short windows, or insufficient data can block fresh entries and force the bot to wait instead of trading into noise.

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
simple-ai-trading risk --paper
simple-ai-trading coordinator
simple-ai-trading universe
simple-ai-trading data-sync --symbol BTCUSDC --interval 1m --full-history
simple-ai-trading api-budget --compact
simple-ai-trading archive-sync --symbol BTCUSDC --interval 1s --cadence monthly
simple-ai-trading archive-sync --top-symbols 50 --quote-asset USDT --interval 1s --cadence daily --require-checksum
simple-ai-trading data-health --interval 1s --market spot --min-rows 1000000 --require-verified-checksum --json
python tools\optimization_round.py --round-id round-004 --market spot --symbols ETHUSDT --interval 1s --require-prefilled-data --min-data-rows 70000 --require-verified-checksum
python tools\optimization_round.py --round-id round-008 --market futures --symbols BTCUSDT,ETHUSDT --interval 1m --objective conservative --require-prefilled-data --min-data-rows 100000 --require-verified-checksum
simple-ai-trading model-blueprint --risk-level conservative
simple-ai-trading model-lab --market futures --objective conservative --objective regular --objective aggressive --max-symbols 5 --full-history
simple-ai-trading ai-review --report data/model_lab/model_lab_report.json
```

`data-sync` writes closed candles, raw exchange snapshots, typed top-of-book spread/depth rows, sync runs, archive-file ingestion records, and API-budget snapshots to the single SQLite market database. With `--full-history`, it pages backward through the venue's maximum allowed kline page size until the exchange returns no older data; normal bounded syncs remain efficient for incremental refreshes. `archive-sync` ingests official Binance public archive ZIPs such as 1-second spot klines from `data.binance.vision` directly into the same SQLite database and records source URL, rows, byte count, SHA-256, Binance sidecar checksum SHA-256 when available, checksum status, and ingestion status. It accepts a single `--symbol`, comma-separated `--symbols`, or `--top-symbols N`, which ranks high-liquidity symbols from current exchange metadata before ingestion so bulk research does not rely on a precompiled pair list. Use `--require-checksum` for promotion-grade archive ingestion; checksum mismatches fail before candle rows are written. Sync artifacts include kline request counts, rows received, coverage ratio, gap count, and Binance rate-limit telemetry captured from response headers when the exchange provides it. `api-budget` reads or refreshes that status; cached samples auto-refresh only when stale, defaulting to 90 seconds. `backtest`, `backtest-chart`, and `backtest-panel` can opt into SQLite evidence with `--execution-db data/market_data.sqlite`, which converts the latest per-symbol bid/ask spread and top-level depth into pessimistic fill assumptions and stores the execution-profile evidence in backtest artifacts.

`data-health` audits the SQLite market database before training or optimization. It reports row counts, UTC spans, expected rows, gap count, coverage ratio, archive-file status counts, and checksum-status counts for every stored symbol/market/interval or for an explicit symbol batch. It exits nonzero when minimum rows, coverage, gap, archive-error, checksum-mismatch, or `--require-verified-checksum` gates fail. Promotion-grade optimization rounds can use the same gates with `tools/optimization_round.py --require-prefilled-data --min-data-rows N --require-verified-checksum`, which blocks training/backtesting instead of silently paging missing data from the network and writes `data-health.json` beside the graph tables. The evidence tool now accepts `--market spot|futures`: spot supports Binance `1s` klines, while the standard USD-M futures kline endpoint must use a supported futures interval such as `1m`. Each round report records both configured leverage and effective leverage, so spot evidence cannot be misread as a futures-leverage backtest.

Liquidity-session controls do not assume that "day trading hours" are fixed forever. The active defaults use observed bar volume by symbol and historical clock bucket in backtests, the `live` command, and the autonomous decision function, so low-liquidity holidays, partial days, overnight crypto liquidity drops, and future stock-market schedule changes can be reflected by the data instead of a static UTC start/end hour.

`backtest-chart` writes an SVG performance chart from the actual mark-to-market equity path produced by the day-trading simulation. When timestamps are present, the chart labels the UTC start/end dates and the simulated duration in days/years instead of presenting an unlabeled sample index. The same command appears in the Windows app.

`model-blueprint` exposes the research-backed model and training roadmap as the same CLI/Windows-app parity command. It separates implemented, evidence-only, research, blocked, sandbox, and advisory model families so future model work cannot silently promote AI forecasts, RL policies, or order-book research into executable trading authority without updating tests and docs.

`model-lab` is the cross-symbol optimization workflow. It automatically ranks high-liquidity symbols from exchange ticker/book data, trains the base GPU model across multiple label target/horizon profiles, serializes meta-label take/downsize/skip policy evidence, requires purged chronological walk-forward evidence for selected candidates, applies a selection-risk gate that deflates the selected score by the number of tried model variants, evaluates Lorentzian-neighbor, rational-quadratic-kernel, and technical-confluence hybrid experts, records hybrid ablation scores showing what happens when each expert family is removed, records feature-group ablation scores for the selected advanced feature vector, then replays every accepted objective under symbol-specific execution stress and final-model temporal robustness windows. Backtests and live/autonomous entry paths apply enabled meta-label policies as pre-entry skip/downsize gates only. Use `--market futures` to research long/short futures behavior without changing saved runtime defaults. Use `--full-history` for promotion-grade research; recent-limit runs are explicitly labeled `binance_recent_limit` and cannot be mistaken for full-history evidence. Every outcome includes a `data_coverage` object with symbol, market, interval, source scope, UTC date span, row count, gap count, coverage ratio, full-history flag, and truth basis. By default, model-lab also consumes `data/autonomous/learning_feedback.json` when present; `--learning-feedback PATH` can point at a specific artifact. A symbol is rejected if any required objective fails profitability, drawdown, trade-count, spread, latency, fee, liquidity-crunch, temporal robustness, statistical edge, selection-risk, data-coverage integrity, or unresolved repeated-loss learning-feedback gates. After individual symbols pass, model-lab also writes `portfolio_risk.json` and rejects the accepted set if combined correlation clusters, effective symbol count, portfolio CVaR, or portfolio drawdown break the risk-level policy. Rejection reports include explicit per-window and portfolio reasons. See [docs/MODEL_RESEARCH_AND_OPTIMIZATION.md](docs/MODEL_RESEARCH_AND_OPTIMIZATION.md) and [docs/MODEL_TRAINING_INSPIRATION.md](docs/MODEL_TRAINING_INSPIRATION.md).

`ai-review` sends a compact, redacted model-lab report to a local structured-output Ollama model and writes `ai_risk_review.json`. It is an advisory risk review with fail-closed output: deterministic model-lab/portfolio failures, missing or failed data-coverage evidence for accepted symbols, failed selection-risk deflation, unresolved learning-feedback promotion blocks, missing or failed AI-vs-ML uplift evidence when AI is enabled, positive hybrid or feature ablation deltas that show a selected component is hurting the accepted score, missing GPU AI capability, sub-multibillion local model evidence, unavailable providers, or invalid model JSON all produce a veto/review-required result rather than an approval.

`coordinator` reads independent loop heartbeats/status for risk, execution, reconciliation, market data, machine learning, AI, and learning feedback, then emits one operator state: `ready`, `waiting`, `review_required`, or `blocked_execution`. Risk/execution/reconciliation can block execution; market data, ML, and AI block new entries; learning feedback does not mutate live positions, but it can block future model promotion when repeated symbol losses have not recovered in stress and temporal validation.

For quick host checks, `model-lab` and `train-suite` accept `--max-candidates N`. This is a smoke/research limiter only; omit it for a full optimization run.

Financial-sanity checks now run on promoted model artifacts, model-lab reports, and AI-review prechecks. They block impossible model dimensions, non-finite parameters, nonsensical probability settings, impossible coverage/drawdown values, accepted reports with zero rows, and failed data coverage. These checks are intentionally conservative and do not turn a backtest into an investment recommendation.

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

`stop` is fail-closed for the local autonomous ledger: it writes `STOPPING` and closes any locally tracked open positions at the latest available mark price, falling back to entry price if no quote is available. `reconcile` reads the signed spot/futures account state, compares exchange exposure against non-paper local open positions, writes `data/autonomous/reconciliation.json`, and exits nonzero on exchange-only, local-only, or quantity-mismatched exposure. Signed `live --live` startup also uses that reconciliation gate; it refuses to manage pre-existing exchange exposure unless it matches a bot-owned ledger position with a bot client order id. Signed CLI opens now use deterministic `sait-o-*` client order ids and signed closes use `sait-c-*`, so a restarted session can prove which exchange exposure is bot-owned before touching it.

Network interruptions are treated as a recovery state, not as a normal trading iteration. The `live` loop keeps retrying market-data reads and records `market_error_retry` events instead of entering on stale data. After connectivity returns, it records a clean recovery observation, waits through `recovery_cooldown_seconds` when configured, and skips fresh entries for that observation step. The autonomous loop adds signed reconciliation before resume: it records a heartbeat that says reconciliation is required, reconciles exchange exposure, checks daily/session loss budgets, checks loss streaks, writes an observation heartbeat, and skips that iteration before allowing any new entry. If reconciliation finds exchange-only exposure, local-only exposure, or a quantity mismatch, the autonomous loop exits fail-closed and does not touch positions that are not represented in the bot ledger.

Closed trades automatically refresh `data/autonomous/learning_feedback.json`. This is the bounded self-improvement loop: it summarizes recurring loss reasons, symbols, sides, loss streaks, and retraining/cooldown review hints. It never edits a live model, loosens risk settings, or changes open positions. The next model-lab run consumes it automatically and blocks a symbol with repeated losses unless the new candidate shows positive stress and temporal recovery evidence.

Signed live-style startup requires a promoted model artifact. A model must carry passing `selection_risk` evidence from the model-lab/training-suite promotion path before `live --live` will use it; stale or hand-written model JSON is rejected before order logic starts. That evidence now includes both a multiple-trials deflated selected score and a two-panel CSCV/PBO-style overfit diagnostic that compares selection ranking with validation ranking. Signed startup also requires `execution_validation` stamped by `model-lab`: symbol-specific liquidity evidence, data-coverage integrity, accepted stress validation, accepted temporal robustness, accepted portfolio risk, and any applicable learning-feedback recovery evidence. Plain `train-suite` artifacts are research/paper artifacts until model-lab stamps them after the final portfolio gate. Paper mode can regenerate a bad model for experimentation, but it will not silently trade with a stale artifact. Authenticated live mode also refuses in-loop retraining; run `model-lab` again and promote a fresh artifact instead.

## Risk Levels

`conservative` is the default:

- `5x` default futures leverage, while spot mode still resolves to `1x`.
- Lower stop-loss capital-at-risk budgets and position caps.
- Longer cooldowns.
- Stricter liquidity/spread thresholds.
- Lower drawdown tolerance.

`regular` and `aggressive` default to `10x` and `15x` futures leverage respectively, but still keep leverage capped at `20x`, require diversification, and preserve exchange/testnet safeguards. Leverage is not treated as an ROI target; it only scales permitted futures notional after stop-loss sizing, position caps, loss budgets, exchange brackets, and reconciliation gates pass.

Position sizing treats `risk_per_trade` as the maximum equity budget intended to be lost at the configured stop-loss distance, then caps gross notional by max position size, leverage, exchange constraints, and available cash. The CLI, live loop, risk report, backtester, optimization evidence generator, and Windows app command surface all use the same stop-loss-sized notional calculation.

Hard capital controls are separate from ROI goals. The conservative profile defaults to a `0.60%` daily loss budget, `1.20%` session loss budget, two-loss streak lockout, three consecutive network errors before recovery-halt messaging, and a 60 second post-reconnect observation cooldown. Regular and aggressive raise those limits gradually, but risk reporting blocks live operation when these controls are disabled or dangerously loose.

## Optimization Evidence

Round-level implementation notes live under `docs/optimization/`. This repo no longer publishes ROI, P&L, drawdown, or chart claims unless they come from exchange-sourced backtests or signed testnet/paper artifacts with the provenance required by [docs/DATA_PROVENANCE_POLICY.md](docs/DATA_PROVENANCE_POLICY.md). Round 001 adds market-quality regime features and risk-aware promotion checks, Round 002 turns closed-trade learning feedback into a promotion gate, and Round 003 adds full-history/data-coverage truth controls plus API-efficiency telemetry. Performance claims must be regenerated from real source data before being documented.

## Live-Market Simulation

The backtester no longer assumes frictionless fills. It models:

- per-symbol spread,
- latency buffers,
- liquidity haircuts for testnet-to-mainnet differences,
- market impact from candle-volume participation,
- taker fees,
- liquidation buffer settings,
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
- Model artifacts and accepted model-lab reports must pass financial-sanity checks for finite parameters, coherent probabilities, valid row counts, valid coverage, and bounded risk metrics.
- Repo-facing performance claims must come from real source data with the provenance required by `docs/DATA_PROVENANCE_POLICY.md`.
- Authenticated live/testnet order loops do not trust requested quantity as filled quantity; they require execution fields or a signed order-status reconciliation.
- Authenticated `live --live` startup refuses unverified exchange exposure; it will only resume an existing position if the local bot ledger proves ownership.
- Autonomous stop closes local open positions to avoid stale ledger exposure.
- `reconcile` must be clean before treating the local autonomous ledger as flat or aligned with exchange state.
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
