# Simple AI Trading

<!-- BEGIN GENERATED BADGES -->
[![andrej-karpathy-skills](https://img.shields.io/static/v1?label=&message=andrej-karpathy-skills&color=555&logo=github&logoColor=white)](https://github.com/forrestchang/andrej-karpathy-skills)
<!-- END GENERATED BADGES -->

Simple AI Trading is a Windows-first, testnet-first autonomous day-trading CLI and desktop app for liquid Binance spot and futures markets. It has been expanded from the original single-pair prototype into a diversified runtime that can manage multiple symbols, measure per-symbol liquidity automatically, train/retrain models, run realistic backtests with execution frictions, and expose the same workflows through both the CLI and Windows app.

This software is experimental trading infrastructure. It does not guarantee profit, 1-2% daily returns, or positive ROI. The goal is to make risk, liquidity, execution, and model checks explicit before any non-mainnet order path is used.

## Current Scope

- Multi-asset day trading on Binance testnet or Demo Trading endpoints.
- Default symbols: `BTCUSDC`, `ETHUSDC`, `BNBUSDC`; users can configure any Binance symbol, then `universe` must prove liquidity before use.
- Conservative risk profile by default, with `conservative`, `regular`, and `aggressive` profiles.
- Mandatory diversification controls: minimum eligible assets, single-asset allocation cap, portfolio risk cap, and max open positions.
- Hard loss-budget controls: daily loss, session loss, consecutive-loss lockout, network-interruption halt, and post-reconnect observation cooldown.
- Futures leverage allowed only up to the app-level safety ceiling of `20x`; default is no leverage (`1x`).
- Profit reinvestment is disabled by default. Enabling it prints a warning because compounding amplifies losses as well as gains.
- CPU-only mode is allowed for wider installability, but AI is disabled there and training/backtesting warns that it will be slower.
- Windows GPU acceleration defaults to DirectML via `torch-directml`, which works across AMD, NVIDIA, and Intel DirectX 12 GPUs.
- AI defaults to a local multibillion model identifier (`qwen2.5:7b`) and a minimum 2B-parameter preflight. AI-assisted signal approval requires explicit holdout uplift over the non-AI ML baseline; otherwise AI remains advisory/review-only.

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

The Windows app is a native C++20 Win32 operator workstation inspired by the SuperZip app structure: PowerShell/CMake build, DPI-aware/resizable Win32 layout, DWM dark window chrome, real listbox/combobox/edit/button controls, grouped operator workflows, and generated workflow metadata. The command picker is still generated from the same argparse command contract as the CLI. The parity test `tests/test_ai_runtime_and_parity.py` fails if a CLI command, option flag, positional argument, or choice is not present in the native contract.

The interface is intentionally simple: the app groups workflows by operator intent instead of exposing every internal model check as a separate task. Complex safeguards run in the background and surface as clear states such as blocked, waiting, review required, or stop-and-close.

Startup behavior:

- The app resolves the repo-local `.venv311` Python and sets `PYTHONPATH` before launching CLI commands, so dev builds do not depend on a globally installed package.
- If DirectML/GPU is available, the Compute workflow reports the active backend in the output console.
- If only CPU is available, the app remains usable, shows a warning, and disables AI.
- The app has direct buttons for AI preflight, AI risk review, risk report, model lab, backtest graph, and stop-and-close local autonomous positions.

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
simple-ai-trading model-blueprint --risk-level conservative
simple-ai-trading model-lab --market futures --objective conservative --objective regular --objective aggressive --max-symbols 5
simple-ai-trading ai-review --report data/model_lab/model_lab_report.json
```

`data-sync` writes closed candles, raw exchange snapshots, and typed top-of-book spread/depth rows to SQLite. `backtest`, `backtest-chart`, and `backtest-panel` can now opt into that SQLite evidence with `--execution-db data/market_data.sqlite`, which converts the latest per-symbol bid/ask spread and top-level depth into pessimistic fill assumptions and stores the execution-profile evidence in backtest artifacts.

`backtest-chart` writes an SVG performance chart from the actual mark-to-market equity path produced by the day-trading simulation. The same command appears in the Windows app.

`model-blueprint` exposes the research-backed model and training roadmap as the same CLI/Windows-app parity command. It separates implemented, evidence-only, research, blocked, sandbox, and advisory model families so future model work cannot silently promote AI forecasts, RL policies, or order-book research into executable trading authority without updating tests and docs.

`model-lab` is the cross-symbol optimization workflow. It automatically ranks high-liquidity symbols from exchange ticker/book data, trains the base GPU model across multiple label target/horizon profiles, serializes meta-label take/downsize/skip policy evidence, requires purged chronological walk-forward evidence for selected candidates, applies a selection-risk gate that deflates the selected score by the number of tried model variants, evaluates Lorentzian-neighbor, rational-quadratic-kernel, and technical-confluence hybrid experts, records hybrid ablation scores showing what happens when each expert family is removed, records feature-group ablation scores for the selected advanced feature vector, then replays every accepted objective under symbol-specific execution stress and final-model temporal robustness windows. Backtests and live/autonomous entry paths apply enabled meta-label policies as pre-entry skip/downsize gates only. Use `--market futures` to research long/short futures behavior without changing saved runtime defaults. A symbol is rejected if any required objective fails profitability, drawdown, trade-count, spread, latency, fee, liquidity-crunch, temporal robustness, statistical edge, or selection-risk gates. After individual symbols pass, model-lab also writes `portfolio_risk.json` and rejects the accepted set if combined correlation clusters, effective symbol count, portfolio CVaR, or portfolio drawdown break the risk-level policy. Rejection reports include explicit per-window and portfolio reasons. See [docs/MODEL_RESEARCH_AND_OPTIMIZATION.md](docs/MODEL_RESEARCH_AND_OPTIMIZATION.md) and [docs/MODEL_TRAINING_INSPIRATION.md](docs/MODEL_TRAINING_INSPIRATION.md).

`ai-review` sends a compact, redacted model-lab report to a local structured-output Ollama model and writes `ai_risk_review.json`. It is an advisory risk review with fail-closed output: deterministic model-lab/portfolio failures, failed selection-risk deflation, missing or failed AI-vs-ML uplift evidence when AI is enabled, positive hybrid or feature ablation deltas that show a selected component is hurting the accepted score, missing GPU AI capability, sub-multibillion local model evidence, unavailable providers, or invalid model JSON all produce a veto/review-required result rather than an approval.

`coordinator` reads independent loop heartbeats/status for risk, execution, reconciliation, market data, machine learning, AI, and learning feedback, then emits one operator state: `ready`, `waiting`, `review_required`, or `blocked_execution`. Risk/execution/reconciliation can block execution; market data, ML, and AI block new entries; learning feedback is advisory and feeds future retraining review.

For quick host checks, `model-lab` and `train-suite` accept `--max-candidates N`. This is a smoke/research limiter only; omit it for a full optimization run.

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

`stop` is fail-closed for the local autonomous ledger: it writes `STOPPING` and closes any locally tracked open positions at the latest available mark price, falling back to entry price if no quote is available. `reconcile` reads the signed spot/futures account state, compares exchange exposure against non-paper local open positions, writes `data/autonomous/reconciliation.json`, and exits nonzero on exchange-only, local-only, or quantity-mismatched exposure.

Network interruptions are treated as a recovery state, not as a normal trading iteration. The autonomous loop keeps retrying at the configured cadence, records a heartbeat that says reconciliation is required, and after connectivity returns it first reconciles signed exchange exposure, checks daily/session loss budgets, checks loss streaks, writes an observation heartbeat, and waits through `recovery_cooldown_seconds` before allowing any new entry. If reconciliation finds exchange-only exposure, local-only exposure, or a quantity mismatch, the loop exits fail-closed and does not touch positions that are not represented in the bot ledger.

Closed trades automatically refresh `data/autonomous/learning_feedback.json`. This is the bounded self-improvement loop: it summarizes recurring loss reasons, symbols, sides, loss streaks, and retraining/cooldown review hints. It never edits a live model, loosens risk settings, or changes open positions; it is evidence for the next model-lab/training review.

Signed live-style startup requires a promoted model artifact. A model must carry passing `selection_risk` evidence from the model-lab/training-suite promotion path before `live --live` will use it; stale or hand-written model JSON is rejected before order logic starts. That evidence now includes both a multiple-trials deflated selected score and a two-panel CSCV/PBO-style overfit diagnostic that compares selection ranking with validation ranking. Signed startup also requires `execution_validation` stamped by `model-lab`: symbol-specific liquidity evidence, accepted stress validation, accepted temporal robustness, and accepted portfolio risk. Plain `train-suite` artifacts are research/paper artifacts until model-lab stamps them after the final portfolio gate. Paper mode can regenerate a bad model for experimentation, but it will not silently trade with a stale artifact. Authenticated live mode also refuses in-loop retraining; run `model-lab` again and promote a fresh artifact instead.

## Risk Levels

`conservative` is the default:

- No leverage by default.
- Lower stop-loss capital-at-risk budgets and position caps.
- Longer cooldowns.
- Stricter liquidity/spread thresholds.
- Lower drawdown tolerance.

`regular` and `aggressive` relax thresholds gradually, but still keep leverage capped at `20x`, require diversification, and preserve exchange/testnet safeguards.

Position sizing treats `risk_per_trade` as the maximum equity budget intended to be lost at the configured stop-loss distance, then caps gross notional by max position size, leverage, exchange constraints, and available cash. The CLI, live loop, risk report, and backtester all use the same stop-loss-sized notional calculation.

Hard capital controls are separate from ROI goals. The conservative profile defaults to a `0.60%` daily loss budget, `1.20%` session loss budget, two-loss streak lockout, three consecutive network errors before recovery-halt messaging, and a 60 second post-reconnect observation cooldown. Regular and aggressive raise those limits gradually, but risk reporting blocks live operation when these controls are disabled or dangerously loose.

## Optimization Evidence

Round-level optimization reports with financial charts live under `docs/optimization/`. Round 001 adds market-quality regime features and risk-aware promotion checks; on the deterministic multi-regime futures holdout, the optimized candidate improved from `-0.80%` ROI and `1.88%` max drawdown to `+3.92%` ROI and `0.10%` max drawdown under the same fee/slippage assumptions. See [docs/optimization/round-001-market-quality.md](docs/optimization/round-001-market-quality.md). This is reproducible engineering evidence, not a live-profit guarantee.

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
- Authenticated live/testnet order loops do not trust requested quantity as filled quantity; they require execution fields or a signed order-status reconciliation.
- Autonomous stop closes local open positions to avoid stale ledger exposure.
- `reconcile` must be clean before treating the local autonomous ledger as flat or aligned with exchange state.
- Autonomous post-outage recovery requires reconciliation, hard loss-budget checks, and an observation cooldown before any new entry.
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
