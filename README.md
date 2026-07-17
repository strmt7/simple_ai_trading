# Simple AI Trading

<!-- BEGIN GENERATED BADGES -->
[![License](https://img.shields.io/github/license/strmt7/simple_ai_trading)](https://github.com/strmt7/simple_ai_trading/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/strmt7/simple_ai_trading/ci.yml?branch=main&label=CI)](https://github.com/strmt7/simple_ai_trading/actions/workflows/ci.yml)
[![super-linter](https://img.shields.io/github/actions/workflow/status/strmt7/simple_ai_trading/super-linter.yml?branch=main&label=super-linter)](https://github.com/strmt7/simple_ai_trading/actions/workflows/super-linter.yml)
[![Ruff](https://img.shields.io/github/actions/workflow/status/strmt7/simple_ai_trading/ruff.yml?branch=main&logo=python&label=Ruff)](https://github.com/strmt7/simple_ai_trading/actions/workflows/ruff.yml)
[![Vulture](https://img.shields.io/github/actions/workflow/status/strmt7/simple_ai_trading/vulture.yml?branch=main&logo=python&label=Vulture)](https://github.com/strmt7/simple_ai_trading/actions/workflows/vulture.yml)
[![cocoindex-code](https://img.shields.io/static/v1?label=&message=cocoindex-code&color=555&logo=github&logoColor=white)](https://github.com/cocoindex-io/cocoindex-code)
[![andrej-karpathy-skills](https://img.shields.io/static/v1?label=&message=andrej-karpathy-skills&color=555&logo=github&logoColor=white)](https://github.com/multica-ai/andrej-karpathy-skills)
<!-- END GENERATED BADGES -->

Simple AI Trading is a Windows-first, testnet-first autonomous day-trading CLI and desktop app focused only on the major BTC, ETH, and SOL Binance spot/futures markets. It has been expanded from the original single-pair prototype into a tightly scoped major-asset runtime that measures per-symbol liquidity automatically, trains/retrains models, runs realistic backtests with execution frictions, and exposes the same workflows through both the CLI and Windows app.

## Beta Warning

`0.1.0-beta.1` is a beta research release. Many advanced AI, live-trading, high-frequency archive, optimization, and Windows-app workflows are still incomplete or non-functional in production terms. Signed execution remains testnet/demo-first and real-money mainnet execution is intentionally disabled. Do not rely on this software to protect capital or produce profit.

This software is experimental trading infrastructure. It does not guarantee profit, 1-2% daily returns, or positive ROI. The goal is to make risk, liquidity, execution, and model checks explicit before any non-mainnet order path is used.

## Current Scope

- Major-asset day trading on Binance testnet or Demo Trading endpoints, limited to BTC, ETH, and SOL.
- Polymarket BTC/ETH/SOL 5-minute markets have a [paper-only parity design](docs/POLYMARKET_PAPER_TRADING.md), prospective public-data capture, strict level-2 replay, source-verified model and causal-retry research, label-free unresolved-market scoring, causal resolution-time capital locking, manual paper orders, and one-shot held-out model/AI replay through the same ownership, reconciliation, Pause/Resume, settlement, and fail-closed Stop semantics as Binance paper mode. Continuous strategy coordination and mark-to-market risk evidence remain incomplete; there is no authenticated or live-money authority.
- New Polymarket captures store each exact public message once in bounded checksummed Zstandard frames; replay reconstructs and rehashes event indexes from that source. Legacy evidence remains readable. A same-sample 160,000-message host benchmark measured a 69.607% database reduction with zero raw/event hash differences; this is storage evidence, not trading performance.
- Default symbols: `BTCUSDC`, `ETHUSDC`, `SOLUSDC`; USD-M futures workflows use the matching `BTCUSDT`, `ETHUSDT`, and `SOLUSDT` contracts.
- Unsupported bases, low-liquidity assets, leveraged-token patterns, and lookalike symbols are rejected before data sync, archive ingestion, universe ranking, or optimization.
- Conservative risk profile by default, with `conservative`, `regular`, and `aggressive` profiles.
- Mandatory diversification controls: minimum eligible assets, single-asset allocation cap, portfolio risk cap, and max open positions.
- Hard loss-budget controls: daily loss, session loss, consecutive-loss lockout, network-interruption halt, and post-reconnect observation cooldown.
- Futures leverage is risk-profile driven when futures mode is enabled: `5x` conservative, `10x` regular, `15x` aggressive, with a hard app-level ceiling of `20x`. Spot trading still resolves to `1x`.
- Profit reinvestment is disabled by default. Enabling it prints a warning because compounding amplifies losses as well as gains.
- CPU-only mode is allowed for wider installability, but AI is disabled there and training/backtesting warns that it will be slower.
- Windows GPU acceleration defaults to DirectML via `torch-directml`, which works across AMD, NVIDIA, and Intel DirectX 12 GPUs.
- Training, retraining, tuning, model-lab feature generation/scoring, external-signal scoring, probability-temperature calibration, threshold selection, backtest simulation, and backtest-panel feature generation use GPU-first `auto` whenever the caller does not explicitly select a backend. Training, scoring, and calibration artifacts record requested/resolved backend evidence; feature builders fall back to the original CPU path if a GPU tensor operation is unavailable. Hybrid-candidate backtest scoring also uses the tensor backend for Lorentzian nearest-neighbor, rational-quadratic-kernel, and technical-confluence models when supported. LightGBM resolves non-CPU training through OpenCL and lets the installed driver select the device by default instead of assuming platform/device `0:0`; operators can set both `SIMPLE_AI_TRADING_OPENCL_PLATFORM_ID` and `SIMPLE_AI_TRADING_OPENCL_DEVICE_ID` together when explicit selection is required.
- AI defaults on, but it fails closed without an accepted local multibillion model. A July 2026 audit found that the old v6 benchmark leaked expected actions through case names, so that evidence is revoked. Fresh v7 prompts exclude and hash-bind case labels: Qwen3 8B scored `9/11`, while Fin-R1 8B, Qwen3.5 9B, and Fino1 8B each scored `8/11`; none passed or has order authority. AI also needs a separate paired after-cost uplift test with at least 30 non-tied outcomes across 90 contiguous days before it can be credited with edge. Exact negative evidence and local model identities are under [`docs/ai/risk-review/latest/`](docs/ai/risk-review/latest/).
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

`simple-ai-trading ai` also reports inferred model size and measured free VRAM. Required GPU headroom is fail-closed: an unknown value, a sub-multibillion model, or a CPU-only backend blocks AI approval. On Windows AMD hosts, the check uses the driver's 64-bit dedicated-memory value and current WDDM dedicated usage instead of the legacy 32-bit WMI field.

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

The Windows app is a native C++20 Win32 operator workstation inspired by the SuperZip structure: DPI-aware layout, native controls, dark window chrome, and Overview, Trading, Research, Risk, Data, System, and Settings pages. Overview keeps Start, Pause, Stop + Close, risk profile, execution mode, leverage, AI, and reinvestment controls together. CLI commands and options are generated from `argparse`; one shared taxonomy places every command in a deliberate workflow group. Generation and parity tests fail on missing, duplicate, or stale entries, while Settings retains the complete expert catalog.

The interface is intentionally simple: the app groups workflows by operator intent instead of exposing every internal model check as a separate task. Complex safeguards run in the background and surface as clear states such as blocked, waiting, review required, or stop-and-close.

Startup behavior:

- The app resolves the repo-local `.venv311` Python and sets `PYTHONPATH` before launching CLI commands, so dev builds do not depend on a globally installed package.
- If DirectML/GPU is available, the Compute workflow reports the active backend in the output console.
- If only CPU is available, the app remains usable, shows a warning, and disables AI.
- `AI on (gated)` means AI is configured, not loaded or approved. Green `AI GPU resident` status requires a post-inference Ollama `/api/ps` check bound to the exact model digest with reported VRAM residency; unloaded, CPU-only, ambiguous, or malformed runtime evidence fails closed. DirectML model compute and Ollama residency are separate contracts.
- The desktop AI toggle and backend model workflows share one policy. The Windows app emits `--enable-ai` or `--disable-ai` from its visible state; direct CLI runs use either explicit switch or inherit the persisted AI setting. Enabled AI still requires the same local-model, GPU-headroom, provenance, residency, and measured-uplift gates.
- The app has direct buttons for Stop + Close, Pause, Reconcile, Positions, and Risk Review. Normal workflow cards are distinct paper, research, graph, data, and settings tasks so the UI does not present multiple similar buttons for the same action.
- The bottom status bar shows the shared CLI API-budget summary. It refreshes opportunistically rather than constantly: automatic refresh is capped to the 60-120 second band and defaults to 90 seconds, while command-completion updates use cached status.
- `tools\smoke_native_windows_ui.ps1` launches the app in dry-run mode, walks every workflow page, clicks the dashboard cards and safety buttons, and then performs a real Compute smoke unless `-SkipRealCompute` is passed.
- `tools\capture_native_windows_app.ps1` creates a DPI-aware PNG artifact and retries complete-frame pixel-health checks after deterministic redraw; undersized or blank captures fail.
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
simple-ai-trading polymarket-record --duration-seconds 660 --database data/polymarket-paper.duckdb --progress-path data/polymarket-recorder-progress.json
simple-ai-trading polymarket-resolve --database data/polymarket-paper.duckdb
simple-ai-trading polymarket-features --database data/polymarket-paper.duckdb
simple-ai-trading polymarket-model --database data/polymarket-paper.duckdb --output data/polymarket-model.json
simple-ai-trading polymarket-verify --artifact data/polymarket-model.json --database data/polymarket-paper.duckdb --output data/polymarket-source-verification.json
simple-ai-trading polymarket-publish --artifact data/polymarket-model.json --database data/polymarket-paper.duckdb
simple-ai-trading polymarket-paper --database data/polymarket-paper.duckdb --action run-model --artifact data/polymarket-model.json --source-verification data/polymarket-source-verification.json --output data/polymarket-paper-model-run.json
simple-ai-trading archive-sync --symbol BTCUSDC --interval 1s --cadence monthly
simple-ai-trading archive-sync --symbols BTCUSDT,ETHUSDT,SOLUSDT --market futures --interval 1s --cadence daily --start-period 2024-01-01 --end-period 2024-01-31 --plan-only --require-checksum --json
simple-ai-trading archive-sync --symbols BTCUSDT,ETHUSDT,SOLUSDT --market futures --interval 1s --cadence daily --start-period 2024-01-01 --end-period 2024-01-31 --require-checksum --progress-path data/archive-sync-progress.json
simple-ai-trading data-health --interval 1s --market spot --min-rows 1000000 --require-verified-checksum --json
simple-ai-trading tick-archive-sync --symbols BTCUSDT,ETHUSDT,SOLUSDT --data-types bookTicker,trades,bookDepth --full-history --plan-only --plan-output docs/microstructure/availability.json
simple-ai-trading tape-depth-train --symbol BTCUSDT --window-days 365 --horizon-seconds 60 --model-profile regularized --feature-set full --compute-backend directml
simple-ai-trading tape-depth-design --risk-level conservative --sampled-count 24 --seed 20260710 --output data/tape-depth-experiment-design.json
simple-ai-trading tape-depth-study --design data/tape-depth-experiment-design.json --symbols BTCUSDT,ETHUSDT,SOLUSDT --compute-backend directml --resume --output-dir data/tape-depth-study
simple-ai-trading tape-depth-prequential --symbols BTCUSDT,ETHUSDT,SOLUSDT --plan-only
simple-ai-trading tape-depth-prequential --symbols BTCUSDT,ETHUSDT,SOLUSDT --study-stage screening --max-folds 4 --model-profile regularized --feature-set core --compute-backend directml --output-dir data/tape-depth-regularized-core
simple-ai-trading tape-depth-select --design data/tape-depth-experiment-design.json --report data/tape-depth-regularized-core/report.json --report data/tape-depth-balanced-full/report.json --output data/tape-depth-selection.json
simple-ai-trading tape-depth-prequential --symbols BTCUSDT,ETHUSDT,SOLUSDT --study-stage confirmation --selection-lock data/tape-depth-selection.json --compute-backend directml --output-dir data/tape-depth-confirmation-run
simple-ai-trading tape-depth-confirm --selection data/tape-depth-selection.json --report data/tape-depth-confirmation-run/report.json --output data/tape-depth-confirmation.json
simple-ai-trading tape-depth-execution-confirm --warehouse data/microstructure.duckdb --design docs/model-research/tape-depth/confirmation-design.json --availability docs/microstructure/availability.json --output-dir data/tape-depth-execution-confirmation --resume
simple-ai-trading microstructure-train --symbol BTCUSDT --candidate-only --stop-loss-bps 25 --take-profit-bps 40
simple-ai-trading microstructure-prequential --input data/microstructure-model.json
simple-ai-trading microstructure-promote --input data/microstructure-model.json
simple-ai-trading microstructure-shadow --input data/microstructure-model.json
python tools\optimization_round.py --round-id round-004 --market spot --symbols ETHUSDT --interval 1s --require-prefilled-data --min-data-rows 70000 --require-verified-checksum
python tools\optimization_round.py --round-id round-btc-eth-sol-futures-1s --market futures --quote-asset USDT --objective conservative --promotion-grade --min-promotion-data-years 2 --require-gpu
simple-ai-trading model-blueprint --risk-level conservative
simple-ai-trading model-lab --market futures --quote-asset USDT --interval 1s --market-db data/market_data.sqlite --require-db-data --objective conservative --objective regular --objective aggressive --max-symbols 3
simple-ai-trading ai-forecast-benchmark --model-size base --backend directml --bootstrap-source
simple-ai-trading ai-review --report data/model_lab/model_lab_report.json
simple-ai-trading ai-uplift --starting-capital 1000 --market-db data/market_data.sqlite
```

`data-sync` writes closed candles, raw exchange snapshots, typed top-of-book spread/depth rows, sync runs, archive-file ingestion records, and API-budget snapshots to the single SQLite market database. With `--full-history`, it pages backward through the venue's maximum allowed kline page size until the exchange returns no older data; normal bounded syncs remain efficient for incremental refreshes. `archive-sync` ingests official Binance public archive ZIPs directly into the same SQLite database and records source URL, rows, byte count, SHA-256, Binance sidecar checksum SHA-256 when available, checksum status, and ingestion status. Spot `1s` uses official kline archives. USD-M futures `1s` uses official `aggTrades` archives and deterministically aggregates real trades into one-second OHLCV candles because Binance does not publish USD-M futures `1s` kline archives. Seconds without trades are represented as carry-forward no-trade candles with zero volume and zero trade count, so coverage is continuous without inventing traded volume. Raw aggregate trades are not duplicated by default; `--store-raw-agg-trades` is an explicit event-time research opt-in. Long second-level backfills should be staged with `--start-period`, `--end-period`, and `--plan-only`; the plan reports listed, filtered, selected, first, and last archive periods plus official S3 byte estimates when available before any ZIP is downloaded. Active downloads and inserts update `data/archive-sync-progress.json` atomically unless another `--progress-path` is supplied. Non-plan archive downloads are blocked above `--max-planned-gb` (`50` by default) so an accidental multi-year command cannot start a huge ingest without an explicit bounded window or raised cap. It accepts a single `--symbol`, comma-separated `--symbols`, or `--top-symbols N`, but all paths reject anything outside BTC, ETH, and SOL quoted in USDC or USDT. `--top-symbols` still ranks the supported majors from current exchange metadata before ingestion so bulk research uses current liquidity rather than stale assumptions. Use `--require-checksum` for promotion-grade archive ingestion; checksum mismatches fail before candle rows are written. Sync artifacts include kline request counts, rows received, coverage ratio, gap count, and Binance rate-limit telemetry captured from response headers when the exchange provides it. `api-budget` reads or refreshes that status; cached samples auto-refresh only when stale, defaulting to 90 seconds. `backtest`, `backtest-chart`, and `backtest-panel` can opt into SQLite evidence with `--execution-db data/market_data.sqlite`, which converts the latest per-symbol bid/ask spread and top-level depth into pessimistic fill assumptions and stores the execution-profile evidence in backtest artifacts.

`data-health` audits the SQLite market database before training or optimization. It reports row counts, UTC spans, span-years, expected rows, gap count, coverage ratio, archive-file status counts, and checksum-status counts for every stored symbol/market/interval or for an explicit symbol batch. It exits nonzero when minimum rows, span, coverage, gap, archive-error, checksum-mismatch, or `--require-verified-checksum` gates fail; bounded windows count missing bars at the requested UTC start and end boundaries, not only gaps between stored rows. Promotion-grade optimization now uses the single fail-closed contract `tools/optimization_round.py --promotion-grade`, which forces the exact BTC/ETH/SOL trio for the selected quote asset, forces `1s`, disables network backfill, requires verified archive checksums, requires zero missing-second gaps, enforces the configured minimum stored history span, and writes `promotion_grade_contract` to `report.json`. If that contract or the critical-analysis layer fails, the tool exits nonzero and the artifacts cannot be treated as performance evidence. Futures `1s` optimization is allowed only with prefilled `aggTrades`-derived candles; otherwise the tool fails instead of calling a nonexistent futures `1s` kline endpoint. The tool defaults to `--model-candidates 32`, evaluating the broad profitability-search prefix of cost-aware label horizon/target/model regularization candidates per symbol before the final holdout; pass a lower `--model-candidates` value only for smoke diagnostics. During multi-candidate rounds, the optimizer builds one advanced feature matrix per symbol/feature-shape, then relabels it per candidate so second-level backtests do not recompute the same features 32 times. `report.json` and `backtest-metrics.csv` record `model_candidate_count`, `model_selected_candidate`, and `model_selection_score`. Signed live startup and `risk --live --model` now require promoted `TrainedModel` artifacts to carry multi-candidate evidence; when the resolved runtime backend is DirectML/CUDA/ROCm/MPS, they also require accelerator evidence for training and probability calibration. The round writes `round-status.json` during long runs, including per-symbol phases for data health, SQLite load, feature cache generation/reuse, candidate relabeling, DirectML training, probability-temperature calibration, threshold calibration progress, holdout scoring, and artifact streaming, plus `data-health.json` beside the graph tables. When hard data gates are active and symbols are not supplied explicitly, optimization now over-samples the live liquidity-ranked universe and keeps the first requested count that also pass local data-health; skipped candidates are recorded as `selection_health_rejections` in the round report. SVG charts are deterministic visual summaries while the full-resolution timeline CSVs remain the source of truth, avoiding multi-megabyte chart bloat without weakening evidence. Each round report records both configured leverage and effective leverage, so spot evidence cannot be misread as a futures-leverage backtest.

Liquidity-session controls do not assume that "day trading hours" are fixed forever. The active defaults use observed bar volume by symbol and historical clock bucket in backtests, the `live` command, and the autonomous decision function, so low-liquidity holidays, exchange maintenance, overnight crypto liquidity drops, and changing venue participation can be reflected by the data instead of a static UTC start/end hour.

`backtest-chart` writes an SVG performance chart from the actual mark-to-market equity path produced by the day-trading simulation. When timestamps are present, the chart labels the UTC start/end dates and the simulated duration in days/years instead of presenting an unlabeled sample index. The same command appears in the Windows app.

`model-blueprint` exposes the research-backed model and training roadmap as the same CLI/Windows-app parity command. It separates implemented, evidence-only, research, blocked, sandbox, and advisory model families so future model work cannot silently promote AI forecasts, RL policies, or order-book research into executable trading authority without updating tests and docs.

The separate `microstructure-action-value-v16` path uses real Binance USD-M
book-ticker and trade archives to build causal one-second L1/tape features. Its
promotion lifecycle is deliberately staged: candidate training, complete
rolling-refit prequential evidence with the terminal period sealed, hash and
row-level evidence verification, one-use terminal reservation, locked
rolling-refit terminal validation, an expiring deployment refit, then a locked
six-hour no-order public-feed shadow. Shadow signals wait for the full modeled
latency before virtual entry, recheck top-of-book participation, apply actual
bid/ask and fee/trigger costs, censor only the planned capture tail, and reject
any feed gap, feature reset, deadline miss, inference failure, stale pending
entry, forced close, or order submission. Capture schema
`binance-usdm-l2-v3` hashes the original stream, synchronized stream, REST
snapshot, manifest, shadow trades, and report. The former
`microstructure-train --evaluate-terminal` shortcut is disabled. Terminal,
refit, shadow, and accepted-runtime loaders independently reject missing or
drifted evidence. A refit produces only `shadow_candidate`; only a passing
`microstructure-shadow` run produces `accepted`. No v16 artifact is currently
accepted or claimed profitable; see
[Model Research and Optimization](docs/MODEL_RESEARCH_AND_OPTIMIZATION.md).

V16 retains the linear cash-return labels introduced in v15 at the observed
executable quotes. A long earns `exit_bid / entry_ask - 1`; a short earns
`1 - exit_ask / entry_bid`. Taker fees and the separate all-trade slippage
stress are charged on the actual entry and exit notionals, so their cost is
`cost_per_side * (1 + exit_notional / entry_notional)`, not a flat two-leg
approximation. Trigger slippage remains a distinct adverse stop/take exit-price
adjustment. The default CLI stress is 1 bps per side in addition to the modeled
taker fee, and every rebuild/promotion/shadow path is bound to that artifact
value.
Feature contract `l1-tape-causal-v8` retains causal 30/60-minute trend,
volatility, range, path-efficiency, liquidity, UTC weekly-cycle, and weekend
context, and adds bid/ask L1 quote depth, relative L1 depth, and signed pressure
against opposing depth. Offline DuckDB construction and live streaming use the
same 107-feature order and require 3,600 clean seconds before inference.
Consequently, the default no-order promotion capture is 25,260 seconds: one
hour of warmup, six complete evaluated hours, and one minute of tail margin.
Because Binance's official historical BBO product contains 320 days, the exact
BBO lane defaults to a 240-observed-day promotion floor; a 365-day floor would
make promotion impossible. Multi-year predictiveness remains the separate
checksummed trade/depth lane and is not mislabeled as exact historical BBO.
The official compact tick plan is tracked at
[`docs/microstructure/availability.json`](docs/microstructure/availability.json):
trade data spans multiple years through 2026-07-09, while official BBO archives
stop on 2024-03-30. The app does not disguise coarse `bookDepth` percentage bands
as a current best bid/ask history.

Full-history tick sync snapshots the official Binance listing before ingestion
and again after ingestion. The immutable snapshot binds both the ZIP and its
`.CHECKSUM` object's S3 ETag, `LastModified`, and byte size. It exits nonzero
until both inventories are identical and every listed daily archive has a
matching SHA-256-bound warehouse manifest. Model datasets require a corpus
certificate covering the exact requested UTC days and quote/trade/depth
products; a missing day cannot silently become zero flow. Certification also
reconciles physical raw/derived partition counts and time bounds, the 100 ms BBO
path, and coarse-depth band groups. Repeat syncs reuse an archive only when both
S3 object versions, verified source/sidecar hashes, schema, and physical
partitions remain unchanged. A physically damaged completed partition is
transactionally rebuilt from the official ZIP instead of being returned as
`skipped`; a failed repair leaves the previous partition uncertified. Binance's
own missing `bookDepth` dates are reported as provider gaps and are never
synthesized or silently forward-filled. Candidate research
has a fingerprinted Latin-hypercube design and chronological successive-halving
contract to avoid retraining every correlated variant at full scale; integration
with the sealed prequential selector remains in progress. Short-window survivors
remain research-only and cannot consume terminal evidence or gain trading
authority.

"Complete history" always means complete through the certificate's explicit
UTC cutoff: markets continue publishing new files, so no honest system can call
an open-ended archive permanently finished. Corpus certificate v3 distinguishes
three states without inference: a listed and checksum/partition-verified file,
an explicitly recorded provider-side `bookDepth` absence, and a missing or
invalid local partition. Only the first two can pass, and the second is allowed
only for `bookDepth`; missing listed BBO or trade data remains fatal. Run
`tick-corpus-audit --output data/tick-corpus-certificate.json` after sync to
write the machine-verifiable certificate. Add
`--strict-book-depth-calendar` when an experiment requires an uninterrupted
depth calendar and should reject even provider-proven absences.

`tape-depth-train` builds a bounded, purged LightGBM direction/return/uncertainty
ensemble from checksummed one-second trade tape and causally joined coarse depth.
Its exact one-second clock keeps real no-trade seconds as an explicit zero-volume,
zero-count row carrying only the last verified trade reference; `trade_observed`
and trade-age features distinguish those rows, and a verified daily trade
manifest is required for every requested date so a missing archive cannot look
like an inactive market.
The v4 matrix also joins lagged BTCUSDT, ETHUSDT, and SOLUSDT returns at or
before each feature second. Peer context has an explicit availability mask,
never reads through the forecast horizon, and is checksum-bound only through
the last observable feature timestamp. It is an ablation candidate, not a
claim that cross-asset context has predictive value.
Its label is the future real trade-reference return after the configured latency,
not an executable fill or after-cost PnL. The artifact is therefore restricted to
`research_candidate` or `rejected`, records `trading_authority=false` and
`execution_claim=false`, and cannot be loaded as an accepted trading model. It is
the long-history forecasting lane; the shorter exact-BBO lifecycle and current
no-order shadow remain mandatory before any execution claim.

The v8 forecast contract eliminates test-distribution tuning: return-magnitude
sample weights use a scale learned only from exact float64 training targets.
The calibration segment freezes the probability transform, direction baseline,
forecast-magnitude gate, and maximum uncertainty width. Evaluation data cannot
change any of them, and replay refits/recomputes every value before accepting the
artifact. Conservative uses calibration quantiles `0.95/0.95/0.75` for
magnitude, directional-confidence margin, and interval width; regular uses
`0.90/0.90/0.90`; aggressive uses `0.80/0.80/0.98`. The probability floor is
`0.5` plus the selected quantile of absolute calibrated probability distance
from `0.5`, so differently sharp models are compared by pre-evaluation rank
rather than an arbitrary fixed probability. A forecast is selected only when
magnitude, calibrated direction, and uncertainty agree. Every compressed
prediction row carries the complete policy. Long/short/action counts are
reported, but no quota forces an entry against risk analysis. The Brier and
majority baselines are frozen from calibration prevalence rather than reading
evaluation labels.

The v8 training backend also makes numerical repeatability an explicit model
contract. CPU fallback enables LightGBM deterministic column-wise training;
OpenCL keeps AMD/NVIDIA acceleration but uses FP64 histogram accumulation, the
upstream mitigation for GPU run-to-run variance. Two consecutive real-data AMD
retrainings produced identical model and prediction SHA-256 fingerprints. This
is a same-runtime repeatability result, not a claim that different LightGBM
builds or GPU architectures will fit byte-identical trees.

`tape-depth-prequential` is the multi-year rolling evidence path. Its default
fold has 730 calendar days of training, separate 30-day tuning and calibration
periods, and a non-overlapping 90-day screening evaluation period. Decisions are
sampled every 20 seconds, while every feature window and target still comes from the
one-second source table. Each fold persists the exact serialized model, a
deterministic compressed row-level prediction table, source/dataset/model hashes,
and a fold metrics row. The final deterministic SVG uses real UTC dates and
shows AUC, rank IC, and gross forecast return with explicit baselines and a
no-fill/no-cost caveat. It never sums overlapping forecast horizons into ROI and
cannot grant trading authority. Full-corpus evidence is not claimed until the
active checksummed acquisition and every planned fold complete.

Gross-forecast survivors must also pass `tape_depth_execution` against the exact
100 ms BBO overlap. This diagnostic suppresses overlapping same-symbol
positions, uses causally available ask-to-bid or bid-to-ask paths, subtracts
two-sided taker fees and explicit stress slippage, rejects stale/crossed/missing
quotes, and enforces an L1 participation cap. It never infers maker fills or
queue priority. The reproducible v8 replay of the exploratory 2024-03-15
discovery reached `+5.5730` bps mean trade-reference gross on 15 conservative
signals. Overlap suppression left 6 scheduled entries, the 10% L1 cap rejected
2, and the remaining 4 averaged `-5.6385` bps net at 5 bps per side. It is
rejected, not a profitability claim. The discovered
20-second candidate is frozen for three untouched dates in the hash-bound
[`confirmation-design.json`](docs/model-research/tape-depth/confirmation-design.json);
those dates were committed before their archives were evaluated.

`tape-depth-execution-confirm` executes that exact design as a separate workflow.
It writes a deterministic plan, one no-overwrite checkpoint per UTC date, the
serialized FP64 model, row-level compressed predictions, exact quote-path rows,
and a final weighted gate report. `--resume` verifies every existing fingerprint
and file hash; it never silently recomputes or replaces an observed period.

The runner computes the causal one-second-derived matrix once per remaining
symbol and retains only 20-second decision rows, bounded by
`--maximum-cached-rows` (`15,000,000` by default). Fold slices share that matrix
in memory but receive their own exact source-manifest binding and dataset hash.
This removes repeated multi-year window calculations without allowing future
features or labels into an earlier fold.

Completed matrices are cached by default inside the same DuckDB warehouse in a
feature-versioned column table. A cache key binds the exact requested UTC range,
target/cadence contract, ordered feature contract, and complete target/peer
source evidence. Writes are transactional; loads reconstruct the matrix and
recompute its dataset fingerprint before use. Cache misses, hits, keys, and
source fingerprints are checkpointed in `dataset-cache-events.json` and the
final report. Use `--no-dataset-cache` only when disk writes are intentionally
undesired; `--maximum-cached-rows` is enforced and forwarded by both CLI and app.

Long rolling runs persist an atomic fold summary after every completed model.
`--resume` reopens only a matching plan, reloads each serialized model, parses
each compressed prediction table, recomputes its metrics and fingerprints, and
skips a fold only when every binding and file hash matches. A changed, missing,
or path-escaping artifact blocks resume instead of silently mixing runs.

Forecaster capacity is independent of trading risk tolerance. The explicit
`regularized`, `balanced`, and `expressive` model profiles are prediction trials
that must be counted and compared on earlier rolling evidence. Conservative,
regular, and aggressive remain execution-policy settings for drawdown, cooldown,
leverage, and sizing; changing risk level with the same model profile produces
the same serialized predictor.

`core`, `tape_derived`, `cross_asset`, and `full` are ordered feature-set
ablations. `cross_asset` adds point-in-time peer and BTC-anchor returns to the
local derived tape, while `full` adds coarse depth last. The model artifact
serializes the exact ordered input names while its dataset fingerprint still
binds the complete source matrix. Every added group must beat the simpler
earlier-fold baseline and confirm later; adding columns is not treated as
automatic improvement.

Tape/depth model selection is physically split into screening and confirmation.
Every candidate run must use `--study-stage screening`, start at fold zero, and
declare 4, 6, 8, or 10 non-overlapping screening folds while leaving at least
two later folds untouched. `tape-depth-design` freezes horizon, decision cadence,
maximum depth age, model profile, feature set, and risk profile before screening.
`tape-depth-select` requires every design candidate exactly once, verifies common
fold boundaries/full-corpus coverage and identical datasets among candidates
that share one dataset configuration, recomputes all candidate metrics, and runs a
complete symmetric-fold forecast-rank PBO diagnostic with a fail-closed 0.20
limit. This is not a PnL/Sharpe PBO or profitability claim. It then writes a
source-report/design-bound winner lock. A confirmation run accepts that lock,
derives the winning horizon, cadence, depth age, profile, feature set, and
terminal fold boundary from it, and fails if the corpus, modeling configuration,
winner, design, or source reports changed.
`tape-depth-study` is the checkpointed operator for this screening stage. It
runs candidates sequentially so GPU jobs do not compete, preserves per-candidate
fold checkpoints, verifies complete reports before `--resume` skips them, emits
nested progress, and invokes the same design-bound selector only after every
declared report exists. It never accesses the confirmation suffix itself.
`tape-depth-confirm` then verifies that the single supplied report contains the
entire untouched suffix. Before either stage trusts a report, it reloads every
serialized model and compressed row-level prediction table, verifies contained
paths and file hashes, and recomputes fold fingerprints, timestamps, metrics,
status, and aggregates against `plan.json`. A failed winner rejects the study;
no runner-up is evaluated. The strongest result is
`confirmed_forecast_candidate`; it is still forecast evidence with no
profitability claim or execution authority.

`ai-forecast-benchmark` is a no-order research workflow for a hash-pinned
financial time-series foundation model. It requires exact BTCUSDT, ETHUSDT, and
SOLUSDT post-pretraining archive coverage, verifies executable source and model
weights before loading, evaluates a random-walk baseline, fits amplitude only
on the earlier half, scores the later half, and records UTC-day bootstrap
uncertainty. DirectML inference runs in bounded child processes because a model
or device fault must not freeze the app. The latest Kronos-base result was
rejected and has no trading authority; its raw table, report, manifest, and
forecast-error graph are under
[`docs/ai/foundation/latest`](docs/ai/foundation/latest/README.md).

`model-lab` is the BTC, ETH, and SOL model-selection workflow. It ranks eligible
markets from exchange liquidity data and compares supervised target definitions,
interpretable signal models, and compact neural ensembles. Training, inference,
probability calibration, and threshold selection use the configured compute
backend; serialized feature and model contracts are shared by backtests, the CLI,
the Windows app, and live scoring.

Every candidate is evaluated with purged chronological walk-forward folds,
multiple-testing adjustment, component and feature ablation, symbol-specific
execution-cost stress tests, temporal robustness checks, and portfolio CVaR and
drawdown controls. Evidence includes source coverage, model identity, trade-level
P&L, fees, win rate, profit factor, drawdown, liquidation and exit-path metrics,
and explicit rejection reasons. Sparse-label candidates fail before training, and
rejected models are configured to prohibit new entries.

Use `--interval 1s --market-db PATH --require-db-data` for archived second-level
market data; the command fails rather than falling back to API candles when the
requested rows are absent. `--full-history` applies only to API research. Detailed
model families, feature contracts, validation rules, and telemetry are documented
in [model research](docs/MODEL_RESEARCH_AND_OPTIMIZATION.md), [training research](docs/MODEL_TRAINING_INSPIRATION.md),
and the [open-source gap analysis](docs/OPEN_SOURCE_GAP_ANALYSIS_AND_RULESET_2026-07-05.md).

`ai-review` sends a compact, redacted model-lab report to a local structured-output Ollama model and writes a versioned, fail-closed `ai_risk_review.json`. Review v4 binds the source evidence, local weight digest, structured response, and exact post-inference provider residency. Deterministic risk/data failures, missing paired AI uplift, harmful ablations, CPU-only or unproven AI execution, unavailable providers, and invalid output all veto approval. The complete evidence contract is in [AI model selection](docs/AI_MODEL_SELECTION.md).

`coordinator` reads independent loop heartbeats/status for risk, execution, reconciliation, market data, machine learning, AI, and learning feedback, then emits one operator state: `ready`, `waiting`, `review_required`, or `blocked_execution`. Risk/execution/reconciliation can block execution; market data, ML, and AI block new entries; learning feedback does not mutate live positions, but it can block future model promotion when repeated symbol losses have not recovered in stress and temporal validation.

For quick host checks, `model-lab` and `train-suite` accept `--max-candidates N`. This is a smoke/research limiter only; omit it for a full optimization run.

Financial-sanity checks now run on promoted model artifacts, generated backtest results, model-lab reports, and AI-review prechecks. They block impossible model dimensions, non-finite parameters, nonsensical probability settings, missing or poor promoted-model probability calibration evidence, impossible coverage/drawdown values, accepted reports with zero rows, missing or failed accepted-outcome data coverage, incomplete Binance source/provenance evidence, missing truth-basis evidence, nonpositive model-row/candle counts, accepted outcomes without real accepted purged walk-forward folds, accepted outcomes without passed selection-risk evidence, accepted stress/temporal reports without measured scenario/window/statistical-edge evidence, accepted outcomes without an accepted portfolio-risk report, accepted portfolio reports whose symbol evidence is missing/duplicated/mismatched, accepted AI uplift without complete baseline/AI/delta metrics, model-size and SHA-256 evidence bindings, contiguous fixed-period matched returns spanning at least 90 days, and a positive moving-block-bootstrap confidence bound, raw backtest liquidation flags/events/losses/exits, and internally inconsistent backtest accounting identities for cash, fees, trade P&L, trade counts, exposure, exit reasons, win rate, path-quality metrics, trade-level timestamps/prices/returns, and equity-curve drawdown. AI uplift policy fields can tighten these minimums but cannot weaken the built-in floors of 30 non-tied pairs, 90 days, a 5% sign test, 2,000 resamples, and 95% confidence. Objective acceptance also consumes the generated-backtest financial-sanity report directly, so no optimizer, model-lab, stress, temporal, or ranking path can promote an incoherent backtest by relying on positive P&L alone. Promoted models must carry calibrated probability evidence with Brier score `<=0.35` and expected calibration error `<=0.20`. These checks are intentionally conservative and do not turn a backtest into an investment recommendation.

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

`stop` is fail-closed and single-writer. The control command records `STOPPING`; only the active execution worker may submit closes, preventing a second CLI or Windows control process from racing it with a duplicate order. Paper mode closes only journal-proven bot inventory against a freshly observed real Binance bid/ask, charges configured fees, preserves partial remainders, and refuses to invent a fill from a cached mark or entry price. Missing worker, broker, quote, integrity, or ownership evidence leaves the position visibly open and returns a nonzero result when no active worker can acknowledge an open position. Signed mode submits closes only when live ownership evidence is complete. Its ledger recomputes after-cost P&L from the exchange-reported fill and the verified conservative taker-fee rate, including proportional entry fees for partial closes. `reconcile` reads the signed spot/futures account state, compares exchange exposure against verified bot-owned non-paper local open positions, writes `data/autonomous/reconciliation.json`, and exits nonzero on exchange-only, local-only, quantity-mismatched, unverified local live exposure, malformed local open-position ledger files, or malformed account payloads. Futures reconciliation requires a signed account payload with a `positions` list; spot reconciliation requires `balances`. Signed `live --live` startup also uses that reconciliation gate; it refuses to manage pre-existing exchange exposure unless it matches a bot-owned ledger position with a bot client order id and exchange fill/acknowledgement evidence. Spot closes require filled or partially filled exchange status; futures reduce-only closes may also use a bot-owned exchange-acknowledged order ID. Signed CLI opens now use deterministic `sait-o-*` client order ids and signed closes use `sait-c-*`, so a restarted session can prove which exchange exposure is bot-owned before touching it.

The worker writes `STOPPED` only after terminal reconciliation proves that the
local ledger and the active paper journal or signed account have no remaining
exposure. Any open position, partial close, journal blocker, integrity error, or
unavailable terminal reconciliation leaves control state at `STOPPING`. The CLI
and shell `close` commands cannot delete either paper or live ledger rows; use
`autonomous stop` so venue closure and ownership reconciliation stay coupled.

Autonomous execution has a venue-neutral execution-lifecycle preflight in `simple_ai_trading.execution_lifecycle`. It is separate from model entry risk: a risk-policy block can prevent new entries while still allowing verified bot-owned emergency closes. Paper mode additionally uses an append-only hash-chained order journal; unresolved orders block new exposure, while close capability remains available when ownership and integrity still reconcile. Ledger corruption, missing required reconciliation, external exchange exposure, quantity mismatch, or unverified bot ownership block both opens and closes. Authenticated autonomous startup, operator stop, risk-close, auto-close-threshold closes, and every live open call use the same lifecycle vocabulary before order submission. `autonomous start --live` is no longer blocked by the old "not wired" guard, but it remains non-mainnet only and still requires credentials, promoted model readiness, API-budget headroom for startup/open, reconciliation, and bot-owned order evidence.
Autonomous stop, risk-close, and auto-close-threshold paths preserve partially
filled exchange closes: the filled quantity is appended to the closed-trade
ledger, the unfilled quantity remains in `open_positions.json` with
`PARTIALLY_FILLED` evidence, and the close/run report is marked incomplete until
the remainder is closed or reconciled.

Network interruptions are treated as a recovery state, not as a normal trading iteration. The `live` loop keeps retrying market-data reads and records `market_error_retry` events instead of entering on stale data. After connectivity returns, it records a clean recovery observation, waits through `recovery_cooldown_seconds` when configured, and skips fresh entries for that observation step. The autonomous loop adds signed reconciliation before resume: it records a heartbeat that says reconciliation is required, reconciles exchange exposure, checks daily/session loss budgets, checks loss streaks, writes an observation heartbeat, and skips that iteration before allowing any new entry. If reconciliation finds exchange-only exposure, local-only exposure, unverified local live exposure, malformed account payloads, malformed local ledger files, or a quantity mismatch, the autonomous loop exits fail-closed and does not touch positions that are not represented in a verified bot-owned ledger row.

Closed trades automatically refresh `data/autonomous/learning_feedback.json`. This is the bounded self-improvement loop: it summarizes recurring loss reasons, symbols, sides, loss streaks, and retraining/cooldown review hints. It never edits a live model, loosens risk settings, or changes open positions. The next model-lab run consumes it automatically and blocks a symbol with repeated losses unless the new candidate shows positive stress and temporal recovery evidence.

Signed live-style startup requires a promoted model artifact. A model must carry passing `selection_risk` evidence from the model-lab/training-suite promotion path before `live --live` will use it; stale or hand-written model JSON is rejected before order logic starts. That evidence includes a multiple-trials deflated selected score, a two-panel CSCV/PBO-style overfitting diagnostic, and one fingerprinted terminal-holdout simulation of the exact final model after hybrid and meta-label fitting. Full-fit fallback and probability-inversion attempts are selected only on the earlier chronological selection sample, and all internal and hybrid-profile attempts count as explicit trials. Candidate search, walk-forward screening, ablation, and policy fitting cannot access the purged terminal suffix. Before that suffix is opened, `model-lab` or `train-suite --symbol SYMBOL` atomically reserves its symbol-, market-, and risk-objective-specific timestamp range in the user-profile governance database at `~/.config/simple_ai_trading/terminal_holdouts.sqlite3` (override only with `SIMPLE_AI_TRADING_TERMINAL_LEDGER`). The completed row binds SHA-256 fingerprints for the terminal rows, exact post-meta/hybrid model, and terminal financial result. Any overlapping prior reservation, including a crashed, rejected, or evaluation-error run, blocks reuse. Authenticated live startup requires the artifact reservation to match that local database; deleting or replacing the ledger invalidates existing live authority. Missing, repeated, nonpositive, rejected, liquidated, mismatched, or malformed terminal evidence blocks serialization or live readiness. The former zero-fold hybrid fallback path has been removed. Signed startup also requires `execution_validation` stamped by `model-lab`: symbol-specific liquidity evidence, data-coverage integrity, accepted stress validation, accepted temporal robustness, accepted portfolio risk, and any applicable learning-feedback recovery evidence. Plain `train-suite` artifacts remain research artifacts until model-lab records final portfolio acceptance. Paper mode can retrain a rejected model for research, but it cannot authorize live orders. Authenticated live mode also refuses in-loop retraining; run `model-lab` again and promote a newly validated artifact instead.

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
New model labels must also clear the configured round-trip fee-plus-spread
floor; AI approval reads that exact frozen label contract from the model and
fails closed when sealed after-cost validation evidence is incomplete. Calls
that cannot reach an entry boundary are rejected before local-model inference,
so they consume no AI tokens and cannot delay position exits. Meta-label
downsizing is also bucket-specific: weaker signals receive capital only when
their own validation bucket has sufficient support and positive after-cost
expectancy; otherwise they are skipped. The exact bucket evidence is preserved
through liquidity overlays and bound into AI review.

Exchange-backed trading caps follow the active symbol's quote and base assets. The persisted runtime field names remain backward-compatible (`managed_usdc` for quote capacity and `managed_btc` for base-asset capacity), but the CLI and app render and enforce them as USDC/USDT plus BTC/ETH/SOL according to the configured pair.

Day-trading objectives require enough activity to demonstrate a repeatable net-of-cost edge, but the closed-trade minimum and trades/day target are evidence requirements, not quotas that force entries. Sparse historical activity fails model or threshold promotion when there is no documented risk rationale. If market-regime or meta-label risk controls explicitly skipped entries during low-predictability conditions, low activity can still satisfy the activity requirement so the bot may remain flat. Path-quality controls also reject superficially profitable candidates when one outlier trade contributes too much of gross positive P&L.

Hard capital controls are separate from ROI goals. The conservative profile defaults to a `0.60%` daily loss budget, `1.20%` session loss budget, two-loss streak lockout, three consecutive network errors before recovery-halt messaging, and a 60 second post-reconnect observation cooldown. Regular and aggressive raise those limits gradually, but risk reporting blocks live operation when these controls are disabled or dangerously loose.

## Optimization Evidence

Round-level implementation notes live under `docs/optimization/`. This repo no
longer publishes ROI, P&L, drawdown, or chart claims unless they come from
exchange-sourced backtests or signed testnet/paper artifacts with the provenance
required by [docs/DATA_PROVENANCE_POLICY.md](docs/DATA_PROVENANCE_POLICY.md).
The latest model-mechanism evidence is
[`action-value/latest`](docs/model-research/action-value/latest/README.md).
[Round 64](docs/model-research/action-value/round-064-positive-expectancy-meta-label-research.md)
records positive-expectancy meta-label and token-free preflight contracts;
[Round 63](docs/model-research/action-value/round-063-cost-aware-ai-gate-research.md)
records the cost-aware label foundation. Neither contains a new performance
result, so neither replaces the latest evidence graphs.
Round 61 completed the synchronized economic replay authorized by Round 60.
It matched `$10,000` long spot with a 1x short perpetual at the same base
quantity, then applied adverse minute bounds, settled funding, actual-notional
taker fees, one extra basis point per fill, and a 1% same-side taker-flow cap.
Source eligibility was `72/72`, `76/76`, and `61/62` for BTC, ETH, and SOL,
but only `30/20/0` episodes supported every modeled fill. The admitted BTC and
ETH subsets still had negative medians (`-6.70/-5.56` committed-capital bps)
and negative bootstrap lower means (`-7.43/-11.13` bps). The carry family is
rejected; tick replay, model training, AI evaluation, leverage, testnet, live
trading, and profitability claims remain unauthorized. Rounds 1-60 remain in
the rolling progress table.

The latest independent execution-replay confirmation remains
[`tape-depth/latest`](docs/model-research/tape-depth/latest/README.md): Round 8
used three untouched, checksummed Binance dates and exact 100 ms BBO replay. It
was rejected with 12 executable trades, `-11.839347` bps weighted mean net
return, and a `0%` positive-net rate. Both evidence tracks preserve their
negative results, source tables, and hash-manifested graphs; neither makes an
ROI, drawdown, profitability, execution, or trading-authority claim.

Round 005 separately evaluated Kronos-base as a forecast-feature candidate over
1,536 post-pretraining BTC/ETH/SOL observations. It failed the random-walk MAE
gate, causal amplitude calibration, and paired day-block confidence gate. Its
graph is explicitly forecast-error evidence, not ROI or P&L. See
[`docs/optimization/round-005-foundation-forecast-gate.md`](docs/optimization/round-005-foundation-forecast-gate.md).

## Live-Market Simulation

The backtester no longer assumes frictionless fills. It models:

- per-symbol spread,
- latency buffers,
- liquidity haircuts for testnet-to-mainnet differences,
- square-root market impact from causal trailing-24h participation, with a
  fail-closed fallback when the estimate is unavailable,
- DB-backed quote-volume/trade-count activity and high-low exit-range stress,
- account-verified taker fees for authenticated startup and documented
  market-specific floors for offline evidence,
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

## Agent Tooling

The repository includes the adapted `cocoindex-code-search`,
`karpathy-guidelines`, research, regression, and documentation skills plus
pinned Ruff, Vulture, and Super-Linter workflows. See
[`docs/AGENT_WORKFLOWS.md`](docs/AGENT_WORKFLOWS.md) for provenance, commands,
and the mandatory semantic-routing contract. README badges are generated from
`.github/readme_badges.json`.

## Test

```powershell
.\.venv311\Scripts\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File tools\build_native_windows.ps1
```

CI enforces two non-substitutable coverage gates: the current whole-repository
branch-coverage ratchet is `83%`, and every changed executable Python line must
be at least `95%` covered through `diff-cover`. The broader floor is explicit
technical debt, not a 95% claim; it must only move upward as older modules gain
tests. A change cannot pass by preserving the legacy baseline while leaving its
new paths untested.

Focused checks used during this revamp:

```powershell
.\.venv311\Scripts\python.exe -m pytest -q tests/test_compute.py tests/test_ai_runtime_and_parity.py tests/test_autonomous.py tests/test_market_universe.py tests/test_backtest.py tests/test_backtest_coverage.py
```

## Release

The beta release tag is `v0.1.0-beta.1`. Python packaging uses the PEP 440-compatible version `0.1.0b1`.

The manual GitHub Actions workflow `beta-release` builds the native Windows app, runs tests and coverage, packages a portable beta ZIP, attaches checksums, and publishes a GitHub prerelease. See [docs/release.md](docs/release.md).
