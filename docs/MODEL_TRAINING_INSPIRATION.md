# Model and Training Inspiration

Research snapshot: 2026-07-06

This dossier translates current model and training research into implementation
direction for Simple AI Trading. It is not a profitability claim. The purpose is
to keep future model work anchored to evidence, realistic market simulation,
AMD-friendly GPU support, and fail-closed validation.

## Design Thesis

The next serious model stack should be layered:

1. A primary signal model proposes long/short/flat probabilities.
2. A regime model decides whether the current market is predictable enough.
3. A meta-label model decides whether to take, skip, or downsize a signal.
4. A risk model converts accepted signals into leverage, stop, cooldown, and
   portfolio-cap decisions.
5. Optional AI/foundation-model components can add forecasts or risk review,
   but they must be backtested as logged features, prove AI-vs-ML uplift, and
   cannot bypass deterministic gates.

The operator interface should stay simple. Market-mode detection, AI uplift
proof, portfolio CVaR, selection-risk deflation, and connectivity recovery
belong in background gates and reports. The user-facing result should be a
clear state such as running, waiting, review required, blocked, paused, or
stop-and-close.

The implemented coordinator follows that shape: risk, execution,
reconciliation, market data, machine learning, AI, and learning feedback publish
independent status. The coordinator combines those statuses into one decision
and blocks only the appropriate surface. Risk, execution, and reconciliation can
block execution; stale market data, missing models, or failed AI capability can
block new entries; learning feedback stays non-mutating in live operation, but
it feeds model-lab promotion and AI review.

Self-improvement is intentionally bounded. Closed trades refresh a learning
feedback artifact with recurring loss reasons, symbol/side loss clusters, loss
streaks, and retraining/cooldown hints. That artifact can influence the next
model-lab cycle and AI review, but it must not mutate a live model, relax risk
limits, or change open positions while the bot is running. If a symbol has
repeated closed-trade losses, the next model-lab promotion for that symbol now
requires positive stress and temporal robustness recovery evidence.

The Bloomberg Opinion article the user referenced shows why this matters:
modern AI tooling can accelerate exchange connectors, news/social ingestion,
LLM scoring, sizing, routing, and portfolio control into working prototypes very
quickly. That is useful inspiration for scope, but it increases operational
risk. The repo response must be stronger gates, better artifacts, and explicit
separation between advisory AI and executable risk controls.

Source:
<https://www.bloomberg.com/opinion/articles/2026-04-28/ai-trading-bots-are-creating-a-major-financial-risk>

## Structured Blueprint Surface

The research direction is now codified in
`src/simple_ai_trading/model_blueprint.py` and exposed through:

```powershell
simple-ai-trading model-blueprint
simple-ai-trading model-blueprint --risk-level conservative --implemented-only
simple-ai-trading model-blueprint --risk-level regular --json
```

This command is intentionally part of the CLI contract, so the Windows app sees
the same roadmap as the terminal. The blueprint separates implemented,
implemented-evidence, research-candidate, blocked, sandbox, and advisory model
families. It also records execution authority so future changes cannot silently
promote an AI forecast, RL policy, or order-book research model into direct
order placement without tests failing.

The same module now contains a tested training-lane map and source catalog.
Every model family must belong to a promotion lane with explicit next build
steps, runtime limits, and validation gates. Research papers and official docs
can guide implementation, while community TradingView scripts are marked as
inspiration only and must not be copied. The contract test fails if a model
family loses its lane, if a source references an unknown model family, or if a
community source lacks a no-copy policy.

Current blueprint principles:

- Implemented supervised and hybrid models can produce candidate signals only
  after objective, temporal, path-quality, and portfolio gates pass.
- Regime and AI-review layers are veto/cooldown/review layers, not alpha engines.
- The trained meta-label layer is a pre-entry execution gate only: it can skip
  or downsize a signal, but it cannot create a new signal or override exits.
- Foundation forecasts are logged features until ablation and no-lookahead
  replay prove that they add value after costs. AI-assisted variants also need
  an explicit uplift artifact showing improvement over the non-AI ML baseline.
- Market-regime gates are allowed to abstain for long periods. In noisy,
  high-reversal, high-volatility, or low-confidence phases, the correct action
  can be to wait for days rather than force a low-edge trade.
- RL is limited to sandboxed meta-control research. It must not emit raw
  buy/sell orders.
- Order-book models stay blocked until symbol-specific depth/top-of-book data
  and realistic fill simulation exist.

## Training Architecture Inspiration

Enterprise quant platforms such as Qlib, FreqAI, FinRL, and NautilusTrader point
to the same pattern: separate data preparation, feature engineering, model
training, portfolio/risk evaluation, and live/sandbox execution. This repo
should stay smaller, but its training flow should keep those boundaries:

1. Build features from cleaned candles plus exchange/liquidity evidence.
2. Train supervised candidates with barrier-aware labels and calibrated
   probabilities.
3. Run regime evidence before trusting the signal distribution.
4. Train or derive meta-label evidence from simulated trade outcomes.
5. Add optional AI/foundation forecasts only as timestamped features.
6. Score candidates with purged walk-forward, temporal robustness, path quality,
   and portfolio stress.
7. Promote only the exact serialized artifact that passed replay.

Sources:

- <https://github.com/microsoft/qlib>
- <https://www.freqtrade.io/en/stable/freqai/>
- <https://arxiv.org/abs/2111.09395>
- <https://nautilustrader.io/docs/latest/concepts/backtesting/>

## July 2026 Research Pass: Model Lessons

This pass checked primary research, official documentation, and public
community indicators for training ideas that fit the current app without
weakening risk controls.

- PatchTST and Temporal Fusion Transformer research support compact sequence
  forecast experiments, but these should start as timestamped forecast features
  with ablation replay, not as direct order engines.
- Chronos-style time-series foundation models can provide probabilistic
  advisory forecasts. They remain blocked from execution authority until
  no-lookahead logs, same-cost replay, and AI-vs-ML uplift evidence show value
  after fees and slippage.
- TimesFM and Moirai-style universal forecasters are useful additions to the
  foundation-model research lane because they target cross-domain zero-shot
  time-series forecasting. In this repo they stay feature-provider candidates
  until the same no-lookahead and uplift gates pass.
- BloombergGPT and FinGPT reinforce the need for financial-domain evaluation
  and multibillion model checks. They do not justify letting an LLM directly
  trade; the implemented AI-uplift gate now fails closed when the local model
  is too small or when the AI-assisted holdout does not beat the non-AI ML
  baseline.
- Fin-R1, Fin-o1, and LLM Open Finance reinforce the near-term AI direction:
  small finance-reasoning models in the 7B/8B class are realistic local
  benchmark candidates, but they are still advisory/uplift components. They
  must be served locally, pass structured finance-risk benchmark cases, and
  prove realized AI-vs-ML holdout uplift after costs before affecting
  autonomous decisions.
- HMM and regime-switching literature supports the abstention design: detect
  volatility/chop/range/trend phases and reduce exposure or cool down when the
  current state is not where the strategy has demonstrated edge.
- LightGBM's OpenCL GPU path is the best near-term AMD-friendly tabular
  candidate. XGBoost GPU remains CUDA-centered, so it should not become the
  default on this Windows/AMD target.
- Bailey/Lopez de Prado PBO and Deflated Sharpe work reinforces the current
  direction: a high backtest score is not enough unless the number of trials,
  walk-forward behavior, holdout replay, and statistical edge evidence are
  visible. The training suite now records a selection-risk report and rejects a
  candidate when its score does not remain positive after a trial-count
  deflation haircut.
- FreqAI reinforces the value of periodic retraining, feature boundaries,
  backtesting/live separation, and adaptive model workflows.
- TradingView Lorentzian, rational-quadratic kernel, and Technical Ratings
  pages are useful product/model inspiration only. The implemented hybrid
  experts are original code and should stay independently tested.
- Ablation evidence must close the loop between research and review. A hybrid
  expert or feature group that looks sophisticated but hurts the accepted score
  is not allowed to hide behind an aggregate ensemble result; compact ablation
  summaries are now part of the AI-review prompt, and positive ablation deltas
  veto before any local model is called.
- The refreshed backtesting literature reinforces that candle-only fills are
  not enough for day trading claims. Future promotion artifacts should disclose
  whether a model was tested with symbol-specific spread, top-of-book depth,
  feed latency, order latency, and queue/fill uncertainty. When those fields are
  missing, the artifact is lower-confidence and cannot justify an HFT-style
  claim.
- Deflated selected score is the minimum acceptable multiple-testing guard.
  The next training-suite step should add CPCV/PBO-style diagnostics for broad
  searches so repeated parameter sweeps do not select a lucky equity curve.
- Implemented update: training-suite selection risk now stores a two-panel
  CSCV/PBO-style proxy over selection and validation scores. It blocks promotion
  when the in-sample winner falls below the out-of-sample median in both
  symmetric views, and it prints `pbo=` in the CLI training summary.
- DeepLOB and HLOB reinforce that credible day-trading/HFT-style modeling
  needs actual order-book tensors, not candle-only proxies. The app should not
  promote an order-book model until it can reconstruct local books, prove depth
  continuity, replay quote-time walk-forward windows, and simulate queue/fill
  uncertainty per symbol.
- Implemented update: v6 advanced features now add order-flow microstructure
  proxies from real second-level quote volume, trade count, taker-buy
  base/quote volume, signed flow imbalance, no-trade seconds, and
  signed-flow/return alignment. This is not a substitute for true order-book
  tensors, but it gives the current BTC/ETH/SOL second-level database a more
  defensible microstructure signal source than OHLCV alone.
- FinMamba/Mamba-style research is useful inspiration for "trade anything"
  workflows because cross-asset relationships and market regimes can change
  quickly. In this repo, those models should begin as point-in-time
  cross-symbol context features for rank, correlation-shock, and diversification
  evidence; they must not directly emit orders.
- FinRL is the right kind of reinforcement-learning inspiration: environment,
  agent, evaluation, transaction-cost, liquidity, and risk-aversion structure.
  That supports keeping RL in a sandboxed meta-control lane until the simulator
  is realistic enough.
- TradingView strategy docs are useful product inspiration for operator
  expectations around market, limit, stop, stop-limit, backtesting, and forward
  testing. They are not a source for copied strategies.
- Bank of England and SEC risk-control material reinforces the same model rule:
  autonomous AI can review or veto, but hard pre-trade limits, stops, kill
  switches, and accountability logs remain deterministic and apply before order
  routing.
- DirectML remains the Windows-first AMD/NVIDIA-compatible research path for
  PyTorch experiments, with ONNX Runtime DirectML available for inference
  parity checks. ONNX Runtime DirectML has practical packaging constraints:
  prefer fixed tensor shapes, avoid shared multi-threaded `Run` calls on one
  DirectML session, and record provider/fallback details. Every accelerated run
  must still persist backend package versions, selected device, VRAM/RAM
  checks, and fallback reason.

Sources:

- <https://github.com/yuqinie98/patchtst>
- <https://arxiv.org/abs/1912.09363>
- <https://github.com/amazon-science/chronos-forecasting>
- <https://arxiv.org/abs/2310.10688>
- <https://arxiv.org/abs/2402.02592>
- <https://arxiv.org/abs/2303.17564>
- <https://arxiv.org/abs/2306.06031>
- <https://arxiv.org/abs/2503.16252>
- <https://arxiv.org/abs/2502.08127>
- <https://huggingface.co/DragonLLM/Qwen-Open-Finance-R-8B>
- <https://arxiv.org/abs/2007.14874>
- <https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html>
- <https://lightgbm.readthedocs.io/en/latest/GPU-Targets.html>
- <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
- <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
- <https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf>
- <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376>
- <https://www.freqtrade.io/en/stable/freqai/>
- <https://hftbacktest.readthedocs.io/en/py-v2.2.0/>
- <https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-II/>
- <https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows>
- <https://github.com/microsoft/DirectML>
- <https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/>
- <https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/>
- <https://www.tradingview.com/support/solutions/43000614331-technical-ratings/>
- <https://arxiv.org/abs/1808.03668>
- <https://arxiv.org/abs/2405.18938>
- <https://arxiv.org/html/2502.06707v2>
- <https://arxiv.org/abs/2402.18959>
- <https://arxiv.org/abs/2011.09607>
- <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- <https://www.tradingview.com/pine-script-docs/concepts/strategies/>
- <https://www.bankofengland.co.uk/speech/2026/june/sarah-breeden-panel-at-the-european-central-bank-forum-on-central-banking-2026>
- <https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/divisionsmarketregfaq-0>
- <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>

## Model Families Worth Implementing

### 1. Regime Detection

Use a regime layer before trading and before choosing model weights. Markets
shift between trend, chop, high-volatility liquidation, low-liquidity drift,
breakout, and mean-reversion states. A single static classifier will overtrade
when the current state differs from its training distribution.

Implementation direction:

- Add a compact regime feature block: realized volatility percentile, ATR
  percentile, spread percentile, volume z-score, trend slope, wick/body
  instability, return autocorrelation, and cross-symbol correlation shock.
- Start with deterministic/GMM-style clustering or simple HMM-like state
  persistence using stdlib-safe math.
- Gate trading when the regime has low historical precision, high realized
  spread, or unstable recent prediction calibration.
- Persist `regime_id`, `regime_confidence`, `regime_trade_count`,
  `regime_expectancy`, and `regime_profit_factor` in model-lab artifacts.

Sources:

- <https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/>
- <https://developers.lseg.com/en/article-catalog/article/market-regime-detection>
- <https://macrosynergy.com/research/classifying-market-regimes/>

### 2. Meta-Labeling

The primary model should not be forced to both discover opportunities and
decide whether each opportunity is worth taking under current costs. A
secondary meta-label model should learn whether a proposed signal survived its
stop/take/time barrier after fees, spread, slippage, latency, and liquidity
haircuts.

Implementation direction:

- First model: predicts opportunity direction or probability.
- Meta model: predicts "take/skip/downsize" using primary probability,
  probability margin, spread, liquidity score, volatility regime, recent
  calibration drift, cooldown state, and portfolio concentration.
- Training labels must be derived from the same execution simulator used in
  backtests, not from close-to-close price movement alone.
- Conservative mode should require high meta-label precision and skip more
  often. Aggressive mode can accept lower meta precision only if drawdown,
  loss-streak, and CVaR gates remain intact.
- Current implementation trains a compact meta-label policy from simulated trade
  outcomes, persists it in model/training artifacts, and applies it as a
  deterministic pre-entry gate in backtests and live/autonomous entry paths.
  Enabled policies can only `take`, `downsize`, or `skip`; malformed enabled
  policies fail closed by skipping the entry.

Sources:

- <https://www.quantresearch.org/Innovations.htm>
- <https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/>
- <https://mlfinpy.readthedocs.io/en/latest/Labelling.html>

### 3. Patch-Based Time-Series Models

PatchTST-style models are attractive for this repo because they process longer
history more efficiently by patching time-series windows and using
channel-independent weights. This fits multi-symbol OHLCV plus engineered
features better than asking an LLM to directly choose orders.

Implementation direction:

- Add a research-only PyTorch/DirectML sequence model candidate behind
  `--model-family patch_transformer`.
- Train it to forecast barrier-adjusted returns, not raw future close.
- Convert its output into logged features first; only promote it to primary
  signal model after walk-forward, temporal, and path-quality gates pass.
- Keep model size small enough for 8 GB VRAM and batch scoring on DirectML.

Sources:

- <https://arxiv.org/abs/2211.14730>
- <https://github.com/yuqinie98/patchtst>

### 4. Time-Series Foundation Forecasts

Chronos and TimesFM show that pretrained time-series models can produce useful
zero-shot or few-shot forecasts. For trading, they should be treated as an
advisory feature generator until this repo proves they survive transaction
costs and non-stationary live-like simulation.

Implementation direction:

- Optional AI feature provider, disabled when CPU-only AI is disabled.
- Generate probabilistic or point forecasts for return distribution, volatility,
  and tail-risk scenarios.
- Log every forecast used in a decision artifact.
- Backtest with the exact forecast timestamps and forbid lookahead.
- Never allow a foundation forecast to override stop-loss, exposure cap, or
  kill-switch gates.

Sources:

- <https://arxiv.org/abs/2403.07815>
- <https://github.com/amazon-science/chronos-forecasting>
- <https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/>
- <https://arxiv.org/abs/2310.10688>

### 4B. Cross-Asset Graph/Sequence Models

The repo is no longer a single-Bitcoin concept. A realistic autonomous
day-trading system needs to understand when assets move together, when
correlations break, and when diversification is fake because the whole accepted
set is one crowded trade. FinMamba-style market-aware graphs and selective
state-space models are useful inspiration for this, but they should not be
treated as magic alpha engines.

Implementation direction:

- Build a point-in-time cross-symbol feature store with return ranks,
  volatility ranks, rolling correlation clusters, quote/liquidity ranks, and
  regime states.
- Train compact graph/sequence candidates to output portfolio-context features:
  rank forecast, correlation-shock risk, diversification warning, and
  market-wide stress probability.
- Require graph sparsity stability and ablation against independent per-symbol
  baselines before any output can influence sizing.
- Keep execution authority at `portfolio_context_features_only` until
  walk-forward, temporal robustness, and portfolio CVaR gates pass.

Sources:

- <https://arxiv.org/html/2502.06707v2>
- <https://arxiv.org/abs/2402.18959>
- <https://arxiv.org/abs/1912.09363>

### 5. Gradient-Boosting Candidates

Tree ensembles are still practical for tabular technical, liquidity, and regime
features. On this user's AMD/Windows target, LightGBM's OpenCL GPU path is more
promising than CUDA-only paths. XGBoost GPU currently centers on CUDA, and
CatBoost GPU introduces nondeterminism that must be handled in validation.

Implementation direction:

- Add optional `lightgbm_opencl` candidate only when the dependency is present
  and GPU capability checks pass.
- Treat XGBoost GPU as NVIDIA/CUDA-specific unless a supported AMD path is
  verified.
- Treat CatBoost GPU as research-only until repeated-seed variance is recorded
  in artifacts.
- Export successful tabular candidates to ONNX only after inference parity is
  proven against native predictions.

Sources:

- <https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html>
- <https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html>
- <https://xgboost.readthedocs.io/en/stable/gpu/index.html>
- <https://catboost.ai/docs/en/features/training-on-gpu>

### 6. Reinforcement Learning

Financial RL is useful for portfolio allocation, execution, and policy research,
but it is dangerous as a direct live decision engine without standardized
environments, realistic frictions, and reproducible benchmarks. It belongs in a
sandboxed research track first.

Implementation direction:

- Build a vectorized simulation interface only after depth/spread/latency data
  is persisted.
- Use RL first for meta-control: position scaling, cooldown length, and
  execution style, not raw buy/sell decisions.
- Require deterministic baselines, repeated seeds, and out-of-sample market
  regime tests before an RL policy can influence autonomous operation.

Sources:

- <https://arxiv.org/abs/2011.09607>
- <https://github.com/AI4Finance-Foundation/FinRL>

## GPU Direction

The repo should keep DirectML as the default Windows path for PyTorch models
because it supports DirectX 12 GPUs across AMD, NVIDIA, and Intel. For ONNX
inference, ONNX Runtime's DirectML execution provider remains relevant, but its
docs now describe DirectML as sustained engineering and point Windows ONNX
deployments toward WinML for future provider selection.

Implementation direction:

- PyTorch training/scoring: `torch-directml` first on Windows.
- ONNX inference: DirectML today, evaluate WinML provider selection before a
  packaged inference runtime is added.
- LightGBM tabular candidates: OpenCL GPU path when available.
- CPU mode: allowed, slower, and no AI approval.
- Every GPU model family must have a CPU fallback for tests, plus a capability
  artifact that records backend, device, package versions, and reason for
  fallback.
- A model cannot be used by signed live startup just because it deserializes.
  Runtime readiness requires the promoted artifact to carry passing
  selection-risk evidence with a positive deflated score; otherwise the operator
  must rerun model-lab.

Sources:

- <https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows>
- <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>
- <https://microsoft.github.io/DirectML/>

## Market Simulation Requirements

Candle-only backtesting cannot honestly claim high-frequency realism. The
current candle simulation is useful, but the next realism upgrade needs
symbol-specific order-book evidence.

Implementation direction:

- Use the persisted typed top-of-book samples from `data-sync` for L1 spread,
  depth, and quote-quality evidence.
- Use `data-sync --full-history` and `model-lab --full-history` for
  promotion-grade research. Recent-limit API pulls are acceptable for smoke
  checks only when the artifact labels source scope, UTC date span, row count,
  gap count, and coverage ratio.
- Add L2 depth snapshots before claiming queue-position or full order-book
  replay realism.
- Add features for spread percentile, depth imbalance, microprice, quote
  volatility, top-level depth, and observed quote update rate.
- Simulate queue/fill uncertainty for limit orders, adverse selection after
  fills, latency between signal and order, and market-impact haircuts for
  position size.
- Keep candle proxies only as fallback when depth data is unavailable, and mark
  the artifact as lower-confidence.

Sources:

- <https://arxiv.org/html/2402.17359v1>
- <https://www.cis.upenn.edu/~mkearns/papers/KearnsNevmyvakaHFTRiskBooks.pdf>
- <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints>
- <https://binance.github.io/binance-api-swagger/>
- <https://arxiv.org/abs/1808.03668>
- <https://arxiv.org/abs/2405.18938>
- <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>

## TradingView-Inspired Feature Ideas

TradingView is useful inspiration for indicator families and operator
expectations, not code to copy. Existing hybrid experts already use Lorentzian
neighbors, rational-quadratic kernels, and technical confluence. Future work
should broaden feature diversity without turning the model into an indicator
dump.

Implementation direction:

- Keep MA/oscillator voting as compact confluence features.
- Add VWAP distance and VWAP reclaim/failure features where volume is reliable.
- Add Supertrend-like ATR band state as a regime feature, not an unconditional
  entry rule.
- Add Bollinger/Keltner squeeze and breakout-confirmation features.
- Backtest each feature family with ablation reports so useless features can be
  removed.

Sources:

- <https://www.tradingview.com/pine-script-docs/language/built-ins/>
- <https://www.tradingview.com/pine-script-docs/concepts/strategies/>
- <https://www.tradingview.com/support/solutions/43000614331-technical-ratings/>
- <https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/>
- <https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/>

## Validation Contract For Future Models

No future model family should be accepted unless it writes:

- candidate family and hyperparameters,
- financial-sanity status for model dimensions, finite parameters, probability
  calibration, row counts, coverage, and bounded risk metrics,
- training backend and device evidence,
- feature signature and data interval,
- data coverage evidence: symbol, market type, interval, UTC span, source
  scope, full-history flag, candle count, model-row count, coverage ratio, gap
  count, and truth basis,
- purged walk-forward results,
- final serialized-model temporal robustness,
- market-regime evidence for each temporal window,
- statistical edge evidence,
- multiple-trials selection-risk evidence and deflated selected score,
- path-quality metrics: profit factor, expectancy, return dispersion, and loss
  streak,
- symbol-specific execution stress,
- portfolio-level correlation and CVaR stress,
- source of any AI/foundation forecast used,
- hybrid and feature ablation deltas for accepted candidates,
- reproducibility seeds or nondeterminism warning.

Implemented objective gates:

- Conservative: require positive expectancy, finite profit factor above 1.10,
  and max loss streak at or below 3.
- Regular: require positive expectancy, profit factor above 1.05, and max loss
  streak at or below 5.
- Aggressive: allow higher variance but require positive expectancy, profit
  factor at or above 1.00, no drawdown stop, and no portfolio CVaR breach.

## Prioritized Backlog

1. Expand meta-label validation with cross-symbol, out-of-sample policy replay
   and per-risk-level precision/drift dashboards.
2. Microstructure feature block from typed top-of-book samples, then L2 depth
   snapshots for queue/fill simulation.
3. LightGBM OpenCL tabular candidate with repeated-seed validation.
4. Patch-transformer research candidate using PyTorch DirectML.
5. Cross-asset graph/sequence feature store for diversification-aware ranking
   and correlation-shock warnings.
6. Foundation forecast feature provider with timestamped no-lookahead logs.
7. RL sandbox for meta-control only after realistic depth simulation exists.
8. Feature ablation reports for every indicator/model family.
9. ONNX/WinML inference parity checks before packaging AI inference into the
   Windows app.

## What Not To Do

- Do not let an LLM choose orders directly.
- Do not claim HFT-grade realism from candle-only data.
- Do not add CUDA-only training as the default path on this AMD-targeted host.
- Do not promote RL policies to live/testnet execution before a realistic
  simulator and repeated-seed evidence exist.
- Do not optimize for final ROI alone; optimize for robust, repeatable net
  returns after spread, fees, latency, slippage, liquidity haircuts, drawdown,
  and portfolio tail risk.
