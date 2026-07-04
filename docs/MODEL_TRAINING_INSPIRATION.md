# Model and Training Inspiration

Research snapshot: 2026-07-04

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
   but they must be backtested as logged features and cannot bypass
   deterministic gates.

The Bloomberg Opinion article the user referenced shows why this matters:
modern AI tooling can accelerate exchange connectors, news/social ingestion,
LLM scoring, sizing, routing, and portfolio control into working prototypes very
quickly. That is useful inspiration for scope, but it increases operational
risk. The repo response must be stronger gates, better artifacts, and explicit
separation between advisory AI and executable risk controls.

Source:
<https://www.bloomberg.com/opinion/articles/2026-04-28/ai-trading-bots-are-creating-a-major-financial-risk>

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

- <https://arxiv.org/html/2504.02281v3>
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

Sources:

- <https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows>
- <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>
- <https://microsoft.github.io/DirectML/>

## Market Simulation Requirements

Candle-only backtesting cannot honestly claim high-frequency realism. The
current candle simulation is useful, but the next realism upgrade needs
symbol-specific order-book evidence.

Implementation direction:

- Persist depth snapshots or rolling top-of-book samples per symbol.
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
- <https://www.tradingview.com/support/solutions/43000614331-technical-ratings/>
- <https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/>
- <https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/>

## Validation Contract For Future Models

No future model family should be accepted unless it writes:

- candidate family and hyperparameters,
- training backend and device evidence,
- feature signature and data interval,
- purged walk-forward results,
- final serialized-model temporal robustness,
- statistical edge evidence,
- path-quality metrics: profit factor, expectancy, return dispersion, and loss
  streak,
- symbol-specific execution stress,
- portfolio-level correlation and CVaR stress,
- source of any AI/foundation forecast used,
- reproducibility seeds or nondeterminism warning.

Implemented objective gates:

- Conservative: require positive expectancy, finite profit factor above 1.10,
  max loss streak at or below 3, and regime-level pass evidence.
- Regular: require positive expectancy, profit factor above 1.05, and max loss
  streak at or below 5.
- Aggressive: allow higher variance but require positive expectancy, profit
  factor at or above 1.00, no drawdown stop, and no portfolio CVaR breach.

## Prioritized Backlog

1. Regime evidence in model-lab artifacts.
2. Meta-label take/skip/downsize model trained on simulator outcomes.
3. Depth/top-of-book data store and microstructure feature block.
4. LightGBM OpenCL tabular candidate with repeated-seed validation.
5. Patch-transformer research candidate using PyTorch DirectML.
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
