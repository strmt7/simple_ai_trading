# Model Research and Optimization

This document records the model direction implemented for the `0.1.0-beta.1`
revamp. The goal is autonomous multi-asset day trading with fail-closed risk
gates, not a promise of guaranteed profit.

## Research Inputs

- TradingView Pine built-ins and public indicator conventions were used as
  conceptual references for common technical features such as RSI, EMA, ATR,
  volume, trend, and volatility:
  <https://www.tradingview.com/pine-script-docs/language/built-ins/>
- The public Lorentzian Classification indicator inspired the Lorentzian
  nearest-neighbor expert. No Pine source was copied:
  <https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/>
- The public Nadaraya-Watson rational quadratic kernel indicator inspired the
  smooth kernel-regression expert. No Pine source was copied:
  <https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/>
- Freqtrade Hyperopt and Protections influenced the separation between
  optimization, cooldowns, stop guards, and drawdown gates:
  <https://www.freqtrade.io/en/stable/hyperopt/> and
  <https://www.freqtrade.io/en/2024.1/includes/protections/>
- Microsoft DirectML was selected for the Windows-first GPU path because it
  supports DirectX 12 GPUs across AMD, NVIDIA, and Intel on Windows:
  <https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows>
- Binance market-data endpoints are used for automatic universe ranking instead
  of static symbol allowlists:
  <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints>

## Implemented Model Zoo

The base classifier remains the advanced logistic/GPU training path already
used by the CLI. The revamp adds a hybrid expert layer stored directly inside
the serialized model:

- `lorentzian_knn`: balanced long/short prototypes selected from chronological
  training rows, scored with Lorentzian distance.
- `rational_quadratic_kernel`: kernel-regression vote over the same prototype
  set, with tunable alpha and length scale.
- `technical_confluence`: deterministic market-regime confluence using the
  existing feature vector for trend, volatility, volume, and mean-reversion.

The optimizer evaluates risk-level-specific weight profiles:

- `conservative`: favors base-model agreement, smoother kernels, and smaller
  expert overrides.
- `regular`: balances base probability, Lorentzian neighbor structure, kernel
  smoothness, and confluence.
- `aggressive`: allows stronger expert contribution, but still has to pass
  backtest gates and drawdown limits.

Accepted hybrid candidates must improve or preserve the objective score and pass
the profitability, drawdown, and minimum-trade gates in
`TrainingObjective.accepts`. If none pass, the original base model is kept.

## Cross-Symbol Model Lab

`simple-ai-trading model-lab` is the iterative optimization workflow. It:

1. Pulls exchange metadata, 24h tickers, and book tickers.
2. Automatically ranks high-liquidity symbols using quote volume, trade count,
   bid/ask spread, exchange status, and quote-asset policy.
3. Fetches recent klines for each ranked symbol.
4. Runs the training suite and hybrid optimizer for one or more objectives.
5. Writes a JSON report and marks an outcome accepted only when all objective
   scores are positive and all risk gates pass.

This is deliberately fail-closed. If live testnet data cannot produce a
profitable, diversified, risk-bounded candidate, the report should reject the
candidate instead of forcing a trade.

## SuperZip Windows-App Alignment

The Windows app follows the SuperZip design direction:

- Native C++20 Win32 app instead of Tkinter.
- PowerShell build script that discovers Visual Studio, CMake, Ninja, and
  Python.
- Double-buffered GDI rendering with Segoe UI fonts and DWM dark caption colors.
- Generated command contract from the Python CLI so the GUI command list cannot
  drift from CLI capabilities.
- Explicit smoke/build tests rather than manual launch assumptions.

The app is intentionally an operator console over the exact CLI contract. New
workflow parity must be added to the Python parser first; then
`tools/generate_windows_contract.py` regenerates the native header and tests
assert that every CLI command appears in the Windows app.

## Non-Negotiable Gates

- No static list of approved symbols. Liquidity must be measured from exchange
  data.
- No mainnet trading by default.
- No leverage above 10x.
- No AI in CPU-only mode.
- No non-profitable accepted model-lab outcome.
- No Windows-app-only workflow.
- No CLI-only workflow.
- Stop/pause controls must remain visible and tested.
