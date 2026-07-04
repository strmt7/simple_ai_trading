# Model Research and Optimization

This document records the model direction implemented for the `0.1.0-beta.1`
revamp. The goal is autonomous multi-asset day trading with fail-closed risk
gates, not a promise of guaranteed profit.

## Research Inputs

- TradingView Pine built-ins and public indicator conventions were used as
  conceptual references for common technical features such as RSI, EMA, ATR,
  volume, trend, and volatility:
  <https://www.tradingview.com/pine-script-docs/language/built-ins/>
- TradingView Technical Ratings inspired the multi-timeframe confluence block:
  moving-average direction, oscillator/candle confirmation, and aggregate
  +1/0/-1-style voting are represented as original numeric features rather
  than copied Pine logic:
  <https://www.tradingview.com/support/solutions/43000614331-technical-ratings/>
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
- Bailey, Borwein, Lopez de Prado, and Zhu's Probability of Backtest
  Overfitting framework influenced the requirement that model-lab reject
  single-path winners and report how a selected model behaves across multiple
  chronological windows:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>
- Bailey and Lopez de Prado's Deflated Sharpe Ratio work influenced the
  project policy of treating high backtest scores as suspect unless the
  selection process and holdout evidence are visible in the artifact:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
- Triple-barrier labeling, event-driven sampling, and purged validation were
  used as training inspiration so labels match stop/take economics instead of
  only future-close direction:
  <https://link.springer.com/article/10.1186/s40854-025-00866-w>
- MLFinPy's triple-barrier docs reinforce the same principle: barriers should
  reflect volatility/risk, not a single fixed close-to-close horizon:
  <https://mlfinpy.readthedocs.io/en/latest/Labelling.html>
- DeepLOB influenced the roadmap for true order-book models; until depth
  snapshots are persisted, this repo uses candle microstructure proxies such as
  body, wick, close-location, ATR, breakout, and volume-surge features:
  <https://arxiv.org/abs/1808.03668>
- FinRL influenced the training-environment boundary: transaction cost,
  liquidity, and risk aversion must live inside the evaluation loop before any
  autonomous model can be accepted:
  <https://arxiv.org/abs/2111.09395>
- Lopez de Prado's Hierarchical Risk Parity work influenced the portfolio
  acceptance layer: individual profitable symbols are not enough if the
  accepted set is concentrated in one high-correlation cluster:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>
- The Basel market-risk backtesting framework influenced the tail-risk gate:
  the portfolio report measures VaR/CVaR-style losses and drawdown from the
  same aligned returns used for model-lab acceptance:
  <https://www.bis.org/publ/bcbs22.pdf>
- NIST AI RMF's govern/map/measure/manage structure influenced the decision to
  make AI/model risk explicit in reports instead of hiding it behind a single
  score:
  <https://airc.nist.gov/airmf-resources/airmf/5-sec-core/>
- Ollama structured outputs and OpenAI structured outputs influenced the
  `ai-review` workflow: local model output is constrained to a JSON schema,
  then validated by deterministic code before it can be treated as an approval:
  <https://docs.ollama.com/capabilities/structured-outputs> and
  <https://developers.openai.com/api/docs/guides/structured-outputs>
- Microsoft DirectML was selected for the Windows-first GPU path because it
  supports DirectX 12 GPUs across AMD, NVIDIA, and Intel on Windows:
  <https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows>
- Binance market-data endpoints are used for automatic universe ranking instead
  of static symbol allowlists:
  <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints>

## Implemented Model Zoo

The base classifier remains the advanced logistic/GPU training path already
used by the CLI. The advanced feature vector now includes:

- multi-window technical rating votes inspired by TradingView's MA/oscillator
  aggregation structure,
- candle microstructure proxies inspired by order-book literature when only
  OHLCV candles are available,
- ATR-normalized trend and breakout features,
- volume-surge confirmation for high-frequency day-trading entries.

The revamp also adds a hybrid expert layer stored directly inside the serialized
model:

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

The training-suite grid deliberately includes lower threshold probes, multiple
label target/horizon profiles, and both forward-return and stop/take-aware
triple-barrier labels for every risk level. This prevents high-confidence-only,
single-horizon, or direction-only candidates from being rejected only because
they never trade or because the label rewards moves that cannot survive fees,
spread, stop losses, and take-profit geometry. The objective gates still require
positive P&L, sufficient closed trades, buy-and-hold edge, and drawdown
discipline. Rejected model-lab candidates now include per-window `reject_reason`
diagnostics so operators can distinguish missing trade count, negative P&L,
buy-and-hold edge failure, drawdown failure, and stopped-by-drawdown failures.
For host smoke checks, `train-suite` and `model-lab` expose `--max-candidates`;
this caps candidate count per objective only when explicitly set and should not
be used to claim a full optimization result.

For futures, threshold calibration stores the same effective neutral-band
threshold used by live and backtest direction logic: values below `0.5` are not
persisted as futures decision thresholds because long/short futures signals use
`score >= threshold` for long and `score <= 1 - threshold` for short.

After the broad grid is ranked, local refinement also tests tighter and wider
exit geometry around the best candidate, including lower take-profit variants
that can close more intraday positions when the first pass fails trade-count
or buy-and-hold edge gates.

Accepted hybrid candidates must improve or preserve the objective score and pass
the profitability, drawdown, and minimum-trade gates in
`ObjectiveSpec.accepts`. If no base candidate survives the hard gates, the
training suite now attempts a small fail-closed hybrid rescue pass over the top
rejected base candidates. A rescued hybrid is serialized only when it passes the
hybrid selection window, the final chronological holdout, and the full-sample
objective gates; otherwise the objective remains rejected.

Model-lab also replays the final serialized model artifact across separate
chronological windows after training is complete. This differs from the
training-suite purged walk-forward gate: purged walk-forward retrains candidates
to select a stable configuration, while `temporal_robustness.json` tests the
exact saved model, including any hybrid expert overlay. Conservative models
must satisfy the strictest window coverage, regular models use the middle
threshold, and aggressive models allow more dispersion while still requiring
positive, non-drawdown-stopped windows. The same artifact now includes a
statistical edge gate: an exact one-sided sign test over positive windows plus a
deterministic bootstrap-style lower confidence bound over mean window return.
This implements the practical lesson from PBO/Deflated-Sharpe research: a high
aggregate score is not enough when the distribution of tested windows still
looks like selection luck.

## Cross-Symbol Model Lab

`simple-ai-trading model-lab` is the iterative optimization workflow. It:

1. Pulls exchange metadata, 24h tickers, and book tickers.
2. Automatically ranks high-liquidity symbols using quote volume, trade count,
   bid/ask spread, exchange status, and quote-asset policy.
3. Fetches recent klines for each ranked symbol.
4. Runs the training suite and hybrid optimizer for one or more objectives.
5. Requires the selected candidate to pass purged chronological walk-forward
   folds before serialization. The purge gap protects against label-lookahead
   leakage between train and test folds.
6. Replays every saved objective model under mandatory symbol-specific stress:
   baseline measured execution, wider spread/slippage, latency spike with a
   liquidity haircut, and combined liquidity crunch with fee/spread/latency
   stress.
7. Replays every saved objective model through `temporal_robustness.json`, a
   separate chronological-window robustness gate for the final serialized
   artifact. This catches models that pass aggregate stress but fail in recent
   or regime-specific windows. The temporal report also records statistical
   edge evidence, including sign-test p-value and bootstrap lower mean return,
   and rejects candidates whose window evidence is too weak for the selected
   risk level.
8. Builds a portfolio-level risk report from aligned symbol returns. This gate
   computes inverse-volatility capped weights, effective symbol count,
   pairwise correlations, high-correlation clusters, portfolio 95% VaR/CVaR,
   and portfolio drawdown.
9. Writes a JSON report plus per-symbol `stress_validation.json`,
   `temporal_robustness.json`, and `portfolio_risk.json`. An outcome is
   accepted only when all objective scores are positive, every stress replay
   passes the objective risk gates, temporal robustness passes, and the
   accepted set passes the portfolio diversification and tail-risk gates.

After model-lab writes a report, `simple-ai-trading ai-review --report ...`
can run a local structured-output model over a compact artifact summary. The
review is intentionally bounded and non-executing: it receives no credentials,
uses the AI capability preflight, requires GPU AI unless the user explicitly
changes runtime settings, validates the JSON schema, and writes
`ai_risk_review.json`. Missing accepted symbols, failed portfolio gates, failed
AI preflight, provider errors, or invalid model JSON all produce a veto/review
result instead of approval.

This is deliberately fail-closed. If live testnet data cannot produce a
profitable, diversified, risk-bounded candidate, the report should reject the
candidate instead of forcing a trade.

## SuperZip Windows-App Alignment

The Windows app follows the SuperZip design direction:

- Native C++20 Win32 app instead of Tkinter.
- PowerShell build script that discovers Visual Studio, CMake, Ninja, and
  Python.
- DPI-aware Win32 layout with Segoe UI fonts, DWM dark caption colors, and
  real listbox, combobox, edit, and button controls.
- Grouped operator workflows instead of an alphabetical command dump, while the
  CLI parity command picker still exposes every generated command.
- Repo-aware command launching that resolves `.venv311` and sets `PYTHONPATH`
  before running the Python CLI from a native app build.
- Generated command contract from the Python CLI so the GUI command list cannot
  drift from CLI capabilities.
- Explicit build, GUI smoke, and automated control-navigation tests rather than
  manual launch assumptions.

The app is intentionally an operator console over the exact CLI contract. New
workflow parity must be added to the Python parser first; then
`tools/generate_windows_contract.py` regenerates the native header and tests
assert that every CLI command appears in the Windows app.

## Non-Negotiable Gates

- No static list of approved symbols. Liquidity must be measured from exchange
  data.
- No mainnet trading by default.
- No leverage above 20x.
- No AI in CPU-only mode.
- No non-profitable accepted model-lab outcome.
- No selected training-suite model without purged walk-forward evidence when
  enough rows are available.
- No single-scenario-only model-lab acceptance.
- No model-lab symbol acceptance when the final serialized model fails
  chronological temporal robustness windows.
- No model-lab symbol acceptance when temporal windows have weak statistical
  edge evidence after selection.
- No model-lab acceptance when the individually passing symbols fail the
  portfolio-level correlation, concentration, CVaR, or drawdown gate.
- No AI review approval unless deterministic model-lab/portfolio gates passed,
  the local AI capability gate passed, and the provider returned valid
  structured JSON.
- No Windows-app-only workflow.
- No CLI-only workflow.
- Stop/pause controls must remain visible and tested.
