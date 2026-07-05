# Model Research and Optimization

This document records the model direction implemented for the `0.1.0-beta.1`
revamp. The goal is autonomous BTC/ETH/SOL day trading with fail-closed risk
gates, not a promise of guaranteed profit.

## Research Inputs

The implementation roadmap is expanded in
[Model and Training Inspiration](MODEL_TRAINING_INSPIRATION.md), which records
the current direction for regime detection, meta-labeling, patch-based
time-series models, foundation-model forecasts, AMD-friendly GPU candidates,
and market-microstructure simulation upgrades.

The same direction is also exposed as a tested CLI/Windows-app parity surface:

```powershell
simple-ai-trading model-blueprint --json
```

Future model work should update that structured blueprint whenever a model
family moves from research to implemented evidence or from advisory evidence to
execution gating. Do not treat roadmap entries as product capabilities: a
capability is executable only when its blueprint status, code path, tests, and
operator docs all agree.

Promotion-grade optimization is also data-health gated. If
`tools/optimization_round.py` is run without explicit symbols and with hard
requirements such as `--require-prefilled-data`, `--min-data-rows`, or
`--require-verified-checksum`, the round builder scans the live liquidity-ranked
universe and keeps only candidates whose local SQLite candles pass the requested
row-count, coverage, gap, archive-status, and checksum gates. Rejected candidates
are recorded in `selection_health_rejections`, so a report cannot hide that a
supported major pair was skipped because the local evidence was too short,
gappy, unverified, or outside the hard BTC/ETH/SOL scope.

For long promotion rounds, use `--require-gpu` unless intentionally profiling
CPU-only behavior. The round resolves the compute backend before symbol work,
records the backend in `round-status.json` and `report.json`, and refuses to run
when the requested backend falls back to CPU. `round-status.json` is updated
inside each symbol at data-health, load, feature-generation, training,
threshold-calibration, holdout-scoring, and artifact-streaming phases. Full-
resolution per-minute graph data stays in CSV; SVG charts render deterministic
downsampled visual summaries so artifact generation does not dominate
GPU-backed training/scoring time.

Feature generation is now part of the accelerated path. Base row construction
uses tensorized prefix/window math for momentum, trend, RSI, EMA, ATR,
volatility, and volume features on the resolved backend, and advanced
objective rows inherit those backend-built base features. The training suite,
optimization evidence generator, model-lab candidate evaluation, robust
validation, live/autonomous readiness rows, and ad-hoc `backtest-panel` all
pass the active compute backend into row construction. If the backend cannot
execute the required tensor operations, the builder falls back to the original
CPU feature path instead of emitting partial rows.

Signed live startup and `risk --live --model` use the same promotion evidence
instead of trusting a model file by name. They block promoted `TrainedModel`
artifacts that do not record bounded multi-candidate selection, and they also
block missing/CPU training or probability-calibration evidence whenever the
resolved live runtime backend is DirectML/CUDA/ROCm/MPS.

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
- TimesFM, Chronos, and Moirai reinforced the value of foundation-style
  probabilistic forecasts for time series, but the repo treats them as logged
  feature providers until no-lookahead replay and AI-vs-ML uplift evidence pass:
  <https://arxiv.org/abs/2310.10688>,
  <https://arxiv.org/abs/2403.07815>, and
  <https://arxiv.org/abs/2402.02592>
- BloombergGPT and FinGPT reinforced the expectation that financial LLM work
  should use multibillion local models and financial-domain evaluation. This
  repo uses that as a capability and governance check, not as permission for an
  LLM to place orders:
  <https://arxiv.org/abs/2303.17564> and
  <https://arxiv.org/abs/2306.06031>
- HMM-style regime research supports the existing abstention policy: a regime
  model should reduce exposure, cool down, or wait during high-noise phases,
  not force a trade:
  <https://arxiv.org/abs/2007.14874>
- Lopez de Prado's Hierarchical Risk Parity work influenced the portfolio
  acceptance layer: individual profitable symbols are not enough if the
  accepted set is concentrated in one high-correlation cluster:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>
- Basel market-risk and expected-shortfall guidance influenced the current
  portfolio gate: diversification is treated as weak evidence when positive
  correlations make several symbols behave like one risk factor. The report
  therefore stores both plain and correlation-adjusted effective symbol counts,
  and the risk-level policy can reject a portfolio whose nominal symbol count
  is diversified but whose correlation-adjusted count is not:
  <https://www.bis.org/bcbs/publ/d352.pdf>
- The Basel market-risk backtesting framework influenced the tail-risk gate:
  the portfolio report measures VaR/CVaR-style losses and drawdown from the
  same aligned returns used for model-lab acceptance. Portfolio weights are
  actual cap-constrained equity weights, so any undeployed allocation remains
  cash reserve instead of being normalized into risky exposure:
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
- Binance spot and USD-M futures kline docs drive the full-history paging
  contract: spot klines are capped at 1000 rows per request, futures klines
  have limit-weight tiers, and the app records recent-limit versus full-history
  scope explicitly instead of treating both as equivalent evidence:
  <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints>
  and
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>
- Binance rate-limit docs influenced request telemetry and backoff handling:
  the client records used-weight/order-count headers when provided and respects
  `Retry-After` on retryable rate-limit responses:
  <https://developers.binance.com/docs/binance-spot-api-docs/websocket-api/rate-limits>

## Implemented Model Zoo

The base classifier remains the advanced logistic/GPU training path already
used by the CLI. The advanced feature vector now includes:

- multi-window technical rating votes inspired by TradingView's MA/oscillator
  aggregation structure,
- candle microstructure proxies inspired by order-book literature when only
  OHLCV candles are available,
- ATR-normalized trend and breakout features,
- volume-surge confirmation for high-frequency day-trading entries.
- market-quality regime features for trend efficiency, downside pressure,
  lagged return autocorrelation, volatility-of-volatility, volume pressure,
  volume/return correlation, ATR pressure, and current volume z-score.

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
Selected training-suite models also receive feature-group ablation replays.
The selected advanced feature vector is replayed with base features, extra
lookback windows, technical-confluence features, market-quality regime
features, nonlinear transforms, and polynomial interactions zeroed out one
group at a time. The report records
acceptance, score, P&L, drawdown, trade count, and delta versus the selected
model. This remains attribution evidence for model selection, and it is also
carried into `ai-review`: if the compact accepted report shows that removing a
hybrid expert or feature group improves the selected score, the AI review
deterministically vetoes before calling the local model.
The training suite also writes a `selection_risk` report for the selected
candidate and serialized model. This deterministic multiple-trials haircut uses
the explored candidate count plus local, ensemble, and hybrid-rescue checks to
deflate the selected score by observed score dispersion. A candidate is not
promoted unless the deflated score remains positive.
AI-assisted alpha has a separate deterministic uplift gate. When AI is enabled,
`ai-review` will not call the local LLM unless every accepted AI-assisted symbol
includes an `ai_uplift` artifact showing the AI-assisted holdout beats the
non-AI ML baseline on realized P&L and expectancy, does not worsen max
drawdown, does not introduce liquidations, does not worsen loss-streak,
profit-factor, win-rate, or downside return/risk evidence when those metrics are
available, has enough closed trades, was produced by a multibillion model, and
passes paired holdout statistical evidence. The paired gate requires enough
trade/window return deltas, a positive-delta rate above policy, an exact
one-sided sign-test p-value below policy, and a positive mean paired delta.
Missing or failed uplift evidence leaves AI in advisory/review-only mode.
After a candidate survives selection, the suite trains a compact meta-label
policy from the accepted model's simulated trade log. The policy
records the signal-strength thresholds that would take, downsize, or skip trades
under the current objective precision target and is persisted in both the model
artifact and `training_suite_summary.json`. Backtests, the legacy live loop, and
the autonomous loop now apply that policy as a deterministic pre-entry
skip/downsize gate. It cannot create entries or override exits, and malformed
enabled policies fail closed by skipping the entry.
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
Score-improving local, ensemble, and hybrid refinements are promoted only when
their validation/full-sample risk snapshot is non-degrading: max drawdown must
stay within tolerance, and P&L plus buy-and-hold edge cannot materially worsen.

Accepted hybrid candidates must improve or preserve the objective score and pass
the profitability, drawdown, and minimum-trade gates in
`ObjectiveSpec.accepts`. If no base candidate survives the hard gates, the
training suite now attempts a small fail-closed hybrid rescue pass over the top
rejected base candidates. A rescued hybrid is serialized only when it passes the
hybrid selection window, the final chronological holdout, and the full-sample
objective gates; otherwise the objective remains rejected.
Accepted hybrid reports also include an ablation table. The optimizer replays
the selected hybrid as base-only and with each expert family removed, then
records acceptance, score, and delta versus the selected hybrid. This makes
Lorentzian, rational-quadratic-kernel, and technical-confluence contribution
visible in `training_suite_summary.json` and model-lab outcomes instead of
trusting a blended score without attribution.

Model-lab also replays the final serialized model artifact across separate
chronological windows after training is complete. This differs from the
training-suite purged walk-forward gate: purged walk-forward retrains candidates
to select a stable configuration, while `temporal_robustness.json` tests the
exact saved model, including any hybrid expert overlay. Conservative models
must satisfy the strictest window coverage, regular models use the middle
threshold, and aggressive models allow more dispersion while still requiring
positive, non-drawdown-stopped windows. The same artifact now includes a
statistical edge gate: an exact one-sided sign test over the selected evidence
sample plus a deterministic bootstrap-style lower confidence bound over mean
sample return. This implements the practical lesson from
PBO/Deflated-Sharpe research: a high aggregate score is not enough when the
distribution of tested evidence still looks like selection luck. When the
backtest produces enough closed trades, the statistical gate uses net trade
returns; otherwise it falls back to chronological-window returns so sparse
strategies are still screened. The report keeps `positive_windows` and
`positive_window_rate` strictly window-level while `positive_samples` and
`positive_sample_rate` describe the trade/window sample used by the sign test.
Each temporal window is also tagged with deterministic market-regime evidence
such as dominant regime, confidence, trend return, realized volatility,
direction consistency, reversal rate, lag-1 autocorrelation, and optional volume
z-score. The objective and suite reports summarize accepted windows and P&L by
regime so model-lab can reveal when a candidate only works in one market state.
Backtest artifacts also record finite profit factor, expectancy, average trade
return, return dispersion, max consecutive loss streak, and positive-P&L
concentration so future objective gates can penalize fragile P&L profiles
instead of looking only at final cash.
The risk objectives now apply those gates when path evidence is present:
conservative requires profit factor above 1.10 with no loss streak above 3,
regular requires profit factor above 1.05 with no loss streak above 5, and
aggressive requires profit factor at least 1.00 with no loss streak above 8.
They also reject models whose positive P&L is dominated by one trade:
conservative caps the largest profitable trade at 55% of gross profit, regular
at 65%, and aggressive at 75%.
All three require positive expectancy.

Optimization-round reports must also pass a critical-analysis layer. A round
that completes but closes zero trades is classified as
`invalid_no_trade_abstention`; a flat strategy equity line is not profitability
evidence, even when the passive baseline is negative. Reports also fail when
there are no accepted symbols, no profitable symbols, or all strategy ROI values
are nonpositive. `tools/optimization_round.py` returns a nonzero exit code for
those verdicts so automation cannot accidentally treat a failed research round
as a successful optimization. Per-symbol metrics include threshold source,
decision threshold, model quality warnings, and meta-label policy reason so a
future no-trade round can distinguish ordinary low confidence from an explicit
fail-closed selection guard. Rejected selection gates are not promoted, but the
optimizer now keeps their final holdout diagnostic rather than installing an
impossible meta-label threshold that forces a flat equity line. This preserves
the real P&L/trade-count evidence while still marking the symbol as rejected.
Threshold calibration also records the best searched diagnostic threshold
separately from the selected safe threshold. When the best searched threshold
produces losing trades and is rejected, `best_closed_trades`,
`best_realized_pnl`, and the exported
`threshold_diagnostic_best_*` fields must still be reviewed; rejected evidence
is evidence of failure, not missing data and not a promotable strategy.
Round model search also includes controlled signal-threshold diversity:
lower-threshold frequency probes and higher-threshold conviction probes are
searched beside label-horizon and triple-barrier variants. These probes are not
lower safety standards; they only broaden the search surface before the same
closed-trade, positive-P&L, edge, drawdown, profit-factor, and expectancy gates
decide whether evidence can be accepted.
Automatic optimization universe selection is strict by default: symbols must
pass the strategy's live liquidity gates at selection time. Research-tier
symbols can be inspected only through an explicit opt-in code path and must not
silently fill a live-style optimization universe.

## Financial Sanity Gates

The repo now applies a separate financial-sanity layer before live-style model
readiness and before AI review. These checks are meant to catch malformed or
analytically incoherent artifacts before they reach an operator:

- model dimensions must match weights, means, and standard deviations,
- model weights, bias, calibration values, scores, drawdowns, coverage ratios,
  and AI uplift deltas must be finite,
- learning rate, L2 penalty, probability temperature, class weights, hybrid
  weights, and neighbor counts must stay inside hard numerical bounds,
- promoted/execution-validated models must include probability calibration
  evidence; calibrated Brier score above `0.35`, expected calibration error
  above `0.20`, or worsened calibrated loss blocks readiness,
- accepted model-lab outcomes must have positive rows and positive objective
  scores,
- raw generated backtests must pass financial sanity before `ObjectiveSpec`
  accepts them; malformed cash, fee, trade-count, trade-P&L, timestamp, return,
  exposure, exit-reason, win-rate, path-quality, or equity-curve identities add
  `financial_sanity_failed` to the rejection reason instead of letting positive
  P&L promote the candidate,
- accepted outcomes must include passed selection-risk evidence for every
  accepted objective; missing reports, nonpositive deflated scores, rejection
  reasons, failed PBO diagnostics, or unknown overfit status block promotion,
- accepted stress and temporal robustness reports must carry measured
  scenario/window/statistical-edge evidence; an accepted flag without those
  measurements blocks promotion,
- accepted coverage cannot have failed integrity, detected gaps, or impossible
  coverage ratios,
- accepted outcomes cannot bypass portfolio risk; missing or failed
  portfolio-risk evidence blocks the model-lab artifact,
- accepted portfolio symbol evidence must be non-empty, unique, and consistent
  across top-level accepted symbols, accepted outcomes, and the portfolio-risk
  report,
- accepted AI uplift evidence must include complete finite baseline, AI, and
  delta metrics, model-size evidence, and paired holdout statistical evidence;
  missing or weak uplift contract fields block the model-lab artifact,
- accepted stress, temporal robustness, and portfolio metrics must remain in
  financial ranges such as drawdown/CVaR/deployed-weight between 0 and 1.

This layer does not prove a model is profitable. It prevents bad math,
impossible metrics, and strange parameterization from being treated as valid
finance evidence.

## Cross-Symbol Model Lab

`simple-ai-trading model-lab` is the iterative optimization workflow. It:

1. Pulls exchange metadata, 24h tickers, and book tickers.
2. Automatically ranks high-liquidity symbols using quote volume, trade count,
   bid/ask spread, exchange status, and quote-asset policy.
3. Fetches klines for each ranked symbol. Default runs are recent-limit
   research/smoke runs; `--full-history` pages backward through venue maximum
   kline batches until the exchange returns no older rows.
4. Runs the training suite and hybrid optimizer for one or more objectives.
5. Records and serializes meta-label take/downsize/skip policy evidence from
   simulated trade outcomes for every selected objective model.
6. Requires the selected candidate to pass purged chronological walk-forward
   folds before serialization. The purge gap protects against label-lookahead
   leakage between train and test folds.
7. Replays every saved objective model under mandatory symbol-specific stress:
   baseline measured execution, wider spread/slippage, latency spike with a
   liquidity haircut, and combined liquidity crunch with fee/spread/latency
   stress.
8. Replays every saved objective model through `temporal_robustness.json`, a
   separate chronological-window robustness gate for the final serialized
   artifact. This catches models that pass aggregate stress but fail in recent
   or regime-specific windows. The temporal report also records statistical
   edge evidence, including sign-test p-value and bootstrap lower mean return,
   records market-regime concentration, and rejects candidates whose window
   evidence is too weak for the selected risk level.
9. Loads `data/autonomous/learning_feedback.json` when present, or an explicit
   `--learning-feedback PATH`, and blocks symbol promotion when repeated
   closed-trade losses for that symbol have not recovered in current stress and
   temporal validation. This is the bounded self-improvement path: it can veto a
   promotion, but it cannot mutate a live model, loosen risk, or alter open
   positions.
10. Builds a portfolio-level risk report from aligned symbol returns. This gate
   computes inverse-volatility capped equity weights, reserve weight, plain
   effective symbol count, correlation-adjusted effective symbol count,
   pairwise correlations, high-correlation clusters, portfolio 95% VaR/CVaR,
   and portfolio drawdown. Cash reserve is carried as zero-return exposure
   during VaR/CVaR and drawdown calculations.
11. Writes a JSON report plus per-symbol `stress_validation.json`,
   `temporal_robustness.json`, and `portfolio_risk.json`. An outcome is
   accepted only when all objective scores are positive, selection-risk
   deflation passes, every stress replay passes the objective risk gates,
   temporal robustness passes, data coverage has no hard integrity failure,
   learning feedback has no unresolved repeated-loss block, and the accepted
   set passes the portfolio diversification and tail-risk gates.

Every outcome now includes a `data_coverage` evidence block. It records symbol,
market type, interval, source scope, requested and used UTC date span, candle
counts, model-row count, gap count, largest gap, coverage ratio, full-history
flags, and a `truth_basis` list that explicitly says the execution results are
simulated rather than exchange fills. Missing candles, missing model rows,
coverage gaps, or coverage below `99.5%` fail promotion. Recent-limit API runs
are not hard failures by themselves, but they are labeled `binance_recent_limit`
and must not be presented as full-history optimization evidence.

Financial sanity re-checks that contract after model-lab writes the report.
An accepted outcome is blocked if `data_coverage` is absent, if the source
scope is missing or marked synthetic/fake/mock/demo/sample, if the source scope
does not identify Binance market data, if required truth-basis entries are
missing, if `candles_used` or `rows_used` is nonpositive, if coverage is below
`99.5%`, or if any measured gap remains. This is deliberate: ROI, drawdown,
selection-risk, stress, and robustness math are not considered financially
usable unless the underlying data evidence is complete and internally
traceable.

After model-lab writes a report, `simple-ai-trading ai-review --report ...`
can run a local structured-output model over a compact artifact summary. The
review is intentionally bounded and non-executing: it receives no credentials,
uses the AI capability preflight, requires GPU AI unless the user explicitly
changes runtime settings, validates the JSON schema, and writes
`ai_risk_review.json`. Missing accepted symbols, failed portfolio gates, failed
selection-risk deflation, positive hybrid/feature ablation deltas on accepted
outcomes, unresolved learning-feedback promotion blocks, failed AI preflight,
provider errors, missing/failed data-coverage evidence, or invalid model JSON
all produce a veto/review result instead of approval. The local model sees
compact data-coverage, selection-risk, hybrid, feature-ablation, and
learning-feedback summaries, but failed coverage integrity, failed deflated
scores, harmful positive ablation deltas, and unresolved repeated-loss blocks
are rejected in deterministic code before provider invocation.

Runtime startup also enforces promotion evidence. `live --live` loads the model
through a readiness gate that requires passing `selection_risk` evidence with a
positive deflated score, and the risk report exposes the same check under
`model promotion evidence`. Paper runs may regenerate an incompatible or stale
model for experimentation, but signed live-style execution cannot use a stale,
hand-edited, or legacy model artifact. Authenticated live mode also disables
in-loop retraining, because an ad hoc model trained during a live session has
not passed model-lab promotion, stress, robustness, ablation, AI-review, or
portfolio gates.

The same readiness gate now requires `execution_validation` on signed live
artifacts. `model-lab` stamps each serialized model after it runs
symbol-specific stress validation and final-model temporal robustness against
the selected liquid symbol, then after the portfolio risk gate is known. The
stamp records the symbol, market type, liquidity measurements, data-coverage
integrity, stress report path, temporal-robustness report path, portfolio report
path, accepted scenario/window counts, portfolio effective-symbol,
correlation-adjusted effective-symbol, CVaR/drawdown/correlation metrics, and
worst realized/drawdown metrics, plus any learning-feedback
recovery decision. A plain `train-suite` model may be useful for research, but
it is not signed-live ready until this execution, data-coverage,
learning-feedback, and portfolio evidence is accepted and persisted into the
model JSON. If individual symbols pass but data coverage, portfolio
diversification, or learning feedback blocks a symbol, model-lab stamps those
model files as not live-ready.

The selection-risk artifact now includes a two-panel CSCV/PBO-style diagnostic.
It ranks candidates by the selection panel and checks where the in-sample winner
lands on the validation panel, then repeats the symmetric validation-to-selection
view. Severe rank inversion, where both views show the in-sample winner falling
below the out-of-sample median, fails promotion even when the raw selected score
and deflated score look positive. This is not a full CPCV implementation over
many purged paths yet, but it is persisted evidence against the most common
backtest-overfit failure mode.

This is deliberately fail-closed. If live testnet data cannot produce a
profitable, diversified, risk-bounded candidate, the report should reject the
candidate instead of forcing a trade.

## Optimization Rounds

Every model-improvement round that changes feature, selection, or risk logic
should write an implementation note under `docs/optimization/`. ROI, P&L,
drawdown, and chart claims are allowed only when generated from exchange-sourced
backtests or signed testnet/paper artifacts with the provenance required by
[Data Provenance Policy](DATA_PROVENANCE_POLICY.md).

- [Round 001 - Market-Quality Regime Features](optimization/round-001-market-quality.md)
  adds the `v5-regime-quality` vector and risk non-degradation checks.
- [Round 002 - Learning Feedback Promotion Gate](optimization/round-002-learning-feedback-gate.md)
  blocks future promotion of symbols with repeated closed-trade losses unless
  the new candidate proves recovery under stress and temporal validation.
- [Round 003 - Full-History Data Coverage and API Efficiency](optimization/round-003-data-coverage-api-efficiency.md)
  adds full-history paging, data-coverage truth records, timestamped chart
  axes, and Binance request-weight telemetry so future optimization reports can
  be audited for timescale, gaps, row counts, and API cost.
- [Round 004 - Regime Entry Gate](optimization/round-004-regime-entry-gate.md)
  adds live/autonomous entry regime-unpredictability gates so the bot can wait
  through volatile chop, mixed low-separation regimes, short windows, or
  insufficient data instead of forcing trades.

## SuperZip Windows-App Alignment

The Windows app follows the SuperZip design direction:

- Native C++20 Win32 app instead of Tkinter.
- PowerShell build script that discovers Visual Studio, CMake, Ninja, and
  Python.
- DPI-aware Win32 layout with Segoe UI fonts, DWM dark caption colors, and
  real listbox, combobox, edit, and button controls.
- Dashboard-style operator shell with header health cards, primary workflow
  cards, safety controls, activity log, and bottom API-budget telemetry instead
  of an alphabetical command wall.
- Grouped operator workflows instead of an alphabetical command dump, while the
  CLI parity command picker still exposes every generated command.
- Repo-aware command launching that resolves `.venv311` and sets `PYTHONPATH`
  before running the Python CLI from a native app build.
- Generated command contract from the Python CLI so the GUI command list cannot
  drift from CLI capabilities.
- Explicit build, GUI smoke, screenshot capture, and automated
  control-navigation tests rather than manual launch assumptions.

The app is intentionally an operator console over the exact CLI contract. New
workflow parity must be added to the Python parser first; then
`tools/generate_windows_contract.py` regenerates the native header and tests
assert that every CLI command appears in the Windows app.

## Non-Negotiable Gates

- Hard product scope is BTC, ETH, and SOL only. Liquidity, spread, exchange
  status, data coverage, and archive integrity must still be measured from
  current exchange/archive data before any symbol is eligible.
- No mainnet trading by default.
- No leverage above 20x.
- No signed futures open order, local margin reservation, or bot-owned position
  ledger record may use leverage above the active Binance notional bracket for
  that order's intended gross notional.
- No live startup or reduce-only close may mutate exchange leverage; leverage
  changes are allowed only immediately before a fresh bot-owned futures open.
- No CLI, live, backtest, or optimization sizing path may let leverage raise
  gross exposure above the configured per-asset allocation cap.
- No signed non-dry operation may run with stop-loss protection disabled.
- No signed non-dry futures operation may run with the liquidation buffer
  disabled.
- No model, threshold, stress, temporal, market-edge, or optimization acceptance
  when a futures backtest records any liquidation event.
- No AI in CPU-only mode.
- No non-profitable accepted model-lab outcome.
- No selected training-suite model without purged walk-forward evidence when
  enough rows are available.
- No single-scenario-only model-lab acceptance.
- No model-lab symbol acceptance when the final serialized model fails
  chronological temporal robustness windows.
- No model-lab symbol acceptance when temporal windows have weak statistical
  edge evidence after selection.
- No fresh live or autonomous entry when rolling market-regime evidence exceeds
  the selected risk profile's `max_regime_unpredictability` threshold, or when
  the live cooldown is still active.
- No fresh live entry from malformed, non-finite, or out-of-range normalized
  market-regime risk scores.
- No model-lab symbol acceptance when the selected score does not remain
  positive after the multiple-trials selection-risk haircut.
- No model-lab symbol acceptance when closed-trade learning feedback shows
  repeated symbol losses and current stress plus temporal recovery evidence is
  not positive.
- No model-lab symbol acceptance when candle coverage is missing, has no model
  rows, contains detected gaps, or has coverage below the integrity threshold.
- No model or accepted AI-reviewed model-lab report when financial-sanity
  checks detect non-finite values, impossible dimensions, incoherent
  probability parameters, impossible row counts, or out-of-range risk metrics.
- No optimization report may claim full-history evidence unless its artifacts
  name the source scope, UTC date span, interval, symbol, row count, coverage
  ratio, and gap count.
- No model-lab acceptance when the individually passing symbols fail the
  portfolio-level correlation, concentration, CVaR, or drawdown gate.
- No AI review approval unless deterministic model-lab/portfolio gates passed,
  selection-risk evidence remains positive, hybrid/feature ablation evidence
  does not show that removing a selected component improves the accepted score,
  AI-vs-ML uplift evidence is present and accepted when AI is enabled, the
  local multibillion AI capability gate passed, and the provider returned valid
  structured JSON.
- No roadmap-only feature may be documented as executable unless the code,
  blueprint status, CLI/Windows parity surface, and tests agree.
- No signed live startup with a stale or unpromoted model artifact.
- No authenticated live in-loop retraining; retrain through model-lab and
  promote a fresh artifact.
- No promoted model when the selection-risk artifact reports severe PBO-style
  in-sample/out-of-sample rank inversion.
- No signed live startup when the model lacks accepted symbol-specific
  execution stress, temporal robustness, and portfolio-risk evidence.
- No signed live or authenticated autonomous startup when current Binance
  request-weight/order-count evidence is at or above the 80% startup threshold
  or the exchange has returned `Retry-After`.
- No signed live startup when one estimated stop-loss hit can exceed the
  tightest active daily, session, or portfolio risk budget.
- No signed live startup with stop-loss geometry that cannot produce a positive
  protective stop price for long exposure.
- No live ledger update from ACK-only order responses; `origQty`, requested
  size, and local fallback prices are not executed-fill evidence.
- No signed spot-roundtrip second leg from an ACK-only first-leg response.
- No generated backtest result may be used as optimization evidence if cash,
  fees, trade counts, exposure, trade-level P&L/return fields, path-quality
  summaries, liquidation counters, or equity-curve drawdown fail the financial
  sanity audit.
- No accepted market-edge report may bypass that generated-backtest financial
  sanity audit.
- No score-improving model refinement if the validation/full-sample risk
  snapshot materially worsens drawdown, P&L, or edge versus buy-and-hold.
- No autonomous post-outage resume until signed exchange exposure reconciles
  cleanly against verified bot-owned local ledger positions, hard daily/session
  loss budgets remain intact, and the reconnect observation cooldown has
  elapsed.
- No clean reconciliation from a malformed signed account payload; futures
  payloads must include `positions`, and spot payloads must include `balances`.
- No clean reconciliation from a corrupt or structurally invalid local
  `open_positions.json`; signed math treats that ledger as untrusted instead of
  silently assuming the bot is flat.
- No coordinator state may allow entries when required risk, execution,
  reconciliation, market-data, machine-learning, or AI heartbeats are stale or
  failed.
- No self-improvement loop may mutate a live model, loosen risk controls, or
  alter open positions; closed-trade learning feedback is evidence for the next
  model-lab/review cycle and can only make promotion stricter.
- No Windows-app-only workflow.
- No CLI-only workflow.
- Stop/pause controls must remain visible and tested.
