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

The 2026-07-10 research refresh adds three constraints to that roadmap:

- [LOBERT](https://arxiv.org/abs/2511.12563) is a message-level LOB foundation
  model. Its tokenization is relevant only after synchronized L2 messages exist;
  the official coarse percentage-band history is not relabeled as that input.
- [Time-series foundation-model benchmark requirements](https://arxiv.org/abs/2510.13654)
  identify overlapping or obscure pretraining corpora and temporal leakage as
  benchmark risks. Foundation candidates therefore require hash-pinned weights,
  known cutoffs, and genuinely post-cutoff rolling evidence.
- The [Finance Agent Benchmark](https://arxiv.org/abs/2508.00828) reports major
  limitations on real financial-research tasks, while
  [Agent Market Arena](https://arxiv.org/abs/2510.11695) finds that agent/risk
  architecture can matter more than backbone choice. The local multibillion LLM
  therefore remains a schema-constrained veto/advisory component. Coordinator,
  memory, and risk-style uplift must be ablated separately, and no LLM receives
  direct order authority.

## Causal L1/Tape Action-Value Model

The `microstructure-action-value-v16` workflow is a separate, fail-closed
research path for BTCUSDT, ETHUSDT, and SOLUSDT USD-M day trading. It does not
feed the legacy candle autonomous loop, and the repository currently has no
accepted v16 artifact or profitability result.

Its implemented lifecycle is:

1. `microstructure-train` rebuilds verified event-time BBO feature bars, joins
   real trade tape, and constructs 1-second causal features with path-aware,
   adverse-first stop/take labels, actual bid/ask execution, taker fees,
   top-of-book quantity, quote age, and a fixed decision cadence.
2. Training uses chronological train, early-stop, probability-calibration,
   policy, selection, and terminal regions. Every adjacent use of labels is
   separated by the horizon/latency purge. Early stopping and Platt calibration
   no longer share rows.
3. A selection pass produces only `candidate`. `microstructure-prequential`
   then performs complete rolling full retrains with fixed selected tree counts
   and hyperparameters. Each fold uses disjoint training, probability-
   calibration, policy-selection, and untouched evaluation intervals. The
   terminal region remains inaccessible.
4. Promotion evidence is canonical JSON plus row-level CSV and SVG. The report
   must use the locked protocol, cover every expected selection row exactly
   once, use the candidate's backend and source build, and pass after-cost,
   drawdown, confidence, baseline, AUC, trade-count, and profitable-fold gates.
   Changing protocol options produces diagnostic evidence that cannot promote.
5. `microstructure-promote` reconstructs the exact dataset, verifies every CSV
   action and non-overlapping fill against the source-bound targets, checks all
   artifact hashes, embeds a typed evidence binding, and only then reserves the
   one-use terminal range. Direct terminal evaluation, deployment refit, and
   accepted-model loading all reject a missing or drifted binding.
6. The terminal pass replays the same locked rolling full-retrain,
   recalibration, and prior-policy-selection protocol over the one-use terminal
   interval. Earlier terminal labels may enter later folds only after their
   exits and embargoes are available. A passing terminal replay produces
   `validated`, not a live model.
7. Deployment refit fixes the selected hyperparameters and tree counts.
   Provisional classifiers learn calibration on a purged recent tail; six final
   estimators then refit on all labeled rows. No terminal metric is used to
   retune the model.
8. A successful refit produces only `shadow_candidate`. It records validation and
   deployment estimator hashes, source build/fingerprint, backend, row counts,
   calibration span, training cutoff, and a hard expiry. A failed refit leaves
   the atomic `validated` artifact available for `microstructure-refit`.
9. `microstructure-shadow` captures at least 25,260 seconds from the locked
   Binance USD-M public depth, BBO, and aggregate-trade streams without loading
   credentials or exposing an order API. Capture schema `binance-usdm-l2-v3`
   hashes the original stream, synchronized stream, REST snapshot, and manifest.
10. Shadow replay uses the production streaming coordinator. A virtual entry is
    queued until the model's full exchange-time latency deadline, top-of-book
    participation is rechecked at that deadline, and exits use observed bid/ask,
    fees, adverse trigger slippage, stop/take barriers, and the validated horizon.
    A 3,600-second causal-feature warmup precedes six complete evaluated hours.
    The planned tail is entry-censored so every evaluated position can close
    naturally. Any feed gap, invalid event, feature reset, deadline miss,
    inference failure, expired/pending entry, forced close, or nonzero order count
    rejects promotion.
11. Only a profitable, positive-utility shadow with at least six complete hours,
    100 decisions, and 20 completed virtual trades produces `accepted`. Report,
    trade-table, capture, candidate, and deployment-estimator hashes are embedded
    in typed evidence and revalidated by the accepted runtime loader.
12. Streaming features use only closed seconds, while liquidity eligibility uses
   the newest independently tracked BBO known at inference time. The scorer
   blocks stale quotes, crossed/invalid books, excess L1 participation,
   unvalidated notional, cadence violations, expired models, and late signals.

Candidate generation does not consume terminal evidence:

```powershell
simple-ai-trading microstructure-train --symbol BTCUSDT --candidate-only `
  --stop-loss-bps 25 --take-profit-bps 40
simple-ai-trading microstructure-prequential `
  --input data/microstructure-model.json
simple-ai-trading microstructure-promote `
  --input data/microstructure-model.json `
  --prequential-report data/microstructure-prequential.json `
  --prequential-predictions data/microstructure-prequential-predictions.csv `
  --prequential-chart data/microstructure-prequential.svg
simple-ai-trading microstructure-shadow `
  --input data/microstructure-model.json `
  --report data/microstructure-shadow/report.json `
  --trades data/microstructure-shadow/trades.csv
```

The old `microstructure-train --evaluate-terminal` path is disabled because it
could retrain a different candidate before terminal access. The promotion
command is the only CLI terminal path. `microstructure-refit --input PATH` is a
recovery path for a terminal-validated artifact; it requires the same embedded
prequential binding and refuses a rebuilt or changed source even when filenames
are unchanged. Both promotion and recovery refits stop at `shadow_candidate`;
they never grant trading authority.

The mandatory no-order shadow measures current public-feed timing and virtual
cross-spread execution, but it still cannot measure the strategy's own market
impact, private order-entry latency, exchange acknowledgements, queue position,
or partial fills because it submits zero orders. Historical and shadow success
therefore do not guarantee live profitability. This limitation is one reason no
v16 artifact is currently accepted or claimed profitable.

### Official Tick-Source Coverage

`tick-archive-sync --full-history --plan-only` enumerates the official Binance
Data Vision S3 index independently for every symbol and product. It also writes
two independently listed, immutable inventory snapshots to the selected
warehouse without downloading an archive; a mismatch fails the plan. Date-bound
plan mode remains read-only. The workflow does not assume that all products
share a launch date or continue through the same day.
The machine-readable 2026-07-10 plan is
[`docs/microstructure/availability.json`](microstructure/availability.json).

The verified listing contains 11,847 files and 295,225,031,410 compressed bytes
(274.950 GiB). BTCUSDT and ETHUSDT trades begin in 2019, SOLUSDT trades begin in
2020, and all three continue through 2026-07-09. Coarse `bookDepth` snapshots
span 2023-01-01 through 2026-07-09. Exact `bookTicker` archives contain only 320
days for each symbol, from 2023-05-16 through 2024-03-30. Binance returns no
official BBO archive after that date.

`bookDepth` rows contain cumulative depth/notional at percentage bands such as
`-0.20%` and `+0.20%`; they are real liquidity observations but are not best bid,
best ask, queue, or spread observations. The software must not relabel them as
BBO or interpolate an invented spread. Long-history research therefore keeps
trade-tape/depth features separate from the shorter BBO model, and any current
execution claim still requires a fresh public-feed shadow. Full corpus ingestion
is in progress; the compact listing plan is not a claim that all files are
already present in the local warehouse.

The warehouse persists each official S3 listing as an immutable inventory
snapshot before and after ingestion, or twice during a metadata-only full-history
plan. Every inventory item includes the ZIP and
`.CHECKSUM` object's opaque S3 ETag, `LastModified`, and exact byte size; the two
snapshots must be identical. A full-history run succeeds only after a corpus
certificate reconciles every listed UTC-day partition against one current
manifest with the exact official URL and compressed byte count, a matching
Binance SHA-256 sidecar, a supported schema, positive raw and derived row
counts, valid exchange-time bounds, and no invalid, duplicate, crossed-book, or
uncanonicalized trade/depth rows. It also reconciles manifest counts and time
bounds against the physical raw and derived DuckDB partitions, validates the
materialized 100 ms BBO execution path, and rechecks coarse-depth band groups.
The certificate records per-product launch and end dates, each provider-side
calendar gap, and the exact common calendar intersection; it never assumes that
BBO, trades, and coarse depth share coverage. The live 2026-07-10 inventory is
calendar-complete for all three trade feeds, while Binance omits some coarse
`bookDepth` dates (including 2023-02-08 and 2023-02-09 for all three symbols).
Those gaps remain explicit in requested-window certification. The tape/depth
builder can admit them only through its `bookDepth`-specific,
provider-proven gap exception. It records the exact dates and emits unavailable
depth after the age limit; it never fills them with synthetic data or stale
liquidity. Both the executable L1 dataset and the long-history
tape/depth dataset require this certificate. An archive plan, downloaded ZIP,
or manifest count alone is insufficient.

Repeat syncs do not redownload an unchanged 275 GiB corpus. Before reuse, the
warehouse batches one physical integrity scan per symbol/product and requires
the exact official URL and byte size, matching ZIP and checksum-object ETags, S3
`LastModified` times no newer than the verified ingestion, matching
source/sidecar hashes, current schema, intact row counts and bounds, and valid
derived partitions. Changed or damaged partitions are excluded from reuse. The
fallback path revalidates physical evidence again and atomically replaces the
partition from the checksummed ZIP; it can no longer return a corrupt completed
manifest as `skipped`. The final corpus certificate remains the authority.

Corpus certificate v3 computes the actual shared calendar overlap from each
product's independent launch and end boundaries. For every hole in that overlap
it records the exact absent product. A hole can pass only when all absent
products are explicitly permitted provider-side `bookDepth` gaps; the exception
never applies to BBO, trades, a listed archive missing locally, an invalid
checksum, or damaged physical rows. Full-history sync and the default
`tick-corpus-audit` use this provider-evidence exception and expose it in JSON;
`tick-corpus-audit --strict-book-depth-calendar` disables it. This makes a
certificate a proof of completeness through its recorded UTC cutoff, while the
normal incremental sync handles newly published days later.

This closes a previous provenance weakness: a missing trade archive can no
longer be interpreted as a day of genuine zero order flow. Action-value v16
requires the exact official BBO and trade products it consumes; coarse depth is
certified separately by the tape/depth lane instead of being imposed on a model
that does not use it. All older action-value artifacts are invalid under v16.

V16 retains the executable target equation sealed in v15. For entry bid/ask
`B0/A0` and exit bid/ask `B1/A1`, long gross return is `B1/A0 - 1` and linear
short gross return is `1 - A1/B0`. If `c` is the per-side sum of taker fee and
additional all-trade slippage stress in basis points, net cost is
`c * (1 + exit_notional_ratio)`. Stop/take trigger slippage is not folded into
`c`; it first moves the observed exit price adversely, after which the same
notional-scaled cost equation is applied. The contract is shared by training,
prequential validation, terminal evaluation, deployment refit, and no-order
shadow replay. The CLI defaults the additional stress to 1 bps per side and
serializes it in the model artifact. V16 training asks LightGBM for deterministic
CPU execution or FP64 OpenCL accumulation to reduce accelerator variance.
The promotion floor defaults to 240 observed UTC days because the official BBO
archive itself spans only 320 days. This is a source-imposed bound, not a claim
that 240 days replaces multi-year validation: multi-year trade/depth forecasts
remain a separate research lane until they can be joined to defensible execution
evidence without inventing historical quotes.

Feature contract `l1-tape-causal-v8` expands the shared offline/live vector to
107 features. It retains the v7 causal 1,800/3,600-second return, volatility,
range, path-efficiency, spread, quote-intensity, trade-intensity, volume, UTC
weekly-phase, and weekend context, then adds bid/ask L1 quote depth, L1 depth
relative to 60/300-second history, and signed 10/60/300-second pressure against
opposing L1 depth. These are context variables, not fixed trading-hour
prohibitions. DuckDB and the streaming coordinator use the same ordered vector
and require 3,600 consecutive closed seconds. The promotion shadow therefore
captures at least 25,260 seconds so warmup is followed by six complete evaluated
hours and tail margin. This responds to measured
regime-transfer failure rather than presuming new alpha:
[TLOB](https://arxiv.org/abs/2502.15757) reports declining cross-condition LOB
predictability and worse results when transaction costs define the target,
while [concept-drift research](https://arxiv.org/abs/2304.01512) motivates
explicit adaptation to nonstationary time-series distributions.

V16 also separates statistical roles in hurdle-class support. Training and
probability calibration each require 256 profitable and 256 non-profitable
eligible rows per side; early stopping requires 64 of each because it selects
tree count rather than estimating the final policy. Every trained artifact
persists all six counts and their role-specific floors. A failure reports the
exact side, role, observed counts, and required minimum. Historical v15/v6
designs remain readable for publication, but the execution runner refuses them;
new experiments must use the current v16/v8 and design-v2 contract.

Reproduce the plan without downloading data:

```powershell
simple-ai-trading tick-archive-sync `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT `
  --data-types bookTicker,trades,bookDepth `
  --full-history --plan-only `
  --plan-output docs/microstructure/availability.json
```

### Long-History Tape/Depth Forecasting

`tape-depth-train` is the implemented research lane for the longer trade and
coarse-depth history. It builds a causal one-second feature matrix with trade
returns, realized volatility, aggressor flow, trade counts, volume, exact UTC
cycle features, the observed `0.20%`, `1%`, and `5%` cumulative depth bands, a
depth-age mask, and depth-curve shape. The depth join is backward-looking only;
a depth snapshot newer than the decision second is never used.

The v4 feature contract also includes causal VWAP deviation, bounded price-path
efficiency, observed-trade rates, short/long quote-volume and trade-intensity
acceleration, and price/flow alignment. These are interpretable
microstructure candidates, not presumed alpha. Their value must be established
by profile and feature-group ablations on the earlier rolling folds and then
confirmed on later untouched folds.

V4 adds a point-in-time cross-asset context block across BTCUSDT, ETHUSDT, and
SOLUSDT. At each feature second it computes 1, 5, 15, 60, and 300-second peer
returns, peer dispersion, target-relative returns, and a BTC-anchor return by
backward as-of joins only. A complete-context flag and maximum peer-trade age
make missing or stale context observable; unavailable values are zeroed only
alongside that false flag. The dataset evidence hash includes the exact peer
trade manifests and stops at the final feature second, not the later label
horizon. This is a testable hypothesis motivated by short-lived lagged
cross-asset effects, not presumed alpha. Cont, Cucuringu, and Zhang report that
lagged cross-asset order-flow imbalance can improve short-horizon return
forecasts while contemporaneous cross-impact adds little after strong local
order-flow features; CryptoGAT separately motivates testing crypto cross-asset
graphs. Both require independent rolling confirmation here:

- https://arxiv.org/abs/2112.13213
- https://arxiv.org/abs/2606.27670

The source clock is complete even when no trade occurs in a particular second.
Such a row carries the most recent verified trade reference for OHLC, records
zero volume and zero trade count, and exposes `trade_observed=0` plus the exact
trade age. It does not invent volume or a transaction. This fill is permitted
only inside a continuous sequence of checksummed daily trade manifests; every
requested UTC date must have a verified manifest, so an absent archive fails
instead of becoming a long carry-forward interval.

The target is deliberately narrow: the real trade-reference return from the end
of the configured latency delay, rounded up to the next observable one-second
boundary, to the exact future horizon. The effective delay and complete target
span are persisted with the artifact. It is a gross
forecast target, not a synthetic spread, executable fill, queue estimate, or
after-cost PnL. A purged chronological train/tune/calibration/evaluation split
fits LightGBM direction, Huber mean-return, and 10th/90th-percentile models,
then records AUC, Brier score, MAE, RMSE, Spearman information coefficient,
interval coverage, and calibration-threshold signed gross return against simple
baselines.

The v8 learner fits every data-dependent policy statistic before evaluation.
The 90th-percentile return-magnitude scale used for sample weights comes only
from exact float64 training targets and is then frozen for train/tune refitting;
calibration and evaluation targets cannot alter model weights or tree strings.
The risk-specific decision policy is also calibration-only. Conservative uses
the 95th percentile of absolute mean forecasts, the 95th percentile of absolute
calibrated probability distance from `0.5`, and the 75th percentile of
forecast-interval width. Regular uses `0.90/0.90/0.90`; aggressive uses
`0.80/0.80/0.98`. The directional quantile becomes a serialized probability
floor of `0.5 + margin`; it does not read evaluation sharpness. Long and short
selection require the mean forecast sign and calibrated
direction model to agree, and interval width must remain below the frozen risk
limit. Calibration prevalence is the fixed Brier/majority baseline; evaluation
prevalence is never read to construct a baseline. Every policy field is
serialized in each row-level prediction table. Replay refits the Platt transform
and recomputes the complete policy from bound calibration rows before scoring.
An evaluation regime may still produce zero actions and be rejected; the report
does not manufacture activity. Per-fold viability requires only five actions or
0.1% of decisions, whichever is larger, while long/short balance is judged over
the multi-fold segment so a one-directional short regime does not force a bad
countertrend trade. This follows the standard leakage rule that learned
transforms and decision thresholds must not be fitted on test observations.

### Multi-Fidelity Candidate Search

`model_experiment.py` provides the precommitted candidate-design and
successive-halving contract for model-research runs. `tape-depth-design` writes
that immutable design before screening, and the sealed `tape-depth-select` path
requires every declared candidate exactly once. It preserves three explicit
tape/depth anchors, stratifies every candidate
dimension with a deterministic randomized Latin hypercube, fingerprints the
complete design, and counts anchors, failures, and eliminated variants in the
cumulative trial burden. The initial space covers forecast horizon, decision
cadence, maximum coarse-depth age, model capacity, and feature-group ablations
for one risk profile at a time. It does not tune latency, fees, spread, or source
quality downward to manufacture performance.

The horizon domain includes `5`, `10`, `15`, `20`, `30`, `60`, `120`, `300`,
and `900` seconds. A real-data discovery screen on 2024-03-15 moved the anchors
to a 20-second regularized cross-asset candidate, a 5-second regularized
tape-derived candidate, and the existing 300-second long-horizon control. This
is search-space allocation, not promotion evidence: the discovery date is
explicitly selection-contaminated and excluded from the immutable
[`confirmation-design.json`](model-research/tape-depth/confirmation-design.json).

`tape_depth_execution.py` is the mandatory exact-BBO diagnostic for a frozen
gross survivor. It greedily suppresses overlapping same-symbol positions, joins
only 100 ms quote buckets available by the modeled arrival time, crosses the
observed ask/bid, subtracts two-sided taker fees and stress slippage, rejects
stale/crossed/missing quotes, and caps order size by observed L1 quantity. It
scales each cost leg by its observed notional and uses cash PnL divided by entry
notional for both long and short returns. It cannot claim maker execution because
historical queue position is absent. In the reproducible v8 replay of the
discovery date, the selected 20-second conservative forecast produced 15 gross
signals at `+5.5730` bps mean trade-reference return. Overlap suppression left
6; the 10% L1 cap rejected 2; the remaining 4 averaged `+4.3617` bps on the
actual quote path and `-5.6385` bps after 5 bps taker fees per side. With an
additional 1 bps slippage per side, mean net was `-7.6385` bps. Every scenario
was rejected and carries no profitability or execution claim.

`tape-depth-execution-confirm` runs the hash-bound design without accepting
timing, cost, model, or date overrides. It writes a deterministic plan, one
no-overwrite checkpoint per UTC period, the serialized model, compressed
row-level predictions, actual quote-path rows, and a weighted aggregate gate
report. Resume verifies checkpoint fingerprints and model/prediction file hashes
before reuse. Aggregate counts, mean net return, hit rate, and rejection classes
are recomputed from the row evidence rather than trusted from summaries.

Round 8 opened the three frozen dates only after the design and runner were
pushed. None of the three gross artifacts passed all forecast gates. Exact BBO
replay produced 1, 0, and 11 executable trades; all 12 were net-negative. The
weighted mean was `-11.839347` bps after two 5 bps taker-fee legs and 1 bps
stress slippage per leg. The aggregate failed five precommitted gates and is
rejected. Hash-manifested source tables and graphs are retained in
[`model-research/tape-depth/latest`](model-research/tape-depth/latest/README.md).

Round 9 was precommitted at implementation commit `8a0eec2` before opening its
seven-day BTCUSDT selection window (`2023-08-14` through `2023-08-20`). Its
bounded corpus certificate reconciled seven official daily `bookTicker` and
seven official daily `trades` archives, including Binance sidecar checksums and
physical warehouse partitions. The run consumed 81,684,026 BBO events and
20,111,284 trades to construct 604,752 causal one-second feature rows. No
terminal interval was opened.

Seven of the 12 predeclared candidates lacked the required profitable and
non-profitable class support after executable spread, two 5 bps taker-fee legs,
two 1 bps additional-slippage legs, and 750 ms latency. The five trainable
300/900-second candidates had selection long AUCs from `0.54096` to `0.59962`
and short AUCs from `0.55750` to `0.60297`, but their mean executable labels
were all negative: long from `-13.30` to `-12.51` bps and short from `-11.62`
to `-10.78` bps. Although 17,858 short rows had positive model-predicted edge,
every non-overlapping threshold policy that used those rows had non-positive
realized drawdown-adjusted utility on its policy segment. The fitted policy
therefore abstained, all five artifacts were rejected, and no zero-trade return
or equity curve was fabricated. Its immutable publication is commit `da8f9f5`;
the cumulative progress table retains the round while the `latest` directory
contains only the newest round, as required by the repository policy.

This result is consistent with two constraints used in the implementation:
[LOBFrame](https://arxiv.org/abs/2403.09267) cautions that predictive scores do
not by themselves establish an actionable strategy, and a recent
[cost-aware crypto study](https://arxiv.org/abs/2606.00060) reports that
transaction costs can reverse apparently useful short-horizon signals. A
post-round decomposition also found that the bounded Newton implementation of
Platt scaling could drive a badly underconfident class estimate to its lower
parameter bounds instead of improving calibration. That defect is remediated
with base-rate initialization and loss-decreasing damped Newton steps, but the
sealed Round 9 evidence remains unchanged and the dates are permanently
selection-consumed.

Round 10 was precommitted in `6f291ff` and pinned the calibration repair at
`58e6ac5` before opening `2023-09-04` through `2023-09-10`. Corpus certificate
`5782bd80...39ab` reconciles exactly seven official BBO and seven official
trade archives against full inventories of 320 and 2,497 files. The bounded
window contains 50,579,048 BBO events and 14,341,489 trades; every sidecar hash
matched and every invalid, duplicate, ordering, crossed-book, and update-ID
regression count was zero.

Nine candidates failed before fitting because the old support gate required
256 examples of both outcomes in each train, early-stop, and calibration role.
This was not a general data shortage. For example, regular 900 seconds had
5,353 profitable long training rows and 2,126 calibration rows, but only 201
long and 116 short early-stop positives. The three 1,800-second candidates
trained successfully. Selection long AUC ranged from `0.46421` to `0.60627`
and short AUC from `0.28943` to `0.56122`; all mean long/short executable labels
remained below `-12.28` bps. The models produced 158 positive predicted-edge
policy rows, but no policy met the 20-trade minimum with positive
drawdown-adjusted utility, so all artifacts abstained and were rejected.

The deterministic post-round diagnostic does not reclassify top-score rows as
trades and does not access terminal data. It shows why threshold relaxation
would be wrong: the regular 1,800-second model's top 100 selection rows predicted
`+1.8338` bps mean edge but realized `-15.5390` bps with zero profitable rows.
The aggressive top 100 realized `-27.0188` bps with zero profitable rows. These
counts, class-support tables, model hashes, and top-score outcomes were bound to
diagnostic SHA-256 `7ca872c2...ea90`. Under latest-only retention, the Round 10
decision remains in the cumulative
[`progress.csv`](model-research/action-value/latest/progress.csv), while the
per-round bundle has been replaced by the newest round. Round 10 is not
profitability evidence.

Round 11 was precommitted in `e7ed78c` before opening its 42-day BTCUSDT
window (`2023-08-14` through `2023-09-24`). Corpus certificate
`3a3851f5...191e` binds 42 BBO and 42 trade partitions to stable full
inventories of 320 and 2,497 official files. The warehouse reconciled
423,616,292 BBO updates and 110,369,564 trades with zero missing or invalid
scope partitions, then produced 3,601,537 causal one-second feature rows. The
complete window is recorded as consumed in immutable registry
`consumed-periods-through-round-011.json`; no date in it remains terminal
evidence.

All 12 v16/v7 candidates trained without class-support failures on the AMD
OpenCL path. Selection long AUC ranged from `0.56649` to `0.79777`, and short
AUC ranged from `0.58232` to `0.76372`. Predictive ranking still did not become
an executable policy: mean long labels ranged from `-12.8170` to `-12.1752`
bps, mean short labels from `-12.1421` to `-11.7518` bps, and 1,468 positive
predicted-edge policy rows produced no threshold with the required
non-overlapping trade count and positive drawdown-adjusted utility. Every
candidate abstained and was rejected; selected-trade return and an equity curve
are therefore undefined.

The consumed-selection diagnostic separates an interesting ranking result from
a valid trading result. Aggressive 900-second top-100 selection rows averaged
`+24.1806` bps with a 77% profitable-label ratio, and aggressive 1,800-second
top-100 rows averaged `+9.4127` bps with an 88% ratio. Their policy-period
top-100 results were only `-1.5940` and `+1.9305` bps, respectively; after
non-overlap, neither policy had the minimum 20 trades. Retrospectively choosing
the better selection tail would be leakage. Two independent reconstruction
runs were byte-identical at file SHA-256 `31a7b553...c8f6f` and canonical
diagnostic SHA-256 `b7f67cb0...3adb`. The Round 11 decision remains in the
cumulative [`progress.csv`](model-research/action-value/latest/progress.csv);
its source tables and charts were superseded under latest-only retention. Round
11 is evidence of regime-transfer and value-calibration failure, not
profitability.

A deterministic consumed-data ablation then tested whether the failure was
limited to the hurdle expected-value formula. On aggressive 900 seconds it
compared the hurdle score, calibrated profitability probability, direct mean,
10th/90th conditional quantiles, downside-adjusted mean, and a reference-rank
ensemble. It also evaluated four causal volatility-scaled CUSUM event filters,
strictly prior-day threshold updates over four lookbacks, and seven event-only
score families from eight estimators trained with average label-uniqueness
weights. The event train roles had
31,686 eligible long and 31,622 eligible short rows; class support remained
ample after removing clock-tick duplicates.

All 14 score families failed policy selection. No static policy was accepted,
none of the 56 prior-only adaptive configurations traded, and none of the 28
CUSUM-filter policy combinations passed. The probability score illustrates why
row-level ranking is not enough: its policy top 100 averaged `-34.3739` bps,
while its consumed selection top 100 averaged `+44.5879` bps with a 98%
profitable-label rate. Those rows were clustered; after chronological
non-overlap, every threshold still had negative policy utility. Event-only
model utility was also negative for all seven scores, from `-496.03` to
`-1,099.65` bps.

Two complete 13-method AMD/OpenCL reruns were byte-identical. The added
distributional score reused the same direct/event model fingerprints; its
regenerable full source JSON has SHA-256 `3bc2f230...9bd6`, canonical ablation
SHA-256 `ca92dff4...ac1e`, and the tracked compact summary has SHA-256
`2d072da6...fd34`. The source-bound summary is retained under
[`model-research/action-value/ablations`](model-research/action-value/ablations/round-011-aggressive-h900-score-ablation-summary.json).
This rejected event gating and uniqueness weighting as sufficient fixes on the
42-day sample. At that point, the proposed next step was a full-history build.

### Round 12 bounded viability

Round 12 tested that assumption before committing to a much larger build. The
v6 design fixed a 52-day BTCUSDT window (`2023-05-16` through `2023-07-06`),
five chronological roles, 300/900-second horizons, three risk profiles, and
three distributional score methods. The checksum-bound corpus certificate is
`975692f0...c33bd`; the one-time causal build reconciled 747,033,065 BBO
updates into 5,532,689 one-second bars. Six AMD/OpenCL fits produced all 18
precommitted candidates with no fit failures.

Every policy tail was negative after 5 bps/side fees, 1 bps/side added
slippage, and 750 ms latency. The best lost `50.6911` bps over 31 trades with
`170.5717` bps max drawdown and profit factor `0.8804`; the worst lost
`621.9029` bps. Each candidate's top 100 selection rows also had a negative
mean outcome (`-3.0395` to `-15.3647` bps). No policy was accepted, no
selection trade was permitted, and leverage was not applied. The July 7
terminal day remains untouched; the evaluated window is sealed in
`consumed-periods-through-round-012.json`.

Round 12 rejected that feature/model/execution family as a viable trading
baseline and did not justify a full-history scale-up or an AI overlay. Its
decision remains in the cumulative
[`progress.csv`](model-research/action-value/latest/progress.csv); latest-only
source tables, hashes, and charts describe only the newest retained round.

### Round 31 frozen chronological confirmation

Round 31 evaluated the exact three-seed Round 30 LightGBM hurdle ensemble and
all twelve frozen thresholds without retraining or recalibration. Its first
precommitted stage covered `2024-01-01` through `2024-02-04` UTC and produced
209,878 valid adaptive-barrier outcomes from official Binance trades and exact
top-of-book archives. Long/short ROC AUC was `0.5783`/`0.5839`, but the ranked
forecasts did not provide acceptable after-cost risk-adjusted performance.

The strongest threshold produced `+79.5577` bps under the stress simulation
over 28 simulated trades, but maximum drawdown was `371.5068` bps and the worst
simulated trade was `-148.4389` bps. It failed the drawdown, positive-day-ratio,
and worst-trade criteria. Every conservative, regular, and aggressive candidate
was rejected, so policy and development targets, predictions, and metrics were
withheld. The reserved `2024-03-30` terminal date was not ingested, queried,
labeled, predicted, or evaluated. Its decision remains in the cumulative
[`progress.csv`](model-research/action-value/latest/progress.csv); the per-round
bundle was superseded under latest-only retention. The current
[`action-value/latest`](model-research/action-value/latest/README.md) bundle is
Round 58. Its value-blind support probe used official BTCUSDT, ETHUSDT, and
SOLUSDT events from 2023-06-01 and rejected symmetric touch making before
training: two-sided fills were only `2.36-3.18%`, one-sided fills were
`28.03-47.19%`, and every two-fill spread p99 was below `1` bps against the
prior frozen `4` bps fee and `6` bps fee-plus-slippage references. The probe
read no returns, costs, P&L, outcomes, or policy thresholds. Round 57 remains
in the cumulative record: queue-fill survival generalized, but all 12
evaluation payoff top quintiles were negative after its frozen costs. None of
these records grants leverage, profitability, AI uplift, execution, or trading
authority.

The v8 backend opts this model family into reproducible training. CPU uses
LightGBM's `deterministic=true` with forced column-wise histograms. OpenCL uses
`gpu_use_dp=true`, LightGBM's
[documented mitigation](https://lightgbm.readthedocs.io/en/stable/FAQ.html)
for non-reproducible GPU histogram sums. On the AMD validation host, two
consecutive real-data fits had
identical serialized-model and evaluation-prediction SHA-256 fingerprints.
Artifacts still record the resolved backend because exact fit identity is not
promised across library builds or device architectures.

`tape-depth-study` operationalizes the sealed screening level. It executes one
candidate at a time, forwards fold-level progress, checkpoints after every
candidate, verifies model/prediction/report evidence before reusing a completed
candidate under `--resume`, and calls the same design-bound selector only when
the declared design is complete. It does not run confirmation or consume the
terminal suffix. `--plan-only` validates and materializes the candidate plan
without opening the warehouse.

Candidate evaluation uses four chronological resource levels:

1. A causal viability screen over several precommitted, non-overlapping short
   windows spread across BTC, ETH, and SOL history. This can reject inactivity,
   clearly negative after-cost expectancy, liquidation, incomplete costs,
   missing source evidence, one-sided degeneration, or excessive drawdown. It
   cannot promote a model or support a profitability statement.
2. Wider cross-regime selection requiring more trades, a positive-window
   majority, nonnegative expectancy, and risk-profile drawdown limits.
3. Rolling prequential evaluation with stronger profit-factor, loss-streak,
   side-balance, activity, and cross-symbol gates.
4. Full certified-history validation for the small frozen survivor set.

Every stage emits `research_only_no_trading_authority` and explicitly records
that it did not consume the terminal holdout. The separate one-use terminal,
deployment-refit, and public-feed shadow lifecycle remains mandatory. An AI
overlay will enter this process only after an executable ML baseline survives;
it must be compared on paired periods under the same fills and costs and show a
positive uplift without worse drawdown or liquidation evidence.

The implementation adapts, rather than blindly invokes, the following methods:

- [SciPy Latin-hypercube documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html)
  defines stratified space-filling marginals. The repository uses a dependency-
  free hash-deterministic equivalent so design replay is stable on Windows.
- [scikit-learn successive-halving documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html)
  motivates allocating larger resources only to survivors. Ordinary shuffled
  folds and generic estimator scores are not used; resources are certified
  chronological market windows and gates are after-cost trading/risk metrics.
- Bailey, Borwein, Lopez de Prado, and Zhu's
  [Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
  motivates retaining the complete selection-trial burden. Halving reduces
  compute; it does not erase multiple-testing risk.
- Binance's [official public-data repository](https://github.com/binance/binance-public-data)
  documents daily/monthly archives, timestamp changes, archive corrections,
  and SHA-256 sidecars used by the corpus certificate.
- Binance's [official USD-M book-ticker stream reference](https://developers.binance.com/legacy-docs/derivatives/usds-margined-futures/websocket-market-streams/All-Book-Tickers-Stream)
  defines the best bid/ask price and quantity fields. Historical replay treats
  them as L1 only and never infers deeper fills or queue position.
- AWS's [official S3 ListObjects reference](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/list_objects.html)
  defines `ETag`, `LastModified`, and `Size`. The implementation treats ETag as
  an opaque content-version identifier because multipart ETags are not MD5
  digests.

```powershell
simple-ai-trading tape-depth-train `
  --symbol BTCUSDT `
  --warehouse data/microstructure.duckdb `
  --window-days 365 `
  --horizon-seconds 60 `
  --total-latency-ms 750 `
  --decision-cadence-seconds 5 `
  --model-profile regularized `
  --feature-set full `
  --compute-backend directml
```

The row limit, DuckDB memory limit, thread count, progress phases, and optional
end date keep training bounded and reproducible. Every source manifest and
archive checksum contributing to the window is embedded in the dataset evidence.
Artifacts can only be `research_candidate` or `rejected`; both carry
`trading_authority=false` and `execution_claim=false`, and the loader rejects a
forged authority field. Model selection still requires full-corpus,
cross-symbol, rolling out-of-sample comparison. Any order-capable descendant
must independently pass exact-BBO replay and the current no-order shadow.

Prediction capacity is not a risk-profile proxy. `regularized`, `balanced`, and
`expressive` are explicit model-profile trials, while conservative, regular,
and aggressive remain downstream execution policies. Holding model profile,
data, and seed constant while changing execution risk level must produce the
same model strings and forecast metrics. This prevents a risk label from hiding
an uncounted hyperparameter trial.

Feature growth is governed by four ordered ablations: `core` contains the
basic causal tape, volatility, range, activity, clock, and no-trade state;
`tape_derived` adds VWAP, path-efficiency, intensity-acceleration, and
price/flow-alignment candidates; `cross_asset` adds the point-in-time peer and
BTC-anchor block; `full` adds the causally joined coarse-depth bands. The
artifact stores the exact model input order, while its dataset hash binds the
complete matrix. A more complex set must improve earlier rolling folds and
later confirmation folds; otherwise the simpler set remains the candidate.

The rolling comparison is implemented by `tape-depth-prequential`. The default
calendar protocol uses 730 days for training, 30 days for early-stopping tuning,
30 days for probability calibration, and the next non-overlapping 90 days for
screening evaluation. Its 20-second decision cadence keeps each 880-day fold below the
five-million-row memory gate while all return, volatility, flow, depth, and
target inputs retain one-second source resolution. Fold boundaries are exact
UTC timestamps rather than percentages, and every preceding label whose exit
reaches or crosses a boundary is purged from that segment.

The 90-day outer block is an efficiency and robustness screen: it avoids
rebuilding nearly identical 820-day matrices every month and requires a frozen
candidate to remain useful across a full quarter. Only candidates that survive
the complete quarterly screen should incur a separate monthly-refit replay and
current shadow. This reduces redundant GPU work without dropping any source
seconds or hiding monthly behavior, because the row-level quarterly predictions
can still be grouped into calendar months.

Within a run, the causal one-second window query is executed once for each
remaining symbol rather than once per overlapping fold. Only cadence-aligned
decision rows are retained, under the separate 15-million
`--maximum-cached-rows` bound. Each fold is a zero-copy contiguous matrix view,
but its dataset fingerprint is rebuilt with the fold's exact checksummed
manifest subset. Feature reuse therefore removes redundant computation without
reusing a later label, calibration row, or source binding.

The complete prefetched matrix is also stored by default as a derived dataset in
the same DuckDB warehouse. The version-specific cache table uses DuckDB's
persistent column compression; no loose NumPy dataset is created. Its SHA-256
key includes the exact target and peer manifests, feature names/version, UTC
range, horizon, latency, cadence, and depth-age contract. A single transaction
writes rows and manifest, and every load recomputes the complete dataset
fingerprint before training. `dataset-cache-events.json` survives resume and
records hit/write/disabled state. `--no-dataset-cache` disables persistence but
does not weaken source verification or the in-memory row bound.

```powershell
simple-ai-trading tape-depth-prequential `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT `
  --training-window-days 730 `
  --tuning-window-days 30 `
  --calibration-window-days 30 `
  --evaluation-window-days 90 `
  --decision-cadence-seconds 20 `
  --plan-only

simple-ai-trading tape-depth-prequential `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT `
  --compute-backend directml `
  --resume `
  --output-dir data/tape-depth-prequential-full
```

The run writes `plan.json`, continuously updated `run-status.json`, one exact
model artifact and deterministic gzip prediction table per fold,
`fold-metrics.csv`, `report.json`, and `forecast-diagnostics.svg`. Every table,
model, prediction batch, plan, and graph is hash-bound. Reported model metrics
are recomputed from the serialized LightGBM strings against exact float64 source
labels; float32 is restricted to learner inputs. The chart has real UTC dates
and explicit random/zero baselines. Its gross-return panel states that it has no
spread, fees, fills, or ROI. Overlapping horizons are never summed into a
performance curve. This remains forecast evidence only until exact-BBO replay,
execution-cost stress, and current no-order shadow independently pass.

Every completed fold is checkpointed in `fold-summaries.json`. `--resume`
requires the same plan and configuration, constrains every relative evidence
path to the run directory, verifies model/prediction file hashes, reloads the
serialized model contract, parses the complete gzip table, and recomputes its
prediction fingerprint and metrics before skipping work. A complete report is
immutable and cannot be resumed.

Timing, profile, and feature-set selection is a physically separate fail-closed stage.
Each candidate run receives only the declared initial screening folds. The
runner requires 4, 6, 8, or 10 non-overlapping screening folds and at least two
untouched later folds per symbol. Supply every screening report exactly once;
selection requires the same symbols, coverage fingerprints, chronological fold
boundaries, and identical dataset fingerprints for reports that share horizon,
cadence, and maximum depth age. The design file and every source report are
hash-bound into the winner lock:

```powershell
simple-ai-trading tape-depth-design `
  --risk-level conservative `
  --sampled-count 24 `
  --seed 20260710 `
  --output data/tape-depth-experiment-design.json

simple-ai-trading tape-depth-study `
  --design data/tape-depth-experiment-design.json `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT `
  --compute-backend directml `
  --resume `
  --output-dir data/tape-depth-study

simple-ai-trading tape-depth-prequential `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT `
  --study-stage screening `
  --max-folds 4 `
  --model-profile regularized `
  --feature-set core `
  --output-dir data/tape-depth-regularized-core

simple-ai-trading tape-depth-select `
  --design data/tape-depth-experiment-design.json `
  --report data/tape-depth-regularized-core/report.json `
  --report data/tape-depth-balanced-tape/report.json `
  --report data/tape-depth-balanced-cross-asset/report.json `
  --report data/tape-depth-expressive-full/report.json `
  --output data/tape-depth-selection.json

simple-ai-trading tape-depth-prequential `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT `
  --study-stage confirmation `
  --selection-lock data/tape-depth-selection.json `
  --output-dir data/tape-depth-confirmation-run

simple-ai-trading tape-depth-confirm `
  --selection data/tape-depth-selection.json `
  --report data/tape-depth-confirmation-run/report.json `
  --output data/tape-depth-confirmation.json
```

Selection aggregates baseline-relative AUC, Brier, MAE, rank IC, gross
calibration-threshold return, and fold-positivity measures over screening only. Every
symbol must beat the direction, prevalence, and zero-return baselines. The
selector also applies deterministic combinatorially symmetric cross-validation
to each declared trial's relative forecast-metric rank across screening folds.
It rejects the study when the estimated probability that an in-sample fold
winner ranks below the median on its symmetric validation folds exceeds 0.20.
This is explicitly a forecast-selection overfit diagnostic, not PnL, Sharpe, or
proof of profitability. The bounded even fold counts keep the complete split
table auditable while preserving the separately sealed terminal suffix. The
winner lock hashes every source report and records the full-corpus coverage
fingerprint, winning profile/feature set, exact terminal boundary, and trial
count. Loading it recomputes selection from unchanged sources. Confirmation
automatically uses only that winner and the complete untouched suffix; manual
fold caps, corpus drift, winner changes, overlap, and incomplete terminal
reports fail closed. Report loading also verifies `plan.json`, relative-path
containment, all artifact/prediction/table/chart hashes, serialized models, and
the complete compressed prediction rows; it recomputes fold fingerprints,
timestamps, metrics, status, and aggregate metrics before selection or
confirmation. A failed winner rejects the experiment, and no runner-up is
evaluated. This enforces the software access path; external copies or human
inspection outside the application remain outside what a local artifact can
cryptographically prove. Both outputs remain forecast evidence, not executable
PnL, trading authority, or profitability claims.

For non-CPU LightGBM work, DirectML remains the general Windows tensor backend
while LightGBM itself uses its OpenCL trainer. Automatic selection now delegates
to the installed OpenCL driver instead of assuming platform/device `0:0`.
Explicit overrides are accepted only when both
`SIMPLE_AI_TRADING_OPENCL_PLATFORM_ID` and
`SIMPLE_AI_TRADING_OPENCL_DEVICE_ID` are valid non-negative integers. On the
current AMD host, a real 100,000-row, 48-feature probe reported LightGBM's GPU
trainer and device `gfx1201`; this capability check is not a model result.

The fold, queue, and latency design follows the documented limits of market-
data replay: a replay cannot infer the strategy's own market impact, queue
position matters, and feed, order-entry, and response latency are distinct.
LightGBM's leaf-only `Booster.refit` is not used because it does not rebuild tree
structure; every validation fold performs a full fixed-protocol retrain.

- HftBacktest order-fill and queue assumptions:
  <https://hftbacktest.readthedocs.io/en/latest/order_fill.html>
- HftBacktest latency components:
  <https://hftbacktest.readthedocs.io/en/v1.8.4/latency_models.html>
- LightGBM `Booster.refit` semantics:
  <https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html>

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
- Scikit-learn's nested-CV guidance states that using the same observations for
  parameter selection and performance estimation produces optimistically
  biased results. The tape/depth workflow applies that separation in calendar
  order: candidate screening artifacts cannot contain terminal folds, and only
  the frozen winner can open the terminal suffix:
  <https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>
- Scikit-learn's leakage and decision-threshold guidance also informs the v8
  train-only sample-weight scale and calibration-only decision policy:
  <https://scikit-learn.org/stable/common_pitfalls.html> and
  <https://scikit-learn.org/stable/modules/classification_threshold.html>
  The same boundary applies to the general training suite: its calibration-fit,
  full-fit fallback, and probability-inversion variants are compared only on
  the chronological selection panel. The winning internal variant is frozen
  before a single validation replay is allowed. A further purged terminal
  suffix is sealed before the search begins and is opened only after the final
  model, hybrid experts, thresholds, and meta-label policy are frozen. The
  purge gap follows the time-series split principle of excluding observations
  between training and test boundaries:
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
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
- DeepLOB influenced the roadmap for true order-book models. Until full depth
  snapshots are persisted, this repo uses candle microstructure proxies plus
  second-level aggTrade-derived order-flow features such as taker-buy ratios,
  signed-flow imbalance, trade-count shocks, quote-volume shocks, no-trade
  ratios, flow/return alignment, flow strength, flow persistence, flow
  acceleration, and price/flow divergence:
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
- The Basel market-risk backtesting framework informed the tail-risk controls:
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

## Implemented Candidate Models

The base classifier remains the advanced logistic/GPU training path already
used by the CLI. The advanced feature vector now includes:

- multi-window technical rating votes inspired by TradingView's MA/oscillator
  aggregation structure,
- candle microstructure proxies inspired by order-book literature when only
  OHLCV candles are available,
- ATR-normalized trend and breakout features,
- volume-surge confirmation for high-frequency day-trading entries,
- market-quality regime features for trend efficiency, downside pressure,
  lagged return autocorrelation, volatility-of-volatility, volume pressure,
  volume/return correlation, ATR pressure, and current volume z-score,
- v9 feature signatures that keep the v8 known-at-entry information-event
  labels and expand the real aggTrade-derived order-flow microstructure block
  from 9 to 13 fields per window with average signed-flow strength, serial flow
  persistence, front/back-window flow acceleration, and price/flow divergence.
  Legacy v8 signatures still rebuild the original 9-field order-flow layout, so
  older evidence artifacts do not silently change dimension.
- side-aware futures threshold calibration that can promote symmetric,
  long-only, or short-only thresholds only when the selection fold improves
  risk-adjusted evidence; the final holdout still has to pass profitability,
  drawdown, liquidation, trade-quality, and market-edge gates.
- a diverse bounded candidate prefix that covers default, long and short
  day-trading frequency probes, intraday triple-barrier probes, focal
  rare-event information-event candidates, session-scale volatility barriers,
  and order-flow event probes before wider sweeps.
- downside-positive label orientation: after probability calibration, models
  trained on short-success labels are inverted into the runtime convention so
  short evidence cannot be accidentally scored as a high-probability long
  signal.

The revamp also adds a hybrid expert layer stored directly inside the serialized
model:

- `lorentzian_knn`: balanced long/short prototypes selected from chronological
  training rows, scored with Lorentzian distance.
- `rational_quadratic_kernel`: kernel-regression vote over the same prototype
  set, with tunable alpha and length scale.
- `technical_confluence`: deterministic market-regime confluence using the
  existing feature vector for trend, volatility, volume, and mean-reversion.

The optimizer evaluates risk-level-specific weight profiles:

- `conservative`: starts with base-model agreement profiles, then also tests
  reduced-base-model-weight fallback profiles where technical confluence, neighbor voting, and
  kernel smoothing can dominate if selection evidence supports them.
- `regular`: balances base probability, Lorentzian neighbor structure, kernel
  smoothness, and confluence.
- `aggressive`: allows stronger expert contribution, but still has to pass
  backtest gates and drawdown limits.

After classifier and hybrid selection, the optimization evidence path now runs
an interpretable strategy-template library. This implements the open-source
pattern seen in stronger bots: broad, cheap entry/exit template sweeps happen
before any template is promoted into the same serialized model path. The current
templates are original momentum breakout, VWAP/RSI mean reversion,
trend-pullback, volatility breakout, volume-flow proxy, order-flow momentum,
flow-reversion, flow-consensus breakout, liquidity-absorption reversal,
micro-flow scalp, VWAP snapback scalp, liquidity-sweep reversal, compression
breakout scalp, volume-synchronized flow, adaptive tape-regime, and
higher-timeframe alignment families. The default full-replay budget stays at
225 static templates, but those replays are now selected from a larger
stratified chronological event-rank pool. The cheap ranker scores both normal
and inverted orientation by after-cost forward-event edge, signal count, and
hit rate on earlier and later slices before choosing which templates deserve
full lifecycle replay. This ranker is only an efficiency and search-quality
layer: it cannot promote a template, and every selected template still has to
pass the same backtest, activity, edge, drawdown, profit-factor, expectancy,
and path-quality gates. Optimization evidence stores the split mode, training
rows, validation rows, and best training/validation edge, signal, and hit-rate
statistics so graph/table regeneration can audit whether a replay candidate was
selected by stable forward evidence or by a small-sample fallback.
Order-flow and higher-timeframe templates receive serialized offsets into the
advanced feature vector, so CPU, DirectML, CLI, Windows app, backtest, and
live/autonomous inference use the same microstructure and broad-regime inputs.
CPU/live rule-alpha scoring now preserves the full serialized feature vector
instead of truncating before the advanced blocks, and DirectML parity is covered
by tests. Each template is tested with normal and inverted
probability orientation plus bounded threshold/stop/take/hold profiles.
Rule-alpha models serialize as `rule_alpha` hybrid experts, so a promoted
template is available without a separate code path. The empirical feature-edge
miner can add one-feature and two-feature interaction tail rules only when an earlier mining slice
and later validation slice both show enough signals and positive net edge after
the modeled cost floor; mined candidates still use the same serialized
`rule_alpha` path and replay gates. Rejected searches record the
best rejected active profile, P&L, closed-trade count, win rate, profit factor,
max drawdown, exit-reason counts, side counts, reject reason, orientation, and
candidate count. Candidate diagnostics also persist the complete candidate set's
active/profitable/accepted candidate counts, static-template candidate count,
empirical mined candidate count, empirical interaction count, forward-event signal count, positive
after-cost forward-event count, best raw event candidate, maximum
closed trades, most-active candidate, best-PnL candidate, and active
family/profile coverage so failed research cannot hide whether it was inactive,
active-but-losing, or directionally negative before full trade replay.

The optimization-round evidence path now runs the same adaptive hybrid candidate set
as a post-base-candidate selection step even when the selected base candidate
failed, but only from a copied model reset to the best diagnostic threshold. It
uses the existing chronological training/selection split, records hybrid
profile and score diagnostics, emits status updates during long hybrid checks,
and keeps the fail-closed no-entry model unless the hybrid replay passes
`ObjectiveSpec.accepts`. This prevents a sophisticated ensemble overlay, or a
rejected diagnostic threshold, from becoming executable just because it exists
in code. The 2026-07-07 `round-broad-rule-alpha-1d-smoke` run used verified
BTCUSDT/ETHUSDT/SOLUSDT futures `1s` data for 2024-06-01, DirectML training and
scoring, conservative 5x futures settings, seven hybrid profiles, the v9
171-feature advanced vector including 13 order-flow microstructure fields per
window, cost-aware rule-alpha stop/take floors, 225 static templates, and an
empirical feature-edge miner that tests one-feature and two-feature interaction
rules. The empirical miner found zero validated one-feature or two-feature
interaction candidates under the chronological sample-count and after-cost edge
gates, so the run still replayed 450 normal/inverted
static-template candidates per symbol. It failed with zero accepted symbols, zero closed
holdout trades, mean ROI `0.0%`, and no liquidations. Best rejected active alpha
profiles were still negative after costs: BTCUSDT guarded momentum breakout
lost `0.6405880579098948` on one closed short, ETHUSDT scalp-3s
liquidity-sweep reversal lost `0.525900571656166` on one closed long, and
SOLUSDT held-180s flow reversion lost `0.5233335205740559` on one closed long.
The broader search did generate more internal trading activity, with 390 BTCUSDT,
416 ETHUSDT, and 390 SOLUSDT active candidates, but zero profitable candidates
after modeled costs. The most-active candidates still lost after costs
(`-13.698733890780659`, `-15.73342255896398`, and `-16.29628526423312` for
BTCUSDT, ETHUSDT, and SOLUSDT), so this remains research evidence rather than
promotion evidence. The added event-study telemetry showed an even earlier
failure: all 450 rule-alpha variants per symbol
produced forward-event signals, but zero had positive net forward edge after the
modeled cost floor.
This is negative research evidence rather than promotion evidence.

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
features, order-flow microstructure features, nonlinear transforms, and
polynomial interactions zeroed out one group at a time. The report records
acceptance, score, P&L, drawdown, trade count, and delta versus the selected
model. This remains attribution evidence for model selection, and it is also
carried into `ai-review`: if the compact accepted report shows that removing a
hybrid expert or feature group improves the selected score, the AI review
deterministically vetoes before calling the local model.
The training suite also writes a `selection_risk` report for the selected
candidate and serialized model. This deterministic multiple-trials haircut uses
the explored candidate count plus local, ensemble, hybrid-profile, full-fit,
and probability-inversion checks to deflate the selected score by observed
score dispersion. `internal_variants_evaluated`, `hybrid_profile_trials`, and
`internal_variant_extra_trials` make those formerly implicit trials auditable.
A candidate is not promoted unless the deflated score remains positive.

Each general-suite candidate now follows a fixed internal chronology. A model
may use its fit/calibration partition for early stopping and probability or
threshold calibration. The calibration-fit model, optional full-fit fallback,
and one probability-inversion variant are then scored on selection rows only.
The best internal variant and threshold are frozen from that panel; ties retain
the less adaptive incumbent. Only that frozen model is replayed once on the
later validation rows and once on the overlapping full-sample sanity panel.
The candidate score is the minimum accepted objective score across selection,
validation, and full-sample replays. Unselected inversion variants expose no
validation or full-sample result. Candidate diagnostics persist the selected
variant, inversion source, every internal selection result, and exact internal
trial count. The full-sample replay is a conservative consistency check, not an
independent out-of-sample estimate; purged walk-forward and model-lab gates
remain required before promotion.

The general suite additionally reserves the latest 20% of every candidate's
chronological rows as a terminal holdout, with the label horizon purged from
the end of development data. Candidate ranking, local/ensemble refinement,
walk-forward screening, hybrid search, meta-label fitting, and feature ablation
cannot access those rows. After all predictive mutations are complete, the
exact serialized model configuration is replayed on the terminal suffix once.
The evidence records `evaluation_count=1`, row count and UTC-compatible bounds,
the full model-row SHA-256 fingerprint, final model family, inversion state,
hybrid profile, meta-label state, objective score, and compact financial result.
Any missing rows, exception, nonpositive score/P&L, objective rejection,
drawdown stop, or liquidation blocks serialization. The final reported score
and deflated score are capped by this sealed result. Model readiness, financial
sanity, and compact AI review all reject missing or malformed terminal evidence.
The old zero-fold hybrid fallback path was removed: failed walk-forward evidence
cannot be replaced with a synthetic passing gate.

The general suite also has a durable cross-run boundary. `train-suite` requires
an explicit `--symbol` to create durable evidence, while `model-lab` forwards
the symbol it actually loaded. A bare candle JSON has no trustworthy asset
identity, so omission remains available only for research and cannot produce a
live-ready artifact. Immediately before the
first terminal backtest, the process uses `BEGIN IMMEDIATE` to reserve the exact
terminal timestamps in `~/.config/simple_ai_trading/terminal_holdouts.sqlite3`.
Overlap is rejected independently for each symbol, market type, and one of the
three supported risk objectives. Reserved rows are never deleted or made
reusable by the application: accepted, rejected, evaluation-error, and
process-interrupted reservations all continue to block overlap. Finalization
stores three separate SHA-256 bindings: derived terminal rows, the exact model
excluding only later governance stamps, and the complete terminal report.
Authenticated live readiness recomputes the model/report hashes and verifies
the reservation against the local database. A copied model without its matching
ledger is research-only until it is revalidated; deleting or replacing the
ledger intentionally invalidates existing live authority. Administrative users
can always delete all local state, which application code cannot prevent, so
the ledger and its filesystem backups must be protected as governance records.

The ledger uses SQLite rollback-journal mode, `synchronous=FULL`, a 30-second
busy timeout, atomic immediate write transactions, and an integrity check on
open. This deliberately avoids concurrent WAL on the bundled SQLite 3.45.1
runtime: SQLite documents single-writer transaction semantics and the behavior
of `BEGIN IMMEDIATE` at <https://sqlite.org/lang_transaction.html>, while its
current WAL guidance documents a rare WAL-reset issue fixed in newer SQLite
releases at <https://sqlite.org/wal.html>. Ledger writes are tiny and occur only
at terminal reservation/finalization, so durability takes priority over WAL
throughput.
AI-assisted alpha has a separate deterministic uplift gate. When AI is enabled,
`ai-review` will not call the local LLM unless every accepted AI-assisted symbol
includes an `ai_uplift` artifact showing the AI-assisted holdout beats the
non-AI ML baseline on realized P&L and expectancy, does not worsen max
drawdown, does not introduce liquidations, does not worsen loss-streak,
profit-factor, win-rate, or downside return/risk evidence when those metrics are
available, has enough closed trades, was produced by a multibillion model, and
passes paired holdout statistical evidence. Pairing uses contiguous,
non-overlapping fixed market periods containing both strategies' returns; trade
lists are not paired by index because vetoes and cooldowns change trade timing.
The gate requires SHA-256 bindings for the common dataset, baseline evidence,
AI evidence, local model artifact, and paired-period table, plus at least 30
periods spanning at least 90 days, a positive-delta rate above policy, an exact
one-sided sign-test
p-value at or below 5%, and a positive 95% moving-block-bootstrap lower bound
from at least 2,000 deterministic resamples. Artifact policy can tighten but
cannot weaken those built-in floors.
Missing or failed uplift evidence leaves AI in advisory/review-only mode.

### Local AI Risk-Review Benchmark

The 2026-07-10 local `finance-risk-review-adversarial-v6` comparison evaluated
Qwen3 8B Q4_K_M and the finance-specialized Fino1 8B Q6_K conversion on 11
schema-constrained adversarial cases. Both returned valid JSON and the expected
action for all 11 cases. Qwen passed every semantic/risk-range gate with score
`0.983409` and mean latency `3.19s`. Fino scored `0.990455` but was rejected by
the all-cases rule because its liquidation rationale did not identify the
explicit `15x` leverage exposure. A higher average score cannot override a
single missed critical-risk concept.

The final v6 scores are deterministic rescores of the persisted v4 normalized
responses. Only scorer aliases and the rational risk range for a flat,
non-urgent provenance conflict changed; model prompts and response hashes did
not. The source payload hashes are embedded in
[`docs/ai/risk-review/latest/comparison.json`](ai/risk-review/latest/comparison.json),
and exact Ollama manifest/base-blob identities are in
[`docs/ai/risk-review/latest/model-provenance.json`](ai/risk-review/latest/model-provenance.json).
Fino is explicitly recorded as a third-party GGUF conversion, not an official
quantization. The report sets `financial_edge_tested=false` and
`trading_authority=false`; it selects a risk reviewer only. A paired,
post-cost, no-lookahead AI-vs-ML uplift benchmark is still required before AI
can be credited with market edge.

After a candidate survives development selection, the suite trains a compact
meta-label policy from the accepted model's development-only simulated trade
log. The policy
records the signal-strength thresholds that would take, downsize, or skip trades
under the current objective precision target and is persisted in both the model
artifact and `training_suite_summary.json`. Backtests, the legacy live loop, and
the autonomous loop now apply that policy as a deterministic pre-entry
skip/downsize gate. It cannot create entries or override exits, and malformed
enabled policies fail closed by skipping the entry. The policy is attached
before the one-shot terminal replay, so its effect is included in sealed final
evidence rather than added after validation.
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
`ObjectiveSpec.accepts` on development selection and validation, then pass the
sealed exact-model terminal gate. If no base candidate survives the purged
walk-forward gates, the objective remains rejected; a hybrid cannot manufacture
or inherit a zero-fold passing result. Offline hybrid fallback research remains in
the optimization workflow, where the selected model is still subjected to that
workflow's separate final holdout and cannot become live authority by itself.
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
Promotion-grade model-lab reports also preserve per-objective
`walk_forward_gate` evidence. A skipped purged walk-forward gate is treated as
missing promotion evidence, not as a pass; accepted outcomes and stamped model
execution-validation records require a positive fold count, every fold accepted,
positive worst score, positive worst realized P&L, and bounded worst drawdown.
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
The v10 advanced feature contract adds a separate
`higher_timeframe_context` group. It derives closed one-minute context bars from
the same candle stream used by the 1-second model and appends broad-regime
return, average-distance, realized-volatility, range, drawdown/bounce, volume,
and trade-count features. The context uses only bars whose close timestamp is
not later than the row timestamp, so it does not leak the current unfinished
minute into training or live inference. This is an evidence surface for broader
day-trading regime alignment; it is not treated as proof of profitability until
the same real-data backtest, activity, risk, and promotion gates pass.
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
- accepted outcomes must include real purged walk-forward evidence for every
  accepted objective; skipped or failed fold evidence blocks promotion,
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
  delta metrics, model-size evidence, hash-bound contiguous fixed-period
  samples, and paired sign-test plus block-bootstrap evidence; missing, weak,
  index-paired, or policy-weakened fields block the model-lab artifact,
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
3. Fetches or loads candles for each ranked symbol. Default runs are
   recent-limit research/smoke API runs; `--full-history` pages backward
   through venue maximum kline batches until the exchange returns no older
   rows. `--market futures --quote-asset USDT --interval 1s --market-db
   data/market_data.sqlite --require-db-data` trains from the local SQLite
   archive and fails instead of silently falling back to API klines when
   second-level rows are missing.
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
   deflation passes, every stress simulation passes the objective risk controls,
   temporal robustness passes, data coverage has no hard integrity failure,
   learning feedback has no unresolved repeated-loss block, and the accepted
   set passes the portfolio diversification and tail-risk controls.

Every outcome now includes a `data_coverage` evidence block. It records symbol,
market type, interval, source scope, requested and used UTC date span, candle
counts, model-row count, gap count, largest gap, coverage ratio, full-history
flags, and a `truth_basis` list that explicitly says the execution results are
simulated rather than exchange fills. Missing candles, missing model rows,
coverage gaps, or coverage below `99.5%` fail promotion. Recent-limit API runs
are not hard failures by themselves, but they are labeled `binance_recent_limit`
and must not be presented as full-history optimization evidence. SQLite-backed
second-level runs are labeled `sqlite_market_data` and include `market_db_path`
in the report; no second-level model-lab claim may be made without those fields.

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

Signed live-style startup and `risk --live --model` now also require live data
evidence from the same model artifact. The serialized model must carry
`execution_validation.data_coverage` for the selected runtime symbol and market,
and that evidence must be SQLite-backed `1s` data, full available history, at
least one year of used span, at least `99.5%` coverage, zero missing-second
gaps, positive candle/model-row counts, and no hard data-integrity warning.
If the runtime interval is not `1s`, startup fails before account reconciliation
or any order loop. This keeps research/paper artifacts usable for experiments
while preventing minute-level, recent-limit, hand-edited, or wrong-symbol models
from reaching signed execution.

The same readiness gate now requires `execution_validation` on signed live
artifacts. `model-lab` stamps each serialized model after it runs
symbol-specific stress validation and final-model temporal robustness against
the selected liquid symbol, then after the portfolio risk assessment is known. The
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
- No optimization report may be presented as promotion-grade day-trading
  evidence unless `tools/optimization_round.py --promotion-grade` wrote a
  `promotion_grade_contract` with `status: pass` for exact BTC/ETH/SOL `1s`
  data, verified checksums, zero gaps, and the configured minimum stored
  history span.
- No second-level model-lab result may be described as DB-trained unless the
  report declares `data_source: sqlite_market_data`, `interval: 1s`, and the
  matching `market_db_path`.
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
