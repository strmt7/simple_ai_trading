# Polymarket model status

> **Beta research software. No paper or live trading authority exists.**

## Current evidence

Round 26 tested the current BTC five-minute 60-second TWAP source on two
one-hour credential-free captures. Neither capture is model-eligible. The first
recorded 1,519,316 messages and two CLOB gaps; a target-free isolated replay
admitted 11 of 13 conditions, excluded one checksum-corrupted condition and one
34 ms trailing stub, and produced no P&L. The second recorded 1,442,594 messages
and five transport gaps. These results reject whole-capture zero-gap repetition
as the next data strategy; future admission is condition-local and requires a
fresh two-outcome baseline plus a connection-bounded executable interval.
The second capture's economic analyzer also failed closed on 87 Binance
Futures trade frames whose reported price and quantity were both zero. Those
frames are preserved as source-integrity evidence and are not treated as price
observations.

Round 27 screened execution mechanics on the 53 condition-eligible Stage 0
markets. Applying each exact exchange message before pairing outcomes left six
after-fee complete-set episodes. None survived the recorded venue delay, and
none survived the optimistic two-delay sequential floor. The best delayed and
sequential costs were 1.014726 and 1.002034 pUSD per complete set,
respectively, before network or order-response latency. The screen observed
60,863 extreme-price states across all 53 markets and 11,884 late-favorite
states across 44 markets. These remain prospective model-value hypotheses;
public quotes do not prove fills, settlement value, or profitability.

Stage 1 is frozen and its Stage 1-A capture started on schedule at 09:00 UTC on
2026-08-15. No Stage 1 feature row or outcome has been accessed. Its three
fixed 10.5-hour primary windows rotate across 09:00, 16:00, and 23:00 UTC
starts and schedule 378 BTC
five-minute intervals across at least three UTC dates. Each slot uses a
separate 2.5 GiB-capped database. A fourth fixed contingency window is allowed
only if target-free replay of all primary slots admits fewer than 300 markets.
The [campaign contract](../round-027-stage1-campaign-contract-v1.json) forbids
target access, fitting, adaptive replacement, and economic claims before the
campaign gate.
The [v16 admission correction](../round-027-campaign-admission-gate-correction-amendment-v16.json)
enforces that gate at the target-store API: one persisted receipt must bind all
primary audits, the total and role-specific population floors, and each role's
exact target-free feature population before any outcome request can begin.
The cumulative
  [v19 source-maintenance amendment](../round-027-static-analysis-remediation-amendment-v19.json)
  preserves the immutable v7 ledger and cumulative v17/v18 amendments while
  binding fail-closed condition-cache aggregate validation. It changes no model,
  feature, target, cost, risk gate, result, or authority.

Round 28 is a preregistered, target-blind BBO ablation over that same campaign,
not a new result. The independent Round 21 sidecar already overlaps the Stage 1
windows and records real-time BTCUSDT spot and USD-M best bid, ask, and top
quantities. Its [frozen contract](../round-028-binance-bbo-matched-ablation-preregistration-v1.json)
adds only 96 BBO-derived fields to the 182 Round 27 fields, stores them as a
small overlay instead of duplicating the base corpus, and excludes every stale
or unavailable BBO decision from both candidates. Base and augmented models
must use identical rows, labels, weights, market-prior offsets, and execution
scenarios. The large OpenMarket null result remains the prior; prediction
uplift alone cannot establish after-cost edge or profitability.

Round 29 is a second target-blind matched ablation over the same frozen rows,
also not a result. The original design named a distance-to-strike/time-remaining
interaction, while the actual 182-field corpus stores those inputs separately;
the L2 residual therefore cannot express that relationship directly. The
[preregistration](../round-029-settlement-state-matched-ablation-preregistration-v1.json)
adds six deterministic fields: current TWAP side, two margin/time transforms, a
smooth variance-standardized diagnostic, signed path efficiency, and a
margin/elapsed-phase interaction. It can compose as 188 fields over Round 27 or
284 fields over Round 28 without copying raw data. It explicitly forbids
reconstructing Chainlink's unpublished TWAP calculation and leaves all frozen
cost, delay, probability, concentration, drawdown, and one-use sealed gates
unchanged. The overlay implementation and synthetic tamper tests pass, but the
matched model/economic path is now source-bound by the
[implementation amendment](../round-029-model-economic-operator-implementation-amendment-v1.json).
It fits the diagnostic and promotion-controlling pairs on identical rows,
binds the persisted selection to the exact input manifest, replays both primary
arms over the same observed books and fixed delays, and rejects source, model,
nested-report, path-alias, or hard-link drift. The bounded runner stops before
economics when probability gates fail. The amendment also binds the
[cross-regime acceptance contract](../../cross-regime-edge-acceptance-contract-v1.json):
aggregate passage cannot establish a bull/bear/sideways/choppy or stressed
liquidity/latency edge, and unsupported states require rejection or abstention.
No Stage 1 feature, outcome, metric, P&L, edge, profitability, or authority has
been accessed or created for Round 29.

The pre-target [selection implementation amendment](../round-028-selection-implementation-amendment-v1.json)
adds chronological whole-condition folds, a 10-minute embargo, calibration-only
scaling, paired stationary bootstrap gates, and fail-closed model persistence.
It also corrects an ECE boundary-binning defect discovered with synthetic tests.
No Stage 1 feature, outcome, selection, economic, or sealed result was used.

The pre-target [matched economics amendment](../round-028-economic-implementation-amendment-v1.json)
now projects both persisted models onto the exact frozen Round 27 fee, depth,
FOK, and latency replay. Promotion requires after-cost uplift at every fixed
delay, a positive paired condition-bootstrap lower bound, all original
execution gates, and no worse maximum drawdown. Books are scanned twice through
a bounded batch factory rather than copied in memory or on disk. This is tested
implementation evidence only; no Round 28 P&L, graph, edge, or profitability
result exists yet.

The source-bound [selection operator amendment](../round-028-operator-implementation-amendment-v1.json)
adds one restart-safe command for input freezing, model selection, and matched
economics. It recursively validates checkpoints and reuses a valid report
without another database scan; a failed probability gate stops before economic
replay. It remains unavailable until the prospective stores are terminal and
contains no result or trading authority.

The pre-outcome [sealed evaluation amendment](../round-028-sealed-evaluation-implementation-amendment-v1.json)
now completes the model-side confirmation path. It loads the exact selected
pair without refitting or retuning, requires passing selection economics before
sealed access, and compares the models on identical Stage 1-C conditions and
after-cost scenarios. A valid terminal artifact is reused without reopening
sealed outcomes or rescanning books. No sealed result or edge claim exists.

The [Round 28 AI preregistration](../round-028-ai-risk-veto-preregistration-v1.json),
[core implementation](../round-028-ai-risk-veto-implementation-amendment-v1.json),
[operator implementation](../round-028-ai-operator-implementation-amendment-v1.json),
and [sealed evaluation implementation](../round-028-ai-sealed-evaluation-implementation-amendment-v1.json)
replace the old 182-feature prompt with the exact 278-feature augmented case.
Qwen3.5-9B and ODA-Fin-SFT-8B remain controls; ODA-Fin-RL-8B is a pending
challenger, not a selected model. Every model is limited to reject, reduce, or
leave unchanged, must be fully GPU-resident under an exact digest, and is
charged measured wall latency in the matched replay. The restart-safe operator
accounts for all three candidates and evaluates qualified candidates in one
bounded order-book scan. The sealed path freezes target-free cases and
inference first, opens outcomes only after an exact nomination exists, and
reuses a valid terminal result without reopening the target store. No Round 28
AI model has yet been run on Stage 1 data, and no AI uplift, edge, or
profitability result exists.

The [v2 sealed-process correction](../round-028-ai-sealed-evaluation-implementation-amendment-v2.json)
supersedes the direct v1 case tool for future runs. A separate authorizer now
reduces passing selection evidence to a metrics-free receipt; the authoritative
case runner accepts that receipt but no target, outcome, resolution, economic,
or PnL path. It also requires a five-second, fully GPU-resident probe using the
actual 278-feature prompt shape before inference. These are source-bound safety
mechanics only, not AI performance or trading evidence.

The [community-strategy review](../round-028-community-hypothesis-review-v1.json)
treats the supplied Reddit analysis as hypotheses, not data. Its open/close
settlement assumption does not match the current 60-second TWAP product, and
its dataset and code are unavailable for audit. Binance lead, late-favorite,
and mean-reversion ideas therefore receive no new threshold; maker split/merge
received an independent [point-in-time reward diagnostic](../paired-maker-reward-screen-v1.md).
Its displayed both-fill surplus was stale, its publicly proven reward payout
floor was zero, and orphan risk remained unbounded by fills or cancellation
evidence. Its conditional share, daily-equivalent, and payback diagnostics are
also invalid because the hypothetical complementary own asks were omitted from
post-quote midpoints; only the both-fill and orphan settlement arithmetic
remain valid. Its Moonshot candidate was also outside the frozen BTC/ETH/SOL scope;
the [scope adjudication](../paired-maker-reward-scope-adjudication-v1.json)
prohibits rerun or prospective continuation. A distinct in-scope candidate
would require its own prospective queue, reward-persistence, and orphan-risk
contract before any execution study.

The first frozen in-scope BTC/ETH/SOL paired-maker source screen stopped after
two public requests because Gamma and the exact CLOB reward source disagreed on
BTC configuration. It reached no books or economics, and its one-attempt
contract prohibits resampling. The
[`terminal receipt`](../crypto-paired-maker-reward-screen-attempt1-failure-v1.json)
also records that future one-use screens must persist source payloads and write
failure evidence before raising validation errors.

A same-day [live product-regime audit](../round-027-live-product-regime-audit-v1-2026-08-15.json)
hash-bound an exact BTC five-minute market to `btc-5m-twap-60`, a 60-second
Chainlink TWAP source, the current crypto fee curve, and an enabled taker-order
delay flag. This matches Round 27's source filter. It also prevents a category
error in the research program: the February-April settlement-manipulation
study used a single price at the open and close, whereas the current product
averages the reference across 60 seconds. Its historical profit figures are
not current edge evidence. A separate target-blind
[risk diagnostic](../round-027-settlement-regime-risk-preregistration-v1.json)
will report final-window flow and post-close reversal without changing the
model, policy, thresholds, or economic gates.

The immutable model contract also contained one mathematically unreachable
economic evidence gate: at most one candidate may be selected for each of 90
conditions, but the original minimum was 100 executions. A frozen
[pre-capture amendment](../round-027-economic-population-amendment-v1.json)
sets the selection and sealed minimums to 60. It is not a trade quota; risk
checks remain unchanged, and fewer fills must yield insufficient evidence.
Before any Stage 1 feature or outcome access, a source audit also found that
the LightGBM candidate was implemented as a standalone classifier rather than
the frozen market-prior residual. The
[offset correction amendment](../round-027-lightgbm-offset-correction-amendment-v1.json)
binds the old and corrected source hashes, trains with the market logit as
`init_score`, and restores that logit around the raw tree margin at inference.
Its synthetic AMD OpenCL check is implementation evidence, not performance or
profitability evidence.
The subsequent
[calibration-identity amendment](../round-027-calibration-identity-correction-amendment-v2.json)
also fixes the persisted identity of a model whose calibration scale is not
`1.0`. The prior scaler changed predictions without recomputing the model
SHA-256, so a non-unit selected model could not be reloaded exactly. Every
allowed scale is now rehashed and round-trip validated before a selection claim
can bind it. This changes no model family, threshold, or economic gate and is
implementation evidence only.
The cumulative
[active-tick amendment](../round-027-active-tick-execution-correction-amendment-v3.json)
then corrects the after-cost replay for Polymarket's documented dynamic tick
changes. Candidate limits use the reconstructed decision-book tick; every book
price must align to its active lattice; and a limit that is invalid on the
execution-book lattice receives no fill credit. The matched AI path carries the
same tick identity without changing its frozen prompt. This is execution-model
correctness evidence, not edge or profitability evidence.
The separate [matched AI contract](../round-027-ai-matched-ablation-contract-v1.json)
also freezes target-free, latency-charged Qwen-versus-ODA selection and a
one-use sealed confirmation. Host compatibility alone is not AI uplift.

Round 27 uses Binance's documented USD-M `@aggTrade` market stream.
A live 33.719-second source probe recorded 29/29 finite positive trades, while
the new source gate independently rejects the legacy v3 feed. The subsequent
one-use 600-second prospective smoke recorded 194,980 raw public messages
across four BTC markets with zero gaps, recorder errors, or integrity errors.
Its independent target-free replay accepted all 1,014 USD-M `aggTrade` records
and all 2,961 spot trade records, with zero invalid, non-positive, or unexpected
trade types. Database plus WAL used 19,410,944 bytes against a 2 GiB cap. The
[source qualification](../binance-usdm-aggregate-trade-source-qualification-v1-2026-08-15.json)
and [prospective smoke result](../round-027-documented-source-smoke-result-v1-2026-08-15.json)
qualify this public source profile for a larger prospective capture only; they
grant no model-data, edge, profitability, paper-trading, or live authority.

The subsequent five-hour Stage 0 capture recorded 6,099,812 raw public
messages and 62 market snapshots. Exactly 59 markets had their complete
five-minute interval inside the recorder bounds; the partial opening and two
trailing markets were excluded without replay. The capture recorded four CLOB
and five RTDS reconnect boundaries, zero recorder or integrity errors, 61,304
valid USD-M aggregate trades, and 161,128 valid spot trades. Isolated replay
admitted 53/59 complete markets through 57 connection-bounded intervals and
excluded six: five atomic-depth checksum disagreements and one uncorroborated
best-book sequence. Eligible coverage ranged from 297.667 to 299.999 seconds.
The [capture result](../round-027-stage0-mechanics-capture-result-v1-2026-08-15.json)
and [target-free condition audit](round-027-stage0-condition-audit/condition-replay-audit.json)
contain no outcome, model, P&L, or trading authority.

After the mechanics result was frozen, a separate target-access claim opened
the exact 53-condition settlement population. Credential-free Gamma and CLOB
terminal responses agreed on all 53 winners: 26 Up and 27 Down, with zero
pending or disagreement. Raw responses are compressed and hash-bound outside
the capture database. This validates settlement mechanics only; Stage 0 is not
model-fitting data and creates no edge or profitability claim.

![Round 27 Stage 0 target-free replay eligibility](round-027-stage0-condition-audit/condition-replay-eligibility.svg)

![Round 27 mechanics diagnostic](round-027-mechanics-diagnostic/mechanics-diagnostic.svg)

![Round 27 settlement mechanics](round-027-stage0-resolution-mechanics/settlement-mechanics.svg)

![Round 26 target-free replay eligibility](round-026-twap60-condition-audit/condition-replay-eligibility.svg)

The [canonical Round 26 audit](round-026-twap60-condition-audit/condition-replay-audit.json),
[v3 analysis failure](../round-026-twap60-development-analysis-failure-v2.json),
the [Round 27 mechanics data](round-027-mechanics-diagnostic/mechanics-diagnostic.json),
the [Round 27 settlement audit](round-027-stage0-resolution-mechanics/settlement-mechanics-audit.json),
the [documented-source smoke contract](../round-027-documented-source-smoke-v1.json),
the [Stage 0 capture contract](../round-027-stage0-mechanics-capture-v1.json),
and [untrusted execution-hypothesis preregistration](../round-027-execution-hypothesis-preregistration-v3.json)
preserve the exact boundaries. No market edge or after-cost profitability has
been found yet.

On 2026-08-10, a credential-free production read probe validated CLOB protocol
V2, two exact BTC five-minute markets using the then-current Chainlink 30-second
TWAP source, matching Gamma/CLOB identities and fees, and all four token books.
The [machine-readable probe](public-clob-live-probe-2026-08-10.json) submitted
no order, accessed no wallet, and proves neither predictive edge,
profitability, nor live readiness. That source regime is retained as historical
transport evidence and is not treated as current after the 60-second audit.

![Round 23 lead-lag performance](../round-023-lead-lag-performance.svg)

Round 23 is the last evaluated mechanism. Public BTCUSDT spot and USD-M trades
reduced condition-equal one-second MSE by `9.22%` on an exploratory 12-condition
selection block (`0.018542` versus `0.020426`). That block was already consumed
and used archive event times rather than measured client receipt times, so the
[result](../round-023-lead-lag-results-v1.json) is neither an economic edge nor
a promotion result.

Round 20 failed after Polymarket changed the BTC five-minute resolution source
from the legacy Chainlink point stream to the `btc-5m-twap-30` contract. Its
failed source lineage is closed and will not be pooled into a successor.

Round 25 v1 was retired before its first eligible receipt. Its generic
Chainlink point-price assumptions are inadmissible for the current source
regime. A bounded probe of that legacy topic was also retired as source
evidence. The active v2 qualification instead observed the distinct
`crypto_prices_twap_thirty` topic with exact E18 value, BTC/USD identity,
source and publisher timestamps, and `window_s=30`. Point-topic frames are
rejected rather than pooled or reinterpreted.

Round 25 v2 is the source-correct successor. Before its 2026-08-11 00:00 UTC
start, a credential-free live qualification observed two exact E18 BTC/USD
30-second TWAP updates and two consecutive protocol-v2 Gamma/CLOB markets. The
clean source commit is `0e2f5c043ed0a0309d0e420b90b667565841d019`; the capture
design, campaign plan, and wire qualification hashes are bound in the
[TWAP-native model design](../round-025-twap-native-model-design-v1.json).
No v2 campaign database, target, model, AI comparison, economic result, or
profitability result existed when that target-blind design was frozen.

The v2 feature engine consumes only exact E18 TWAP observations received by the
decision time. It rejects missing exact openings, stale or future data, stream
gaps, conflicting duplicate timestamps, and broken 30-second grids. It exposes
path and transport features but no structural probability. Comparing exact
event-boundary TWAP values is only a frozen settlement hypothesis until it is
matched against official resolutions; nearest-tick, Binance, and independently
reconstructed TWAP substitutions are prohibited.

The [v2 terminal contract](../round-025-terminal-receipt-materialization-design-v2.json),
receipt auditor, and target-blind joint materializer/store are implemented for
the active v2 plan/state/result schemas. They make one read-only, WAL-free receipt pass, rebuild one strict
top-20 CLOB lane, admit only exact E18 TWAP observations, and persist only the
16 target-blind endpoints and bounded 64-step histories selected per condition.
The store binds exact market, token, role, source-snapshot, and receipt-audit
identities. No v2 terminal receipt audit or materialization result exists.

Official resolution collection is separately implemented behind a persisted
target-access claim. It requires complete identity agreement and terminal
winner agreement from credential-free Gamma and CLOB responses, keeps pending
conditions unlabeled, and atomically publishes a distinct compressed target
store only after full population and role-count audit. A prestart live check
found that both public origins offer a Cloudflare response cookie. The frozen
[v3 transport contract](../round-025-official-resolution-collection-contract-v3.json)
therefore rejects preexisting or outbound cookies but immediately discards
offered response cookies before validation, retry, or return. The
[post-close probe](../round-025-official-resolution-transport-probe-v1-2026-08-10.json)
jointly validated one terminal BTC market across both origins with zero cookies
retained and no credentials or orders. It is transport evidence only; no target
collection, pre-close identity proof, resolution authority, or target store
exists.

The [post-capture coordinator v4](../round-025-post-capture-coordinator-contract-v4.json)
binds the corrected resolution transport and exact-receipt economic contracts.
Its single-writer,
self-hashed recovery state orders terminal audit, feature publication, bounded
resolution batches, train/calibration-only loading, one-time transform fitting,
finite model fitting, target-free selection prediction freeze, one-use
selection access, and immutable predictive evaluation. Pending resolutions
return before any model access. An economic scan is prohibited unless the
predictive gate nominates exactly one frozen candidate. The host-neutral
`tools/run_polymarket_round25_post_capture.py` command adds an outer
single-writer lock, creates the terminal manifest once, emits flushed JSON
progress, and resumes bounded resolution batches. Its live-host pre-terminal
check opened no capture database, created no output, and submitted no order.

The [economic replay contract v2](../round-025-economic-replay-contract-v2.json)
then fixes one exact receipt-time source scan, captured market-specific fees
and venue delays, FAK depth walking, and 81 combinations of risk profile,
latency, displayed depth, and adverse ticks. It uses initial capital only,
never credits midpoint fills or hidden liquidity, and persists no authority.
Its result also preserves a self-hashed primary-scenario condition series for
all three risk profiles, so performance graphs remain data-reconstructible.
The operator is implemented. No coordinator run, fitted model, prediction
panel, nomination, economic result, predictive edge, or profitability result
exists.

The frozen predictive ledger combines 37 TWAP features with 111 causal CLOB
features at the same 250 ms decision receipt. Its six finite families are the
market-prior control, phase-isotonic control, regularized logistic residual,
two shallow LightGBM residuals, and a compact causal multitask TCN residual.
Development uses only the v2 plan's chronological train, calibration, and
selection intervals with one condition purged on each side of each boundary
and minimum condition counts of `2000/400/400`. Sixteen target-blind,
phase-balanced endpoints give every condition total weight one. Labels may
come only from official Polymarket winning-token evidence after terminal source
semantics are verified.

The [control-fit contract](../round-025-control-fit-contract-v1.json) freezes
quarter-phase weighted isotonic calibration and the bounded L2-logistic
residual before capture starts. The control fitter is implemented but no
control model has been fitted: every public fit path rejects fewer than
`2000/400` train/calibration conditions, and no prediction grants trading
authority.

The [LightGBM fit contract](../round-025-lightgbm-fit-contract-v1.json) fixes
both shallow tree candidates, condition-bounded leaves, calibration-only early
stopping, capability-tested CPU/OpenCL/CUDA resolution, serialized-model
identity, and the same bounded prior-residual output. The LightGBM operator is
implemented but no Round 25 tree has been fitted.

The [sequence contract](../round-025-sequence-materialization-contract-v1.json)
uses 64 condition-local 250 ms rows, resets after gaps, zero-pads with a valid
history mask, and excludes unavailable +1s/+5s auxiliary targets from loss.
Its sequence materializer is implemented but no sequence corpus has been
materialized.

Selection inference uses the separate
[target-free sequence contract](../round-025-target-free-sequence-inference-contract-v1.json).
It exposes only causal tensors and source receipts: terminal labels, auxiliary
targets, resolution authority, and target-bearing dataset identity are absent.
The builder is implemented, but no real selection batch exists.

The [TCN fit contract](../round-025-tcn-fit-contract-v1.json) freezes a compact
14,371-parameter causal residual network, exact three-seed ensemble, bounded
16-condition batches, calibration-only early stopping, and minimum `2000/400`
train/calibration conditions. The lazy fitter, artifact verifier, and reusable
target-free inference runtime are implemented. A bounded
[AMD DirectML host probe](../round-025-tcn-directml-host-probe-2026-08-10.json)
completed one real forward/backward update and byte-identical state reload on
an RX 9070 XT. That is runtime mechanics evidence only: no seed model or
ensemble has been fitted, and no predictive, economic, profitability, AI, or
trading claim follows.

The [predictive evaluation contract](../round-025-predictive-evaluation-contract-v1.json)
freezes the 10 paired challenger/score hypotheses, one-hour circular condition
blocks, 10,000 shared bootstrap paths, and studentized Romano-Wolf stepdown.
The evaluator is implemented behind a durable one-use boundary: all six model
outputs must be frozen in a target-free panel before selection targets can be
opened, access is restart-recoverable, and repeated access or event-chain
tampering fails closed. ECE, balanced accuracy, and ROC AUC are descriptive;
only simultaneous log-loss and Brier improvement can nominate a candidate.
No real prediction panel, target access, evaluation, or nomination exists.

The [model-ledger contract](../round-025-model-ledger-contract-v1.json) binds
all six fitted candidates to the source commit, implementation hashes, exact
train/calibration populations, and resolution authority. The target-free
preparation operator binds that ledger, the terminal-receipt audit, every
condition-level TCN batch, and all six probability arrays before the one-use
selection lock. The durable ledger and preparation mechanics are implemented;
no real ledger, prepared prediction, fitted candidate, or result exists.

The v2 plan has no test role. The
[candidate amendment](../round-025-twap-native-candidate-selection-amendment-v2.json)
therefore requires a separate prospective campaign after model selection.
Qwen3.5-9B and Fin-R1-8B failed the frozen 10-second per-entry runtime boundary
and cannot re-enter it. The current [AI risk contract](../round-025-ai-risk-advisory-contract-v6.json)
limits exact-digest Qwen3 4B to one of seven entry-only risk ceilings; trusted
code derives veto, size, cooldown, and audit semantics. AI cannot create or
enlarge a trade, change side, override deterministic safety, or affect exits.

A bounded [AMD host probe](../round-025-ai-risk-advisory-host-probe-v7-2026-08-10.json)
returned a valid constrained action in `1.8152` seconds with the exact model
fully resident on the RX 9070 XT and unloaded it afterward. The probe used one
target-free numerical fixture, no market outcome, no credential, and no order.
It proves runtime mechanics only, not useful judgment. The final target-free
[reachable safety battery](../round-025-ai-risk-scenario-host-probe-v3-2026-08-10.json)
retained 11 valid responses but its reporter incorrectly marked every check
false after expecting the obsolete 10-row ledger. A frozen
[no-inference correction](../round-025-ai-risk-scenario-host-probe-v3-correction-2026-08-10.json)
verified every original packet, advisory, telemetry hash, and unload state.
Benign and most isolated soft hazards returned `allow`; elevated portfolio risk
and the combined soft crisis returned `veto`. Maximum latency was `2.0685`
seconds. This is constrained safety-behavior evidence only, not economic value.

Fin-R1 8B was also evaluated once as a slower 60-second regime supervisor. A
pre-inference [inventory/reporting failure](../round-025-fin-r1-regime-supervisor-infrastructure-failure-v1-2026-08-10.json)
was corrected without changing the prompt, scenarios, model, or digest. The
superseding [seven-case host probe](../round-025-fin-r1-regime-supervisor-host-probe-v2-2026-08-10.json)
then produced seven valid, fully GPU-resident responses in `2.6296` seconds on
average, but returned `normal` for every case, including the combined crisis.
The [rejection](../round-025-fin-r1-regime-supervisor-rejection-v1-2026-08-10.json)
bars prompt tuning, economic evaluation, and execution integration for this
exact role. Finance QA benchmarks did not translate into useful regime-risk
discrimination.

The strongest installed prior general-risk candidate, Qwen3 8B, then ran the
same packet ledger, prompt, protocol, and gates. Its
[host probe](../round-025-qwen3-8b-regime-supervisor-host-probe-v1-2026-08-10.json)
also returned seven valid `normal` actions, including the crisis packet, at
`2.6730` seconds average latency. The resulting
[mechanism rejection](../round-025-qwen3-8b-regime-supervisor-rejection-v1-2026-08-10.json)
closes further model cycling on this slow regime-supervisor design. The fast
Qwen3 4B entry reviewer is unchanged; its economic value still requires the
prospective matched uplift experiment.

The [AI uplift contract](../round-025-ai-uplift-evaluation-contract-v2.json)
and evaluator require a new chronological population after selection with at
least 500 conditions and 50 actual interventions. The control and AI paths must
share the selected prediction, pre-AI decision, and execution/cost/latency
scenario. Ten thousand paired circular block resamples must show positive
after-cost return uplift without worse expected shortfall or maximum drawdown.
A pass is only a development nomination. No model training, candidate
selection, AI uplift, edge, profitability, paper authority, or live authority
exists yet.

The research-only [four-arm supervisor uplift contract](../round-025-fin-r1-regime-supervisor-uplift-contract-v1.json)
and clustered evaluator compare ML-only, fast Qwen, slow Fin-R1, and their
minimum-risk hierarchy using 10,000 bootstrap paths over supervisor windows.
They establish experiment mechanics only. Because Fin-R1 failed its frozen
behavior gate, this economic experiment is closed for that candidate.

The core capture uses no Binance input, credentials, account, wallet, model, or
execution authority. Round 21 remains a separate credential-free public
Binance predictor capture with no Polymarket execution or safety dependency.
Round 23 remains the latest graph and evaluated mechanism; profitability,
paper authority, and live authority are all false.

![Held-out predictive metrics](charts/round14-held-out-metrics.svg)

Round 14's subsequent one-hour, after-cost BTC five-minute shadow **failed**:
12 events produced `-9.87720` quote net PnL, `0.455725` profit factor, and
`11.69847` maximum drawdown. This rejects the model for paper or live
promotion. The canonical
[economic evaluation](../evidence/round-014-btc-5m-shadow-hour-evaluation-v1.json)
overrides the earlier predictive-only result below.

![Round 14 cumulative after-cost shadow P&L](shadow/cumulative-pnl.svg)

The plotted values are preserved in the
[exact event-outcome table](shadow/event-outcomes.csv).

Round 14 tested a frozen BTC five-minute direction model on all 287 eligible
conditions from 2026-06-22 UTC. The shallow Binance-flow LightGBM challenger
recorded log loss `0.644667` versus
`0.691317` for the best control, a
`6.75%` relative skill.
Balanced accuracy was `0.6208` and the
paired 95% block-bootstrap improvement interval was
`[0.03544,
0.05865]`.

This is **predictive evidence only**. Polymarket spread, queue position, fills,
latency, fees, settlement, redemption, inventory risk, and PnL were not tested,
so it is not a profitability or execution claim.

Round 25 v2 independently captures the current Polymarket TWAP-source regime.
Round 21 captures only optional public Binance predictor receipts in another
worktree and database. Neither capture is model-eligible before its current
source contract and terminal audit pass. A predictor outage can only block
signal-dependent new entries; Polymarket recovery, reconciliation, risk
reduction, Stop, and exits remain independent.

The finite development ledger contains three regularized logistic
residuals, two shallow LightGBM residuals, and one compact causal TCN residual.
Model design v6 adds one target-free basis feature:
`logit(Polymarket market prior) - logit(Chainlink structural probability)`.
Because the residual models use the structural log odds as their offset, a
unit coefficient can represent the executable market prior exactly instead of
approximating that disagreement in probability space. This is a preregistered
representation change, not evidence of predictive uplift or profitability.
Before the full six-candidate fit, `polymarket-round21-ablate-basis` now runs a
bounded paired screen: three regularized logistic residuals without the basis
and the same three with it. Regularization selection uses the first half of
`tune_calibration`, Platt calibration uses the second half, and the untouched
`tune_selection` role supplies paired block-bootstrap intervals. The basis is
retained only when the lower 95% improvement bound is strictly positive for
both condition-equal log loss and Brier score; rejection is final for model
design v6. The command is available through both CLI and generated Windows
workflow contracts, but cannot run until the terminal capture and corpus exist.
No real ablation result exists yet.
Both `polymarket-round21-fit-core` and
`polymarket-round21-fit-matched` require that accepted result as an explicit
input. Validation binds its self-hash, design, publication manifest, terminal
transport, development targets, row ranges, and condition counts before any
full fit; the core fit additionally requires exact dataset hashes. A rejected,
malformed, mismatched, or missing result blocks fitting.
The TCN consumes 16 causal 250 ms rows, resets at every condition or cadence
gap, uses equal condition weight, and is capped at `12 * feature_width + 649`
parameters. It is a bounded challenger, not an edge claim; adding larger
sequence or foundation models is prohibited in this round. TCN training
assigns every condition exactly `1/16` total weight in its shuffled batch.
The final partial batch therefore has proportionally lower total weight instead
of over-weighting its fewer conditions. Causal history boundaries are computed
once and reused across all mini-batches. A bit-identical one-million-row host
benchmark reduced that boundary pass from 0.185285 seconds to 0.007757 seconds
(`23.89x`); this is implementation-throughput evidence, not model performance.

The terminal program
evaluates one development-selected candidate against five probability controls
and all 81 after-cost profile/scenario ledgers. The local
candidate selector compares each candidate's ordered, condition-equal log loss
directly with the best candidate. It uses a Bartlett Newey-West standard error
and circular block-bootstrap intervals with the target-blind block rule
`min(n, max(2, ceil(sqrt(n))))`; consecutive five-minute conditions are not
treated as independent. This reduces false-promotion risk but is not a
performance improvement or edge result. The local
AI comparison is finite: exactly `qwen3.5:9b`, `fin-r1:8b`, and `fino1:8b` use the
same matched development population and deterministic baseline. The pretest
manifest binds the repository, selected layer, terminal captures, model,
economic matrix, test population, and any nominated AI identity. Test access is
then consumed once in a synchronous SQLite state machine; completion, failure,
or interruption cannot reopen it. The shared CLI/Windows
`polymarket-round21-ai-development` workflow implements this terminal path only
when the operator supplies `--acknowledge-one-use-test-access` together with new
store, pretest, claim, and sealed-result paths. The test dataset and target
hashes bind the consumed claim, access, and frozen population. Any nominated AI
is rerun on those exact sealed cases, and its delayed permissions are rewalked
through the same receipt population before comparison with the deterministic
baseline. Its report is canonically rehashed and its exact local model digest
is verified before holdout access, then checked again before inference. No
claim has been opened and no test data has been scored. The unpublished v4
sealed bundle also requires the economic matrix and predictive verdict to bind
the same access-derived dataset and target-manifest hashes; a valid matrix from
another population cannot be substituted. All 81 ledgers must also carry the
same ordered condition/outcome cohort, and its condition/event hash must equal
the predictive population. Completion stores the full validated bundle in the
durable v2 one-use ledger. The shared CLI/Windows
`polymarket-round21-recover-sealed` command can export it after an output-device
failure without reopening test access or rerunning inference.

Historical AI timing is not presented as a measured past host receipt. Every
historical case embeds a canonical virtual event-time schedule, exact causal
market-evidence hash, and explicit false flags for observed historical dispatch
and provider load. Current local queue and inference latency is measured and
shifted onto that schedule before the 81 economic scenarios are evaluated.
Prospective cases remain a distinct schema requiring exact host receipt times.
Case preparation indexes selected probability rows once rather than scanning
the full panel once per condition. The v4 timing contract is explicitly bound
to the current AI-veto design and is included in the one-use repository seal.

A bounded [AMD DirectML training probe](../round-021-directml-training-host-probe-2026-08-03.json)
fit all six `core` candidates in memory. LightGBM resolved to the RX 9070 XT
through OpenCL and the TCN resolved to the same GPU through DirectML, with no
fallback. The fixture was synthetic and the worktree was dirty, so this is only
host integration evidence, not predictive, economic, or reproducibility proof.

A second bounded [TCN v5 DirectML probe](../round-021-directml-tcn-v5-host-probe-2026-08-03.json)
verified the target-blind epoch-rotating endpoint schedule, training-seed
provenance, serialization, and payload validation on the same AMD GPU. It used
32 synthetic conditions and two epochs, wrote no market dataset, and grants no
predictive, profitability, paper, or live authority. The schedule exposes more
within-contract states across epochs without increasing the eight
condition-equal training endpoints per condition per epoch; early stopping
retains a fixed midpoint-stratified panel.

A fresh [probability-basis v6 DirectML probe](../round-021-directml-probability-basis-v6-host-probe-2026-08-03.json)
constructed the new four-column core matrix, verified the exact log-odds
disagreement column, and trained and validated the TCN on the RX 9070 XT through
DirectML with no fallback. It used 32 synthetic conditions, 64 rows per
condition, and two epochs. It used no financial data and supplies no predictive,
economic, profitability, paper, or live-trading evidence.

Matched v2 target-free host probes exercised
[Qwen3.5 9B](../round-021-qwen3.5-9b-target-free-candidate-host-probe-2026-08-03.json),
[Fin-R1 8B](../round-021-fin-r1-8b-target-free-host-probe-2026-08-03.json), and
[Fino1 8B](../round-021-fino1-8b-target-free-host-probe-2026-08-03.json) on the
same frozen numeric packet. Qwen and Fin-R1 produced valid vetoes in 8.322 and
7.940 seconds with zero provider failures and full AMD-GPU residency. Fino1
repeated a schema-valid approval despite `0.005` after-fee edge being below the
hard `0.02` minimum; the shared boundary rejected it as an invalid response,
converted it to a fail-closed veto, and granted no permission. The earlier
[schema-only attempt](../round-021-fino1-8b-target-free-host-probe-attempt1-invalid-approval-2026-08-03.json)
is retained as audit evidence. These are compatibility findings, not model or
financial results. Full development must still evaluate every frozen candidate
on the same economic population, bind exact digests, and unload each model under
the [official lifecycle contract](https://docs.ollama.com/faq) before the next.

The target-free prospective path is implemented but cannot activate without an
accepted sealed result. It converts at most 16 contiguous 250 ms causal rows
into the same inference panel used offline, resets on a condition or cadence
gap, and records exact model, batch, row, latency, and abstention evidence. Its
in-memory coordinator serializes redundant Polymarket CLOB and Chainlink input;
any Polymarket source gap interlocks new entries for that market. Optional
Binance state is discarded and rewarmed independently after a disconnect. It
never opens a database and imports no wallet, order, ledger, or execution
boundary. The credential-free public session now drives two redundant CLOB
channels and one RTDS Chainlink channel. Its strict wire classifier preserves
sequence continuity across the empty opening frame and subscription-history
snapshot observed on the current RTDS service, while excluding both controls
from features. On 2026-08-02, a bounded 15-second live-host transport probe
processed 19,147 frames (19,131 CLOB, 14 live Chainlink updates, two controls)
with zero gaps and queue high-watermark two. A separate three-second probe
through the real union and feature coordinator processed 2,106 frames into 243
union events and three live Chainlink prices, with zero gaps or core
interlocks. Neither probe used credentials, an account, Binance, persistence,
model scoring, or trading authority. These are runtime observations, not
predictive, economic, paper, or live-trading evidence. No prospective
prediction or qualification result exists yet.

Repeated inference no longer reconstructs model state at every 250 ms row.
The scorer validates and compiles immutable feature transforms, LightGBM
boosters, and TCN weights once, then reuses them under a lock. A bounded CPU
fixture with 80 rows and six candidates measured 3.682 ms mean reused
prediction versus 10.355 ms for load-plus-predict, with exact prediction-hash
parity. This is runtime evidence only. The public session also rolls to each
new exact BTC five-minute condition with a fresh feed and coordinator, retries
Gamma discovery while scores remain blocked, and treats feed-processing exit
as a critical failure. A 2026-08-02 three-second live-host session probe made
three successful discoveries and processed 1,309 frames with zero gaps and a
queue high-watermark of two. It used no credentials, account, Binance,
persistence, model scoring, or authority.

The promoted Round 21 decision adapter is also implemented but cannot activate
without a separately verified live promotion. It consumes a Polymarket-only,
hash-bound portfolio rebuilt from exact parent lots and fee-verified cash-flow
evidence. At each consumed 250 ms timestamp it emits at most one initial entry,
exact-parent reduction, or complementary lock. Entries use the selected
outcome's lower probability bound. Reductions use the outcome's upper bound and
must retain positive executable after-fee edge. Locks have priority only when
the original average cost plus the fresh complementary cost leaves positive
guaranteed profit; the execution coordinator independently proves that event
maximum loss cannot rise. Realized event loss remains in the ledger after a
flat exit and cannot be reset by re-entry. Books older than 500 ms, excessive
directional depth participation, insufficient event time, weak edge, active or
unknown transitions, and evidence mismatches fail closed. The adapter owns no
wallet, order, balance, reconciliation, Stop, settlement, or Binance state;
submission remains inside the promotion-gated Polymarket coordinator.

Restart-safe composition is now implemented. The sealed evaluation is exported
as one strict nested bundle containing the predictive verdict, all 81 economic
ledgers, and the selected result. Runtime loading verifies the promotion's exact
model-file and evaluation-file SHA-256 values before reconstructing either
object, then requires the model's internal artifact identity to match the
accepted sealed result. `polymarket-live --action autonomous` selects this
Round 21 stack only for a five-minute promotion; Round 16 pins are required only
for a fifteen-minute promotion. This is wired code, not live qualification: no
accepted Round 21 result, prospective record, authenticated lifecycle
qualification, or live-authority promotion exists.

The independent live boundary now enforces a strict lifecycle-qualification
capability at the actual opening-submission call. Its canonical report must
prove an authenticated user stream, exact bot-owned order identity, and a
confirmed fill-and-parent-close path, followed by a forced reconnect
followed by authoritative reconciliation, unchanged foreign state, and a flat
final account. It binds the source commit, bot, horizon, wallet, and credential
fingerprint, stores no secret material, and grants no authority by itself. The
verifier and generated CLI/Windows controls are implemented. A valid
cancellation-only report remains diagnostic and cannot unlock opening; no real
fill-and-close report exists because this host has not submitted a Polymarket
order.

The separate no-order shadow path is now restart-auditable without weakening
that boundary. It loads exact expected model and sealed-evaluation file hashes,
starts the same rolling public session, and writes target-free scores to a
dedicated SQLite ledger. Each run pre-binds the model, sealed result, and
feature layer. Canonical prediction bytes are bounded, zlib-compressed,
byte-hashed, and SHA-256 chained under immutable update/delete triggers.
Duplicate condition/timestamp writes are idempotent only when their payload is
identical; chronology regression, divergent duplicates, post-terminal writes,
decompression bombs, and semantic tampering fail closed. Full replay audit is
post-terminal only, avoiding a long reader lock against the active 250 ms
writer. Completion, interruption, and failure remain collection states, not
model, profitability, paper, promotion, or live authority. No shadow run or
prospective prediction has been produced. The generated CLI/Windows contract
exposes only `run` and `audit` for this path as
`polymarket-round21-shadow`; it accepts no credential or promotion option.

The optional Binance predictor is a separate credential-free in-memory
sidecar. It connects only when the sealed model layer requires spot or
spot-plus-USD-M features, streams public BTCUSDT `bookTicker` and trade events,
using the [official WebSocket stream contract](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams),
and has no Binance account or execution API. A disconnect clears and rewarms
only optional features. On 2026-08-02, a five-second live-host probe processed
50 spot and 116 USD-M records with zero gaps or reconnects and queue
high-watermark one. A preceding probe exposed the current USD-M `bookTicker`
`ps`/`st` schema extension; the strict parser was updated and regression-tested
before the successful rerun. The rolling session now retains the exact active
identity until its event boundary instead of polling Gamma every second. A
post-change five-second integrated probe made one discovery and processed 2,013
Polymarket frames plus 44 spot-sidecar records, with zero gaps and queue
high-watermarks two and one. The authenticated supervisor uses the same
event-boundary refresh rule while still synchronizing current, next, and owned
user-stream subscriptions on every decision cycle. This is transport evidence
only.

Terminal opening now starts with a separate transport manifest. It fails before
the scheduled campaign end, preserves every interrupted or failed segment and
its coverage hole, and admits only hash-valid complete/degraded runs for later
exact receipt replay. Transport coverage is not condition admission or model
eligibility; redundant-union reconstruction remains mandatory.

The target-blind core materializer now reconstructs each segment's redundant
CLOB union during that same exact receipt pass. It applies per-condition lane,
joint-gap, book, Chainlink open/close, reconnect, role, and causal-availability
gates across all 1,200 five-minute decisions. This is implemented mechanics,
not a completed corpus or model result.

Its publication sink is also implemented but has not run: one same-volume
directory rename publishes separate development and sealed-test DuckDB files,
canonical admission ledgers, and condition-sized Zstandard level-3 binary64
feature chunks bound to the terminal receipt audit. Failed audits remove the
entire staging directory. The pretest v4 seal accepts this validated publication
rather than caller-supplied digests. A sealed-test audit requires its exact claim
and access hash in the durable one-use ledger's consumed state.

The shared CLI and generated Windows workflow now expose the missing terminal
operator boundary. `polymarket-round21-terminal` refuses access before the
scheduled campaign end and atomically writes the hash-bound transport manifest;
`polymarket-round21-corpus` then performs the exact receipt audit and publication.
Neither command opens outcomes or grants trading authority.

Model fitting no longer needs a private storage shortcut: the corpus API exposes
one-pass, hash-audited snapshot readers. Development rows are readable normally;
sealed-test rows require the exact consumed one-use claim and access hash.
`polymarket-round21-fit-core` now assembles official CLOB/Gamma consensus targets
and fits only the frozen core development baseline; its hash-bound artifact records
`trained_layers=["core"]`. The independent
`polymarket-round21-sidecar-terminal` command excludes interrupted sidecar
segments and opens no payloads or outcomes. After both terminal manifests exist,
`polymarket-round21-fit-matched` performs one WAL-free exact-receipt replay,
invalidates optional state at every recorded gap, reconnect, and segment end, and
fits all three layers on the same Polymarket decisions, outcomes, and frozen
development roles. `polymarket-round21-evaluate-development` hash-checks the
result against those exact panels, reconstructs exact top-20 books during one
terminal receipt audit, and advances all 81 fee/latency/depth/adverse-tick/risk
ledgers without materializing a month of books. Optional layers use the exact
matched core population and compare already-streamed matrices, so the receipt
scan is not duplicated. This is development evidence across train and tune roles,
not the untouched seven-day test. Sealed-test access, profitability claims, and
every trading authority remain false.

The AI comparison can consume already-streamed matrices. Its convenience path
advances deterministic and delayed-veto matrices from one condition stream,
sharing only the validated causal book lookup. Cash, inventory, cooldown, risk,
permission, and execution state remain separate. Binance supplies optional
public predictor data only; no Binance account or execution path exists in the
Polymarket comparison.

An optional Binance feature layer cannot pass development because it is
independently profitable. Its selected economic matrix must pass every ledger,
and all 81 matched deltas against the Polymarket-only core must also pass. A
rehashed result that marks a rejected optional comparison as passed is invalid.

`polymarket-round21-ai-development` binds the complete finite AI experiment.
Its first exact source pass builds baseline economics and one historical case
per selected development condition. It then benchmarks the three digest-pinned
local models through a separate resumable response cache. One second source
pass advances all three 81-ledger AI matrices together and applies the frozen
candidate selector. The artifact preserves every case, response, comparison,
and rejection. By default it opens no sealed target. Only the explicit one-use
acknowledgement path may open that target; neither path grants paper or live
trading authority. The terminal path freezes the pretest before writing
development output,
consumes the sealed holdout once, reruns only the nominated AI on the exact
sealed cases, and writes a strict restart-loadable result bundle. A no-edge
verdict is a valid terminal outcome and grants no authority.

The economic engine no longer retains every 250 ms abstention object in every
ledger. It hash-binds the full decision path and retains first/last audit points
plus non-abstain transitions. A flat-inventory condition is collapsed only after
proving from the exact source-bound envelopes and books that neither outcome can
meet the raw minimum edge before costs; any possible edge runs the full stateful
simulator. On this host, the controlled 100-decision/81-ledger no-action case
improved from 9,681 to 588,099 ledger-decisions/second. Its metrics, utility, and
executed-action count match the slow materialized reference. These are software
throughput checks, not profitability results.

Economic replay validation now recomputes condition chronology, cash
continuity, gross and net P&L, daily aggregates, realized drawdown, and every
derivable metric from the retained condition evidence. Settled cash is also
carried into the next conservative equity peak, so a profitable condition
cannot make the following condition's drawdown appear artificially smaller.
The deterministic daily bootstrap is cached only by its exact Decimal series
and identity; caching changes no statistic or evidence hash.

The multi-action risk ledger now carries condition-start cash and pre-condition
drawdown through every 250 ms decision. Its event downside is current cash plus
the guaranteed paired payout, not just open-lot cost basis. This prevents a
loss-making reduction followed by a flat re-entry from resetting the event cap.
Every candidate is bounded at zero and full fill, so a paired reduction or other
transition that increases worst-case downside is rejected when it would breach
the event, daily-loss, or drawdown limit. Risk-reducing locks and reductions
remain available during a gate.

The live ownership ledger now applies the same core invariant at execution:
event downside is reconstructed after restart from exact confirmed BUY cost,
SELL proceeds, redemption payout, remaining parent cost basis, and paired
payout. Targeted reductions can consume only one confirmed unreserved parent.
Complement locks use only unpaired bot inventory and must remain both
guaranteed-profitable after fees and non-increasing in maximum event loss.
Live entry also requires an explicit dedicated-wallet capital basis. A
restart-derived, hash-bound snapshot uses exact parent cost basis, fee-inclusive
SELL proceeds, and verified redemption payout to rebuild daily realized P&L,
settled drawdown, consecutive losing conditions, cooldown, and remaining risk
headroom. The proposal binds that snapshot and the maximum permitted inventory
downside; both are refreshed before submission and the final executable quote
cannot exceed them. This gate cannot block reductions, locks, or Stop.
The affected Round 16, Round 21, autonomous, and live execution checkpoint
passes 674 tests. This is implementation evidence only; no account was
connected, no order was submitted, and no profitability or authority claim was
opened.

The external [OpenMarket study](https://arxiv.org/abs/2607.26245) is a negative
benchmark, not training evidence. Its synchronized archive contains more than
727 million deduplicated rows and reports a median 347 ms Polymarket response
after large Binance moves on the collector clock, but its walk-forward model
still underperformed the same-time Polymarket mid prior and lost after stated
fees and slippage. Round 21 therefore requires strict
improvement over every frozen control, at least 1,800 resolved conditions over
seven calendar days, all 81 economic gates, and explicit optional/AI ablations.
Passing those gates still grants neither paper nor live authority.

Two other 2026 studies are research controls, not claimed edge. A
[75-million-snapshot Polymarket study](https://arxiv.org/abs/2605.00864) found
executable single-market anomalies rare, short-lived, and commonly constrained
by shallow depth. A
[Binance-option/Polymarket threshold study](https://arxiv.org/abs/2606.19517)
found longer-horizon cross-venue wedges, but its hourly threshold contracts do
not establish a five-minute directional signal. These findings support the
existing no-forced-activity rule, exact displayed-depth sizing, and matched
after-cost controls; they do not justify a larger model or more trades by
themselves.

The compact TCN uses the repo's device-neutral training boundary. A live-host
probe completed forward, backward, exact-weight reload, and inference on an AMD
RX 9070 XT through DirectML without an unsupported-operator fallback. DirectML
is not assumed: CUDA, ROCm, XPU, MPS, and CPU remain valid runtime paths. This is
runtime capability evidence only, not model-performance evidence.

Round 21 model design v8 retains v7's target-free, condition-equal
feature-support gate and corrects the causal rolling-window contract. Endpoint
returns and realized variation now start from the latest receipt available at
the window boundary. A genuine move is therefore retained when only one new
tick arrives inside a short window; no future tick, exchange timestamp, target,
or outcome is used. The feature policy and complete downstream design chain
were superseded, preventing predecessor rows from being mixed with corrected
rows. This is a feature-integrity correction, not evidence of predictive edge
or profitability.

Model design v9 adds target-free receipt-timing state needed to test the
cross-venue lead-lag hypothesis without inferring event order from venue source
clocks. It records separate Up/Down Polymarket book ages, Binance spot and
USD-M BBO ages, and signed spot-minus-USD-M BBO receipt skew from the local
collector clock. Future receipts remain rejected. The candidate program,
training schedule, controls, execution matrix, risk policy, AI candidates, and
gates are unchanged. No capture target or outcome was used, and no edge or
profitability follows from the representation change.

Round 16 is a separate, preregistered BTC fifteen-minute comparison. Its
historical/live one-second feature transform is bit-identical, but no model
result exists yet. Before held-out access, its pretest artifact must freeze
train-only feature-support bounds and label-blind tune-only settlement-anomaly
thresholds. A future prospective scorer can activate only from caller-pinned
artifacts that pass every predictive gate; it has no execution authority.

The resumable workflow is intentionally phase-separated:
`python tools/run_polymarket_round16_screen.py status`, then `identities`,
`features`, `development-targets`, and `fit`. The one-use `test-targets` phase
requires `--acknowledge-one-use-test-access`; only then may `evaluate` and
`export` run. No command grants trading authority.

## Audit

- [Evaluation artifact](round-014-evaluation.json)
- [Sealed pretest artifact](round-014-pretest.json)
- [Candidate metrics](tables/round14-candidates.csv)
- [Every held-out decision](tables/round14-decisions.csv)
- [UTC condition series](tables/round14-conditions.csv)
- [Decision-offset metrics](tables/round14-decision-offsets.csv)
- [Cross-round progression](tables/optimization-progress.csv)
- [AI risk-model rejection record](ai-risk-models-rejected.json)
- [Round 21 causal-feature contract](../round-021-causal-feature-policy-v3.json)
- [Round 21 core corpus-materialization contract](../round-021-core-corpus-materialization-design-v3.json)
- [Round 21 execution stress contract](../round-021-executable-action-policy-v3.json)
- [Round 21 matched-model contract](../round-021-matched-model-design-v9.json)
- [Round 21 probability-basis ablation](../round-021-probability-basis-ablation-design-v1.json)
- [Round 21 probability-envelope contract](../round-021-probability-envelope-design-v6.json)
- [Round 21 multi-action policy](../round-021-multi-action-policy-design-v8.json)
- [Round 21 economic-replay contract](../round-021-economic-replay-design-v6.json)
- [Round 21 matched-comparison contract](../round-021-matched-economic-comparison-design-v6.json)
- [Round 21 AI veto contract](../round-021-ai-veto-design-v7.json)
- [Round 21 AI candidate-selection contract](../round-021-ai-candidate-selection-design-v7.json)
- [Round 21 historical AI timing contract](../round-021-ai-historical-schedule-design-v7.json)
- [Round 21 terminal sealed-evaluation contract](../round-021-terminal-sealed-evaluation-design-v7.json)
- [Round 21 terminal transport-manifest contract](../round-021-terminal-transport-manifest-design-v1.json)
- [Publication integrity](publication-integrity.json)
- [Round 25 AI risk contract](../round-025-ai-risk-advisory-contract-v6.json)
- [Round 25 AI host mechanics](../round-025-ai-risk-advisory-host-probe-v7-2026-08-10.json)
- [Round 25 AI safety behavior](../round-025-ai-risk-scenario-host-probe-v3-correction-2026-08-10.json)
- [Round 25 AI uplift contract](../round-025-ai-uplift-evaluation-contract-v2.json)
- [Round 25 target-free sequence inference](../round-025-target-free-sequence-inference-contract-v1.json)
- [Round 25 fitted-model ledger](../round-025-model-ledger-contract-v1.json)
- [Round 25 v2 capture design](../round-025-twap-core-capture-design-v2.json)
- [Round 25 v2 campaign plan](../round-025-twap-core-campaign-plan-publication-v2-2026-08-10.json)
- [Round 25 v2 TWAP wire qualification](../round-025-twap-wire-source-qualification-v2-2026-08-10.json)
- [Round 25 RTDS wire correction v2](../round-025-twap-wire-schema-correction-v2.json)
- [Round 25 terminal receipt materialization v2](../round-025-terminal-receipt-materialization-design-v2.json)
- [Round 25 joint feature materialization v2](../round-025-joint-feature-materialization-contract-v2.json)
- [Round 25 official resolution collection v3](../round-025-official-resolution-collection-contract-v3.json)
- [Round 25 official resolution transport probe](../round-025-official-resolution-transport-probe-v1-2026-08-10.json)
- [Round 25 economic replay v2](../round-025-economic-replay-contract-v2.json)
- [Round 25 post-capture coordinator v4](../round-025-post-capture-coordinator-contract-v4.json)
- [Round 25 Fin-R1 supervisor rejection](../round-025-fin-r1-regime-supervisor-rejection-v1-2026-08-10.json)
- [Round 25 slow LLM supervisor mechanism rejection](../round-025-qwen3-8b-regime-supervisor-rejection-v1-2026-08-10.json)

Regenerate Round 14 tables and charts from the closed local evidence database
with `python tools/publish_polymarket_round14_historical.py`. After an
intentional current-status documentation change, refresh the host-neutral
integrity manifest with `python tools/update_polymarket_latest_manifest.py`.
