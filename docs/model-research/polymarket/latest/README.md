# Polymarket model status

> **Beta research software. No paper or live trading authority exists.**

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

Round 20/21 is a separate 30-day, target-blind BTC five-minute program scheduled
from 2026-07-30 23:40 UTC through 2026-08-29 23:40 UTC. After a host reboot, the
core Polymarket capture resumed in a new attested segment and the optional,
credential-free Binance predictor resumed in its own worktree and database. A
single-lane Polymarket reconnect gap is explicitly recorded; the frozen
contract rejects affected conditions rather than silently treating them as
continuous or invalidating unrelated conditions. The optional predictor has no
execution, wallet, risk, settlement, Stop, existing-position supervision, or
closure dependency. If configured and unavailable, only signal-dependent new
entries abstain; Polymarket recovery and exits continue independently.

No target, model, AI, economic, or profitability result exists while capture is
active. The finite development ledger now contains three regularized logistic
residuals, two shallow LightGBM residuals, and one compact causal TCN residual.
The TCN consumes 16 causal 250 ms rows, resets at every condition or cadence
gap, uses equal condition weight, and is capped at `12 * feature_width + 649`
parameters. It is a bounded challenger, not an edge claim; adding larger
sequence or foundation models is prohibited in this round. The terminal program
evaluates one development-selected candidate against five probability controls
and all 81 after-cost profile/scenario ledgers. The local
candidate selector compares each candidate's ordered, condition-equal log loss
directly with the best candidate. It uses a Bartlett Newey-West standard error
and circular block-bootstrap intervals with the target-blind block rule
`min(n, max(2, ceil(sqrt(n))))`; consecutive five-minute conditions are not
treated as independent. This reduces false-promotion risk but is not a
performance improvement or edge result. The local
AI comparison is finite: exactly `qwen3:8b`, `fin-r1:8b`, and `fino1:8b` use the
same matched development population and deterministic baseline. The pretest
manifest binds the repository, selected layer, terminal captures, model,
economic matrix, test population, and any nominated AI identity. Test access is
then consumed once in a synchronous SQLite state machine; completion, failure,
or interruption cannot reopen it. No claim has been opened and no test data has
been scored.

The target-free prospective path is implemented but cannot activate without an
accepted sealed result. It converts at most 16 contiguous 250 ms causal rows
into the same inference panel used offline, resets on a condition or cadence
gap, and records exact model, batch, row, latency, and abstention evidence. Its
in-memory coordinator serializes redundant Polymarket CLOB and Chainlink input;
any Polymarket source gap interlocks new entries for that market. Optional
Binance state is discarded and rewarmed independently after a disconnect. It
never opens a database and imports no wallet, order, ledger, or execution
boundary. No prospective prediction or qualification result exists yet.

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
entire staging directory. The pretest v2 seal accepts this validated publication
rather than caller-supplied digests. A sealed-test audit requires its exact claim
and access hash in the durable one-use ledger's consumed state.

The external [OpenMarket study](https://arxiv.org/abs/2607.26245) is a negative
benchmark, not training evidence: its published out-of-sample logistic baseline
did not establish tradable after-cost edge. Round 21 therefore requires strict
improvement over every frozen control, at least 1,800 resolved conditions over
seven calendar days, all 81 economic gates, and explicit optional/AI ablations.
Passing those gates still grants neither paper nor live authority.

The compact TCN uses the repo's device-neutral training boundary. A live-host
probe completed forward, backward, exact-weight reload, and inference on an AMD
RX 9070 XT through DirectML without an unsupported-operator fallback. DirectML
is not assumed: CUDA, ROCm, XPU, MPS, and CPU remain valid runtime paths. This is
runtime capability evidence only, not model-performance evidence.

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
- [Round 21 AI candidate-selection contract](../round-021-ai-candidate-selection-design-v1.json)
- [Round 21 terminal sealed-evaluation contract](../round-021-terminal-sealed-evaluation-design-v1.json)
- [Round 21 terminal transport-manifest contract](../round-021-terminal-transport-manifest-design-v1.json)
- [Round 21 core corpus-materialization contract](../round-021-core-corpus-materialization-design-v1.json)
- [Publication integrity](publication-integrity.json)

Regenerate from the closed local evidence database with
`python tools/publish_polymarket_round14_historical.py`.
