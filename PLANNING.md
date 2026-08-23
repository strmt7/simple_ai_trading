# Product Direction

Simple AI Trading is a beta BTC/ETH/SOL day-trading research platform. The
near-term objective is trustworthy paper and testnet/Demo operation, not a
profit promise or a shortcut to mainnet.

## Priorities

1. Prove source-complete, causal, after-cost model evidence before promotion.
2. Keep Binance and Polymarket capital, ownership, and execution independent.
3. Preserve one backend contract across the CLI and native Windows app.
4. Make deterministic risk, reconciliation, Pause, and Stop independent of AI.
5. Keep storage, API use, and model experiments bounded and observable.

## Product Defaults

| Setting | Default |
| --- | --- |
| Risk | Conservative |
| AI | Requested, but enabled only after GPU/model/provenance gates pass |
| Reinvest profits | Off |
| Spot leverage | `1x` |
| Futures leverage | `5x` conservative, `10x` regular, `15x` aggressive |
| Maximum futures leverage | `20x` application ceiling |
| Binance execution | Paper or testnet/Demo only |
| Polymarket execution | Disabled; BTC live boundary has no authority |

## Promotion Standard

A model advances only with reproducible source provenance, causal splits,
explicit spread/fee/latency/liquidity costs, adequate trade activity, positive
after-cost evidence, bounded drawdown, and untouched holdout results. AI must
also show matched uplift over the frozen machine-learning decision. The v5 gate
requires identical metric-unit provenance, recomputed AI-minus-baseline deltas,
strict aggregate P&L/ROI/expectancy improvement, and non-worsening maximum
drawdown; otherwise AI remains a veto-only research component.

The current state and next admissible experiment are recorded in the
[Binance](docs/model-research/action-value/latest/README.md) and
[Polymarket](docs/model-research/polymarket/latest/README.md) status pages.

AI model-size gates infer parameter count from the model identity. An optional
operator value is only a consistency assertion: it cannot supply or override
the inferred count, and any mismatch rejects the uplift report. Live-review v3
also binds the canonical Ollama name, exact metadata parameter count, display
size, metadata hash, model digest, and terminal-model fingerprint into each
hash-chained case. Historical v2 records remain readable but can never satisfy
the authoritative uplift gate.

The remaining AI identity limitation is fail-closed: a custom or `:latest`
model name whose size cannot be inferred by the generic uplift assessor may be
rejected even when the live v3 record contains an exact count. Resolve this in
a later bounded API change using provenance-attested parameter evidence; do not
weaken the current gate or turn an operator assertion into evidence.

## Active Experiment Order

1. Preserve the active Round 75 Binance capture through its fixed
   `2026-08-23T12:00:00Z` boundary and the independent Round 21 Binance sidecar
   through its fixed `2026-08-29T23:40:00Z` boundary. Do not read targets or
   outcomes early. Round 75 currently has 639 fail-closed missed-slot receipts
   and repeated storage-resource-gate failures; do not salvage, retry, or alter
   its frozen process before the contract-defined terminal audit.
2. Run each contract's target-blind terminal, continuity, source, gap, role,
   population, lease, database/WAL, and resource audits. Failed capture gates
   close the affected lineage without salvage or adaptive thresholds.
3. After Round 75 terminalization, preserve and reconcile the detached
   `simple_ai_trading-model-dev` worktree onto `main`. It currently contains
   97 content paths that differ from or are absent on published `main`, plus
   220 dirty-status paths already equivalent to `main`, all based on an older
   ancestor. Never clean it, bulk-stage it, or overwrite newer analyzer fixes.
4. Revalidate every integrated contract hash, focused domain test, and
   generated CLI/native parity surface before treating local files as
   implemented evidence.
5. Only then proceed through the frozen source, causal split, execution-cost,
   delay, access-ledger, sample, model, and one-use evaluation gates. AI remains
   veto/downsize-only until a matched causal uplift test passes.

The currently pushed checkpoint has no reproducible after-cost edge,
profitability claim, or trading authority. Local unintegrated files do not
change that fact.

The next target-blind model experiment is frozen in the
[Round 76 preregistration](docs/model-research/action-value/round-076-mark-conditioned-duration-preregistration-v1.json).
It permits one mark-conditioned, censor-aware interarrival-density challenger
against unchanged v5 controls. Implementation and evaluation remain blocked
until Round 75 terminal adjudication and source reconciliation complete.

## Repository Quality Gate

Model work is not complete while repository analyzers or dependency checks are
red. Fix findings at their type, schema, or ownership boundary; do not hide
them with broad ignores. Run focused tests while editing, then run the full
local matrix once at the final checkpoint. Push only after that matrix passes,
and close the checkpoint only after current-main CI, Ruff, Vulture,
Super-Linter, DeepSource, secret scanning, and available GitHub dependency and
code-scanning checks have been verified. An unavailable scanner is reported as
unverified, never as zero vulnerabilities.

Latest source revision `beffdb2e` passes every GitHub Actions workflow but fails
DeepSource run `168f4123-432e-4b4f-9d4c-a48395b2724f`, which re-raised a
repository-wide 1,268-issue Python inventory. Treat that as an unresolved
bounded analyzer program, not 1,268 confirmed vulnerabilities and not a reason
to revert the AI model-size integrity correction. The exact counts, first page,
and resumption order are in `docs/CONTINUATION.md`.

At the `2026-08-23T02:26Z` closeout, exact revision `42a3721f` passed CI, Ruff,
Vulture, Super-Linter, Python and native C++ CodeQL, and DeepSource. GitHub's
available APIs reported zero open code-scanning, Dependabot, and secret-scanning
alerts, with only `main` and no open pull request. This bounded snapshot is not
proof that no undiscovered vulnerability exists. Historical Bandit and
DeepSource inventories remain triage evidence, not current vulnerability counts
or permission for broad suppressions. See `docs/CONTINUATION.md` for the exact
inventory and integration hazard. Closeout revision `f50c3a83`
removed the current AI-uplift type-contract blockers; DeepSource then displayed
nine changed-scope complexity findings. All nine paths have now been decomposed
in behavior-preserving batches. The five `financial_sanity.py` entry points fell
from Radon scores `188/125/64/36/30` to `2/2/5/1/1`, with exact serialized-report
fingerprints and rejection-order tests protecting the evidence contract. This
closes that bounded maintainability batch; it is not a vulnerability, model-edge,
or profitability claim.

DeepSource's first analysis of the decomposed module applied a stricter helper
threshold and reported 13 residual complexity findings plus two simplification
findings. The simplifications are fixed in the follow-up checkpoint. The 13
helper findings are recorded exactly in `docs/CONTINUATION.md`; they remain a
bounded maintainability queue and must not be presented as vulnerabilities or
silenced with broad analyzer exclusions.

Exact revision `67dab5d8` passed every GitHub Actions workflow. DeepSource
confirmed that the four targeted model-evidence findings were gone and reported
the expected nine residual changed-scope complexity findings. The next bounded
batch now decomposes those nine paths without changing calculations, thresholds,
diagnostics, or serialized report order. Locally, all target scores are `14` or
lower, no replacement helper exceeds `14`, 51 focused and 235 affected tests
pass, and all 133 changed executable lines have coverage. This is maintainability
evidence only. DeepSource analyzed exact revision `bb6abac7`, resolved all nine
targeted findings, and introduced no replacement-helper finding. That wider run
then exposed six type-contract diagnostics in adjacent modules and one test
complexity diagnostic. The follow-up fixes those exact seven items plus five
adjacent local Mypy findings, with all 11 changed executable source lines covered.
Exact code revision `13f33f8a` removed those seven target diagnostics, but its
touched-file DeepSource analysis resolved 208 findings and exposed 27 broader
legacy maintainability findings. The first rendered page is inventoried in
`docs/CONTINUATION.md`; the remaining findings must be exported from that exact
run before another bounded batch starts. Formatting correction `5f6e790c`
passed DeepSource, Ruff, and Vulture. That narrow pass does not erase the
`13f33f8a` inventory or constitute a project-wide security result.

The next bounded market-data integrity batch addresses two findings from that
inventory without broad formatter churn. `MarketDataStore.coverage_quality`
falls from Radon `18` to `7`, with pure helpers scoring `5` and `4`. Top-of-book
ingestion now rejects NaN and infinite prices or quantities, uses an
overflow-safe midpoint, and rejects non-finite derived depth notional before
any write. The 322-test affected matrix passes and all 29 changed executable
lines have coverage. Ruff, scoped Mypy, Pylint errors-only, and strict
simplification checks pass. Bandit still reports the ten documented dynamic-SQL
heuristics outside changed lines. DeepSource analyzed exact code revision
`6370235f`, resolved the targeted coverage-complexity and unsafe-comparison
findings, and introduced only two style findings in the new regression test;
the follow-up removes both. Exact follow-up revision `8d5ecc1e` passed hosted
DeepSource, Ruff, Vulture, and Super-Linter; its longer CI run remained in
progress at the final refresh. This is data-integrity and maintainability
evidence, not model-edge or profitability evidence.

The bounded model-maintainability batch decomposes
`_rule_alpha_score_from_values` from Radon `35` to `3`; no extracted helper
exceeds `12`. It also marks the technical and rule-alpha probability helpers as
static. A frozen numeric contract covers every rule-alpha family, the default
family, empirical two-feature behavior, missing context, non-finite values,
short vectors, and negative trade direction. The 213-test affected matrix
passes; the post-format 20-test numeric contract passes; a deterministic
parent-versus-current harness produced 2416 exact float-hex matches; and all
160 changed executable lines have coverage. Scoped Mypy, Ruff, Pylint
errors-only, Bandit, and strict simplification checks pass locally. No score,
coefficient, calibration, serialization, model result, P&L, risk setting, or
trading authority changes. Exact code revision `31c1e02e` was pushed; its
hosted Ruff range gate found only the two newly decorated signatures needed
formatting. Narrow follow-up `c349aca6` fixes those signatures. Hosted
DeepSource, Ruff, and Vulture pass on that follow-up. Documentation checkpoint
`291c758a` also passed DeepSource, Ruff, Vulture, and Super-Linter; its CI run
remained in progress at the final refresh.

The first bounded SQL-safety checkpoint replaces the optional dynamic predicate
in `latest_microstructure_capture` with one static, parameter-bound query. Its
regression proves passed-only selection, unrestricted selection, and rejection
of an injection-shaped symbol. The full 20-test market-data matrix passes and
both changed executable lines have coverage. Scoped Mypy, Ruff, Pylint
errors-only, and formatter checks pass. Bandit `B608` findings in
`market_store.py` fall from ten to nine without suppression. Exact revision
`63ecb4f2` passed hosted DeepSource, Ruff, Vulture, and Super-Linter; its CI run
remained in progress at the final refresh.

The current follow-up removes the remaining nine dynamic predicate joins. Every
time window and limit is now bound in static SQL; omitted windows use SQLite's
full signed-integer range and an omitted limit binds SQLite's unlimited `-1`.
All timestamp columns are `INTEGER NOT NULL`, and `EXPLAIN QUERY PLAN` confirms
covering-index searches for candles, aggregate trades, top-of-book snapshots,
futures reference bars, and funding rates. The 47-test affected matrix passes,
all 31 changed executable lines have coverage, and scoped Mypy, Ruff, Pylint
errors-only, Bandit, and changed-range formatting checks pass. Bandit now
reports zero findings in `market_store.py`; this does not close dynamic-SQL or
security findings in other modules. Exact revision `646bbdbe` passed hosted
DeepSource, Ruff, and Vulture; its Super-Linter and CI runs remained in progress
at the final refresh.

The current model-evidence SQL batch replaces runtime table, ordering, and
placeholder interpolation in the Polymarket Ridge and MLP materializers with
complete allowlisted select and insert statements. Idempotence, transaction,
runtime-evidence, and tamper-detection behavior remains unchanged. All 19 tests
in the combined Ridge/MLP module pass and all ten changed executable lines have
coverage. Ruff, Pylint errors-only, Bandit, and changed-range formatting checks
pass; both files now report zero `B608` findings. Their one remaining Bandit
`B311` item is SHA-seeded pseudo-random model initialization required for
reproducibility, not a cryptographic or security decision.

The bounded type follow-up closes all 22 scoped Mypy diagnostics in those two
modules. Stored report objects, arrays, integers, and finite numeric values now
have explicit runtime boundaries; Ridge split and candidate reconstruction no
longer relies on unchecked object coercion; and MLP bootstrap quantiles must be
exactly two finite values. The combined 26-test matrix passes and changed-line
coverage clears the 95% gate. Ruff and scoped Mypy pass. Bandit still reports
only the documented non-security `B311` item in these files and no `B608`.
No dataset, feature, target, fit, bootstrap sample, coefficient, model score,
policy, P&L, risk setting, trading authority, or profitability claim changes.

DeepSource analyzed exact type checkpoint `eef01801` and failed with 29
displayed issues, 9 resolved issues, and 8 classified as introduced. Its first
rendered page showed seven MLP complexity findings, one comparison
simplification, one critical unguarded `next()` finding, and optional-row type
findings in `polymarket_fit_claim.py` and `polymarket_recorder.py`. The page did
not expose per-card introduced status, so those categories must not be equated
with the eight-item introduced count. The exact inventory and next order are in
`docs/CONTINUATION.md`.

The next bounded AI-runtime safety batch replaces two optimization-removable
worker-pipe assertions with explicit stdin/stdout/stderr validation. A malformed
foundation-model launcher is now stopped, reaped, and removed from supervisor
state before a non-restartable startup error is returned. The focused two-test
matrix passes, all seven changed executable lines have coverage, and the file's
Bandit `B101` count falls from two to zero without suppression. This changes no
model, prediction, inference payload, timeout, backend choice, risk setting,
P&L, authority, or profitability claim.

The immediate model-evidence contract follow-up removes the unguarded MLP
validation-trial lookup, requires exactly one trial for a selected threshold,
and validates the fit-claim schema probe as one binary integer row before it is
indexed. The two fit-claim dynamic-identifier queries are now complete static
statements selected by an explicit Ridge/MLP identity allowlist; unregistered
table/column pairs fail closed. It also applies DeepSource's exact backend
comparison simplification. The combined 36-test evidence matrix passes, all 22
changed executable lines have coverage, and scoped Mypy, Ruff, and Pylint pass.
Bandit reports zero `B608` in `polymarket_fit_claim.py`. The adjacent recorder
optional-row finding is deliberately not changed: that file is bound by the
immutable Round 27 ledger and requires a reviewed cumulative successor to v18.
No training, prediction, model selection, score, P&L, risk, authority, or
profitability claim changes.

The bounded MLP validation closeout then decomposes backend, member, and
ensemble evidence validation. Their Radon scores fall from `17/25/16` to
`4/5/6`; every extracted helper scores `10` or lower. Pre-edit canonical
fingerprints freeze the backend identity, all three member identities, and the
ensemble identity. Ten focused tests cover valid evidence plus every extracted
fail-closed branch; the 40-test affected matrix passes and changed executable
line coverage is `58/58`. Ruff and formatting pass. A scoped Bandit run reports
no high- or medium-severity finding and retains only the documented source
`B311` deterministic-bootstrap warning. Bare Mypy retains two diagnostics in
unchanged SciPy/training lines and reports none in this validation change.
This is maintainability and evidence-integrity work only: no feature, target,
fit, weight, prediction, threshold, result, P&L, risk, authority, or
profitability claim changes. The report validator, fit-member routine, top-level
fit orchestration, and materializer remain separate bounded complexity tasks.

The report-validation follow-up decomposes `PolymarketMLPReport.validated` from
Radon `61` to `7`; its largest helper scores `12`. It preserves exact canonical
SHA-256 identities for both valid states: validation rejection with untouched
test evidence sealed, and validation admission with complete test evidence.
Eleven malformed contracts cover identity, threshold-grid, admission,
validation-statistic, sealed-test, evaluated-test, canonical-reason, and report
hash failures. The 59-test affected matrix passes at `81.54%` module coverage,
and changed executable-line coverage is `57/57`. Ruff and formatting pass;
scoped Bandit retains only the documented deterministic-bootstrap `B311` and
reports no high- or medium-severity finding. No split, feature, label, fit,
weight, prediction, threshold, partition access, result, P&L, risk, authority,
or profitability claim changes. `_fit_member`, top-level fit orchestration, and
materialization remain three separate bounded tasks.

The fit-member follow-up decomposes `_fit_member` from Radon `17` to `3`; every
training helper scores `5` or lower. A deterministic CPU contract freezes the
input SHA-256, all member parameters, the complete 40-epoch trace, best epoch,
six prediction float-hex values, canonical replay drift, and epoch-progress
payloads. The member and trace SHA-256 values remain exact after extraction.
Direct safety tests cover non-finite weighted loss rejection and rate-limited
batch heartbeats. All 62 affected tests pass and changed executable-line
coverage is `88/88`. Ruff and formatting pass; scoped Bandit reports no high-
or medium-severity finding and only the existing deterministic-bootstrap
`B311`. No tensor dtype, RNG, batch ordering, optimizer operation, loss,
gradient clipping, early stopping, checkpoint, model parameter, prediction,
threshold, partition access, P&L, risk, authority, or profitability claim
changes. Top-level fit orchestration and materialization remain separate tasks.

The top-level follow-up decomposes `fit_and_evaluate_polymarket_mlp` from Radon
`47` to `1`; its largest phase helper scores `8`. Parent authority, causal data
preparation, device fitting, same-seed replay, validation admission, test access,
test evaluation, and report construction are explicit contracts. The existing
150-group parent replay produces byte-identical dataset, Ridge parent, split,
ensemble, member, bootstrap, validation-trial, report, and callback hashes. The
validation-rejected branch returns before any untouched-test row, label,
prediction, or metric access. Controlled mechanics tests cover the admitted
path, parent replay mismatches, all admission and test reason families, optional
progress callbacks, and reproducibility failure without producing financial
evidence. All 71 affected tests pass at `90.68%` module coverage; changed
executable-line coverage is `162/162`. Ruff, formatting, and Mypy with SciPy
stubs pass. Scoped Bandit reports no high- or medium-severity finding and only
the existing deterministic-bootstrap `B311`. No model input, target, tensor,
fit, prediction, threshold, partition policy, evidence identity, P&L, risk,
authority, or profitability claim changes. Materialization is the sole
remaining MLP complexity task in this queue.

The materialization closeout decomposes `materialize_polymarket_mlp_report`
from Radon `23` to `2`; its largest persistence helper scores `8`. Identity
validation, validation/test replay, canonical row construction, schema setup,
existing-evidence verification, and transactional insertion are now explicit
private phases. Static allowlisted SQL, row and sort order, canonical JSON/hash
inputs, idempotence, runtime-evidence repair, tamper rejection, and rollback
semantics are preserved. The 59-test focused matrix passes, including the real
DuckDB create/existing/tamper path, and changed executable-line coverage is
`84/84`. Ruff and formatting pass; scoped Bandit reports no high- or
medium-severity finding. The three scoped Mypy diagnostics are unchanged lines
outside this batch. This closes the documented MLP complexity queue only; it
does not establish model edge, profitability, production authority, or
project-wide analyzer closure.

DeepSource's first exact analysis of this closeout (`d52f623b`) reported 11
introduced findings: five `PYL-R0201` findings in the new in-memory test fakes
and six `TYP-062` findings on one optional aggregate row in
`polymarket_recorder.py`. The narrow follow-up marks the five fake methods
static and passes the current Pylint `no-self-use` extension. Exact follow-up
revision `502568f7` passed DeepSource with zero issues; hosted Ruff, Vulture,
Super-Linter, and the Windows Python/native-UI job also passed. Its long Linux
test-and-coverage job was still running at the bounded refresh.

The ledger-bound follow-up then closes the six recorder findings without an
assertion or analyzer ignore. A dedicated validator requires one exact
two-field row containing nonnegative integer frame and message counts before
any indexing, comparison, or report serialization. Missing, wrong-width,
boolean, non-integer, and negative evidence fails closed. The cumulative Round
27 v19 amendment preserves v7/v17/v18 and binds the new recorder hash to every
original frozen hash. The 39-test affected contract matrix passes, including
real DuckDB cache creation/replay, downstream model/stage-capture source
bindings, and publication reconstruction; all 9 changed executable lines have
coverage. Scoped Mypy with runtime dependencies, Ruff, formatting, manifest,
terminology, and markdown checks pass. No model, feature, target, result, cost,
risk limit, authority, edge, or profitability claim changes.

Frozen source ledgers are immutable. Later analyzer or safety maintenance in a
bound dependency requires a cumulative, predecessor-bound amendment; updating
code without advancing that replacement layer is a CI-blocking provenance
failure. Round 27's cumulative v19 amendment records the current `compute.py`
and `polymarket_recorder.py` evidence-integrity changes without rewriting v7,
v17, or v18.

Exact recorder checkpoint `ec04b2f6` passed DeepSource, Ruff, Vulture,
Super-Linter, and the Windows job. Linux completed the full suite but failed
one test that required a Windows MLP training SHA-256 on Linux. The bounded
replacement keeps exact input and evidence self-hashes, epoch/progress-to-trace
reconciliation, and the production `1e-5` replay tolerance; model fitting and
all financial behavior are unchanged. Exact replacement `d93de892` passed all
five GitHub workflows and DeepSource with zero changed-scope Python issues.

Pinned CodeQL `v4.37.8` now scans Python and the native Win32 C++ target with
`security-extended` queries. Its first runs exposed an eagerly loaded optional
Torch runtime, masked Windows smoke failures, a DPI-audit mismatch, a six-pixel
minimum-layout overflow, two URL host-confusion defects, and one
environment-controlled smoke-log path. Those defects are fixed with focused
regressions. The two credential-fingerprint findings were source-traced and
dismissed individually as non-password false positives. Exact current authority
`42a3721f837dc0681399c0affc65d4b706e76fe3` passed CI, Ruff, Vulture,
Super-Linter, Python and native C++ CodeQL, and DeepSource. GitHub's available
APIs reported zero open code-scanning, Dependabot, and secret-scanning alerts,
with only `main` and no open pull request. This is bounded scanner evidence, not
a claim that no undiscovered vulnerability exists.

## Resume Here

A new development session must start with
[docs/CONTINUATION.md](docs/CONTINUATION.md). It records the fixed data cutoff,
current authority, completed work, safety invariants, and the next ordered
tasks. Repository evidence and live GitHub state must still be verified before
making changes.
