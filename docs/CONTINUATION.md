# Continue Development

This is the authoritative handoff for a new development session. Read
`AGENTS.md` and `docs/AGENT_START.md` first, then verify every stateful claim in
the repository and on GitHub before changing code.

## Current Truth

- The repository is beta `0.1.0-beta.1`. No model has production trading
  authority, and no reproducible after-cost edge currently supports a
  profitability claim.
- Development happens only on `main`. Do not create a long-lived branch or
  discard unintegrated work from another worktree.
- Binance scope is BTC, ETH, and SOL in paper or testnet/Demo mode.
- Polymarket is an independent BTC 5-minute/15-minute track. Its live-capable
  boundary is disabled by default and has no capital authority.
- Historical research has one exclusive repository cutoff:
  `2026-08-14T00:00:00Z`. Do not move it or refetch the latest history during
  each iteration. The machine contract is
  `docs/model-research/research-data-snapshot-contract-v1.json`.
- Prospective captures after the cutoff are isolated, append-only experiments.
  They must never be merged into the frozen historical snapshot or used after
  their preregistered access boundary is violated.
- The old Round 21/27 PIDs (`33804`, `34228`, `65092`, `36844`) are no longer
  alive. Do not restart them from this historical note.

## Live Host Handoff: 2026-08-22

The following state was reverified through approximately
`2026-08-22T17:14Z`. It is host state, not model evidence, and must be
rechecked before any action.

The `2026-08-23` closeout refresh found `main` synchronized with `origin/main`
at parent `1ea2387cc905f1d9a4a7fab73061f842091e1c42` before the final bounded
materializer edit. Round 21 wrapper PID `35008` and child PID `35264` were
responsive. The Round 75 scheduled supervisor was `Ready` and its last result
was `0`. Neither
capture worktree, process, state file, database, or schedule was changed. These
are ephemeral observations and must be reverified after resumption. GitHub
exposed only `main`, no open pull request, zero open Dependabot alerts, and zero
open secret-scanning alerts. Code scanning returned `404: no analysis found`;
it remains unavailable and must not be represented as zero vulnerabilities.

- Main and `origin/main` were synchronized at model-evidence refactor revision
  `9f5efe1573281306a1dde413afe353fcc309bee3` before this documentation
  closeout. GitHub's remote branch inventory contained only `main`, with no
  open pull requests. Always verify
  `git rev-parse HEAD`, `git rev-parse origin/main`, and the remote branch
  inventory directly before resuming.
- Round 75 prospective Binance capture is active from the detached worktree
  `C:\trader\simple_ai_trading-model-dev`. The latest supervisor record was
  `service_healthy`, with no credentials, orders, or trading authority.
  Scheduled task `SimpleAITrading-Round75-Continuous-Capture-Supervisor-v1`
  was ready and its `2026-08-22T17:13:15Z` invocation returned `0`. The service
  state was fail-closed at slot `674`, phase `waiting_for_next_fixed_slot`.
  The state tree contained 36 completed slot directories and 639 immutable
  missed-slot receipts, all classified as
  `service_observed_start_window_elapsed_without_reservation`, with exact host
  cause not established and automatic retry forbidden. The service stderr log
  also contained repeated `Round 75 storage resource gate failed` records.
  Do not infer causality between those facts, salvage missed slots, or modify
  the frozen process; perform the contract-defined terminal resource and
  continuity audit after the campaign ends at `2026-08-23T12:00:00Z`.
- The independent Round 21 Binance spot/futures sidecar is active from clean,
  detached commit `4a5912574a9157d79fecc53bf68ed6f01bb8dac8` in
  `C:\trader\simple_ai_trading-round21-sidecar-v2`. Wrapper PID `35008` and
  physical child PID `35264` were alive. Its state reported segment `12`,
  phase `capturing`, `123162645` written public messages, twelve recorded
  stream gaps, zero recorder errors, no credentials, no execution connection,
  no model-data eligibility, and no paper or live authority. Its frozen
  scheduled end is `2026-08-29T23:40:00Z`.
- All process IDs above are ephemeral snapshots. Process rotation is expected;
  never infer failure, completion, or permission to restart from PID drift.
- Never stop, restart, stage, clean, reset, switch, merge, or modify either
  active capture worktree from this handoff. PID death alone is not a terminal
  verdict; use the contract-defined state, lease, process ancestry, scheduled
  task, database/WAL, and independent terminal audit together.

### Unintegrated Worktree

The Round 75 worktree is detached at
`c42219d47dc781a46411a4ec96838f8a26c3924c`, which is an ancestor of current
`main`. Relative to that old base, Git reports 99 tracked changes and 218
untracked paths. That status is not an unpublished-work count: comparison with
published `main` at `49195b214cdbfd8a188bafa71426e4e7889478c2` found 218
byte-identical files, two local deletions already absent from `main`, 71 files
with different content, and 26 local files absent from `main`. Therefore 97
local content paths require post-capture review; the other 220 must not be
needlessly copied or recommitted. At the live snapshot its tracked binary diff
hashed to Git object
`bf7f896c3fa2b17a7a7a34887b2d3fe04cb4be54`; the sorted untracked
path/content manifest had SHA-256
`f622661d58b6cf044d9e0946c7115ceffa379de553f2eaecaee5b999b02be13b`.
The newest untracked-file write was `2026-08-10T21:43:27Z`.

These hashes are preservation aids, not validation or approval. Do not commit
or change HEAD in that worktree while Round 75 is live: the capture is bound to
its frozen implementation and a repository identity change could invalidate
it. After contract-defined terminalization, snapshot the state again, preserve
the complete diff and untracked manifest, and reconcile it onto `main` with a
reviewed three-way integration. Never use `git clean`, destructive reset, or a
blind overwrite. All new development after integration belongs on `main`.

### Security Snapshot

- GitHub reported zero open Dependabot alerts, zero open secret-scanning
  alerts, no open pull requests, and only `main`. GitHub code scanning returned
  `404: no analysis found`; it is unavailable, not zero.
- `uv lock --check` passed. Dependency manifests and the lock have not changed
  since pip-audit found no known vulnerabilities in the complete exported GPU
  and DirectML dependency stacks. The exact pushed revision
  `053e71ec9959c7b51c73d6983fb574e03f4173ea` passed hosted actionlint and
  Zizmor `1.25.2` through Super-Linter.
- A fresh Bandit audit of `src/simple_ai_trading` reported 522 audit findings:
  zero high severity, 251 medium, and 271 low.
  The medium set is dominated by 248 dynamic-SQL (`B608`) review items. Its
  three `B113` request findings are false positives because each displayed call
  supplies an explicit bounded timeout. The remaining findings still require
  source-by-source triage; this audit does not prove vulnerability or safety.
- `market_store._table_columns` was source-traced: its only caller supplies the
  fixed internal literal `archive_files`, so that `B608` item has no
  attacker-controlled source-to-sink path and needs no speculative patch.
- `MarketDataStore.latest_microstructure_capture` now uses one static query for
  both passed-only and unrestricted selection, with the symbol and mode bound as
  parameters. A direct regression includes an injection-shaped symbol. The
  first checkpoint reduced the module's `B608` count from ten to nine without
  suppression. The follow-up converts the remaining nine query builders to
  static SQL with bound full-range windows and limits. Bandit now reports zero
  findings in `market_store.py`. This is module-level closure only; do not infer
  project-wide dynamic-SQL or vulnerability closure.
- The Polymarket Ridge and MLP report materializers now use complete allowlisted
  select and insert statements instead of interpolated table, ordering, and
  placeholder strings. Bandit reports zero `B608` findings in both files. The
  remaining `B311` item is SHA-seeded pseudo-random model initialization used
  for deterministic reproducibility, not a secret or security decision.
- The final MLP materializer closeout separates identity checks, causal replay,
  canonical row construction, schema creation, existing-evidence verification,
  and transactional insertion. `materialize_polymarket_mlp_report` falls from
  Radon `23` to `2`; the largest extracted helper is `8`. The 59-test focused
  matrix passes, including real DuckDB create/existing/prediction-tamper behavior
  and direct runtime-tamper, missing-runtime repair, replay-drift, and rollback
  checks. Changed executable-line coverage is `84/84`; Ruff and formatting pass.
  Scoped Bandit reports no high- or medium-severity finding. Scoped Mypy reports
  only three annotations on unchanged lines `841`, `1374`, and `1401`; this batch
  adds no diagnostic. The locked environment resolves, the installed Python
  environment audit reports no known third-party vulnerability, and GitHub
  reports zero open Dependabot and secret-scanning alerts. GitHub code scanning
  still returns `404: no analysis found`, and DeepSource remained red at the
  pre-closeout parent. Therefore project-wide vulnerability and analyzer closure
  remain unproven.
- Exact DeepSource run `62f6f94a-e0e5-4e1b-a0fe-7ccd6f093ebd` on revision
  `d52f623b` resolved two findings and reported 11 introduced findings. Five are
  `PYL-R0201` on the new test fakes at original lines `36`, `46`, `61`, `141`,
  and `157`; the immediate follow-up marks those methods static and passes the
  current Pylint `no-self-use` extension. Six are `TYP-062` on the same optional
  aggregate row in `polymarket_recorder.py` at lines `3685`, `3686`, `3694`,
  `3695`, `3701`, and `3702`. That recorder is bound by the immutable Round 27
  source ledger. A future fix must reject an absent or malformed aggregate row
  before indexing and advance the cumulative ledger amendment with provenance
  tests in the same reviewed change; do not add a type assertion, ignore, or
  unbound edit merely to silence DeepSource. Exact follow-up revision
  `502568f7` passed DeepSource with zero issues in three minutes 22 seconds;
  hosted Ruff, Vulture, Super-Linter, and Windows Python/native-UI also passed.
  The Linux test-and-coverage job remained in progress at the bounded refresh.
- A Windows child-process reproducer proved that a fake `git.exe` in the
  process current directory could satisfy an unqualified `git` invocation.
  Package bootstrap now removes empty, relative, duplicate, and
  current-directory PATH entries before any operational module runs and sets
  `NoDefaultCurrentDirectoryInExePath=1` on Windows. The paired exploit test
  proves the unguarded child executes the fake binary while the guarded child
  does not. The 22 `B607` calls remain scanner-visible because Bandit cannot
  model package bootstrap; they are not suppressed. Explicit absolute PATH
  directories remain a host trust boundary and still require controlled
  installation and permissions.
- Two silent terminal-holdout audit exceptions were removed. If terminal
  evaluation or result fingerprinting fails, the model remains rejected; a
  successful ledger update records `evaluation_error`, while a failed update
  leaves the durable reservation in fail-closed `reserved` state and now adds
  `terminal_audit_error` to rejection diagnostics without replacing the
  primary failure. The affected 96 training-suite and terminal-ledger tests,
  Ruff, and a focused Bandit scan pass.
- The remaining two `B110` and one `B112` findings were then removed without
  suppressions. Per-device CUDA memory-probe failures are debug-observable and
  do not hide usable peer devices. A failed Polymarket recovery rollback is
  attached to the primary exception with unknown transaction state. Round 74
  testnet transport cleanup attempts WebSocket, listen-key, and HTTP-session
  closure, reports every failure, and does not mask an active primary error.
  Focused tests, mypy, Ruff, and the full `B110`/`B112` scan pass. These changes
  do not alter model outputs, evidence, risk limits, or trading authority.
- AI uplift schema `ai-uplift-v5` now fails closed when aggregate ROI does not
  strictly improve over the machine-learning baseline. It records the exact
  metric alias used by each treatment, rejects baseline/AI alias mismatches,
  normalizes explicit drawdown and win-rate percentage aliases, validates
  count/range inputs, and requires strict aggregate P&L, ROI, and expectancy
  improvement without worse maximum drawdown. The independent financial-sanity
  reader validates v5 provenance, recomputes every reported delta, and applies
  the same directional gates so a hand-edited `accepted` flag cannot bypass the
  production assessor. This is evidence-contract hardening, not AI uplift,
  market-edge, or profitability evidence.
- Code checkpoint `5c220bcfa34028640fb3ac00d8953314c291d7fd`
  passed 189 focused AI/uplift/review/financial-sanity/Polymarket tests, local
  Ruff check, Pylint errors-only, targeted Bandit, terminology, lock, and diff
  checks. Hosted Vulture and Super-Linter passed. Hosted Ruff failed because a
  newly added long expression did not satisfy its changed-range format gate;
  the closeout formats only the new ranges and locally reproduces zero overlap.
  DeepSource failed with 32 displayed issues, 34 resolved, and 13 classified as
  introduced. The blocking items were strict object-to-number type errors in
  the newly reclassified `ai_uplift.py`; the remainder shown were legacy
  complexity findings. The closeout adds typed period/bootstrap contracts and
  guarded finite/integer/reason parsing. Current mypy then reports zero issues
  in both changed source modules, compared with 18 on exact parent `486f0506`.
  The local closeout passes 191 affected tests and 96% changed-line coverage,
  plus Ruff check, Pylint errors-only, targeted Bandit, markdownlint,
  terminology, lock, and diff checks. Hosted Ruff, Vulture, Super-Linter, and
  the Windows Python/native-UI job then passed on closeout revision
  `f50c3a8302090b15bce8075f425716dcf2afc659`. Its full Linux
  test-and-coverage job was still running at the handoff and is not a pass
  until GitHub reports a successful terminal result. DeepSource reanalysis
  resolved the displayed type-contract findings. It still failed with only
  `PY-R1000` cyclomatic-complexity findings visible in the changed Python
  scope: four in `ai_uplift.py` and five in `financial_sanity.py`. These are
  maintainability findings, not reported security vulnerabilities; preserve
  behavior and reason ordering if addressing them in a later bounded refactor.
- Full Linux CI on documentation closeout revision `68c1a422` exposed 29
  cascading Round 27/28 failures. The immutable v7 source ledger and cumulative
  v17 replacement layer no longer recognized the later `b5144c98`
  exception-path changes in `compute.py` and `polymarket_recorder.py`. The
  cumulative v18 maintenance amendment preserves both predecessors and binds
  the original ledger hashes to the current implementations. It changes no
  model mathematics, feature, target, cost, threshold, risk gate, authority, or
  result. The exact previously failing cluster now passes 70 tests locally with
  100% changed-line coverage, plus Mypy, Ruff/format, Pylint errors-only,
  targeted Bandit, lock, manifest, and diff checks. Repair revision `580e4a0e`
  subsequently passed full hosted CI, Ruff, Vulture, Super-Linter, DeepSource,
  and the Windows Python/native-UI smoke job.
- The four `ai_uplift.py` complexity findings from `f50c3a83` were decomposed
  in revision `9f5efe15` without changing output construction or rejection
  order:
  `AIUpliftPolicy.__post_init__` fell from 24 to 1, `_matched_period_deltas`
  from 22 to 9, `_statistical_evidence` from 20 to 2, and `assess_ai_uplift`
  from 33 to 3. The affected matrix passes 217 tests and the 156 changed
  executable lines have 100% coverage. Mypy, Ruff/format, Pylint errors-only,
  targeted Bandit, and Radon passed locally. Exact-SHA hosted CI, Ruff,
  Vulture, Super-Linter, DeepSource, and the Windows Python/native-UI smoke job
  also passed.
- The five deferred `financial_sanity.py` paths were subsequently decomposed
  without changing calculations, check order, labels, paths, metrics, limits, or
  report serialization. Radon scores fell from `188` to `2` for
  `build_model_lab_financial_sanity_report`, `125` to `2` for
  `build_backtest_financial_sanity_report`, `64` to `5` for
  `_selection_risk_checks`, `36` to `1` for
  `_ai_uplift_period_evidence_checks`, and `30` to `1` for
  `_probability_calibration_checks`. The module's highest remaining score is
  `23`. Four fixed SHA-256 fingerprints cover complete good/bad backtest and
  model-lab reports. Locally, 49 focused financial-sanity tests and the 233-test
  affected matrix pass; changed executable-line coverage is `95.7%` (`741`
  lines, `32` missing), clearing the `95%` gate. Ruff/format, Pylint
  errors-only, targeted Bandit, and Radon pass. Mypy reports no error in the
  changed module but still follows imports into three unchanged modules with
  nine recorded type errors. At that local checkpoint, hosted results were
  unverified; the next bullet records the first exact-SHA analyzer result.
- DeepSource failed revision `eaff89305fca00073e8d701bd8a85ad0acc0d920`
  with 15 changed-scope maintainability findings and no displayed security
  finding: 13 helper functions exceeded its strict complexity threshold, one
  nested conditional was collapsible, and one comparison could be chained.
  The two simplification findings are fixed in the immediate follow-up. The
  residual complexity queue is `_walk_forward_gate_checks` (`23`),
  `_terminal_holdout_check` (`23`), `_ai_uplift_count_checks` (`22`),
  `_ai_uplift_period_contract_checks` (`19`), `_market_edge_checks` (`19`),
  `_ai_uplift_rate_checks` (`19`), `_ai_uplift_bootstrap_checks` (`18`),
  `_backtest_sequence_checks` (`18`), `_backtest_log_identity_checks` (`18`),
  `_probability_deterioration_checks` (`17`),
  `build_model_financial_sanity_report` (`17`), `_data_coverage_checks` (`16`),
  and `_accepted_portfolio_symbol_checks` (`16`). Do not broaden this batch or
  suppress the rule during closeout. A later agent should decompose one coherent
  contract family at a time under the fixed full-report fingerprints and then
  verify the exact new DeepSource inventory.
- The next local model-evidence batch reduces four functions from that exact
  queue without changing calculations or diagnostics:
  `_ai_uplift_period_contract_checks` falls from `19` to `8`,
  `_ai_uplift_bootstrap_checks` from `18` to `9`,
  `_probability_deterioration_checks` from `17` to `8`, and
  `build_model_financial_sanity_report` from `17` to `1`. The largest extracted
  predicate scores `12`, below DeepSource's observed threshold. Accepted and
  rejected model reports add two fixed SHA-256 fingerprints to the four existing
  complete report contracts. Locally, 49 focused tests and the 233-test affected
  matrix pass, and all `62` changed executable lines have coverage. Scoped Mypy,
  Ruff/format, Pylint errors-only, targeted Bandit, strict simplification checks,
  and Radon pass. This is maintainability evidence only; hosted DeepSource must
  confirm the residual inventory on the exact pushed SHA.
- Exact revision `67dab5d8d4acbb12d71077a2fa30cd62ac5f03f3` subsequently
  passed CI, Ruff, Vulture, Super-Linter, and Windows native-UI smoke.
  DeepSource confirmed that the four targeted findings were gone and reported
  exactly the expected nine residual changed-scope complexity findings.
- The current local batch decomposes all nine residual paths. Their Radon scores
  are now `5/3/6/14/14/12/14/1/1` in the documented DeepSource order, and no
  replacement helper exceeds `14`. The six complete accepted/rejected report
  fingerprints remain unchanged. Locally, 51 focused tests and the 235-test
  affected matrix pass, all 133 changed executable lines have coverage, and
  scoped Mypy, Ruff/format, Pylint errors-only, targeted Bandit, strict
  simplification checks, and Radon pass. Hosted DeepSource must analyze the exact
  pushed checkpoint before this changed-scope queue is called closed.
- DeepSource analyzed exact revision
  `bb6abac7b33cd3ffe089c05872283beef3bc13f0`, resolved all nine targeted
  findings, and introduced no replacement-helper finding. The run remained red
  for six adjacent type-contract diagnostics in `market_store.py`, `model.py`,
  and `terminal_holdout_ledger.py`, plus one complexity diagnostic in
  `test_financial_sanity.py`. The current follow-up fixes those seven exact items
  and five adjacent local Mypy findings. All 11 changed executable source lines
  have coverage; the focused five-file runtime matrix, direct legacy action-value
  tree test, scoped Mypy, Ruff, strict simplification checks, and Pylint
  errors-only pass. Pylint's only unfiltered errors are the unavailable optional
  `torch` imports. The file-wide Bandit scan still reports the documented dynamic
  SQL and assertion triage queue, but no finding is on a changed line.
- Exact code revision `13f33f8a2b78e826579d65611763f40fef60e9cc`
  removed the seven targeted DeepSource diagnostics. Its broader touched-file
  analysis resolved 208 findings and introduced 27 legacy maintainability
  findings. The first rendered page included complexity in
  `MarketDataStore.coverage_quality`, `_rule_alpha_score_from_values`,
  `_serialized_mlp_output`, `_signed_payoff_lightgbm_ranker_prediction`,
  `predict_payoff_evidence`, `_temperature_scan_torch`, and
  `calibrate_probability_temperature`; two instance-independent methods; one
  protected-member access; and a suggested rewrite of `not value > 0.0`.
  Do not apply that comparison rewrite directly because it would accept NaN.
  Export and classify all 27 findings from the
  [exact DeepSource run](https://app.deepsource.com/gh/strmt7/simple_ai_trading/run/baab8e6e-dbe8-4653-8d1c-e884778ea8df/python/)
  before changing another coherent family. Formatting correction `5f6e790c`
  passed DeepSource, Ruff, and Vulture; that one-line-delta result does not
  erase the parent run's 27-item inventory.
- The next bounded market-data batch safely implements two items from that
  inventory. `MarketDataStore.coverage_quality` drops from Radon `18` to `7`;
  its extracted empty-window and streamed-gap helpers score `5` and `4`.
  Top-of-book ingestion explicitly rejects non-finite and non-positive prices
  or quantities, calculates the midpoint as `bid + spread / 2` to avoid finite
  input overflow, and rejects non-finite derived depth notional before opening
  a transaction. Direct regressions cover NaN, positive and negative infinity,
  zero, negative quantity, bounded empty windows, null defensive rows, finite
  extreme inputs, overflow rejection, and absence of partial writes. The
  322-test affected matrix passes and changed-line coverage is `29/29`.
  Scoped Mypy, Ruff, Pylint errors-only, strict simplification checks, and
  Markdown lint pass locally. Bandit still reports ten pre-existing `B608`
  dynamic-SQL heuristics outside changed lines and no high-severity issue.
  DeepSource analyzed exact revision `6370235f`, resolved the targeted
  coverage-complexity and unsafe-comparison findings, and introduced only
  `PY-W0072` and `PTC-W0062` in the new test. The follow-up constructs the
  parameterized payload in one expression and uses one compound context manager;
  its focused eight-test file, Mypy, Ruff, and formatter checks pass. Exact
  follow-up revision `8d5ecc1e7c0807892055643d4bd92cbeea70f7ac` passed hosted
  DeepSource, Ruff, Vulture, and Super-Linter. Its longer CI run remained in
  progress at the final refresh. Do not infer that the parent 27-item queue is
  otherwise closed.
- The bounded rule-alpha batch decomposes
  `_rule_alpha_score_from_values` from Radon `35` to `3`; its extracted helpers
  score at most `12`. It also makes `_technical_probability` and
  `_rule_alpha_probability` static without changing their call contract. A
  frozen numeric regression covers every named rule-alpha family, the default
  family, empirical two-feature behavior, absent higher-timeframe context,
  non-finite values, an empty feature vector, and negative trade direction.
  The 213-test affected matrix passes. After the mechanical formatting pass,
  the 20-test exact-score contract passes; a deterministic in-memory harness
  comparing the parent and current implementations produced 2416 exact
  float-hex matches; and changed-line coverage is `160/160`. Scoped Mypy, Ruff,
  Pylint errors-only, Bandit, and strict simplification checks pass locally. No
  score, coefficient, calibration, serialization, model outcome, P&L, risk
  threshold, or trading authority changes. Exact code revision
  `31c1e02e64b8c8056a437f2fefa20d283dac2601` was pushed to `main`. Its hosted
  Ruff range gate found only the two newly decorated signatures needed
  formatting. Narrow follow-up
  `c349aca6337bc5940d471a60c325f05d5a6d4f3b` fixes those signatures; its
  20-test exact-score contract and local changed-range reproduction pass.
  Hosted DeepSource, Ruff, and Vulture pass on that follow-up. Documentation
  checkpoint `291c758ab559932c24d6dd6d530315b625f13e9a` also passed DeepSource,
  Ruff, Vulture, and Super-Linter. Its CI run remained in progress at the final
  refresh and must not be recorded as a pass until its exact terminal result is
  read.
- The first bounded SQL-safety checkpoint removes dynamic predicate assembly
  from `latest_microstructure_capture` without changing its public contract. A
  new isolated regression covers passed-only selection, unrestricted selection,
  case normalization, and an injection-shaped symbol. The complete 20-test
  market-data matrix passes; changed-line coverage is `2/2`; and scoped Mypy,
  Ruff, Pylint errors-only, and formatter checks pass. Bandit reports nine
  remaining `B608` heuristics in `market_store.py`, down from ten, with no
  suppression. Those nine use internal fixed predicate fragments and bound
  external values. Exact revision
  `63ecb4f2af0f4bb62566105084f873ce25dd9ad9` passed hosted DeepSource, Ruff,
  Vulture, and Super-Linter. Its CI run remained in progress at the final
  refresh.
- The current follow-up replaces those remaining nine predicate joins with
  static, parameter-bound SQL. Omitted windows bind SQLite's full signed-integer
  range, and an omitted result limit binds SQLite's unlimited `-1`; the filtered
  schema columns are all `INTEGER NOT NULL`. Representative `EXPLAIN QUERY PLAN`
  checks retain covering-index searches for candles, aggregate trades,
  top-of-book snapshots, futures reference bars, and funding rates. The 47-test
  affected matrix passes, changed-line coverage is `31/31`, and scoped Mypy,
  Ruff, Pylint errors-only, Bandit, and changed-range formatting checks pass.
  Bandit reports zero findings in `market_store.py`. Exact revision
  `646bbdbe5d3a45637462fca83755cfcc47645b6c` passed hosted DeepSource, Ruff,
  and Vulture. Its Super-Linter and CI runs remained in progress at the final
  refresh. Other modules retain their own SQL review queues.
- The current model-evidence SQL batch replaces the Ridge and MLP materializer's
  runtime table, ordering, and placeholder interpolation with complete
  allowlisted SQL statements. The full 19-test combined module passes, including
  fit-claim reservation, deterministic fitting, idempotence, transaction,
  runtime-evidence, CLI parity, and tamper-detection paths. Changed-line coverage
  is `10/10`; Ruff, Pylint errors-only, Bandit, and changed-range formatting
  checks pass; and both files now report zero `B608` findings.
- The bounded Ridge/MLP type follow-up closes all 22 scoped Mypy diagnostics.
  Explicit stored-object, stored-array, strict-integer, and finite-number guards
  now protect pipeline batches, action metadata, split groups, candidate rows,
  selected-policy evidence, and report reconstruction. The MLP bootstrap also
  requires exactly two finite quantiles before constructing evidence. The full
  19-test module and seven focused malformed-evidence contracts pass together;
  changed-line coverage clears the 95% gate, and Ruff/format and scoped Mypy
  pass. Bandit reports no `B608` in either file and retains one low-severity
  `B311` at the SHA-seeded deterministic bootstrap RNG. That RNG is required for
  reproducible statistical evidence and is not used for secrets or security.
  No dataset, feature, target, fit, bootstrap sample, coefficient, model score,
  policy, P&L, risk setting, authority, or profitability claim changes. Hosted
  analysis of the exact pushed checkpoint remains required.
- DeepSource analyzed exact checkpoint
  `eef018010903fc851783ceab9314f03f88892ebf` and failed with 29 displayed
  issues, 9 resolved, and 8 classified as introduced. Its first rendered page
  contained seven MLP complexity findings in
  `PolymarketMLPBackendEvidence.validated`, `PolymarketMLPMember.validated`,
  `PolymarketMLPEnsemble.validated`, `PolymarketMLPReport.validated`,
  `_fit_member`, `fit_and_evaluate_polymarket_mlp`, and
  `materialize_polymarket_mlp_report`; one comparison simplification in backend
  validation; one critical unguarded `next()` in MLP materialization; and
  optional-row type findings in `polymarket_fit_claim.py` and
  `polymarket_recorder.py`. The public page did not identify introduced status
  per card, so do not map those eleven rendered items onto the eight-item
  introduced count. Ruff, Vulture, Windows smoke, and Super-Linter passed on
  that exact revision; Linux coverage remained in progress at the last refresh.
- A fresh in-memory Bandit scan of exact `eef01801` reported 508 findings: zero
  high severity, 237 medium, and 271 low. The inventory was 234 `B608`, 183
  `B101`, and 91 other items. The immediate AI-worker lifecycle batch replaces
  the two assertions in `foundation_worker_client.py` with an explicit
  three-pipe gate. A hanging malformed launcher is closed, terminated, reaped,
  and removed from supervisor state before startup fails. Its two focused tests
  pass, changed executable-line coverage is `7/7`, and focused Mypy, Ruff,
  Pylint errors-only, and Bandit pass. The file now has zero `B101` findings
  without suppression. No model, prediction, inference payload, timeout,
  backend, P&L, risk, authority, or profitability claim changes.
- Exact worker checkpoint `1ad4b84a03ba0778e4a89074354f39eb984632d0`
  passed hosted DeepSource, Ruff, Vulture, Windows smoke, and Super-Linter. Its
  Linux coverage job remained in progress at the last refresh.
- The immediate model-evidence contract follow-up replaces the unguarded MLP
  `next()` with a helper that requires exactly one validation trial for the
  selected threshold. Missing and duplicate matches now fail with an explicit
  evidence-contract error. The fit-claim information-schema probe now requires
  one row containing exactly `0` or `1` before indexing. Both fit-claim dynamic
  identifier queries are replaced by complete static Ridge and MLP statements;
  any unregistered report table/parent-column pair now fails identity
  validation. During post-capture Round 75 integration, inventory every new
  fit-claim caller and add its complete static statements before enabling that
  model family; never reopen arbitrary identifier construction. DeepSource's
  exact backend comparison simplification is also applied. The full 19-test
  Ridge/MLP module and 17 focused malformed-evidence
  contracts pass together; changed executable-line coverage is `22/22`, scoped
  Mypy, Ruff, and Pylint errors-only pass, and Bandit reports zero `B608` in
  `polymarket_fit_claim.py`. The adjacent `polymarket_recorder.py` optional-row
  finding is intentionally deferred because the file is bound by Round 27's
  immutable source ledger; changing it requires a reviewed cumulative successor
  to v18 and must not occur casually during the active capture. No model fit,
  prediction, score, policy, P&L, risk, authority, or profitability claim
  changes. Hosted analysis of the final pushed checkpoint remains required.
- Exact parent `75160c3d4c2f88011026448b6153dc91dbcd83de` passed hosted Ruff,
  Vulture, Windows smoke, and Super-Linter at the closeout refresh. Its Linux
  coverage job was still running. DeepSource failed that exact parent because
  blocking complexity findings remained; the exact run ID is
  `f4b06373-b6fd-4781-aa6d-2999dbdc696b`. Pending is not pass, and this result
  is not a project-wide vulnerability count.
- The final bounded MLP validation batch decomposes only
  `PolymarketMLPBackendEvidence.validated`,
  `PolymarketMLPMember.validated`, and
  `PolymarketMLPEnsemble.validated`. Their Radon scores fall from `17/25/16` to
  `4/5/6`; every new helper scores `10` or lower. Pre-edit SHA-256 fingerprints
  freeze the backend identity, each seeded member identity, and the ensemble
  identity. Ten focused tests cover canonical evidence and every extracted
  fail-closed branch. The 40-test affected matrix passes at `78.42%` module
  coverage and changed executable-line coverage is `58/58`. Ruff and format
  pass. Scoped Bandit reports no high- or medium-severity finding and only the
  existing source `B311` deterministic-bootstrap item. Bare Mypy retains two
  diagnostics at unchanged SciPy/training lines and reports none in the changed
  validation paths. No feature, target, split, fit, weight, prediction,
  threshold, evidence identity, result, P&L, risk, authority, or profitability
  claim changes. Hosted checks on the final pushed SHA must be read directly.
- Exact validation checkpoint
  `3fda7ff7b9caa5b97dbc0378c819fcd459814a1b` passed hosted Windows smoke,
  Ruff, Vulture, and Super-Linter. Its Linux coverage job remained in progress
  at the refresh. DeepSource failed with blocking issues under exact run ID
  `22ab322b-8b09-4dfd-bab8-4f0d6d38ec2a`; do not infer the residual issue
  inventory from its aggregate failure state.
- The report-validation follow-up decomposes
  `PolymarketMLPReport.validated` from Radon `61` to `7`; its largest helper
  scores `12`. Parent behavior is frozen with canonical no-trade report SHA-256
  `f24c8a92186309073c4d07be3c1ce2fbc3012af5a59eda3e77e8b38b35a90f9c`
  and admitted report SHA-256
  `9c471198599a852778f6e0500245095a201effda708e969eea1aff82540da05b`.
  Eleven malformed reports cover identity, threshold-grid, admission,
  validation-statistic, sealed-test, evaluated-test, canonical-reason, and hash
  failures. The 59-test affected matrix passes at `81.54%` module coverage;
  changed executable-line coverage is `57/57`. Ruff and format pass. Scoped
  Bandit reports no high- or medium-severity finding and retains only the
  existing deterministic-bootstrap `B311`. Bare Mypy retains the same two
  diagnostics at unchanged SciPy/training lines and reports none in the changed
  report-validation paths. No split, feature, target, fit, weight, prediction,
  threshold, partition access, evidence identity, result, P&L, risk, authority,
  or profitability claim changes. Hosted checks on the pushed revision remain
  required.
- Exact report-validation checkpoint
  `ab1cab7bc6a0f5f49a4900b285710b2acf751db7` passed hosted Windows smoke,
  Ruff, Vulture, and Super-Linter. Its Linux coverage job remained in progress
  at the refresh. DeepSource failed with blocking issues under exact run ID
  `b0c9481e-2a8b-404f-a68e-7fbfe6e2bc21`; do not infer its issue inventory from
  the aggregate state.
- The fit-member follow-up decomposes `_fit_member` from Radon `17` to `3`; all
  extracted helpers score `5` or lower. A deterministic CPU contract freezes
  input SHA-256
  `a88b58b8e0bfc9165f87c3b8977028d0366c1e0ae13392136d4def96f34920f3`,
  member SHA-256
  `f9da99f54c0c3a57deb3aec51b4d7c63c7a41dec053e8a20e1702a2cf927ae14`,
  trace SHA-256
  `7052e95e3fad0a990df1646e73c1117804b97f741836c2c9cfff33ea5ba08b89`,
  best epoch `20`, epoch count `40`, all six prediction float-hex values,
  replay drift, and epoch-progress SHA-256
  `2574cb15bcc3fa2b6fca5c985253dc1d33a402ee6f89a82ab58358b9e3c13a07`.
  Direct tests also cover non-finite weighted loss rejection and rate-limited
  batch heartbeats. All 62 affected tests pass; changed executable-line
  coverage is `88/88`. Ruff and format pass. Scoped Bandit reports no high- or
  medium-severity finding and only the existing deterministic-bootstrap
  `B311`. Bare Mypy retains the same two diagnostics at unchanged
  SciPy/training-orchestration lines and reports none in the fit-member change.
  No tensor dtype, RNG, permutation, optimizer operation, loss, gradient clip,
  early stopping, checkpoint, parameter, prediction, threshold, partition
  access, P&L, risk, authority, or profitability claim changes. Hosted checks
  on the pushed revision remain required.
- Exact fit-member checkpoint
  `442dc3d59a7d208f7c9dccdf7472b487abf1a876` passed hosted Windows smoke,
  Ruff, Vulture, and Super-Linter. Its Linux coverage job remained in progress
  at the refresh. DeepSource failed with blocking issues under exact run ID
  `d07c554b-884a-4135-8ecf-17982d6a7461`; do not infer its issue inventory from
  the aggregate state.
- The top-level orchestration follow-up decomposes
  `fit_and_evaluate_polymarket_mlp` from Radon `47` to `1`; its largest phase
  helper scores `8`. Parent authority, causal data preparation, device fitting,
  reproducibility, validation admission, admission-gated test evaluation, and
  immutable report construction are now explicit. The 150-group replay remains
  byte-identical at dataset SHA-256
  `1dbe87ac256d2f6851235d2cb38eafa982e62e1be7e5e84c14f095eaeadacf29`,
  parent SHA-256
  `a9adaf23a8fd1851636cadfebb04f5f4a65d4ce63f9b028f33dcf119ec3cf21d`,
  ensemble SHA-256
  `e05f81ca3391f7bb802e5183cddb4a3a8b563807d74bacf6deac50f693215e7d`,
  report SHA-256
  `c2d130107c162845fbd44b65c13f44131978538d5f60f8716cb2fd6d7b7f1af9`,
  and non-batch callback SHA-256
  `20bb664a7c2317719f3e6764e7516f6b21403da9a831726b62e1e0ec616d8a12`.
  Member, split, bootstrap, and validation-trial hashes also match the recorded
  parent values. The rejected branch returns before untouched-test rows, labels,
  predictions, or metrics are accessed. Nine controlled mechanics tests cover
  admitted execution, replay mismatches, all reason families, optional progress,
  and reproducibility failure without creating model or financial evidence.
  All 71 affected tests pass at `90.68%` module coverage; changed executable-line
  coverage is `162/162`. Ruff and format pass. Mypy with ephemeral SciPy stubs
  reports zero issues; the locked environment without that optional stub package
  reports only the known SciPy import diagnostic. Scoped Bandit reports no high-
  or medium-severity finding and only the existing deterministic-bootstrap
  `B311`. No model input, target, tensor, fit, prediction, threshold, partition
  policy, evidence identity, P&L, risk, authority, or profitability claim
  changes. Hosted checks on the pushed revision remain required.
- Exact parent `486f0506d60857e291941c7f17580c172b8ea5ca`
  passed hosted Ruff, Vulture, Super-Linter, and DeepSource. Its longer CI run
  was still in progress at the snapshot.
- The last full DeepSource backlog verification on `2026-08-16` found about
  28000 active findings, including 118 security findings. DeepSource passed
  revision `053e71ec9959c7b51c73d6983fb574e03f4173ea` with no blocking issues or
  failing metrics; the historical project-backlog count was not refreshed and
  must not be represented as current.
- DeepSource failed exact revision
  `c6ff3ac286eca7d58938ff9eded3c461f9ad60ea` with 107 Python findings, 88
  classified as introduced. Its first rendered page contained fifteen major
  type-contract findings in `backtest.py`, `execution_simulation.py`, and
  `financial_sanity.py`; it did not identify those findings as security
  vulnerabilities. Revision
  `16c40e26d278caf5eba166609fb70716325adaf1` added explicit
  numeric-conversion guards, a typed scoring-backend payload, unambiguous
  optional trade-limit and threshold-variant types, and primitive-only sanity
  diagnostics. Its hosted DeepSource run reduced the exact inventory to 86
  findings, reported 536 resolved findings, and classified 75 remaining
  findings as introduced. The follow-up closeout revision removes the next
  page's loop-variable closure captures, one no-op assignment, one unnecessary
  generator, and a simple comparison warning. It also makes two accelerated
  scorer parity tests execute through Torch on CPU when DirectML is absent.
  Focused behavioral tests, mypy, Ruff, Bandit, and Pylint pass locally. Large
  function-complexity findings and the rest of the DeepSource backlog remain.
  DeepSource failed follow-up revision
  `2457d9d59f2c680998d2e296a0d4ce5639c5f049`: its headline Python count fell
  to 51, while the change summary reported 61 issues, 10 resolved and 51
  introduced. Its first rendered item was the intentionally deferred
  cyclomatic-complexity finding for `_batch_probabilities_torch`; do not infer
  that the remaining backlog is fixed or that every finding is a vulnerability.

### Round Closeout Checkpoint

Code revision `5c220bcfa34028640fb3ac00d8953314c291d7fd` and closeout revision
`f50c3a8302090b15bce8075f425716dcf2afc659` were pushed to `main` with the
AI-uplift v5 contract described above. At the snapshot, GitHub showed only
`main`, no open pull requests, zero open Dependabot alerts, and zero open
secret-scanning alerts. GitHub code scanning returned `404: no analysis found`
and remained unavailable. Hosted Ruff, Vulture, Super-Linter, DeepSource, and
the Windows job passed on documentation closeout `68c1a422`; full Linux CI
failed only the 29 source-provenance tests described above. Cumulative v18
repair `580e4a0e` then passed every hosted workflow and DeepSource. The four
`ai_uplift.py` findings were closed by `9f5efe15`, which passed every available
hosted workflow and DeepSource. The five `financial_sanity.py` findings were
then closed locally under full-report fingerprint tests as described above;
their hosted status must be read from the exact final checkpoint. GitHub still
reported zero open Dependabot and
secret-scanning alerts; code scanning remained unavailable with
`404: no analysis found` and is not a zero-alert result.
No model score, label, fill, cost, threshold, model outcome, trading authority,
or profitability claim changed. Verify every available hosted check on the
final pushed SHA; pending and unavailable checks are not passes.

This is an interruption-safe checkpoint, not product completion. Round 75 and
Round 21 remain active under their frozen contracts. The 97 modified, 2 deleted,
218 untracked, and 71 differing plus 26 locally-only path counts above are a
historical preservation snapshot against an older `main`; they must be
recomputed after terminalization because `main` has advanced. Do not reconcile
any path before Round 75 terminalization.

## Completed Foundation

This section describes pushed `main` only. It predates the unintegrated
Round 74/75 worktree above and must not be used to infer that newer local work
is published, tested on current `main`, or safe to discard.

- CLI and native Windows commands share one parser-derived contract.
- Deterministic ownership, reconciliation, Pause, Stop, loss, liquidity,
  latency, rate-limit, and stale-state gates remain outside AI control.
- Binance and Polymarket execution, capital, credentials, and order ownership
  are separate.
- Current model and AI research records preserve rejected, blocked, and
  non-authoritative outcomes instead of relabeling them as success.
- Round 29 now has a hash-bound, target-blind six-field settlement interaction
  overlay and preregistration. It repairs a design/implementation gap for the
  linear residual candidate without changing frozen Round 27/28 rows or
  pretending to reproduce Chainlink's unpublished TWAP method. Synthetic
  transform, zero-variance, composition, and tamper tests pass; no Stage 1 row,
  outcome, model metric, P&L, or authority was produced.
- Dependencies, imported agent workflows, release automation, and concise
  operator documentation are integrated into the main-line closeout gate.
- The 2026-08-16 DeepSource remediation addresses the complete 219-finding
  inventory across 25 Python modules without analyzer suppressions. Ruff,
  format, Vulture, terminology, provenance, badge, and manifest checks pass
  locally. The final full pytest run passed at 100%; source branch coverage is
  76.42%, and changed-line coverage is 95% (765 executable lines, 38 missing).
  Native Windows build, CLI parity, smoke, launcher, and layout checks also
  pass; the audited 2250x1470 render has no clipping or overlap.
- A complete Codex Security diff review found no reportable findings across all
  30 changed product-source files. Dependency audits of both supported GPU
  stacks found no known vulnerabilities. Zizmor auditor and pedantic scans pass
  without a long-lived coverage-service token. Coverage remains enforced in CI
  and published as a GitHub artifact; the unavailable, unprovisioned Codecov
  upload is not treated as a repository quality gate. Before this commit, live
  GitHub showed zero open Dependabot alerts, zero open secret-scanning alerts,
  and only `main`; code scanning was unavailable and remains unverified. Hosted
  CI, Ruff, Vulture, Super-Linter, and DeepSource are never inferred from local
  runs; this checkpoint is valid only if they pass on its pushed commit.
- Immutable Round 27 source-ledger and Round 28/29 preregistration hashes remain
  preserved. Canonical source-only remediation amendments bind every
  semantics-preserving analyzer refactor before any Stage 1 feature or outcome
  access. Round 28 correction v3 supersedes v2 only for those source bindings;
  it creates no data, model result, P&L, or authority. The generated publication
  manifest now lists Round 29 as the latest research round without claiming a
  result or authority.

### Latest Model Audit

A read-only audit on `2026-08-22` traced the unintegrated Round 74 v136
target-free pretraining objective through feature-view masking, target
construction, loss normalization, causal encoders, chronological purged
splits, segmented population weighting, paired random-state restoration, and
the focused failure-mode tests. The implementation matched its frozen
mask-aware contract: event targets come from untouched observations, masked
continuous dimensions are excluded from both target scoring and the loss
denominator, segmented rows are visited once without cycling, and validation
uses the same eligible-row weighting. No evidence-backed defect was found, no
training or profitability claim was made, and no source or capture worktree was
changed. Re-run the focused tests only after the capture is terminalized and
the differing files are safely reconciled onto `main`.

The same audit identified a post-capture, preregistration-only challenger: keep
v5 unchanged and compare a separate next-mark plus conditional-duration-density
objective on identical encoders, splits, seeds, populations, supervised
training, and after-cost evaluation. The duration target must come from the
untouched next event even when timing input is masked. Call it a conditional
marked-duration density, not a complete point-process likelihood, unless the
dataset exposes observation-end censoring and terminal survival. Promotion
requires paired proper-loss improvement without run/symbol/task subgroup
degradation and then downstream after-cost improvement. Do not implement or
evaluate it before Round 75 terminalization; it has no present edge claim.

## Next Work

1. Reverify both live captures and their canonical state files. Let Round 75
   reach its fixed terminal boundary and let the Round 21 sidecar reach its
   separately frozen boundary. Do not inspect targets or outcomes early.
2. Terminalize each capture only through its frozen contract. Run target-blind
   source, continuity, gap, role, population, lease, database/WAL, and resource
   audits before deciding whether any row is admissible. A failed gate closes
   that lineage; it does not permit salvage or threshold changes.
3. After Round 75 terminalization, preserve and re-hash the entire detached
   worktree before integration. Recompute its content comparison against the
   then-current `main`; the 2026-08-22 snapshot had 71 differing and 26
   locally-only files, while 220 status paths were already equivalent. Inspect
   every differing path and preserve the analyzer fixes already on `main`; do
   not bulk-copy or stage all dirty-status paths. Commit and develop only on
   `main` after that point.
4. Treat the unintegrated Round 74/75 documents and implementation as
   unverified until focused domain tests, contract hashes, generated CLI/native
   parity, and the affected analyzer gates pass on the integrated tree. Do not
   publish or infer their model claims from local file presence.
5. Continue security triage in bounded batches. The `B110` and `B112` queue is
   now zero. Trace remaining Bandit/DeepSource dynamic SQL source-to-sink paths.
   Distinguish fixed or validated identifiers and trusted executable contracts
   from actual injection or path-hijack exposure. Add direct regressions for
   every behavior change; do not suppress the backlog broadly. Continue from
   the exact DeepSource inventory on the final closeout SHA. The first
   type-contract batch reduced the headline Python count from 107 to 86 and the
   follow-up reduced it to 51. The nine changed-scope AI-uplift and
   financial-sanity complexity findings were later closed in two bounded,
   behavior-preserving batches. DeepSource then exposed the 13 smaller helper
   findings listed above under its stricter threshold. Exact revision `67dab5d8`
   closed four and confirmed the remaining nine. The subsequent local batch
   addresses all nine, and exact revision `bb6abac7` confirmed their closure.
   The seven-item type/test follow-up is closed at `13f33f8a`; continue from its
   exact 27-item DeepSource inventory recorded above. The market-data batch
   closed the coverage-complexity and unsafe-comparison family at exact
   follow-up revision `8d5ecc1e`. The rule-alpha scorer and two static-method
   findings are closed at follow-up `c349aca6`. The first SQL-safety checkpoint
   removed one of ten `market_store.py` dynamic-query heuristics at `63ecb4f2`;
   follow-up `646bbdbe` removes the other nine without suppression or index loss.
   The current Ridge/MLP materializer batch removes four more internal-identifier
   construction sites through explicit statement allowlists, and its bounded
   type follow-up closes the two modules' 22 scoped Mypy diagnostics. Continue
   from the exact `eef01801` DeepSource inventory above. The critical MLP lookup
   and unbound fit-claim row boundary are closed in `75160c3d`. Preserve the
   recorder finding until a cumulative source-ledger amendment can be reviewed.
   Backend, member, and ensemble validation complexity is closed in the final
   bounded closeout under canonical identity fingerprints and fail-closed tests.
   Report validation is also closed under exact admitted/no-trade identities and
   rejection tests. Fit-member training-loop complexity is closed under exact
   numeric and progress fingerprints. Top-level orchestration is also closed
   under exact full-report and callback fingerprints plus admission-gated test
   mechanics. The materializer is now closed under real DuckDB idempotence,
   transaction rollback, runtime-evidence repair, replay-drift, and tamper tests.
   Continue source-to-sink triage in other modules from exact hosted results,
   not assumed residual counts. Replace runtime
   assertions only when an explicit fail-closed error and cleanup contract
   exists; do not mechanically rewrite typing assertions.
   Do not treat the narrow `5f6e790c` pass or any bounded follow-up as closure
   of the broader backlog or as a project-wide security result.
6. Evaluate a model only after source, causal split, cost, delay, access-ledger,
   sample, and implementation bindings are complete. AI remains veto/downsize
   only until matched, latency-charged causal uplift is demonstrated.
7. Run focused checks while integrating, then the full local matrix once at the
   final checkpoint. On this Windows host invoke pytest through
   `uv run --locked python -m pytest`; the direct `uv run pytest` console entry
   does not place the repository's `tools` package on `sys.path`. Push `main`,
   verify all hosted workflows and available scanners on that exact SHA, and
   keep the GitHub branch inventory to `main`.

## Safety Invariants

- Conservative remains default; reinvestment remains off.
- Leverage is a ceiling, not edge. Any gate may reduce size to zero.
- Only provably bot-owned orders and parent-bound positions may be changed.
- Unknown market, account, fill, fee, order, or redemption state blocks new
  exposure. Reconnect recovery reconciles first and observes fresh books before
  deciding.
- AI never submits orders, raises risk, overrides a safety gate, or blocks a
  close.
- Canonical numeric evidence is JSON/CSV. Charts and prose are derived views.
  Never invent, repair by hand, or silently replace market data or results.

## Evidence Map

- Binance status: `docs/model-research/action-value/latest/README.md`
- Polymarket status: `docs/model-research/polymarket/latest/README.md`
- Round 29 preregistration:
  `docs/model-research/polymarket/round-029-settlement-state-matched-ablation-preregistration-v1.json`
- Model rules: `docs/MODEL_AND_SIGNAL_VALIDATION.md`
- Data cutoff: `docs/model-research/research-data-snapshot-contract-v1.json`
- Agent and CI workflow: `docs/AGENT_WORKFLOWS.md`
- Product direction: `PLANNING.md`
- Live local capture and integration boundary: this document's
  `Live Host Handoff: 2026-08-22` section

Suggested continuation request:

> Read `AGENTS.md`, `docs/CONTINUATION.md`, and `PLANNING.md`; verify the current
> repository, active processes, and GitHub state; then continue the ordered
> work without weakening frozen evidence or safety gates.
