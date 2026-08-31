# AGENTS

Read `docs/AGENT_START.md` first. Hash-bound evidence and executable contracts
override prose.

## Hard Rules

- Work in this session only; use no subagent.
- AI Git history must use `AI agent <>` for author and committer. Read
  `docs/AI_COMMIT_IDENTITY.md` before committing or auditing; never use a human,
  host, tool, CI, global-config, or noreply identity.
- Binance is BTC/ETH/SOL testnet, Demo, or paper only. Polymarket research is
  BTC/ETH/SOL; its BTC-only live-capable boundary defaults off. No live-money authority exists.
- Conservative is default. Leverage is a ceiling, never edge. Profitability,
  ROI, readiness, and drawdown claims require source-bound after-cost evidence.
- Aggregate performance never establishes an all-regime edge. Apply
  `docs/model-research/cross-regime-edge-acceptance-contract-v1.json`; an
  unsupported bullish, bearish, sideways, choppy, volatility, liquidity, or
  latency slice must reject promotion or abstain from new exposure.
- Risk, ownership, reconciliation, Stop, and close controls are deterministic.
  AI may only veto or reduce risk after matched uplift evidence and may never
  override a safety gate or block a close.
- Polymarket terminal state requires authenticated exact-order or fill evidence.
  Stop may cancel and sell only bot-owned hashes and
  parent-bound lots; foreign state is never modified.
- Future books, labels, resolutions, fills, and PnL never enter inference.
  Unknown order or redemption state blocks new exposure. Polymarket settlement
  never auto-deploys wallets or creates token approvals.
- A public fill after an oracle deadline does not prove its maker order was
  created after finality. Any finalized-winner latency study must bind the exact
  undisputed non-ignore on-chain state before hypothetical or owned order entry;
  a bid resting before that state is directional exposure, not structural carry.
- Never print, prompt, log, serialize, test, document, or commit credentials,
  secrets, tokens, signed requests, or unredacted secret fields.
- Do not label the common `0.00010000` USD-M funding value as a funding-rate
  cap. It is normally the standard eight-hour interest-rate plateau produced by
  the premium-index clamp; bind any actual cap or floor from the applicable
  current `fundingInfo` terms before using cap semantics.
- Preserve testnet, dry-run, diversification, liquidity gating, and the app's
  `20x` leverage cap unless a stricter frozen contract applies.
- The installed CLI and native app both invoke `simple_ai_trading.entrypoint`.
  Register independent command extensions there and keep
  `command_contract.py` on the same parser; do not create frontend-only
  Polymarket controls.
- No network calls in tests unless explicitly stubbed. Do not hard-code host
  capabilities; detect and record effective backends and fallbacks.
- Edge discovery is the current priority. Public, unauthenticated, read-only
  source, market-data, and blockchain research may proceed iteratively when
  each request tests a distinct question or materially refreshes stale evidence.
  Record request bounds and retain raw evidence, but do not require a commit,
  push, or hosted CI run before exploratory public requests. The stricter
  frozen one-use workflow remains mandatory for authenticated, account-specific,
  funded, order-capable, or state-changing operations.
- Before spending a historical trade or book request on an exact-payoff family,
  use immutable retained terminal outcomes to test settlement consistency when
  they can answer that question. Settlement consistency supports only the payoff
  identity; it never proves sub-floor acquisition, atomic execution, fees,
  capacity, owned fills, or profit.
- For an integer-score additive cover `Under A + Under B + Over G`, prove both
  the event and its complement before looking at price. The complete condition
  is `A+B-1 <= G <= A+B`; the lower bound alone is insufficient because a
  game-under state can leave both team-under legs losing. Enumerate the complete
  terminal state space and reject any one-sided implication before venue access.
- An issuer's at-par mint or redemption right proves only the payoff identity.
  Before any venue refresh, reuse retained finite-size spreads and reject unless
  the executable gap exceeds account fees plus transfer, redemption, delay,
  failure, compliance, custody, operating, tax, and alternative-yield costs.
  Never infer repository-account access, eligibility, timing, capacity, or
  completion from the legal right; event-triggered discounts require one frozen
  finite-size book, not venue polling.
- Before researching another stablecoin issuer one by one, construct or consult
  the complete retained same-unit parity frontier and source-bind only a row that
  can materially dominate the strongest admitted candidate after the frozen fee
  and operational hurdle. A lower gross diagnostic quote is not a reason to
  spend another issuer request; update the existing issuer-dislocation family
  instead of multiplying duplicate hypotheses.
- A documented comma-separated multi-market query can still exceed the venue's
  practical backend limit. If a frozen one-use request times out, preserve the
  exact response and journal and terminalize that population; never split,
  narrow, paginate, reorder, or alias it after observing the outcome.
- Invoke repository Python tools through the locked project environment, using
  `uv run --locked python -m tools.<module>`, so package imports and dependency
  versions match the checkout before any outcome-sensitive access.
- Raw non-browser requests to rendered Binance Academy and product-documentation
  pages have repeatedly returned HTTP 202 shells of about 2 KB with none of the
  visible economic or schema terms. Do not use such a dynamic page as the sole
  positive source in a one-use contract when the retained Agent Native index,
  a machine-readable primary schema, or a predeclared rendered field extraction
  can answer the question. If an exact dynamic page is nevertheless uniquely
  necessary and fails this way, retain the response and stop without URL aliases.
- Search the retained official Agent Native index and exact safe source sections
  before downloading a multi-megabyte complete reference. Secret-scan any full
  documentation response before staging because public examples can contain API
  keys and private-key blocks; if they do, preserve the response receipt hash,
  mechanically extract and hash-bind only the exact required secret-free section,
  and remove the unrelated full payload. Never commit public example secrets.
- Before any public research request, search retained request journals and source
  bindings for the exact canonical method and URL. Reuse a same-day successful
  immutable response unless the registered retry trigger requires a fresh mutable
  observation. A different output path or research label does not make the same
  request novel.
- Before treating a structural mechanism as novel, search both the prioritized
  hypotheses and terminal screens in the structural-edge registry, then `rg`
  mechanism aliases and payoff identities across retained artifacts and tools.
  An existing terminal family may advance only on its literal retry trigger;
  otherwise stop before building another collector. A zero-request retained-data
  audit may be kept only when it materially strengthens or corrects the existing
  adjudication, and it must update that family instead of creating a duplicate.
- Before freezing any request for an existing family, inspect its complete
  registry row: canonical artifacts, current status, next action, every
  prohibited shortcut, and the literal retry trigger. A compact rank/trigger
  projection is triage only and cannot authorize access.
- Family-specific tests must not pin the mutable whole-registry SHA-256 or the
  mutable global accepted-edge count. Verify the registry's embedded canonical
  self-hash plus the family's exact artifact, rank, scope, and terminal or retry
  state; keep exact global registry hashes and accepted counts only in central
  registry/frontier integrity and the newly accepted family's proof. This
  prevents every valid discovery from forcing unrelated test edits.
- Outcome-sensitive contracts must never bind a mutable current registry,
  frontier, manifest, or rolling artifact path as their only retained source.
  Snapshot the exact consumed bytes under an immutable source path before
  access. If an older contract already points at a mutable path, preserve the
  contract and outcome, recover the exact expected bytes into a hash-bound
  immutable sidecar, and verify that sidecar instead of rolling back current
  state or blindly updating the historical expected hash.
- A frozen current-state screen terminalizes only its exact population, time,
  size, and source contract. Material new primary evidence of recurrent
  event-time violations may reopen a separately preregistered recurrence study;
  it does not justify repeating an isolated snapshot or promoting the family.
- Historical mechanism-linked profit that mixes realized cash flows with
  imputed residual, merged, minted, or opportunity-valued inventory is a
  research lead, not after-cost profit. Separate the cash-realized core from
  mark-to-market inventory and bind every unwind, fee, delay, and capacity cost
  before using it in an edge claim.
- Accepted-edge count is a scope ledger, not a profitability score. Portfolio
  reviews must separate source-demonstrated recurring direct cash, same-principal
  yield or capital overlays, pre-existing-activity savings or rebates, and
  external-user or client revenue. Prioritize the first class and material
  improvements to its account, cost, capacity, and persistence evidence. Never
  present the other three as standalone market profit or spend requests and
  tests merely increasing their count.
- Call an edge stable and current only when the exact account is qualified, the
  mechanism is currently available, positive after every external cost and best
  feasible alternative, and its recurrence is demonstrated without an
  unresolved time-limited or material rate dependency. Historical payouts,
  public APRs, acceptance predicates, and fee schedules do not satisfy this
  standard by themselves.
- Before every venue HTTP request, classify the exact endpoint and method from
  a current primary API contract. Never infer that an endpoint is public from
  its path, product name, nearby public endpoints, or an unauthenticated error;
  if the security classification is absent or contradictory, do not call it.
- Documentation that a venue supports a feature does not prove that the live
  production configuration currently deploys it. For Binance Spot SOR, require
  the current public `exchangeInfo.sors` configuration before requesting books
  or considering a signed test order; an absent optional field is a terminal
  configuration result until a material official or live configuration change.
- HTTP `GET` does not prove read-only semantics. Classify account mutation from
  the documented operation as well as the verb; Binance
  `/sapi/v1/soft-staking/set` changes activation state and requires separate
  mutation authority. Never credit Soft Staking yield to pending-order frozen
  assets, Auto-Subscribe allocations, or inventory needed for prompt liquidity.
- Conflicting current primary-source terms are an unresolved gate. Do not pick
  a rate, fee, eligibility rule, or effective date by page location, apparent
  recency, or convenience; preserve both sources and require an explicit
  effective-date source or realized post-change evidence before promotion.
- A promotion-extension announcement does not override a conflicting current
  fee schedule that still publishes the expired end time. Credit zero current
  discount until an explicit correction, withdrawal, replacement, or updated
  schedule resolves the effective interval; never retry a consumed article
  endpoint merely to obtain a cleaner copy of the favorable terms.
- Search-result snippets, locale mirrors, and cached previews are discovery
  leads only. Open and retain the current primary page or endpoint before
  source-binding a rate, fee, eligibility rule, inventory, or effective date.
- A browser-renderable dynamic documentation route is not automatically byte-
  retainable through a direct HTTP client. Before freezing an outcome-sensitive
  documentation capture, use an already hash-bound native index or preflight
  that the exact route returns the required contract bytes. Preserve HTTP 202
  with an empty body as a consumed null response; do not retry an alias to
  manufacture the rendered text.
- Freeze text gates against the exact retained representation. For Markdown,
  either include formatting delimiters literally or preregister one mechanical
  normalization before access; raw substring checks do not ignore bold or
  inline-code markers. If a consumed response contains the required semantics
  but fails only on an uncontracted formatting difference, preserve the failed
  contract and journal, adjudicate the retained bytes offline, and never refetch
  merely to repair the phrase gate.
- A current official runtime registry that explicitly declares itself the
  single source of truth, together with a dated official deprecation or cutover
  notice, resolves an older repository deployment table for address selection.
  Preserve the stale source as superseded provenance; never let it keep a
  retired contract in an execution path or treat address resolution as proof of
  account access, cost, or profitability.
- A displayed `Max` or `Up to` APR is not a full-principal account rate. Bind
  the exact base and bonus tiers, caps, minimum, end time, account eligibility,
  and fees before promotion. A public calculator is only a sensitivity: if its
  horizon, tier, or campaign state is not source-exposed, do not reverse-engineer
  those terms from outputs or call the estimate realized economics.
- Treat Binance Yield Arena as a product-discovery surface, never as a distinct
  edge or one comparable APR. Before searching it, alias-check the exact article
  code, product, asset, and payoff identity against retained artifacts. Route
  each offer into its existing Simple Earn, staking, holding-promotion, or
  conditional-conversion family and bind that product's exact terms; Arena
  branding or a changed headline alone is not a retry trigger.
- Promotion dates stated for one named asset or product do not flow into adjacent
  rows in a current-offers table. Bind each asset's own effective start and end;
  without them its guaranteed forward public reward floor is zero, even when a
  current APR and tier cap are displayed.
- Before sampling variants of a reward endpoint, source-bind its filter and
  aggregation semantics and request the documented superset once. For
  Polymarket raw market rewards, `sponsored=true` folds sponsored daily rates
  into `rate_per_day`; a default response cannot close total reward funding,
  and Gamma spread/size metadata does not prove a current funded pool.
- An omitted parameter in a newer official client method does not prove that
  its default response is aggregation-equivalent to an older filtered response.
  Never compare populations across different parameterizations without an
  explicit current semantics contract; source-contract drift is terminal, not
  permission to discover the new population ceiling by adaptive pagination.
- Treat values surfaced by a rewards page or search result as discovery only;
  never require them to equal exact API values or use them in economics. Before
  requesting a book, reconcile condition ID, token identity, minimum size, and
  maximum spread between exact Gamma and exact sponsored-condition responses,
  and take the dated daily rate only from that exact reward response. A mismatch
  between those exact sources is terminal for the frozen candidate and does not
  authorize a retry, parameter change, replacement market, or favorable-value
  selection. A stale or misattributed discovery snippet is a methodology error,
  not evidence that agreeing exact sources conflict.
- Cross-token cost and reward comparisons must use a source-bound executable
  conversion into one exact unit. A one-for-one stablecoin assumption or a
  different quote currency is a labeled sensitivity only and cannot support an
  after-cost profitability claim. Public evidence that an automation contract
  captured a reward also does not prove independent access to its batching,
  relayer, permissions, fees, or first-block execution path.
- Treat venue fee promotions as exact pair-and-order-role overlays, never as a
  reason to change quote inventory or create volume. A zero maker fee may be
  credited only to an owned maker fill on the currently listed pair; it cannot
  be double-counted with a BNB discount, rebate, or reward. Any quote-asset
  switch first requires its own executable spread, basis, conversion, fill,
  settlement, and opportunity-cost proof.
- A fee-precision quantum is a bounded cost adjustment, not a standalone edge.
  Never manufacture, self-match, churn, split, or reroute volume to chase fee
  rounding. Bind the exact market minimum order, tick, fee-assessment
  aggregation, partial-fill behavior, and owned organic fills; absent those,
  the guaranteed saving is zero. Cap any credited rounding saving below one
  documented fee quantum per independently proven assessment.
- A contingent order list is not an atomic round trip. For Binance Spot OPO or
  OPOCO, the pending sell activates only after the working buy fully fills;
  partial fills remain unprotected. Bind received-quantity commission handling,
  trigger-time filters, locked and unlocked residuals, exact symbol capability,
  owned fills, and a precommitted sequential comparator before valuing its lower
  capital requirement or removed client request. Feature support alone proves
  no latency, fee, fill, price-improvement, or profit floor.
- A published market-data update interval is not measured arrival latency or
  profit. Compare like-for-like streams: Binance SBE Diff Depth is documented at
  20 ms versus JSON Diff Depth at 100 ms, but both SBE best bid/ask and JSON book
  ticker are documented real-time. Bind key type, decoder, source and receive
  clocks, continuity, reconnects, gaps, host path, decision timing, and owned or
  paper execution before crediting any freshness value.
- A FIX session disconnect, Logout, or heartbeat failure does not imply exchange
  cancellation of resting orders unless an exact retained contract says so.
  Treat Binance FIX `UNORDERED` and "better performance" language as qualitative
  candidates only. Bind non-live venue support, Ed25519 `FIX_API` permission,
  TLS/SNI, message-handling mode, in-flight count, clocks, acknowledgements,
  unknown timeout reconciliation, account-wide limits, mass-cancel scope, and an
  identical non-FIX comparator before crediting latency or stale-order value.
- Bind Polymarket fee schedules per exact retained market population. Crypto,
  Sports, and other event categories can have different taker rates and rebate
  fractions; never reuse a prior event's hard-coded schedule. If a consumed
  book request exposes schedule drift after raw retention, preserve the failed
  runner and adjudicate the immutable response once against a separately frozen
  exact schedule instead of refetching for a cleaner result.
- Polymarket Gamma `outcomePrices` can behave like midpoint diagnostics while
  the same market exposes a materially different `bestAsk`. Never let
  `outcomePrices` alone authorize a CLOB request. For a YES leg, require the
  retained side-specific `bestAsk`; for a NO leg, require a direct NO ask when
  available or use `1 - YES bestBid` only as a conservative rejection proxy.
  Missing side-specific ask evidence blocks escalation, and every surviving
  row still requires an exact current book batch before any economic claim.
- Treat soccer exact-score implications as one cross-family graph, not a reason
  to rebuild collectors market by market. The retained August 29 population
  already exhausts match result, first scorer, BTTS, full-total, team-total,
  halftime-result, and second-half-result identities. Reopen only on a distinct
  active event whose exact common rules preserve the payoff floor and whose
  side-specific rejection prices are strictly below it; stale event prices and
  `outcomePrices` never authorize a book.
- Treat soccer anytime-goalscorer/Over-0.5 and corner-count markets as part of
  that same anti-repeat boundary. The retained August 29 page already exhausts
  all 43 goalscorer-total packages and 1,820 full/half/team corner monotone,
  additive-partition, and parity packages. Reopen only under their literal
  rank-31 future-distinct-event triggers; do not rebuild or reprice them.
- Treat the 2026 GLW bStock/perpetual timing-gap episode as terminal. The exact
  Special debit matched the gross dividend but arrived 1.003 seconds after the
  snapshot, so no pre-snapshot gap existed. Never repeat, retry, paginate,
  alias, extend, repair, or book-capture that episode.
- A record-date weekend or holiday gap is not an ex-dividend timing gap. Bind
  the official exchange ex-dividend date using the applicable exchange calendar
  and settlement rule before valuing any bStock snapshot. Treat the current
  GOOGL episode as terminal: its holiday-adjusted ex-date equals the snapshot,
  its prior same-underlying gross-matching Special debit arrived only 7 ms after
  midnight, and the current public conservative net-distribution floor is zero.
  Never attempt a public-network millisecond snapshot/funding race. Rank 34 may
  reopen only under its literal future material pre-ex-date snapshot and
  positive source-bound net-distribution-floor trigger.
- Before freezing a time-bounded prospective capture, prove that its duration,
  phase alignment, retained observation tail, and required source timestamps can
  supply every minimum-sample gate. Elapsed duration and zero transport gaps do
  not prove analyzable intervals; record source-boundary continuity separately.
  If an unchanged mechanism already fails an economic gate, do not spend another
  capture merely to repair sample count unless a precommitted decision could
  still change.
- Before launching a one-use WebSocket capture, freeze reconnect behavior,
  maximum unobserved gaps, duration accounting, and terminal acceptance. A
  transport close that ends the runner before its frozen duration consumes the
  test and fails closed; never silently call the shorter file complete or rerun
  it under the same contract.
- In Windows/PowerShell capture wrappers, compare absolute instants with
  `DateTimeOffset` or normalize both operands to UTC before comparison. Never
  compare `DateTime.UtcNow` directly with a parsed local-kind `DateTime`; its
  tick comparison can wait past the frozen UTC boundary without issuing a
  request.
- An outcome-sensitive runner may enforce only gates mechanically enumerated in
  its frozen contract. Do not add an unstated stricter price, mark, balance, or
  data-quality equality in implementation; displayed rounded fields are not
  exact economic identities. If a consumed one-use run exposes such a mismatch,
  preserve its journal and raw responses, fail closed, and do not spend a repair
  capture when the retained evidence cannot change the economic decision.
- A candidate boolean must mechanically conjoin every frozen run-level gate as
  well as every row-level gate. If a consumed runner omits a run-level gate,
  preserve it and adjudicate the retained values separately; never rerun when
  every omitted gate actually passed. In wide screens, rank and display feasible
  rows only after latency, capacity, freshness, residual, fee, and stress gates;
  infeasible gross headline rows must not appear ahead of actionable candidates.
- Before an exhaustive graph or path extension, compute the rule-only topology
  cardinality and worst-case retained-output size before freezing. If either is
  large, freeze a lossless optimistic rejection bound and bounded diagnostics;
  retain every actual candidate and rejection identity required by the decision,
  but never allocate one summary object per provably rejected route-size row.
- Fixed-NegRisk multi-NO conversion value is additive after a base path. Optimize
  its exact best subset algebraically and use bounded meet-in-the-middle only when
  an exact profitable-path count is required; never enumerate all `2^n - 1`
  subsets. Preflight the supported outcome ceiling before requesting books.
- After a consumed runner fails, inventory every already-saved raw response for
  observations beyond the failed decision boundary before requesting anything
  again. An exact retained row may support a separately frozen, materially
  distinct public follow-up such as receipt reconciliation, but that follow-up
  may not repair, reinterpret, or complete the consumed contract's adjudication.
- A future-event terminal reconciliation must start strictly after the last
  retained source row and end only at the exact preregistered post-boundary
  allowance. Once the executable event window has closed, never request books
  merely to reconstruct a missed opportunity; terminal history is mechanism
  evidence only and a future episode requires a new prospective contract.
- An outcome-sensitive contract timestamp must be generated from the actual UTC
  clock before access, never typed as a rounded or anticipated time. The runner
  must reject a missing, offset-free, unparsable, or future `frozen_at_utc`
  before issuing its first request. If a retained run exposes timestamp metadata
  error, preserve the original hash-bound contract and capture and adjudicate it
  separately; never rewrite the consumed contract or rerun merely to repair
  metadata.
- Treat every conjunctive retry trigger literally: each event, time, authority,
  and data-state clause needs independent evidence before the triggered action.
  A passed date does not prove that a required market event occurred. If one
  public request is itself the preregistered event-discovery observation, label
  it that way, freeze its one-use consequences, and never reinterpret a null
  result as permission for an adaptive poll or conditional downstream capture.
- Before any time-gated request, reconcile the executable contract's exact
  not-before instant against the complete registry row and every current
  frontier or continuation instruction that names the request. A disagreement
  blocks access until the non-authoritative secondary artifact is corrected;
  never choose the earlier timestamp or average conflicting gates.
- A recurring-series title, recurrence label, recent update, or volume does not
  prove that future events are deployed. For a source-selected future series,
  either source-prove an actual event deployment before the outcome request or
  make one bounded event-discovery request its own terminal stage. An empty
  catalog may reopen only on its explicit deployment or nonoverlapping
  not-before trigger; never refine immediately from a short-horizon binary
  series to another unproved forward horizon.
- Before claiming complete coverage from a paginated public catalog, prove a
  source-bound population/page ceiling or freeze an explicitly partial rank or
  cursor boundary. A non-null cursor at the request ceiling is incomplete, not
  a zero-candidate universe, and does not justify a larger adaptive rerun.
- For a time-delta catalog, preflight worst-case arrivals from the densest
  retained tail against the venue's observed effective page cap, not the larger
  requested limit. Require capacity headroom to cross the frozen cutoff; if the
  bound cannot fit, do not spend the request until a documented server filter or
  explicitly partial decision can answer the question.
- For weighted or rate-limited APIs, freeze both the cumulative request weight
  and a limit-derived pacing schedule. After the first retained page, validate
  ordering, timestamp phase, cursor progression, and cross-source alignment on
  retained evidence before continuing pagination; successful JSON parsing alone
  is not an aggregation preflight.
- When searching retained minified JSON, JSONL, HTML, or other potentially
  single-line raw evidence, use filename-only matches, a bounded parser, or
  aggregate counts. Never let `rg -n` print an unbounded matching payload line
  to the console; raw retention is not permission to flood logs or prompts.
- Before printing parsed JSON, inspect only its top-level keys and collection
  counts first. Never serialize an unknown object or array to the console;
  select named scalar fields and explicitly bounded rows only. This applies to
  PowerShell `ConvertTo-Json` as well as search tools.
- For cross-contract funding histories, audit the actual returned row ceiling,
  ordinal settlement schedule, and timestamp skew before choosing a join. Do not
  discard valid leg-specific cash flows merely because independently published
  settlement timestamps differ by milliseconds; preserve the original failed
  method and adjudicate from retained raw responses without an adaptive refetch.
- Before acquiring price or book data for a funding-only dispersion candidate,
  compute the outcome-dominating perfect-foresight maximum-minus-minimum funding
  bound with zero execution cost and the full frozen capital hurdle. If that
  bound fails any preregistered role, terminalize the exact funding family and
  do not request prices; forecast-dependent price alpha is a separate family
  and cannot rescue a failed funding edge.
- Source-bind the documented semantics of every timestamp before using it as
  quote freshness or cross-source synchronization. A ticker transaction time,
  rolling-window close time, event time, or HTTP receipt is not automatically a
  best-bid/ask update time. Preserve a consumed test that used the wrong field,
  then freeze any retained-data correction separately; use HTTP request bounds
  only as observation-window provenance and never as proof of the book's last
  internal update.
- Before a one-use CLOB batch screen, preflight the actual book-array ordering
  semantics or freeze acceptance of either strictly monotone direction. Do not
  assume the documented bid or ask sort order; if a retained complete batch is
  consistently reversed, preserve the consumed runner and adjudicate offline
  without refetching or allowing the outcome-aware correction to promote it.
- Every current multi-book candidate gate must mechanically conjoin both a
  source-bound oldest-book-age ceiling and a cross-book timestamp-skew ceiling.
  Synchronization alone does not make uniformly stale books current; a displayed
  metadata price never substitutes for executable depth.
- When one complete catalog produces multiple outcome-sensitive candidates,
  freeze one deterministic candidate ordering before the first depth request.
  Failure of that selected candidate consumes the catalog escalation: do not
  cherry-pick runners-up from the already-observed catalog without a literal new
  event, source, recurrence, or material economics trigger.
- Before every one-use HTTP request, durably journal the method, URL, and exact
  request-body hash before access. Catch HTTP error responses as evidence: save
  their body, status, and completed receipt before raising. If the provider's
  bounded-history limits are not source-proved, freeze a conservative range;
  never learn the limit by adaptively retrying a consumed outcome query.
- Before any one-use file-backed request, create and verify every raw-output and
  journal parent directory before network access; the runner must independently
  fail before HTTP if a destination parent is missing or unwritable. A curl
  write error after response bytes arrive consumes the request and may not be
  repaired by retrying the same source under a new filename or endpoint alias.
- Do not consume a one-use request inside an ephemeral orchestration callback
  that retains the response only in process memory. Preflight the complete
  byte-to-durable-file path on synthetic UTF-8 bytes, prejournal the request,
  and make the HTTP client itself atomically persist the raw body and receipt;
  an unavailable decoder or callback failure after access is a consumed run.
- Treat a documented keyset `limit` as a ceiling, not proof that the service
  will return that many rows. A returned cursor makes the frozen population
  incomplete. Never follow it or depth-test a partial-page winner unless the
  cursor budget was fixed independently of outcomes before the first request;
  prefer a narrower distinct window that is prospectively complete.
- An exact duplicate-payoff identity is a mechanism, not recurring positive
  economics. Before spending a source-continuous capture or authenticated paper
  probe on a maker-first lead, require one distinct resolved-leg instance to
  remain positive after its frozen hedge stress. If the independent instance is
  negative, stop the family until a material price, fee, rule, or market change.
- For an exact sports monotone-payoff package, use retained Gamma
  `outcomePrices` only as a rejection-only optimistic gate after proving market
  identity and resolution semantics. If every package's displayed price sum is
  at or above its guaranteed payout floor, stop before requesting CLOB books.
  Gamma prices may never accept, promote, or prove executable profitability;
  only a package below the floor may advance to one frozen exact-depth screen.
- Before calling a sports payoff-lattice screen complete, enumerate exact
  deterministic subset relations across compatible time scopes as well as
  within each market family. Cumulative statistics such as first-period and
  full-game totals can create cross-period implications; preserve a narrower
  consumed adjudication that missed one, then freeze the retained-data
  correction separately before any depth access.
- In NFL margin lattices, a half-half-tie moneyline and favorite minus 0.5 can
  share the same integer win threshold while differing in the actual-tie state.
  Order them by exact statewise payout dominance; do not reject the event merely
  as a duplicate threshold, and do not adaptively request depth after an offline
  outcome-aware correction.
- A rejection-only catalog result must retain every tested relation's exact
  identity, price sum, payout floor, and decision, not only candidate rows and
  aggregate counts. If a consumed runner omits rejected rows, preserve it and
  reconstruct the complete retained page offline without refetching before
  making a coverage or best-row claim.
- Store large retained catalog artifacts as deterministic compact JSON (or a
  partitioned row format) once canonical content is fixed. Do not spend Git,
  review, or CI resources on whitespace expansion of thousands of evidence
  rows, and never drop required relations merely to reduce file size.
- PowerShell preflight guards must parenthesize each `Test-Path` operand and set
  terminating error behavior before a guarded one-use command. The runner must
  independently enforce the same one-use boundary, so a shell parser or
  non-terminating-error mistake cannot consume a duplicate request.

- Binance Spot's Price Range Execution Rule is an execution-safety constraint,
  not a standalone edge: an otherwise identical marketable `LIMIT IOC` or
  `LIMIT FOK` order can impose the same or a tighter user-selected worst price.
  Do not poll `executionRules` or `referencePrice` merely to credit avoided
  slippage against an avoidably unbounded `MARKET` order. Source-bind the exact
  current rule and model `EXECUTION_RULE_PRICE_RANGE_EXCEEDED` residual quantity
  only when a separately frozen candidate materially depends on unbounded taker
  execution or exact residual-expiry behavior.
- In an exact Polymarket crypto threshold ladder whose common rules resolve YES
  only when the same source value is strictly above its threshold, `YES(L)`
  plus `NO(H)` for `L < H` has a one-pUSD optimistic rule-consistent floor and
  pays two pUSD only between the thresholds. Independent condition disputes,
  cancellations, or inconsistent resolutions are additional downside, not part
  of that floor. Screen every lower-higher pair from one exact complete event;
  Gamma is rejection-only. Do not request books unless at least one displayed
  sum is strictly below one pUSD, and never cherry-pick a sibling ladder after
  a consumed complete event fails that gate.
- Do not equate a strict-above threshold indicator with cumulative range bins
  when exact boundary values resolve to the higher bin. At `x = T`, threshold
  YES is zero while the cumulative upper-range indicator is one. The exact
  equality gate therefore fails; only `NO(T)` plus all upper-range YES claims
  has an optimistic common-rule one-pUSD floor. Screen that weaker cross-event
  coverage identity only on a distinct nonconsumed same-source, same-instant
  pair with complete boundary rules and contemporaneous frozen populations.
  Exhaust both valid directions at every shared boundary: threshold NO plus
  cumulative upper-range YES, and threshold YES plus lower bins through the
  starting bin. Gamma is rejection-only; if exact depth is already negative at
  zero fee, stop before fee endpoints and never refetch stale or skewed books.
- When one current retained response exposes multiple simultaneously deployed
  BTC, ETH, or SOL sibling pairs for the same payoff family, screen every pair,
  shared boundary, and valid direction before selecting one global best row.
  Precommit deterministic tie-breaking; never book-test one asset while hiding
  an equal or worse sibling, and stop when the global best merely equals its
  optimistic payout floor before fees.
- Before spending another market request, enumerate every distinct exact-payoff
  family that one retained complete population can test. Freeze each family
  separately before examining its economic rows, reuse only the hash-bound raw
  bytes, and preserve independent terminal outcomes. Reuse is an efficiency
  gain, not permission to combine hypotheses, adapt gates, or promote one
  family's displayed prices as another family's executable depth.
- For aligned binary interval markets that use one source-continuous opening
  and closing value, three adjacent `Up` intervals imply the covering interval
  is `Up`, and three adjacent `Down` intervals imply the covering interval is
  `Down`. This authorizes rejection-only transitive packages only after exact
  partitioning, source/value continuity, and equality semantics are proved;
  matching titles or nominal horizons alone prove nothing. Screen every scoped
  asset and both directions, then apply one deterministic global gate.
- Gamma may encode `outcomes`, `outcomePrices`, and `clobTokenIds` as JSON
  strings inside otherwise parsed event objects. A retained-input preflight
  must assert the exact representation and exercise the production field
  decoder before an implementation-hash-bound contract is frozen. Preserve a
  pre-economic local failure and freeze a new contract; never rewrite it away.
- Before refreshing books for a Polymarket maker-reward overlay, reconstruct the
  exact minimum-size one-leg orphan loss from retained evidence and reconcile
  each exact condition with `sponsored=true`. Gamma reward minimum and spread
  fields are eligibility metadata, not funding proof. If even an impossible
  100 percent share of every remaining exact pool does not strictly exceed the
  maximum orphan loss, or an exact sponsored population is empty, stop without
  refreshing books, accessing an account, or repeating the condition.

## Working Method

Use the pinned Karpathy baseline from
`multica-ai/andrej-karpathy-skills@2c606141936f1eeef17fa3043a72095b4765b9c2`:
think first, state material uncertainty, keep changes small, preserve contracts,
and verify reproducibly. Do not load upstream `EXAMPLES.md`.

1. Inspect `git status`.
2. Read the nearest source, matching test, and relevant local skill. Use the
   artifact routed by `docs/AGENT_START.md` only when needed.
3. Use exact `rg` first. For broad semantic routing, use the external
   `cocoindex-code-search` workflow with at most five results, then confirm each
   candidate in live source. Never build its index during high system load.
4. Freeze causal inputs, costs, roles, rejection gates, and test access before
   viewing a new model outcome.
   Before a multi-request public screen, validate parsing and aggregation on one
   retained response. Use non-alias helper names and property-bearing objects;
   retain raw responses before downstream calculations so a local calculation
   failure reuses evidence instead of refetching it. For a large discovery or
   inventory response, persist it and print only a bounded aggregate in the same
   request; never stream the full payload to the console as the only copy.
5. Keep edits scoped, match existing patterns, remove only resulting orphans,
   and never revert unrelated work.
6. Keep numeric evidence in canonical JSON/CSV and regenerate charts from it.
   Generated charts and prose are not result authority.
7. Format code before generating implementation-hash-bound evidence; later
   source edits require regeneration.
8. Before freezing a one-use runner, execute its import-only or `--help`
   preflight through the same locked entry mode used for capture. In this
   repository, prefer `uv run python` so the `uv.lock` dependencies are present;
   never substitute the Windows Store `python` alias or an unverified global
   interpreter. Verify every required transport import before freezing the
   contract. For a retained-input runner, also parse every hash-bound raw input
   through the exact production loader and exercise the full pre-network
   contract validator before freezing; `--help` proves only imports and CLI
   wiring, not encoding or retained-schema compatibility. A runner that imports
   `tools.*` must be invoked as
   `uv run python -m tools.<module>` from the repository root; do not discover a
   runtime, dependency, or module-path difference after freezing evidence. If a
   pre-main import still fails, journal it as a local unconsumed preflight error
   before any corrected invocation.
9. Before adding a runner at a reusable-looking path, check path ownership with
   `git ls-files --error-unmatch <path>`. Never replace a tracked generic runner
   with a one-event implementation: pass it a new frozen contract or use a
   separately named wrapper. If replacement is discovered only after a request,
   do not rerun or rewrite the consumed contract; preserve the exact consumed
   implementation in an immutable hash-bound sidecar, restore the reusable path,
   and record both bindings in the terminal artifact.
10. Discovery and search surfaces may select an exact current primary source
    but never supply its economic values. If the one-use retained primary bytes
    disagree in currency, rate, scope, or formatting, preserve the failed
    contract and consumed response, exclude every discovery value, and perform
    at most one zero-network adjudication of the retained bytes. Never refetch,
    alias, or loosen the consumed contract to recover the expected answer.

Do not broadly read the README, historical round designs, generated SVG, or
large CSV files. The detailed workflow and imported-tool provenance are in
`docs/AGENT_WORKFLOWS.md`; broad architecture starts with
`docs/SIMILAR_TRADING_REPOS_REVIEW.md`.

## Verification

- Invoke repository tests as `uv run python -m pytest`; the `uv run pytest`
  console entry may omit the repository root and cause false collection errors
  for tests that import `tools.*`.
- During iteration, run the smallest focused test and Ruff check. Every new
  branch needs a direct assertion, including normal and fallback error paths.
- At a behavior checkpoint, run the complete affected-domain suite once.
- Run full pytest and coverage only for shared core, release preparation, or a
  significant final handoff; do not repeat them after each edit.
- CLI changes require parser/handler coverage and generated native-contract
  parity. Model/backtest changes require contract, causal-split, economic-gate,
  persistence, and tamper tests for that domain.
- Run `tools/update_readme_badges.py --check` after badge changes. The README
  badge block is generated and must not be hand-edited.

Completion requires implemented behavior, focused tests, relevant live or
artifact validation, synchronized CLI/Windows metadata, and truthful blockers.
