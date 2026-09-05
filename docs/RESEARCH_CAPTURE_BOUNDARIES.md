# Research Capture Boundaries

Mandatory companion to `../AGENTS.md`. Read this file completely before source
selection, market-data access, or any research capture. These rules are moved
intact from the root instructions; moving them does not reopen any consumed
study or change a protected boundary, exact retry trigger or trading authority.
Hash-bound executable contracts and evidence retain precedence over prose.
Paths in the preserved rules below are repository-root-relative unless stated
otherwise. The original text is preserved from commit
`fcc542285e439f20da82b1d6f484831c60d6309f`, with line endings normalized only.

- Before freezing or fetching a source selected from any retained index,
  deduplicate its exact canonical URL against retained contracts, source results,
  and request journals. If that URL was already consumed and its exact retry
  trigger is not independently satisfied, reuse the hash-bound bytes offline;
  do not refetch or alias it. Family-level novelty does not override an exact-
  URL one-use boundary. Correct the registry lineage when a novelty inventory
  missed this reuse instead of spending another request.
- When a public source URL does not prove its payload format before access,
  freeze the admitted format or treat content-format discovery as a source gate.
  Always retain raw bytes and the request journal before decoding. A text-only
  runner must record non-UTF-8 or binary input as a clean fail-closed format
  result instead of raising after access. Hash-bound PDF or other binary bytes
  may be inspected offline to strengthen rejection, but they cannot repair or
  promote a text contract, and no format alias or retry is allowed.
- Predict.fun is a separate BNB Chain prediction venue, not Binance Exchange.
  Binance Exchange or testnet credentials never authorize Predict.fun API,
  wallet, account, order, reward, or eligibility access. A logged-out Predict.fun
  page may expose a public book and a Polymarket comparator, but a comparator
  probability is not an executable ask, a missing ask is not free, and rendered
  discovery is rejection-only unless exact rules, side-specific depth, fees, and
  timestamps are durably source-bound under a prospective contract.
- Current Predict.fun maker rebates are only an incremental overlay on an
  independently justified legitimate organic eligible maker fill: credit the
  exact asset actually distributed after every incremental cost, never infer a
  profitable market-making strategy from the advertised fraction, and never
  create volume, self-trade, wash-trade, manipulate balances, abuse cancellation,
  or post non-executable orders to farm rebates or Predict Points. Predict Points
  have zero admitted monetary value without a deterministic official redemption
  contract. Matching crypto title and interval are not payoff identity: current
  Predict BTC/USDT point-report Up/Down/Flat rules differ from Polymarket BTC/USD
  60-second-TWAP equality-to-Up rules and cannot form a deterministic hedge.
- Predict.fun public-chain research must start from an exact official deployed
  contract or explorer reference. Never guess deployments from selectors,
  bytecode similarity, admin neighbors, audit examples, or testnet addresses.
  Protocol collateral yield is not trader income when the current source assigns
  claims to yield managers; require explicit trader beneficiary, rate, base,
  distribution, and redemption terms before crediting any value.
- Before spending a historical trade or book request on an exact-payoff family,
  use immutable retained terminal outcomes to test settlement consistency when
  they can answer that question. Settlement consistency supports only the payoff
  identity; it never proves sub-floor acquisition, atomic execution, fees,
  capacity, owned fills, or profit.
- For a cross-venue funding candidate, freeze the population, chronological
  training/validation/test roles, timestamp normalization, orientation rule,
  execution cost, two-leg capital hurdle, and quote-unit stress before viewing
  funding values. Select orientation on training only and require every role to
  remain positive after all frozen costs plus within-role stability gates.
  The frozen capture contract must embed those exact adjudication choices or
  bind a separately frozen adjudication contract; a schema check before that
  freeze may print field names and row counts only, never rate, price, first-row,
  last-row, minimum, maximum, or other economic values. If economic values leak
  first, mark that population promotion-ineligible and require a disjoint
  prospectively frozen confirmation rather than pretending the freeze was clean.
  Request no premiums, books, accounts, credentials, or orders when that cheap
  funding-only prefilter fails; never rescue it by resampling or changing the
  orientation after observation.
- Treat a timezone-naive economic timestamp as ambiguous even when its wall-
  clock text looks familiar. Stop before economic output and preserve the
  failure. A sole mechanical correction may be refrozen only when a current
  primary source independently anchors the venue schedule and timezone; bind
  that source, require one exact timestamp grammar, and leave population,
  alignment, roles, orientation, costs, gates, and raw inputs byte-identical.
  Otherwise enumerate every plausible alignment conservatively or reject the
  population; never inherit the workstation timezone or silently append UTC.
- A new listing on only one venue does not by itself reopen a cross-venue
  funding family. First freeze one current counterpart-instrument inventory and
  require an exact common underlying, share class, contract unit, quote unit,
  and settlement mapping. If no exact counterpart exists, stop before every
  funding value, ticker, premium, price, book, fee, or account request; a related
  prediction market, company name, index, wrapper, or merely similar ticker is
  not a hedge match.
- For an integer-score additive cover `Under A + Under B + Over G`, enumerate
  the zero-payout state before looking at price. It exists exactly when both
  component scores reach their thresholds while their sum stays below `G`, so
  the complete one-pUSD-floor condition is `G <= A+B`; no lower bound on `G` is
  required. Generalize to `G <= sum(component thresholds)` for more components,
  and use side-specific `bestAsk` or conservative `1 - bestBid`, never midpoint-
  like `outcomePrices`, for the rejection gate.
- For any scalar threshold ladder, prove that every leg uses the same scalar,
  source, observation instant or interval, precision, boundary convention, and
  exceptional fallback before looking at price. If those rules are identical,
  `YES(X > L) + NO(X > H)` has a one-pUSD floor for every `L < H`; enumerate
  every ordered pair, not only adjacent thresholds. Use the rendered `Buy Yes`
  and `Buy No` buttons only as a discovery rejection gate, then side-specific
  `bestAsk` or conservative `1 - bestBid` for a frozen advance. A card
  probability, search snippet, stale crawl, midpoint, or blank button is not an
  ask. Multiple deadline buttons on one rendered event card can be separate
  market probabilities rather than labeled acquisition sides; never combine
  them as `NO(earlier) + YES(later)` economics. If that ambiguity alone creates
  an apparent strict sub-floor package, freeze at most one exact primary
  metadata reconciliation and treat its prices as rejection-only. Stop before
  Gamma or CLOB when the cheapest unambiguous discovery package is already at
  or above one pUSD, and charge time value through resolution before calling
  any strict sub-floor package an edge.
- When scalar-threshold rules accrue only after each market's creation,
  `YES(lower) + NO(higher)` additionally requires the lower-threshold market to
  start no later than the higher-threshold market. Compare exact instants,
  including subsecond precision. If the lower market starts later, the higher
  threshold can trigger in the gap and leave both legs at zero; exclude that
  pair before economics unless the gap is source-proved event-free.
- For any proposed two-market Boolean cover, enumerate all four joint truth
  states before looking at price. If `A => B`, `NO(A) + YES(B)` has a one-pUSD
  floor. If `A` and `B` are mutually exclusive, `NO(A) + NO(B)` has a one-pUSD
  floor. If they are collectively exhaustive, `YES(A) + YES(B)` has a one-pUSD
  floor. Calling two markets complements requires both mutual exclusion and
  collective exhaustion. Different end times, emergency-action windows,
  observation sources, boundary rules, or exceptional fallbacks can leave a
  joint state uncovered; write that state down explicitly instead of pricing it
  as impossible. Apply the same rendered-button rejection gate and frozen
  side-specific advance rules as for a scalar ladder.
- For an exact union identity `B = OR(A1...Ak)`, each
  `NO(Ai) + YES(B)` implication package has a one-pUSD floor, while the complete
  replication `YES(A1) + ... + YES(Ak) + NO(B)` has a one-pUSD floor only when
  every union member is present and the rules prove both directions. A missing
  member leaves a zero-payout state and may never be treated as a free leg. For
  rejection only, sum the known direct YES asks plus conservative `1-bestBid`
  for composite NO before requesting a missing member: if that nonnegative-leg
  lower bound is already at or above one pUSD, stop without the extra metadata,
  book, fee, account, or credential request.
- In one rule-complete fixed-NegRisk event, every unordered pair of distinct
  outcomes is mutually exclusive, so `NO(Ai) + NO(Aj)` has a one-pUSD floor and
  pays two pUSD when another outcome wins. An all-YES complete-set screen does
  not exhaust this pairwise-NO family. Freeze and enumerate every pair before
  economics, use a direct NO ask or conservative `1-bestBid`, and treat a
  missing selected-side price as incomplete rather than free. Stop before books
  when no pair is strictly sub-floor. For retained Gamma fee metadata, an
  absent `feeSchedule` is zero fee only when the same frozen row explicitly has
  `feesEnabled=false` and an empty `feeType`; every enabled row still requires
  a supported exact positive schedule. Preserve a consumed pre-output schema
  failure and refreeze that sole mechanical correction instead of rerunning or
  rewriting it.
- A pairwise-NO screen does not exhaust larger mutually exclusive NO packages.
  For any `k >= 2` distinct outcomes in one complete fixed-NegRisk event,
  buying all `k` NO tokens has a `(k - 1)`-pUSD floor. At fixed common quantity,
  sort price-complete NO legs once and evaluate the cheapest prefix for every
  cardinality; for fee-and-tick stress, sort by each leg's additive stressed
  unit cost. This exactly tests whether any known-price subset can pass without
  enumerating billions of dominated combinations. Missing legs remain
  incomplete, and retained post-price discovery is hypothesis generation only,
  never prospective promotion evidence.
- For a complete mutually exclusive fixed-NegRisk event, the primitive
  direction-independent long-only package basis is: the all-YES complete set,
  every same-market `YES + NO` binary straddle, and every optimal `k`-NO
  cardinality frontier. Adding a positive-cost YES leg to a proper NO subset
  without covering every selected NO cannot raise its minimum payout. Freeze
  this complete basis before accessing a prospectively selected recurring
  event; require a strict side-specific after-fee-and-one-tick source gate
  before any book request, even when the gross near miss is only one tick.
- When projecting a pairwise comparator into a multi-outcome winner market,
  align tie payouts before calling the implication Boolean. If the pairwise
  market resolves an equal comparison 50-50 while the multi-outcome market
  awards one full winner by a separate tie-break, the naive
  `NO(A wins) + YES(A beats B)` package has only a 0.5-pUSD floor when `A` and
  `B` tie and `A` wins the multi-outcome tie-break. Price against that smaller
  floor or add exact tie coverage; never assume a one-pUSD floor. If one known
  required leg already costs at least the tie-aware floor, stop before metadata,
  book, fee, account, or credential requests.
- For ordinal rank mutual exclusion, both events must rank the same entities
  with the same observation view, source instant, fallback, ambiguity handling,
  and deterministic tie-break. A best-model view and a lab-rank view are
  different payoff functions even when both name companies. `Other` can denote
  different entities at different ranks, identity-free placeholders are not
  cross-event identities, and spelling variants are not aliases without source
  proof. Only an exact named identity can support `NO(rank i) + NO(rank j)` for
  distinct positions; freeze every named package before economics and require
  exact fees plus one adverse tick per leg before any book request.
- For a projection cover from one complete joint partition to `k` mutually
  exclusive exact marginal outcomes, buy `NO` on every marginal outcome and
  `YES` on every joint row whose projection is their union. The package has a
  `k`-pUSD floor only after every projected joint row, hidden sibling, boundary,
  and fallback is enumerated and both events use the same underlying scalar and
  resolution rules. Use conservative `1 - bestBid` for marginal NO legs and
  direct `bestAsk` for joint YES legs. Apply the exact fee curve and at least one
  adverse tick per leg before requesting books; a nonpositive stressed floor
  terminalizes the exact family without depth access.
- A rendered multi-outcome count is not proof that its visible cards exhaust an
  exact NegRisk event. If the rules name an unrendered fallback or Gamma can
  retain hidden or inactive siblings, a strict visible all-YES subfloor may
  authorize at most one prospectively frozen exact metadata reconciliation.
  Enumerate every returned market, including inactive fallbacks, and require the
  complete event set to be active, acquisition-capable, mutually exclusive, and
  exhaustive before any book request. Never omit an expensive hidden sibling or
  treat an inactive market, blank button, missing ask, or rendered FAQ count as
  a zero-cost leg.
- A rendered two-party election-winner card is subject to the same rule. Gamma
  can retain an inactive `Other` outcome and inactive candidate or placeholder
  siblings even when only Democratic and Republican cards render and their YES
  asks sum below one pUSD. Reconcile the complete exact event once; never infer
  exhaustiveness from the two-party prose, rendered FAQ count, or active-only
  market set, and never omit a returned inactive sibling from a complete-set
  acquisition package unless its zero-payout status is source-proved.
- A rendered FAQ outcome count can describe only the active or displayed rows,
  not the complete Gamma event population. Generated candidate events can retain
  dozens of inactive `Person X` placeholders, named inactive candidates, a
  closed active candidate, and `Other` even when only a small active set renders.
  Reject before books unless every returned market is open, active,
  acquisition-capable, and rule-exhaustive; never party-map, classify, or omit
  identity-free placeholders to manufacture a projection or complete-set floor.
- A nominally shorter calendar window is not a subset when it begins before the
  longer window. Source-prove that every already elapsed exclusive segment did
  not trigger before upgrading the remaining short-window event to an
  implication. Do not spend a historical price-source request on that proof
  when the package already costs at least its floor under the optimistic subset
  assumption. An unproved optimistic implication may reject research spend but
  may never advance a candidate, authorize depth, or support an edge claim.
- A cumulative-deadline market and an exact-date partition are not projections
  of the same event when their creation instants differ. The earlier market's
  exclusive prehistory can make cumulative YES true while every post-creation
  exact-date row pays zero. Source-prove that no qualifying event occurred in
  that gap before assigning a floor. For rejection only, price the optimistic
  package with every exact partition leg plus the cumulative complement; never
  subtract a stale headline `No release` probability from an all-YES sum or mix
  rendered snapshots. Charge one adverse tick and the exact fee on every leg.
- For two cumulative deadlines whose rules begin at market creation,
  `NO(earlier) + YES(later)` has a one-pUSD floor only when the later-deadline
  market starts no later than the earlier-deadline market and every other rule
  is identical. Compare exact timestamps, not calendar labels. A later-market
  creation gap permits earlier YES with later NO unless a source proves no
  qualifying event occurred in the gap; reject that pair before economics.
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
- For canonical top-level JSON self-hashes, use
  `uv run --locked python -m tools.hash_json_without_field <path> --field <name>`
  instead of an ad hoc interpreter snippet or a test failure as a hash oracle.
- Keep broad `rg` discovery out of retained raw single-line JSON payloads unless
  the payload itself is the explicit target. Search source and artifact metadata
  first, then parse the exact retained file with a bounded field projection; a
  multi-megabyte one-line match wastes output without improving coverage.
- On Windows, invoke focused tests as `uv run --locked python -m pytest ...`.
  The `uv run --locked pytest ...` console-script form can omit the repository
  root from `sys.path` here and produce a false `ModuleNotFoundError: tools`
  during collection; do not debug or modify imports until the module form has
  been tried. The exact publication reconstruction selector is
  `tests/test_polymarket_publication.py::test_current_status_manifest_reconstructs_every_artifact`;
  do not guess a filename from the test name.
- Raw non-browser requests to rendered Binance Academy and product-documentation
  pages have repeatedly returned HTTP 202 shells of about 2 KB with none of the
  visible economic or schema terms. Do not use such a dynamic page as the sole
  positive source in a one-use contract when the retained Agent Native index,
  a machine-readable primary schema, or a predeclared rendered field extraction
  can answer the question. If an exact dynamic page is nevertheless uniquely
  necessary and fails this way, retain the response and stop without URL aliases.
- Treat every Binance CMS announcement catalog as a category-specific discovery
  page, not a complete or strictly current cross-category feed. A page may omit a
  newer Futures, listing, promotion, or product announcement while still returning
  HTTP 200. Freeze one bounded page only when its exact catalog is relevant, label
  a no-new-row result to that page and cutoff only, and never infer that Binance as
  a whole has no new trigger. A discovered title or article code is not economics:
  freeze the exact detail article and any announcement-designated FAQ before using
  eligibility, tier, fee, duration, or product terms.
- Testnet credentials never prove mainnet promotion, VIP, portfolio, copier, AUM,
  commission, or account eligibility. Keep them out of public program gates and
  require the exact designated mainnet read-only authority before any owned-state
  claim; every application, agreement, evidence disclosure, trade, transfer, or
  account change remains separately authorized.
- Search the retained official Agent Native index and exact safe source sections
  before downloading a multi-megabyte complete reference. Secret-scan any full
  documentation response before staging because public examples can contain API
  keys and private-key blocks; if they do, preserve the response receipt hash,
  mechanically extract and hash-bind only the exact required secret-free section,
  and remove the unrelated full payload. Never commit public example secrets.
  A PEM/assigned-key regex alone is insufficient: include provider-specific
  credential patterns before staging (including AWS access-key IDs embedded in
  site bundles). Never bypass push protection to publish documentation. If
  extraction removes original bytes, preserve their receipt hashes and an
  explicit section-disposition record; do not claim full-body reconstruction.
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
- Before browsing, downloading, or source-validating a paper, search its exact
  title, DOI, arXiv ID, SSRN ID, author-title pair, and mechanism aliases across
  the registry, action-value artifacts, and `docs/CONTINUATION.md`. A retained
  source-bound hit is not a new literature trigger; reuse its adjudication and
  advance only on that family's literal retry trigger.
- Direct SSRN `Delivery.cfm` PDF requests can return a Cloudflare HTTP 403 HTML
  challenge even when the public abstract is indexed. Freeze the exact primary
  PDF once, retain the failure, and stop without retrying query variants or
  mirrors. An indexed abstract remains discovery only: never use its Sharpe,
  return, drawdown, fee, fill, venue, or direction-neutral claims to authorize
  a collector, book request, or order experiment without byte-retained complete
  methodology and reproducible code or data.
- For XML or other structured public sources, freeze semantic parser gates for
  namespace-expanded element names and exact required fields instead of a
  literal serialized root tag whose whitespace or attribute layout can vary.
  Preserve a consumed representation-only failure, adjudicate its retained
  bytes offline, and never refetch merely to repair serialization formatting.
- Do not run broad text searches over raw or canonical one-line JSON payloads.
  List candidate files first, exclude `data/`, every path segment named `raw`,
  every `*.raw` payload, and every `*.json` file from prose/code alias searches
  by default. Parse only the exact JSON keys needed for retained-data audits.
  This prevents multi-megabyte
  console dumps from wasting context and resources. When JSON artifacts are
  relevant, list their names first and use a structured parser on only the
  selected files and predeclared fields; never send a matching one-line payload
  to stdout. A narrow exact-filename search is allowed only when its output is
  independently bounded.
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
- When one retained primary announcement covers several exact events, enumerate
  every sibling and apply the cheapest independent decisive gate to the whole
  set before researching them serially. If shared terms leave the conservative
  cash-flow floor at zero, terminalize each exact sibling before issuer dates,
  funding, or books; a different ticker does not make the shared terms new.
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
- On the rendered Rewards surface, an earnings value of zero or a competition
  dash is account/dashboard presentation, not proof of zero competitors, full
  owned share, or any positive payout floor. The CLOB book `timestamp` is
  officially the snapshot timestamp; compare it with receipt time and reject a
  stale snapshot without refreshing it or selecting a sibling. Do not reinterpret
  it as a harmless last-change time merely because the levels look attractive.
- A rendered Polymarket market page's `End Date` can be a user-facing
  resolution date rather than Gamma `endDate`. Never freeze equality between
  those fields or invent a Gamma timestamp from the rendered date. Bind the
  exact Gamma end only from a retained Gamma source, or omit that equality gate
  and let the prospectively frozen Gamma request establish the exact horizon;
  every other identity, active-state, fee, token, and reward gate still applies.
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
- Polymarket US cent rounding is a receipt-level accounting overlay only. Never
  split, combine, or time orders to optimize rounding, and never infer partial-
  fill or settlement grouping from the published formula alone. Credit only an
  exact positive owned maker rebate on independently justified legitimate
  organic flow after its applied schedule, receipt grouping, adverse selection,
  inventory, hedge, and every incremental cost are reconciled.
- Polymarket US Volume and Fill incentive rewards are receipt-level overlays on
  activity that must already be independently justified. Never create volume,
  fills, orders, accounts, or eligibility to obtain them; never self-trade,
  wash-trade, spoof, cycle related accounts, abuse cancellation, or double
  count maker rebates, liquidity rewards, spreads, or redemption. Advertised
  pools, examples, and caps are not expected value. Credit only an exact owned
  reward after current eligibility, allocation, receipt, fill quality, and
  every incremental cost are reconciled.
- Polymarket US promotional credit is trading collateral, not cash, and its
  face amount remains non-withdrawable. Do not accept it as a structural edge:
  only resulting trading proceeds may possibly become withdrawal-eligible, so
  positive value requires a favorable trading outcome unless an expressly
  permitted state-complete direction-independent conversion is source-proved.
  Public sources conflict on campaign eligibility, minimum deposit, and whether
  incentive funds themselves may become withdrawable. Never create duplicate,
  synthetic, related, or misrepresented eligibility; onboard, deposit, trade,
  hedge, or liquidate to evade program controls; or ignore reversal authority
  for suspected abuse, self-dealing, coordinated activity, or inconsistent use.
  Any account, deposit, order, fund, or state change requires separate explicit
  authority.
- Treat every trial fund, voucher, bonus credit, and promotional notional as a
  ledger with separate face value, earned reward, acquisition cost, validity,
  lock, regional and account eligibility, reversal, and withdrawal fields. A
  nonwithdrawable face value is never cash, principal, collateral, or profit.
  Credit only the exact deterministic positive reward that reaches the owned
  account after tax, conversion, operating, and every opportunity cost; Points
  or another redemption currency are not free. Public terms can establish the
  mechanism but never an owned amount, APR, eligibility, receipt, or stable
  forward floor. Testnet credentials do not authorize or evidence a mainnet
  reward product. Claiming, redeeming, accepting terms, subscribing, converting,
  transferring, withdrawing, or any other state change requires separate
  explicit authority.
- Polymarket US mutually exclusive and directional collateral return is only a
  buying-power overlay on independently existing eligible positions. It is not
  cash income, trading profit, reduced actual risk, or a reason to open or
  increase either leg. Require exact account enablement and an owned before/after
  margin and buying-power receipt. Released buying power may be used only in a
  different event, hypothetical new-order offsets do not count in pre-order
  buying-power checks, and closing an offset requires returning the release and
  can be rejected when buying power is already deployed. Credit value only
  after an independently justified positive different-event use, exit ordering,
  close capacity, persistence, fees, and every incremental cost are reconciled.
- Polymarket Gamma `outcomePrices` can behave like midpoint diagnostics while
  the same market exposes a materially different `bestAsk`. Never let
  `outcomePrices` alone authorize a CLOB request. For a YES leg, require the
  retained side-specific `bestAsk`; for a NO leg, require a direct NO ask when
  available or use `1 - YES bestBid` only as a conservative rejection proxy.
  Missing side-specific ask evidence blocks escalation, and every surviving
  row still requires an exact current book batch before any economic claim.
- Treat exact-date, cumulative-deadline, interval, and no-release siblings for
  one underlying event as one implication graph, not isolated rendered pairs.
  Bind each condition's start instant and calendar timezone: an exact date can
  imply a later deadline only when that deadline condition existed before the
  exact calendar day began; mutual exclusion with an earlier deadline requires
  that deadline condition to start no earlier than the exact-date event unless
  complete intervening release history is source-proved. Exhaust every valid
  retained relation with conservative side-specific acquisition evidence in
  one zero-network pass. A row must remain strictly positive after exact fees
  and at least one adverse tick per leg before any CLOB request is justified.
  Terminalize the complete retained population when no row survives; do not
  retry it as date aliases or hand-selected pairs.
- Never infer an all-different assignment across independently resolved rank or
  ordinal events from labels alone. Bind deterministic tie handling, the exact
  observation metric and instant, corporate-action handling, and outcome
  identity across every event before using a matching or permutation floor.
  Identity-free placeholders cannot be mapped by letter or position, and an
  `Other` bucket can represent different entities at different ranks. If those
  semantics are absent, fail at the source gate before reading price fields or
  building an assignment optimizer.
- A subset condition implies a broader aggregate only over their shared
  observation interval. Bind exact condition starts as well as deadlines and
  timezones. If the subset starts earlier, a qualifying event in the creation
  gap followed by no aggregate-interval event makes `NO(subset) +
  YES(aggregate)` pay zero unless complete retained history removes that state.
  Never assume a later aggregate condition inherits earlier subset history;
  fail at the source gate before reading prices when any gap counterexample
  remains.
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
  rows only after latency, capacity, freshness, residual, fee, and stress-test acceptance criteria;
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
- Compute canonical contract and result hashes with the repository's Python
  canonicalizer: remove only the named hash field, then use sorted keys,
  compact separators, and ASCII JSON. Never substitute PowerShell
  `ConvertTo-Json`; its serialization is not the executable canonical byte
  contract. If a wrong hash is caught before validation, output creation, or
  network access, correct the hash only and preserve the frozen question,
  population, decision gate, and request boundary.
- Immediately before writing such a contract, capture the UTC clock from the
  host and use that observed instant or an earlier exact second. After computing
  the canonical hash, call the runner's offline validator with the contract path
  resolved to an absolute path. A validation failure before output-path creation
  and before network access leaves the request unconsumed; correct only the
  invalid pre-access field, recompute the hash, and record the correction rather
  than weakening or changing any outcome-sensitive population or decision gate.
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
  equality gate therefore fails. In particular, a top label such as `>T` does
  not override the higher-bracket rule: range `NO(>T)` plus threshold `YES(T)`
  can pay zero at exact equality and is not a hedge. Only `NO(T)` plus all
  upper-range YES claims has an optimistic common-rule one-pUSD floor. Screen
  that weaker cross-event coverage identity only on a distinct nonconsumed
  same-source, same-instant pair with complete boundary rules and
  contemporaneous frozen populations.
  Exhaust both valid directions at every shared boundary: threshold NO plus
  cumulative upper-range YES, and threshold YES plus lower bins through the
  starting bin. Gamma is rejection-only; if exact depth is already negative at
  zero fee, stop before fee endpoints and never refetch stale or skewed books.

- For adjacent daily closes `x0` and `x1` at one shared strict threshold `T`,
  the packages `day1 NO(T) + day0 YES(T) + daily Up` and
  `day1 YES(T) + day0 NO(T) + daily Down` each have an optimistic common-rule
  one-pUSD floor when all three contracts use the same exact source, timestamp,
  close definition, precision, and exceptional-settlement rules. A 50/50 daily
  equality payout does not break that floor because one threshold leg already
  pays one when `x0 = x1`. Prove those rule identities before price, exhaust
  every shared threshold and both directions, treat a displayed daily
  probability as neither an ask nor a lower bound, and stop before Gamma or
  books when even an explicitly labeled optimistic rendered diagnostic is far
  above the floor. Only a distinct synchronized side-specific prefilter that is
  strictly sub-floor may authorize one separately frozen exact book batch.
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
- Prove that interval markets share the exact observation function before any
  price or API request. A Binance candle open/close comparison, a Chainlink
  start/end comparison, an interval TWAP-versus-start comparison, and a
  noon-to-noon close comparison are different payoff functions even when the
  asset and nominal endpoints look aligned. A source, timestamp endpoint,
  aggregation, or tie-rule mismatch stops the package at the rendered-rule
  discovery gate; it is not permission to request Gamma or CLOB data.
- Gamma may encode `outcomes`, `outcomePrices`, and `clobTokenIds` as JSON
  strings inside otherwise parsed event objects. A retained-input preflight
  must assert the exact representation and exercise the production field
  decoder before an implementation-hash-bound contract is frozen. Preserve a
  pre-economic local failure and freeze a new contract; never rewrite it away.
- Retained Gamma JSON is UTF-8 even when canonical downstream artifacts are
  compact ASCII, and scalar `groupItemTitle` labels can carry display suffixes
  such as `+`. Preflight the unchanged raw bytes with UTF-8 and freeze any
  mechanical label normalization before economics; never discover encoding or
  numeric-label assumptions inside an outcome-sensitive run.
- Before refreshing books for a Polymarket maker-reward overlay, reconstruct the
  exact minimum-size one-leg orphan loss from retained evidence and reconcile
  each exact condition with `sponsored=true`. Gamma reward minimum and spread
  fields are eligibility metadata, not funding proof. If even an impossible
  100 percent share of every remaining exact pool does not strictly exceed the
  maximum orphan loss, or an exact sponsored population is empty, stop without
  refreshing books, accessing an account, or repeating the condition.
