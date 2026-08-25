# Continue Development

This is the authoritative handoff. Verify every drift-prone claim before acting.
Development belongs only on `main`; do not create another development branch.

## Closeout State

- The last fully hosted-verified baseline before this structural-parity
  checkpoint is `336114411aab0ad4ed6fae18047245dc420789b2`. CI, Ruff, Vulture,
  Super-Linter, CodeQL, and DeepSource passed that exact revision. GitHub exposed
  only `main`, and the available APIs reported zero open Dependabot,
  code-scanning, and secret-scanning alerts. Reverify the publication commit;
  zero alerts never proves zero undisclosed vulnerabilities.
- The repository is beta `0.1.0-beta.1`. No model has production authority or a
  demonstrated long-lived after-cost edge. Binance remains paper/testnet/Demo;
  Polymarket remains independent, disabled by default, and unpromoted.
- The one historical cutoff is `2026-08-14T00:00:00Z`. Do not move it or fetch
  the newest history on each iteration. Prospective experiments remain isolated
  from that frozen snapshot.

## Round 75 Terminal Verdict

The campaign ended at `2026-08-23T12:00:00Z`. A read-only host audit observed no
owned service or capture process, a released lease, and a valid
`campaign_terminal` service state. The scheduled task's triggers ended at the
same boundary and must not restart the campaign.

Canonical evidence:

- `round-075-terminal-campaign-audit-2026-08-23.json`: rejected incomplete
  campaign; artifact SHA-256
  `94a556887adf33996168de260a8a172952bc64040273a7ff2dfb373f2c2f50d6`.
- `round-075-wal-copy-recovery-2026-08-23.json`: controlled recovery on copies
  only; artifact SHA-256
  `18fa37db65aa19fed16128d4aaf0af10cb0606c95192b46dfca0b0a28e932751`.
- `round-075-post-campaign-amendment-v1.json`: frozen-source preservation and
  non-restartable supervisor v3; artifact SHA-256
  `755a6ed89482e63f118d251ee5d20669c0c899bd26d29e17a7c48b0fe2d84f37`.

Facts: 720 slots were preregistered; 35 produced results, 33 were admitted, 684
were missed, and slot 67 remained incomplete. All admitted epochs are training
role. Tuning and test have zero admitted epochs. Raw training-role eligible
anchor time is `28,903,469,878,300 ns` against `394,740,000,000,000 ns`
required. Shard 002 retains a `9,431,058` byte WAL. WAL recovery on an exact
copy added 196 frames and 62 REST rows but no terminal report, confirming the
payload belongs to the incomplete run and is inadmissible.

Consequences: do not open the original databases, replay or delete the original
WAL, use slot 67, materialize targets, train, tune, inspect sealed tests, or make
accuracy, AI-uplift, edge, profitability, ROI, or trading-authority claims from
Round 75.

## Current Model-Gate Verdicts

- Binance Round 76 is blocked before implementation. Its preregistration required
  a passing Round 75 terminal population; Round 75 failed source continuity,
  role quotas, and train/tune/test coverage. The candidate was not implemented,
  trained, or rejected by a model result. Canonical adjudication:
  `round-076-round75-source-gate-adjudication-v1.json`.
- Polymarket Round 29 is blocked before feature, target, or model access. Stage 1
  produced one terminal primary slot, one incomplete slot with a 3,483,426-byte
  WAL, and no third primary slot. The frozen requirement is three primary dates
  and at least 300 audited eligible markets. Replaying slot B cannot create the
  missing third date. Canonical adjudication:
  `round-029-stage1-readiness-adjudication-2026-08-23.json`.
- Polymarket live-promotion schema v2 now requires strict semantic cross-regime
  evidence bound to the exact model, commit, market variant, risk profile, and
  frozen evidence roles. Hash-valid placeholder reports and caller-asserted
  gates no longer suffice. This control grants no edge, profitability, paper,
  or live claim; no real promotion artifact currently exists.
- The current reproducible action-value status, CSV, and graph are in
  `docs/model-research/action-value/latest-status/`. Round 72 remains the latest
  completed model evaluation and was rejected. Rounds 73-76 contain no invented
  model, trade, ROI, or profitability metrics.

## Source-Continuity Recovery Checkpoint

The source-only recovery design is now implemented and hash-bound in
`docs/model-research/prospective-source-continuity-recovery-design-v1.json`.
Its generic `prospective_capture_gate.py` primitive is shared only for capture
integrity; Binance and Polymarket schedules, storage, quotas, strategies,
capital, targets, and promotion evidence remain independent.

The key correction is slot-local failure containment. Every future capture
window must use a unique database namespace. A failed slot and its WAL may be
terminally quarantined, after which a later already-preregistered window may
continue only if its role quota remains mathematically recoverable. Shared
storage, a WAL on a passed slot, an elapsed slot without a terminal disposition,
target or outcome access, adaptive replacement, or an unrecoverable role quota
fails closed. This prevents the Round 75 pattern, where one shared-shard WAL
blocked every later slot, without salvaging or reusing any failed lineage.

This is a design-only implementation checkpoint. No exact future schedule is
frozen, no host preflight or activation receipt exists, no scheduled task was
created, and no capture, target, model, P&L, edge, profitability, paper, or live
authority was opened. Direct behavior and publication-lineage coverage pass 31
tests; the implementation module itself has 100% statement and branch coverage.

## Structural-Parity Triage

The 2026-08-25 screens deliberately moved away from another directional model
iteration. They evaluate target-free payoff identities whose economics do not
depend on predicting bull, bear, or sideways markets. Both are current-state
diagnostics, not accepted edges.

- Polymarket: the official BTC/ETH/SOL tags contained 1,427 unique active events,
  22 fixed negative-risk events, and one augmented event that was excluded. At
  five-share displayed depth, every possible NO-subset conversion was evaluated
  together with all-YES buy/hold and mint/sell identities. No event had a
  gross-positive path. The closest event, `Bitcoin vs. Gold vs. S&P 500 in
  2026`, was exactly flat before fees and `-0.11804` pUSD after the recorded
  fee curve. Every event's on-chain question count and zero conversion fee were
  independently checked. Canonical result:
  `docs/model-research/polymarket/structural-parity-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `9d7a2c61ae29cb6b29fd3f417ed3e40f1ea08fb2cb6729d20372648abdc448e2`.
- Polymarket logical implications: a later canonical fetch contained 1,473
  unique scoped active events. Exact same-event rule identities yielded 53
  threshold groups and 12 deadline groups: 2,572 implication pairs were
  evaluated and 1,514 had five-share displayed depth. Buying
  `YES(weaker) + NO(stronger)` produced no gross-positive pair. The best gross
  pair, Bitcoin reaching $190,000 versus $200,000 by December 31, 2026, was
  exactly flat and `-0.01372` pUSD after the recorded fee curve. One missing
  market deadline, one missing fee schedule, and two Gamma/CLOB tick-size
  disagreements were explicitly excluded; no term was inferred. Canonical
  result:
  `docs/model-research/polymarket/logical-implication-parity-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `c77c5c6e2e525898f334bd81c54d1b60673226b7488b2833f2f15e17e4de1f78`.
- Polymarket cross-condition duplicates: 1,478 unique active BTC/ETH/SOL-tagged
  events contained 607 eligible non-negative-risk binary order-book markets.
  Exact question matching produced one candidate group: two separate
  MicroStrategy-bankruptcy conditions. Canonical payout-rule fingerprints
  differed in both `description` and `group_item_title`, so the title was not
  treated as proof of equivalence. Zero exact payout-rule duplicate groups
  advanced to depth or fees. Canonical result:
  `docs/model-research/polymarket/duplicate-contract-parity-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `7eab53089f904d647538de29193dcfa33bfabaa73440161d5fdec706b7bcb5b1`.
- Binance: ten tradable scoped spot symbols yielded 24 simple three-leg cycles.
  Seven were positive only in the zero-fee upper bound. The best was
  `USDC -> BTC -> USDT -> USDC` at `0.6461833` bps gross with a break-even
  commission below `0.21539` bps per leg. It fell to `-21.8384` bps under a
  7.5-bps reference and `-29.3258` bps under a 10-bps reference. Those scenarios
  are not assertions about an account's actual fees; exact account/pair fees
  require authenticated evidence. Canonical result:
  `docs/model-research/action-value/binance-spot-structural-parity-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `53498bbf4c1ea7af78f3d05819d965ea3e227b1fa8457c958e5721982b1f3f69`.
  Binance's official spot contract confirms that exact account evidence needs
  one signed `GET /api/v3/account/commission` response per leg. The path's
  sides are BUY BTCUSDC, SELL BTCUSDT, and BUY USDCUSDT; each fee must combine
  the correct taker plus buyer/seller standard, special, and tax components.
  BNB discount may be applied only when both returned flags and sufficient BNB
  are proved. The current process had neither designated ephemeral credential
  variable, so no signed query was attempted and application configuration was
  not inspected. Canonical evidence gate:
  `docs/model-research/action-value/binance-spot-triangle-account-fee-gate-v1.json`,
  result SHA-256
  `304a78180be3375a3453384ad71948c24e52ffeba2f1482cb97711e59aa4a688`.
- Binance options: 1,538 tradable unit-one BTCUSDT, ETHUSDT, and SOLUSDT
  contracts formed 50 same-underlying, same-expiry, same-side chains. Exact
  lot-aligned enumeration covered 26,688 vertical-dominance pairs and 338,904
  arbitrary-strike convexity triples. Two ticker-only convexity candidates
  showed `0.05` USDT credits, but displayed depth repriced both exact minimum
  portfolios to a `-0.15` USDT gross loss. Both candidate paths also failed the
  frozen age/skew gate. The screen therefore stopped after one depth sweep,
  before authenticated commission, margin, or atomic-execution work. Canonical
  result:
  `docs/model-research/action-value/binance-option-parity-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `ceca2f61ab1da16285190afcb90c276a10b032fb7d264c90656aaf2f7266c253`.
- Binance option boxes: the source-bound option snapshot formed 25
  same-underlying, same-expiry chains with both calls and puts. Exact
  minimum-lot enumeration covered 13,344 strike pairs; 5,637 long boxes and
  8,917 short boxes had all four ticker sides. Six strict short boxes showed
  ticker credits `0.028`-`0.088` USDT above their fixed expiry liabilities, and
  one near-expiry long box showed `0.05` USDT nominal carry before costs. The
  candidate-only depth sweep found zero executable boxes because each lacked at
  least one required displayed side; every four-leg timestamp set also failed
  the age/skew gate. No second sweep or fee/margin work was justified.
  Canonical result:
  `docs/model-research/action-value/binance-option-box-parity-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `e85b4e270e707c0faa47e3f373d6d345fce448fcbcbcf6a6f39bceec7d9eb229`.
- Binance quarterly cash-and-carry: one catalog fetch and one spot/futures book
  pair per selected contract covered BTCUSDT and ETHUSDT current/next quarters
  at 12 quantities. All 12 displayed gross bases were positive; nine cleared a
  stated 35-bps sensitivity hurdle. December BTC retained 108.75-110.16 bps and
  December ETH retained 48.54-51.49 bps after that hurdle. This is unqualified,
  not accepted: the hurdle is not authenticated account cost evidence, while
  collateral opportunity cost, liquidation protection, settlement charges,
  and delivery-index versus executable spot-exit basis remain unresolved.
  Canonical result:
  `docs/model-research/action-value/binance-quarterly-carry-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `9c9f75565128cd62372ad1971bab09d910583e27e5c47d8eeaeda4e9177b99a2`.
- Binance quarterly delivery basis: a separately frozen audit produced
  post-delivery spot mismatch values, but its time interpretation is invalid.
  All 16 historical `deliveryTime` values were at 00:00 UTC, while all four
  independently captured current exchange-catalog `deliveryDate` values were at
  08:00 UTC. The audit had treated the historical field as an exact spot-window
  epoch without proving that equivalence. Its mismatch values and rejection of
  hold-to-delivery are therefore non-authoritative. Do not resample or assume a
  `+8h` correction. Original provenance:
  `docs/model-research/action-value/binance-quarterly-delivery-basis-audit-v1-2026-08-25.json`,
  result SHA-256
  `5476fdb43a24bd2d3a31c10321de968f63fd33eca20603e4891fa8d838a134a4`.
  Authoritative adjudication:
  `docs/model-research/action-value/binance-quarterly-delivery-basis-timestamp-adjudication-v1.json`,
  result SHA-256
  `1f669b24c09917e8b080515e8733ba0adea68e74745e2cfafc9dd8f9a45c7f88`.
  Binance's official rule now source-binds the normal quarterly schedule to the
  last Friday at 08:00 UTC, with possible postponement under extreme conditions.
  Timing artifact:
  `docs/model-research/action-value/binance-quarterly-delivery-time-semantics-v1.json`,
  result SHA-256
  `2a52b558f8bc1332cbf2deb41c4e8d4f01bf44d4276ebcc901b3768d4d8516db`.
  A separate 16-contract, 32-request pre-delivery unwind contract is frozen at
  `docs/model-research/action-value/binance-quarterly-pre-delivery-unwind-contract-v1.json`,
  result SHA-256
  `f61a8c9dfd86274292c5dae154120871ea5358e2a5ca004b92574e6bdcb7657c`.
  Its one-use audit stopped after the first futures/spot pair (two requests):
  the expired futures endpoint returned 70 rows through 08:09 UTC, including
  ten flat rows at/after the scheduled 08:00 delivery with zero volume and zero
  trades. This violated the frozen exact-60/no-later-bar cutoff gate. Do not
  rerun or salvage the 60 pre-delivery rows. Returned-kline presence is not an
  authenticated order-state cutoff test; a future design needs authoritative
  order-state evidence or source-bound trade-count/volume semantics. Terminal
  audit result SHA-256
  `07556c4c128fdde32b8bc3ade55134e25eedec157715585aac9e561d87ac9e5a`;
  adjudication result SHA-256
  `e45df8dbffdb8e8e09a542ad3cf2f2f7fe855a775c10f9c07cfa30b290505521`.
  No historical basis result or accepted edge was produced.
- Polymarket paired-maker rewards: the frozen Moonshot candidate's one-tick
  hypothetical 20-share YES+NO bids summed to 0.940 for 1.20 pUSD displayed
  both-fill gross. The books were 8,074 ms old and failed the 5,000 ms gate.
  Public data proves a zero reward payout floor; a separately labeled
  conditional calculation reported 0.2980 pUSD/day against 9.42 pUSD maximum
  orphan settlement loss, but its share, daily-equivalent, and payback values
  are invalid because the hypothetical complementary own asks were omitted
  from the post-quote midpoints. The both-fill and orphan settlement arithmetic
  remain valid. The event is augmented negative-risk, so no
  event-wide payout identity was assumed. The Moonshot condition is outside the
  frozen BTC/ETH/SOL Polymarket research scope and is retained only as a
  negative methodology audit; no rerun or prospective continuation is allowed.
  Canonical result:
  `docs/model-research/polymarket/paired-maker-reward-snapshot-v1-2026-08-25.json`,
  result SHA-256
  `3ed963fe2ff3473dba6c9b5146d842130d4f67ed3a0e8673451330133c68c0b0`.
  Scope adjudication:
  `docs/model-research/polymarket/paired-maker-reward-scope-adjudication-v1.json`.
- The separately frozen BTC/ETH/SOL paired-maker reward screen made one bounded
  live attempt. It stopped after the Gamma request and exact BTC reward request
  because the reward configuration disagreed across those sources. No books or
  candidate economics were reached, no output snapshot was created, and the
  contract prohibits retry or replacement. The terminal receipt is
  `docs/model-research/polymarket/crypto-paired-maker-reward-screen-attempt1-failure-v1.json`.
  The failed tool also taught a workflow correction: every future one-use live
  screen must retain its request ledger and decoded source payloads and write a
  terminal failure receipt before propagating any validation exception.
- The official crypto maker-rebate schedule is now isolated from liquidity
  rewards. Its source-bound arithmetic artifact is
  `docs/model-research/polymarket/crypto-maker-rebate-economics-v1.json`, result
  SHA-256
  `09f67265772716873c625ada816140332430697894f30af22c05bd0dd6422c8a`.
  A 50+50 share example at 0.49+0.49 has 1.00 pUSD complete-set spread and
  0.3498600 pUSD nominal unrounded rebates conditional on both maker fills.
  This is not an edge: the publicly proven payout floor is zero, fill and queue
  evidence are absent, and one orphan can still lose 24.50 pUSD without rebate
  credit.
- Polymarket complete-set holding rewards are a distinct market-direction-
  independent hypothesis. Official mechanics allow splitting 1 pUSD into one
  YES plus one NO and later merging the pair back to 1 pUSD. The Help Center
  says eligible position value includes current YES and NO shares at their
  latest mid-prices and lists BTC, ETH, and SOL 2026 price events. However, two
  current official sources disagree on the annual rate (3.25% versus 4.00%). At
  the lower rate, 35 bps of total friction needs 39.3077 days to break even.
  The rate is discretionary, payout limits may be introduced, random hourly
  sampling is not exact accrual, and split-originated paired eligibility has
  not been explicitly or empirically proved. This is promising enough for a
  later authenticated payout/cost study, but its future payout floor is zero
  and it is not an accepted edge. Canonical economics:
  `docs/model-research/polymarket/complete-set-holding-reward-economics-v1.json`,
  result SHA-256
  `b15b9039848094057322387c9aed3a555a8ca32020af97689fc6b26e16114561`.
  Current public readiness evidence now identifies 26 BTC, 15 ETH, and 14 SOL
  markets whose Gamma records simultaneously report active, open, accepting
  orders, an enabled order book, and holding rewards. One exact BTC $100,000
  market had live midpoints 0.325 and 0.675, preserving a 1.000 pUSD reward
  mark for the split complete set. Official relayer docs say successful relayed
  split and merge operations have zero direct user gas. Gamma flags remain
  candidate filters, not account eligibility or payout proof, and costs outside
  those relayed CTF calls remain unmeasured. Canonical readiness evidence:
  `docs/model-research/polymarket/complete-set-holding-reward-readiness-v2.json`,
  result SHA-256
  `c12bd9d75503c679c1e19800773a002f1d3e82dfad91338cb7bd3ea24dd6a964`.

The shared arithmetic is now in `structural_parity.py` and `logical_parity.py`.
Binance option payoff arithmetic is isolated in `option_parity.py`.
Do not repeat payoff formulas in shell snippets. Tag pages discover event IDs;
canonical event endpoints bind contract terms. Missing market-level deadlines
or fee schedules and Gamma/CLOB execution-term disagreements are exclusions,
not invitations to substitute parent fields or defaults. First prove the payoff
identity and current gross upper bound; stop immediately when it is
nonpositive. Only a gross-positive candidate may consume time on exact fees,
filters, atomicity, latency, fills, inventory, gas, capacity, persistence, and
cross-regime adjudication. Public books never prove fills, and no snapshot
grants paper, testnet, or live authority.

The Binance option workflow adds a request-efficiency contract learned from a
failed exploratory confirmation: fetch the contract catalog once, fetch the
all-symbol ticker once for discovery, and request depth only for ticker-positive
candidates. A 429 must stop without an immediate retry and honor
`Retry-After`; contract metadata must never be fetched again inside confirmation
sweeps. Ticker prices have no displayed quantities and are never execution
evidence. Once all candidates are nonpositive at depth, do not poll again.
Format the module and tool before a source-bound run; changing either afterward
invalidates its recorded implementation hash and permits only one deliberate
regeneration, not open-ended resampling.

Evidence timestamps must use integer epochs or explicit invariant UTC/RFC3339
parsing; locale-dependent implicit date parsing is prohibited. Similar quarter
or expiry labels across Binance products do not establish a shared settlement
timestamp or value. The option/future historical comparison showed different
settlement times, so no cross-product parity identity may proceed without exact
rule and numeric-epoch binding. For Polymarket rewards, the corrected
instantaneous denominator is old aggregate `Q1 + Q2`, never the minimum of the
two aggregates. Public books do not reveal per-maker grouping, queue position,
sampling persistence, or final reward allocation; without that evidence the
provable payout lower bound is zero. A physical binary BUY order also appears
as a complementary ASK at `1-price`; hypothetical post-quote midpoints must
include that mirrored own ask while the physical order is scored only once.

## Protected Local Work

`C:\trader\simple_ai_trading-model-dev` remains detached at
`c42219d47dc781a46411a4ec96838f8a26c3924c`. Its terminal evidence is frozen.
The latest read-only preservation snapshot reports:

- 99 tracked status entries and 218 untracked paths;
- binary diff Git hash `bf7f896c3fa2b17a7a7a34887b2d3fe04cb4be54`;
- untracked manifest SHA-256
  `2b81dd7f8c70bf319ac4b40725c1dd06fc6d0d2be119e45dd064e83d6428d50f`,
  calculated as SHA-256 of sorted UTF-8 records containing path, NUL, lowercase
  file SHA-256, and LF;
- newest untracked write `2026-08-10T21:43:27.4494537Z`.

Never clean, reset, switch, commit, or blindly copy that worktree. Review its
remaining content paths against current `main` with a three-way comparison and
integrate only work that is both unpublished and still valid. The exact Round
75 v4 implementation has already been preserved in `main`; do not duplicate it.

The 2026-08-23 three-way audit is recorded in
`docs/model-research/model-dev-three-way-audit-2026-08-23.json`. Current `main`
descends from the frozen commit; 214 working files match `main` exactly and 15
additional frozen blobs occurred earlier in `main` history. An AST comparison
found no frozen-only top-level source symbol. One still-valid AI edge-floor
regression test was integrated manually; the stale activation-era publication
and all bulk-copy paths remain rejected. Keep the worktree frozen.

The Round 21 sidecar worktree
`C:\trader\simple_ai_trading-round21-sidecar-v2` remains protected through
`2026-08-29T23:40:00Z`. Its process IDs are ephemeral. Do not touch it until a
contract-defined terminal audit proves the boundary has passed and the process,
lease, state, database, and WAL agree.

## Next Work

1. Reverify `main`, GitHub branches, alerts, and exact-SHA hosted workflows for
   this closeout.
2. Keep the completed model-dev three-way audit frozen. Do not bulk-integrate
   stale or divergent files; reevaluate a specific path only when a current task
   requires it.
3. Do not rerun rejected Binance funding carry, quarterly carry, two-sided touch making,
   Polymarket binary complete-set taking, negative-risk parity, logical
   threshold/deadline implication parity, Binance spot triangles, Binance
   option vertical/convexity parity, option box parity, or Polymarket exact
   cross-condition duplicate discovery or paired-maker reward snapshots as if
   repetition could create an edge.
   A repeat is justified only by a frozen prospective sampling contract or
   materially new fee/execution evidence.
4. Do not run Binance Round 76 or Polymarket Round 29 from their failed source
   campaigns. After the protected Round 21 sidecar reaches its terminal boundary,
   use the source-continuity recovery design to freeze separate Round 77 Binance
   and Round 30 Polymarket activation contracts. Each must bind an exact fixed
   schedule, unique per-slot storage, role capacities, host supervision, and a
   pre-market activation receipt. Do not share schedules or storage between
   venues.
   A distinct Polymarket maker-rebate study may be designed meanwhile, but it
   may activate only after the protected sidecar boundary and only under a
   frozen prospective contract measuring authenticated maker fills, queue and
   cancellation latency, adverse selection, realized rebate payment, and
   complete orphan P&L. Nominal rebate algebra is not a substitute.
   The complete-set holding-reward hypothesis likewise requires an exact
   candidate BTC/ETH/SOL market, authenticated rewards eligibility, paired
   balance and daily payout, a reconciled split-to-merge cycle, every cost
   outside the documented gasless relayer path, alternative cash yield, and
   capacity. Do not infer account eligibility from help prose or a Gamma flag,
   and do not activate it before the protected boundary.
5. Reject any candidate that fails bull, bear, sideways, choppy, high-volatility,
   liquidity-stress, or latency-stress after-cost slices. Abstention is required
   where evidence is unsupported; no strategy can guarantee profit or prevent
   every future loss.
6. Keep Binance and Polymarket strategies, capital, ownership ledgers, Stop, and
   promotion evidence independent. Binance data may be a causal Polymarket
   feature only when timing provenance proves it arrives first.
7. Evaluate the night-effect idea as a separate stock-market hypothesis using
   exact exchange calendars, auction mechanics, overnight gaps, spreads, fees,
   taxes, borrow, capacity, and causal timestamps. It has no current crypto or
   trading authority.
8. Perform final walk-forward validation only after source continuity,
   representative train/tune/test coverage, after-cost economic gates, and
   cross-regime gates pass. Walk-forward is not a substitute for those gates.

## Verification Scope

The focused Round 75 closeout passes 20 tests. The cross-regime promotion change
passes 62 affected tests before final publication checks. Run the smallest
affected checks during development, then full CI once before publication. Do
not repeat unchanged expensive suites between adjacent edits.

The previous verbose handoff and chronology are preserved byte-for-byte in:

- `docs/archive/agent-history/AGENT_START-before-2026-08-23-closeout.md.txt`
  (SHA-256 `2ba0ee28f38a9f5d2a177cf4b270fe924517e88f6a9511dd7acb3507ab7907c5`)
- `docs/archive/agent-history/CONTINUATION-before-2026-08-23-closeout.md.txt`
  (SHA-256 `2170f14bcfdf49674c576b8fd7d42aa02dc4569c48ba1f643ec6ad43c8d30b18`)
