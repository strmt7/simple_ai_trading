# Agent Start

Read `AGENTS.md`, then `docs/CONTINUATION.md`. Those two files are the current
operating contract. Historical handoff text is archived under
`docs/archive/agent-history/`; use it only for provenance.

## Current Truth

| Boundary | State |
| --- | --- |
| Release | `0.1.0-beta.1`; experimental |
| Development branch | `main` only |
| Binance | BTC, ETH, and SOL; paper or testnet/Demo only |
| Polymarket | Independent BTC 5-minute/15-minute research; disabled by default |
| Accepted edge | One scoped structural edge positive after direct relayer split/merge cost: Polymarket complete-set holding yield for existing idle on-platform pUSD; not deployment-ready or fully external-cost-qualified |
| Live-money authority | None |
| Historical cutoff | `2026-08-14T00:00:00Z` |

No model, AI component, backtest, capture, or paper result may be described as
profitable without reproducible source-bound after-cost evidence. AI may veto
or reduce risk only; it never creates positions, selects leverage, overrides a
safety gate, blocks Stop, or submits an order.

## Non-Negotiable Gates

- Aggregate performance is not an all-market edge. Promotion requires causal
  bull, bear, sideways, choppy, high-volatility, liquidity-stress, and
  latency-stress slices under
  `docs/model-research/cross-regime-edge-acceptance-contract-v1.json`.
  Unsupported regimes must abstain from new risk.
- Only provably bot-owned orders and positions may be modified. Unknown order,
  fill, fee, balance, redemption, or reconciliation state blocks exposure.
- Risk, ownership, reconciliation, Pause, and Stop remain deterministic and
  independent of model or AI availability.
- Historical labels, future books, fills, resolutions, and PnL never enter live
  inference. Secrets never enter prompts, logs, tests, artifacts, or commits.
- Leverage changes exposure, not edge. No leverage profile bypasses liquidity,
  drawdown, cooldown, ownership, execution-cost, or regime gates.

## Research State

- Round 75 ended and is rejected. Its metadata-only terminal audit found 35
  result slots, 33 admitted training epochs, 684 missed slots, incomplete slot
  67, no tuning or test epochs, and a retained WAL. The training-role raw
  eligible anchor was `28,903,469,878,300 ns` versus
  `394,740,000,000,000 ns` required. Do not train, tune, access sealed targets,
  or claim an edge from this campaign.
- The exact Round 75 sources are preserved under
  `docs/model-research/action-value/round-075-frozen-v4-source/`. Active
  supervisor v3 treats an expired campaign as non-restartable.
- Polymarket Round 29 is blocked before feature, target, or model access. Stage
  1 produced one terminal primary slot, one incomplete slot, and no third
  primary date; it cannot satisfy the frozen three-date/300-market source gate.
- The 2026-08-25 target-free structural-parity screens found no accepted edge.
  Polymarket had zero gross-positive paths across 22 fixed BTC/ETH/SOL
  negative-risk events and zero gross-positive logical-implication bundles
  across 2,572 threshold/deadline pairs. Binance's best three-leg spot cycle
  was only 0.6462 bps gross and required less than 0.2154 bps fee per leg to
  break even. The exact three spot fee queries are now defined from Binance's
  official commission contract, but the designated ephemeral credential
  variables were absent from the current process and no signed request was
  authorized or sent. The fee-gate artifact is
  `binance-spot-triangle-account-fee-gate-v1.json`, result SHA-256
  `304a78180be3375a3453384ad71948c24e52ffeba2f1482cb97711e59aa4a688`;
  this remains non-executable and is not an edge. A separate official API
  review identified two new all-stablecoin triangles whose six spot legs were
  advertised at zero maker and taker fees for eligible users. The first frozen
  attempt stopped before admitting a book sample because Binance reports zero
  `MARKET_LOT_SIZE` steps and the implementation did not fall back to positive
  `LOT_SIZE`; that failure is preserved and not reused. The sole corrected
  recovery then journaled 600 synchronized six-symbol book responses over
  299.729 seconds. Across both orientations and 100, 1,000, and 10,000 quote
  sizes, zero of 7,200 evaluations was positive after exact whole-token
  rounding, displayed top-level capacity, and a 3-bps operational stress. The
  best observed maximum was still -5.0008 bps. Canonical recovery:
  `binance-zero-fee-stablecoin-cycle-recovery-v2-2026-08-26.json`, result
  SHA-256
  `f44f283b311ebf8b3302dba4e1d5d6be0b956a2657483786229611f82ed5da88`.
  This exact zero-fee stablecoin-cycle family is terminal; do not resample,
  reduce stress, or substitute nonsynchronous trades. A separate official API
  review found account-specific liquidity-program overview, performance,
  weekly final-rebate, and spot rebate-history endpoints. The symbol commission
  endpoint explicitly excludes spot maker rebates and the BNB discount effect.
  The public static program document returned 403, both designated ephemeral
  variables were absent, and no signed request was sent. Maker rebates therefore
  remain an unproved structural hypothesis, not a current fee or edge claim.
  Canonical account gate: `binance-spot-maker-rebate-account-evidence-gate-v1.json`,
  result SHA-256
  `19e6d69f73a1f723680aec51b82709ab912e7437f6e7889e89fc74ff834ac88f`.
  Binance option
  vertical/convexity parity covered 365,592 exact payoff
  identities across 1,538 unit-one contracts. Two ticker-only candidates
  disappeared at displayed depth, where every exact minimum portfolio was
  already negative before fees. A distinct fixed-payoff box screen found six
  strict short-box and one nominal long-box ticker candidates across 13,344
  strike pairs; every candidate lacked executable fresh depth. Do not repeat
  these screens without a frozen prospective sampling contract or materially
  new execution evidence. A separate option put-call parity versus quarterly
  future study found 20 non-synchronous gross-positive ticker combinations
  across 192 common expiry-strike pairs. It is not execution evidence. An exact
  eight-GET historical settlement-value audit was frozen at
  `binance-option-future-settlement-equivalence-contract-v1.json`, result
  SHA-256
  `63a57771fe7042381bea0ac052889550738b4890b6c01fadc279e793189b4291`.
  Its hosted-verified one-use run stopped after the first request returned a
  valid empty array for the oldest window. The durable journal is terminal,
  the remaining seven requests and any synchronized depth screen are
  prohibited, and numeric settlement equivalence remains unproved. The
  terminal adjudication is
  `binance-option-future-settlement-equivalence-terminal-adjudication-v1.json`,
  result SHA-256
  `a5d34919c4e9c94ca794b73dea57c96bf9c6f9e968cc6c77f240f243c7597601`.
  Do not rerun or adapt this contract. A future historical endpoint study with
  unproved retention must preregister newest-first ordering so availability is
  tested before older windows consume the one-use attempt.
  A separate Binance delta-hedged BNB fee-inventory screen tested whether an
  equal-base BNB perpetual short could neutralize BNB held only for eligible
  spot-fee discounts. The first valid funding response returned 500 rows rather
  than the requested 1,000 and was source-limited to four complete inner
  months. A separately frozen recovery added exactly one older non-overlapping
  page and merged 1,000 rows without changing the symbol, scenarios, or gate.
  It rejected the mechanism: the worst complete month cost the short hedge
  35.6129 bps, requiring 14.24516 times monthly spot turnover to break even in
  the primary non-authoritative 10-bps-fee/25%-discount scenario. Canonical
  recovery: `binance-bnb-fee-discount-hedge-recovery-v1-2026-08-25.json`,
  result SHA-256
  `85d0be66391b53bef87dda33ea73acaf6995d0200e6423de7999d44a8fed3c8f`.
  This is terminal and not an accepted edge. Do not paginate again, resample
  books, loosen the turnover gate, or substitute a public discount for the
  signed account commission response. Future one-use contracts must prove that
  their bounded request budget can physically supply the required horizon.
  A source-first USD-M/COIN-M perpetual funding review also stopped before any
  market request. Official schemas expose separate public funding endpoints and
  COIN-M `contractSize` and `marginAsset`, but do not bind a complete
  cross-product payoff, collateral hedge, funding conversion, or
  liquidation-safe path. Neutralizing the required coin collateral otherwise
  reintroduces the already-terminal spot/perpetual carry family. The generated
  Go account Markdown also says `No authorization required` for commission
  sections while Binance's official Python transport signs both commission
  GETs; use the stricter signed classification. Canonical triage:
  `structural-edge-source-triage-v2-2026-08-25.json`, result SHA-256
  `3df17e93866cbf53617340dd422a91945c8a1924d4ca736b76c5f78f4c9a5575`.
  Do not build a funding or book collector without materially new official
  payoff/collateral semantics and the designated account evidence.
  A newly identified Binance stable-value yield-allocation candidate compares
  BFUSD, RWUSD, and exact same-currency alternatives without forecasting crypto
  direction. Official APIs expose signed read-only rate history and
  account-specific subscription/redemption quota details for both products and
  the current eligible USDT/USDC flexible alternatives.
  Both designated ephemeral credentials were absent, so no signed request was
  made and no current net-yield edge is claimed. A logged-out RWUSD promotion
  was explicitly excluded from evidence. Canonical gate:
  `binance-stable-yield-allocation-evidence-gate-v1.json`, result SHA-256
  `3096867474c4b5a0b3f893645bac68081ceb3783ad14393261e6d88793b64a8a`.
  Do not freeze another signed collector until both credentials exist; never
  use a marketing APR, example fee, assumed daily cadence, or zero alternative
  yield as a shortcut.
  A materially new public route now exists: `BFUSDUSDT` and `BFUSDUSDC` are
  live spot markets, so BFUSD can be screened against its 1:1 subscription and
  redemption identity without predicting BTC, ETH, or SOL direction. The exact
  `BFUSDUSDT` depth had a 1.0000 bid and 1.0001 ask; at 100, 1,000, and 10,000
  BFUSD, neither buy-then-redeem nor subscribe-then-sell was positive under the
  clearly labeled 10-bps spot and typical 10-bps product-fee sensitivities.
  Public daily trade bars since 2025-08-13 ranged from 0.995 to 1.08, which
  justifies a future executable-depth trigger but does not prove historical
  depth or fills. Canonical screen:
  `binance-bfusd-spot-redemption-parity-v1-2026-08-26.json`, result SHA-256
  `566be5e515ac14d38377b6a6b42101cc9b8a65585142053791b759efbd77f6bb`.
  Do not poll the book until signed account commission, quota, exact fee, and
  reward evidence sets executable thresholds.
  Current public promotions add one conditional path, not a stable accepted
  edge. Binance published an 8.07% effective APR for the first completed week
  of its RLUSD/XRP campaign. It has no stated individual cap, but requires an
  eligible account, RLUSD collateral, and at least 500 USD average daily genuine
  Margin or Futures volume; future weekly APRs are unknown and the campaign ends
  2026-09-11. Never manufacture volume to qualify. The contemporaneous 14-day
  USDT and USDC Flexible bonuses have maximum combined gross rewards of only
  about 1.34 USD and 0.58 USD at their public approximate base rates and capped
  bonus tiers. Canonical triage:
  `binance-public-promotion-yield-triage-v1-2026-08-26.json`, result SHA-256
  `fe34f9aaf64a0ec920b0cf7cc7fd1141d30880d1205454779327e41fd7521b1c`.
  Observe the fixed August 28 and September 4 RLUSD updates; do not extrapolate
  one week, illustrative examples, or system timezone into profitability or
  account eligibility.
  A separate 2026 bStock screen covered 67 trading tokenized-stock symbols
  against Binance's public external reference price. `SNXXBUSDT` was the only
  selected live outlier and remained gross-positive at 1,000 and 5,000 USDT
  after a labeled 20-bps spot-plus-stock cost sensitivity, by 0.6463 and 2.0877
  USDT respectively. The reference is not an executable Binance Stocks sale
  quote, so this is not after-cost evidence. A direction-neutral long-bStock /
  short-same-underlying-perpetual diagnostic found DRAM, MU, MRVL, and SNDK;
  each cleared a labeled 30-bps round-trip sensitivity in every available
  complete inner month, but only two or three complete months exist. Exact
  account costs, eligibility, conversion state, multiplier-to-hedge mapping,
  margin/capital costs, and permanent scope authority are absent. Canonical
  research-only artifact:
  `binance-bstock-reference-parity-v1-2026-08-26.json`, result SHA-256
  `73fec22cc61fc8be0c792a78c0340fcb163b9bf7862708796d50521a9c44a8ac`.
  A later one-shot causal screen removed the apparent funding promise. Of 66
  matched bStock/perpetual pairs, 60 had an exact multiplier and 57 were
  unexposed confirmation candidates. Zero passed the frozen training gate
  after a 60-bps round-trip stress, a 10%-annual two-leg capital hurdle,
  family-adjusted block bootstrap, risk gates, and eight regime slices. A
  separately frozen top-20% equal-weight basket then averaged -81.80 bps in
  validation and -108.20 bps in test; only 1/12 and 0/12 symbols respectively
  were positive, and every aggregate regime slice was negative. Canonical
  results are `binance-bstock-funding-full-universe-v1-2026-08-26.json`, SHA
  `ad3fbc7a09ff6b467955eeef8bf1e8df4ba7d20ca9e7659fcaf75069da622d3f`,
  and `binance-bstock-ranked-basket-v1-2026-08-26.json`, SHA
  `0cf6e3aae168e0c483634e78fd824a80be9e58269f02e9b01a6d9c9c46578a8f`.
  The funding family is terminal: do not weaken, rerank, or resample it. The
  reference-conversion hypothesis remains account, quote, cost, and scope
  gated; neither result is an accepted edge.
  A source-first follow-up also identified WBETH/ETH and BNSOL/SOL conversion
  parity as a distinct direction-neutral candidate. Binance's official staking
  conversion-rate, quota, reward, and operation-history paths are signed, both
  ephemeral credentials remained absent, and no collector or book screen was
  opened. Public liquid-staking-token books do not prove the same account can
  redeem at a current ratio, fee, quota, or arrival time. The source-triage
  artifact is `structural-edge-source-triage-v1-2026-08-25.json`, result
  SHA-256
  `509f63910c77a582680849e779317396962d06edeffa537e7d5ce8e18a984cb2`.
  The same artifact repairs the structural registry's missing terminal entry
  for Round 61 elevated-funding spot/perpetual carry and prohibits manufactured
  Polymarket taker-tier volume, self-matching, wash trading, and treating
  one-time bonuses as persistent edge.
  Polymarket cross-condition duplicate discovery found
  one repeated exact question across 607 eligible binary markets, but its two
  canonical rule sets differed; zero exact payout-rule duplicates advanced to
  pricing.
- A distinct Binance USDT-versus-USDC perpetual funding differential is also
  terminal. Two recent 500-settlement BTC/SOL candidates survived corrected
  mark-price and conservative FX accounting, but the frozen 2,898-settlement
  full-history recovery rejected both. BTC failed selection stress, validation,
  and the up, sideways, regular-volatility, and continuation slices. SOL failed
  validation, test, most regimes, and the fee gate. Zero of two candidates
  passed. The terminal artifact is
  `binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json`, result SHA-256
  `8e30be61daaecabd3546e41cdc204d20b8ad38e0fc80c3c9aa96092266a3abe5`.
  Do not repeat this public backfill. Its v3 predecessor also exposed a workflow
  defect by discarding fetched payloads after a later FX validation failure;
  v4 was the sole recovery and durably journaled all 20 responses.
- The newer Binance U-settled BTC/ETH perpetual versus matched USDT perpetual
  funding differential is terminal under its frozen static orientation. Across
  171 aligned settlements per base, training selected short USDT and long U,
  but gross funding reversed negative in both validation and test for BTC and
  ETH. Every role was negative after the frozen 20-bps round-trip execution and
  two-leg capital hurdles. Zero candidates passed, so no book or account
  escalation is permitted. Canonical result:
  `binance-u-usdt-funding-differential-v1-2026-08-26.json`, result SHA-256
  `486b1aa261ae41fd8d8aeb19f0fea5bb01305d24927ccd72624bdd8afb7895d7`.
  Treat unchanged static quote-stable perpetual funding differentials as the
  same terminal family unless a new payoff or incentive changes the economics.
- The 2026-08-25 direction-neutral carry/reward diagnostics also found no
  accepted edge. Binance quarterly spot/future basis was gross-positive at all
  12 tested sizes and cleared a stated 35-bps sensitivity hurdle at nine, but
  exact account costs, collateral/liquidation economics, and delivery-index
  versus spot-exit basis remain unresolved. A frozen eight-delivery audit used
  historical `deliveryTime` values at 00:00 UTC as exact spot-window epochs,
  but the independently captured current exchange catalog uses 08:00 UTC for
  every quarterly `deliveryDate`. The eight-hour semantic mismatch invalidates
  the audit's post-delivery mismatch values and hold-to-delivery rejection. Do
  not resample that audit. Binance's official quarterly-delivery rule now binds
  the normal schedule to the last Friday at 08:00 UTC, while allowing extreme
  postponements. A separate pre-delivery unwind audit stopped terminally after
  two requests: the expired futures endpoint returned ten flat, zero-volume,
  zero-trade rows at/after the scheduled cutoff, violating the frozen
  no-later-bar gate. Do not rerun or salvage its pre-delivery rows. Kline
  presence is not authenticated order-state evidence; no historical basis
  result was accepted. Read-only mainnet evidence authority was subsequently
  granted, and an exact seven-signed-GET commission/configuration contract is
  frozen. The two required ephemeral process variables were absent, so zero
  authenticated requests were made. Do not recover secrets from chat, shell
  history, logs, or repository files. When both variables are available, run
  the frozen capture once and use it only as a fee gate before any fresh book
  sampling. A two-leg long-current/short-next quarterly calendar spread was also
  rejected algebraically: its initial credit is reduced by the unknown exit
  spread or terminal far basis, so it is residual curve exposure rather than
  locked carry. Zero request and no backtest were justified; do not repeat it
  as a fixed-payoff edge. Polymarket
  paired-maker quoting had
  a stale displayed both-fill surplus of 1.20 pUSD, but the public reward payout
  floor is zero and the 9.42 pUSD orphan-loss bound remains valid. Its reported
  conditional share, daily-equivalent, and payback are invalid because the
  hypothetical complementary own asks were omitted from the post-quote
  midpoints. Its Moonshot candidate is outside the BTC/ETH/SOL research scope
  and is retained only as a negative methodology audit. Do not repeat either
  snapshot; future Polymarket reward work must be BTC/ETH/SOL and satisfy the
  evidence gates in the screen document. The first frozen in-scope crypto
  screen then stopped after two public requests because BTC reward settings
  disagreed between Gamma and the exact CLOB reward endpoint. It reached no
  books or economics and is terminal without resampling.
- The separate official crypto maker-rebate schedule has exact conditional
  filled-order arithmetic, not an accepted edge. At 50 shares bid on each side
  at 0.49, the unrounded nominal rebates total 0.3498600 pUSD and raise the
  conditional both-fill value from 1.00 to 1.3498600 pUSD. Public evidence still
  proves no positive payout lower bound, queue position, fill probability, or
  orphan protection; the one-fill settlement loss bound remains 24.50 pUSD
  without rebate credit. Do not turn this arithmetic into a profitability claim
  or activate a capture while the Round 21 sidecar is protected.
- Polymarket complete-set holding yield is now a validated, narrowly scoped
  structural edge after direct relayer split/merge cost for existing idle pUSD
  already on Polymarket. It does
  not require a market-direction forecast: equal mergeable YES and NO shares
  preserve the complete-set value while the current holding program pays yield.
  The canonical economics artifact is
  `complete-set-holding-reward-economics-v1.json`, result SHA-256
  `b15b9039848094057322387c9aed3a555a8ca32020af97689fc6b26e16114561`.
  A later public readiness capture found 26 BTC, 15 ETH, and 14 SOL markets
  with active/open/orderbook/holding-reward flags. The exact BTC $100,000
  candidate had live YES+NO midpoints of 0.325+0.675=1.000 pUSD. Official
  relayer documentation also confirms zero direct user gas for successfully
  relayed split and merge operations. A capped exact-market diagnostic then
  used the wrong activity subtype: official SDK bindings distinguish holding
  `YIELD` from generic `REWARD`, so its blank-condition `REWARD` rows were
  liquidity rewards and not holding-yield evidence. The readiness artifact is
  `complete-set-holding-reward-readiness-v2.json`, result SHA-256
  `2d3650d65f248294395fcac336c6650e0c6bc332cb490c6f0bac70bc11244e2c`.
  The corrected public reconciliation found one wallet with 150 YES plus 150
  NO mergeable shares in the BTC $45,000 condition and no other currently
  holding-reward-eligible position value. Fourteen consecutive `YIELD` rows
  paid 0.1816 pUSD; all 14 Polygon receipts exactly reconcile transfers from
  the holding-yield distributor to that wallet. The realized annualized gross
  rate was 3.15638%, and every payout maps to 21-24 hourly samples under the
  current official 3.25% formula; the stale 4% rate does not fit. Canonical
  reconciliation:
  `complete-set-holding-yield-reconciliation-v3-2026-08-26.json`, result SHA-256
  `48e31f3d6021d28946fa1f143f65ff0f6baf9a222424f41e76c2d89875796abe`.
  A separate cross-asset capture then reconciled explicit on-chain split origin,
  subsequent merges, the remaining equal complete sets, and 28 of 28 positive
  daily `YIELD` transfers for ETH and SOL. ETH retained 440 of 550 split sets
  after 110 merged and received 0.5377 pUSD over 14 days; SOL retained 449 of
  550 after 101 merged and received 0.5488 pUSD. Both map to 330 of 336 possible
  hourly samples and realized about 3.186% annualized gross. Canonical
  cross-asset reconciliation:
  `complete-set-holding-yield-cross-asset-v4-2026-08-26.json`, result SHA-256
  `eda29a314218e1724e39984e2712a4351d9e697503d4583d391c89a060ba53ea`.
  The v5 net-economics adjudication combines the three cases: 1,039 pUSD of
  demonstrated principal produced 1.2681 pUSD over 14 days across 42 of 42
  positive daily payouts, or 3.182019% principal-weighted annualized. Because
  the documented relayer pays split/merge gas and the CTF identities restore
  one pUSD per complete set, all three cases remain positive after direct
  mechanism cost. They also remain positive against a 3% annual alternative
  before external friction, but the weighted spread is only 18.2019 bps: a
  10-bps external friction needs 200.53 days to break even, and only 6.3333
  bps can be tolerated over 127 days. No case beats a 3.25% alternative after
  realized hourly sampling and payout rounding. Canonical adjudication:
  `complete-set-holding-yield-net-economics-v5-2026-08-26.json`, result SHA-256
  `dff80903a20d9bfc8e3402eea01dad8a8f5ee39b0427690514cc30b9fe9dcb85`.
  The accepted scope still excludes bridging, wrapping, withdrawal, custody,
  tax, failed operations, and the exact best eligible alternative yield. The
  BTC wallet still lacks split-origin
  lineage, while the ETH and SOL cases close that limitation for their public
  wallets. None proves future rewards; the rate is discretionary and caps may
  be introduced. No deployment, account, funding, transaction, paper, or live
  authority exists. Continue public monitoring without touching Round 21.
- A shared source-continuity gate now permits only slot-local failure
  containment for future, separately activated Binance and Polymarket
  campaigns. It is design-only: no future schedule, capture, target, model, or
  authority is active.
- The cross-venue Binance Prediction Trading versus Polymarket parity screen
  stopped before market access. Official generated Markdown labeled market list
  unauthenticated, but the generated Java transport attached
  `binanceSignature`, and the live no-key request returned HTTP 400 with
  `-2014 API-key format invalid`. Zero market payloads or books were viewed.
  Do not inspect stored credentials or retry without explicit read-only
  credential authority and a new frozen authenticated contract. Result artifact
  `cross-venue-prediction-parity-screen-v1-2026-08-25.json`, result SHA-256
  `628e63106bc3c0e28c36dcad094b7d7ac500ecd14dfff827287030c2dbbb3d72`;
  no edge or trading authority exists.
- The independent Round 21 sidecar remains protected until
  `2026-08-29T23:40:00Z`. Do not stop, restart, stage, clean, reset, switch,
  commit, or modify its process, worktree, state, database, or WAL.
  This protection applies only to that capture and its assets; it does not
  block separate read-only Polymarket research in the main worktree.

## Task Routing

| Work | Read first |
| --- | --- |
| Current plan or handoff | `docs/CONTINUATION.md` |
| Binance model/backtest | `docs/model-research/action-value/latest/README.md` |
| Polymarket model | `docs/model-research/polymarket/latest/README.md` |
| Structural parity | `structural_parity.py`, `logical_parity.py`, and the three 2026-08-25 snapshots |
| Structural edge priorities | `docs/model-research/structural-edge-priority-registry-v1.json` |
| Structural source triage | `docs/model-research/structural-edge-source-triage-v1-2026-08-25.json` |
| Quarterly carry | `quarterly_carry.py`, `quarterly_carry_account_evidence.py`, and `binance-quarterly-carry-screen-v1.md` |
| Maker rewards | `polymarket_liquidity_rewards.py` and `paired-maker-reward-screen-v1.md` |
| Model promotion | `docs/MODEL_AND_SIGNAL_VALIDATION.md` and cross-regime contract |
| Execution/risk | `docs/LIVE_MARKET_SIMULATION.md` and venue runbook |
| AI | `docs/ai/risk-review/latest/comparison.json` |
| Windows/CLI parity | `src/simple_ai_trading/command_contract.py` and parity tests |
| CI/release | `docs/AGENT_WORKFLOWS.md` and `docs/release.md` |

Before editing, verify `git status`, `git worktree list`, active processes,
scheduled tasks, `origin/main`, open alerts, and the exact evidence boundary.
Never infer current host state from an old PID or archived note.
