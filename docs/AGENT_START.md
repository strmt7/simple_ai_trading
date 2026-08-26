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
| Accepted edges | Four scoped structural edges: Polymarket complete-set holding yield for existing idle on-platform pUSD; Binance Soft Staking yield for already-held idle non-order ETH/SOL Spot inventory not needed for prompt liquidity; Binance LDUSDT yield only for already-required futures collateral; and bounded just-in-time BNB fee reduction only for independently justified organic BTC/ETH/SOL Spot flow under exact full-consumption and risk gates. None is deployment-ready or fully account-and-external-cost-qualified. |
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
- A public Polygon pUSD parity screen stopped at the optimistic marginal fee
  gate. At pinned block 92,696,858, the exact USDC.e wrap-then-sell loop lost
  22.4388 bps after the 30-bps Uniswap v3 pool fee, while the native-USDC
  buy-then-unwrap route lost 0.3328 bps after its 1-bp fee before price impact,
  gas, and the unclosed native-USDC/USDC.e basis. Do not request quotes or
  resample unchanged pools. Reopen only for a materially lower-fee same-asset
  route or a source-bound deviation exceeding fee, gas, and finite-size impact.
  Canonical result: `polymarket-pusd-external-parity-v1-2026-08-26.json`,
  SHA-256 `c15f1e131aa18d705aa6ce507c0f921b7a559664db91a352a244d8df9ddb0f99`.
- A separate public Polymarket UMA proposer-reward screen found positive
  aggregate observed economics but rejected a stable or publicly accessible
  edge. Thirty-nine BTC/ETH/SOL questions offered 23.4 USDC.e total reward
  against 19,500 USDC.e locked for 600 seconds. Under an explicitly
  non-authoritative USDC.e-equals-USDT valuation, charging all scoped proposal
  transactions and all eight resolution transactions left 4.3644 USDC.e
  equivalent, or 2.2382 bps. However, one of five actual proposal batches was
  already negative after observed gas, allocated resolution gas, and capital
  cost; aggregate POL-price headroom was only 22.9723%; and 38 of 39 scoped
  rewards were claimed in the first Polygon block. Do not repeat the current
  cluster or treat protocol/private automation as public execution. Canonical
  gate: `polymarket-resolution-proposer-reward-gate-v1-2026-08-26.json`,
  SHA-256 `ee76a40a86e1c777006c697798d0ad3da20609cadd1c2d8f6bf039ecb79f2155`.
- The strongest new Polymarket execution lead is buying a finalized winner at
  0.999 after an undisputed UMA proposal expires but before the adapter closes
  the market. A fixed five-hour public screen covered 195 BTC/ETH/SOL hourly
  markets. Two of five clusters contained on-chain-confirmed taker sells into
  maker bids after exact finality, totaling 278.14 winning shares and 0.27814
  pUSD gross. Current exact market terms make fees taker-only, official terms
  make the maker fee zero, and successful relayed redemption is user-gasless.
  This is a positive direct-cost execution lead, not an accepted edge: public
  history does not reveal when the maker bids were created, and a bid resting
  before finality has directional risk. Do not expand history. After the
  protected boundary and only with explicit authenticated paper authority,
  freeze one minimum-size post-finality order-acceptance probe. Canonical gate:
  `polymarket-finalized-winner-redemption-latency-gate-v1-2026-08-26.json`,
  SHA-256 `3df84b6639c409ffca472bb4566e623ac78f160e7d8bc66795009f619edfdcb1`.
- Binance Soft Staking is now an accepted scoped incremental-yield edge only
  for identical ETH or SOL already held idle in Spot, outside pending orders,
  outside Auto-Subscribe, and not needed for prompt execution or withdrawal.
  The current official page displays 0.50% estimated APR for both assets and
  the current FAQ specifies daily native-token rewards with no additional fee.
  Do not buy or retain either asset for this yield, count frozen order balances,
  or treat advertised flexibility as guaranteed liquidity under stress. The
  signed product list and reward history remain unproved; activation is a
  state-changing signed `GET` and needs separate authority. Canonical gate:
  `binance-soft-staking-idle-spot-yield-gate-v1-2026-08-26.json`, SHA-256
  `9ded119650ed1679795cca8616935015bc8bf48850bfcc509ba28486e94bd9a7`.
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
  A materially distinct bounded just-in-time BNB fee buffer is now an accepted
  scoped incremental cost edge for already-intended independently profitable
  organic Spot flow. The official current Spot BNB discount is 25%, and the
  public Convert catalog permits a `0.01` USDT USDT-to-BNB minimum, about 562
  times smaller than the current `5.61864` USDT spot-lot fallback. With full
  immediate consumption and a 100-bps acquisition-cost stress, that minimum
  buffer saves `0.0032` USDT at zero BNB move and does not break even until a
  `24.2424%` adverse BNB move; principal at risk is capped at one cent for the
  first reconciliation. This is not deployment-ready, standing inventory,
  hedging, risk-free arbitrage, or authority to create volume. Exact signed
  account/symbol eligibility, positive standard commission, exact-order fee,
  executable quote cost, consumption, short holding interval, residual, and
  owned deduction remain mandatory. Canonical gate:
  `binance-bnb-just-in-time-fee-buffer-gate-v1-2026-08-26.json`, result SHA-256
  `b97eed6a93070d5e29b26d1a47757c9be49e0296332c8019a64388ba936c3b6b`.
  Both credential variables remain absent, so no signed or funded action was
  made. Do not poll public books or refit; when both credentials exist, freeze
  one commission plus exact-order test contract, then require separate funded
  authority for one `0.01` USDT maximum fully consuming reconciliation. The
  reverse public minimum is `0.000014` BNB, so do not assume a partial residual
  from the minimum inbound buffer is independently unwindable.
  A materially different BNB candidate now stacks BNB Simple Earn base rewards
  with realized Launchpool, Megadrop, and HODLer distributions while keeping the
  equal-base BNBUSDT short. Binance's current seven-day BNB offer advertises
  0.35% APR plus applicable airdrop rewards. The base rate is only 0.67123 bps
  over seven days, versus an 18.4917-bps worst rolling 21-payment short-funding
  cost in the already-frozen 1,000-row hedge history. Realized account airdrops
  must therefore clear at least 17.82047 bps before every other cost in that
  window. Both credential variables remain absent, so no signed request was
  made and no edge is claimed. Canonical gate:
  `binance-bnb-stacked-reward-hedge-evidence-gate-v1.json`, result SHA-256
  `0bfc615af743f4ba352201ff2f06e2abf0f0c8fec56b548a0e19791faf25f8ed`.
  This does not reopen the terminal fee-discount-only family: do not refresh its
  funding or books. Once both credentials exist, account principal, Simple Earn
  reward, and all-asset dividend history are the decisive new evidence.
  A June 2026 paper on Binance options versus Polymarket BTC thresholds is now
  source-triaged as a distinct direction-neutral statistical hypothesis, not
  exact arbitrage. Its 16-trade pooled net-alpha confidence interval crosses
  zero. The current public catalog contains 83 active BTC point thresholds and
  337 Binance BTC calls but zero exact strike-and-expiry pairs. All 20
  same-date/strike pairs leave an eight-hour terminal gap; only 15 had two-sided
  option quotes. Their mean model midpoint wedge was 1.2559 percentage points,
  the maximum absolute wedge was 2.7804 points, and zero cleared the paper's
  4.27-point historical mean friction term. No current economic edge is
  claimed. Canonical gate:
  `binance-polymarket-option-threshold-wedge-gate-v1-2026-08-26.json`, result
  SHA-256
  `22a99f25de487774ac4d22f4666a242fe3cb961e31f7f610de7a079cd6d9d7e7`.
  Do not resample until an exact payoff mapping appears or a two-sided model
  wedge exceeds 4.27 points, then freeze a prospective executable-cost study.
  The retained Round 26 public BTC capture also exposed a distinct conditional
  execution lead. In 10 consecutive complete 5-minute conditions, the exact
  closing Chainlink TWAP direction matched the later resolution, aggregate
  winner bids grew only after local receipt of that observation, and later
  winner-side seller fills implied 13.40488 pUSD gross at public event prices.
  Both Up and Down conditions contributed, so no direction forecast is needed.
  This is not yet an accepted edge: aggregate book events do not prove a fresh
  authenticated order was accepted after observation, public trades do not
  prove owned fill lineage or queue position, the capture was one degraded BTC
  hour, and all costs remain unbound. Canonical gate:
  `polymarket-post-observation-maker-window-gate-v1-2026-08-26.json`, result
  SHA-256
  `03dcb88790b96bcaed6a58dc921abff5244e3b2eecd3a39e8f4e82c412f49392`.
  A clean prospective BTC/ETH/SOL interval then showed post-observation winning
  bid growth in 3/3 conditions, but qualifying later winner-side seller fills
  recurred only for BTC (1/3 conditions; 0.01022 pUSD public gross). All three
  resolved Up, so direction balance and cross-regime persistence remain absent.
  Prospective artifact:
  `polymarket-post-observation-prospective-v2-2026-08-26.json`, result SHA-256
  `079925ec06eda0cdfc5851d71d7fc76df96de6f03883bcc70edc0f36da28d421`.
  Freeze a non-overlapping multi-interval public contract before collecting
  more; do not refit to this interval. Do not submit an order. Only after
  `2026-08-29T23:40:00Z` and explicit authority may one minimum-size no-crossing
  authenticated order-acceptance probe be frozen.
  Binance LDUSDT is now the highest-priority Binance structural lead. Official
  product guidance says eligible USDT Simple Earn Flexible assets can become
  LDUSDT, remain USD-M Multi-Assets margin, and keep Real-Time APR rewards.
  Public LDUSDTUSD and USDTUSD index histories produced 505 aligned daily closes
  over 504 days: normalized LDUSDT appreciated 2.78729%, or 2.01091% compound
  annualized; the latest 7-day pace was 2.74952% annualized. Public exchange
  information still marks LDUSDT margin-eligible. This is a validated gross
  incremental collateral-yield mechanism. Official current terms bind a 99.9%
  collateral value ratio, the same conversion ratio both ways, and zero
  additional swap fees. It is accepted only for eligible LDUSDT already held as
  margin already required by an independently justified futures strategy, with
  the haircut fully budgeted and no liquidation or auto-exchange. It is not a
  reason to open a futures position and is not deployment-ready. Canonical gate:
  `binance-ldusdt-margin-yield-gate-v1-2026-08-26.json`, result SHA-256
  `6c2b81a8067faac80efb56f586d89bc308cb69b4fae0ec8504adc3aa2f3ff49d`.
  When both designated ephemeral credentials exist, freeze one GET-only account
  prequalification. A minimum-size conversion round trip then requires separate
  explicit funded authority and must not open a futures position.
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
  A separate offline stack used BFUSD as reward-bearing collateral for the long
  Binance BTCUSDT perpetual hedge against a short Polymarket BTC Perpetual. It
  reused the retained non-authoritative BTC diagnostic and made no venue
  request. The aggregate already-held-collateral break-even is 5.1825556081%
  annual BFUSD yield, but every fixed role clears only above 14.1066194737% in
  the worst role; typical 10-bps subscription plus 10-bps redemption friction
  raises that worst-role threshold to 25.6329352632%. The current public guide
  states that BFUSD rates vary daily and exact rate, reward, and quota evidence
  is signed USER_DATA. Both designated credentials are absent, so no edge is
  claimed and no book request is justified. Canonical gate:
  `polymarket-perps-binance-bfusd-collateral-stack-gate-v1-2026-08-26.json`,
  result SHA-256
  `5919c6fdf73b15c7774feefb8b0a57129c3a43f668b2e4d5f6aed528a094dbe3`.
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
  Binance has also announced a USD1 Flexible promotion from 2026-08-27 through
  2026-09-25: 7% fixed Bonus Tiered APR on at most 1,500 USD1 plus an
  approximately 1.5% variable Real-Time APR. Against the current USDT Flexible
  alternative, the conservative 28-bonus-day target case adds 25.0228 bps.
  The worst public 30-day USD1USDT close move was -23.9545 bps, leaving only
  0.9683 bps after the contemporaneous 0.1-bp displayed round-trip spread and
  before unknown commissions, account eligibility, reserve, and redemption
  risk. The promotion had not started, the latest listed attestation was June
  2026, and the live public reserve dashboard returned no collateralization
  ratio. This is therefore a high-priority conditional time-limited candidate,
  not a stable accepted edge. Canonical gate:
  `binance-usd1-simple-earn-promotion-gate-v1-2026-08-26.json`, result SHA-256
  `230b1524f337964394a45ffe047adfd19b35b339a7735866a15cafdd7549c6f1`.
  At activation, confirm terms and exact fee treatment once; do not assume
  public Convert bounds, a displayed spread, or an issuer redemption claim is
  an executable fee-free round trip.
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
  A later public recurrence study proved that the program does make material
  payments, without proving a trading edge. All top ten weekly crypto-volume
  wallets had `MAKER_REBATE` receipts on each of seven UTC dates. The top
  monthly crypto-volume wallet's 2026-08-25 payment decomposed into 905 markets;
  668 BTC/ETH/SOL rows paid 7,017.331032 USDC, 94.10% of its 7,457.259568-USDC
  daily total, across 5m, 15m, and 4h markets. Public activity omits fills,
  queue, capital, adverse selection, inventory, and orphan P&L, and wallet-level
  receipt-to-volume ratios mix incompatible public fields. The fresh-order
  payout floor therefore remains zero and no edge is accepted. Canonical
  recurrence:
  `crypto-maker-rebate-public-recurrence-v2-2026-08-26.json`, result SHA-256
  `c992e0e1febc1a9789289cb129c166280ee0192cab203d3a6935a8c40e949612`.
  Do not poll public wallets again unless terms change. After the protected
  boundary, only owned authenticated fills and complete after-cost inventory
  reconciliation can decide this candidate.
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
  A frozen current-state valuation-uplift screen then covered all 55 eligible
  BTC/ETH/SOL markets in four public requests. Every market had both midpoints,
  and every equal YES-plus-NO complete set summed to exactly 1.0000 pUSD at
  displayed precision. Zero cleared the 1.0884312538 midpoint-sum threshold
  required to beat a 3.25% alternative plus 10-bps friction over 127 days.
  Do not collect books or history for this unchanged uplift idea. Canonical
  result: `complete-set-midpoint-uplift-v1-2026-08-26.json`, SHA-256
  `33cdc53555f8bbdecf6a9977a77d2c3bc004dab4bff27abb36eac4452f96e5a3`.
  The accepted scope still excludes bridging, wrapping, withdrawal, custody,
  tax, failed operations, and the exact best eligible alternative yield. The
  BTC wallet still lacks split-origin
  lineage, while the ETH and SOL cases close that limitation for their public
  wallets. None proves future rewards; the rate is discretionary and caps may
  be introduced. No deployment, account, funding, transaction, paper, or live
  authority exists. Continue public monitoring without touching Round 21.
- The active Binance USD1/WLFI holding airdrop is accepted only for already-held
  idle USD1 in a published eligible account category. The current realized base
  APR is 5.46%; the 6.55% boost is creditable only when at least 300 USD1 of
  lowest-hourly daily Futures open interest already exists for an independently
  justified organic strategy. Simple Earn is not in the closed eligible-account
  list, so the same principal cannot receive both yields under the published
  contract. Credentials, region eligibility, exact account rates, reward-sale
  costs, and USD1 principal risk remain unresolved. Do not acquire USD1 or open
  Futures exposure for this promotion. Refresh only on the fixed 2026-08-27,
  2026-08-28, and 2026-09-04 triggers. Canonical gate:
  `binance-usd1-wlfi-holding-airdrop-gate-v1-2026-08-26.json`, result SHA-256
  `c67367932b440d6f4a23330a17c405c0e15b0021b0484575a0b0efcc6e9238a6`.
- The Binance U Flexible promotion is accepted only for already-held idle U in
  an eligible non-EEA master account: regular users receive a public headline
  8.5% APR on the first 5,000 U through 2026-09-14. Buying U for the promotion
  is rejected because only 10.3422 bips remain after the current USDT
  alternative, displayed zero-fee spread, and worst observed 19-day close move
  before issuer and account risks. The issuer's terms give ordinary secondary
  holders no direct redemption right or reserve claim, and its homepage lists
  only a December 2025 attestation. Do not infer region from the host timezone,
  acquire U for the reward, or call the promotion stable or deployment-ready.
  Canonical gate: `binance-u-flexible-idle-holding-yield-gate-v1-2026-08-26.json`,
  result SHA-256
  `1dd11be5bb81d7d9bf278a27cdb0df4dc25a1db41c3bce005308c7aa639cbf25`.
- The RWUSD VIP extension is accepted only as an automatic bonus on RWUSD
  already held for an independent reason: 1.1452 bips over 22 days for VIP 1-3
  up to 30,000 RWUSD or 4.5808 bips for VIP 4-9 up to 200,000 RWUSD. Do not
  subscribe to RWUSD or chase VIP status for this bonus. Exact account quota,
  alternative yield, and USDC redemption costs remain behind the existing
  signed prequalification gate. Canonical gate:
  `binance-rwusd-existing-vip-bonus-overlay-gate-v1-2026-08-26.json`, result
  SHA-256
  `076f428ea9bccc0dc9c1a0c605ac469db27fedb7941ac6728260cf98da667e51`.
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
| Post-observation maker window | `docs/model-research/action-value/polymarket-post-observation-maker-window-gate-v1-2026-08-26.json` |
| LDUSDT margin yield | `docs/model-research/action-value/binance-ldusdt-margin-yield-gate-v1-2026-08-26.json` |
| USD1 holding airdrop and Simple Earn allocation | `docs/model-research/action-value/binance-usd1-wlfi-holding-airdrop-gate-v1-2026-08-26.json` |
| U Flexible idle-holding yield | `docs/model-research/action-value/binance-u-flexible-idle-holding-yield-gate-v1-2026-08-26.json` |
| Existing RWUSD VIP bonus overlay | `docs/model-research/action-value/binance-rwusd-existing-vip-bonus-overlay-gate-v1-2026-08-26.json` |
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
