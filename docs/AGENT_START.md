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
| Accepted edges | Nineteen scoped structural edges: Polymarket complete-set holding yield for existing idle on-platform pUSD; Polymarket pUSD taker-fee rebates only for independently justified legitimate organic BTC/ETH/SOL taker flow after the direct-wallet tier is effective; Polymarket builder fees only on bona fide independently existing third-party orders routed through an owned app with an account-confirmed active disclosed positive rate; Polymarket referral rewards only for authentic external referrals when the account already independently cleared the lifetime-volume threshold; Polymarket Perps referral fee share only for authentic external traders within account-confirmed available invites and without volume-based invite unlocking; Binance Soft Staking yield for already-held idle non-order ETH/SOL Spot inventory; LDUSDT yield only for already-required futures collateral; just-in-time BNB fee reduction only for independently justified organic Spot flow; current quote-native BTC/ETH/SOL promotional fee reduction without quote acquisition or extra volume; the current TradFi perpetual zero-maker and reduced-taker fee overlay only for independently justified organic flow with the exact current symbol and actual fill role; the current Binance Stocks promotional trading-spread reduction only for independently justified organic direct-stock flow with the exact previewed order tier and realized fee; USD1/WLFI holding-airdrop yield only for already-held eligible USD1; the fixed USD1 Simple Earn bonus only on the first 1,500 independently already-held idle USD1 when its mutually exclusive balance-specific route beats the holding airdrop after all transition and opportunity costs; U Flexible yield only for already-held eligible non-EEA U; the automatic RWUSD VIP bonus only on independently required existing RWUSD; current USDT Flexible bonus yield only for independently held idle eligible USDT; the current automatic USDe holding reward only for eligible USDe already independently held on Binance for at least 24 hours; Binance Square's base 20% Write to Earn commission only on authentic external readers' independently existing eligible fee-bearing trades attributed to genuinely useful content; and Binance Referral Pro's base 20% Spot/Margin and 10% one-year Futures fee commission only for authentic independently acquired new external users. None is deployment-ready or fully account-and-external-cost-qualified. |
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

- The one-use Polymarket holding-yield post-conflict refresh is consumed and
  must not be rerun. It failed closed after five of nine public requests on an
  unstated exact-current-value gate: SOL's equal mergeable 591.11-share pair
  displayed separately rounded values totaling 591.1099. Retained BTC and ETH
  rows nevertheless each uniquely match 24 sampled hours at 3.25%, not 4%, for
  the first wholly post-conflict interval. Because SOL activity and all three
  receipts were not requested, the current rate remains unqualified. A repair
  capture cannot change the frozen v5 economic rejection versus a 3.25%
  alternative, so reopen only after a material rate, program, payout, or
  comparator change. Canonical adjudication:
  `complete-set-holding-yield-post-conflict-v7-failure-adjudication-2026-08-29.json`,
  SHA-256
  `448b068aa5c1b34c6012a5fadafa449ed9ef125afc310b7901b9f68285510f71`.
- Binance Stocks FPSL is a materially distinct direction-independent overlay
  for U.S.-listed stocks or ETFs that are already independently owned and fully
  settled. The current FAQ says actual loaned shares accrue interest daily and
  distributions arrive monthly while selling remains available, but there is
  no loan guarantee and the public sources omit the account's annualized rate
  and fee-share percentage. The public forward income floor is therefore zero.
  Loaned shares are not SIPC-covered, voting rights are lost, and dividends
  become cash-in-lieu. Do not buy stock or enable FPSL. Read-only account
  evidence requires explicit authority; enabling FPSL requires a separate
  explicit account-state authorization. Canonical result:
  `binance-stocks-fpsl-existing-inventory-yield-overlay-candidate-v1-2026-08-27.json`,
  SHA-256
  `3fe1801a6cbf442ab1ce79d1f3bd4586542d97414aea954b0bbd9a55a85453e1`.
- The current widest public structural reward lead is Binance's first
  U.S.-stock transfer-in program for independently held inventory. Its fixed
  bonuses equal 250, 150, 66.67, 40, 25, and 20 bps at the six tier thresholds;
  the first three retain 192.47, 92.47, and 9.13 bps after an illustrative 10%
  annualized 21-day liquidity hurdle. It is not accepted: the 300,000-USDC pool
  is first-come, account eligibility and region are unknown, the program excludes
  U.S., U.K., EEA, and other restricted users, transfer settlement generally
  needs at least 14 business days, and the bonus has a 21-day transfer-out
  restriction. Do not buy or transfer stock. Account eligibility needs explicit
  read-only authority; submitting any transfer needs separate high-impact
  authority. Canonical result:
  `binance-existing-stock-transfer-reward-overlay-candidate-v1-2026-08-27.json`,
  SHA-256
  `3ecb4f39848719f788b6853bd90120d1809379b8d81b5419da4b1bbc957fec3d`.
- Binance's time-limited bStock Spot LP promotion adds bStock maker-share
  thresholds of 0.05%, 0.10%, 0.30%, and 0.60%; tiers 2 through 4 advertise
  0.4, 0.6, and 0.8-bp maker rebates across all symbols when the bStock tier
  exceeds the original tier. The first effective week starts
  `2026-09-01T00:00:00Z`. This is not accepted or profitable evidence: exact
  account tier, denominator, organic volume, fills, hedges, and realized rebates
  are absent. Do not fetch books, generate volume, apply, or place orders.
  Canonical result:
  `binance-bstock-spot-lp-all-symbol-rebate-overlay-candidate-v1-2026-08-27.json`,
  SHA-256
  `d279f8ab88875c812e6691fa500fdfde741f2e2fbca19ee240b4c0d4a579d607`.
- A one-use public Lite Loan/stablecoin-yield screen found one narrow,
  time-limited USD1 candidate and rejected U and plain USDT. After the frozen
  worst 30-day USD1 close decline, the USD1 route retained only 1.1079 to
  1.3278 bps at 100 to 1,000 USDT loan sizes. It is not accepted, stable,
  profitable, or deployment-ready. The offer ends at
  `2026-08-27T23:59:59Z`; do not borrow, convert, subscribe, or repay without
  separate explicit funded authority. Exact account eligibility may be checked
  before expiry only with both credentials and explicit signed GET-only
  authority. Canonical result:
  `binance-lite-loan-stablecoin-yield-curve-v1-2026-08-27.json`, result SHA-256
  `65f223a245fa1bb65a8fd791275da0dbd71d3c52ee2d232ac1420feb198b129d`.
- A one-use public broad-crypto funding-carry preflight corrected the earlier
  BTC/ETH/SOL-only research boundary without changing execution scope. It
  deterministically selected 17 exact Binance Spot/USDT and USD-M USDT crypto
  perpetual pairs by the smaller current 24-hour leg volume, retained 23 raw
  public responses plus a durable journal, and found zero funding-only passes.
  Every training, validation, and test role was negative after the frozen
  32-bps round-trip stress and 10% annual opportunity cost on each of two
  capital legs; every family-adjusted bootstrap lower bound was negative. Do
  not resample the current-liquidity-selected history or relax its gates.
  Canonical result:
  `binance-broad-crypto-funding-carry-preflight-v1-2026-08-27.json`, result
  SHA-256 `095009a36a5c6a8a5a2dfdfb3e57ebe6183721bb84600518552ccf6d463617c8`.
- A materially distinct Binance-Hyperliquid cross-venue funding-spread
  extension exactly reproduced the public June 2026 study archive and added a
  frozen 70-day 2026-06-18 through 2026-08-26 public extension for BTC, ETH,
  SOL, and DOGE. All four assets had complete 1,680-hour venue coverage, but
  after 20 bips round trip and exact entry/exit premium-basis drift their APRs
  were only 0.23645%, 2.16957%, 1.27718%, and 1.17155%; the primary BTC/ETH
  basket was 1.20301%. Every asset and the basket failed the same-timestamp
  3.86% DGS3MO hurdle. This direction-neutral family is rejected without
  refitting or resampling; reopen only after a material venue-fee, funding,
  basis, or hurdle change. Canonical result:
  `binance-hyperliquid-cross-venue-funding-spread-extension-v1-2026-08-27.json`,
  SHA-256 `23eb54dfd19890d984d73156ef05950f7362f8fffe081b93cc5d471f59f62755`.
- Official Binance Options RFQ material proves a materially different
  execution path for predefined same-expiry two-leg call and put spreads: the
  RFQ bypasses the public book and all predefined legs execute together. It
  does not prove a four-leg box, documented RFQ API, account eligibility,
  minimum quantity, exact quote cost, or profit. The displayed-book vertical
  and box screens remain terminal; only the two-leg vertical execution
  architecture is reopened. Request no quote without explicit quote-request-
  only authority, and never confirm or execute without separate authority.
  Canonical triage:
  `binance-options-rfq-fixed-payoff-execution-triage-v1-2026-08-27.json`, result
  SHA-256 `64943efe0c6ad16f8d02f78548afef38f919448d2da87c7573e825a2eeefd6b9`.
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
- The current logged-out Binance Earn pages expose a materially stronger but
  still unaccepted liquid-staking lead for independently idle inventory: ETH
  Staking displays 2.20% and SOL Staking 4.65%, respectively 170 and 415 bips
  above the 0.50% Soft Staking comparator. Those uplifts can absorb at most
  13.97260274/34.10958904 bips over 30 days or 41.91780822/102.32876712 bips
  over 90 days before becoming nonpositive. Exact same-account conversion
  ratio, quota, commission, redemption period, owned reward lineage, delay,
  liquidity, and alternative-yield costs remain absent, so the public net
  profit floor is zero. Do not stake or redeem. Canonical candidate:
  `binance-existing-idle-eth-sol-liquid-staking-yield-candidate-v1-2026-08-27.json`,
  SHA-256
  `b7fc84d0be3968d31afeb801b7a40ee0d382724b11281c28733a8145d12ee035`.
- The adjacent delta-neutral ETH/SOL Soft Staking plus short-USDT-perpetual
  stack is terminal under the retained 500-row funding histories. The first
  local calculation correctly stopped after discovering that the response had
  500 rows rather than the requested limit of 1,000; the final chronological
  roles were therefore derived from the actual response count as 300/100/100.
  At 0.50% Soft Staking APR and 32 bips round-trip execution stress, ETH still
  fails training and SOL fails training before opportunity cost. Every role for
  both assets is negative after even one 10% annual capital-leg hurdle. The
  maximum required APR across roles is 0.59006995% for ETH and 3.73322000% for
  SOL after execution stress alone, or 10.59006995% and 13.73322000% after one
  capital hurdle. Do not repeat unchanged history or acquire inventory for this
  stack. Canonical terminal result:
  `binance-soft-staking-delta-neutral-funding-stack-terminal-v1-2026-08-27.json`,
  SHA-256 `591fb98b9a8e58365c67c4a281d1fda3de674b42f1f868a42d98acf2ab19ae68`.
- BTC Simple Earn is a new extension candidate for that same idle-native-token
  family, not a tenth accepted edge. The current official page shows BTC
  Flexible at `0.27% Max` with an APR breakdown of `0.02% + 0.25%`; it does
  not expose the minimum, bonus cap, bonus end, account eligibility, or fees.
  After requesting the public one-year calculator control, visible estimates
  were 0.00002172 BTC on 0.01 BTC, 0.00003997 on 0.1 BTC, 0.00022247 on 1 BTC,
  and 0.00202922 on 10 BTC. This confirms that the maximum is not a whole-
  balance rate but cannot prove the hidden tier contract. Do not buy or retain
  BTC, reverse-engineer the cap, or subscribe. Exact signed product, position,
  and reward-history GETs are the next evidence gate only after both ephemeral
  credentials and explicit read-only authority exist; every subscription or
  redemption still needs separate funded authority. Canonical candidate:
  `binance-btc-simple-earn-idle-yield-candidate-v1-2026-08-26.json`, SHA-256
  `193495029148d0022fe1bf4158442226705a7f62a22dbf0eafdbf9a53bece785`.
  The same artifact rejects the adjacent current `0.2%~0.41%` locked BTC
  On-Chain Yields headline: Binance classifies it as high risk, makes rewards
  protocol-dependent and unguaranteed, and says protocol failure can lose
  assets. Do not trade a higher headline for an unbound principal-loss path.
- The current logged-out VIP Earn page now exposes exact BTC, ETH, and SOL APR
  ranges, so absence of credentials no longer justifies repeating that public
  lookup. Zero in-scope row has positive displayed maximum uplift: BTC VIP
  `0.25%~0.41%` equals the best visible non-VIP maximum, ETH VIP
  `1.70%~1.90%` trails ETH Staking at 2.20%, and SOL VIP `3.78%~4.50%`
  trails SOL Staking at 4.65%. A 2025 locked-products PDF is 441 days before
  the 2026-08-12 VIP Earn launch and is inadmissible as current evidence unless
  a current official page adopts it. Reopen only on a material current terms
  change or explicit signed GET-only account evidence. Canonical terminal
  snapshot:
  `binance-vip-earn-public-btc-eth-sol-comparator-terminal-v1-2026-08-27.json`,
  SHA-256
  `cd41cad8e0053b9d41ddda64fd4ad8a86a163307ddcc9fabc805c56b9c5028c9`.
- The 2026-08-25 target-free structural-parity screens found no accepted edge.
  Polymarket had zero gross-positive paths across 22 fixed BTC/ETH/SOL
  negative-risk events and zero gross-positive logical-implication bundles
  across 2,572 threshold/deadline pairs. Negative-risk conversion is no longer
  globally terminal as a research family: the new primary paper
  `arXiv:2608.00666` reports 36 adapter-supported positive NO-side CLOB episodes
  and approximately 1.086 million USDC of historical converter-linked estimated
  profit. That estimate mixes realized proceeds with imputed residual and merged
  inventory, so it is not current cash-realized after-cost profit. A retained
  2026-08-09 replay for the fixed three-outcome `Bitcoin vs. Gold vs. S&P 500 in
  2026` event initialized all six books and evaluated 796 exact
  received-timestamp-batched five-share states; zero was gross positive. The old
  one-state snapshot remains terminal only under its exact contract. A completed
  five-minute public event-time capture then screened all 22 fixed events and
  found one gross-positive source frame for the same three-outcome event at five
  and twenty shares. The two positives were two size evaluations of one frame,
  not recurrence. The all-taker paths were `-0.07082` and `-0.28328` pUSD after
  current fees. A Bitcoin-NO maker input at 0.82 left only `0.00740` and
  `0.02960` pUSD before conversion and external costs. An exact recent Polygon
  receipt independently proves 66.72 Bitcoin-NO shares filled as maker at 0.82,
  and official source plus one exact successful conversion prove the current
  V2-collateral-adapter-to-legacy-adapter route. That conversion used all three
  NO positions (`indexSet=7`), not the candidate one-NO route, and its 479,446
  gas units cover the whole outer transaction. Reusing those units at one
  current Polygon gas recommendation and executable POLUSDT ask makes the
  five-share margin negative and leaves only `0.0092570280188902383064000`
  pUSD minus a USDT sensitivity at twenty shares before every other cost; USDT
  is not assumed equal to pUSD. That adapter-address conflict is now resolved:
  the current official Contracts registry explicitly declares itself the single
  source of truth, labels `0xd91E80...35296` as the deprecated CLOB-v1 adapter,
  and lists `0xadA200...6eAab` as the current pUSD NegRisk collateral adapter;
  the dated changelog says the V1 relayer route was fully retired on 2026-07-17.
  The current official source at commit `ccc0596` confirms that the V2 wrapper
  invokes the legacy adapter internally. This removes only the address-identity
  blocker. No source-bound post-fill Gold-YES/S&P-YES books, queue ownership,
  candidate index-set account access, exact user gas/relayer charge, latency, or
  after-cost profit exists. Both
  documented historical-price endpoints returned points outside their requested
  time window and were rejected. Canonical maker-input gate:
  `polymarket-negrisk-maker-input-gate-v1-2026-08-26.json`, result SHA-256
  `d4e02d2d1cc6b0a598265af734b29f62aec6145bc5a1cc3b3d65771ba2031d2a`.
  Canonical address resolution:
  `polymarket-negrisk-v2-adapter-address-resolution-v1-2026-08-27.json`, result
  SHA-256 `e11810a0215521cb5ad0c0c966340b4ff943760fda516e7841430fe057fe25fe`.
  Its predecessor is the recurrence gate:
  `polymarket-negrisk-converter-recurrence-gate-v1-2026-08-26.json`, result
  SHA-256 `ff8b2eddeaab155327ad0d1542c0b75602342b45571443a4de61f8904165f030`.
  One frozen 24-hour public capture of only the six event tokens was launched
  under contract SHA-256
  `9d32e66b6d150434e4b978daafa1ea9482066230f253da4c86eb9a18504717da`.
  Verify its process and terminal file before acting; never use its partial raw
  file, restart it, or start a duplicate. It may only decide queue-censored fill
  and causally subsequent unwind feasibility, not trading authority.
  A separate current organic-taker overlay is accepted only as a pUSD fee
  reduction on independently justified legitimate BTC/ETH/SOL taker flow. One
  complete public UTC day contained 1,202 BTC/ETH/SOL and other crypto taker
  trades whose current crypto fee curve reconstructs 302.8176185015 pUSD; the
  next on-chain `TAKER_REBATE`
  payment was 54.5062 pUSD, matching the documented Gold 18% rate within 0.001
  pUSD. At a 0.50 entry, the 0.07 fee curve is 3.5% of trade notional, so Gold
  saves 63 bips; do not misread 1.75 pUSD per 100 shares as 1.75% of the 50
  pUSD trade. The fee page retains a generic `USDC` label, but current V2 source
  defines fees in exchange collateral and the deployed exchange returns the
  official pUSD proxy, whose on-chain symbol is `pUSD`; no parity assumption is
  used. Current terms conflict between immediate threshold activation and
  the next daily update, so credit only an account-confirmed tier after the
  completed daily update. Never manufacture volume or count a level-up bonus.
  Canonical overlay:
  `polymarket-organic-taker-rebate-overlay-v1-2026-08-26.json`, SHA-256
  `6a3f907dbebd0c7cc894d95054231540e50cd8e28e6264840a2840be8ac72865`.
  A retained-state-only NegRisk overlay proves the queue-free all-taker route
  still fails: it needs a 58.6161231584% rebate before external costs, above the
  documented 50% maximum, which leaves `-0.010410` and `-0.041640` pUSD at five
  and twenty shares. Gold would raise the maker-input twenty-share margin from
  `0.02960` to `0.096272` pUSD before external costs, but does not prove the
  wallet tier, queue fill, conversion, subsequent books, or same-unit costs.
  Canonical non-promoting overlay:
  `polymarket-negrisk-taker-rebate-overlay-v1-2026-08-26.json`, SHA-256
  `fbbaf4ff7a7d93f8cf5d306a829ff00518d82c9802be674fdace864cea907a60`.
  A distinct retained Round 27 binary complete-set rescore exactly reproduced
  the zero-tier latency baseline, then found that Gold's one sub-unit sequential
  minimum was ex-post ordering hindsight, not causal. Diamond is the first tier
  with one source-time lower-cost-leg-first historical survivor at `0.98407728`
  pUSD per complete set, or `0.07961360` pUSD at five shares before external
  costs. Zero episodes survived the venue-delay simultaneous check or both leg
  orders, and the opposite order cost `1.09417472`. This is a historical
  candidate only. Do not replay or refit it; retry only after a direct wallet
  confirms Diamond or Obsidian after a completed daily update, then preregister
  one current prospective causal capture. Canonical overlay:
  `polymarket-round27-complete-set-taker-rebate-overlay-v1-2026-08-26.json`,
  SHA-256
  `948f47d9d0c2fb6cbf441da1147ae07006a897f307141dfd6ae25c85e47f13d2`.
  The current structural-edge registry accepted count is seventeen. Binance's
  best three-leg spot cycle
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
  The old static program document returned 403, but the current dynamic official
  fee page now source-proves enrollment at zero maker fee and higher tiers of
  `-0.0040%`, `-0.0060%`, and `-0.0080%` after weekly maker-volume-share hurdles
  of `0.15%`, `0.50%`, and `1.00%`; tier 1 also publishes `0.05%` share or
  `25,000,000` USD weekly volume. This closes the public-rate gap only. Both
  designated ephemeral variables remain absent and no signed request was sent;
  account enrollment, organic qualifying volume, owned fills, queue/adverse
  selection, inventory unwind, and realized rebates remain unproved. Do not
  manufacture volume for a maximum displayed rebate of only 0.8 bps.
  Canonical account gate: `binance-spot-maker-rebate-account-evidence-gate-v1.json`,
  result SHA-256
  `d2adda1c5ab4b561e0c238e1e874cc72edaee15ebadafbb76703251f9cd99e10`.
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
  A second, non-overlapping Binance fee overlay is accepted for already-intended
  quote-native BTC/ETH/SOL Spot flow. The current official table lists zero
  maker fees on all six FDUSD and U pairs; VIP2-9 additionally receive zero U
  taker fees and zero maker/taker fees on the three USD1 pairs. BTC/ETH/SOL USDC
  remain in the all-user taker promotion: a regular user pays `0.09500%` instead
  of `0.100%`, or `0.071250%` instead of `0.07500%` with BNB. The deterministic
  filled-notional saving is therefore 10 or 7.5 bps for a regular zero-maker
  fill and 0.5 or 0.375 bps for a regular USDC taker fill. This does not justify
  acquiring a promotional quote asset, chasing VIP status, changing execution
  role, or creating volume; spread, basis, conversion, queue, fill, settlement,
  and opportunity cost remain with the independently profitable strategy. Never
  credit BNB against a zero commission or double-count any adjustment. Canonical
  gate: `binance-spot-promotional-fee-overlay-v1-2026-08-26.json`, result SHA-256
  `f951d167b3abbb89afc39a29671b9a4cb6929661f13a957e553a8fad439ce9e6`.
  Both credentials remain absent, so account/region eligibility and exact-order
  commission evidence remain unproved and no signed or funded action occurred.
  The same current official fee surface exposes an additional accepted scoped
  overlay for independently justified organic TradFi perpetual flow. Every
  displayed regular/VIP tier currently has `0.0000%` maker fees, and displayed
  taker fees are reduced to `0.0400%` through `0.0085%` before the separate BNB
  discount. Against the simultaneously displayed standard USD-M USDT table,
  positive comparator savings range from `18` to `200` USD per `1,000,000` USD
  of notional. This is a fee overlay only: refresh the exact fee table before
  every otherwise authorized order, credit zero maker fees only to an actual
  owned maker fill, and never change price, role, or volume to chase the rate.
  The public table exposes no promotion end, exact account commission and all
  underlying strategy costs remain mandatory, and no order authority exists.
  Canonical edge:
  `binance-tradfi-perpetual-current-fee-overlay-edge-v1-2026-08-27.json`,
  result SHA-256
  `705cb3da615c1873623e7f5be31f0d8cf672c3db9635a5ba971407cf6e715b6c`.
  A distinct current fee-overlay candidate is Binance's `VIP 6 for Six`
  promotion. The public Growth Track is for current Binance VIP1-5 users who
  can verify VIP3+ on another exchange; the Reactivation Track is for users who
  held Binance VIP3-6 in 2025 and can be applied automatically by email. At the
  current published fee table, moving to VIP6 saves from `15` to `710` USD per
  `1,000,000` USD of already-intended fee-bearing flow across the scoped Spot
  and USD-M tier, maker/taker, and BNB-discount examples. This is a material
  candidate, not an accepted edge: the logged-out public floor is zero because
  this account's track eligibility, selection, approval, exact effective
  interval, exact symbol commissions, future organic flow, and incremental
  costs are unknown. Never create volume, borrow BNB, apply, contact an account
  manager, or disclose external exchange records without the separately
  required authority. Canonical candidate:
  `binance-vip6-for-six-organic-fee-overlay-candidate-v1-2026-08-27.json`,
  result SHA-256
  `f638cb6f565c1ee18c9dc065c5f4fc6506442f00833193d23c287bdf9d8ec74d`.
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
  the worst role. The official live product page displays only 5.03% seven-day
  average APR and 5.12% last-day APR. It now states zero purchase fee, Fast
  Redemption free for the first 500 BFUSD then 0.1%, and two-day Standard
  Redemption at 0.025%. Those terms improve the older guide sensitivity, but
  even fee-free-under-500 collateral misses the fixed-role gate by more than
  8.98 percentage points. Do not request credentials, accounts, funding, or
  books unless a materially new official displayed APR first reaches
  14.1066194737%. Exact rate, reward, and quota evidence remains signed
  USER_DATA. Canonical gate:
  `polymarket-perps-binance-bfusd-collateral-stack-gate-v1-2026-08-26.json`,
  result SHA-256
  `a6ff387d70d33c40951e36de93eff7c810b2291dbefff5ecb0f3953880fe7878`.
  Current public promotions add one conditional path, not a stable accepted
  edge. Binance published an 8.07% effective APR for the first completed week
  of its RLUSD/XRP campaign. It has no stated individual cap, but requires an
  eligible account, RLUSD collateral, and at least 500 USD average daily genuine
  Margin or Futures volume; future weekly APRs are unknown and the campaign ends
  2026-09-11. Never manufacture volume to qualify. The contemporaneous USDT
  promotion has a source-bound 14-day window and maximum combined gross reward
  of about 1.34 USDT at the advertised approximate base rate and capped bonus.
  The same article's USDC table row displays approximately 2.5% Real-Time APR
  plus 5% on the first 200 USDC, but the dated promotion sentence names only
  USDT. No USDC start or end time is published there, so its guaranteed forward
  public reward floor is zero; never transfer USDT's dates to USDC. Canonical
  triage:
  `binance-public-promotion-yield-triage-v1-2026-08-26.json`, result SHA-256
  `26efd481a5ff424ca17ec803bb6a1a3ae8949d1fe0fc31a03e20a35d08d031ac`.
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
  The one permitted post-activation refresh confirmed the same 7% bonus terms
  and a 0.10001-bip public top-book spread, but the frozen conservative margin
  remains only 0.96835 bips before exact account costs, peg risk, and redemption
  risk. It remains active but unaccepted and not stable. Canonical activation
  refresh: `binance-usd1-simple-earn-activation-refresh-v1-2026-08-27.json`,
  result SHA-256
  `f8106a93155813a3130bc925a3f4b223fad16b6133ee073226251d25175ecf06`.
  Do not assume public Convert bounds, a displayed spread, or an issuer
  redemption claim is an executable fee-free round trip.
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
  A distinct 15-minute/4-hour August crypto-TWAP liquidity-reward screen then
  corrected the old daily-rate mistake before capture: every daily reward
  equivalent must be prorated to the exact market lifetime, and only a fixed
  100-times competition-stressed full-market reward may cover the maximum
  orphan loss. Its one attempt again stopped after Gamma and the first exact
  BTC 15-minute reward GET because exact reward identity was not proved. No
  books were requested. The transient payload was lost because the collector
  embedded sources only on success, repeating an already documented workflow
  defect. The terminal receipt is
  `crypto-twap-liquidity-reward-screen-attempt1-failure-v1.json`, result SHA-256
  `e486f2928a326e6829cbe3c07aad5a47bb25a63783a935273606df00cea98c66`.
  The collector now journals every public response before validation and a
  focused test enforces that correction. Never retry this exhausted window or
  infer whether the lost response was empty, duplicated, or mismatched.
  The current official program later added an exact 550,000 dollar five-minute
  allocation across BTC, ETH, SOL, XRP, HYPE, BNB, and DOGE, which materially
  reopened only that previously excluded duration. One exact seven-market
  source screen started at 2026-08-27 06:45:05 UTC and retained both public
  responses before validation. All seven exact markets existed with a
  50-share reward minimum, 4.5-cent maximum spread, and identical taker-only
  0.07/exponent-1/0.2-rebate fee schedules, but raw Gamma omitted the optional
  `clobRewards` field on every row. The screen stopped before books because no
  exact dated per-market daily allocation was available. Its terminal artifact
  is `crypto-twap-5m-liquidity-reward-screen-attempt1-terminal-v1-2026-08-27.json`,
  result SHA-256
  `319c6aedbb5491e56e68cc3fdf95f366766ce4a070dcadec3213471ff938120d`.
  Do not infer a daily rate from the monthly program cap or retry the same
  source configuration. The current SDK model documents `clobRewards` as
  optional; never treat a normalized SDK field as guaranteed raw Gamma output.
  The current official SDK then exposed a materially distinct unsigned public
  `/rewards/markets/current?sponsored=true` population. A frozen 07:20:05 UTC
  join retained all 54 rows through the terminal `LTE=` cursor and found zero
  exact condition matches for the same seven five-minute markets. It again
  stopped before books. Canonical result:
  `crypto-twap-5m-current-rewards-list-join-v1-2026-08-27.json`, SHA-256
  `62940fa602d71259aab1326eb038069b23df6f247dd898a952f493fcebc38e6f`.
  Both current public allocation-source paths are terminal; do not repeat them
  absent another program change or a genuinely new exact per-market dated
  allocation response.
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
- The mutually exclusive USD1 Simple Earn route is now separately accepted only
  for at most the first 1,500 USD1 that is independently already held idle. The
  logged-out public product page displayed `8.62% Max`, decomposed as a variable
  `1.62%` Real-Time APR plus a fixed `7%` promotion bonus. The fixed bonus alone
  exceeds the current `5.46%` holding-airdrop base by 154 annualized bps; after
  forfeiting one airdrop day, its gross break-even is 3.5455 days. At a 1,500
  USD1 balance, immediate fixed-bonus allocation beats the sensitivity of
  waiting through the airdrop and then subscribing by only 0.5063 USD1, or
  3.3753 bps. Never count both rewards on the same principal or infer that the
  displayed USD1/U/USDC/USDT rate ordering justifies a conversion. Exact account
  eligibility, capacity, liquidity need, transition cost, tax, and redemption
  timing must all be proved before any subscription, which requires separate
  authority. Canonical adjudication:
  `binance-usd1-simple-earn-versus-holding-airdrop-allocation-edge-v1-2026-08-27.json`,
  result SHA-256
  `a4158bf059f4f5ad839b2f504c08c4afc65615260b4171533866f4c2337494e0`.
- The current first-USD-deposit Promotion A is a distinct high-margin but
  unaccepted action-gated candidate for a genuinely first-time eligible user.
  It advertises a 15 USD-equivalent SPCXB voucher to the first 1,000 users who
  register, deposit at least 100 USD, and complete at least 200 USD of eligible
  trade volume; distribution is within 30 hours after both tasks. A frozen
  public one-way purchase of 201 USDT cost 200.93166 USD and the theoretical
  reward had 15.6135 USDT of displayed liquidation value. After labeled 10-bp
  task and reward-sale fees, a 20-bp round-trip SPCX hedge fee, and four funding
  intervals at the worst short-pay rate in the latest 20 rows, 15.3490 USDT
  equivalent remained for every deposit, bank, FX, withdrawal, tax, basis, and
  operating cost. This is not profit: first-time status, jurisdiction, first-
  come capacity, deposit fees, task completion, exact entitlement, rounding,
  and future reward value are account-specific. Never pre-hedge an unproved
  reward or manufacture round-trip volume. Every registration, BPay activation,
  deposit, trade, hedge, claim, sale, and withdrawal requires separate explicit
  authority. Canonical candidate:
  `binance-first-usd-deposit-spcxb-reward-hedge-candidate-v1-2026-08-27.json`,
  result SHA-256
  `e0b6ed9311d2a022abee417a677b952e83cf918fc6b396804f5cba39fd83d4ed`.
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
  `6f44b65e5aa85d33cc02e8611a372162cf00f4162fdff99828a31cf498ced6f9`.
- The RWUSD VIP extension is accepted only as an automatic bonus on RWUSD
  already held for an independent reason: 1.1452 bips over 22 days for VIP 1-3
  up to 30,000 RWUSD or 4.5808 bips for VIP 4-9 up to 200,000 RWUSD. Do not
  subscribe to RWUSD or chase VIP status for this bonus. Exact account quota,
  alternative yield, and USDC redemption costs remain behind the existing
  signed prequalification gate. Canonical gate:
  `binance-rwusd-existing-vip-bonus-overlay-gate-v1-2026-08-26.json`, result
  SHA-256
  `076f428ea9bccc0dc9c1a0c605ac469db27fedb7941ac6728260cf98da667e51`.
- Current USDT Flexible bonuses are accepted as one scoped, time-limited
  same-asset yield family only for independently held idle USDT in an eligible
  verified master account. The global offer adds 4% Bonus APR to the first 500
  USDT through 2026-09-07; subscribing before 2026-08-27 exposes at most 12
  accrual days, or 13.1507 bips and 0.6575 USDT. Mutually exclusive new-user
  variants offer 30% for seven days on the first 200 USDT in a published LATAM
  country list, or 15% in Brazil. Never assume those offers stack, infer region,
  register, deposit, acquire USDT, or move prompt-liquidity/collateral principal
  to chase them. Canonical gate:
  `binance-usdt-flexible-current-bonus-overlay-v1-2026-08-26.json`, result
  SHA-256
  `0126a1feef4e8bb5c46a7b7cab45e2471857a2e105fe0f41d73d4710b6abceda`.
- The current Binance USDe automatic holding reward is accepted only as a
  same-token gross increment on eligible USDe already independently held on
  Binance for at least 24 hours. The current reference APR is 4%, producing
  1.09589041096 bips per eligible day or 0.7671232877 USDe per 1,000 over seven
  days. The campaign is ongoing until further notice, uses a random daily
  minimum-balance snapshot, and pays automatically each Monday. Do not acquire,
  deposit, convert, borrow, or retain USDe for this reward; do not treat the APR
  as guaranteed fiat return or principal protection; and do not double-count
  collateral use or separate Ethena yield. Exact KYC, jurisdiction, eligible
  balance, completed holding duration, and owned distribution remain unproved.
  Canonical gate:
  `binance-usde-existing-holding-reward-edge-v1-2026-08-26.json`, result SHA-256
  `4640635514ad43ed846660c204a95c0d59ed75ac3ccbf5f17a0b70f3d5726f6a`.
- The Polymarket Builder Fee mechanism is accepted only as a gross pUSD overlay
  on bona fide independently existing third-party matched flow through an owned
  application with an account-confirmed active, positive, explicitly disclosed
  rate. The official formula is matched pUSD notional times configured bps over
  10,000; current maxima are 100 bps taker and 50 bps maker. Never attach the
  code to operator, related-party, wash, circular, self-referred, or manufactured
  activity, and never create or reroute trades for fees, tiers, grants, or
  rewards. Account eligibility, external flow, owned payouts, demand effects,
  legal obligations, and all operating costs remain unproved; gross fees are not
  after-cost profit. Canonical overlay:
  `polymarket-organic-third-party-builder-fee-overlay-v1-2026-08-26.json`,
  result SHA-256
  `8c070b6a4b07070ffdd5ba703da1ca3788faffcb4d748633a18269dc02c17885`.
- BFUSD existing-holding yield is not accepted. The current product page says
  daily rewards are paid in BFUSD, while the current FAQ and governing terms say
  a USD stablecoin and permit the reward asset to change. The same-unit forward
  floor is zero until an effective source or owned reward history reconciles the
  asset; do not assume one-for-one conversion. Canonical conflict gate:
  `binance-bfusd-existing-holding-reward-unit-conflict-gate-v1-2026-08-26.json`,
  result SHA-256
  `54fe3d3e23a92290debdc67d1e7e19ecac6c06441c045f1aa21fe3e62558c03c`.
- Binance Smart Arbitrage is terminal as packaging of the already rejected
  matched-base spot-perpetual funding family. Official terms retain spot-maker
  and futures-taker entry and exit fees, spread, funding reversal, manual exit,
  basis, and liquidation risk; no distinct fee, execution, or capital subsidy
  was proved. Do not resample Round 61 or treat delta-neutral as risk-free.
  Canonical adjudication:
  `binance-smart-arbitrage-terminal-family-adjudication-v1-2026-08-26.json`,
  result SHA-256
  `03b652fcd7e50c0671abbfb73f68f69509a2e5d7f75d8166f6b74743eab630d3`.
- Polymarket's current referral program is accepted only as a gross pUSD
  overlay for authentic independently acquired external users when the account
  already independently exceeds 10,000 USD lifetime Polymarket volume. Direct
  rewards are 10% and indirect rewards 5% of net fees after the referred user's
  tier rebate, paid daily until the earlier of Platinum or 30 days. Never self-
  refer, use controlled or linked accounts, manufacture qualifying volume, or
  double-count Builder Fees or another reward without explicit combination
  terms. Eligibility, attribution, owned payouts, and all acquisition and
  operating costs remain unproved. Canonical overlay:
  `polymarket-organic-referral-net-fee-overlay-v1-2026-08-26.json`, result
  SHA-256
  `f7aec4a5340cba42abb120a43cda1ed1fa4d5b03632b3c062c0d00d7b5636cf0`.
- Binance Flexible Loan collateral-yield retention is a candidate, not an
  accepted edge. Current official sources say Simple Earn Flexible collateral
  continues earning, but all current asset, rate, LTV, position, income, and
  reward inputs are signed USER_DATA. The designated credentials are absent;
  no signed request was sent and the public after-cost floor is zero. Do not
  borrow, subscribe, repay, adjust LTV, acquire collateral, use leverage, or
  double-count idle yield. Canonical gate:
  `binance-flexible-loan-simple-earn-collateral-yield-gate-v1-2026-08-26.json`,
  result SHA-256
  `ac010265c5236152907ac7b3c12ce13104f473b4cc61c5db43fb8b28c6678182`.
- Binance Advanced Earn Discount Buy and Dual Investment are terminal for the
  market-situation-independent search. Their nominal APR accompanies locked,
  settlement-price-dependent conversion: Discount Buy can use 50% or 100% of
  stablecoin principal to buy crypto at the preset target, while Dual
  Investment Buy Low and Sell High have cash-secured-put-like and covered-call-
  like exposure. This is option-like direction risk, not neutral yield. Do not
  simulate APR, target, or duration grids unless materially new evidence proves
  a complete executable option-equivalent mispricing or external subsidy.
  Canonical adjudication:
  `binance-advanced-earn-conditional-conversion-terminal-adjudication-v1-2026-08-26.json`,
  result SHA-256
  `15f160e3d54f0be09611bb36901b1d9061a2a173643c0562996ecb2824320a3f`.
- Binance Square Write to Earn is accepted only as a direction-independent
  gross USDC overlay at the current base 20% rate on authentic external
  readers' independently existing eligible fee-bearing trades after engagement
  with genuinely useful attributed content. Do not credit conditional 30% or
  50% weekly leaderboard totals, zero-fee or self trades, content older than
  seven days, unattributed activity, or weekly earnings below the 0.1 USDC
  payout threshold. Never manufacture reader activity or encourage unsuitable,
  leveraged, or loss-making trades for commission. KYC and regional eligibility,
  owned attribution and payouts, audience demand, content, compliance, tax, and
  operating costs remain unproved. Canonical overlay:
  `binance-square-organic-write-to-earn-fee-overlay-v1-2026-08-26.json`,
  result SHA-256
  `29ec95146998535fde295dfc830a2639b9d10964e7f9e36c17e44e628dc454d1`.
- Binance Referral Pro is accepted only as a direction-independent gross fee
  overlay at the public base tier: 20% of authentic referred users' fee-bearing
  Spot and Margin fees and 10% of their Futures fees for one year after Futures
  activation. Referral Lite and Pro are mutually exclusive per new user. Do not
  credit higher quarterly performance tiers, self or controlled accounts,
  zero-fee or invalidated trades, restricted regions, prohibited advertising,
  or the same fee under Write to Earn or another commission program. Account
  eligibility, attribution, payout asset and timing, acquisition, disclosure,
  compliance, tax, and operating costs remain unproved. Canonical overlay:
  `binance-organic-referral-pro-fee-overlay-v1-2026-08-26.json`, result SHA-256
  `8a29116879fd90cb0f8fc11d9780a8dccbff8afc2d3ea685e671921f651e64d1`.
- Polymarket Perps referrals are a separate accepted direction-independent gross
  fee overlay: 20% of authentic external referred traders' Perps fees, paid
  weekly, only while the account has a confirmed available invite. Never use
  operator or referred-user volume to unlock the 100, 250, or 500 invite tiers;
  never self-refer, request trades, or double-count prediction-market referral
  rewards or Builder Fees. Account code, available invites, attribution, exact
  fee and payout asset, owned weekly payout, acquisition, compliance, tax, and
  operating costs remain unproved. Canonical overlay:
  `polymarket-perps-organic-referral-fee-overlay-v1-2026-08-26.json`, result
  SHA-256
  `4bebea610dc9406d598627035f4e6e815e6a4daeb64944d7ba2ec9f55b6b7d71`.
- A new primary study materially reopens only the distinct Polymarket live NBA
  full-game moneyline/spread implication family. Across 173 games in February-
  March 2026 it reports 290 active episodes, a 16-second median duration, and
  101.01-bps median yield, but it assumed zero NBA trading fees. Current
  official terms instead list a `0.05` sports taker rate and require each
  market's exact fee schedule. The paper's forward-filled 3.6-to-5.5-second
  books, retail depth, and zero realized middle payouts do not prove current
  profit. This remains an unaccepted candidate. Wait for future active NBA
  full-game markets; then first prove exhaustive same-game payoff rules,
  including integer-handicap push and overtime states, before one synchronized
  public all-taker after-fee recurrence capture. Do not reopen the terminal
  threshold/deadline or single-market complete-set families. Canonical contract:
  `polymarket-live-nba-moneyline-spread-combinatorial-parity-reopen-v1-2026-08-26.json`,
  result SHA-256
  `70cfc7b2ae1cb256e7a8c08c9af33fa8524d2308a8c18400d5a2b7d93c966fe3`.
- A separate primary paper materially reopens exact dependent-subset parity
  across two multi-outcome Polymarket markets, but not its headline profit
  claim. The paper's four numerically enumerated cross-market pairs total
  `95,156.71` USD; its `39,587,585.02` USD headline spans mostly single-
  condition and within-market strategies. It used executed-trade VWAPs, up to
  2.5 hours of forward-fill, no current fees, and semantic dependency labels;
  it also says five cross-market cases but gives only four values. This is an
  unaccepted candidate. Admit a future pair only after a machine-checked
  exhaustive joint payoff table proves exact subset-indicator equality, then
  require one-batch all-leg asks, current per-market fees, displayed common
  depth, synchronization, and every external cost. Do not use LLM or historical
  likelihood as payoff proof, truncate outcomes, or double-count the NBA
  subfamily. Canonical contract:
  `polymarket-cross-market-dependent-subset-parity-reopen-v1-2026-08-26.json`,
  result SHA-256
  `0838bea50b70a8d9e102f40146b2ddf041bc06db3039736d312b9f309c72fc6d`.
- Binance Launchpool is a distinct direction-independent candidate only for an
  independently already-held idle supported stablecoin. Current official
  guidance describes USDC/FDUSD-style pools, hourly rewards, early unlock with
  accrued rewards retained, and principal returned to Spot. The latest concrete
  2026 example used USDC, U, and USD1 for a two-day OPN campaign beginning
  March 3; it is historical, and its exact end timestamp is not stated. The
  current Launchpool page returned WAF-empty HTTP 202, so no active project,
  account eligibility, allocation, APY, owned reward, or executable sale value
  is proved. Do not acquire or redirect principal, poll the empty page, assume
  stablecoin parity, or value token allocation before owned distribution and
  an executable sale. Wait for a new official campaign announcement. Canonical
  candidate:
  `binance-stablecoin-launchpool-idle-inventory-reward-candidate-v1-2026-08-26.json`,
  result SHA-256
  `f898914a56fe61c063ca0eaf8d02fc91ea8bf527dd3ff49289527db524d286c3`.
- Polymarket's Positions Framework exposes a materially distinct exact Boolean
  parity candidate between two underlying CLOB outcomes and Combo RFQ
  positions. For terminal values `A,B` in `[0,1]`, including fractional
  cancellation payouts, `A+B = YES(A and B) + NO(not A and not B)`. Current
  public catalog evidence includes Combo-enabled same-game WNBA legs, but the
  catalog is not a quote and no approved-builder credentials, authenticated
  RFQ, executable CLOB batch, or after-cost recurrence was available. The
  candidate is unaccepted. Do not repeat the terminal broad Combo catalog
  screen or treat implication alone as the identity. Only when approved-builder
  access and explicit quote-request-only authority exist, request minimum-size
  nonaccepted BUY and SELL quotes; inspect CLOB books only if exact RFQ fields
  leave positive conservative headroom. Canonical candidate:
  `polymarket-combo-rfq-boolean-parity-candidate-v1-2026-08-27.json`, result
  SHA-256
  `08fb223f771c5793da944497f37f4067238e7fd2b40fa2427293dbf7b55c4116`.
- The broad sports Combo requester-overround family is separately terminal.
  Never use the Combo positions endpoint's default status set for historical
  return research: it omitted redeemed winners in discovery and created a
  false all-loss sample. Explicitly request
  `RESOLVED_WIN,RESOLVED_PARTIAL,RESOLVED_LOSS`, use
  `gross_entry_cost_usdc` rather than the near-zero remaining
  `entry_cost_usdc` on redeemed winners, and subtract attributed buyer fees
  before interpreting any opposite-side proxy. The corrected unseen ranks
  251-1000 validation lost 73,368.711836 pUSD for the opposite-side gross proxy
  before maker costs and failed PNL-cohort, training, test, wallet-cluster, and
  date-cluster gates. Do not repeat leaderboard mining without a less selected
  population or direct maker quote ledger. Canonical terminal result:
  `polymarket-combo-maker-overround-validation-v1-2026-08-27.json`, result
  SHA-256
  `416daf4d279e06a2353127e642d588a39ae85be0709c2d7498896c1d182847ee`.
- Binance bStock dividend reinvestment versus stock TradFi-perpetual funding
  has one closed family and one materially distinct timing candidate. Historical
  AMAT and MSFT special negative funding debits matched their declared gross
  dividends within three micro-USDT per matched unit. Because bStock
  receives only the net dividend after deductions, direct pre-adjustment long
  bStock plus short perpetual contributes `N-D=-F<0` before every other cost;
  do not repeat it. GLW is different only because its Friday 2026-08-28 ex-date
  precedes the Monday 2026-08-31 bStock snapshot. Do not assume the special
  funding time or credit the gross dividend. After 2026-08-28, check funding
  history once. Only if the actual special debit occurs before the snapshot may
  one synchronized post-adjustment public book batch run after the conversion
  pause and before the deposit-withdrawal pause. The public net-distribution
  floor is zero, so this is unaccepted and not deployment ready. Canonical
  candidate:
  `binance-bstock-dividend-perp-funding-timing-gap-candidate-v1-2026-08-27.json`,
  result SHA-256
  `c073b61271886a5add71c2578caa889dfb97b1245327ae746bd517a91e52530d`.
- A distinct NOK event is the first observed non-US bStock dividend exception
  to the gross-dividend perpetual-debit pattern. Nokia declared a `0.0462` USD
  gross NYSE amount with 2026-07-27 ex-date and 2026-07-28 record date. The
  two negative NOKUSDT funding rows at the Binance NOKB snapshot and eight
  hours later cost a matched short only `0.008865103` USDT per unit. This is
  positive gross upper headroom, not profit: Binance reinvests only the net
  dividend, and the exact historical multiplier increment, entitlement,
  executable books, every cost, and recurrence remain unproved. The current
  multiplier is source-bound at `1.002349416320445721`; it cannot reconstruct
  the historical increment. Do not poll. Reopen only after a new Nokia Board
  resolution states an exact amount and Binance publishes a matching NOKB
  announcement. Canonical candidate:
  `binance-nok-bstock-dividend-perpetual-underdebit-candidate-v1-2026-08-27.json`,
  result SHA-256
  `79118e0e9a32a17d0d79040746068b94e6ec545179958a29dc45f3b8771434bb`.
- A public exact-ticker screen normalized Ondo, bStock Spot, and stock
  perpetual wrappers by each Ondo `sharesMultiplier`. Sixty tickers overlapped;
  41 had a positive point gap and ten were at least 10 bps, led by AXTI at
  `22.2047` bps. These are not executable spreads: official Binance source
  classifies `tokenInfo.price / sharesMultiplier` as a reference value whose
  feeds can update asynchronously, and no Ondo market-order quote was available.
  Binance Alpha public full depth subsequently closed the missing
  executable-looking ask question for the complete active transferable
  four-contract population. All minimum common quantities fit top-level depth,
  but none survived the frozen 20 bps pre-account stress: CRCL 0 bps, TSLA
  3.4592101470, COIN 6.5313231372, and MSTR 7.2132724213 gross. Do not repeat
  either screen. Reopen only after a material Alpha fee, execution, or book-
  architecture change capable of clearing 20 bps; no account, quote, order,
  transfer, paper, or live authority was used or added. Canonical artifacts:
  `binance-ondo-bstock-stock-perpetual-wrapper-parity-candidate-v1-2026-08-27.json`,
  result SHA-256
  `8bcf6f7bfa0cca6dab1fd6fd854a331d5ee41366ac6f9c0244b62a8f3545f475`;
  `binance-alpha-ondo-perpetual-parity-contract-v1.json`, contract SHA-256
  `2f08c6b0a8509d9d51db7716d5dde499c3a1937b68eafe77b2970e4da8311b59`;
  and `binance-alpha-ondo-perpetual-parity-v1-2026-08-27.json`, result SHA-256
  `a3d474e9010b92c9454a5bc04b5a7f586656c8bc5842cecc61baaa508c2d8bc3`.
- Binance Stocks has one accepted scoped, time-limited cost overlay. The current
  public fee page saves 5 bps versus the normal trading spread strictly above
  340 USD, or 0.18 USD per order strictly below 340 USD, through
  `2026-08-31T00:00:00Z`. The page labels both tiers as inclusive at exactly
  340 USD, so precredit zero there until a current order preview and owned
  realized fee resolve the ambiguity. Apply the saving only to independently
  justified organic direct-stock flow; never resize, acquire quote inventory,
  or trade to chase it. Account, jurisdiction, symbol, tax, spread, preview,
  and realized-fee evidence remain unproved, so it is not deployment-ready.
  Canonical result:
  `binance-stocks-current-fee-overlay-edge-v1-2026-08-27.json`, SHA-256
  `d4f02be559d9267abbea28ccefb48f4886f375b359ce7274b90b6585b828160a`.
- The older native-stock/TradFi-perpetual parity screen remains an incomplete
  result for its frozen 14-symbol population: 13 rows completed and zero
  survived 30 bps stress. Current official all-symbol Stocks stream behavior
  materially expanded the discoverable population, so the old KLAC-only
  recovery instruction is invalid and must not run. One exploratory expanded
  public screen found no after-public-fee positive row but did not retain a
  canonical raw population, so it cannot terminalize the current universe or
  support profit. Reopen only after a material fee, basis, or stream-
  architecture change, using a preregistered exhaustive population boundary
  and retaining raw quotes before every calculation. The old canonical result
  remains `binance-native-stock-perpetual-parity-v1-2026-08-27.json`, SHA-256
  `2776ff86fddf78e7e87860c6b9500cb237fce5af908a4840d351ae0cc2eff930`.
- A new primary paper materially supports a maker-first/taker-hedge complete-set
  execution design, but the strict public-fill reconstruction rejects a stable
  edge claim. In one exact 2026-04-27 Polymarket-v1 daily partition, 159
  conservative same-actor sequences across 105 BTC/ETH/SOL markets had only 75
  current-fee-sensitive positives, a `1.012037` median complete-set cost, and
  `-9.33095786` pUSD aggregate sensitivity P&L. ETH was negative overall, SOL
  had zero sequences despite 203,085 scoped fills, and 12 of 24 UTC hour bins
  were aggregate-negative. Positive delayed sequences require favorable price
  movement unless the opposite executable ask existed at the maker fill; the
  dataset has no placements, cancellations, books, or queue. Do not expand the
  historical tape. After the protected boundary and only with explicit
  authenticated paper authority, test one minimum post-only maker order only
  when a synchronized opposite ask locks positive after every cost, then hedge
  an owned fill immediately at exact quantity and reconcile every orphan.
  Canonical candidate:
  `polymarket-maker-first-taker-hedge-complete-set-candidate-v1-2026-08-27.json`,
  result SHA-256
  `4fe308ddeb6fd080bbd8548347a095762d8fc67eb5820fb0c7b3c2d6b7430d69`.
- Primary cross-venue microstructure evidence now rejects all-situation BTC
  five-minute market making. Across 1,613 top-decile manipulation-pressure
  cycles, 227 classified market makers lost `0.62M` USD, `381` USD per cycle,
  and were negative in `58.6%`; in 14,460 normal cycles they made `3.11M` USD,
  `215` USD per cycle, and were negative in `37.7%`. The paper explicitly leaves
  net-of-changing-fee P&L to future work, and its PushIntensity label uses the
  completed final-ten-second window, so it is not a causal live filter. Its
  separate fifteen-minute BTC test found the manipulation footprint largely
  absent but did not test profitability. The first future maker cohort is
  therefore fifteen-minute only. Five-minute stays excluded until a separate
  current latency-stress preflight proves every owned order cancel-confirmed
  before the settlement-risk window without future data. Never trade Binance
  spot to influence settlement. Canonical regime gate:
  `polymarket-maker-execution-manipulation-regime-gate-v1-2026-08-27.json`,
  result SHA-256
  `7d3387289a7e82b33fa52c03b2bc134864259a001c3d28524745026bb83db387`.
- Polymarket's official changelog reduced the crypto taker delay from 250 ms to
  50 ms effective `2026-08-17T11:00:00Z`. This leaves resting makers 80% less
  delayed time in which a stale quote might be cancelled, a negative protection
  change that invalidates 250 ms as a current forward-execution assumption but
  does not prove a numeric adverse-selection loss or current PnL. Do not
  resample books on this change alone. Any future authorized execution contract
  must use 50 ms or the exact current market value and fail closed when absent.
  Canonical regime artifact:
  `polymarket-crypto-taker-delay-regime-change-v1-2026-08-27.json`, result
  SHA-256
  `c7b785a1fbf4d6380033810338b2cf2845399f2a7464688c8ac36427b375a777`.
- Current sports execution rules do not turn maker rebates or live-NBA parity
  into a protected edge. Marketable sports orders enter a configured delay and
  cannot be cancelled while pending; the help center lists a three-second
  general delay and a one-second NBA/MLB test, while the compact market-info
  `itode` field only identifies a separate crypto/finance delay path and does
  not expose a numeric duration. The current crypto duration is 50 ms from the
  changelog; do not infer a finance duration. Therefore each all-taker leg or post-maker hedge
  must be treated as independently delayed and revalidated, with full orphan
  risk. Current official pages also conflict between 15% and 20% sports maker
  rebates, so credit zero until effective or owned payout evidence resolves it.
  Never reuse any crypto or finance delay constant for sports. Canonical gate:
  `polymarket-sports-taker-delay-maker-protection-gate-v1-2026-08-27.json`,
  result SHA-256
  `4847ec7828e598950da9a455170b66a529d9a5d671bfb4c37a57a36f608b9627`.
- The static high-price favorite taker-buy hypothesis is terminal despite the
  primary author's pooled favorite-longshot calibration pattern. Its latest
  chronological PWI role failed the frozen persistence gate, and the retained
  causal BTC/ETH/SOL action-value translation failed after current fee,
  execution, settlement, and capital stresses. The 90-95 cent band was
  unstable across roles, assets, winning direction, and time to close; the
  95-99 cent band lost 43.3821 pUSD across 951 independent conditions and was
  negative in every required slice. Do not download more partitions, refit
  bands, or infer favorite execution from longshot trades. Canonical result:
  `polymarket-favorite-longshot-bias-preflight-v1-2026-08-27.json`, result
  SHA-256
  `31cd01740e48b2dc0c76e9ca7820b0348aa7d04e403d0aeb71560000b9630c93`.
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
| Live NBA moneyline/spread implication candidate | `docs/model-research/action-value/polymarket-live-nba-moneyline-spread-combinatorial-parity-reopen-v1-2026-08-26.json` |
| Cross-market exact dependent-subset parity candidate | `docs/model-research/action-value/polymarket-cross-market-dependent-subset-parity-reopen-v1-2026-08-26.json` |
| Binance idle-stablecoin Launchpool reward candidate | `docs/model-research/action-value/binance-stablecoin-launchpool-idle-inventory-reward-candidate-v1-2026-08-26.json` |
| Polymarket Combo RFQ versus CLOB Boolean parity candidate | `docs/model-research/action-value/polymarket-combo-rfq-boolean-parity-candidate-v1-2026-08-27.json` |
| Polymarket terminal broad sports Combo requester-overround validation | `docs/model-research/action-value/polymarket-combo-maker-overround-validation-v1-2026-08-27.json` |
| Binance bStock dividend/perpetual funding timing-gap candidate | `docs/model-research/action-value/binance-bstock-dividend-perp-funding-timing-gap-candidate-v1-2026-08-27.json` |
| Binance bStock Spot LP all-symbol rebate overlay | `docs/model-research/action-value/binance-bstock-spot-lp-all-symbol-rebate-overlay-candidate-v1-2026-08-27.json` |
| Binance existing-stock transfer reward overlay | `docs/model-research/action-value/binance-existing-stock-transfer-reward-overlay-candidate-v1-2026-08-27.json` |
| Binance Stocks FPSL existing-inventory yield overlay | `docs/model-research/action-value/binance-stocks-fpsl-existing-inventory-yield-overlay-candidate-v1-2026-08-27.json` |
| Binance NOK bStock dividend/perpetual under-debit candidate | `docs/model-research/action-value/binance-nok-bstock-dividend-perpetual-underdebit-candidate-v1-2026-08-27.json` |
| Binance Ondo/bStock/stock-perpetual wrapper parity candidate | `docs/model-research/action-value/binance-ondo-bstock-stock-perpetual-wrapper-parity-candidate-v1-2026-08-27.json` |
| Binance Alpha/Ondo/stock-perpetual exact-book terminal screen | `docs/model-research/action-value/binance-alpha-ondo-perpetual-parity-v1-2026-08-27.json` |
| Binance native-stock/TradFi-perpetual parity incomplete screen | `docs/model-research/action-value/binance-native-stock-perpetual-parity-v1-2026-08-27.json` |
| Polymarket maker-first/taker-hedge complete-set candidate | `docs/model-research/action-value/polymarket-maker-first-taker-hedge-complete-set-candidate-v1-2026-08-27.json` |
| Polymarket maker execution manipulation regime gate | `docs/model-research/action-value/polymarket-maker-execution-manipulation-regime-gate-v1-2026-08-27.json` |
| Polymarket crypto taker-delay regime change | `docs/model-research/action-value/polymarket-crypto-taker-delay-regime-change-v1-2026-08-27.json` |
| Polymarket sports taker-delay maker-protection gate | `docs/model-research/action-value/polymarket-sports-taker-delay-maker-protection-gate-v1-2026-08-27.json` |
| Polymarket terminal favorite-longshot bias translation | `docs/model-research/action-value/polymarket-favorite-longshot-bias-preflight-v1-2026-08-27.json` |
| Polymarket August crypto-TWAP liquidity-reward terminal screen | `docs/model-research/polymarket/crypto-twap-liquidity-reward-screen-attempt1-failure-v1.json` |
| Post-observation maker window | `docs/model-research/action-value/polymarket-post-observation-maker-window-gate-v1-2026-08-26.json` |
| LDUSDT margin yield | `docs/model-research/action-value/binance-ldusdt-margin-yield-gate-v1-2026-08-26.json` |
| Terminal ETH/SOL Soft Staking delta-neutral funding stack | `docs/model-research/action-value/binance-soft-staking-delta-neutral-funding-stack-terminal-v1-2026-08-27.json` |
| Existing-idle ETH/SOL liquid-staking yield candidate | `docs/model-research/action-value/binance-existing-idle-eth-sol-liquid-staking-yield-candidate-v1-2026-08-27.json` |
| Terminal public BTC/ETH/SOL VIP Earn comparator | `docs/model-research/action-value/binance-vip-earn-public-btc-eth-sol-comparator-terminal-v1-2026-08-27.json` |
| USD1 Simple Earn activation refresh | `docs/model-research/action-value/binance-usd1-simple-earn-activation-refresh-v1-2026-08-27.json` |
| Accepted USD1 Simple Earn versus holding-airdrop allocation | `docs/model-research/action-value/binance-usd1-simple-earn-versus-holding-airdrop-allocation-edge-v1-2026-08-27.json` |
| First-USD-deposit SPCXB reward hedge candidate | `docs/model-research/action-value/binance-first-usd-deposit-spcxb-reward-hedge-candidate-v1-2026-08-27.json` |
| Accepted current TradFi perpetual fee overlay | `docs/model-research/action-value/binance-tradfi-perpetual-current-fee-overlay-edge-v1-2026-08-27.json` |
| Accepted current Binance Stocks fee overlay | `docs/model-research/action-value/binance-stocks-current-fee-overlay-edge-v1-2026-08-27.json` |
| Binance VIP 6 for Six organic-fee overlay candidate | `docs/model-research/action-value/binance-vip6-for-six-organic-fee-overlay-candidate-v1-2026-08-27.json` |
| Lite Loan and fixed-bonus stablecoin yield curve | `docs/model-research/action-value/binance-lite-loan-stablecoin-yield-curve-v1-2026-08-27.json` |
| U Flexible idle-holding yield | `docs/model-research/action-value/binance-u-flexible-idle-holding-yield-gate-v1-2026-08-26.json` |
| Existing RWUSD VIP bonus overlay | `docs/model-research/action-value/binance-rwusd-existing-vip-bonus-overlay-gate-v1-2026-08-26.json` |
| Current USDT Flexible bonus overlay | `docs/model-research/action-value/binance-usdt-flexible-current-bonus-overlay-v1-2026-08-26.json` |
| Existing USDe automatic holding reward | `docs/model-research/action-value/binance-usde-existing-holding-reward-edge-v1-2026-08-26.json` |
| Organic third-party Polymarket builder-fee overlay | `docs/model-research/action-value/polymarket-organic-third-party-builder-fee-overlay-v1-2026-08-26.json` |
| BFUSD reward-unit conflict gate | `docs/model-research/action-value/binance-bfusd-existing-holding-reward-unit-conflict-gate-v1-2026-08-26.json` |
| Binance Smart Arbitrage terminal adjudication | `docs/model-research/action-value/binance-smart-arbitrage-terminal-family-adjudication-v1-2026-08-26.json` |
| Organic Polymarket referral net-fee overlay | `docs/model-research/action-value/polymarket-organic-referral-net-fee-overlay-v1-2026-08-26.json` |
| Binance Flexible Loan collateral-yield gate | `docs/model-research/action-value/binance-flexible-loan-simple-earn-collateral-yield-gate-v1-2026-08-26.json` |
| Binance Advanced Earn conditional-conversion terminal adjudication | `docs/model-research/action-value/binance-advanced-earn-conditional-conversion-terminal-adjudication-v1-2026-08-26.json` |
| Binance Square organic Write to Earn fee overlay | `docs/model-research/action-value/binance-square-organic-write-to-earn-fee-overlay-v1-2026-08-26.json` |
| Binance organic Referral Pro fee overlay | `docs/model-research/action-value/binance-organic-referral-pro-fee-overlay-v1-2026-08-26.json` |
| Polymarket Perps organic referral fee overlay | `docs/model-research/action-value/polymarket-perps-organic-referral-fee-overlay-v1-2026-08-26.json` |
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
