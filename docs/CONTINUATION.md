# Continue Development

Current canonical status after the September USD1 campaign source gate:
accepted scopes 37, ranked hypotheses 65, terminal observations 186,
and stable current account-qualified after-all-cost edges zero. Registry
SHA-256 is
`39a69bce3a5544cc33d2d6383cc7f9578ff8c2305fe36caa1c57b0e5f053de1a`;
durability-audit SHA-256 is
`333c90c4cc471c6f2fa5544b2eecd367f8288009abe8278a3dbf9cf4120bcfe2`.

## September 4 resumed review and USD1 campaign

Whole-product reevaluation is explicitly requested: include research ideas,
economics, plans, interfaces and architecture, not only code/security checks.
Read `docs/review/2026-09-04/product-reassessment.md` for the first keep/change/
investigate decisions and open coverage. The current corrective patch changes
misleading native empty-state labels, console malformed-artifact/environment
handling, and the optional Tornado pin from 6.5.7 to 6.5.8. It does not implement
the proposed edge-centric interface or prove the whole repository reviewed.

The next substantive finding was false portfolio-return alignment: disjoint
histories were paired by row tails, and unequal horizons by end timestamp only.
The corrected risk module requires shared start/end observations and rejects
duplicate endpoints and nonfinite returns. The 23-test portfolio/model-lab
domain passes, with existing synchronized-data acceptance preserved. Historical
portfolio acceptance needs an independent interval-alignment audit before reuse;
no protected data or old outcomes were reopened.

The paused USD1 source work is retained under adjudication
`f3eaf05ff817dba843d992c643718f0b6f26fbd57bb5bfd15159522b0899114d`.
The September 4 to October 2 campaign exists independently of the ended
August campaign. Its 150 million WLFI pool does not determine per-user returns;
the first-period APRs and valuation remain placeholders. One frozen public
reconciliation may run after September 11 at 18:00 UTC or an independently
observed material official update. No previous article is being refetched now.

The user's comprehensive review is in progress. Inventory coverage does not
mean semantic review completion. The review must distinguish an unavailable
guaranteed lower bound, a negative stress scenario, negative expected value,
and evidence of economic impossibility. Historical contracts and captures
remain immutable; corrections require explicit new analysis and independent
confirmation before promotion. See `docs/REVIEW_2026_09_04.md` for coverage,
findings, and the current research priorities.

## 2026-09-04 WDCB/NVDAB dividend source-floor rejection

Binance published a distinct current announcement for WDCB and NVDAB cash
dividend support. This was a new source candidate, not automatic satisfaction
of rank 34's conjunctive retry trigger. Exact-URL deduplication found no prior
retained contract, result, or journal match. A frozen one-use public CMS GET
then retained 69,675 bytes and bound WDCB, NVDAB, their respective
2026-09-08T00:00:00Z and 2026-09-10T00:00:00Z snapshots, and the shared rule
that only the net cash amount after applicable withholding taxes, fees, costs,
and other deductions is reinvested into additional same-security units or
fractions.

The precommitted cheapest independent gate fails for both siblings. The exact
article publishes no positive gross cash amount per bStock unit, no finite
upper bound or complete formula for every deduction, and no deterministic
positive final units or multiplier increment. WDCB and NVDAB therefore each
have a zero source-bound conservative net-distribution floor. The full rank-34
retry trigger is not satisfied, so no issuer-calendar, perpetual-adjustment,
book, funding, fee, account, credential, order, fund, or rushed pre-suspension
request is justified. This is a terminal rejection for only these two exact
episodes, not evidence about any future dividend event.

Do not refetch or alias this announcement. Reopen rank 34 only for a future
independent episode where current primary terms first bind a strictly positive
conservative net-distribution floor and the bStock snapshot materially precedes
the official exchange ex-dividend adjustment. Source-contract, source-result,
and adjudication SHA-256 values are respectively
`36019fbf294ae3f3182ed75c6a4e35e1f305fd1ef7d44e59d24609ca663e3f6f`,
`fd294181f625d62967254bc97de6f55930b6af9194b97cc9280f493c82702e21`,
and `82bd2e0b0461b930218da3b7e01756cc2a5572d823bfc49586421fa3a7d5ce98`.
Raw and request-journal SHA-256 values are
`252d23d4de41ed250d2e5d8585cecd964aeeafb8b09d376a5b0dd38e51726882`
and `352a9de98ae662c0727c97c560afb19ccd33c58eeb49ef5ebbb4ab264c46c71a`.
No credentials were used, and the protected Polymarket capture was untouched.

## 2026-09-04 Binance September 2 bStock listing trigger

The higher-ranked rank-12 retry trigger was satisfied by Binance's exact
September 2 announcement for CRWDB, MRNAB, SQQQB, and STXB. A frozen one-use
official CMS GET retained the complete current article. The independently
frozen public inventory delta then advanced from 68 to 72 rows with no removed
ticker. All four new rows have an exact one-share multiplier. Conditional
current futures metadata proved active exact equity TradFi perpetual matches
for CRWD, MRNA, and SQQQ; `STXUSDT` is not an equity TradFi match and was
excluded rather than confused with the Stacks crypto contract.

The precommitted rule reported all three matches but selected only the
lexicographically first, CRWD, for economics. Before price access, a separate
contract froze two top-of-book requests and a 50-bip rejection hurdle. The
CRWDBUSDT spot ask and CRWDUSDT perpetual bid started 262 ms apart and were both
exactly 213.07 USDT per share. Gross entry headroom was exactly zero; the fixed
stress made the row -1.06535 USDT per share, or -50 bips. The sequence stopped
before depth, funding, fees, accounts, credentials, orders, funds, transactions,
or protected Polymarket capture access.

Only the exact CRWD observation is terminal. MRNA and SQQQ remain unscreened
and are not inferred from CRWD. To prevent adaptive cherry-picking after seeing
CRWD, do not inspect them now. Nasdaq's official 2026 calendar closes the U.S.
market for Labor Day on September 7. At or after
2026-09-08T13:35:00Z, freeze one separate lexicographically next MRNABUSDT to
MRNAUSDT two-request top-book test with the same 50-bip gate. Request full depth
and adverse funding only if it survives. This is not an accepted edge or a
deployment-ready strategy.

Listing source-contract, listing source-result, inventory-delta contract,
inventory-delta result, CRWDB spot source-contract, CRWDB spot source-result,
CRWD futures source-contract, CRWD futures source-result, top-book contract, and
top-book result SHA-256 values are respectively
`e232dcfdb16039fb5dbb3e30f55e1470046b2ee571def91317f70b1267526601`,
`72e5428bc2b96f8bd05867479fedddda40665458548ce4c0456f107d1bb24e4b`,
`a264751084738d226669ea181409e8e357d77c41cd53f9955931b12b3a84f00d`,
`ba5ebb29ce89c8cc09bde5066bbe4a0cfc7dc11fa926f5a00de6321054caba26`,
`ce797e1624a240d93048870bc7de25a962a40ad0889f0f3c5f36b9a0d1d65e5f`,
`ae771bb195ea5882365c91b0e193be74d31ee98fd3c071265805975bc4f80a7e`,
`5b2246de062f787a04e6e137d2321155d30aa4c77ac34dc81b6c8b472e3f4614`,
`0f9251248a65f04853e73330d36568e4822f9187f4c5fb92b84fa2b7c1fd2401`,
`4b6091bbeda43d6b6609c3fc9980ffc28b19ce015b8dad8dcb5ddc9891c65dd2`,
and `a02367c4c0f7461a29ba857f947dc870ab6539be36419d7b068e5e685b58c744`.
The retained article, current bStock inventory, futures metadata, CRWDB book,
and CRWD book raw SHA-256 values are
`fde73b7185b6660cefe859197551b1c789a79841316f9610f41e6abe72a3fcda`,
`0550a22f138562caee8cb22962c494b911feaca43e830a9e28c8bae3c2fddff6`,
`91beaa5253d9e89d9eabfaad901f6c8888da20066394576acc22434438560f5e`,
`bce7837961ccefe919a2bbc5aad4cddc499691a1f7597aea3734761f8d86c581`,
and `201a7131d2b4fad337ac9a9cbf9ba3fd4b799d8b4adb8a72507f36903d770b05`.

## 2026-09-04 Binance distinct crypto-option population rejection

Rank 47's distinct-population trigger was satisfied. One prospectively frozen,
non-polling public `exchangeInfo` GET found 1,488 currently eligible BTC, ETH,
and SOL crypto options. Relative to the hash-bound 1,576-symbol baseline, 356
were new and 444 had disappeared. Exact source-bound deduplication proved zero
overlap with the consumed 508-symbol August 31 population and removed the two
already screened December 25 94,000 BTC call and put symbols, leaving exactly
354 genuinely unscreened contracts: 174 BTC, 90 ETH, and 90 SOL.

A timestamp preflight initially failed because its typed freeze instant was
later than the observed clock. No network request ran and no output existed.
Only that invalid field was replaced with the actual observed
2026-09-04T18:40:01Z instant and the self-hashes were recomputed before access;
the request, population, and decision rules did not change. `AGENTS.md` already
forbids guessed future freeze times, so no weaker or redundant rule was added.

Before price access, a separate contract froze exactly two public requests,
the complete 354-symbol identity hash, side-correct terminal-payoff formulas,
positive displayed-entry requirements, and a 33.5-bip fixed fee and expiry-basis
stress. The option-ticker and futures-book requests started 245 ms apart and
returned every scoped ticker. Of 354 rows, 226 had positive option asks and
perpetual entry sides. Only `BTC-260905-80500-P` had a positive displayed gross
floor, 11.10 USDT per BTC. The frozen stress cost 266.555815 USDT and changed
that row to -255.455815 USDT, or -32.1049826 bips. Zero rows survived, so no
depth, funding, fee-account, credential, order, fund, transaction, or protected
Polymarket capture access followed.

This exact population is consumed and is not an edge. Do not refetch, poll,
reprice, subset, or request depth for it. Reopen rank 47 only for a distinct
active BTC, ETH, or SOL option population after the retained
2026-09-04T18:49:36.306Z snapshot; a material fee, settlement, tick, depth,
funding, expiry-basis, capital, or architecture change; or an independently
observed non-polling displayed terminal floor strictly above every applicable
cost. Then apply the same rejection-first sequence.

Population source-contract, population source-result, population gate-contract,
population result, option-ticker source-contract, option-ticker source-result,
futures-book source-contract, futures-book source-result, price-prefilter
contract, and price-prefilter result SHA-256 values are respectively
`b41ca8937b8cfa0275bda15a97d40879f19df515e544862185dd1057e28ea305`,
`be1f95a1d4affa5c700c5f2e5d9fa1f04f087767803394fbb1d5b0bd3d216dd0`,
`36f556f034dffa1f73d87283b8ab6cd8f01ccf561c244e39290a49df4456a415`,
`8c10dd5d039bb0753207e86ee14e6761b46f160cf21a2114ba7fd6470632972c`,
`b99d763b3db26b4168bde76d34241fd80709f77d89df51391f0bb5649b529138`,
`ecd384396fcb5432526f79fa67d9ac0dd560b1f631d30787974f27403427ce63`,
`a1d42e3e9f9c78f4d50a5df84c22bdc7b879dfe133c5370d3af2c8e9e42d2c4f`,
`3dbe4ba86772573e170091ee648f649412e23c5b8b6f5da98fade89241e74b71`,
`7990c1d0ff3f37fd031e190921eb6bd07075a0495725acf121730e5ba634e0de`,
and `fbd10642d35ed469b7cc7f5554681d90d074be71fe718498738011af2fcb3b8e`.
Raw population, option-ticker, and futures-book SHA-256 values are
`06da75d0e6ebacc007a73925213240c370751a79d0cb4ae624d044044746a396`,
`b71de1f16145e96824cb456d87df7d7683d49cd4cf0124fcfc3d163027625222`,
and `5f03b4c945760bb65c8e60c212fa81be4ca65794ff09de62e4a6f7662acd1be4`.
Their journal SHA-256 values are
`dac125f9f49e9dfcd615eb4e9c2ba53a791c87008169355a49ab85d4570f6bc4`,
`7d8b79a7ba8a0002829faee2dbf95becff3dec48a377cbfca9b267642c43b3d6`,
and `48d7f66357ddd33e7ae9823ef738d64b920467b220673441d689fafbb77d9147`.

## 2026-09-04 Binance CXMT four-hour funding trigger

A fresh official Binance notice materially changed nine TradFi perpetuals from
eight-hour to four-hour funding effective 2026-09-04T08:15:00Z, with the
per-settlement cap/floor changed from plus or minus 2 percent to plus or minus
1 percent. Exact retained inventories prove a literal active `CXMTUSDT` to
`CXMT-USD` cross-venue match, so rank 43 has a distinct post-change history
question. No alias or merely related product was used.

This is not yet an edge. Three old 2-percent settlements per day and six new
1-percent settlements per day both have the same arithmetic 6-percent absolute
daily cap. The change affects timing, not the maximum daily bound, and proves
nothing about the sign or persistence of realized funding. Sampling now would
also be underpowered: twelve complete new four-hour settlements do not exist
until the 2026-09-06 08:00 UTC settlement.

At or after 2026-09-06T08:10:00Z, freeze the receiving-leg orientation before
history access and run one exact twelve-settlement CXMT test split 6 training,
3 validation, and 3 test. Every role must clear a fixed 20-bip execution hurdle
plus two 500-bip annual capital hurdles. Request no books, fees, accounts,
credentials, orders, funds, or transactions before a persistent survivor.
Source contract/result, trigger contract/result, raw, and journal SHA-256 values
are
`54e2ba6f701a13ba30da2c0035fbff41f5cd72b1d2bd3401bbd88b153324fa2d`,
`0943e3e603a75d3b3af28ed0c356b10f1980219bca66d98e1f8f1f40b2416ebd`,
`b03eb43353a1e935123cbbc3671f8d9739c91d1c2900867478ca3749e65372ed`,
`99d0fded6f7378d0b398b33cd5221f515704cd8c31c3b360e9773eac784f6402`,
`3b570055daeccf38d9d9bc9d9130cd11b99142e4dbf1777a6af835ffa9c044da`,
and `8fdf47248870819a032bc9bb856130c6685bd7856a54db7c6bba4322ede649bb`.
No credentials or protected Polymarket capture bytes were touched.

## 2026-09-04 Binance scheduled-yield publication gate

Rank 8's exact scheduled trigger was satisfied after the promised Friday
18:00 UTC publication deadline. Before access, three contracts froze the two
official public CMS requests, exact article identities and period rows, numeric
row regexes, calculations, and fail-closed decisions. The two requests ran once
in parallel and retained 132,816 USD1 bytes and 127,552 RLUSD bytes.

Both official articles missed their own displayed update deadline. The USD1
fourth and final distribution row still contains three `To be updated on
2026-09-04` placeholders and its article timestamp remains
2026-08-28T08:19:00Z. The RLUSD third-distribution row still contains two such
placeholders and its article timestamp remains 2026-08-28T08:25:00Z. The frozen
numeric patterns therefore match neither row. Do not infer either APR from
prior weeks, rewrite the test, or retry an alias, locale, article, or parameter.

The USD1 holding campaign has ended and has zero forward holding-reward floor;
its three published positive base APRs remain historical scoped evidence. The
separate fixed 7 percent USD1 Simple Earn candidate remains only under its prior
account, capacity, remaining-horizon, transition, opportunity-cost, issuer,
redemption, tax, custody, and all-cost gates. RLUSD remains unaccepted with two
published completed-week APRs, a missing third value, and a zero public forward
floor for its unpublished fourth week. No accepted-edge count changes.

The September 4 captures are consumed. Do not poll the same pages. Reopen this
publication question only if an independently observed official timestamp
advances beyond either retained August 28 baseline, on a material official
economics change, or at the distinct RLUSD fourth-distribution trigger after
2026-09-11T18:00:00Z. Contract, USD1 source-result, RLUSD source-result, and
adjudication SHA-256 values are
`e9cb7b1923e55afa4bbe8bb64b1ff9c28598251d40671e84030e715deac747f5`,
`6b94591d7e4486a3e9b96dfe6eb0bd4d56e0a89c5ef3b159dae77a71f6b4f346`,
`e2674a34cf658810c9aa97e4e8710813ce4d83fc4cc734ee76b028df146a672b`,
and `5728de0226526a070c44179f3dc4eb673098349b38e62e646722a6f548f126bb`.
No credentials, accounts, orders, funds, transactions, or protected Polymarket
capture bytes were touched.

## 2026-09-01 Binance Options block-trade single-leg rejection

Rank 37's account-gated RFQ route raised one cheaper public architecture
question: whether Binance's separately documented Options block-trade API can
atomically carry a four-leg box. Exact-URL deduplication found zero retained
matches. Before direct access, a one-use contract froze the exact official
source and required at least four supported legs before any economic or
authenticated request.

The exact public GET returned HTTP 200 and durably retained 40,487 bytes. The
generic text runner then stopped on `UnicodeDecodeError` because the
extensionless source is a four-page PDF, not UTF-8 text. The request was not
retried or aliased. Complete offline extraction and visual review of the
hash-bound PDF proves that `POST eapi/v1/block/order/create` creates a New Block
Trade Order, accepts a mandatory `legs` list, and explicitly limits it to
`Max 1 (only single leg supported)`. Acceptance is a separate trade endpoint,
and account history is USER_DATA.

The one-leg limit terminalizes this programmatic workaround. It cannot execute
a two-leg vertical or four-leg box atomically and therefore cannot remove the
multi-leg execution risk needed for fixed-payoff parity. No quote, book,
commission, account, credential, order, fund, transaction, or protected capture
was used. Rank 37's separate predefined two-leg Options RFQ UI route remains
unchanged: it may reopen only with explicit quote-request-only authority plus
eligible RFQ account access, and confirmation or execution requires separate
authority.

Contract, raw-response, request-journal, and adjudication-result SHA-256 values
are
`5cbf54045024dfa9c5bb63bebbdb5caa706f2b35a43894c2f8758305851784ab`,
`06abdba3d906ddbcc03400cb74f01926c0b410929e8db8b1305b1ae474510882`,
`1e22651e53b827483ab6ecb1a1a50997966c2ffdf40bde2647686b802a14d3c1`,
and `31f0149202f997f0ba7015629d6c3df1e9f86b6580c5f98a34c024d2f3da7b6d`.
The public-source capture helper now classifies future non-UTF-8 responses
fail-closed with detected-format and decode metadata after durable retention,
instead of crashing. This is an efficiency correction only; it does not repair
this consumed capture or weaken any source or economic gate.

## 2026-09-01 Binance Futures Bonus Voucher downside rejection

One prospectively frozen unauthenticated public CMS GET retained the exact
current official Futures Bonus Voucher FAQ. It says free tokens can be used as
same-cryptocurrency USD-M or COIN-M Futures collateral, losses can be offset
against the bonus amount, generated profits can be withdrawn or transferred,
the bonus principal cannot be directly withdrawn, and unused credits may be
revoked after 30 days.

Those headlines do not prove a free option. The FAQ expressly says it is not
legal terms and may be outdated. It does not specify bonus-versus-user-fund
loss ordering, commissions, funding, liquidation, deficits, ADL, insurance
recovery, clawbacks, mixed-balance or other-asset protection, additional-margin
requirements, exact depletion accounting, or transfer and withdrawal effects.
The fully nonrecourse gate therefore fails. The narrower organic-cost-overlay
gate also fails because exact depletion and mixed-balance mechanics are absent.

The hypothesis is terminal before linked controlling terms, prices, funding,
books, accounts, credentials, Rewards Hub, voucher actions, positions, orders,
or funds. Do not refetch, alias, or use testnet credentials as mainnet evidence.
Reopen only on rank 65's material official downside-term change or a distinct
prospectively frozen complete controlling-terms package selected before access,
plus an independently owned-voucher read-only trigger. Every claim, activation,
deposit, trade, position, transfer, withdrawal, or state change requires
separate explicit authority.

Contract, source-result, raw-response, request-journal, and terminal-result
SHA-256 values are
`dd35c75a5f84ff75e8623e15082596fa4ca25cbe8ea090236f8592cfa5faab57`,
`971e87c3d6e95b37bf1a565f2054f828b55e13a4f8d746578a5d1682c5c8d1e3`,
`f7325515673a162e8b21f6a15ba9e2fe02aaa0b999c8877ff51f2161c538d710`,
`7e5db43bb0bee51c15fc8d5ad7dd0823e58a4504be1ae72ac339e26b22741ee7`,
and `57822cbd67822d8c8d61e6d353db99b42dc9895ff19e51e6825beee0a4f66423`.
No account, credential, order, fund, transaction, or protected capture was used.

## 2026-09-01 Binance Trading Fee Rebate Voucher scoped edge

One prospectively frozen unauthenticated public CMS GET retained the exact
current official Trading Fee Rebate Voucher FAQ. It proves that eligible Spot,
Margin, or Futures fees actually paid after activation are rebated at the
voucher-specific trade-back percentage. The daily refund is credited to Spot
Wallet in USDT, USDC, or BNB the next day. It uses the net paid fee after VIP
and BNB discounts, is limited by remaining voucher balance, and expires on the
individual voucher date. Referral commissions, liquidation and delivery fees,
some bots, Alpha trades, unavailable regions, suspicious activity, wash trades,
bulk accounts, and sub-accounts are excluded or revocable.

The accepted scope is narrow: exact positive owned rebate actually credited to
Spot on independently justified organic eligible fee-bearing activity after
every voucher-acquisition, Points-opportunity, activity, execution, tax,
conversion, custody, delay, failure, and operating cost. It is a deterministic
cost overlay, not a profitable underlying trading strategy. Never trade merely
to obtain or consume a voucher; never create volume, self-trade, wash-trade,
spoof, manipulate, use fake users, or double-count BNB, VIP, zero-fee, maker,
referral, affiliate, or other discounts.

No owned voucher, product, percentage, balance, maximum reward, expiry, region,
eligibility, organic paid fee, Spot receipt, persistence, deployment readiness,
or positive public forward floor is proved. Testnet credentials are irrelevant
to mainnet voucher state and remained unused. Do not refetch or alias the FAQ,
open Rewards Hub, claim, activate, spend Points, trade, or mutate state. Reopen
only on rank 64's exact material official change or explicit signed GET-only
mainnet authority plus an independently existing voucher; every state change
requires separate explicit authority.

Contract, source-result, raw-response, request-journal, and adjudication
SHA-256 values are
`f7d8d67913e8813e16e62ee031c29ead28715676b01b5b8ea1e604b645025a15`,
`73698c080c26f0806df8168d88a2ce7f6fdcbece2b35a35fd35eab71ef059de1`,
`fbb3f32ba39b853b0774343c88e40c32893be4da209a02a09559bfcb62c832d8`,
`ed063592a2a156c84e71b2f509a6ef29699526e0b625bdd0122eb50204ef4cde`,
and `f2d8c368459ac59d440e388c0a43e976ab33aa0404132455327ad7001fce17aa`.
No account, credential, order, fund, transaction, or protected capture was used.

## 2026-09-01 Binance Futures Free Position downside rejection

One prospectively frozen unauthenticated public CMS GET retained the exact
current official Futures Free Position FAQ. The source proves that the voucher
opens a position with given trading margin; contract pair, order type, leverage,
and margin mode are voucher parameters; the user may choose Bullish/Long or
Bearish/Short; generated profits may be withdrawn or transferred; and the
position may be closed at any time with stop-loss control.

The decisive downside terms are missing. The FAQ does not say that position
losses, commissions, funding fees, liquidation charges, deficits, clawbacks,
or other liabilities are confined to the supplied margin. It does not say that
the user's other Futures or Spot balance, collateral, deposits, or assets cannot
be debited, encumbered, or liquidated. It also does not prove zero acquisition
cost, no additional collateral requirement, owned eligibility, exact position
economics, expected value, or realized receipts.

Supplied margin plus withdrawable upside is therefore insufficient to model the
user payoff as `max(position PnL, 0)`. The zero-personal-capital free-option
hypothesis is terminal before prices, books, accounts, credentials, Rewards Hub,
voucher activation, terms acceptance, positions, orders, or funds. Do not
refetch, alias, follow the adaptively discovered terms link, or use testnet
credentials as mainnet evidence. Reopen only on rank 63's exact material
downside-term change or distinct prospectively frozen complete official terms
package plus independently owned-voucher read-only trigger. Every activation,
side selection, position, transfer, withdrawal, or state change requires
separate explicit authority.

Contract, source-result, raw-response, request-journal, and terminal-result
SHA-256 values are
`d0b6f83bac0f2f4f5aedcb631e2d7ce2cadc3ff441f5e765219cd788f7c1c5ca`,
`317aeb0f872c9f74c542aef3e7eaaa45eefce05adea7a8e8174166f1189519fb`,
`092b39ce35545530e9ae730fd64399308b8e7535ef704686409d7e681729b66a`,
`03d212d56ef5fbbc21e8a24bb38909d3a5d528644cc6ec9ee11f304950b616d8`,
and `a258fff5e8a96f8a36de8aca28d72d15d7e8d29c26a262d9df2c673a5b750fff`.
No account, credential, order, fund, transaction, or protected capture was used.

## 2026-09-01 Binance Token Voucher source-gate rejection

One prospectively frozen unauthenticated public CMS GET retained the exact
current official ordinary Token Voucher FAQ. Four of five mandatory phrase
gates passed. The source uses `with no restrictions`, while the contract had
guessed the unobserved lexical variant `without any restrictions`. The frozen
decision requires rejection if any exact phrase is absent, so the exact public
direct-cash hypothesis is terminal. It was not rewritten, aliased, or refetched.

The retained bytes substantively describe token vouchers as rebates and say a
redeemed voucher credits free tokens to Spot Wallet for unrestricted use,
transfer, or withdrawal. Those mechanics are hypothesis-generation evidence
only and have zero creditable value under the failed contract. No owned voucher,
acquisition lineage, token, amount, expiry, region, account eligibility,
realized credit, tax, conversion, withdrawal, custody, operating cost,
profitability, or deployment readiness was proved.

The process error was assuming an exact synonym from discovery rather than
freezing text observed in the selected primary source. `AGENTS.md` now requires
exact source-observed phrases, minimal observed substrings, or prospectively
frozen alternative forms; a failed phrase gate remains failed after access.
Do not refetch or alias the FAQ, open Rewards Hub, use credentials, or rescue the
result with account access. Reopen only on rank 62's exact material official
change or independently owned-voucher read-only reconciliation trigger. Every
claim, redemption, transfer, withdrawal, conversion, order, or state change
requires separate explicit authority.

Contract, source-result, raw-response, request-journal, and terminal-result
SHA-256 values are
`7defa837fd8622ecc8ff94cff3fca2ed35125302859ae8ff4141fb81ab47bd8b`,
`5235ee14ad6d7d17c69938d7215f5266048610fac7148de8c27375eba4ddab45`,
`6376d31fe3ee6f721e5110c7f31a06a75610a33546864c1500af2472ffe50d53`,
`ddc0e4f38c5861efc386b1caa8f7a9d9b2e78a870c8a543d7d5a40a372016dd4`,
and `1776c7bcd318edd3c703446f7e2ca746922efc307c2c295075dd0b58eb03c752`.
No account, credential, order, fund, transaction, or protected capture was used.

## 2026-09-01 Binance Simple Earn Trial Fund scoped edge

One exact current official Binance FAQ was frozen and retained through a
single unauthenticated public CMS GET. It proves that Simple Earn Trial Fund
face value cannot be withdrawn or transferred, is used only for APR reward
calculation, and is not redeemed to Spot. Using the voucher creates a locked
position; its APR rewards are distributed to Spot daily for the voucher period,
and early redemption is not allowed. The account presentation exposes the
exact token, amount, APR, and validity rather than making them public constants.

The accepted scope is therefore narrow: exact positive Spot APR rewards from an
independently awarded zero-cost voucher, after tax, conversion, operating, and
every other incremental cost. The face value is never cash or profit. A voucher
acquired with Binance Points or another asset must charge its full opportunity
cost. No owned voucher, positive APR, token, amount, validity, regional or
account eligibility, receipt, persistence, deployment readiness, or positive
public forward floor is proved; stable current account-qualified edges remain
zero.

Do not refetch or alias the consumed FAQ, open Rewards Hub, or use testnet
credentials. Reopen only on a material official term change or with explicit
signed GET-only Binance mainnet authority plus an independently existing
voucher and exact owned voucher and Spot-reward reconciliation. Claiming,
redeeming, accepting terms, subscribing, converting, transferring, withdrawing,
or any state change requires separate explicit authority.

Contract, source-result, raw-response, request-journal, and adjudication
SHA-256 values are
`41998867e54deac8bc4e394a88c948b0b407782d6f543bbbb7b027d06705d3aa`,
`70bcaf46a96aa095003289cab9720312490c13a1bcb6cfdb099f11b0da5ed0ed`,
`a5317a6dc24cabd626d37a11da1d4d26f742ee488c08b108cc84cdf4fefb8178`,
`16a7e118b182f791f927e2c76f9d538ffa351a97a65e70a232a0af82c59f1baa`,
and `d1738d58015b2ee1e34d8dc3bc24eee15cceeeec8730b2ef4c1b8d35a5f30f70`.
No credential, account, order, fund, transaction, or protected capture was used.

## 2026-09-01 Binance-Backpack cross-venue funding rejection

The distinct BTC, ETH, and SOL Binance-USDT versus Backpack-USDC perpetual
funding family was frozen before any funding value was emitted. The population
contained 270 Binance eight-hour buckets per asset and required the exact 2,160
Backpack hourly intervals that end within them. Training, validation, and test
roles were 135, 67, and 68 buckets. Orientation was selected on training only.
The prefilter charged 20 bips round-trip execution, a 10 percent annual two-leg
capital hurdle, 25 bips of USDC/USDT stress, and 25 bips of custody, latency,
and failure stress before any premium, basis, or book request.

Seven frozen unauthenticated venue GETs retained one 89-market Backpack
inventory, 2,200 Backpack hourly rows per asset, and exactly 270 Binance rows
per asset. A v1 preflight stopped before economic output because Backpack's
live interval-end strings were strict whole-hour ISO labels without timezone
markers. That failure is preserved. A separately retained current official
Backpack source anchors the exchange-wide hourly funding change at 08:00 UTC
and states that funding is debited or credited at each interval end. The v2
contract changed only that strict timestamp interpretation; population,
alignment, roles, orientation, costs, gates, and raw inputs remained immutable.

Every training-only orientation was short Backpack and long Binance. Gross
funding was often positive, but zero asset passed every after-cost role. BTC,
ETH, and SOL test net results were respectively `-118.9728466210`,
`-113.6156966210`, and `-110.2172366210` bips. BTC validation gross was
negative, and SOL's validation second half was negative. The exact population
is terminal before premiums, basis, books, accounts, credentials, orders, or
funds. Do not refetch, resample, realign, refit orientation, or weaken costs.
Reopen only on a material venue funding, fee, quote-unit, custody, latency,
capital, or execution-architecture change, or a distinct prospectively frozen
nonoverlapping population.

Preregistration, v1 failure, v2 contract, and v2 result canonical SHA-256
values are
`3e59d011fedc55d81f3fdd78b75ef20fcd9f3c7b7b89dd24a704335d4b372baf`,
`411575adf40b65ddecc945ce4af423b3a07b21863dd3bbb285d240f37e7c5032`,
`baf18425361d1dc750296ce619ad9caf5ef74e6f8109e1b3994e19c8778354d4`,
and `ef665320c7cb3537d5f1a9b296fb95ca545def685fe7bb25abbda52c1193f133`.
Inventory, Backpack BTC/ETH/SOL, and Binance BTC/ETH/SOL raw SHA-256 values are
`b515ac0b293694af23c2de22bf7d281b8ba470a02187bcdfcb07f692228d4bf7`,
`0fabf713e50e2353f4fdf311c2ac01b053adce64a1b4536eac0e0591f5d4afb8`,
`405b0385c808b33b3d11d7c2eaeb4d9a9fa3886b6e90ba2294bd8354408c8226`,
`c16c7c727342858104a74a2251e48621843c768454c63117c840088f00bcef75`,
`caa7222b1e3ce9f412cd51d84952f30f3e88550b1b1839f26386100dae386483`,
`8d9cef0513bad8dfb77f83a49f462e74ee5ab03915b1304363be0803d27e1a48`,
and `8150891d65d0d2257b719b4ecc3d54a574d62905b76a6b09d6bd34e6e0a484e7`.
No credential, account, order, fund, transaction, or protected capture was used.

## 2026-09-01 Binance-Paradex cross-venue funding source gate

Official Paradex sources proved the public history schema, exact BTC/ETH/SOL
USD perpetual instruments, USDC settlement, continuous pro-rata funding, an
eight-hour reference rate, and positive funding as long-pays/short-receives.
Paradex itself aggregates Binance and other venues into funding, so the frozen
question was limited to a possible timing or smoothing spread.

Before any funding value was viewed, the exact four public requests, 90 Binance
buckets, 45/22/23 chronological roles, training-only orientation, hourly-to-
eight-hour normalization, execution, capital, USDC/USDT, custody/latency, and
sampling hurdles were frozen. One current market inventory and three bounded
histories were retained. The inventory returned 2,474 rows and exceeded its
two-MB ceiling. Every funding history returned 5,000 rows, exceeded its one-MB
ceiling, and included a continuation cursor.

The preregistration forbade pagination, adaptive cadence changes, resampling,
or changed roles and costs. The exact population is therefore terminal before
economics. No funding value was printed, viewed, or used; no basis, book,
premium, account, credential, order, fund, transaction, or protected capture
was touched. Do not refetch, alias, paginate, or consume the exact cursors.
Reopen only for a distinct prospectively frozen nonoverlapping population or an
independently known material API cadence, pagination, or architecture change.

The reusable failure was an underspecified pagination plan and arbitrary byte
ceilings. `AGENTS.md` now requires uncertain-cadence cursor endpoints to freeze
conditional traversal, maximum pages, total rows and bytes, deduplication, and
stop conditions before first access, with page ceilings derived from the
documented maximum row count.

Terminal adjudication SHA-256 is
`98c5441b5f9828664c387b9f9b8a05e4eef919a3857054c8bbd73faa4c3ea30a`.
Inventory raw and journal SHA-256 values are
`f0c0299701b6e52f3bbe539affa0c2c8bf75ccd9f32a98a084bdec678c429c0c`
and `b6537f0a293a74705692a2059e461b31a5f2753735979d492292294ec640be4e`.
BTC, ETH, and SOL funding raw SHA-256 values are
`71520ae9b08536953d480414abe00b5cc4a6d8017fa7143b3c7de8a1072b6fd2`,
`8af42e6a6128119737b66892a478ffd8ad9d32f510975df92d129c52a607cfae`,
and `70c95d73b8635f55862f43ed35ebcfd15ed8d8d7b96e720e7ad545a548f0fba4`.

## 2026-09-01 prospective NYC September 2 complete long-only basis screen

The next deterministic member after the consumed September 1 NYC daily
high-temperature event was selected before any rendered page, search result,
Gamma response, or other price-bearing source was opened. The exact September
2 slug, expected eleven-outcome count, complete payoff basis, side-specific
price rules, five-share quantity, fee model, one-tick stress, and one-use request
were frozen first.

One exact public Gamma GET returned event 940515 with all eleven active,
accepting-order fixed-NegRisk markets. The complete primitive long-only basis
covered the all-YES complete set, all ten price-complete same-market
`YES + NO` binary straddles, and every optimal `k`-NO cardinality frontier.
Zero packages had strict positive metadata headroom, and zero survived current
fees plus one adverse tick per leg.

The best package was the `69°F or below` same-market straddle. Its conservative
side-specific cost was `1.001` pUSD per share against a one-pUSD floor, for
`-0.005` pUSD gross at five shares. After the frozen tick-and-fee stress its
floor was `-0.01625` pUSD. The strict source gate failed, so no book or fee
endpoint request was made.

This exact event is terminal. Do not refetch, alias, reprice, request books, or
select a sibling. Reopen only for a distinct event selected and frozen before
prices or a material price, fee, tick, rule, adapter, or market-architecture
change. Raw response and journal SHA-256 values are
`2f2481f1479fd1a5a36c2fce3be374b9d692e7616014e0486044d9648709ae1c`
and `2991b25fd601391d38aaeb1fb378cf3cf1872a23d7bfb006a8689fd4334caf75`.
Contract and result SHA-256 values are
`ecb4f25a9585e724078595907c4f35bc087ce10ec2045e31e2af8a804d19ec22`
and `68232f3fa8f8ba9a6a747ce1b1c06428bbd1ad7e6b1a190b5120610b97ab9c8b`.
No account, credential, order, fund, transaction, or protected capture was
used.

## 2026-09-01 retained fixed-NegRisk k-NO cardinality frontier

The pairwise screen did not mathematically exhaust larger NO packages. For any
`k` distinct mutually exclusive outcomes, buying every NO has a `(k - 1)`-pUSD
floor. At fixed common quantity, sorting price-complete legs once yields the
cheapest subset for every cardinality; sorting each leg by additive
fee-and-one-tick unit cost yields the stressed frontier. This reduced
2,148,007,910 possible subsets to 41 exact frontier rows without brute force.

The frozen zero-network screen found zero strict metadata candidates and zero
stressed candidates. The best metadata frontier was nine NO legs in event
624242: cost `8.012` pUSD per share against an eight-pUSD floor, or `-0.060`
pUSD at five shares. The best stressed frontier was eight NO legs in the same
event and lost `0.150` pUSD. The retained prices had already been exposed by
the earlier pairwise screen, so this result was explicitly frozen as
hypothesis-generation evidence only and could never promote an edge.

Do not brute-force the dominated subsets, refetch, reprice, request books, or
treat this post-price result as prospective evidence. Contract and result
SHA-256 values are
`1443f1b6390762c25faceddb03be73b3afa85a3659d662d95ffdb5727d349df0`
and `a1e93af849bc7ce31f9b17e1498023d36385ebcd7b53252e2e797c4a2515bd90`.

## 2026-09-01 retained fixed-NegRisk pairwise-NO audit

The hash-bound September 30 Global catalog contains three complete fixed-
NegRisk events with 19, 31, and 5 markets. Any two distinct outcomes in one
event are mutually exclusive, so `NO(A) + NO(B)` has a one-pUSD floor: if A or
B wins one NO pays, and if another outcome wins both NO tokens pay. This is a
distinct direction-independent package family not exhausted by the earlier
all-YES complete-set screen.

The prospectively frozen v1 zero-network run failed before producing economic
output because a fee-disabled market had no `feeSchedule`. That failure is
preserved and was not rerun or rewritten. A retained-byte fee-shape audit found
19 markets with `feesEnabled=false`, empty `feeType`, and no schedule, plus 36
enabled `tech_fees` markets with explicit schedules. The frozen v2 correction
therefore treats absence as zero only under the explicit disabled-fee shape;
every enabled market still requires an exact supported positive schedule.

The one-use v2 screen exhausted all 646 unordered pairs. Of those, 481 had
complete side-specific rejection prices and 165 remained price-incomplete.
Zero complete pairs cost strictly below their `1.000` pUSD floor, and zero
survived current fees plus one adverse tick per leg. The best pair was `No
Meeting by September 30` plus `Switzerland` in event 624242 at `1.084` pUSD per
share. At five common shares it lost `0.475` pUSD after the frozen tick-and-fee
stress.

This exact three-event/646-pair population is terminal. Do not repeat, refetch,
reprice, request books, or adaptively fill the 165 missing prices. Reopen only
for a distinct unconsumed complete fixed-NegRisk event or a material price,
fee, tick, rule, or architecture change. No network, credential, account,
order, fund, transaction, or protected capture request was used. Canonical v2
adjudication SHA-256 is
`951bde07b04cd1257d6be4e4d6e163a3b6a8adc3198bf653eb2504f791d23520`.

## 2026-09-01 retained Iran-island OR implication audit

The hash-bound September 30 Global catalog contained three active individual
conditions for Farsi, Hengam, and Hormuz Island plus a distinct active
four-island condition covering Farsi, Hengam, Hormuz, or Kharg Island. Under
optimistic exact rule alignment, each individual condition implies the
four-island OR condition, so `NO(individual) + YES(composite)` has a one-pUSD
floor without a market-direction forecast.

The zero-network retained screen rejected all three packages before books. Each
individual YES best bid was `0.010`, making the conservative NO ask proxy
`0.990`; the composite YES best ask was `0.028`. Every package therefore cost
`1.018` pUSD for a `1.000` pUSD floor. The alternative complete OR replication
already cost at least `1.041` pUSD from the three known YES asks plus the
composite NO proxy before adding the missing nonnegative Kharg YES leg. A Kharg
metadata request could not rescue that lower bound and was not made. The
composite rule ambiguity can only weaken the optimistic implication.

The exact family is terminal. Do not refetch, alias, request Kharg metadata,
reprice, or request books or fees. Reopen only for a distinct rule-complete
family with a prospectively observed strict side-specific sub-floor package.
No network, credential, account, order, fund, transaction, or protected capture
was used. Canonical adjudication SHA-256 is
`95d820d802b5eb4f327e33ec025b3d6d4a5e6b2e0693ef4b7d3f83e0a069e307`.

## 2026-09-01 Polymarket US participant-program exact-URL deduplication

Rank 58 selected the exact official URL
`https://docs.polymarket.us/incentives/user-programs.md`. Before freezing a new
request, retained contracts, source results, and request journals proved that
the same URL had already been prospectively captured under rank 51 earlier that
day. Its 4,959 hash-bound bytes say Daily Trading details will be published once
that program is live and say the same for Deposit and Trading. The overview
names generic deposit and trading requirements plus possible credits but gives
no deterministic live eligibility, task, reward, cap, timing, expiry, payment,
withdrawal, reversal, or after-cost economics.

The correct action was offline reuse, not a duplicate request. Rank 58 is
terminal on the retained current source, with public and stable account-
qualified floors zero, and is not accepted or deployment-ready. The mechanism-
index inventory had deduplicated economic family names but missed exact selected-
URL reuse. `AGENTS.md` now requires every index-selected source to be checked
against retained contract, result, and journal URLs before a freeze; a new
family label cannot override a consumed source retry policy.

Do not refetch or alias the exact source without an independently known material
change. Never manufacture deposits, trades, volume, days, accounts, referrals,
or eligibility; self-trade, wash-trade, spoof, coordinate abuse, manipulate, or
use promotional credit face value as cash. Every onboarding, deposit, trade,
order, fund, or state change requires separate explicit authority. All five
families from the retained mechanism index are now adjudicated. Continue with
the highest existing registry hypothesis only when its exact retry trigger is
independently satisfied.

Rank 58 adjudication SHA-256 is
`cba93ed78972affef86459fba0ad070431b803087d404bb7b9fe92eb284e17dd`.
Reused retained contract canonical/file, source-result, raw, and journal hashes
are
`9f237e7b0092d60553364c396d20822d2f7fe805535d451c4114a38c94839c92`,
`9b24ea0835bde9c3e76289aa6a728767950bcf8e3f6cff211ecf11e4f7fe4dc2`,
`8bf6ca023bf62623b5c0334a4866f3df4d5cbe4f9460e333b8c620d46c99caad`,
`3a1987bbff88abf718c43b190db8b40d040ab5ba9c42141920e250e186f9d1ec`,
and `c22e4a529233a4fb2dbf2c80008c86b1073a36b5e9b0191e3eb7a1e42992ac0c`.
Registry and durability-audit SHA-256 values become
`c038dde89dc80e2712d8adbbcd90056b27b625be062467f34b592e233e5e7eb8`
and `bcf28988454195e319db22ba3f31ef1ad1e13d157d332ff84b2c7a7ad4981556`.
Accepted edges remain 35, ranked hypotheses remain 58, terminal families become
172, and stable current account-qualified after-all-cost edges remain zero. No
network, account, credential, signed request, deposit, trade, order, fund,
transaction, or protected capture was touched by this adjudication.

## 2026-09-01 Polymarket US combo RFQ source gate

The exact rank 57 source identified by the retained official documentation
index was prospectively frozen before access. One current official Markdown GET
retained 4,071 bytes. The Retail RFQ API is authenticated beta for explicitly
enabled users. An RFQ requests two-sided liquidity for an exact combo symbol;
the lifecycle includes quote creation or replacement, acceptance of one side,
and maker confirmation during last look. Its private stream is best effort,
without replay or a separate subscription acknowledgement, and reconnect
requires reconciliation through authenticated REST reads.

The decisive execution semantics reject an atomic-fill assumption:
`QUOTE_STATUS_EXECUTED` and `quoteExecuted` mean paired exchange orders were
submitted and their order IDs recorded, not that the orders filled. RFQ orders
enter the normal combo order book, may trade with other resting liquidity, and
can leave unfilled quantity resting if either side enables `restRemainder`.
Using an RFQ is optional because the combo can be traded through the normal
Orders API.

The overview proves no authentic quote, atomic fill, guaranteed price
improvement, fee, last-look rejection rate, remainder configuration, exact
payoff, or after-cost result. Public and stable account-qualified profit floors
are zero. Rank 57 is terminal on public mechanics and is not accepted or
deployment-ready. An authentic finite-size comparison would require explicit
separate quote-only authority, beta-enabled account access, a prospectively
frozen independently required exact combo and side, a payoff-identical public-
leg counterfactual, complete fill/remainder/reconciliation/failure costs, and a
fresh quote frozen before access. Quote acceptance and every state change would
still require separate authority.

Do not refetch RFQ, combo, endpoint, schema, streaming, institutional, trader-
guide, HTML, FAQ, or search aliases. Never manufacture RFQs, quotes, orders,
fills, volume, counterparties, or activity; self-trade, wash-trade, spoof,
manipulate a reference price, abuse information, or abuse cancellation. Rank 58
participant Daily Trading / Deposit and Trading programs was the next source-
qualified hypothesis at that checkpoint and is adjudicated above through exact-
URL retained-source reuse.

Contract canonical/file, source-result, raw, journal, and adjudication SHA-256
values are
`dca10fd4fb7ff27d8c9e88670cbc57e79d5a12e683f4ea18d7dbc22f5415fd96`,
`c51e5372f9218a79a480652d626cfe03ef5e06eb26b63643344f485ce548185e`,
`210cb590e3bbdd50ab653a80f5265d461237d7f5a9e162afa2165e320472d43e`,
`ab97a5110e8f5ee7aae0996cb6931fdea52a813c78512a7a3a01106bb4414c4b`,
`bee6c4768ac645746def16ae8be8d1cd0ff0ff3a41a72991012079304af289c7`,
and `d1a4075cde9f3f3bd830067a8b70ae1f1c3a19a7de97e8788f16d6536c42bc20`.
Registry and durability-audit SHA-256 values become
`7a6dd4d5cb7aed3b1dc15062ce017387b873ea64da9c7f3a5feb62c894c3d077`
and `e6b22e14da4128fa3e1b37c5214776360dcb41651e77f6cef47348abe5416e02`.
Accepted edges remain 35, ranked hypotheses remain 58, terminal families become
171, and stable current account-qualified after-all-cost edges remain zero. No
account, credential, signed request, authenticated access, RFQ, quote, order,
fund, transaction, or protected capture was touched.

## 2026-09-01 Polymarket US market-maker source gate

The exact rank 56 source identified by the retained official documentation
index was prospectively frozen before access. One current official Markdown GET
retained 488 bytes. It says the program rewards approved market makers for
providing liquidity across a wide range of contracts in given categories and
provides an institutional contact route to learn more or apply.

The page publishes no deterministic approval criteria or guarantee, eligible
categories or markets, quote uptime, size, spread, depth, inventory or fill
obligations, performance measurement, reward formula, amount, asset, cadence,
cap, discretion, withholding, clawback, suspension, termination, renewal, or
change rights. It therefore proves no positive public after-all-cost floor and
no profitable liquidity strategy. Accepting an exact-positive future owned
payment predicate would be tautological and add no predictive or deployable
edge. Rank 56 is terminal on current public terms and is not accepted, account-
qualified, stable, or deployment-ready.

Do not refetch or alias the consumed source, contact or apply, or manufacture
quotes, orders, fills, liquidity, volume, accounts, or eligibility. Never self-
trade, wash-trade, spoof, abuse cancellation, or post non-executable quotes.
Contact, application, negotiation, onboarding, configuration, quote, order,
cancellation, fill, transfer, fund, and state changes require separate explicit
authority. Rank 57 combo RFQ was the next exact unconsumed source-qualified
hypothesis at that checkpoint and is adjudicated above.

Contract canonical/file, source-result, raw, journal, and adjudication SHA-256
values are
`d3fb2c4ac6975acc612760624fc126ae382ff232e8f8e2e1f8e67577f37038bb`,
`af73d12cad2f1d459dcda88e29cfe78e63baa78d707f9324b58d8bf76846be92`,
`4acef72e1c83f2861df3a099e189355b30d6e7f15a5d54728a120f6299aa39b1`,
`d4b568df596d93c7e7d1b7da7d74602349449b59535644abafe2909dc7245d42`,
`07d538a24afc46dd5d1d4adac8d34a6a3890dc1fed5db392536154f3caaac20e`,
and `af80c16faa5666735cc74f17d5212ec95d226bc0ed58e8d247811eab06ded381`.
Registry and durability-audit SHA-256 values become
`afdc6b4f6d82f4011ce723076a9cff22e1cf21c45a4995a28ca9bfca83a86b4e`
and `b6e2d132f31b0879214ef2e81658bd48fa781bfaaea89536a8c1eb2c13c98f18`.
Accepted edges remain 35, ranked hypotheses remain 58, terminal families become
170, and stable current account-qualified after-all-cost edges remain zero. No
account, credential, signed request, contact, application, quote, order, fund,
transaction, or protected capture was touched.

## 2026-09-01 Polymarket US partner vendor-fee overlay

The exact rank 55 source identified by the retained official documentation
index was prospectively frozen before access. One current official Markdown GET
retained 8,591 bytes and proved a distinct market-direction-independent cash-
flow primitive for approved partners. A partner computes a fixed USD fee under
its agreement with a participant; Polymarket validates that the declaration is
well formed, non-negative, and allowed for the firm, then records it against the
order. No fee cash moves at order time, and the declared amount is not
automatically adjusted for partial fill or cancellation.

The resulting accrual is an unsecured receivable. Polymarket does not reserve
or underwrite it, no API exposes accrued balance, and participant trading losses
can cause collection rejection for insufficient funds. The partner must keep a
shadow balance and reconcile a daily report whose delivery method remains TBD.
Collection is one net participant-account-to-partner-funding-account
`VENDOR_FEES` transfer at most once per participant account per day. The page is
beta and subject to change.

The thirty-fifth accepted scope is only exact positive owned USD vendor-fee
cash actually collected on bona fide independently existing external
participant orders under an already approved integration and a disclosed,
consented agreement, after uncollectible receivables, waivers, refunds,
reversals, acquisition, support, compliance, tax, opportunity, and every
incremental cost. This is external-user business revenue rather than autonomous
trading profit. The public after-all-cost floor is zero; account qualification,
collectability, owned receipt, persistence, after-cost margin, stability, and
deployment readiness remain unproved.

Do not refetch or alias the consumed source. Never treat a declaration, accrual,
report row, or receivable as collected cash; create fake or related users;
manufacture orders, volume, or activity; or charge undisclosed, unconsented,
deceptive, excessive, or disallowed fees. Contact, application, onboarding,
configuration, order, fee declaration, transfer, collection, fund, and state
changes require separate explicit authority. Rank 56 approved market-maker
contracts was the next exact unconsumed source-qualified hypothesis at that
checkpoint and is adjudicated above.

Contract canonical/file, source-result, raw, journal, and adjudication SHA-256
values are
`10f643129214fe6d104ba04b119884e1880a0478a61df20c1c062907b61a82b7`,
`801f416555efdd418f1e0e5d26642badda0f3eb97e1a944128154ba32ae023e4`,
`6fc0c7918d461f21fb352fdf2af95e7ba6eaa77ddaa0f3d1ac0760507559fb71`,
`5ee5063318cf5c5ad4991fa1c0a52430a03d91df2981021e5bcc96f83c18eae8`,
`6e774dcc56c1a658e021656bd133988f02f9eacf42fccb31744cb464898fb3d3`,
and `92b881d5b426b6e271550dbd3f3fa6b73b1c8164d326d2885528dc4524f57b25`.
Registry and durability-audit SHA-256 values become
`d2f93c88d50905be28cefe0f215e72137c8b9bbb7f9d73741b87925691413b44`
and `8fb854400727c2b40666047a8df5ac42b183082db8c8ce2f3c77ce16a95828af`.
Accepted edges become 35, ranked hypotheses remain 58, terminal families become
169, and stable current account-qualified after-all-cost edges remain zero. No
account, credential, signed request, order, fund, transaction, or protected
capture was touched.

## 2026-09-01 Polymarket US mechanism index and referral gate

One prospectively frozen exact official `llms.txt` GET retained 47,687 bytes:
350 lines, 342 link rows, and 300 unique absolute HTTP URLs. Offline
title-description deduplication found five unrepresented economic families:
Referral Incentive, partner vendor fees, approved market-maker contracts, combo
RFQ execution, and participant Daily Trading / Deposit and Trading incentives.
The frozen campaign allowed exactly one linked economic source, so Referral
Incentive advanced under its own contract and the other four remained unopened
at that checkpoint.

The 446-byte referral page says approved affiliates receive rewards for
referring new Participants and provides only a contact address. It publishes no
approval criteria or guarantee, eligibility, attribution, qualifying activity,
payout formula, cap, duration, payment asset or timing, expiry, anti-abuse,
reversal, or program-change terms. The overview's `Open` label cannot override
the exact approved-affiliate condition. Public and account-qualified floors are
zero, and accepting a tautological exact-positive-payment predicate would add
no predictive or deployable value. Rank 54 is therefore unaccepted and terminal
on current public terms. Rank 55 partner vendor fees was subsequently consumed
and adjudicated above; ranks 56–58 remain source-qualified in order.

Do not repeat the current index while its raw SHA-256 remains
`308b0a29cf69d8b50fd9f5c841a21b51cebb5842d988fe4d53f77322fd980d60`;
do not refetch the referral page or open the Refer-A-Friend FAQ alias. Never
create fake users, accounts, KYC, referrals, deposits, trades, quotes, fills, or
volume. Contact, applications, invitations, onboarding, account access, orders,
funds, and state changes require separate explicit authority.

Index contract/source-result/raw/journal/adjudication SHA-256 values are
`7625c1acceffc154492429b20b5218ab32f5f9e77f2e180e67377c8f6eb4866f`,
`43eb515fb75c5a1891c78c998e261d8f8f2c60707c5034db0d7aec9e70314c52`,
`308b0a29cf69d8b50fd9f5c841a21b51cebb5842d988fe4d53f77322fd980d60`,
`2a1405b81ed8d586fe1a18d445febcbbd223451f840b022ed15fa0ffad320522`,
and `214f9807ed7d9120478be6e75daf889ee8aeb30f8bea194c3af8d5eb4e8c251f`.
Referral contract/source-result/raw/journal/adjudication SHA-256 values are
`cb8dca1749f6c2d58e60d9a72d993072a45be22308c7b13771481c25b98d3d11`,
`62494e77c882afe747e87056d64661f2abc1f80e58bcee4a1725cc4f8e43d935`,
`626d5f5ff0b21598dbf4d89cc643cc4b14a6d619a8a0b269782122b65c165fa2`,
`383076375ec8ce2659c3bb5fe15feb789279942cdfd8e230d56cb73d1d4bc362`,
and `87cce3f6175f12b2b5604db57d0f35fcfec1868577d0e0d92a99a82e189bf6d6`.
Registry and durability-audit SHA-256 values become
`8041cee138044411dc4c8357c7dc8e0e8b8a4cc5b4493f3ddc475db4aac1303f`
and `454c2c0e2f3f3eea07a58c6d8ec92be3dd85aac39d48b26d836b415c3d29c3bb`.
At that checkpoint accepted edges remained 34, ranked hypotheses became 58,
terminal families became 168, and stable current account-qualified after-all-
cost edges remained zero.

## 2026-09-01 Binance Web3 prediction OTC source failure

Official Binance discovery exposed a previously unrepresented Web3 Wallet
Prediction Trading OTC block-trade surface. It is potentially a
direction-independent price-concession overlay for an independently required
legitimate large prediction trade, and is structurally distinct from Binance
Exchange Spot block matching and Polymarket's public CLOB. The exact developer
page was frozen before access. Its one-use public GET returned HTTP 202 with
zero bytes, preserved in the raw response and durable request journal.

The search-rendered endpoint and `secretToken` examples are discovery-only.
They prove no current public quote access, authentication contract, authentic
counterparty, minimum size, fee, settlement, fillability, payoff identity,
public-book counterfactual, or after-cost concession. Public profit floor is
zero; the candidate is not accepted or deployment-ready. Do not retry locale or
rendered aliases, use cached snippets, guess tokens, or request private quotes.
Reopen rank 53 only on a material byte-retainable official documentation/API
change, or with explicit separate read-only authority, an independently
required legitimate large prediction trade, and an exact same-payoff
public-book counterfactual. Every quote acceptance, cancellation, negotiation,
order, fund, wallet, or account mutation requires separate explicit authority.

Contract canonical/file, raw, journal, and failure-adjudication SHA-256 values
are
`2441d38840ceb9a442a649b5631c2ef0c8ef73c58afdc61f7b015cbc91f92b50`,
`e90e728adc4ffce850a6b1789b298cda30cac1dc5a03b2fd25a67dc8710f4505`,
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
`5ff12552f4dba7fa15210bf13a073f4c6f01d8646f4e288b3bf930b91c200961`,
and `02a2085182a0ff292aaf5b7f4544a3d4d3c5a813bab9e7adca9638fac8500525`.
Registry and durability-audit SHA-256 values are
`9b2236283e5ec3c877be2fa0cabb423699d3708bd2cb1a6d656e26dace4aaefb`
and `dfb85ebfcc78b7d05da92036860701dbdae54953c04911a1fc6e47cad3dbd762`.
Accepted edges remain 34, ranked hypotheses become 53,
terminal families become 166, and stable current account-qualified after-all-
cost edges remain zero. No account, credential, signed request, quote, order,
fund, wallet, transaction, or protected capture was touched.

## 2026-09-01 Binance-Bybit funding-spread rejection

A genuinely distinct direction-independent cross-venue funding family was
cleanly preregistered before any funding value was viewed. Official Bybit
sources established the public funding-history endpoint, exact linear
instrument fields, funding interval, and signed-rate cash-flow convention. Six
bounded public unauthenticated GETs then proved live linear USDT-settled
BTCUSDT, ETHUSDT, and SOLUSDT perpetuals and retained exactly 90 eight-hour
funding rows per asset aligned to the existing Binance population.

The training-only orientation was long Bybit and short Binance for all three
assets. Zero assets survived every chronological 45/22/23 training,
validation, and test role after the frozen 20-bip round-trip execution cost,
10% annual two-leg capital hurdle, and 25-bip cross-venue custody, transfer,
latency, and failure stress. BTC gross funding was positive in every role but
economically dominated; ETH and SOL validation gross funding was negative.
Every test net was worse than -62 bips. No basis or book request was justified.

Do not paginate, resample, reverse after observation, alter roles, weaken
costs, or request basis or books for this population. Reopen only on a material
Bybit or Binance funding, fee, quote-unit, custody, transfer, latency, capital,
or execution-architecture change. Preregistration, adjudication-contract, and
result hashes are
`6cbf711127949566a3ad8e7dc50f3bd38700e96987ca06e6d1a3f75abee77480`,
`00cfb669c0a0300ba1b1de919e03910d0e1aa1c33c1955b696625973a2da0963`,
and `eb8d9badab2dd2d4869b2d52154a49815a2bb9b7034098bdf475163ac8b9303f`.
Accepted edges remain 30, ranked hypotheses remain 47, terminal families become
145, and stable current account-qualified after-all-cost edges remain zero. No
account, credential, signed request, order, fund, transaction, basis, book, or
protected capture was touched.

## Polymarket US portfolio-margin collateral return

The current official documentation index exposed two previously unrepresented
capital-efficiency mechanisms. Both exact sibling sources were frozen before
either was opened, preventing the first source from selecting the second. Two
one-use public Markdown GETs then proved:

- mutually exclusive collateral return applies when the account feature is
  enabled and the participant holds shorts across instruments in the same event
  where exactly one outcome can occur; the documented two-short example reduces
  margin by the smaller offsetting position size;
- directional collateral return applies only when a lower-ranked long offsets a
  higher-ranked short in the same ordered event; the reverse direction receives
  no relief, one long can offset multiple higher shorts, and the matching
  algorithm starts with the highest-ranking eligible short and long;
- freed buying power can be used only in different events, not the event that
  generated the release;
- closing an offset requires returning released collateral, and the close is
  rejected when sufficient buying power is unavailable;
- hypothetical collateral return from proposed new orders is excluded from
  pre-order buying-power checks.

The former promotional-credit edge remains revoked. The replacement
thirty-fourth accepted scope is exact positive owned buying-power release
actually credited on independently existing eligible mutually exclusive shorts
or lower-ranked-long versus higher-ranked-short directional positions, only
when released buying power has an independently justified positive after-cost
use in another event. This is deterministic market-direction-independent
portfolio-margin relief, not cash income, trading profit, reduced actual risk,
or a reason to create positions. Account enablement, owned before/after receipt,
positive different-event use, position lineage, exit ordering, close capacity,
persistence, fees, tax, and every incremental cost remain unproved. Public and
stable account-qualified profit floors remain zero.

Mutually exclusive contract/source-result/raw/journal SHA-256 values are
`e0d4a453180d9f14a72975a54c1d7d2356a4752734b8e2d737b9cc82bb800eda`,
`2ac41a12efc654017471472fe4c428d0b5f8f9a2dee903bae91a19f7a1bde83b`,
`7b56aad70dc5ea4550e31b429c052d7f2600774eb5940b330866f1f257a83164`,
and `0a86b98946a58aeb146b6cdad354e7f964cb4d1f1883644ee0d94f57316a5974`.
Directional contract/source-result/raw/journal SHA-256 values are
`773b300f89697ebee8b9dc992dcd9d979dc46cdc5da07e40e23eb3aa580067cd`,
`db19108f54904e67ee9cad7b3150ea0d8629cd721a1dbbc88dd2604e1bf8b72e`,
`2b0556c4d4f41c10ae8654eaffd04356320489fdfa2b46c644b2c54efd179362`,
and `40d3e25dbf2b4f780106d3ebcb085aab0a36b07393cb3d549c5f0a7b7785dc05`.
Adjudication, registry, and durability-audit SHA-256 values become
`d1e50f0f650f09b2180eb2b9312c07c9e4004612d8f86831ebf73411d22630a2`,
`22f0e5d5c557e2626eff4094ba938a8e81a3d8be07a794f5d3a81d8f053f046a`,
and `f8fb509a2dba3086228f7b0ea2fb804269bac9206858f89b80626c947b9f8ab0`.
Accepted edges return to 34, ranked hypotheses become 52, terminal families
become 165, and stable current account-qualified after-all-cost edges remain
zero. No price, BBO, book, fee, account, credential, order, deposit, fund,
transaction, or protected capture was touched.

## 2026-09-01 Polymarket NFL period moneyline-spread rejection

One frozen zero-network adjudication reused the exact complete retained
September 13 early NFL catalog without a refetch. It exhausted 227
direction-independent quarter and second-half relations pairing the favorite's
same-period moneyline outcome with the opponent side of a negative
half-integer favorite spread. A favorite win, opponent win, period tie, and
whole-game cancellation pay at least one pUSD; the period-tie payout is 1.5
pUSD and overtime is excluded by the retained period rules.

Of those relations, 215 had complete side-specific rejection prices using only
retained `bestAsk` or conservative `1-bestBid`; zero cost strictly less than
their one-pUSD guaranteed floor. The best complete package was Eagles 2H
moneyline plus Commanders against Eagles -0.5 in the 2H spread at 1.53 pUSD per
share. The remaining 12 relations are price-incomplete because retained
selected-side quotes are missing; missing quotes are not free legs.

Do not refetch, repair, reprice, select a sibling, or request books or fees for
this consumed period-margin population. Reopen only for a distinct unconsumed
population or a material rule, fee, tick, or market-architecture change.
Contract and result hashes are
`f6526f49a92a7822be866a6801fbef8a3546c6df3ba7f1c2100ce8b59673e3c4`
and `e58f7dab3c752863a4a546c83141cac0e3177c0d27d16b87d1217baed9549706`.
Accepted edges remain 30, ranked hypotheses remain 47, terminal families become
144, and stable current account-qualified after-all-cost edges remain zero. No
network, account, credential, signed request, order, fund, transaction, or
protected capture was touched.

## 2026-09-01 Polymarket September 13 NFL team-period graph rejection

One frozen zero-network adjudication reused the exact complete retained
September 13 early NFL catalog without a refetch. It exhausted 84,508
direction-independent relations across all 12 events: full, team, half,
quarter, and team-period total ladders plus team-to-game, team-to-half,
halves-to-game, team-halves-to-team, quarters-to-half, and quarters-to-game
additive covers.

Of those relations, 76,607 had complete side-specific rejection prices using
only retained `bestAsk` or conservative `1-bestBid`; zero cost strictly less
than their one-pUSD guaranteed floor. The best complete package was Buccaneers
vs. Bengals Over 51.5 plus Under 52.5 at 1.04 pUSD per share. The remaining
7,901 relations are explicitly price-incomplete because retained selected-side
quotes are missing. Missing quotes are not zero-cost legs and block any
escalation.

Do not refetch, repair, reprice, select a sibling, or request books or fees for
this consumed catalog graph. Reopen only for a distinct unconsumed population
or a material rule, fee, tick, or market-architecture change. Contract and
result hashes are
`91272f05bbef81ba0dbf9b2bddee2d5d3418cdd12f33d9ef060cc2b582f5c560`
and `556ee348cf5246f422df80da958a2d96402a2900dc88878cd561fb02f1bd5b02`.
Accepted edges remain 30, ranked hypotheses remain 47, terminal families become
143, and stable current account-qualified after-all-cost edges remain zero.
No network, account, credential, signed request, order, fund, transaction, or
protected capture was touched.

## 2026-09-01 Polymarket UAE-to-any-Arab creation-gap rejection

Rendered discovery found an apparently direction-independent five-cent cover:
`NO(Iran targets the UAE by September 15) + YES(Iran targets any Arab country
by September 15)`. Two prospectively frozen, one-use, public unauthenticated
Gamma GETs retained the exact UAE and aggregate deadline events.

The country subset, qualifying military action, interception, territory,
attribution, proxy exclusion, reporting-conflict, and source definitions align.
The GST cutoff is one hour earlier than the AST aggregate cutoff and is also
deadline-safe. The condition starts are not: UAE begins at
`2026-08-18T22:49:24Z`, while the aggregate begins only at
`2026-08-31T18:37:55Z`, leaving 1,108,111 seconds uncovered. A qualifying UAE
action in that gap followed by no qualifying aggregate-interval action makes
both proposed tokens pay zero. The guaranteed floor is therefore zero.

The source gate failed before price inspection. No `bestBid`, `bestAsk`, or
`outcomePrices` field was used for the decision, and zero CLOB book, fee,
account, credential, signed, order, fund, transaction, or protected requests
were made. Do not retry or assume a later aggregate condition inherits earlier
subset history. Reopen rank 31 only for source-proved aligned starts or complete
intervening-event evidence that removes every creation-gap counterexample.

Aggregate and UAE contract hashes are
`856e8ad6a4ca83ff41267950794b75ee783e350ca4ad55c9c6f6c196cc7dcef2`
and `d17cf334d4fbecaf269e6c3b2d779d57dfb61be5b0a1414a29ba0b8a06dcb88b`;
capture-result hashes are
`a5dfe50ee40388c843677275def2d166657e103a2287ba717cc66ddc47e45603`
and `c41d4ce06afa52ba0308f92e51b9c48b1ccd5b6cc5cecb49c4198314cbf229ce`;
the adjudication hash is
`c6634add2f5fac96fe5c3b08c4236b3a0ad483c37efea03bd1dd1ab41e398941`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
142, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit hashes are
`aa36f2488fa336e45fbff6f31e9644612e3a2d6faf78138f634b00bfb08194f5`
and `c9c1d57cc402cd54372b59c12b2052238e9ab6b8eb68732fa1427dcf56971508`.

## 2026-09-01 Polymarket public-company rank assignment source rejection

Three prospectively frozen, one-use, public unauthenticated Gamma GETs retained
the exact largest, second-largest, and third-largest company events with 25,
29, and 29 markets. The market-cap metric, September 30 market-close instant,
and consensus-of-credible-reporting source align, but the cross-event payoff
proof does not.

Each event is an independently resolved NegRisk partition and none specifies
how exact market-cap ties or corporate actions map across first, second, and
third place. The hidden `Company`-letter populations differ by rank and have no
source-proved identities. `Any other company` can also denote different
companies at different ranks. Therefore an all-different matching or
permutation floor would require unstated tie and identity assumptions.

The source gate failed before price inspection. No `bestBid`, `bestAsk`, or
`outcomePrices` field was used for the decision, and zero CLOB book, fee,
account, credential, signed, order, fund, transaction, or protected requests
were made. Do not retry, alias-map placeholders, or assume ordinal tie
semantics. Reopen rank 31 only if exact rules add deterministic cross-rank tie
and identity semantics or a distinct family proves them before economics.

Rank-one, rank-two, and rank-three contract hashes are
`c60773469228f8b94f43e62fae1b8fcacf87c69f2a97cb2ed075656bd2a1555d`,
`77806e15cd1326385ecdf3860c760a11f24df8612b77d570dab30e6b86fdc862`,
and `c89545cb8fe40944bff0f44b978573f9fec215cdaa9758847a8ca5c8dc9341f8`;
capture-result hashes are
`e338dfd907f90c758aee7c04aa5cabd4ef7391eff34bb303d6289c22c370869c`,
`59cc6a074cbc29088490d7208481a90fa4750861e616a824b9f8790f2457c97b`,
and `c945347331b14bbb5ace8c68660fa3169e99b4e62fae78a0347ea74fd726f077`;
the source adjudication hash is
`47d7c9f000b9e8445d227a4b9b35ebeaf0681e8b6bfbdaaf2ebcb359e054839e`.

Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
141, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit hashes are
`9c793173647d9a3f50e0be1f3696a16614517fb6d921dc9efb9f2151532b2939`
and `0d25d77c6d84cad2bb356d15469cea7f6d0c3032e38ab18e0040388ed29466c4`.

## 2026-09-01 Polymarket Mythos release implication-graph rejection

A distinct direction-independent release-timing family was tested as one graph,
not as repeated hand-selected pairs. Two prospectively frozen, one-use, public
unauthenticated Gamma GETs retained the exact 38-market Mythos release-date
event and 10-market cumulative-deadline event. A separately frozen zero-network
adjudicator then exhausted 208 creation-order-safe implication,
mutual-exclusion, and no-release-exclusion packages across 30 future exact dates
and eight deadlines.

Zero of 208 conservative side-specific metadata packages cost strictly less
than their one-pUSD payout floor, and zero survived current taker fees plus one
adverse tick per leg. The best metadata package was `NO(no release through
September 30) + NO(released by September 7)` at 1.17 pUSD per share. At the
five-share common minimum it loses 0.99878 pUSD after the frozen fee-and-tick
stress, before depth, time value, non-atomicity, latency, unwind, capital, or
operating cost. No CLOB book or fee-endpoint request was justified.

This exact two-event population is terminal. Do not retry, refresh, alias,
reprice, or book-capture it. Reopen rank 31 only for a distinct unconsumed,
source-proved release family or a material rule, fee, tick, or market-
architecture change. Exact-date and deadline metadata contract hashes are
`bcaf6ca43afab29fa4d8dcd614e36522e93bfbf9ec09761a19e7efbe63052765`
and `482f3afab76c71f60ce049cfb310d64decfcadea7294d5f9f0218860d32cca3e`;
capture-result hashes are
`db3f62ddb586c898b3a8b9e2258cb6db005c3c7c66253093afe4b73738a3e13c`
and `2b51a74a707a29029ba7d1392adc56115f4caf08b1a5b4406008fb9dbfca6ab8`;
graph contract and adjudication hashes are
`fa9dc2f724d03cf869865ed1bd0f40b46c2274c11e1c121d0290757c2446f65b`
and `f515939e699129b50b7570fc7157cc9d9f70951a3b8a3127555152603ddf782e`.

Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
140, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit hashes are
`98a39ba8c2a565104cd49df26b41e711fc770c57433a26fdb6d90ba20d6f6300`
and `fa699d282419535be266c71a5e0ff465cde3ecddbf7fbb4ce15a0c63f30c86ff`.
No account, credential, signed request, order, fund, transaction, or protected
capture was touched.

## 2026-09-01 Polymarket Gemini Pro interval/deadline parity rejection

A distinct direction-independent candidate passed rendered discovery:
`NO(no Gemini Pro release by November 1) + NO(Gemini Pro released by October
31)` appeared to cost 0.97 pUSD for a one-pUSD floor. Two prospectively frozen,
one-use, public unauthenticated Gamma GETs retained the exact interval and
deadline events with 11 and 21 markets respectively.

The substantive qualifying-model, GA-promotion, public-access, private-beta,
placeholder, ET-calendar, and resolution-source rules align. Closed No siblings
also exclude a qualifying release during the roughly ten-hour selected-market
creation gap, so the same-target implication survives the optimistic rule gate.
The exact conservative acquisition proxies do not: `NO(no release by November
1)` is `1 - 0.33 = 0.67` pUSD and `NO(released by October 31)` is
`1 - 0.67 = 0.33` pUSD. Their 1.00-pUSD sum only equals the guaranteed floor
before fees, ticks, depth, time value, non-atomicity, latency, unwind, capital,
or operating cost. The strict sub-floor gate failed, so zero CLOB book or fee
endpoint requests were justified.

This exact two-event population is terminal. Do not retry, refresh, alias,
reprice, or request books or fees. Reopen rank 31 only for a distinct,
unconsumed, source-proved same-target interval/deadline pair whose conservative
side-specific package is strictly below its floor, or a material rule, fee,
tick, or market-architecture change. Interval and deadline contract hashes are
`56fdf5aa9445190d1e2683f68a1f77fa9b5590e9535f05c31e7457bd5aeb68b9`
and `d9c3633af7c754ba0aef5fdcc319eb3cf03b8a083ecb19ff2c6056007983cb8e`;
capture-result hashes are
`748cdb55075ff118848ea71608622571e89912763a33a271eef0674a84ad4309`
and `7748a2c988adf2cc4efebed0686ebc8bd49ad6f1f23166b2096de99ffbf14df5`;
adjudication hash is
`a1e7562a56408bf3634f60585a6b64be89e401b7f1a25875762eed42c6908ac6`.

Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
139, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit hashes are
`097e92b82ef74899f08c972747d268fd4554beb92f1fdbcb23c5316767a19703`
and `62ebb5280c1d3ded0a0298c97ac1cad0189d733be57918e9bf3a1da4bbb4884f`.
No signed request, account, credential, order, fund, transaction, or protected
capture was touched.

The pre-access contract hashes were initially computed with PowerShell
`ConvertTo-Json`, which does not reproduce the repository's canonical sorted,
compact ASCII JSON. Offline validation had not run and no output path or network
request existed. The hashes alone were corrected with the Python canonicalizer,
both absolute-path validators passed, and only then were the exact one-use GETs
consumed. `AGENTS.md` now prohibits that serializer substitution.

## 2026-09-01 Binance SHEIN quanto to native-stock exact-share rejection

The new `HK0625USDT` SHEIN quanto announcement also satisfied rank 38's literal
new-TradFi-instrument trigger. The cheapest decisive question was whether
Binance's native Stocks product could supply the exact same HK0625 Hong Kong
share, not whether a company-name-related US security existed. Current official
Binance Stocks documentation limits the native product to US-listed NYSE or
NASDAQ common shares and US-listed ETFs, with bare US-equity symbols and USDC
as the default quote. That universe cannot contain the HK0625 Hong Kong share.

A source-retention contract was frozen for the exact official documentation
page. Its one permitted unauthenticated GET returned HTTP 202 with an empty
body. Preserve that failure; do not retry an alias or alternate download. The
separately rendered official page is only a negative universe gate and cannot
promote or price a candidate. An ADR, US wrapper, tokenized stock, bStock,
Ondo, xStock, company-name match, or other share class is not fungible with
HK0625.

The branch is terminal before native quote streams, futures, FX, funding, fees,
or books. Contract, failure, and adjudication canonical SHA-256 values are
`3fd064e94d8e591d874a816c495c34b878fb0f6979662a7dcca03625ab43c9d0`,
`4baf2f1b3a5bce69cc23575234c4ee27405136c222f8d60904adc0ba4e2bec73`,
and `6ff02e16bc0c8e8f6e833da8a221eee532868b103b6ef078ca16b72495ecf139`.
Reopen rank 38 only for a future official listing whose exact share class
exists in the documented native US-stock universe or a material product-
universe or stream-architecture change.

## 2026-09-01 Polymarket pairwise-to-multiway tie rejection

A distinct September private-company valuation-growth projection was rejected
without a Gamma, CLOB, or fee request. The pairwise Anthropic-versus-OpenAI
market resolves equal percentage growth 50-50. The five-company greatest-
growth market instead awards one full winner among tied companies using the
highest final valuation. Therefore `NO(Anthropic wins five-way) +
YES(Anthropic beats OpenAI)` does not have a one-pUSD Boolean-implication
floor. In the state where the two companies tie and Anthropic wins the five-way
tie-break, it pays only 0.50 pUSD.

The required five-way `NO` leg alone displayed 0.97 pUSD. Total package cost is
therefore at least 0.97 pUSD against the correct 0.50-pUSD floor: at most
-0.47 pUSD before the missing pairwise leg, fees, depth, time value,
non-atomicity, latency, or unwind cost. The exact family is terminal before any
venue API request. Reopen only when pairwise and multiway tie payouts are
aligned or exact tie coverage is included and the complete rendered package is
strictly below its tie-aware floor. Canonical adjudication SHA-256 is
`bcb49e2e1bdba407fe8121a37ff56dcddd770b1957e1add2c95f225fdb76fded`.

Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
138, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 is
`06e9c43228f09c23d6ad26dde21216eed7d350fbfe19453380a7bf10247565b9`;
durability-audit SHA-256 is
`a6c3d63d971bf16aacfec64492dc13660c203d93e740c4fade4122f2fc50a5d8`.
No account, credential, signed request, order, fund, transaction, or protected
capture was touched.

## 2026-09-01 Polymarket joint-to-marginal state-cover checkpoint

A distinct direction-independent Boolean decomposition passed the rendered
discovery gate: `NO(Democratic House control) + YES(D Senate, D House) +
YES(R Senate, D House) + YES(joint Other)` displayed at 0.983 pUSD for a
one-pUSD state-cover floor. The `Other` leg was mandatory; omitting it would
leave an unmatched joint state with Democratic House control uncovered.

Two prospectively frozen one-use public Gamma GETs retained the exact joint and
marginal events. The joint event is an active five-market NegRisk partition
containing DD, DR, RD, RR, and `Other`. Its House rules align with the marginal
event on control definition, seat universe, runoffs, party attribution,
independent caucusing, ambiguity fallback, end date, and sources. The marginal
event actually contains nine markets, including seven hidden inactive siblings,
but the selected Democratic market's `NO` token still pays in every state where
Democratic House control is false; no hidden sibling was priced as free.

Exact Gamma rejection prices tightened the four-leg sum to 0.993 pUSD, leaving
only 0.035 pUSD gross headroom at the five-share minimum. Every selected market
has the current 0.04, exponent-1, taker-only fee schedule. The source-bound fee
curve charges 0.11778 pUSD across the four legs, so the package floor is
-0.08278 pUSD after fees and before depth, ticks, time value, non-atomicity,
unwind, latency, capital, or operational costs. No book or fee-endpoint request
was justified.

This exact two-event population is terminal. Do not retry, refresh, reprice,
alias, omit `Other`, or request books. Reopen rank 31 only for a distinct exact
joint-to-marginal family whose rejection-only package remains strictly positive
after current fees before depth, or a material fee, rule, or market-architecture
change. No credential, account, order, fund, transaction, or protected capture
was touched. Accepted edges remain 29, ranked hypotheses remain 47, and terminal
families become 132. Joint and marginal contract SHA-256 values are
`8201e7535d07b7340a612f10473a43718ed1efa350b6d804dd5762a57cab64b7`
and `1ce1e4450da6cc36e7b2c64702bad00ce52fc0b33b9937353ae8ecb16bdbf634`;
adjudication SHA-256 is
`738a9d89f9334e2cfc91f4c21b474528d5222cab222455856cb8289c055ac3ff`.
Registry SHA-256 is
`264076c9ab6c42043689bf0d53043c2520236c493859cfd1f159acaea78e0cac`;
durability-audit SHA-256 is
`a0c062913d17787d027b2973c947a5d8fb4d2be81d018ee90f9ca354034d6285`.

## 2026-09-01 Polymarket policy-partition discovery screen

A bounded rendered-page pass screened eight distinct, exhaustive central-bank
decision partitions before any Gamma, CLOB, fee, account, credential, order,
fund, transaction, or protected-capture access. Every known direct `Buy Yes`
subtotal rejected at or above its one-pUSD complete-set floor: Bank of Canada
1.005, Bank of Japan 1.024, Reserve Bank of Australia 1.006, Bank of Russia
1.018, Fed 1.016, ECB 1.009, and Bank of England already exceeded 1.024 before
its missing displayed side. Reserve Bank of New Zealand already exceeded 1.003
before its missing displayed `Decrease` ask. Missing or zero-rendered sides
remain unavailable, never free.

These values are discovery gates, not executable economics or accepted edge
evidence. Do not refetch these exact September decision pages merely to chase
mill changes. Reopen rank 31 only for a distinct, unconsumed, rule-complete
partition whose every required side is displayed and whose direct acquisition
sum is strictly below its exact payoff floor; then freeze one exact metadata
reconciliation before any depth or fee request. This pass changes no accepted,
ranked, or terminal-family count and requires no registry or publication-hash
rewrite.

## 2026-09-01 Polymarket streaming-service partition checkpoint

A distinct fixed-NegRisk partition passed the cheapest rendered discovery gate:
the eight visible `Which streaming service will win the most Emmys?` YES asks
summed to 0.99 pUSD for a nominal one-pUSD complete-set floor. The rendered
rules also named an unlisted `Other` fallback, so no book request was permitted
without exact metadata reconciliation.

One prospectively frozen, one-use, public unauthenticated Gamma GET retained
44,700 bytes and reconciled nine markets, not eight. The ninth market is
`Other` (`Will another streaming service win the most Emmys?`); it is inactive
and its direct YES `bestAsk` is 1 pUSD. The complete nine-market rejection sum
is therefore 1.99 pUSD for a one-pUSD floor before fees, depth, time value,
latency, atomicity, or unwind risk. Stop before books and fees. Do not retry,
refresh, alias, omit `Other`, or treat the rendered page as exhaustive.

No credential, account, order, fund, transaction, or protected capture was
touched. Accepted edges remain 29, ranked hypotheses remain 47, and terminal
families become 131. Contract and adjudication SHA-256 values are
`90525ae26b9d336473ad3e676bc8a229fc90bbabb0b8d7ceeb6230e15c1e5f30`
and `9ffdd9272bb4b18644aad2c42e713a270689c2fed2145cd91079ed6b7ba82c05`.
Registry SHA-256 is
`89cd6f271e75fb716682976870b5d74d4fd06b8a27c8933f88cb0019c993a533`;
durability-audit SHA-256 is
`7232a04530aff3090453f32f6f7819fa29e102975b6134775a6219a3a64beeba`.

## 2026-09-01 Binance-OKX funding checkpoint

A distinct Binance-OKX USDT perpetual funding family was preregistered before
any funding value access. One complete public inventory proved exactly one live,
linear, USDT-settled BTC, ETH, and SOL swap. Three one-use public history calls
returned 296 rows each and aligned the exact retained 90-bin Binance population,
so no Binance refresh, basis, book, account, credential, order, fund, or
protected-capture request occurred.

The 45-row training role selected long OKX / short Binance for all assets and
held that orientation fixed for 22-row validation and 23-row test. Gross
train/validation/test carry was BTC 3.66193/-1.34312/0.73452 bips, ETH
5.85134/-2.04475/-0.80671, and SOL 2.97859/-2.46280/1.08712. Zero assets
survived the frozen 20-bip round trip, 10% annual capital hurdle, and 25-bip
custody/transfer/latency/failure stress. Validation gross was negative for all
three, ETH test gross was negative, and every role was economically dominated
by the frozen hurdles. Stop before basis and books.

Do not paginate, resample, change alignment, refit orientation, or weaken the
costs for this exact population. Reopen only on a material OKX or Binance
funding, fee, quote-unit, custody, transfer, latency, capital, or execution-
architecture change. Preregistration and adjudication result SHA-256 values are
`68de39a4ff7db0b01a204f410b637f513fcdb5058705b8a34949991acc8a585b`
and `3bd57553ea5f40ac1141a48ee3095b02e3c6e6c41012de93d96d95ebe13b006e`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
130, registry SHA-256 is
`eb915168778e0824dc80bb405c2e1c5657916c0750e6f12bc50ea687024b88a4`,
and durability-audit SHA-256 is
`159f9940502865da19cdcc6df40f32c385497b723e31988725da5951286d061f`.

## 2026-09-01 Binance-Lighter funding checkpoint

A new direction-neutral cross-venue family was screened efficiently. One
public Lighter inventory request fixed active BTC, ETH, and SOL perpetual IDs,
then three one-use public hourly funding requests reused retained Binance
BTCUSDT, ETHUSDT, and SOLUSDT histories. No redundant Binance refresh, basis,
book, account, credential, order, fund, or protected-capture request occurred.
The endpoint's exclusive boundaries produced 718 hourly rows and 88 complete
aligned eight-hour buckets per asset.

Training-only orientation selection and fixed validation/test roles left no
survivor after 20 bips round-trip execution, a 10% annual capital hurdle, 25
bips USDC/USDT stress, and 25 bips custody/bridge/latency stress. BTC gross
train/validation/test spread was 3.1217/2.5119/-3.4251 bips; ETH was
10.0548/0.4170/3.1349; SOL was 4.4760/3.8258/15.1961. BTC reversed in test,
ETH failed validation persistence, SOL failed training persistence, and every
role total was economically dominated by capital cost. Stop before basis and
books. Reopen only for a material Lighter or Binance funding, fee, quote-unit,
custody, bridge, latency, capital, or execution-architecture change.

A bounded post-capture schema command printed the first and last economic rows
before the offline adjudication contract was finalized. The retained source
contracts themselves were prospectively frozen, but the result is explicitly
promotion-ineligible and would have required a fresh disjoint confirmation had
anything survived. `AGENTS.md` now requires schema inspection to print only
field names and counts until every funding adjudication choice is hash-bound.
Adjudication result SHA-256 is
`88e62dfd4ab11ab498c56fca530c5b5875054e3819b4068f532fd0ca29e4db0d`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
129, registry SHA-256 is
`333a9118e9339a7b1e0d3ce1ddbfc5f382c1d77f006cb81254b160bcd0f221a8`,
and durability-audit SHA-256 is
`fe8c01b5389fdef3b4a2acd6bc87fec85ac1da65f3111ce018b6ba0da29bb3a9`.

This is the authoritative handoff. Verify every drift-prone claim before acting.
Development belongs only on `main`; do not create another development branch.

## Latest Edge R&D Checkpoint

- The current public Rewards surface exposed a distinct September valuation
  allocation family. A rules-first single-market choice selected Anduril at or
  above 122.5B because it was unconsumed, near-balanced, and displayed a 50
  pUSD/day pool with a 20-share minimum and 4-cent band. Every displayed value,
  competition dash, and earnings value remained discovery-only.

  One prospective exact-source contract then reconciled the exact market and
  sponsored condition: active binary YES/NO tokens, zero maker fee, 5-share
  executable order minimum, 20-share reward minimum, 4.5-cent exact reward
  band, 50 pUSD/day, and 30.7062 reward days through the exact Gamma horizon.
  `market_competitiveness` was 0.842025, but no primary source maps it to an
  account-independent owned share; the public reward payout floor remained
  zero.

  A separately frozen one-use two-token book returned in 233 ms with zero
  cross-token timestamp skew. Joining the retained best bids 0.47 YES and 0.50
  NO cost 0.97 per complete set, producing 0.60 pUSD both-fill gross on 20
  shares and 10 pUSD maximum orphan settlement loss. One-tick improvement cost
  0.99, remained nonmarketable, and left 0.20 pUSD both-fill gross with 10.20
  pUSD orphan loss. Even though the impossible 100% remaining-pool bound was
  1,535.26 pUSD, the official snapshot was 79,200 ms old against the frozen
  10,000 ms gate. Current official API docs define this field as the order-book
  snapshot timestamp, so the age is a real freshness failure rather than a
  harmless last-change time.

  This exact market is terminal: do not retry, refresh, reprice, or select a
  September valuation sibling. No credential, account, order, cancellation,
  fund, on-chain, or protected-capture action occurred. The source contract
  SHA-256 is
  `1db484e6aa584f9d6609f8eab98bc51104bd1d6aeb3dfd17e46c823e6755c84e`,
  source result SHA-256 is
  `5315ab57f4a40356f5cb80aa226c77ec975fda182e0eadeb46c4c25c19c82d51`,
  book contract SHA-256 is
  `3d917d1eb6458a0632d4ce3e22279d59582c40f016a3bfc735f153bfec29fa9b`,
  and book result SHA-256 is
  `e840ceeed3a9bf6dc82a63b54e0d5fadcf737cafebd34bce1335aa5af3b1e2d8`.
  Accepted edges remain 29, ranked hypotheses remain 47, and terminal families
  become 128. Registry SHA-256 is
  `8953013ab62dd510724c702ade0480234c95e6139f3ff57054e4cc12cbd352a3`;
  durability-audit SHA-256 is
  `14f3fdfcabaccf344e6ac7573412425ccecd5132e7e84428a1a40bd7c564bf11`.

- A newly listed Qwen Flash (3.9+) event satisfied rank 31's distinct exact
  two-deadline trigger. Its common rules prove that `NO(released by October 31)
  + YES(released by December 31)` has a one-pUSD terminal floor: release by the
  earlier deadline pays one, release between deadlines pays two, and no release
  by the later deadline pays one. The rendered New card superficially summed to
  0.99, but those values were separate market probabilities rather than labeled
  acquisition sides.

  One prospective contract froze the exact slug, title, two deadline groups,
  rule fragments, implementation hash, request URL, output paths, and zero-live-
  authority boundary before access. Its one public unauthenticated Gamma GET
  returned October `NO` 0.50 plus December `YES` 0.52 = 1.02 pUSD. Both markets
  had fees enabled with the retained 0.04/1 taker-only schedule, so the package
  failed even before books, fee computation, latency, failure unwind, and
  capital-time cost. No CLOB, account, credential, order, fund, on-chain, or
  protected-capture request was justified.

  The contract SHA-256 is
  `1079c0c1aaa133e6e3598d9f9d15985f323b698d689127a02e25d7310acf655e`,
  result SHA-256 is
  `e6438db7e061ed11a86c6c20ea0c80d38bdab9121e13b7ebd2c20583a5c9b85e`,
  and retained raw response SHA-256 is
  `9e5b26aa91e9535fd88d8e570b049db3e030a21ca5b42262f5b70df8d763ee14`.
  The exact event is consumed: do not refresh, reprice, alias, or request its
  books. `AGENTS.md` now records the reusable efficiency rule that multiple
  deadline buttons on one rendered card are not acquisition sides. Freeze at
  most one exact metadata reconciliation when that ambiguity alone creates an
  apparent candidate, and keep Gamma rejection-only.

  Accepted edges remain 29, ranked hypotheses remain 47, and terminal families
  become 127. Registry SHA-256 is
  `4dfb9c701d88737992999978a77bdb8554ea2f6e47f6194c95a6db0112e8b70b`;
  durability-audit SHA-256 is
  `1c90bd52fbc61e8b3fba8c521170b87710ab7640696a17388dbec4ad59a23b8d`.

- A bounded current-page pass screened eight distinct, previously unrecorded
  end-of-September monotone equity ladders: OPEN, NFLX, PLTR, META, NVDA, TSLA,
  MU, and SPCX. Each uses one common final-close scalar and common fallback
  rules across its thresholds. Therefore, for `L < H`, `YES(above L) +
  NO(above H)` has an optimistic one-pUSD terminal floor independent of market
  direction. None of the displayed side-specific acquisition packages was
  strictly below that floor. The closest was META `YES(above $680) + NO(above
  $700)` at 0.13 + 0.98 = 1.11 pUSD before fees, depth, latency, failure unwind,
  and capital-time cost.

  Rendered buttons remain discovery-only, not executable economics. No Gamma,
  CLOB, Binance market, account, credential, order, fund, or protected-capture
  request followed. Do not repeat the exact OPEN, NFLX, PLTR, META, NVDA, TSLA,
  MU, or SPCX end-of-September pages. Reopen rank 31 only for a distinct
  nonconsumed same-rule ladder whose displayed side-specific package is already
  strictly sub-floor, then freeze one exact primary/depth sequence.

  The current official Binance announcement/developer delta supplied no new
  Launchpool, stock/perpetual, stock-option, or funding-cash-flow trigger. The
  frozen quarterly spot-future carry candidate still requires exact Mainnet
  fee evidence; posted Testnet credentials do not satisfy that trigger and were
  neither used nor persisted. Public Polymarket Combo catalog/configuration also
  does not reopen rank 33: the retained candidate already proves RFQ quoting is
  approved-builder/authenticated and explicitly forbids treating catalog prices
  or `rfqEnabled` as executable quotes.

  A broad alias command accidentally admitted a retained one-line `*.raw`
  payload even though JSON suffixes were excluded. `AGENTS.md` now requires
  broad prose/code searches to exclude every raw path segment and `*.raw`
  payload explicitly; selected machine artifacts must be parsed by exact field.

- A bounded public rendered-page pass screened eight distinct, previously
  unrecorded week-of-August-31 scalar ladders: DXY, EWY, Natural Gas, PLTR,
  SpaceX, WTI, Gold, and Silver. Each exact event used a common within-event
  high/low scalar, observation window, boundary convention, and fallback, so
  its ordered monotone packages have a one-pUSD terminal floor. None of the
  displayed side-specific discovery packages was strictly below that floor.
  The closest was WTI `YES(above $110) + NO(above $115)` at 0.005 + 0.999 =
  1.004 pUSD before fees, depth, latency, failure unwind, and capital-time cost.

  Rendered buttons remain discovery-only, so this is a near-miss rather than an
  executable, accepted, stable, or profitable edge. No Gamma, CLOB, Binance
  market, account, credential, order, fund, or protected-capture request
  followed. Do not refetch these exact pages merely to chase a four-mill gap.
  Reopen rank 31 only when a distinct nonconsumed same-rule package is already
  strictly below its floor on side-specific discovery asks, then freeze one
  exact primary and depth sequence. The posted Binance Testnet credentials were
  neither used nor persisted and do not satisfy Mainnet-only account-evidence
  triggers.

  This pass also exposed an avoidable search-hygiene defect: excluding only the
  action-value JSON tree still allowed another large one-line retained JSON to
  flood an alias search. `AGENTS.md` now excludes every JSON file from broad
  prose/code alias searches by default and requires structured exact-field
  parsing for selected JSON artifacts. The Polymarket boundary is clarified:
  order-capable work remains BTC/ETH/SOL and disabled, while public
  unauthenticated structural discovery may cover other markets under the same
  fail-closed gates.

- A second bounded rendered-page pass screened eight distinct September
  equity/index hit ladders: RKLB, NVDA, TSLA, AAPL, SPY, AMZN, META, and MU.
  Their current rules use the same within-event Pyth one-minute high/low scalar,
  regular-session scope, exact-price convention, and split adjustment. No
  fully visible monotone package was strictly below one pUSD. The closest new
  row was SPY `YES(dip to 730) + NO(dip to 720)` at 0.20 + 0.83 = 1.03 pUSD
  before fees, depth, latency, failure unwind, and capital-time cost. No Gamma, CLOB,
  Pyth-history, account, credential, order, fund, or protected-capture request
  followed.

  The AMZN weekly/monthly pages exposed a reusable cross-window trap. The week
  of August 31 begins before the September monthly window, so the weekly event
  is not a literal subset until the already elapsed August 31 session is proved
  not to have hit the shared strike. The visible shared-strike packages already
  rejected under the optimistic subset assumption, so requesting Pyth history
  merely to prove the weaker identity would waste a source call. `AGENTS.md`
  now requires this rejection-first ordering. The current official Binance Spot
  and USD-M Futures changelogs were also checked and contain no post-retained structural
  deployment change; no Binance market or account endpoint was called.

- A bounded public rendered-page Boolean-cover screen checked current Fed
  count/deadline markets, September XAU/XAG/WTI hit ladders, and several
  same-event deadline ladders before Gamma, CLOB, account, credential, order,
  fund, or protected-capture access. No observed guaranteed package was
  strictly below its floor. The closest exact implication was XAU
  `YES(hit >= 5,100) + NO(hit >= 5,200)` at 0.06 + 0.94 = 1.00 pUSD; equality
  does not survive fees, time value, depth, or failure risk. The Fed
  mutual-exclusion package `NO(cut by the December meeting) + NO(zero cuts in
  full-year 2026)` cost 0.89 + 0.121 = 1.011 pUSD for a one-pUSD floor.

  The superficially cheap opposite construction is not a complement: a first
  emergency cut after completion of the December meeting but before December
  31 makes both `YES(cut by the December meeting)` and `YES(zero full-year
  cuts)` pay zero. `AGENTS.md` now generalizes the state-table gate: implication,
  mutual exclusion, and collective exhaustion authorize different guaranteed
  packages, and complementarity requires both of the latter two properties.
  Rendered values remain bounded discovery evidence only, not current
  executable economics or a terminal population. Reopen only when a distinct
  same-rule package is already strictly sub-floor on side-specific discovery
  asks, then freeze one exact source and depth sequence.

  Focused verification initially used `uv run --locked pytest` and failed at
  collection because the Windows console-script path omitted the repository
  root, producing `ModuleNotFoundError: tools`. The locked interpreter form
  `uv run --locked python -m pytest tests/test_agent_workflows.py -q` passed all
  21 tests. `AGENTS.md` now records that invocation rule; do not change imports
  to repair this environment-only launcher failure.

- A bounded public rendered-page discovery pass generalized rank 31's exact
  monotone-threshold identity beyond crypto without spending Gamma, CLOB,
  account, credential, order, fund, or protected-capture requests. Every sampled
  2026 college-football team win-total page uses the same scalar rule: `Yes`
  pays when regular-season wins exceed its listed threshold, with championship,
  bowl, playoff, cancellation, and no-data handling stated consistently. Thus,
  for one team and thresholds `L < H`, lower-threshold `YES` plus
  higher-threshold `NO` has a rule-consistent one-pUSD terminal floor
  independent of team performance.

  Thirty-one official event-page discovery snapshots, including both liquid
  and zero-volume ladders, were checked with the displayed `Buy Yes` and
  `Buy No` buttons. No observed ordered pair was strictly below one pUSD; the
  best was Colorado State `YES(>1.5 wins)` plus `NO(>2.5 wins)` at 0.88 + 0.17
  = 1.05 pUSD. This is a bounded discovery rejection, not complete catalog
  coverage, current executable economics, or a terminal family result. Search
  crawls and rendered buttons are excluded from accepted economics. Reopen
  rank 31 only for a distinct nonconsumed same-rule scalar ladder whose
  side-specific discovery acquisition sum is already strictly below its floor;
  only then freeze one exact primary event capture and require depth, fees,
  resolution risk, failure unwind, and capital-time cost.

- A tempting cross-market decomposition of the four-way `US economic state at
  the end of 2026?` partition was rejected before any API access. That event
  resolves from December 2026 unemployment and inflation, while the separately
  traded `How high will US unemployment go in 2026?` and `How high will
  inflation get in 2026?` ladders trigger on any qualifying month during 2026.
  The source families overlap but their observation functions are not equal, so
  neither ladder can replicate the December-only partition. The active August
  unemployment nine-bin complete set was also rejected at the rendered layer:
  all displayed `Buy Yes` asks summed to 1.277 pUSD for a one-pUSD complete-set
  floor. Do not substitute shared BLS sources or matching numeric thresholds
  for exact time-scope equality.

- Rank 47's literal later-population trigger was satisfied with zero new
  network access. The retained complete Binance Options `exchangeInfo` at
  `2026-08-31T17:29:50.562Z` contained 1,576 eligible BTC, ETH, and SOL
  symbols. The already-retained all-options ticker at
  `2026-08-31T22:44:19.025Z` contained 1,578 scoped symbols, with exactly two
  additions: `BTC-261225-94000-C` and `BTC-261225-94000-P`. The already-
  retained synchronized USD-M Futures book was reused; no source was refetched.

  A prospectively frozen zero-network screen applied the existing side-specific
  long-option plus opposite-perpetual payoff identity and 33.5-bip fixed fee and
  expiry-basis stress. The call had a displayed 2,625 USDT ask and a gross
  terminal floor of `-17,970.60` USDT per BTC before fixed stress. The put had
  no positive displayed ask, so it was price-incomplete and could not advance.
  Zero rows had a complete positive gross floor and zero survived fixed stress.
  No current `exchangeInfo`, ticker, book, depth, funding, account, credential,
  order, fund, or protected-capture request was made.

  This exact two-symbol retained snapshot is terminal, not a profitable or
  accepted edge. Do not refetch, repair the missing ask, or poll rank 47 again
  within the same UTC day. Reopen only after `2026-09-01T22:44:19Z` for a later
  distinct population or another literal material fee, settlement, tick, depth,
  funding, basis, capital, or nonpolling above-all-cost trigger. Contract and
  result SHA-256 values are
  `03b6e4c160d10493fe090353440aae1b987fbc7195845a0d8691a0a76cf09f6d`
  and `b425fd356f5033e24313c5dbb166d381c045a4e794107a791acd49ed61992ab2`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  become 126, registry SHA-256 is
  `6ad94f7f192f61f7f47f146139985debe5da1252a6a1b7470e53581c5b9d9bf2`,
  and durability-audit SHA-256 is
  `4f7196ba69e68d8e28bfc02f3f60c1f063456a310a456a9bcbb8a0d39868aed8`.

- Rank 30's literal public trigger was satisfied by an official rendered NFL
  deployment outside every consumed exact event and time window. One frozen
  complete Gamma keyset GET covered the previously unconsumed September 13
  `00:00:00Z` through `20:25:00Z` gap. It returned 12 events, included 10 active
  rule-complete events, and proved 2,978 exact full-game margin and total
  monotone payoff relations without books or fees.

  The catalog's 120 midpoint-like `outcomePrices` candidates were diagnostic
  only. A zero-network exhaustive side-specific correction evaluated 2,965
  price-complete relations with first-outcome `bestAsk` or conservative
  `1-bestBid` for the second outcome. Zero were strictly below their one-pUSD
  floor; the best was a Jets-Titans total package at 1.04 pUSD. Thirteen
  Atlanta-Pittsburgh relations genuinely lacked the required `bestBid`, while
  Bears-Panthers and Bills-Texans were excluded for duplicate logical
  thresholds. The exact population is therefore price-incomplete, not an
  exhaustive rejection, but a strongest candidate is unknowable and no depth,
  fee, refetch, repair, sibling selection, account, credential, order, fund, or
  protected-capture request is permitted.

  Two zero-network failures were preserved rather than overwritten. V1 exposed
  an unnecessary dependency on an unused side field; after that was corrected,
  v2 exposed a genuinely missing selected-side price. The shared adjudicator now
  validates only the selected side, explicitly retains price-incomplete
  relations, blocks depth for incomplete populations, and reproduces historical
  complete-result hashes unchanged. This corrects the workflow without
  refetching or substituting midpoint data.

  Catalog contract/result SHA-256 values are
  `821e45f53f134dbc14e5f1d94bc657680ac047a74d188365b5c0d29154cab8bf`
  and `126b1dc61fa379458aaa88a8edef899be437d33fda6cba1a7a29e3536cbe856f`;
  final offline contract/result SHA-256 values are
  `714f4688f00111be75d9cb39de668bed7101b3e54debb454f85ffdd2775d9c76`
  and `1965d997ba11fdeb51cf5bac40e9a13569640d724b4f58624a59fa230e9d69f9`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  become 125, registry SHA-256 is
  `1647539495270c7e732dd9fe8d421cc69456f6d7ccbb53e7655d34cfc7316f85`,
  and durability-audit SHA-256 is
  `0195a2498172aec1a92e86a3e553e6e344e2c87980761a2fa4c59ab6bead33d7`.

- Rank 19's Polymarket Perps/Binance spot carry near-miss has a materially
  tighter public fee bound. The retained August 29 official documentation index
  exposed a dedicated Perps fee page that the August 26 carry attempt had not
  captured. One prospectively frozen public unauthenticated Markdown GET
  retained 3,187 exact bytes and proved that a zero-volume Polymarket Perps
  account pays 1.25 bips per maker fill, or 2.50 bips for entry plus exit;
  taker entry plus exit costs 8 bips. The top maker tier rebates 0.50 bip per
  fill, but its one-billion-dollar volume tier and temporary beta-account
  exception are not attributed to any repository account.

  The retained 30-day BTC short-Polymarket-Perps/long-Binance-spot result had
  only 25.7920876712 bips left after funding, the conditional 6% OI reward, and
  the two-leg 5% annual capital hurdle. The new baseline Polymarket maker fee
  therefore leaves 23.2920876712 bips before every Binance fee, cross-venue
  basis, quote conversion, transfer, custody, and failure cost. Combining it
  with the retained standard Binance maker sensitivity leaves only
  3.2920876712 bips; a retained conditional zero-maker quote route leaves
  23.2920876712 bips but is not current account or same-unit route evidence.
  The exact account must still prove the one-million-dollar daily-average gross
  OI reward, its fee tiers, eligible quote route, and every external cost before
  one synchronized finite-size basis/book study. No funding refresh, books,
  accounts, credentials, orders, funds, or protected captures were used.

  The fee-source contract, capture, and zero-network adjudication SHA-256 values
  are `d262e39f404530dbc5c7451fa930f3c9a6180f32ba334e842c5fa467fa2565d1`,
  `864efb46a06dda7a7af76e1dda766fe25870f80c78a1bfb322c83033acee7529`,
  and `69fc78d24c28106e7c8f96a6bca32e92f4591582698261398bcdb77a6fc2fa78`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  remain 124, registry SHA-256 is
  `d4a134ac80079622c76e0f39bcf30dba4a5d0ae539635dde93f5f802b3005cbb`,
  and durability-audit SHA-256 is
  `a520e2e0a41bb053cfeefe6752acecfe95d471c21fffe02b9512c06c46be7e96`.

- A distinct adjacent-day threshold-plus-direction identity was derived and
  screened before Gamma, books, fees, accounts, or orders. For one shared
  strike `T`, `Sep 2 NO(T) + Sep 1 YES(T) + daily Up` and the symmetric
  `Sep 2 YES(T) + Sep 1 NO(T) + daily Down` each have an optimistic one-pUSD
  common-rule floor under the current same-asset Binance noon-close rules. The
  daily 50/50 equality case does not create a hole: at equal closes, one of the
  two strict-threshold legs already pays one. Every shared strike and both
  directions were then screened across BTC, ETH, and SOL using only available
  rendered discovery snapshots of threshold acquisition buttons and the
  rendered daily probability as an explicitly non-executable optimistic
  diagnostic. The best displayed
  totals were approximately 1.429, 1.370, and 1.450 pUSD, respectively, for a
  one-pUSD floor. Snapshot refresh times were not synchronized and a displayed
  probability is not an ask, so this does not terminalize the family or
  establish economics;
  it does show that the visible package is roughly 37% or more too expensive
  and justifies zero Gamma or CLOB requests. Reopen only for a distinct
  synchronized side-specific prefilter strictly below one pUSD, never by
  treating a probability, last trade, midpoint, missing ask, or zero display as
  executable acquisition.

- A current rendered-screen lead in the September 3 Ethereum range and
  strict-above events was rejected before Gamma, books, fees, accounts, or
  orders. The tempting range `NO(>3,000)` plus threshold `YES(3,000)` package
  is not an exact complement: the range rules assign an exact 3,000 close to
  the higher bracket while the threshold requires a strictly higher close, so
  both purchased legs pay zero at exact equality. Current rendered acquisition
  asks also moved to 0.999 plus 0.005, above the one-pUSD target even before
  fees. Treat displayed values as discovery only and never let a top-bin label
  override an event's explicit boundary assignment. `AGENTS.md` now states the
  zero-payout state directly so this false identity cannot authorize Gamma or
  book access in a later run. The same pass screened every visible valid
  cross-event coverage package and every lower-YES plus higher-NO threshold
  pair across the newly deployed September 7 BTC, ETH, and SOL siblings. The
  best rendered coverage sums were 1.98, 1.98, and 3.96 pUSD; the best ladder
  sums were 1.03, 1.03, and 1.04 pUSD, respectively, for one-pUSD floors.
  Rendered asks remain discovery-only, but all six global minima already reject
  before Gamma, depth, fees, accounts, or orders; do not deepen or cherry-pick
  this September 7 sibling population absent a distinct rule or market change.

- A distinct Binance Wallet routed Aster USD1-versus-Binance USDT perpetual
  funding lead has been rejected before any basis, book, account, credential,
  order, or fund request. One prospectively frozen public Aster inventory
  capture proved active BTCUSD1, ETHUSD1, and SOLUSD1 contracts. Three sibling
  funding captures then aligned 209 eight-hour rows per asset with retained
  Binance BTCUSDT, ETHUSDT, and SOLUSDT histories. Orientation was selected on
  the first 104 rows only and held fixed for 52-row validation and 53-row test
  roles.

  All three assets failed the frozen 20-bip round trip, 10% annual two-leg
  capital hurdle, and 23.954486-bip USD1/USDT stress. BTC and ETH had negative
  validation and test gross funding; SOL had positive validation gross but
  negative test gross. No asset passed even the training role after all costs,
  so the frozen rejection-first contract prohibited premiums, books, accounts,
  credentials, orders, or funds. Do not repeat, resample, change the training
  orientation, or weaken the costs for this exact 209-row population. Reopen
  only on a material Aster or Binance funding, fee, basis, USD1 conversion,
  custody, capital, or execution-architecture change. Prefilter contract and
  result SHA-256 values are
  `20ef24c12e5b18960ef491e07e51855868c18355adfb1e9fb3c3c465f28b307e`
  and `c8983f0668fdd157add672940769f5ceb3db91c2cbe495ce27b8034dd491da91`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  become 124, registry SHA-256 is
  `400313770bcd2cbf542e97f5a1d1fcf59b66dbf76121ecbd784e107bed4767c8`,
  and durability-audit SHA-256 is
  `13910f293d085c63f8ac6fb001e06b26c23dae7481af8852af1f1e7165fcc73f`.

- A 2026-09-01 zero-request discovery pass rejected three superficially
  attractive Polymarket shortcuts before Gamma or CLOB access. The current
  official rendered hourly BTC rules use Binance one-hour candle open and
  close, a five-minute SOL market uses Chainlink start/end values, and the
  daily ETH market uses Binance noon-to-noon closes plus a 50/50 equality case.
  These are different observation functions, so there is no proved
  cross-horizon interval payoff identity. The same discovery pass found every
  rendered Buy package for the new September 3 and September 5 BTC
  range/threshold pairs and the August 31-September 6 ETH hit/daily-threshold
  implications at or above its optimistic guaranteed floor. Discovery values
  are excluded from economics: this pass authorizes zero deeper requests and
  changes neither terminal-family counts nor the registry. `AGENTS.md` now
  makes source, timestamp endpoint, aggregation, and tie-rule equivalence a
  mandatory pre-price gate so nominally aligned titles cannot waste another
  request.

- The distinct active Polymarket `Ethereum above ___ on September 2?`
  threshold ladder satisfied rank 31's literal trigger. A prospectively frozen
  corrected v2 screen made one exact public unauthenticated Gamma event GET and
  excluded midpoint-like `outcomePrices` from economics. All 11 active markets
  and all 55 direction-independent lower-YES plus higher-NO packages had
  side-specific rejection prices: lower YES used `bestAsk`, while higher NO
  used the conservative `1 - bestBid` proxy. Zero packages were strictly below
  their guaranteed one-pUSD terminal floor. The cheapest was 2000 YES plus
  2100 NO at 1.004 pUSD, so this exact event fails before fees, ticks, latency,
  failure unwind, or depth.

  This September 2 event is terminal with zero book, fee, account, credential,
  order, fund, on-chain, or protected-capture requests. Do not repeat or
  reprice it, and do not select a BTC, SOL, or adjacent-date sibling after
  observing the result. Reopen only on rank 31's registered literal trigger for
  a distinct nonconsumed exact population whose side-specific rejection proxy
  is already strictly sub-floor. Contract and result SHA-256 values are
  `1ef97e13565e958fa67c380abb6fb9e519d4207e1a66a843ce2f9c2fc46bf8b7`
  and `a83c568656fe15000624d0ae872abc75955bd42581714a1a926b114bc4206f33`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  become 123, registry SHA-256 is
  `77a777ed40570ebd4773b431c20d32057fb36ce2f648011e503870fe58700352`,
  and durability-audit SHA-256 is
  `58109b3122be2876d4a6f8acd2a028bd13cc06255d6d379a8f2b4063802897ff`.

- Rank 47's literal new-population trigger was satisfied and consumed
  efficiently. A frozen zero-network audit compared the exhausted 1,410-symbol
  August 27 BTC/ETH/SOL crypto-option population with the already retained
  complete August 31 Binance Options `exchangeInfo`. The current population had
  1,576 eligible symbols: 508 exact additions and 342 removals. A separately
  frozen two-request public prefilter then captured the complete Options ticker
  and USD-M Futures book ticker once and evaluated only those 508 additions.
  Four hundred thirteen had positive option-ask and executable perpetual-entry
  sides, but zero had even a positive gross long-option plus
  opposite-perpetual terminal floor. They therefore fail before the frozen
  33.5-bip option fee, settlement fee, futures round-trip, and expiry-basis
  stress, and before funding, ticks, capital, depth, or account costs.

  This exact August 31 delta is terminal with zero option-depth, funding,
  account, credential, order, or fund requests. Do not refresh, subset, reprice,
  or rebuild it. Reopen rank 47 only for a later distinct active BTC/ETH/SOL
  option population or another literal material fee, settlement, tick, depth,
  funding, basis, capital, or independently observed above-all-cost trigger.
  The population-delta contract and result SHA-256 values are
  `15604a7006b324bdd873481c79d5ac4ec34551a1d39d80904b546eb50ea441bc`
  and `001abaada3b352235cbc38228dec6b6176a26cdfb33e208f0c3467f858cf9446`.
  The price-prefilter contract and result SHA-256 values are
  `5e2d0c36588530a4c4bf176bfe874a43af6f0063b636670fd4d599d3356bd5af`
  and `93d2ed3c9b6041f9ffcc7f9579f184687113049051a421f9fc048d2d4e309eee`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  become 122, registry SHA-256 is
  `fc026e6e2c76d6d6589f964dbe52fe140e126d87ee005c96e786289f40f9664c`,
  and durability-audit SHA-256 is
  `6d72a1f54af7e18351133f9bd329acfc9f2800f429274f0d564228efdd9e1d0e`.

- A serious Polymarket existing-inventory opposite-lock candidate passed one
  frozen out-of-sample historical validation. The rule was selected only on the
  retained August 25 public-wallet day. One preregistered public unauthenticated
  August 30 wallet-day GET then returned 1,964 complete rows and produced 66
  causal 1-60 second locks across 27 BTC, ETH, and SOL conditions, 1,486.108451
  matched shares, and 217.6302921908223351705204892211 pUSD of locked
  historical cashflow after charging the full 0.07 fee on both legs and
  stressing the later hedge one adverse 0.01 tick. Both UTC halves and all
  three assets were positive; maximum single-condition PnL share was
  0.2807103473 below the frozen 0.35 ceiling. Retained official primary-source
  bytes define the fee as `shares * feeRate * p * (1-p)` in USDC. A zero-network
  child correction kept the frozen lock set fixed and conservatively ceiled
  both fees of every matched fragment to the published 0.00001 pUSD quantum.
  All 66 locks remained positive, the upper-bound rounding drag was only
  0.0005909365952300705204892211 pUSD, and corrected locked cashflow was
  217.6297012542271051 pUSD.

  A separately preregistered zero-network robustness audit kept every lock
  fixed and crossed 1/2/3/5/10 total adverse hedge ticks with fixed per-lock
  costs. At the frozen gate of five ticks plus 0.05 pUSD on every lock,
  aggregate cashflow remained 154.9035632142271051 pUSD and BTC, ETH, and SOL
  aggregates were each positive, but only 43 of 66 locks, or 65.15%, remained
  individually positive versus the required 80%. The gate therefore failed.
  Median extra whole-tick capacity above baseline was eight, but the minimum
  and p10 were zero. This materially narrows the claim: aggregate historical
  surplus exists, but cross-lock stability does not.

  This is not an accepted edge, current executable profit, or authority to
  initiate the first leg. It is an existing-inventory risk-reduction overlay:
  only an independently justified owned first leg may be completed, and the
  matched YES+NO quantity then has a direction-independent one-pUSD terminal or
  merge identity. The public wallet is not owned, authorized, or assumed
  reproducible. Current owned per-lot basis, finite-size book,
  merge/redemption, external cost, and failure-unwind evidence remain absent,
  so the public forward floor is zero. Do not repeat, narrow, paginate, alias,
  resample, or cherry-pick the consumed August 30 locks. The next literal
  trigger is explicit read-only account authority plus independently
  preexisting eligible BTC, ETH, or SOL binary inventory with exact per-lot
  basis and external-cost ceiling, or a material fee/merge/redemption/execution
  change. Reject every exact lot that is not positive after all costs; any order
  requires separate explicit authority. Candidate SHA-256 is
  `4ee6b1d3a54b6b112f9f031dc5cb91cb2abc2943119f512dfb26c79fe6c93a01`;
  validation SHA-256 is
  `b81af57f094f1ff75bcb77f9938ec7c84791af4e1cecb44b3402dac17d4dc1df`.
  Fee-rounding correction SHA-256 is
  `b423b44e57bfd329220256facf4b9eabe45371b267e7a91ea08aa11a666be204`.
  Robustness contract and result SHA-256 are
  `cc52cb4cafd7d36c432e39b1610e352bf513f405d2244432f7bcbc26dca2ea6f`
  and `3f2bc8f2ea70345700062f43766bd1110299f7636367385daf4ce944f129046e`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  become 121, registry SHA-256 is
  `669324eaae8533fb51fae63078f561499b74666a511847fe99f7c8b48eba085e`,
  and durability-audit SHA-256 is
  `3b4cbdacb890c13ce7f91ed1eb31feee65dc468503afbe0281f23f4760678467`.

- The exact-title literature delta for `Arbitrage in Perpetual Contracts`
  (SSRN 5262988) is terminal as a source failure, not a Binance edge. The one
  frozen exact primary PDF GET returned HTTP 403 and a retained 5,625-byte
  Cloudflare HTML challenge. The indexed abstract is discovery only and cannot
  authorize a clamp-bound collector or retained-data screen. Do not retry a
  query variant, alias, mirror, or locale route. Reopen only on a materially
  revised complete primary version from the authors or a public reproducible
  repository with the exact formula, code, data, and after-cost execution
  method. Canonical adjudication SHA-256 is
  `dd5058136c809425b833266f9a1bce568bab6e692301a2adbffec8b18b017275`.

- The apparent August 31 Polymarket Sports maker-rebate change is terminal as
  a source-quality event, not an edge. A current search-index rendering exposed
  25%, a 0.03 Sports taker-fee rate, and USDC wording, but one frozen canonical
  `.md` GET returned exactly the prior 5,945 bytes and SHA-256
  `8d2c6562bd1b3376bc3fc1557a60efef5aa3c1d856c7f8dcc405139a07e9ba2a`,
  still stating 15%, 0.05, and pUSD. Discovery snippets are never economic
  inputs. The prior official cross-surface conflict and zero public forward
  Sports rebate floor remain. Do not retry or alias this exact drift while the
  canonical hash is unchanged; reopen only on a canonical byte change plus an
  effective-dated reconciliation, exact effective per-market fraction, or
  separately authorized owned fill-and-payout evidence. Canonical adjudication
  SHA-256 is
  `0874302d3acb6e641f38f693eafaa94d2a3b86167c0f796691efc5707b5cf64a`.
  Accepted edges remain 29, ranked hypotheses remain 47, terminal families
  become 120, registry SHA-256 is
  `6a7d560501abdd37f3a88db28a708ef69dd0e83668639ea5d4bac9836cee6dc3`,
  and durability-audit SHA-256 is
  `bd0f48ebd0b56e14ffdc988c0a72ba9c016e6a4c8db7d0b743242e26003efac5`.

- Binance COIN-M inverse versus USD-M linear same-asset perpetual funding is
  terminal for the retained BTC/ETH/SOL population. One frozen public
  eight-request sequence retained current exchange configuration and 499
  aligned funding rows per pair. A training-only orientation selected short
  COIN-M and long USD-M for all three assets, but combined validation plus test
  gross was only 3.9516, 11.8749, and 25.2915 bips versus one 32-bip entry-exit
  hurdle. Every asset also had an out-of-sample negative calendar month. A
  zero-request causal lagged-sign rescue required 116 to 172 turnover units per
  out-of-sample role and lost 220.1847 to 330.5917 bips at only two bips
  one-way. Do not repeat, resample, request books, or use credentials for this
  family. Reopen only after a material COIN-M or USD-M funding cash-flow,
  collateral, settlement, fee, or contract-architecture change. Canonical
  adjudication SHA-256 is
  `560c42b8bbebf9c10ac1292c44043001fe4d26ccf85217f71b93095e72bf9230`.

- Binance Spot OPO and OPOCO now have a source-bound received-quantity execution
  candidate. Pinned official terms prove that only the working BUY balance is
  required, received funds are locked, the contingent SELL quantity accounts
  for commission and lot filters, and the exchange activates it after the BUY
  fully fills. One exact frozen public Testnet `exchangeInfo` response confirmed
  current OTO, OPO, and OCO support on BTCUSDT, ETHUSDT, and SOLUSDT, matching
  the retained production flags and core filters. This removes one post-fill
  client submission from a frozen manual sequential comparator and avoids
  pre-funding the pending sell, but partial fills remain unprotected and zero
  monetary latency, fill, fee, or after-cost profit floor is proved. The
  standalone-profit claim is terminal; the rank-5 overlay may advance only with
  separate explicit Spot testnet order authority plus an independently required
  minimum-size organic buy-then-sell question and an identical-quantity manual
  comparator. Do not repeat, narrow, reorder, or alias the consumed Testnet
  configuration request. Canonical candidate SHA-256 is
  `4680faf2b4b4e36466cbc7ace4de2a2214430cbe4a5d537a3997e3f43bc47cc8`.

- Binance Spot SBE Diff Depth is a second source-bound direction-independent
  execution-information candidate. The retained current official SBE source
  documents 20 ms depth updates, microsecond timestamps, and a 50 ms top-20
  stream versus the retained JSON source's fastest 100 ms depth cadence. It
  requires an Ed25519 API key but no extra market-data permission. Both SBE
  best bid/ask and JSON book ticker are documented real-time, so the nominal
  depth cadence cannot be carried into a top-of-book or monetary claim. The
  one-use source gate failed only because bold Markdown delimiters wrapped one
  required phrase; retained bytes were adjudicated offline and must not be
  refetched or aliased. Advance only with a designated ephemeral Ed25519 key,
  explicit read-only market-data authority, a byte-exact decoder, and a
  precommitted same-host same-symbol dual-feed capture. Canonical adjudication
  SHA-256 is
  `6b01c825831657d8dac8a33efb196bbd63d64698288e15cf9b1ff27be9b4aa77`.

- Binance Spot FIX now has a source-bound execution-risk candidate. The retained
  current official contract says `UNORDERED` should perform better with multiple
  messages in flight, FIX ExecutionReport push should perform better, and one
  mass-cancel message cancels every account order on one symbol across
  connections. It documents no automatic cancel-on-disconnect, no non-live
  order-entry support, and no measured latency, fill, or profit floor. The
  168,282-byte receipt hash is preserved, while only a mechanically extracted
  15,222-byte secret-free exact excerpt is committed because the unrelated full
  official document contains a public example private key. The
  candidate requires confirmed non-live support, an ephemeral Ed25519 key with
  `FIX_API`, separate session and order authority, and an identical-intent
  multi-flight comparator. Canonical candidate SHA-256 is
  `177d9d119c8a57c86df03b136bdfc2c880b0220ac50c51bce20e84cbe21f1755`.

- The exact Polymarket BTC/ETH/SOL interval-composition identity now has a
  retained settlement audit across 25 aligned sets, 100 terminal markets, and
  50 direction-independent four-leg packages. Every package paid at least its
  one-pUSD floor; observed payouts ranged from one to three pUSD. This supports
  the payoff identity, not profitability: the current displayed screen still
  had a 1.990 pUSD best package for a one-pUSD floor, and no atomic executable
  sub-floor acquisition is proved.
- The separately frozen one-use historical trade test made one public
  unauthenticated request for 48 exact condition IDs. It returned HTTP 408 and
  exposed zero trade rows. Preserve the raw error and journal; never retry,
  split, narrow, paginate, reorder, or alias that consumed population. Reopen
  only for a future distinct aligned population that first passes a strict
  rejection-only sub-floor gate, then freeze a prospective exact live CLOB
  package capture. No account, credential, order, fund, on-chain, or protected
  capture access occurred.
- The GLWUSDT terminal special-funding reconciliation is preregistered, not yet
  runnable. At or after `2026-08-31T00:10:00Z`, run only the frozen one-request
  nonoverlapping history delta in
  `binance-glw-special-funding-terminal-reconciliation-contract-v2-2026-08-30.json`.
  It may establish mechanism timing only: the 2026 execution window is closed,
  book capture is prohibited, and profitability remains rejected. Preserve any
  failure and do not retry, paginate, alias, or extend the interval.
- A distinct Polymarket created-event delta after
  `2026-08-30T16:07:16.021321Z` consumed one public request and returned the
  endpoint's effective 100-row cap with a non-null cursor; all 100 rows were
  newer than the cutoff. The population was therefore incomplete, zero packages
  were screened, and no book or fee request was authorized. Never paginate,
  narrow, refresh, alias, or repeat this delta. Future time-delta catalogs must
  prove worst-case arrivals fit below the observed cap before access.

## Closeout State

- The last fully hosted-verified baseline before this structural-parity
  checkpoint is `336114411aab0ad4ed6fae18047245dc420789b2`. CI, Ruff, Vulture,
  Super-Linter, CodeQL, and DeepSource passed that exact revision. GitHub exposed
  only `main`, and the available APIs reported zero open Dependabot,
  code-scanning, and secret-scanning alerts. Reverify the publication commit;
  zero alerts never proves zero undisclosed vulnerabilities.
- The repository is beta `0.1.0-beta.1`. No model has production authority or a
  demonstrated long-lived after-all-cost edge. Twenty-nine narrowly scoped structural
  edges are accepted: Polymarket holding yield for existing idle on-platform
  pUSD; Polymarket pUSD taker-fee rebates only for independently justified
  legitimate organic BTC/ETH/SOL taker flow after the direct-wallet tier is
  effective; exact realized positive Polymarket crypto maker rebates only on
  independently justified legitimate organic owned BTC/ETH/SOL maker fills
  after every incremental cost; Binance Soft Staking yield for already-held idle non-order ETH/SOL
  Spot inventory; LDUSDT or independently existing RWUSD reward retention only
  for already-required USD-M Futures collateral under the exact applicable
  account haircut and limits;
  fail-closed one-intermediary Binance Spot route savings only for an
  independently required legitimate same-account organic conversion after
  exact account fees, filters, fresh finite-size depth, residuals, extra-leg
  stress, and failure-unwind costs remain strictly cheaper than the direct route;
  just-in-time BNB fee reduction; the exact 5% Cross Margin interest reduction
  only on an independently existing legitimate borrow when sufficient BNB is
  already held in the same Cross Margin account for unrelated reasons and every
  incremental cost is lower than the discount; current quote-native BTC/ETH/SOL promotional
  fee reduction; the current TradFi perpetual zero-maker and reduced-taker fee
  overlay only for independently justified organic flow with the exact current
  symbol and actual fill role; the current Binance Stocks promotional trading-
  spread reduction only for independently justified organic direct-stock flow
  with the exact previewed order tier and realized fee; the current bStocks zero-
  maker fee only for independently justified organic maker flow through
  `2026-09-30T23:59:00Z`, with the exact owned fill role and counterfactual
  account fee; exact realized positive Binance Spot Liquidity Program final
  rebates only on independently justified legitimate organic owned maker fills
  after every incremental cost; USD1/WLFI holding-airdrop
  yield; the fixed USD1 Simple Earn
  bonus on at most the first 1,500 independently already-held idle USD1 only
  when its mutually exclusive balance-specific route beats the airdrop; U
  Flexible yield; automatic
  RWUSD VIP bonus yield; current USDT Flexible bonus yield; and the current
  automatic USDe holding reward only for eligible USDe already independently
  held on Binance for at least 24 hours; Polymarket builder fees only on
  bona fide independently existing third-party matched orders routed through an
  owned app with an account-confirmed active disclosed positive rate; and
  Polymarket referral rewards only for authentic external referrals after the
  account independently cleared the lifetime-volume threshold; and Binance
  Square's base 20% Write to Earn commission only on authentic external readers'
  independently existing eligible fee-bearing trades attributed to genuinely
  useful content; Binance Referral Pro's base 20% Spot/Margin and 10%
  one-year Futures fee commission only for authentic independently acquired new
  external users; and Polymarket Perps' separate 20% fee share only for authentic
  external traders within account-confirmed available invites without volume-
  based invite unlocking; and exact realized monthly Binance Institutional Loan
  interest rebates only on an independently required existing eligible loan when
  KYB VIP eligibility and the applicable performance target were independently
  satisfied without creating trading volume, Open Interest, Net Asset Value,
  borrowing, leverage, or collateral exposure; and exact realized positive
  Binance CAAS markup trade commission only from independently existing bona
  fide external client trades under an already active disclosed fee group and
  markup configuration after every incremental cost; and exact realized
  positive own-account Binance Link-and-Trade kickback income only from
  independently justified legitimate organic Spot flow on an already linked
  rebate-working account after balance reconciliation and every incremental
  cost. Each Binance
  edge is limited to independently required inventory or organic flow under its
  canonical account, liquidity, cost, and non-manufactured-volume gates.
  A current BTC Simple Earn Flexible product is only an extension candidate:
  its public `0.27% Max` headline is a hidden-tier `0.02% + 0.25%` breakdown,
  not a whole-balance rate, and exact account tiers, costs, and eligibility are
  unproved. It does not increase the accepted count.
  Binance Exchange Link Spot and Futures commission rebates are a distinct
  unaccepted candidate: the retained current official index proves exact
  endpoint names, but exact security classification, schema, eligibility, rate,
  organic flow, owned payout, and all costs remain unproved. It also does not
  increase the accepted count.
  None is deployment-ready. Binance remains paper/testnet/Demo; Polymarket
  remains independent, disabled by default, and unpromoted.
- The one historical cutoff is `2026-08-14T00:00:00Z`. Do not move it or fetch
  the newest history on each iteration. Prospective experiments remain isolated
  from that frozen snapshot.
- A delimiter-safe 2026-08-30 identity audit found 656 already-shared `main`
  commits with legacy prohibited AI-style identities and 1,496 commits with the
  required `AI agent <>` identity. The most recent prohibited commit was 102
  commits behind `181000ac6e3d55434f2a6f076b4ad8ca0144ab37`; there were no
  local PR-head refs, and a fresh anonymous-contributor API response contained
  zero anonymous rows. This is surfaced, not repaired: do not force-rewrite
  shared history without explicit approval. Identity audit and commit are
  separate gates; inspect the audit result before any mutating Git command.

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
  That artifact is terminal only as a frozen one-state snapshot, not as a
  lifetime verdict on event-time recurrence. A new primary study,
  `Executable Arbitrage and Market Efficiency in Prediction Markets`
  (`arXiv:2608.00666v1`, submitted 2026-08-01), reports 2,098 positive
  unsupported YES-side CLOB episodes but 36 positive adapter-supported NO-side
  episodes, with only five exact-duration NO-side episodes and a 7.99-second
  median. It also estimates approximately 1.086 million USDC of converter-linked
  profit. The estimate combines realized proceeds with imputed values for
  residual/merged YES inventory and imputed opportunity values for split-minted
  NO inventory; it is therefore a mechanism lead, not current cash-realized
  after-cost profit. The official adapter at pinned commit
  `f78b35b0863b4308a431ca307d06f49b2ea65e78` independently confirms the
  executable NO-to-YES subset conversion and optional per-event conversion fee.
  A retained public PMXT preflight filtered all 24 hourly 2026-08-09 partitions
  to the six tokens in event `106981`. Once every token had an authoritative
  full book, 796 exact received-timestamp-batched five-share states spanning
  11,100.796 seconds had zero gross-positive path and zero crossed state. This
  rejects that bounded event-day preflight but does not estimate the broader
  recurrence reported by the paper. Do not repeat another isolated snapshot or
  call historical mark-to-market estimates realized profit. The now-actionable
  research contract is one bounded source-continuous event-time study across
  fixed non-augmented active BTC/ETH/SOL negative-risk events; exact fees,
  latency, atomicity, capacity, and cash-realized outputs are gated behind a
  continuity-valid gross-positive state. Canonical gate:
  `docs/model-research/action-value/polymarket-negrisk-converter-recurrence-gate-v1-2026-08-26.json`,
  result SHA-256
  `ff8b2eddeaab155327ad0d1542c0b75602342b45571443a4de61f8904165f030`.
  That bounded public study is now complete. One five-minute connection captured
  14,156 frames across all 22 fixed events. Only event `106981` was gross
  positive, and only in one delivered source frame at the five- and twenty-share
  ladder points. Those are two quantity evaluations, not two independent states.
  Current all-taker fees make the paths `-0.07082` and `-0.28328` pUSD. Moving
  only the Bitcoin-NO input to a zero-fee maker fill leaves merely `0.00740` and
  `0.02960` pUSD before conversion and every external cost. The relevant full
  books arrived in one subscription frame but their source timestamps span
  202.407 seconds, so transport continuity does not prove atomic event-time
  continuity. A separate exact Polygon V2 receipt from 2026-08-24 proves one
  85.1-share Bitcoin-NO taker sell consumed 7.38 maker shares at 0.83 and 66.72
  at exactly 0.82, both with zero maker fee. This proves price-level maker-fill
  feasibility for some queue position. It does not bind a newly placed order's
  queue or the first post-fill Gold-YES and S&P-YES books. Official source and
  one exact successful conversion now prove the current V2 collateral adapter
  calls the legacy adapter, but the observed conversion used all three NO
  positions (`indexSet=7`), not the candidate one-NO index set. Its 479,446 gas
  units cover the whole outer transaction. Reusing those units at one current
  Polygon gas recommendation and executable POLUSDT ask makes the five-share
  margin negative and leaves only `0.0092570280188902383064000` pUSD minus a
  USDT sensitivity at twenty shares before other costs. This is not same-unit
  after-cost proof. The prior V2 README deployment-address conflict is now
  resolved by the current official Contracts registry, which explicitly declares
  itself the single source of truth, labels `0xd91E80...35296` as the deprecated
  CLOB-v1 adapter, and lists `0xadA200...6eAab` as the current pUSD NegRisk
  collateral adapter. The dated official changelog says the V1 relayer route was
  fully retired on 2026-07-17, and current official source at commit `ccc0596`
  confirms that the V2 wrapper invokes the legacy adapter internally. This
  removes only the address-identity blocker. Exact account conversion access,
  approvals, user gas or relayer charge, latency, adverse selection, and
  after-external-cost profit are unproved. The official batch and single-token
  historical-price endpoints both
  returned current points outside the requested 2026-08-24 window; those
  responses were retained and rejected without guessed retries. Canonical gate:
  `docs/model-research/action-value/polymarket-negrisk-maker-input-gate-v1-2026-08-26.json`,
  result SHA-256
  `d4e02d2d1cc6b0a598265af734b29f62aec6145bc5a1cc3b3d65771ba2031d2a`.
  Canonical address resolution:
  `docs/model-research/action-value/polymarket-negrisk-v2-adapter-address-resolution-v1-2026-08-27.json`,
  result SHA-256
  `e11810a0215521cb5ad0c0c966340b4ff943760fda516e7841430fe057fe25fe`.
  A distinct organic-taker fee overlay is now accepted without changing the
  NegRisk verdict. The official fee page defines the current crypto fee as
  `shares * 0.07 * price * (1-price)` and says takers can earn a portion back.
  One complete public 2026-08-25 UTC window contained 1,202 BTC/ETH/SOL and
  other crypto taker trades. Their fee curve reconstructs 302.8176185015 pUSD;
  the next public activity row and exact successful on-chain pUSD transfer both
  equal 54.5062 pUSD, or 17.999679% of that fee, matching the documented Gold
  18% tier within 0.001 pUSD. The fee rate is per fee curve, not trade notional:
  at 0.50 the fee is 3.5% of trade value and Gold saves 63 bips. This is a
  current V2 pUSD calculation: although the fee page retains a generic `USDC`
  label, official current source denominates fees in exchange collateral, the
  deployed exchange returns the official pUSD proxy from `getCollateral()`, and
  that token returns `pUSD`. No USDC/pUSD parity assumption is used. This is only a
  direct-cost reduction on independently justified legitimate organic taker
  flow. It grants no volume, order, account, or profit authority; never
  self-match, wash trade, manufacture volume, count a one-time bonus, or use a
  public wallet as account eligibility. Current official text conflicts between
  immediate threshold activation and the next daily update, so the stricter
  completed-update rule controls. Canonical overlay:
  `docs/model-research/action-value/polymarket-organic-taker-rebate-overlay-v1-2026-08-26.json`,
  result SHA-256
  `6a3f907dbebd0c7cc894d95054231540e50cd8e28e6264840a2840be8ac72865`.
  Applying those exact tier fractions to the retained NegRisk state closes the
  queue-free alternative without another market request. The exact nonzero
  `0.04` event fee establishes documented fee-enabled eligibility; its mixed
  subject matter is not assigned a category by assumption. At both five and
  twenty shares, all-taker break-even requires a 58.6161231584% rebate before
  external costs, above the maximum documented 50% Obsidian tier. Obsidian
  therefore still leaves `-0.010410` and `-0.041640` pUSD. A confirmed Gold tier
  would improve the maker-input twenty-share pre-external-cost margin from
  `0.02960` to `0.096272` pUSD, but queue ownership, one-NO conversion access,
  causally subsequent output books, and exact tier activation remain unproved.
  Even the Gold five-share direct-gas overlay is only
  `0.0037250280188902383064000` pUSD minus USDT, not same-unit after-cost profit.
  This does not change the NegRisk or accepted-edge verdict. Canonical overlay:
  `docs/model-research/action-value/polymarket-negrisk-taker-rebate-overlay-v1-2026-08-26.json`,
  result SHA-256
  `fbbaf4ff7a7d93f8cf5d306a829ff00518d82c9802be674fdace864cea907a60`.
  At that earlier checkpoint, accepted edges were thirteen. One frozen 24-hour public capture of only the six
  event tokens was launched under internal contract SHA-256
  `9d32e66b6d150434e4b978daafa1ea9482066230f253da4c86eb9a18504717da`.
  That terminal artifact is now audited. The connection closed after
  `19635.343` seconds (`22.726091435%` of the frozen duration), so continuity
  failed. One 99.986314-share Bitcoin-YES buy removed the mirrored Bitcoin-NO
  0.82 bid, but it cleared only `1.246314` shares beyond the initially visible
  98.74-share queue. That is insufficient for either the five- or twenty-share
  frozen hypothetical order, so no queue-censored input fill and no causal
  output unwind are admissible. Never restart or duplicate this consumed
  contract. The no-rebate five-share direct-gas sensitivity remains negative;
  a hypothetical tier-adjusted cross-unit sensitivity does not replace the
  failed post-fill gate. Canonical terminal adjudication:
  `docs/model-research/action-value/polymarket-negrisk-maker-input-prospective-terminal-v1-2026-08-29.json`,
  result SHA-256
  `613453649f84407d6216e72228bdb16005b0a5c290c6bd58fa522007de5317e5`.
  A separate retained-data binary complete-set rescore found a new historical
  candidate without another venue request. The five-hour 2026-08-15 Round 27
  cohort covered 53 BTC five-minute markets; its zero-tier path exactly
  reproduced six same-state episodes, zero venue-delay survivors, and zero
  sequential survivors. Gold creates one `0.99986788` optimistic sequential
  minimum only by choosing the better leg order after both later books are
  known; the source-time lower-rebated-cost-leg-first rule has zero Gold
  survivors and a best cost of `1.00870748`. Diamond is the first tier with one
  causal historical survivor: `0.98407728` pUSD per complete set, leaving
  `0.01592272` pUSD per share or `0.07961360` pUSD at five shares before every
  external cost. It still has zero venue-delay and zero both-order survivors;
  the opposite ordering costs `1.09417472`. This is one historical episode, not
  an accepted edge or profit claim. Do not replay, resize, or refit the cohort,
  use the ex-post minimum as an execution rule, or manufacture tier volume.
  Retry only after a direct wallet confirms Diamond or Obsidian after a completed
  daily update, then freeze one separate current prospective causal capture.
  Canonical overlay:
  `docs/model-research/action-value/polymarket-round27-complete-set-taker-rebate-overlay-v1-2026-08-26.json`,
  result SHA-256
  `948f47d9d0c2fb6cbf441da1147ae07006a897f307141dfd6ae25c85e47f13d2`.
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
- Binance zero-fee stablecoin cycles: official current promotion evidence
  justified two distinct triangles, `USDT-U-RLUSD-USDT` and
  `USDC-U-USD1-USDC`, with all selected legs advertised at zero maker and taker
  fees for eligible users. The first frozen attempt failed before admitting a
  book sample because every symbol exposed a zero `MARKET_LOT_SIZE.stepSize`
  and the implementation incorrectly rejected it instead of falling back to
  the positive whole-token `LOT_SIZE`. Its one-record exchange-info journal and
  implementation hash are preserved in the terminal-failure artifact. The sole
  recovery fixed both that interpretation and the evidence-order defect by
  fsyncing every raw response before parsing or economics. It captured 600
  synchronized six-symbol responses over 299.729 seconds. Both orientations at
  100, 1,000, and 10,000 quote units produced zero stressed-positive samples;
  the best of all 7,200 evaluations was -5.0008 bps after exact whole-token
  rounding, displayed top-level capacity, and the frozen 3-bps operational
  stress. Canonical recovery:
  `docs/model-research/action-value/binance-zero-fee-stablecoin-cycle-recovery-v2-2026-08-26.json`,
  result SHA-256
  `f44f283b311ebf8b3302dba4e1d5d6be0b956a2657483786229611f82ed5da88`.
  The family is terminal and not an edge. Do not resample it, weaken the
  operational stress, or use nonsynchronous trades/candles as fills.
- Binance spot maker rebates: current official API documentation exposes the
  same account's liquidity-program overview, daily/weekly performance, weekly
  final rebates, and spot rebate history. It also explicitly says the symbol
  commission endpoint excludes the spot market-maker rebate rate and BNB
  discount effect. The historical static document returned HTTP 403, but the
  current dynamic official fee page now proves zero maker fees by enrollment;
  tier 1 requires `0.05%` weekly maker-volume share or `25,000,000` USD, while
  tiers 2-4 publish `-0.0040%`, `-0.0060%`, and `-0.0080%` after `0.15%`,
  `0.50%`, and `1.00%` share hurdles. This closes only the public-rate gap. Both
  credential variables remain absent, so same-account enrollment, pair scope,
  organic qualifying volume, owned fills, queue/adverse selection, inventory
  unwind, realized rebates, and after-cost profit remain unproved. Do not reuse
  taker triangles as maker economics or manufacture volume for at most 0.8 bps.
  Canonical evidence gate:
  `docs/model-research/action-value/binance-spot-maker-rebate-account-evidence-gate-v1.json`,
  result SHA-256
  `d2adda1c5ab4b561e0c238e1e874cc72edaee15ebadafbb76703251f9cd99e10`.
- Binance USDT/USDC perpetual funding differential: a distinct equal-base,
  opposite-position screen initially found recent BTC and SOL candidates. The
  corrected recent cash-flow screen retained them, but exact fee ceilings were
  only 1.58 and 4.75 bps per leg after the captured spread. The fixed
  full-history recovery then evaluated 2,898 common settlement epochs per
  candidate from January 2024, selected orientation only on the oldest third,
  and applied separate validation, test, time, direction, volatility,
  reversal/continuation, and USDCUSDT 0.98/1.02 stress-test acceptance
  criteria. BTC failed
  selection stress, validation, and four market-regime slices. SOL failed
  validation, test, most regimes, and its fee gate. Zero candidates passed.
  The v3 attempt also violated the one-use evidence rule by discarding fetched
  payloads after a later missing-FX validation error. Its sole v4 recovery
  durably journaled all 20 responses before evaluation and is terminal; do not
  repeat the backfill. Canonical result:
  `docs/model-research/action-value/binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json`,
  result SHA-256
  `8e30be61daaecabd3546e41cdc204d20b8ad38e0fc80c3c9aa96092266a3abe5`.
- Binance U-settled BTC/ETH perpetual versus matched USDT perpetual funding is
  also terminal under the frozen static orientation. Each base had 171 aligned
  settlements split oldest-first into 85 training, 43 validation, and 43 test
  rows. Training selected short USDT and long U; gross funding then reversed
  negative in validation and test for both bases, and all six base-role results
  were negative after the frozen execution and two-leg capital hurdles. Zero
  candidates passed, so the contract forbids book/account escalation. Do not
  repeat an unchanged quote-stable funding-differential screen or hindsight
  switch orientation by role. A materially new payoff or incentive is required.
  Canonical result:
  `docs/model-research/action-value/binance-u-usdt-funding-differential-v1-2026-08-26.json`,
  result SHA-256
  `486b1aa261ae41fd8d8aeb19f0fea5bb01305d24927ccd72624bdd8afb7895d7`.
- Binance USD1-settled BTC/ETH perpetual versus matched USDT perpetual funding
  is terminal under its frozen static orientation and conservative FX stress.
  BTC had 309 aligned settlements and ETH had 165; training selected short
  USD1 and long USDT for both. Although raw funding remained positive in each
  chronological role, every training, validation, and test result was negative
  after the frozen 20-bps execution hurdle, two-leg capital hurdle, and the
  observed 23.9545-bps worst 30-day USD1USDT decline stress. Zero of two public
  candidates passed. Do not repeat this unchanged screen or escalate to account
  data. Canonical result:
  `docs/model-research/action-value/binance-usd1-usdt-funding-differential-v1-2026-08-26.json`,
  result SHA-256
  `0d82da55d7687b0f26dc12103f38394b3325542a62af10abf4d34609fa5a6e79`.
- Polymarket Combos are source-proved atomic multi-leg RFQ executions, but
  executable quote requests require authenticated builder/account access. The
  public combo-leg catalog screen retained 5,000 unique volume-ranked markets
  under its frozen 50-request ceiling and still received a continuation cursor,
  so complete-catalog coverage is unproved. The observed population contained
  one genuine BTC-related leg, zero ETH legs, zero Solana legs, and one rejected
  sports false positive caused by `sol` abbreviating Solihull. No scoped
  multi-leg candidate existed. Do not repeat or enlarge the same incomplete
  contract. Canonical result:
  `docs/model-research/action-value/polymarket-combo-catalog-v1-2026-08-26.json`,
  result SHA-256
  `61dd2782963295bae975ed4f929b6191f23bfe0970ea02d56c6ec0fc9412a2ae`.
- Polymarket Perps now expose BTC, ETH, and SOL perpetuals plus a documented
  conditional 6% APR OI reward, but the account or mapped entity must sustain at
  least $1 million of daily-average gross OI. The frozen Perps-versus-Binance
  perpetual attempt retained a complete BTC source window, then stopped without
  retry on the first `429` during ETH pagination. It also exposed an unfrozen
  exact-timestamp assumption: Polymarket funding rows arrived 4-172 ms after the
  UTC hour, so exact equality produced zero aligned rows. A non-authoritative
  nearest-hour BTC diagnostic failed all training, validation, and test roles
  after conditional reward, 20-bps execution, and two-leg capital hurdles.
  Terminal artifact SHA-256:
  `01bf5ea0a3f293e8e14b2f484ac2715c6c326ddfab57dba51873bf16e76a62e4`.
  A separate offline source-derived hedge then tested short Polymarket BTC Perps
  against equal-base long Binance BTC spot, which avoids a second funding leg.
  Missing Polymarket hours were valued at zero. All three fixed ten-day roles and
  every direction/volatility slice had positive excess before one-time friction,
  but only 25.7920876712 bps remained after the conditional reward and two
  unlevered 5% annual capital hurdles. The frozen 50-bps round-trip-and-basis
  gate therefore rejected it at -24.2079123288 bps. It is a ranked conditional
  lead, not an edge; require exact authenticated eligibility and an all-in cost
  ceiling below 25.7920876712 bps before any book request. Canonical result:
  `docs/model-research/action-value/polymarket-perps-binance-spot-oi-carry-v1-2026-08-26.json`,
  result SHA-256
  `0d89a4fee61cf51c2f5f8c2491c9d8c3cc4a5e08046206dbe2ba20f7ba3ea934`.
  A distinct reward-stack sensitivity reused the retained BTC diagnostic without
  another venue request: short Polymarket BTC Perps, long Binance BTCUSDT
  perpetual, and use BFUSD as reward-bearing collateral on the Binance leg.
  The full 592-hour window would need at least 5.1825556081% annual BFUSD yield
  for already-held collateral, but the fixed-role persistence gate needs
  14.1066194737% in the worst role. The official live product page displays
  5.03% seven-day average APR and 5.12% last-day APR. It now states zero
  purchase fee, Fast Redemption free for the first 500 BFUSD then 0.1%, and
  two-day Standard Redemption at 0.025%; even fee-free-under-500 collateral
  misses the fixed-role gate by more than 8.98 percentage points. The validated
  2.01091% LDUSDT alternative would
  contribute only 13.5897 bps and leave the full window 21.4340 bps negative.
  Exact rate, rewards, and quota endpoints are signed USER_DATA. This is not a
  stable or accepted edge, and the inherited alignment remains non-authoritative.
  Do not request credentials, accounts, funding, or books unless a materially
  new official displayed APR first reaches 14.1066194737%. Canonical gate:
  `docs/model-research/action-value/polymarket-perps-binance-bfusd-collateral-stack-gate-v1-2026-08-26.json`,
  result SHA-256
  `a6ff387d70d33c40951e36de93eff7c810b2291dbefff5ecb0f3953880fe7878`.
- Binance Prediction Trading versus Polymarket was frozen as a distinct
  structural screen, permitting only exact BTC/ETH/SOL payout-rule equivalence.
  It stopped before market access. The official generated Markdown said market
  list required no authorization, while the same pinned connector's generated
  Java transport attached `binanceSignature`; the live no-key request returned
  HTTP 400 with `-2014 API-key format invalid`. Two attempts of one exact URL
  produced zero market payloads; no detail, Polymarket, or book request followed.
  Do not inspect stored credentials or retry without explicit read-only
  credential authority and a new frozen authenticated contract. Contract:
  `docs/model-research/action-value/cross-venue-prediction-parity-screen-contract-v1.json`,
  result SHA-256
  `6cfb13c1088ab4356f8a037df0d2f059e94fba03029a6746badfe3a6d2ea9f5c`.
  Terminal result:
  `docs/model-research/action-value/cross-venue-prediction-parity-screen-v1-2026-08-25.json`,
  result SHA-256
  `628e63106bc3c0e28c36dcad094b7d7ac500ecd14dfff827287030c2dbbb3d72`.
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
- Binance option put-call parity versus same-expiry quarterly futures is a
  distinct fixed-payoff candidate. The frozen snapshots contain 192 common
  BTC/ETH expiry-strike pairs, 184 with at least one complete ticker path, and
  20 non-synchronous gross-positive combinations. The best stale value was
  101.70 USDT per base unit, but ticker prices have no displayed quantity and
  the futures books arrived about 2.25 hours later, so this was request
  justification only. The frozen eight-GET audit was hosted-verified and run
  exactly once. Its first request, for BTCUSDT on 2025-09-26, returned HTTP 200
  with a valid empty array. The raw body was durably journaled before the
  zero-row validation stopped the run. No other request, retry, adaptive time
  shift, or depth screen is permitted. The empty response neither proves nor
  disproves common settlement values; it leaves the required numeric identity
  unproved. The contract had ordered its windows oldest-first without first
  source-binding retention or archive availability. Future bounded historical
  endpoint contracts with unknown retention must preregister newest-first
  ordering and fail closed if the newest response cannot prove record
  availability. Contract:
  `docs/model-research/action-value/binance-option-future-settlement-equivalence-contract-v1.json`,
  result SHA-256
  `63a57771fe7042381bea0ac052889550738b4890b6c01fadc279e793189b4291`.
  Terminal adjudication:
  `docs/model-research/action-value/binance-option-future-settlement-equivalence-terminal-adjudication-v1.json`,
  result SHA-256
  `a5d34919c4e9c94ca794b73dea57c96bf9c6f9e968cc6c77f240f243c7597601`.
  Retry only if an official static history or explicit retention contract
  becomes available, or after 2026-09-25 under a separately frozen prospective
  contract.
- Binance BFUSD/RWUSD stable-value yield allocation is a direction-neutral
  candidate, not a market-invariant or accepted edge. Official current API
  documentation exposes signed read-only `rateHistory` and `quota` GETs for
  both products plus flexible-product-list GETs for exact eligible USDT/USDC
  alternatives, including account-specific subscription capacity and fast and
  standard redemption fee/delay fields. The current process had neither
  designated ephemeral credential variable, so zero signed requests were made.
  A logged-out Binance Earn page displayed an RWUSD 3.36% APR promotion but no
  comparable BFUSD rate; that observation is excluded from economic evidence.
  Prequalification must compare timestamp-integrated rates with the best exact
  same-currency alternative yield and every entry, redemption, delay, custody,
  tax, and opportunity cost. Subscription/redemption remains a separately
  authorized funded stage. Canonical gate:
  `docs/model-research/action-value/binance-stable-yield-allocation-evidence-gate-v1.json`,
  result SHA-256
  `3096867474c4b5a0b3f893645bac68081ceb3783ad14393261e6d88793b64a8a`.
  A new public BFUSD spot route was then screened against the 1:1 subscription
  and redemption identity. `BFUSDUSDT` had a 1.0000 bid and 1.0001 ask; no
  100, 1,000, or 10,000 BFUSD buy/redeem or subscribe/sell path was positive
  under the labeled 10-bps spot and typical 10-bps product-fee sensitivities.
  Its 379 public daily bars ranged from 0.995 to 1.08, so a prospective depth
  trigger is justified after exact account costs are known, but those trade
  extrema are not historical depth or fill evidence. Canonical screen:
  `docs/model-research/action-value/binance-bfusd-spot-redemption-parity-v1-2026-08-26.json`,
  result SHA-256
  `566be5e515ac14d38377b6a6b42101cc9b8a65585142053791b759efbd77f6bb`.
  A current public-promotion triage then found a conditional, direction-neutral
  RLUSD reward candidate. The first two completed campaign weeks published
  8.07% and then 5.78% effective APR with no stated individual cap, but participation requires an
  eligible account, RLUSD collateral, and at least 500 USD average daily genuine
  Margin or Futures volume. Future weekly APRs are unknown and the campaign ends
  2026-09-11. Treat qualifying activity as zero incremental cost only when it is
  independently genuine pre-existing volume; never create churn or self-trades.
  The current USDT Flexible promotion has a source-bound 14-day window and a
  capped combined gross reward of about 1.34 USDT at the advertised approximate
  base rate. The same article's USDC current-offers row displays approximately
  2.5% Real-Time APR plus 5% on the first 200 USDC, but its dated promotion
  sentence names only USDT. No USDC start or end time is published there, so
  USDT's dates must not be transferred to USDC and its guaranteed forward public
  reward floor is zero. Neither public observation proves account eligibility.
  Observe the scheduled 2026-09-04 and 2026-09-11 RLUSD updates without
  extrapolating the completed weeks. Canonical triage:
  `docs/model-research/action-value/binance-public-promotion-yield-triage-v1-2026-08-26.json`,
  result SHA-256
  `26efd481a5ff424ca17ec803bb6a1a3ae8949d1fe0fc31a03e20a35d08d031ac`.
  A separate public USD1 Flexible promotion begins 2026-08-27 and ends
  2026-09-25. It offers a fixed 7% bonus on at most 1,500 USD1 plus an
  approximately 1.5% variable base rate. The conservative 28-bonus-day case
  exceeds the contemporaneous USDT alternative by 25.0228 bps. However, the
  worst observed 30-day USD1USDT close move was -23.9545 bps, so only 0.9683
  bps remain after the current displayed 0.1-bp round-trip spread and before
  every unknown commission, conversion, eligibility, reserve, and redemption
  cost. One public 30-day window was worse than -19.1694 bps; none was worse
  than -29.1694 bps, but an intraday low reached 0.9000. The latest attestation
  month listed by BitGo was June 2026 and the live public reserve dashboard did
  not return a current collateralization ratio. The offer subsequently
  activated with the same terms; the one permitted public refresh observed a
  0.10001-bip top-book spread but did not improve the frozen 0.96835-bip margin
  before unproved account costs and peg or redemption risk. It remains a
  conditional time-limited candidate, not a stable edge or deployment
  authority. Canonical gate:
  `docs/model-research/action-value/binance-usd1-simple-earn-promotion-gate-v1-2026-08-26.json`,
  result SHA-256
  `230b1524f337964394a45ffe047adfd19b35b339a7735866a15cafdd7549c6f1`.
  Post-activation refresh:
  `docs/model-research/action-value/binance-usd1-simple-earn-activation-refresh-v1-2026-08-27.json`,
  result SHA-256
  `f8106a93155813a3130bc925a3f4b223fad16b6133ee073226251d25175ecf06`.
  Exact account product eligibility, current rates, capacity, entry and exit
  costs remain signed evidence and any funded action still needs separate
  authority.
- Binance bStock structural parity is a promising research-only candidate
  outside the current BTC/ETH/SOL execution scope. A public 67-symbol screen
  selected `SNXXBUSDT`; exact displayed depth stayed positive at 1,000 and
  5,000 USDT after a labeled 20-bps spot-plus-stock cost sensitivity, by only
  0.6463 and 2.0877 USDT. Binance documents eligible-user 1:1 stock/bStock
  conversion with no conversion fee, but its public external reference price
  is not an executable Binance Stocks sale quote. Separately, direction-neutral
  long-bStock / short-matching-TradFi-perpetual diagnostics for DRAM, MU, MRVL,
  and SNDK cleared a labeled 30-bps round-trip sensitivity in every available
  complete inner month; the histories contain only two or three complete
  months. Account eligibility, exact commissions, synchronous stock quotes,
  conversion state, hedge-unit mapping, margin/capital costs, prospective
  persistence, and permanent scope authority remain absent. Canonical artifact:
  `docs/model-research/action-value/binance-bstock-reference-parity-v1-2026-08-26.json`,
  result SHA-256
  `73fec22cc61fc8be0c792a78c0340fcb163b9bf7862708796d50521a9c44a8ac`.
  A frozen full-universe follow-up matched 66 bStock/perpetual pairs, retained
  60 exact-multiplier pairs, and excluded three previously observed exact
  tickers from 57 confirmation candidates. Zero passed the training-only gate
  after the 60-bps round-trip stress, 10%-annual two-leg capital hurdle,
  family-adjusted block bootstrap, risk limits, and eight regime slices.
  Sixteen had positive point-estimate training net, but all 57 had nonpositive
  adjusted bootstrap lower bounds and 56 failed at least one slice. Canonical
  result:
  `docs/model-research/action-value/binance-bstock-funding-full-universe-v1-2026-08-26.json`,
  SHA-256
  `ad3fbc7a09ff6b467955eeef8bf1e8df4ba7d20ca9e7659fcaf75069da622d3f`.
  Before opening validation/test outcomes, a distinct top-20% training-ranked
  equal-weight basket was frozen. It failed validation at -81.80 bps and test
  at -108.20 bps; only 1/12 then 0/12 symbols were positive and every
  aggregate direction, volatility, and path slice was negative. Canonical
  result:
  `docs/model-research/action-value/binance-bstock-ranked-basket-v1-2026-08-26.json`,
  SHA-256
  `0cf6e3aae168e0c483634e78fd824a80be9e58269f02e9b01a6d9c9c46578a8f`.
  The current bStocks zero-maker promotion does not reopen this funding family.
  A retained-data counterfactual erased the full frozen 60-bp two-leg execution
  stress, an impossible upper bound because the promotion removes only the
  otherwise applicable bStocks maker fee on qualifying fills. The selected
  basket still lost `21.8046206700` bps in validation and `48.1975271954` bps
  in test; cross-sectional bootstrap lower bounds remained `-42.0480030822`
  and `-55.4780525299` bps, and test remained zero of 12 positive. No funding
  or book request is justified for this promotion. Canonical retained
  counterfactual:
  `docs/model-research/action-value/binance-bstocks-zero-maker-carry-retained-counterfactual-v1-2026-08-29.json`,
  SHA-256
  `2af1504748f51ad36c18c76162a91e82803395311932d73f1809d2781dfc4fb7`.
  The funding family is terminal without parameter retry. Conversion parity
  remains account, executable-stock-quote, exact-cost, and permanent-scope
  gated. Neither path is an accepted edge and neither permits repeated polls.
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
  Read-only mainnet account-evidence authority was received afterward. A
  GET-only capture is frozen at
  `docs/model-research/action-value/binance-quarterly-carry-account-evidence-contract-v1.json`,
  result SHA-256
  `901c16bf3e7e4082339f3ddd2a910a904a3cd46d51c0dc16f7074c16351145e5`.
  It binds two exact spot commission responses, four exact quarterly-futures
  commission responses, one minimal futures account-configuration response,
  fresh venue time before every signed GET, zero retries, and a durable
  secret-free request journal. The required mainnet API key and secret process
  variables were absent at preflight, so no authenticated request or one-use
  attempt occurred. Do not search chat, logs, history, or repository files for
  the raw values. When both variables become available to the process, run the
  frozen capture once; use the result as a non-synchronous fee rejection gate,
  not as a current edge or reason to refresh books prematurely.
- Binance two-leg quarterly calendar spread: long current-quarter and short
  next-quarter futures in equal base quantity does not lock the initial
  far-minus-near credit. Before expiry, gross PnL equals the initial spread
  minus the exit spread; at near expiry, it equals the initial spread minus the
  unknown next-quarter-future-minus-spot basis. This is residual curve exposure,
  not a fixed payoff. The mechanism failed before estimation, so zero request
  and no backtest were justified. Do not repeat this family as locked carry.
  Canonical adjudication:
  `docs/model-research/action-value/binance-quarterly-calendar-spread-mechanism-adjudication-v1.json`,
  result SHA-256
  `4add8a3aea01ebb13e743a85793681b8ca7a8884035daf5cb371f3f2b09900b0`.
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
- The separate August crypto-TWAP liquidity-reward screen excluded 5-minute
  markets and froze six exact BTC/ETH/SOL 15-minute and 4-hour identities. It
  corrected a material economic error before capture: a daily reward equivalent
  must be multiplied by exact market milliseconds divided by 86,400,000, and
  the frozen public escalation gate requires the 100-times competition-stressed
  full-market conditional reward to cover maximum orphan settlement loss. The
  one attempt stopped after Gamma and the first exact BTC 15-minute reward GET
  because exact reward identity was not proved; no books were requested. The
  transient responses were not retained because the implementation wrote
  sources only on success, repeating the workflow defect documented above.
  Do not retry or infer the lost shape. Canonical failure receipt:
  `docs/model-research/polymarket/crypto-twap-liquidity-reward-screen-attempt1-failure-v1.json`,
  result SHA-256
  `e486f2928a326e6829cbe3c07aad5a47bb25a63783a935273606df00cea98c66`.
  The tool now atomically journals request intent, raw response, and response
  hash before parsing; focused tests enforce the correction for any materially
  triggered future contract.
- The official current program then materially changed by adding a 550,000
  dollar five-minute allocation across BTC, ETH, SOL, XRP, HYPE, BNB, and DOGE.
  That reopened only the previously excluded duration. A frozen seven-market
  attempt at 2026-08-27 06:45:05 UTC retained its official documentation and
  Gamma responses before validation. All seven exact markets were returned
  with a 50-share reward minimum, 4.5-cent maximum spread, and the same
  taker-only 0.07/exponent-1/0.2-rebate fee schedule, but raw Gamma omitted the
  optional `clobRewards` field on every row. The collector stopped before any
  book request, so no exact daily allocation, reward economics, candidate, or
  profit was established. Canonical terminal artifact:
  `docs/model-research/polymarket/crypto-twap-5m-liquidity-reward-screen-attempt1-terminal-v1-2026-08-27.json`,
  result SHA-256
  `319c6aedbb5491e56e68cc3fdf95f366766ce4a070dcadec3213471ff938120d`.
  Do not retry the same source configuration or derive a daily rate from the
  monthly program cap. The current SDK model makes `clobRewards` optional; a
  normalized SDK field is not proof that raw Gamma populates it.
- A second frozen attempt used the materially distinct unsigned public current
  rewards list documented by official SDK commit
  `41c642c2d056e5a697fd5962498ca5a7313ac8ef`. At 2026-08-27 07:20:05 UTC,
  `/rewards/markets/current?sponsored=true` returned its complete 54-row
  population in one page with the terminal `LTE=` cursor. Zero row matched any
  of the seven exact five-minute condition IDs, so the collector stopped before
  books. Canonical terminal artifact:
  `docs/model-research/polymarket/crypto-twap-5m-current-rewards-list-join-v1-2026-08-27.json`,
  result SHA-256
  `62940fa602d71259aab1326eb038069b23df6f247dd898a952f493fcebc38e6f`.
  This closes the distinct current-list source path as well as Gamma; do not
  repeat either absent another program change or a genuinely new exact dated
  per-market allocation response.
- A distinct complete-set inventory stack screen did not repeat paired-maker
  books or wallet polling. Gamma metadata showed 12 of the 55 currently
  holding-reward-eligible BTC/ETH/SOL annual thresholds with positive reward
  spread and size fields, but exact-condition reward requests with
  `sponsored=true` proved that only five have any current pool. Each is capped
  at 0.001 pUSD/day, all ETH candidates and seven others have zero current
  configuration, and the combined five-market pool is only 0.005 pUSD/day.
  Even assigning an entire market pool to one 20-share minimum order is only
  0.5 bp/day; an illustrative BTC $160,000 one-leg settlement orphan is 18.64
  pUSD, or 18,640 days of that whole pool. Official sources also do not state
  that tokens committed to resting orders continue earning the holding program.
  No books, authentication, or order experiment is economically warranted.
  Canonical terminal gate:
  `docs/model-research/polymarket/complete-set-liquidity-reward-stack-gate-v1-2026-08-26.json`.
  Result SHA-256
  `f0df2e95e36315850b7f4e8b6600e742490ee34151317628a5304e35c037146e`.
  Do not poll unchanged configurations. Reopen only after explicit official
  stacking semantics, a materially larger exact-condition pool, and a frozen
  loss-bounded restoration design.
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
  A distinct public recurrence pass then proved realized program payments but
  not profitable making. All top ten weekly crypto-volume wallets received
  `MAKER_REBATE` activity on all seven queried UTC dates. For the top monthly
  crypto-volume wallet, the 2026-08-25 market-level endpoint returned 905 rows
  totaling 7,457.259568 USDC. An exhaustive condition join found 668 BTC, ETH,
  and SOL markets paying 7,017.331032 USDC, or 94.10% of that wallet-day total,
  across 5m, 15m, and 4h contracts. The weekly cohort's 38.5082-bp aggregate
  receipt-to-crypto-volume number is diagnostic only: activity can include
  other categories, leaderboard volume is not maker-only, period boundaries
  are not identical, and public PnL accounting semantics are unspecified.
  Neither source exposes queue, quote duration, paired fills, inventory path,
  adverse selection, orphan loss, or complete costs. The payout floor for a
  fresh hypothetical order remains zero and no edge is accepted. Canonical
  recurrence:
  `docs/model-research/polymarket/crypto-maker-rebate-public-recurrence-v2-2026-08-26.json`,
  result SHA-256
  `c992e0e1febc1a9789289cb129c166280ee0192cab203d3a6935a8c40e949612`.
  Do not repeat public wallet polling unless the program terms change.
- Polymarket complete-set holding yield is now an accepted structural edge
  after direct relayer split/merge cost only for existing idle pUSD already on
  Polymarket. Official mechanics
  allow splitting 1 pUSD into one YES plus one NO and later merging the pair
  back to 1 pUSD. Equal mergeable balances therefore remove outcome direction
  while the holding program is active. Canonical economics:
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
  `2d3650d65f248294395fcac336c6650e0c6bc332cb490c6f0bac70bc11244e2c`.
  The v2 public diagnostic used the wrong subtype. Official SDK bindings define
  account-level holding payments as `YIELD`; generic `REWARD` rows are a
  different program and the selected wallet's latest receipts came from the
  liquidity-reward distributor. Do not repeat that diagnostic.
  The corrected public reconciliation identified wallet
  `0x3fb5c98d825651d7efd2bd48a5d02c2d86c96f2f` with exactly 150 YES plus
  150 NO mergeable shares in “Will Bitcoin dip to $45,000 by December 31,
  2026?” and no other currently holding-reward-eligible position value. Its 14
  consecutive public `YIELD` rows total 0.1816 pUSD, and all 14 Polygon receipts
  exactly reconcile pUSD transfers from the holding-yield distributor. The
  3.15638% realized annualized gross rate and 328/336 implied sampled hours fit
  the current official 3.25% hourly formula exactly at four-decimal payout
  truncation; the stale 4% rate does not. Canonical reconciliation:
  `docs/model-research/polymarket/complete-set-holding-yield-reconciliation-v3-2026-08-26.json`,
  result SHA-256
  `48e31f3d6021d28946fa1f143f65ff0f6baf9a222424f41e76c2d89875796abe`.
  The independent ETH/SOL follow-up reconciles split-origin lineage on-chain:
  ETH has 440 complete sets remaining from a 550 split after 110 merged, while
  SOL has 449 remaining from a 550 split after 101 merged. Each wallet contains
  only its equal mergeable pair. All 28 daily `YIELD` transfers reconcile to
  successful Polygon receipts; both assets map to 330/336 sampled hours and
  about 3.186% realized annualized gross. Canonical cross-asset evidence:
  `docs/model-research/polymarket/complete-set-holding-yield-cross-asset-v4-2026-08-26.json`,
  result SHA-256
  `eda29a314218e1724e39984e2712a4351d9e697503d4583d391c89a060ba53ea`.
  The source-bound v5 net-economics adjudication aggregates 1,039 pUSD of
  demonstrated principal, 1.2681 pUSD of reward over 14 days, and 42/42
  positive daily payouts. The principal-weighted realized rate is 3.182019%.
  The documented relayer pays successful split/merge gas and the CTF split and
  merge identities preserve principal, so this is positive after direct
  mechanism cost. All three cases remain positive versus a 3% annual
  alternative before external friction, but the weighted spread is only
  18.2019 bps. Ten bps of external friction takes 200.53 days to recover at
  that alternative yield, the 127-day friction budget is only 6.3333 bps, and
  none of the realized cases beats a 3.25% alternative. Canonical result:
  `docs/model-research/polymarket/complete-set-holding-yield-net-economics-v5-2026-08-26.json`,
  SHA-256
  `dff80903a20d9bfc8e3402eea01dad8a8f5ee39b0427690514cc30b9fe9dcb85`.
  A bounded same-day continuity check then found contradictory current
  Polymarket terms: the dated Help Center page still says 3.25%, while the
  developer `Positions & Tokens` page says 4.00%. No new `YIELD` row existed
  after either source capture, so 4.00% is not adopted as operative or
  realized. BTC's 150 and ETH's 440 complete sets were unchanged; SOL had
  increased from 449 to 591.11, so its post-change denominator must be reset.
  The accepted historical 3.182019% result remains intact, but the current
  prospective rate is unqualified until realized post-conflict evidence or an
  explicit effective-date source resolves the contradiction. Canonical gate:
  `docs/model-research/polymarket/complete-set-holding-yield-rate-conflict-gate-v6-2026-08-26.json`,
  SHA-256
  `17c23b1bf821256a573b8685ea4c5725d1c1315a4ca6449395e75635b51678d9`.
  The frozen one-use post-conflict refresh was consumed on 2026-08-29 and
  failed closed after five of nine requests. BTC and ETH retained exact equal
  mergeable balances, no non-YIELD activity during the selected daily
  intervals, and their first wholly post-conflict payouts of 0.0133 and 0.0391
  pUSD each uniquely map to 24 sampled hours at 3.25%, not 4%. The runner then
  stopped because SOL's separately rounded displayed current values summed to
  591.1099 rather than its exact 591.11 equal-share balance. That 0.0001 pUSD
  difference was an unstated implementation gate, not a balance change. No SOL
  activity or receipt request was made, so the current rate remains fail-closed
  unqualified; do not repair or rerun because the retained 3.25% evidence cannot
  improve the v5 economics against a 3.25% alternative. Canonical failure
  adjudication:
  `docs/model-research/polymarket/complete-set-holding-yield-post-conflict-v7-failure-adjudication-2026-08-29.json`,
  SHA-256
  `448b068aa5c1b34c6012a5fadafa449ed9ef125afc310b7901b9f68285510f71`.
  V7's already retained BTC and ETH activity payloads each contained one later
  daily `YIELD` row beyond the selected rate-conflict interval. A distinct
  frozen monitor made exactly two public Polygon receipt requests and no Data
  API requests. Both rows reconciled to successful exact pUSD transfers from
  the holding-yield distributor: BTC paid 0.0133 pUSD after 86,474 seconds and
  ETH paid 0.0391 pUSD after 86,724 seconds. This is additional current payout-
  mechanism continuity, not a v7 repair or three-wallet rate qualification;
  current-rate, external-cost, ownership, and deployment gates remain closed.
  Canonical continuity result:
  `docs/model-research/polymarket/complete-set-holding-yield-continuity-receipts-v8-2026-08-29.json`,
  SHA-256
  `2eb7b434170afb195cc4f4faef8260ac4ec30b655c20fc07ee1bc9acbdfe090d`.
  The next retained daily-window trigger was later satisfied. A frozen
  one-request BTC activity pulse found exactly one new `YIELD` row for 0.0133
  pUSD at `2026-08-30T00:13:40Z`, an 86,290-second interval. The pulse stopped
  before receipt access. One separately frozen transaction-specific Polygon
  receipt request then reconciled the exact successful distributor-to-wallet
  pUSD transfer. BTC now has 18 observed positive rows. This is stronger
  current payout continuity, not a v7 repair, full three-wallet current-rate
  qualification, deployment authority, or positive public profit floor for
  new capital. Do not repeat v9 or v10; absent a material official terms
  change, no next single-wallet continuity pulse is allowed before
  `2026-08-31T02:15:30Z`. Canonical receipt result:
  `docs/model-research/polymarket/complete-set-holding-yield-payout-receipt-v10-2026-08-30.json`,
  SHA-256
  `4e57d6c0216886144fb89f8ae69b11a2eee4db37149ce6c956adecf293b7b927`.
  A separate frozen valuation-uplift screen covered every currently eligible
  BTC/ETH/SOL market in four public requests. All 55 markets returned both
  token midpoints, and every equal YES-plus-NO complete set summed to exactly
  1.0000 pUSD at displayed precision. Zero cleared the 1.0884312538 threshold
  required to beat a 3.25% alternative plus 10-bps friction over 127 days.
  History and book escalation are prohibited for this unchanged idea. Canonical
  result:
  `docs/model-research/polymarket/complete-set-midpoint-uplift-v1-2026-08-26.json`,
  SHA-256
  `33cdc53555f8bbdecf6a9977a77d2c3bc004dab4bff27abb36eac4452f96e5a3`.
  Acceptance remains deliberately narrow: it excludes capital not already on
  the platform and does not zero bridge, wrapping, withdrawal, custody, tax,
  failed-operation, or exact best-alternative-yield costs. The original BTC
  wallet still lacks split-origin
  proof, but that limitation is closed for the two new public cases. Fourteen
  days do not guarantee future payouts; the official rate is discretionary and
  caps may be introduced. No deployment, authenticated account, funded, paper,
  or live authority exists.

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

The canonical structural-edge priority and retry-trigger registry is
`docs/model-research/structural-edge-priority-registry-v1.json`, result SHA-256
`e712a9086d31944b42f93270256c393c6d8ab38997c20b7f8638cd4aa9088a34`.
Advance only the highest-ranked hypothesis whose trigger is actually satisfied.
This prevents account-blocked or terminal screens from being rerun as if more
snapshots could create an edge.

The static Polymarket high-price favorite taker-buy hypothesis is terminal.
The primary author's pooled calibration snapshot still shows realized win
rates of 97.7573% and 99.4049% in the 92.5% and 97.5% price bins, but the
chronological persistence contract failed: the latest PWI role had median
Prelec alpha 0.9653 against the frozen below-0.90 gate, while the earliest
longshot-gap role was negative in only 53.27% of weeks against the 60% floor.
The stricter current 1,000-non-bot-trade methodology floor excluded 37 PWI
weeks, including a retained non-null 207-trade row that conflicted with the
published suppression rule.

The causal BTC/ETH/SOL translation was worse. On the retained 2026-04-27 v1
partition, the first actual five-share-or-larger taker BUY per condition and
fixed band was charged the current 0.07 crypto fee curve, 0.001 pUSD per-share
execution/settlement stress, and a 10% annual capital hurdle through
resolution. The 90-95 cent band earned only 5.0267 pUSD across 703 conditions
in aggregate, with a negative family-adjusted lower bound and negative
training, test, BTC, SOL, Up, and below-60-second slices. The 95-99 cent band
lost 43.3821 pUSD across 951 conditions; training and test were negative and
every BTC, ETH, SOL, Up, Down, early, and late slice was negative. Do not
download more partitions, refit the bands, or convert pooled trade counts into
independent-market confidence. Canonical result:
`docs/model-research/action-value/polymarket-favorite-longshot-bias-preflight-v1-2026-08-27.json`,
result SHA-256
`31cd01740e48b2dc0c76e9ca7820b0348aa7d04e403d0aeb71560000b9630c93`.

The public Polygon pUSD external-parity lead is terminal at its captured pool
state. Official contracts prove exact one-to-one USDC.e-to-pUSD wrapping and
pUSD-to-USDC.e unwrapping, but the block-pinned USDC.e pool exposed only a
7.5839-bps pUSD premium against a 30-bps fee. The optimistic marginal complete
loop therefore lost 22.4388 bps before price impact and gas. The distinct
native-USDC pool exposed only a 0.6673-bps pUSD discount against a 1-bp fee, so
it lost 0.3328 bps before price impact, gas, and the still-unclosed
native-USDC/USDC.e basis. No Quoter or wallet call was justified. Canonical
result: `docs/model-research/action-value/polymarket-pusd-external-parity-v1-2026-08-26.json`,
SHA-256 `c15f1e131aa18d705aa6ce507c0f921b7a559664db91a352a244d8df9ddb0f99`.

A distinct public Polymarket UMA resolution-proposer reward screen is terminal
at the current reward, gas, and competition state. Thirty-nine recently closed
BTC/ETH/SOL questions each offered 0.6 USDC.e reward against a 250-USDC.e
custom bond plus 250-USDC.e final fee for 600 seconds. Every request settled
without dispute and the proposed price matched the resolved price. The five
scoped proposal transactions consumed 159.7365077952 POL; charging all eight
resolution transactions added 9.5001027951 POL. At the captured 0.11226
POLUSDT sensitivity and a 10%-annual collateral hurdle, the 23.4-USDC.e reward
pool left 4.3644 USDC.e equivalent, or 2.2382 bps on 19,500 USDC.e, only if
USDC.e is valued one-for-one with USDT. This is not a stable or accessible
edge: one actual two-market proposal batch was already negative after observed
proposal gas, pro-rata resolution gas, and capital cost; full-charge aggregate
break-even had only 22.9723% POL-price headroom; 38 of 39 scoped proposals
landed in the first Polygon block; and public events do not prove access to the
observed proposer contracts or relayers. No wallet, approval, proposal,
resolution, credential, account, or funded action occurred. Do not resample an
unchanged cluster. Reopen only after a material source-bound reward, bond,
liveness, batching, or public-access change and then freeze an all-cost,
gas-stressed, multi-day competition study. Canonical gate:
`docs/model-research/action-value/polymarket-resolution-proposer-reward-gate-v1-2026-08-26.json`,
SHA-256 `ee76a40a86e1c777006c697798d0ad3da20609cadd1c2d8f6bf039ecb79f2155`.

The highest-priority Polymarket execution lead is now the short interval after
an undisputed UMA request becomes `Expired` but before the adapter resolves and
closes the market. OptimisticOracleV2 makes this a fixed proposed-price state:
with no disputer and `expirationTime <= current time`, `hasPrice` is true and
settlement copies `proposedPrice` into `resolvedPrice`. A fixed public screen
covered five consecutive 09:00-13:00 UTC hourly clusters, 39 initialized
BTC/ETH/SOL questions per cluster, and 195 proposed-price-equals-resolved-price
markets in total. The bounded multi-market trade endpoint reached no row
ceiling. Two clusters contained one qualifying trade each. At 12:10:16 UTC,
193.67 finalized ETH Down shares sold at 0.999, 16 seconds after finality and
45 seconds before adapter close. At 13:10:52 UTC, 84.47 finalized BTC Down
shares sold at 0.999, 52 seconds after finality and 26 seconds before close.
Both Polygon settlements succeeded. Because the public endpoint was frozen to
taker-side rows, `SELL` implies the maker bought the winning token. The two
fills total 278.14 shares and 0.27814 pUSD fixed redemption discount. Both exact
CLOB markets report a 0.001 tick, five-share minimum, and taker-only fee curve;
official current terms charge makers zero, and successful relayed redemption
costs the user zero gas. A 10%-annual opportunity-cost sensitivity from each
fill to adapter close is only 0.00003457 pUSD; the same hurdle breaks even after
3.6537 days of redemption delay.

This is positive observed direct-mechanism evidence, not an accepted or stable
edge. Three of five clusters had no qualifying trade, both observed winners
were Down, capacity is only 278.14 shares, and public history does not expose
maker order creation time. A bid resting before UMA finality bears outcome risk
and is prohibited from this mechanism. Do not expand historical pages. After
`2026-08-29T23:40:00Z` and only with explicit authenticated paper authority,
freeze one minimum-size post-only bid created after a pinned exact Expired,
undisputed, non-ignore, unpaused, unflagged, unresolved state; require clean
acceptance, fill or rejection/cancellation, owned lineage, and gasless
redemption reconciliation. No credential, account, order, approval, wallet,
redemption, or funded action occurred. Canonical gate:
`docs/model-research/action-value/polymarket-finalized-winner-redemption-latency-gate-v1-2026-08-26.json`,
SHA-256 `3df84b6639c409ffca472bb4566e623ac78f160e7d8bc66795009f619edfdcb1`.

The highest-priority current Binance structural edge is now Soft Staking for
otherwise idle ETH or SOL already held in Spot. The public product page
currently displays 0.50% estimated APR, a 0.1-ETH minimum and 4,000-ETH cap,
and a 1-SOL minimum and 400,000-SOL cap. The official FAQ specifies daily
native-token rewards calculated as eligible daily-average balance times current
APR divided by 365, with no additional participation fee. At the displayed
rate, the minimum balances accrue 0.0005 ETH or 0.005 SOL per year before
rounding and external costs. This is accepted only relative to continuing to
hold the identical native-token quantity idle. It does not justify buying,
retaining, or forecasting ETH or SOL.

The scope is narrower than the product headline. Pending-order frozen assets
earn no reward, Auto-Subscribe allocations take priority, activation enrolls
all eligible Spot tokens rather than a selected asset, APR can change daily,
and the liquidity disclaimer says prompt return is not guaranteed under stress.
Inventory required for orders, withdrawal, collateral, settlement, or any
latency-sensitive operation is excluded. Exact account eligibility, signed APR,
eligible balance, activation, reward rounding, and owned distribution history
remain unproved. The official API exposes signed product-list and reward-history
GETs. Its separate `/sapi/v1/soft-staking/set` operation changes account state
despite also using GET, so it requires explicit mutation authority and must not
be called during read-only prequalification. No credential or account action
occurred. Canonical gate:
`docs/model-research/action-value/binance-soft-staking-idle-spot-yield-gate-v1-2026-08-26.json`,
result SHA-256
`9ded119650ed1679795cca8616935015bc8bf48850bfcc509ba28486e94bd9a7`.

A current logged-out public comparison now identifies ETH and SOL liquid
staking as the stronger idle-native-token lead, but not yet an accepted edge.
ETH Staking displays 2.20% and SOL Staking 4.65%, respectively 170 and 415 bips
above the 0.50% Soft Staking comparator. At those displayed uplifts, the
maximum total-cost budgets are only 13.97260274 and 34.10958904 bips over 30
days, 41.91780822 and 102.32876712 bips over 90 days, and 170 and 415 bips over
365 days. Current official Binance Academy material says subscription produces
WBETH or BNSOL, whose conversion ratio includes rewards; the BNSOL glossary
says redemption may take days.
Exact same-account quota, commission, conversion ratio, redemption period,
eligibility, owned rewards, and every delay/liquidity/alternative-yield cost
remain absent. The public forward net floor is therefore zero. Do not stake,
redeem, convert, transfer, trade, or use the receipt token as collateral.
Canonical candidate:
`docs/model-research/action-value/binance-existing-idle-eth-sol-liquid-staking-yield-candidate-v1-2026-08-27.json`,
result SHA-256
`b7fc84d0be3968d31afeb801b7a40ee0d382724b11281c28733a8145d12ee035`.

The market-direction-independent extension that combines long ETH or SOL Spot
Soft Staking with an equal-notional short USDT perpetual is now terminal under
the retained funding evidence. Both official funding responses contain 500
rows, despite the original request limit of 1,000. The first local calculation
that assumed the request limit stopped on empty validation data; no false
result was preserved. The final calculation derives its chronological
300/100/100 training, validation, and test roles from each validated response's
actual row count. At the current 0.50% Soft Staking APR and a frozen 32-bip
round-trip execution stress, ETH's training role is negative and SOL's training
role is negative even before opportunity cost. Every role for both assets is
negative after one or two 10% annual capital-leg hurdles. The maximum Soft
Staking APR needed across roles is 0.59006995% for ETH and 3.73322000% for SOL
after execution stress alone; with one 10% capital-leg hurdle the thresholds
are 10.59006995% and 13.73322000%. Do not buy or retain either asset for this
stack, infer response size from a requested limit, or rerun the unchanged
500-row history. Reopen only after a material Soft Staking rate, funding, fee,
margin-reuse, or opportunity-cost change. Canonical terminal result:
`docs/model-research/action-value/binance-soft-staking-delta-neutral-funding-stack-terminal-v1-2026-08-27.json`,
result SHA-256
`591fb98b9a8e58365c67c4a281d1fda3de674b42f1f868a42d98acf2ab19ae68`.

BTC Simple Earn is now recorded as an unaccepted candidate extension to this
same idle-native-token family. The official current BTC page displays Flexible
Simple Earn as principal protected in token amount at `0.27% Max`, with an APR
breakdown of `0.02% + 0.25%`. Its own FAQ requires the product-specific minimum,
identity verification, and regional availability, but the public page exposes
neither the minimum nor the bonus cap, end date, exact account eligibility, or
fees. After clicking the public one-year control and waiting three seconds per
input, the calculator showed 0.00002172 BTC on 0.01 BTC, 0.00003997 on 0.1 BTC,
0.00022247 on 1 BTC, and 0.00202922 on 10 BTC. The declining effective return
proves the headline maximum cannot be credited to the whole balance; the page's
own disclaimer and hidden horizon/tier state prevent reverse-engineering an
exact contract from those estimates. No credential or account action occurred.
Do not acquire or retain BTC for this yield, use operational BTC, enable Auto-
Subscribe, or subscribe. With both ephemeral credentials and explicit read-only
authority, freeze one exact BTC Flexible product-list, position, and reward-
history prequalification; subscription and redemption require separate funded
authority. Canonical candidate:
`docs/model-research/action-value/binance-btc-simple-earn-idle-yield-candidate-v1-2026-08-26.json`,
result SHA-256
`193495029148d0022fe1bf4158442226705a7f62a22dbf0eafdbf9a53bece785`.
The same artifact rejects the adjacent current `0.2%~0.41%` locked BTC
On-Chain Yields headline. Binance classifies On-Chain Yields as high risk,
states rewards depend on the protocol and are not guaranteed, assigns smart-
contract risk to users, and warns protocol failure can lose assets. The public
BTC page does not identify the exact offering or bind principal return, reward
asset, lock, redemption, fees, slashing, or protocol terms. Do not reopen on a
higher APR alone.

The logged-out VIP Earn page is now publicly enumerable, but its current
BTC/ETH/SOL rows do not create a displayed-rate edge. BTC VIP at
`0.25%~0.41%` merely equals the best visible non-VIP maximum; ETH VIP at
`1.70%~1.90%` trails the public 2.20% ETH Staking row; SOL VIP at
`3.78%~4.50%` trails the public 4.65% SOL Staking row. Zero in-scope displayed
maximum-uplift rows survive, and exact same-product duration, quota,
redemption, account eligibility, and costs remain missing. A separately found
official locked-products PDF was last modified 2025-05-28, 441 days before the
2026-08-12 VIP Earn launch, so it is discovery provenance rather than current
rate evidence. Do not repeat the public snapshot unless a material rate or
terms change occurs. Canonical terminal snapshot:
`docs/model-research/action-value/binance-vip-earn-public-btc-eth-sol-comparator-terminal-v1-2026-08-27.json`,
result SHA-256
`cd41cad8e0053b9d41ddda64fd4ad8a86a163307ddcc9fabc805c56b9c5028c9`.

The next accepted Binance structural edge is LDUSDT incremental margin
yield. Official product guidance says eligible USDT Simple Earn Flexible assets
can become LDUSDT, remain usable as USD-M Multi-Assets margin, and continue
earning Real-Time APR. A public source-bound normalization of LDUSDTUSD by
USDTUSD contains 505 aligned daily closes over 504 days. It returned 2.78729%
cumulative appreciation, 2.01091% compound annualized over the full horizon,
and a 2.74952% annualized latest-seven-day pace. Public exchange information
still marks LDUSDT margin-eligible. This is a validated public gross incremental
mechanism only for collateral already required by an independently justified
futures strategy. Official current guidance binds a 99.9% collateral value
ratio, the same current conversion ratio in both directions, accumulated reward
value on redemption, zero additional swap fees, and a 300,000 LDUSDT VIP-0
limit. The observed 2.01091% annualized increment remains above 2.00% even after
assigning a 10% annual alternative yield to only the extra principal required by
the 0.1% collateral-value haircut. This is now accepted only for eligible
LDUSDT already held as already-required margin, with the haircut fully budgeted
and no liquidation or auto-exchange. It is not a standalone strategy, reason to
open risk, or deployment-ready result. Canonical gate:
`docs/model-research/action-value/binance-ldusdt-margin-yield-gate-v1-2026-08-26.json`,
result SHA-256
`6c2b81a8067faac80efb56f586d89bc308cb69b4fae0ec8504adc3aa2f3ff49d`.
When both designated ephemeral credentials exist, freeze one GET-only account
prequalification. Only a positive gate plus separate explicit funded authority
permits one minimum-size conversion round trip without opening a futures
position.

The next conditional execution lead is Polymarket's post-observation
oracle-to-CLOB-close maker window. The retained Round 26 public capture contains
10 consecutive complete BTC 5-minute conditions. In every row,
the exact closing Chainlink TWAP direction matched the later resolution,
aggregate winner bids grew only after local receipt of that observation, and a
later public winner-token seller fill was recorded. Both Up and Down rows
contributed 13.40488 pUSD of gross `size * (1 - price)` in total. This is not an
accepted edge: aggregate events do not prove a fresh authenticated order was
accepted after observation, public fills do not prove owned lineage or queue
position, the evidence is one degraded BTC hour, and fees, rebates, settlement,
redemption, relayer, opportunity, and pUSD-availability costs remain unbound.
Canonical gate:
`docs/model-research/action-value/polymarket-post-observation-maker-window-gate-v1-2026-08-26.json`,
result SHA-256
`03dcb88790b96bcaed6a58dc921abff5244e3b2eecd3a39e8f4e82c412f49392`.
A clean 660-second prospective capture then retained 578,480 raw messages,
578,430 normalized events, 12 conditions, and zero gaps or integrity errors.
The one fully observed interval for each asset showed winning-side 0.99/0.999
bid growth after local receipt of the exact closing TWAP in 3/3 conditions.
Only BTC had qualifying later winner-token seller fills: three events implied
0.01022 pUSD public gross; ETH and SOL had none during the retained 113.766
second post-close window. All three conditions resolved Up. This disproves a
strong current cross-asset fill-recurrence claim and is neither direction
balance nor regime persistence. Prospective artifact:
`docs/model-research/action-value/polymarket-post-observation-prospective-v2-2026-08-26.json`,
result SHA-256
`079925ec06eda0cdfc5851d71d7fc76df96de6f03883bcc70edc0f36da28d421`.
A frozen non-overlapping 1,380-second follow-up then completed with 1,227,321
raw messages, 1,227,214 normalized events, and zero stream gaps, recorder
errors, or integrity errors. It produced two complete intervals per asset with
both Up and Down represented. Winner high-bid growth recurred in 6/6, but the
qualifying later winner sell-fill fractions were only 1/2 BTC, 1/2 ETH, and 0/2
SOL versus the frozen 75% gate; SOL public gross was zero. The mechanism is a
terminal public recurrence failure and must not receive another unchanged
capture. An exact 11:45 UTC Chainlink TWAP boundary was absent for all assets,
which also exposed that fixed duration plus zero transport gaps does not prove
analyzable sample supply. Canonical result:
`docs/model-research/action-value/polymarket-post-observation-prospective-v3-2026-08-26.json`,
result SHA-256
`7b9f21cf3c1a65a709d5e52867877b9d79a9bf17f7a4df448a2fb92a32757e16`.
Only a material program-term or execution-architecture change may reopen it;
these artifacts grant no order, paper, account, funding, or live authority.

The 2026-08-25 source-first triage is canonical at
`docs/model-research/structural-edge-source-triage-v1-2026-08-25.json`, result
SHA-256 `509f63910c77a582680849e779317396962d06edeffa537e7d5ce8e18a984cb2`.
It repairs an omission in the terminal registry: Round 61 already rejected
matched-base elevated-funding spot/perpetual carry after capacity and after-cost
gates, so that family must not be rediscovered from its earlier funding-only
pass. It also preserves Binance WBETH/ETH and BNSOL/SOL conversion parity as a
distinct direction-neutral hypothesis, but every official Binance REST
conversion-rate and quota input reviewed is signed and both ephemeral credential
variables remain absent.
No collector, book sample, signed request, or edge claim was opened. Public LST
books alone cannot prove redeemable value.

The Binance delta-hedged BNB spot-fee-discount inventory mechanism is now
terminal. It holds BNB only to pay eligible spot commissions and shorts equal
base BNBUSDT perpetual inventory; fee consumption would still require hedge
rebalancing. The initial hosted-verified one-use screen made six public GETs,
used no credentials, and placed no orders. Its requested 1,000-row funding page
validly returned 500 rows, leaving only four complete inner months and exposing
a planning defect: the request budget had not been proved capable of supplying
the gate's six-month horizon. A separately committed, hosted-verified recovery
made exactly one older non-overlapping request, merged 1,000 settlements across
ten complete inner months, and retained the original symbol, scenarios, and
decision thresholds. Aggregate short funding was positive 164.9780 bps, but
the worst complete month was negative 35.6129 bps. Under the primary
non-authoritative 10-bps standard commission and 25% discount scenario, that
loss requires 14.24516 times monthly spot turnover to offset, so the unchanged
primary robustness gate failed. The terminal recovery is
`docs/model-research/action-value/binance-bnb-fee-discount-hedge-recovery-v1-2026-08-25.json`,
result SHA-256
`85d0be66391b53bef87dda33ea73acaf6995d0200e6423de7999d44a8fed3c8f`.
Do not request another page, resample books, loosen the gate, or treat this cost
reduction as a standalone edge. For every future one-use historical screen,
bind the endpoint's effective page capacity or preregister sufficient
non-overlapping pagination before activation; a requested `limit` is not proof
of returned horizon.

A materially distinct just-in-time BNB fee-buffer policy is now accepted as a
scoped incremental Binance cost edge, not a standalone strategy and not a
reopening of the terminal delta-hedged inventory. Binance's current official
guide states that eligible Spot standard commissions paid in BNB receive a 25%
discount without changing trading frequency or strategy. The public Convert
catalog currently permits USDT-to-BNB from `0.01` USDT, versus a `0.008` BNB
spot minimum costing `5.61864` USDT at the source-bound ask: about 562 times
less unhedged principal. If the minimum Convert buffer is fully consumed by an
imminent independently justified organic trade, a 100-bps acquisition-cost
stress still yields `0.0032` USDT net savings at zero BNB move and breaks even
only after a `24.2424%` BNB decline between acquisition and fee deduction.
Even at only 25% consumption with symmetric 100-bps entry and residual-exit
costs, the modeled adverse-move threshold is `6.1099%`; absolute principal at
risk remains capped at `0.01` USDT for the first reconciliation. This does not
make BNB price risk disappear. Exact account and symbol eligibility, positive
standard commission, exact-order commission, executable Convert quote cost,
minimum consumption, holding interval, residual, and owned fee deduction remain
fail-closed gates. Both designated credentials are absent, so no signed request,
quote acceptance, trade, or funded action occurred. Canonical gate:
`docs/model-research/action-value/binance-bnb-just-in-time-fee-buffer-gate-v1-2026-08-26.json`,
result SHA-256
`b97eed6a93070d5e29b26d1a47757c9be49e0296332c8019a64388ba936c3b6b`.
Never manufacture or enlarge volume, use the discount to rescue an unprofitable
trade, keep standing BNB inventory, or call this risk-free. The reverse public
catalog also requires at least `0.000014` BNB, so a partial minimum-buffer
residual cannot be assumed independently unwindable. Once both credentials
exist, freeze one exact-symbol commission and exact-order test contract; only
after it passes and separate funded authority exists may one `0.01` USDT maximum
quote and immediate fully consuming organic fee deduction be reconciled.

A separate current Binance promotional-fee overlay is accepted for exact
quote-native BTC/ETH/SOL Spot flow that was independently justified before the
discount. The dynamic official fee page lists `0%` maker fees on BTC, ETH, and
SOL against both FDUSD and U. Pre-existing VIP2-9 accounts also receive `0%`
taker fees on the three U pairs and `0%` maker/taker fees on the three USD1
pairs. The current USDC promotion explicitly includes BTC/USDC, ETH/USDC, and
SOL/USDC; a regular taker pays `0.09500%` rather than `0.100%`, or `0.071250%`
rather than `0.07500%` when independently eligible to pay fees in BNB. Thus the
regular-user direct saving is 10 or 7.5 bps for an actual promotional maker fill
and 0.5 or 0.375 bps for an actual USDC taker fill. Never acquire or retain a
promotional quote asset, chase VIP status, change execution role, or create
volume for this overlay. A quote switch requires separate executable spread,
basis, conversion, fill, settlement, and opportunity-cost proof; a zero fee may
not be double-counted with BNB, rebates, or rewards. Both credentials remain
absent, so exact account/region eligibility and exact-order commission evidence
are unproved; no signed or funded action occurred. Canonical gate:
`docs/model-research/action-value/binance-spot-promotional-fee-overlay-v1-2026-08-26.json`,
result SHA-256
`f951d167b3abbb89afc39a29671b9a4cb6929661f13a957e553a8fad439ce9e6`.

The current official fee surface also establishes an accepted scoped TradFi
perpetual fee overlay on independently justified organic flow. All displayed
regular/VIP levels currently show `0.0000%` maker fees, while taker fees range
from `0.0400%` for regular/VIP1 to `0.0085%` for VIP9 before the separate BNB
discount. Against the same-session standard USD-M USDT table, the positive
displayed comparator savings span `18` to `200` USD per `1,000,000` USD of
notional. This is not standalone profit or permission to trade. Refresh the
exact current table before each otherwise authorized order, apply the rate only
to the exact current account and symbol, and credit zero maker cost only after
an owned maker fill. Never change price, role, or volume to chase the rate; the
underlying strategy must independently clear spread, slippage, queue, adverse
selection, latency, partial-fill, funding, basis, liquidation, opportunity,
tax, and operating costs. The public table exposes no end date, so it cannot be
carried forward without refresh. Canonical edge:
`docs/model-research/action-value/binance-tradfi-perpetual-current-fee-overlay-edge-v1-2026-08-27.json`,
result SHA-256
`705cb3da615c1873623e7f5be31f0d8cf672c3db9635a5ba971407cf6e715b6c`.

A materially distinct current fee overlay is Binance's `VIP 6 for Six`
promotion. The Growth Track requires an actively trading current Binance
VIP1-5 account plus verifiable VIP3+ status on another exchange; the
Reactivation Track requires Binance VIP3-6 history in 2025 and may be applied
automatically by email. The current public tables imply `15` to `710` USD of
gross savings per `1,000,000` USD of already-intended fee-bearing flow across
the retained Spot and USD-M tier, role, and BNB-discount examples. The highest
example is a standard-fee VIP1 Spot taker moving from `0.100%` to `0.029%`; the
lowest is a BNB-discounted VIP5 Spot taker moving from `0.02325%` to `0.02175%`.
This remains an unaccepted account-gated candidate with a zero public forward
floor: account track eligibility, selection, approval, exact effective
interval, exact commissions, future organic volume, and incremental costs are
unproved. The article's stated Phase 1 ends `2026-10-16`; do not assume a late
approval receives two full months beyond that date. No application, account
manager contact, external-record disclosure, borrowing, trade, or account
request occurred. Canonical candidate:
`docs/model-research/action-value/binance-vip6-for-six-organic-fee-overlay-candidate-v1-2026-08-27.json`,
result SHA-256
`f638cb6f565c1ee18c9dc065c5f4fc6506442f00833193d23c287bdf9d8ec74d`.
Only with explicit read-only account evidence authority may current tier,
commission, 2025 history, and an already-applied upgrade be checked. Applying,
contacting, disclosing external VIP records, borrowing BNB, or trading each
requires separate explicit authority; never manufacture volume or value the
unbounded loan/Earn claims.

A distinct Binance BNB reward-stack candidate now reuses that frozen hedge
history without reopening the fee-discount-only screen. The current seven-day
BNB Simple Earn offer advertises 0.35% APR plus applicable Launchpool, Megadrop,
and HODLer rewards. The advertised base rate supplies only 0.67123 bps over
seven days, while the worst rolling 21-payment window in the frozen 1,000-row
BNBUSDT short history cost 18.4917 bps. Realized account airdrops must therefore
clear at least 17.82047 bps before commissions, spread, basis, collateral,
liquidation protection, tax, custody, opportunity cost, and alternative yield.
Public token allocations cannot establish per-BNB yield because the eligible
denominator and executable sale proceeds are unknown. The required signed
Simple Earn position/reward and all-asset dividend endpoints are source-bound,
but both designated ephemeral credentials are absent; zero signed requests were
made. Canonical gate:
`docs/model-research/action-value/binance-bnb-stacked-reward-hedge-evidence-gate-v1.json`,
result SHA-256
`0bfc615af743f4ba352201ff2f06e2abf0f0c8fec56b548a0e19791faf25f8ed`.
Do not acquire BNB or open a hedge to chase an unannounced retrospective
snapshot. When both credentials exist, capture only account principal, realized
rewards, conservative reward sale values, and exact costs against the frozen
hedge history; do not paginate funding or resample books.

A June 2026 paper reporting Polymarket BTC threshold overpricing relative to
Binance option-implied binary values is preserved as a distinct statistical
lead, not a locked payoff. Its pooled net-alpha result used only 16 proxy trades,
had `p=0.053`, and its HAC interval crossed zero. A complete current public
catalog join found 83 active Polymarket BTC point thresholds and 337 Binance BTC
calls, but zero exact same-strike same-expiry pairs. Twenty same-date/strike
pairs all retained an eight-hour terminal payoff mismatch; only 15 had two-sided
Binance option quotes. A contemporaneous reproduction produced a 1.2559-point
mean model midpoint wedge and a 2.7804-point maximum absolute wedge. Zero pairs
cleared the paper's own 4.27-point historical mean friction term. Canonical gate:
`docs/model-research/action-value/binance-polymarket-option-threshold-wedge-gate-v1-2026-08-26.json`,
result SHA-256
`22a99f25de487774ac4d22f4666a242fe3cb961e31f7f610de7a079cd6d9d7e7`.
Do not repeatedly resample. Retry only if an exact settlement identity appears
or a two-sided same-date model wedge first exceeds 4.27 points; then freeze one
prospective synchronized-depth, exact-fee, hedge-rebalancing, capacity, and
cross-regime study.

The source-first USD-M versus COIN-M perpetual funding hypothesis is also
terminal as a separate public edge family without consuming market data. The
official SDK exposes unauthenticated `/fapi/v1/fundingRate` and
`/dapi/v1/fundingRate`, while COIN-M exchange information separately exposes
`contractSize` and `marginAsset`. Those fields do not prove equal base delta, a
complete inverse terminal payoff, collateral neutrality, funding conversion,
or a liquidation-safe transfer path. Hedging the mandatory coin collateral
with spot-equivalent or linear perpetual exposure otherwise reintroduces the
already-terminal spot/perpetual carry family. Account fees remain signed: the
official Python connector uses `sign_request` for both commission endpoints,
despite generated Go AccountAPI Markdown incorrectly printing
`No authorization required`. The stricter classification controls. With both
designated credentials absent, zero signed or public venue requests were made.
Canonical source triage:
`docs/model-research/structural-edge-source-triage-v2-2026-08-25.json`, result
SHA-256 `3df17e93866cbf53617340dd422a91945c8a1924d4ca736b76c5f78f4c9a5575`.
Do not build a funding-history or book collector unless materially new official
payoff and collateral semantics first prove a distinct identity and the exact
account evidence trigger is satisfied.

Polymarket taker-tier rebates are cost reductions for legitimate organic taker
flow, not authority to manufacture a complete-set volume loop. Do not self-match,
wash trade, create inauthentic volume, ignore fee precision and market-specific
minimums, or treat a one-time level-up bonus as persistent edge. The current
August crypto TWAP liquidity-reward caps are materially relevant, but the one
public 15-minute/4-hour exact-identity screen is now terminal after its first
BTC reward response failed the condition-identity gate before books. Do not
retry that window. Reopen only after a material exact endpoint, configuration,
or program change, with source journaling active before every validation. Any
authenticated reward, order, cancellation, or fill study still requires
explicit account plus paper or funding authority.
The later exact 550,000 dollar five-minute program allocation was such a
material change, but its one source-bound attempt also terminated before books:
all seven exact raw Gamma rows omitted the optional `clobRewards` allocation.
The distinct complete official current-rewards list then returned 54 terminal-
cursor sponsored rows with zero of those seven exact condition IDs, and also
stopped before books. Both public source configurations are terminal. Reopen
only when a genuinely new public response supplies an exact per-market dated
daily allocation or the official program allocation changes again; never infer
a daily rate by dividing the monthly cap.

The 2026-08-30 official rewards-page discovery signal did not justify a new
economic screen by itself. A frozen one-use source-contract delta attempted the
current official v2 client's default `/rewards/markets/current` method, which
sends `next_cursor` but no `sponsored=true` parameter. Every one of the four
allowed pages returned its maximum 500 rows, and the fourth response still had
the nonterminal offset-2,000 cursor. The retained 2,000 rows are complete only
for the consumed prefix, not for the current rewards population.

That default response also cannot be compared as a population delta with the
prior complete 54-row `sponsored=true` baseline: only four baseline condition
IDs appeared in the retained prefix, all four row representations differed,
and 1,996 prefix identities were not in the filtered baseline. An omitted
parameter in a newer client method is not proof of aggregation equivalence.
The run therefore stopped before Gamma metadata, books, fees, accounts, or
economic selection; the public payout floor remains zero. Do not follow the
cursor, retry another page budget, or select a partial-page winner. Reopen only
after a current primary source explicitly defines bounded comparable filter and
aggregation semantics, or source-selects a distinct exact per-market dated
endpoint before access. Canonical contract and terminal result SHA-256 values:
`c6aef4b1395e6a0c311a6a64b4e679d6dafbd6ac3e7b50860585e230ca0c9484` and
`0a885b96eadbc0109ec3da70a8670680a23813a5f492d61abf89e49f3e892481`.

A genuinely distinct exact-market retry was then source-selected for the active
Elon Musk 40-64 posts market. The first frozen attempt correctly retained exact
Gamma and `sponsored=true` reward responses but incorrectly treated a stale or
misattributed 104 pUSD/20-share/4-cent discovery tuple as an economic equality
gate. The exact sources actually agreed on 50 shares and 5.5 cents; the exact
reward rate was 53 pUSD/day. A separately frozen correction reused those bytes
without refetching and made the one previously unrequested book call. It failed
freshness at 6,408 ms, while one-tick 0.48 and 0.53 bids were both marketable,
summed to 1.01, and lost 0.50 pUSD if both filled before reward uncertainty.
A separately frozen zero-network best-bid join earned 0.50 pUSD both-fill gross
but risked 26 pUSD orphan loss. Even observed displayed competition required
43.554 reward days versus 3.693 remaining; 100-times competition required
4,306.850 days. Do not retry or justify a fresh capture. Canonical rejection:
`facecfaa3b92d905c700083c7b8afe153adc495403ceabc91e417bdb248d059b`.
The accepted-edge count remains 21, the ranked-hypothesis count remains 44, and
the registry SHA-256 is
`de28d80cc4b0b9cd1bd3f9954cb840dcaefe46fcc0fdf9ac3fd53218169370cb`.

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
parsing; locale-dependent implicit date parsing is prohibited. Artifact creation
and update times must come from the host clock or a captured source epoch, never
from an estimated narrative time, and must not be later than the committing
checkpoint. Every Polymarket recorder preregistration manifest must include its
`manifest_sha256`, computed over the canonical body before insertion. Validate
that body before opening the target evidence database when practical. If
validation has already created a database and then fails, preserve it and use a
new filename; never overwrite the failed attempt. Similar quarter
or expiry labels across Binance products do not establish a shared settlement
timestamp or value. The historical futures settlement endpoint exposes a
00:00 `deliveryTime` date marker while the current futures catalog and option
catalog align relevant expiries at 08:00. This does not prove settlement-value
equivalence and does not authorize adding eight hours to historical fields. A
cross-product parity identity may proceed only through the frozen calendar-date
comparison and then exact current rule and numeric-epoch binding. For
Polymarket rewards, the corrected
instantaneous denominator is old aggregate `Q1 + Q2`, never the minimum of the
two aggregates. Public books do not reveal per-maker grouping, queue position,
sampling persistence, or final reward allocation; without that evidence the
provable payout lower bound is zero. A physical binary BUY order also appears
as a complementary ASK at `1-price`; hypothetical post-quote midpoints must
include that mirrored own ask while the physical order is scored only once.

The active Binance USD1/WLFI holding airdrop is now accepted only as a scoped,
time-limited gross holding-yield edge for USD1 that is already held in an
eligible published account category. The official extension reports 4.85% then
5.46% then 5.27% realized base APR and 5.82%, 6.55%, then 6.33% boosted APR
through the first three weekly distributions. The boost requires the lowest hourly USD1 Futures open
interest to remain at least 300 USD1 each day; never open or retain a position
to farm it because the current boost adds only 2.0904 bips over seven days
before funding, spread, basis, liquidation, margin, and operational costs.
Simple Earn is absent from the announcement's closed list of eligible account
categories, so never credit the airdrop and Simple Earn to the same principal.
From 2026-08-27, the logged-out public Simple Earn page displays `8.62% Max` on
the first 1,500 USD1, decomposed as a variable `1.62%` Real-Time APR plus a
fixed `7%` bonus. The fixed bonus alone exceeds the current 5.46% airdrop base
by 154 annualized bps; after one forfeited airdrop day its gross break-even is
3.5455 days. For a 1,500 USD1 balance, immediate fixed-bonus allocation beats
the sensitivity of waiting through the airdrop and subscribing afterward by
only 0.5063 USD1, or 3.3753 bps. This mutually exclusive allocation is now
accepted only for independently already-held idle USD1, after exact account,
capacity, liquidity, transition-cost, tax, and redemption-timing proof. Never
credit both rewards to the same principal, acquire or retain USD1 for this
bonus, or use the displayed USD1/U/USDC/USDT rate ordering to justify a
conversion. These are public sensitivities, not account instructions:
credentials are absent, region and product eligibility are unproved, USD1
principal risk remains, and exact reward-sale costs are unknown. Refresh the
final airdrop once on 2026-09-04; the 2026-08-28 refresh is consumed and must
not be repeated.
Canonical holding-airdrop gate:
`docs/model-research/action-value/binance-usd1-wlfi-holding-airdrop-gate-v1-2026-08-26.json`,
result SHA-256
`c67367932b440d6f4a23330a17c405c0e15b0021b0484575a0b0efcc6e9238a6`.
Canonical allocation adjudication:
`docs/model-research/action-value/binance-usd1-simple-earn-versus-holding-airdrop-allocation-edge-v1-2026-08-27.json`,
result SHA-256
`a4158bf059f4f5ad839b2f504c08c4afc65615260b4171533866f4c2337494e0`.
The scheduled public distribution refresh is
`docs/model-research/action-value/binance-scheduled-yield-distribution-refresh-v1-2026-08-29.json`,
result SHA-256
`c5feb852830adadd497aa287460d1a3132e324fbbbdaa5f608890acebc43e252`.
At the latest 5.27% base, the mutually exclusive fixed 7% Simple Earn bonus has
173 annualized bps of gross uplift, a 3.0463-day break-even after one forfeited
airdrop day, and only 0.4977 USD1 same-week excess at the 1,500 USD1 cap before
unproved account and transition costs. The RLUSD second completed-week APR fell
from 8.07% to 5.78%; its next-week public reward floor remains zero.

A zero-network remaining-horizon correction now prevents the activation-day
28-day USD1-versus-USDT stress margin from being treated as current. Using the
unchanged source-bound 7% USD1 fixed bonus, 3% approximate variable USDT
alternative, 4% USDT fixed bonus on 500 through September 7, 23.9544864757-bip
worst retained 30-day USD1 close decline, and 0.1000095009-bip displayed round-
trip spread, a new 2026-08-30 allocation has 25.5707762557 bips of incremental
reward and only 1.5162802791 bips after the frozen stress. That is just
0.2274420419 quote units at the 1,500 cap before every unproved cost. The same
case has only 0.0550930645 bips for a September 1 subscription and becomes
negative on September 2 before any unproved account, conversion, redemption,
issuer, tax, custody, or operating cost. This is not a public profit floor.
Do not roll it daily or refresh books merely because another accrual day passed.

The current first-USD-deposit Promotion A is a distinct high-margin public
candidate, not an accepted or recurring edge. It offers a 15 USD-equivalent
SPCXB voucher to the first 1,000 registered genuinely first-time eligible users
who deposit at least 100 USD and complete at least 200 USD of eligible trade
volume, with distribution within 30 hours after task completion. The frozen
one-way 201-USDT task route cost 200.93166 USD at the displayed ask. The
theoretical voucher quantity had 15.6135 USDT of displayed liquidation value;
after labeled 10-bp task and reward-sale fees, 20-bp round-trip SPCX hedge fees,
and four funding intervals at the worst short-pay rate in the latest 20 rows,
15.3490 USDT equivalent remained for every deposit, bank, FX, withdrawal, tax,
basis, and operating cost.

The public forward floor is still zero. Exact first-time and regional
eligibility, first-come capacity, account deposit fee, task accounting,
deterministic voucher entitlement, rounding, and future executable value are
unproved. SPCXB public metadata reports a 1.0 multiplier to SPCX, but that does
not prove hedge basis convergence or delivery timing. Never open the hedge
before account-confirmed task completion and deterministic entitlement because
a zero reward leaves a naked short; never churn or round-trip solely to create
volume. Registration, BPay activation, deposit, trade, hedge, claim, sale, and
withdrawal each require separate explicit authority. Canonical candidate:
`docs/model-research/action-value/binance-first-usd-deposit-spcxb-reward-hedge-candidate-v1-2026-08-27.json`,
result SHA-256
`e0b6ed9311d2a022abee417a677b952e83cf918fc6b396804f5cba39fd83d4ed`.

The current Binance U Flexible promotion is also accepted only as a scoped,
time-limited gross yield overlay for U that is already held idle in an eligible
non-EEA master account. Regular users are offered approximately 0.5% Real-Time
APR plus 8% Bonus APR on the first 5,000 U through 2026-09-14; VIP 1-9 users
are offered the same approximate base plus 5% on the first 500,000 U. For a
regular user subscribing before 2026-08-27, 19 eligible days produce 44.2466
bips gross. Buying U for the promotion is rejected: only 10.3422 bips remain
after the current USDT alternative, displayed zero-fee round-trip spread, and
the worst observed 19-day U close move, before every exact account and issuer
risk. United Stables' own terms say ordinary secondary holders have no direct
redemption right or claim on reserves; the public homepage lists only a
December 2025 attestation, and the live PoR page did not server-render numeric
reserve values. Never infer EEA status from the host timezone, buy U for the
headline APR, or treat issuer reserve language as insolvency protection.
Canonical gate:
`docs/model-research/action-value/binance-u-flexible-idle-holding-yield-gate-v1-2026-08-26.json`,
result SHA-256
`6f44b65e5aa85d33cc02e8611a372162cf00f4162fdff99828a31cf498ced6f9`.

The RWUSD VIP promotion extension is accepted only as an automatic incremental
bonus on an independently required existing RWUSD position. Over a conservative
22 forward days, VIP 1-3 adds 1.1452 bips on at most 30,000 RWUSD and VIP 4-9
adds 4.5808 bips on at most 200,000 RWUSD. A new RWUSD subscription is not
accepted: exact account rate, quota, alternative yield, USDC redemption fees
and delay, and product risks remain behind the existing signed six-GET gate.
Do not chase VIP status, enlarge RWUSD, or treat RWUSD as a stablecoin,
transferable token, direct RWA claim, or guaranteed instant redemption.
Canonical gate:
`docs/model-research/action-value/binance-rwusd-existing-vip-bonus-overlay-gate-v1-2026-08-26.json`,
result SHA-256
`076f428ea9bccc0dc9c1a0c605ac469db27fedb7941ac6728260cf98da667e51`.

Current USDT Flexible bonuses are accepted as one scoped, time-limited
same-asset gross-yield family only for independently held idle USDT in an
eligible verified master account. The global offer adds 4% Bonus APR to the
first 500 USDT through 2026-09-07; a subscription before 2026-08-27 has at most
12 bonus-accrual days, producing 13.1507 bips or 0.6575 USDT. Separate new-user
offers add 30% for seven days on the first 200 USDT for a closed published LATAM
country list, or 15% for Brazil; those produce 57.5342 and 28.7671 bips on capped
principal. Do not assume bonuses stack, infer region or new-user status, or
register, deposit, acquire USDT, or move operational principal to chase them.
Exact account eligibility, product capacity, variable Real-Time APR, owned
distributions, and redemption behavior remain unproved; credentials are absent
and no subscription or account request was made. Canonical gate:
`docs/model-research/action-value/binance-usdt-flexible-current-bonus-overlay-v1-2026-08-26.json`.
Result SHA-256
`0126a1feef4e8bb5c46a7b7cab45e2471857a2e105fe0f41d73d4710b6abceda`.

The current Binance USDe automatic holding reward is accepted only as a scoped
same-token gross increment for eligible USDe already independently held on
Binance for at least 24 hours. The current reference APR is 4%; that equals
1.09589041096 bips per eligible day and 0.7671232877 USDe per 1,000 over seven
days. Official terms make the campaign ongoing until further notice, calculate
rewards from a random daily minimum eligible-balance snapshot, and distribute
USDe automatically each Monday. The product page and legal terms disagree on
the exact Monday payout hour, so no exact hour is claimed. Never acquire,
deposit, convert, borrow, or retain USDe for the reward; treat USDe as fiat or
principal protected; or double-count collateral use or separate Ethena yield.
Exact KYC, jurisdiction, eligible-account balance, completed holding duration,
owned reward rows, and weekly distributions remain unproved. Canonical gate:
`docs/model-research/action-value/binance-usde-existing-holding-reward-edge-v1-2026-08-26.json`,
result SHA-256
`4640635514ad43ed846660c204a95c0d59ed75ac3ccbf5f17a0b70f3d5726f6a`.

Polymarket Builder Fees are a distinct current direction-independent gross
revenue overlay, not trading alpha. The official formula is matched pUSD
notional times the configured builder fee rate in bps divided by 10,000, with
current maxima of 100 bps taker and 50 bps maker. Accept it only for bona fide
independently existing third-party matched flow through an owned application,
an account-confirmed active positive fee setting, and explicit disclosure and
consent. Never use operator, related-party, self-referred, circular, wash, or
manufactured orders or create and reroute volume for the fee. Account tier,
active rates, external flow, owned pUSD payout, demand effects, legal duties,
and all operating costs are unproved, so this is not deployment-ready or an
after-cost profit claim. Canonical overlay:
`docs/model-research/action-value/polymarket-organic-third-party-builder-fee-overlay-v1-2026-08-26.json`,
result SHA-256
`8c070b6a4b07070ffdd5ba703da1ca3788faffcb4d748633a18269dc02c17885`.

Do not promote BFUSD existing-holding yield. The live product page says the
daily reward is BFUSD, while the current rate FAQ and effective 2026-01-05
governing terms say a USD stablecoin and permit the reward asset to change.
That unresolved primary-source conflict makes the same-unit forward reward floor
zero; never assume BFUSD, USDT, USDC, or another stablecoin are one-for-one.
Canonical gate:
`docs/model-research/action-value/binance-bfusd-existing-holding-reward-unit-conflict-gate-v1-2026-08-26.json`,
result SHA-256
`54fe3d3e23a92290debdc67d1e7e19ecac6c06441c045f1aa21fe3e62558c03c`.

Binance Smart Arbitrage is terminal as an operational wrapper around the same
matched-base spot-perpetual funding carry rejected in Round 61. Official terms
still charge spot-maker and futures-taker entry and exit fees and retain spread,
basis, funding reversal, manual exit, and liquidation risk. No distinct fee,
execution, or capital subsidy was found; do not resample ordinary carry, use
trailing three-day funding as a forward edge, or call delta-neutral risk-free.
Canonical adjudication:
`docs/model-research/action-value/binance-smart-arbitrage-terminal-family-adjudication-v1-2026-08-26.json`,
result SHA-256
`03b652fcd7e50c0671abbfb73f68f69509a2e5d7f75d8166f6b74743eab630d3`.

Polymarket's current Referral Program is accepted only as a gross pUSD overlay
for authentic independently acquired external users when the referrer already
independently exceeds 10,000 USD lifetime Polymarket volume. Direct referrals
pay 10% and indirect referrals 5% of net fees after the referred user's own tier
rebate, daily until the earlier of Platinum or 30 days after signup. Never self-
refer, use controlled or linked accounts, manufacture qualifying or fee volume,
or double-count the same fee base with Builder Fees or another reward without
explicit combination terms. Omnibus third-party integrations are ineligible.
Account eligibility, attribution, owned payouts, acquisition costs, compliance,
tax, and operating costs are unproved, so gross reward is not after-cost profit.
Canonical overlay:
`docs/model-research/action-value/polymarket-organic-referral-net-fee-overlay-v1-2026-08-26.json`,
result SHA-256
`f7aec4a5340cba42abb120a43cda1ed1fa4d5b03632b3c062c0d00d7b5636cf0`.

Binance Flexible Loan collateral-yield retention is a distinct candidate, not
an accepted edge. Current official sources say Simple Earn Flexible collateral
continues earning while pledged to an isolated overcollateralized open-term
loan. A direct market-direction-independent same-asset loop is not publicly
available: the live official page explicitly says same-cryptocurrency
collateral-loan pairs are unsupported, and the newer 2026 overview does not
reverse that rule. Two reciprocal cross-asset isolated positions are not the
same mechanism and retain independent path-dependent liquidation risk. Exact
current collateral and loan eligibility, reward and interest rates, LTV
thresholds, positions, income, and owned rewards are all signed USER_DATA. The
designated ephemeral credentials are absent, so no signed request was sent and
the public after-cost floor is zero. Never borrow, subscribe, repay, adjust LTV,
acquire collateral, reinvest loan proceeds, or double-count idle yield.
Canonical gate and same-asset adjudication:
`docs/model-research/action-value/binance-flexible-loan-simple-earn-collateral-yield-gate-v1-2026-08-26.json`,
result SHA-256 `ac010265c5236152907ac7b3c12ce13104f473b4cc61c5db43fb8b28c6678182`;
`docs/model-research/action-value/binance-flexible-loan-same-asset-loop-adjudication-v1-2026-08-30.json`,
result SHA-256
`7106bb072533327d3154c773ba2c1969ff7891df6ec16d314f2d8f1b410f48e6`.

Binance Advanced Earn Discount Buy and Dual Investment are terminal for this
market-situation-independent search. Discount Buy is a locked non-principal-
protected structured product whose settlement can convert 50% or 100% of the
stablecoin principal into crypto at the target price. Dual Investment Buy Low
and Sell High likewise embed cash-secured-put-like downside or covered-call-like
foregone upside. Their nominal APR is compensation within a direction-dependent
conditional-conversion payoff, not a positive worst-state floor. No account
quote or subscription was requested. Do not run APR, strike, or duration grids
without a materially new principal subsidy or complete executable option-
equivalent mispricing. Canonical adjudication:
`docs/model-research/action-value/binance-advanced-earn-conditional-conversion-terminal-adjudication-v1-2026-08-26.json`,
result SHA-256
`15f160e3d54f0be09611bb36901b1d9061a2a173643c0562996ecb2824320a3f`.

Binance Square Write to Earn is accepted only as a direction-independent gross
USDC overlay at the current base 20% rate on authentic external readers'
independently existing eligible fee-bearing trades after engagement with
genuinely useful attributed content. Do not credit conditional 30% or 50%
leaderboard totals, self or zero-fee trades, content older than seven days,
unattributed activity, or weekly earnings below the 0.1 USDC payout threshold.
Never manufacture reader activity or encourage unnecessary, unsuitable,
leveraged, or loss-making trades for commission. Account KYC and region,
attribution, owned USDC payout, audience demand, content production, compliance,
tax, and operating costs remain unproved, so the gross commission is not net
profit. Canonical overlay:
`docs/model-research/action-value/binance-square-organic-write-to-earn-fee-overlay-v1-2026-08-26.json`,
result SHA-256
`29ec95146998535fde295dfc830a2639b9d10964e7f9e36c17e44e628dc454d1`.

Binance Referral Pro is accepted only as a direction-independent gross fee
overlay at the public base tier: 20% of authentic referred users' fee-bearing
Spot and Margin fees and 10% of their Futures fees for one year after Futures
activation. Referral Lite and Pro are mutually exclusive for the same new user.
Do not credit higher performance tiers, self or controlled accounts, zero-fee
or invalidated trades, restricted regions, prohibited advertising, or the same
fee under Write to Earn, Affiliate, Broker, or another commission program.
Account eligibility, attribution, payout asset and timing, acquisition,
disclosure, compliance, tax, and operating costs remain unproved, so gross
commission is not net profit. Canonical overlay:
`docs/model-research/action-value/binance-organic-referral-pro-fee-overlay-v1-2026-08-26.json`,
result SHA-256
`8a29116879fd90cb0f8fc11d9780a8dccbff8afc2d3ea685e671921f651e64d1`.

Polymarket Perps referrals are a separate accepted direction-independent gross
fee overlay: 20% of authentic external referred traders' Perps fees, paid
weekly, only while the account has a confirmed available invite. Never use the
operator's or referred users' trading volume to unlock the 100, 250, or 500
invite tiers; never self-refer, request trades, or double-count the separate
prediction-market referral program, Builder Fees, or another commission.
Account code, available invites, attribution, exact fee and payout asset, owned
weekly payout, acquisition, compliance, tax, and operating costs remain
unproved, so gross fee share is not net profit. Canonical overlay:
`docs/model-research/action-value/polymarket-perps-organic-referral-fee-overlay-v1-2026-08-26.json`,
result SHA-256
`4bebea610dc9406d598627035f4e6e815e6a4daeb64944d7ba2ec9f55b6b7d71`.

A new primary study materially reopens only the distinct Polymarket live NBA
full-game moneyline/spread implication family. Across 173 games from 2026-02-04
through 2026-03-04 it reports 290 active episodes, a 16-second median duration,
101.01-bps median yield, and average executable size of about 14.79 shares when
the $100 budget was constrained. Those are historical leads, not current
profit: the paper explicitly assumed zero NBA fees, forward-filled asynchronous
books sampled every 3.6 to 5.5 seconds, and observed zero middle payouts.
Current official terms list a `0.05` sports taker rate and say exact fee schedules
must be read per market. The current contract therefore assigns zero value to
the middle, rejects unmapped integer-handicap push or overtime states, and waits
for future active NBA full-game events before one synchronized public all-taker
after-fee recurrence capture. It does not reopen the terminal threshold/deadline
or single-market complete-set families. Canonical contract:
`docs/model-research/action-value/polymarket-live-nba-moneyline-spread-combinatorial-parity-reopen-v1-2026-08-26.json`,
result SHA-256
`bb2d030b9465ded6cc4ce0ba894719d60ecd812d673432a10941db1779d0d758`.

A separate primary paper materially reopens exact dependent-subset parity
across two multi-outcome Polymarket markets, but its headline must not be
misstated. The four numerically enumerated cross-market pair profits sum to
`95,156.71` USD; `39,587,585.02` USD covers all reported strategies and is
dominated by single-condition and within-market activity. Its analysis used
executed-trade VWAPs, up to 5000 blocks or about 2.5 hours of forward-fill, a
historical no-fee regime, and semantic dependency discovery. The text also says
five cross-market extraction cases but enumerates only four numeric pairs. A
New York dependency justified by historical voting behavior is expressly
rejected as non-deterministic. The candidate therefore admits only a complete
machine-checked joint payoff truth table proving subset-indicator equality,
followed by one-batch all-leg asks, exact current fees, displayed common depth,
synchronization, and every external cost. It is unaccepted and must not double-
count the specialized NBA family. Canonical contract:
`docs/model-research/action-value/polymarket-cross-market-dependent-subset-parity-reopen-v1-2026-08-26.json`,
result SHA-256
`0838bea50b70a8d9e102f40146b2ddf041bc06db3039736d312b9f309c72fc6d`.

A one-use current-sports discovery then tested the exact Polymarket event title
`Colorado Rockies vs. Washington Nationals`, which the public spread page
displayed as a current popular market. The frozen public Gamma keyset query
returned HTTP 200 with zero events, no cursor, and no partial population. It
made no price, book, account, or authenticated request. This terminalizes only
that displayed title lead; do not adapt the title or rerun it. The broader
cross-market exact-subset family remains open only when a future active pair has
complete rules and a machine-proved payoff implication. Contract SHA-256:
`99559dd57d8ba1520fd4f607c4e4e56cea1070a2798536941af10134e4376aed`;
result SHA-256:
`e5ce48b6b0521a5ba2fe58ae17316e703ab2155934a126e603eeadf81e219d9c`.

A distinct current official MLB page then exposed the postponed Boston Red Sox
versus New York Yankees June 6 event with an August 29 countdown. One exact
public slug request proved that the event remained active and open with all 16
embedded markets accepting orders: moneyline, NRFI, seven spreads, and seven
totals. The retained rules machine-prove 37 monotone subset relations across
the moneyline, team-margin ladders, and total-run ladder. For subset `A` inside
superset `B`, buying `B` plus the complement of `A` pays at least one pUSD per
share in `A`, `B`-only, outside-`B`, and both-market cancellation states.

The frozen one-request public CLOB batch retained all 30 exact books. Every ask
array was strictly descending even though the current API reference describes
ascending asks, so the frozen runner failed before economics. The runner,
contract, journal, and raw response remain unchanged. A no-refetch offline
adjudication reversed only the fully audited retained arrays and is explicitly
outcome-aware and promotion-ineligible. All 37 five-share packages were
negative after each market's current `0.03`, exponent-one taker fee, displayed
depth, and two adverse ticks per leg. The best, Yankees margin at least four
versus at least five, still lost `0.53262` pUSD against a five-pUSD guaranteed
floor. Do not refetch or retry this event. Reopen only for a distinct future
active complete-rules pair, with both strict book-order directions accepted in
the contract before access. Inventory result SHA-256:
`e274e3b05227022eb8c021fecdfa1a42e369ba30175ba906b24d6fd8459da80d`;
batch contract SHA-256:
`231ebf5e9078bc14c8acd3d8274bc98c0200776621b830b64c689a34cdd204b8`;
adjudication SHA-256:
`1e75e049abb116955294d878830f940491fe4044f09c7e3564ad2761c0129178`.

Binance Launchpool is a distinct direction-independent candidate only for an
independently already-held idle supported stablecoin. Current official guidance
describes USDC/FDUSD-style pools, hourly accrual, early unlock with accrued
rewards retained, and principal returned to Spot. The latest concrete 2026
example used USDC, U, and USD1 for a two-day OPN campaign beginning March 3;
it is historical, and the retained source does not state its exact end timestamp.
The current Launchpool page returned WAF-empty HTTP 202, so no active project,
account eligibility, allocation, APY, owned reward, or executable reward-token
sale value is proved. The public forward floor is zero. Do not acquire, swap,
borrow, retain, or redirect principal; do not assume stablecoin parity or value
an allocation before owned distribution and an executable sale. Wait for a new
official campaign announcement. Canonical candidate:
`docs/model-research/action-value/binance-stablecoin-launchpool-idle-inventory-reward-candidate-v1-2026-08-26.json`,
result SHA-256
`f898914a56fe61c063ca0eaf8d02fc91ea8bf527dd3ff49289527db524d286c3`.

Polymarket's Positions Framework exposes a materially distinct exact Boolean
parity candidate between underlying CLOB outcomes and Combo RFQ positions. For
all terminal redeemable values `A,B` in `[0,1]`, including `0.5` cancellation
payouts, `A+B = (A*B) + (1-((1-A)*(1-B)))`: two underlying outcomes exactly
replicate `YES(A and B)` plus `NO(not A and not B)`. The retained public Combo
catalog boundary contains 500 unique volume-descending markets and current
Combo-enabled same-game WNBA legs, but its non-null cursor proves incomplete
coverage. Catalog prices are not Combo quotes. No approved-builder credentials,
authenticated quote, CLOB batch, account state, order, or after-cost recurrence
was obtained, so this is an unaccepted candidate with a zero public profit
floor. The earlier broad catalog population screen remains terminal; this new
candidate is a payoff identity and should advance only when approved-builder
access plus explicit quote-request-only authority permits minimum-size
nonaccepted BUY and SELL RFQs. Canonical candidate:
`docs/model-research/action-value/polymarket-combo-rfq-boolean-parity-candidate-v1-2026-08-27.json`,
result SHA-256
`08fb223f771c5793da944497f37f4067238e7fd2b40fa2427293dbf7b55c4116`.

The broad Polymarket sports Combo requester-overround idea is now terminal and
must not be confused with that exact Boolean parity candidate. An initial
discovery screen was invalid because the Combo positions endpoint's default
listing omitted redeemed winners; the frozen correction explicitly requested
`RESOLVED_WIN,RESOLVED_PARTIAL,RESOLVED_LOSS` and excluded every discovery
wallet before examining sports leaderboard ranks 251 through 1000. The unseen
validation retained 6,264 resolved Combo YES positions across 162 wallets and
77 first-entry UTC dates. Buyers lost only 12,641.550729 pUSD, or 0.2928%,
after 86,010.262565 pUSD of attributed buyer fees. Once those fees are removed
because they are not maker revenue, the opposite-side gross spread proxy lost
73,368.711836 pUSD before any seller fee, hedge, collateral, funding, Last
Look, latency, or operating cost. The PNL-ranked cohort, chronological training
role, chronological test role, wallet-cluster lower bound, and date-cluster
lower bound were all negative. Do not repeat leaderboard mining or infer a
house edge from requester losses; reopen only with a materially less selected
population or a direct maker quote ledger. Canonical result:
`docs/model-research/action-value/polymarket-combo-maker-overround-validation-v1-2026-08-27.json`,
result SHA-256
`416daf4d279e06a2353127e642d588a39ae85be0709c2d7498896c1d182847ee`.

Binance bStock dividend reinvestment versus stock TradFi-perpetual funding has
now been separated into a terminal direct family and a materially distinct
calendar-timing candidate. At the 2026-08-20 AMAT and MSFT bStock snapshot, the
special negative funding debits charged to shorts were respectively
`0.529999005708` and `0.909997601100` USDT per matched unit, matching the
declared gross dividends of `0.53` and `0.91` within three micro-USDT.
Since Binance reinvests only the net bStock dividend after deductions, direct
pre-adjustment long bStock plus short perpetual has dividend contribution
`N-D=-F<0` before basis, fees, slippage, and capital costs. Do not repeat that
same-day capture or treat closing and reopening the short as a free hedge.

GLW has a separate prospective ordering question: its Friday 2026-08-28
ex-date precedes Binance's Monday 2026-08-31T00:00Z bStock snapshot. The frozen
one-use pre-snapshot observation ran on 2026-08-29 after the conversion pause.
Its one public funding-history response retained eight rows from 2026-08-27T00Z
through 2026-08-29T08Z: all were `Regular`, none was negative, and none was
`Special`. The required exact `0.28`-matching adjustment gate therefore failed,
so the five-request conditional synchronized depth, funding, and filter batch
did not run. This result does not prove a later pre-snapshot adjustment cannot
occur; it consumes the preregistered observation and prohibits opportunistic
polling or a 2026 GLW book capture. After 2026-08-31T00:00Z, at most one newly
frozen terminal history reconciliation may establish mechanism timing only.
Any executable study must use a future independent weekend event under a new
prospective contract. Gross headroom remains diagnostic only, and the current
conservative net-distribution floor is zero.
GS has no distinct weekend gap and remains inside the rejected direct family.
Canonical candidate:
`docs/model-research/action-value/binance-bstock-dividend-perp-funding-timing-gap-candidate-v1-2026-08-27.json`,
result SHA-256
`c073b61271886a5add71c2578caa889dfb97b1245327ae746bd517a91e52530d`.
One-use observation result:
`docs/model-research/action-value/binance-glw-special-funding-trigger-result-v1-2026-08-29.json`,
result SHA-256
`823448f115ecf7fe3e7fe8862855f40dfd351ed041fce2aa94196d069c8d585a`.

NOK is a materially distinct non-US dividend exception. Nokia's primary source
states a `0.0462` USD gross NYSE amount, 2026-07-27 ex-date, 2026-07-28 record
date, and 2026-08-06 payment. Binance made the 2026-07-28T00:00Z NOKB snapshot
eligible for a net dividend multiplier adjustment. The NOKUSDT short paid only
`0.001941732` and `0.006923371` USDT per matched unit at the snapshot and eight
hours later, totaling `0.008865103`; the signed short cash flow from ex-date
through payment date was `-0.0053709158406413` USDT because ordinary later
funding partly offset those debits. The `0.037334897` gross upper headroom is
not profit. The exact net historical multiplier increment, update time,
entitlement, synchronized executable books, fees, basis, funding horizon, and
recurrence are unproved. The current on-chain and Binance multiplier agree at
`1.002349416320445721`, but current state cannot reconstruct historical state.
Do not poll the preliminary 2026-10-27 record date. Reopen only after a new
Nokia Board resolution gives an exact amount and Binance publishes a matching
NOKB announcement. Canonical candidate:
`docs/model-research/action-value/binance-nok-bstock-dividend-perpetual-underdebit-candidate-v1-2026-08-27.json`,
result SHA-256
`79118e0e9a32a17d0d79040746068b94e6ec545179958a29dc45f3b8771434bb`.

The next declared non-US recurrence comparator is already terminal without a
current Binance request. TSMC's official dividend pages set the prior
2026-06-11 ADR payment at 0.939325 USD gross and 0.742067 USD net of at-source
withholding, while the next ex-date is 2026-09-16 with an estimated 1.11 USD
gross ADR payment. Retained TSMUSDT funding contains exactly one `Special` row
at the prior ex-date: -0.00233910 at a 407.23 mark. The matched short debit was
0.952551693 USDT, leaving -0.013226693 versus the gross dividend and
-0.210484693 versus the net dividend before every ordinary funding, execution,
basis, fee, tax, and capital cost. No current funding or books were requested;
do not repeat the September TSM event. Canonical result:
`docs/model-research/action-value/binance-tsm-bstock-dividend-underdebit-v1-2026-08-30.json`,
result SHA-256
`82acc3529620f1d9c728eac24ea0fb256f228e4065650c766dff057d198a5e60`.

A separate exact three-wrapper screen normalized current Ondo tokenized-stock
point values by `sharesMultiplier` and matched them to bStock Spot and stock
perpetual executable top-of-book symbols. Sixty exact tickers overlapped, 41
had positive point gaps, and ten were at least 10 bps; AXTI led at `22.2047`
bps. This is a candidate population, not an executable spread. Official
Binance source says `referencePrice = tokenInfo.price / sharesMultiplier` and
warns that the on-chain token and stock feeds update at different frequencies.
No executable Ondo ask, fee, gas, slippage, transfer, settlement, short-funding,
exit, or atomicity evidence exists. The nonexecuting Agentic Wallet quote client
is not installed, and no quote was requested. Binance Alpha subsequently
supplied a documented public unauthenticated full-depth endpoint for the exact
active transferable Ondo contracts. A contract frozen before validation covered
the complete four-contract Alpha/perpetual population: CRCL, TSLA, COIN, and
MSTR. All four minimum common quantities fit displayed top-level capacity, but
zero survived the frozen 20 bps pre-account stress. Gross entry headroom was
0 bps for CRCL, 3.4592101470 for TSLA, 6.5313231372 for COIN, and a best
7.2132724213 for MSTR, before commission, settlement/network cost, funding,
exit basis, and non-atomic leg risk. Seven public GETs completed in 4,489 ms;
the 2,412,967 retained raw bytes are locally hash-bound by the result. Do not
repeat this current snapshot. Reopen only after a material Alpha fee, execution,
or book-architecture change capable of clearing the 20 bps gate. Canonical
artifacts:
`docs/model-research/action-value/binance-ondo-bstock-stock-perpetual-wrapper-parity-candidate-v1-2026-08-27.json`,
result SHA-256
`8bcf6f7bfa0cca6dab1fd6fd854a331d5ee41366ac6f9c0244b62a8f3545f475`;
`docs/model-research/action-value/binance-alpha-ondo-perpetual-parity-contract-v1.json`,
contract SHA-256
`2f08c6b0a8509d9d51db7716d5dde499c3a1937b68eafe77b2970e4da8311b59`;
and
`docs/model-research/action-value/binance-alpha-ondo-perpetual-parity-v1-2026-08-27.json`,
result SHA-256
`a3d474e9010b92c9454a5bc04b5a7f586656c8bc5842cecc61baaa508c2d8bc3`.

Binance Stocks now has one accepted scoped, time-limited cost overlay. The
current public fee page saves 5 bps versus the normal trading spread strictly
above 340 USD, or 0.18 USD per order strictly below 340 USD, through
`2026-08-31T00:00:00Z`. At exactly 340 USD the page labels both tiers as
inclusive, so precredit zero until a current order preview and owned realized
fee resolve the ambiguity. Apply the saving only to independently justified
organic direct-stock flow; never resize, acquire quote inventory, or trade to
chase it. The zero account, regulatory, and USDC/USD conversion fees are
current baseline cost absences, not extra promotional savings. Account,
jurisdiction, symbol, tax, spread, preview, and realized-fee evidence remain
unproved, so the overlay is not deployment-ready. Canonical result:
`docs/model-research/action-value/binance-stocks-current-fee-overlay-edge-v1-2026-08-27.json`,
SHA-256
`d4f02be559d9267abbea28ccefb48f4886f375b359ce7274b90b6585b828160a`.

A distinct bStocks fee overlay is now accepted in the same deliberately narrow
way. Binance's 2026-08-28 announcement extends zero maker fees on all supported
bStocks pairs through `2026-09-30T23:59:00Z`. Credit only the exact otherwise
applicable account maker fee on an owned qualifying maker fill from an
independently justified organic bStocks order. A bot order, post-only
submission, or displayed strategy does not prove maker execution. Never create
or enlarge volume, use the fee to rescue a negative trade, or double-count it
with BNB discounts or liquidity-provider rebates. Account eligibility, exact
counterfactual commission, owned fill role, realized zero fee, and every
underlying trade cost remain unproved, so this is neither standalone profit nor
deployment-ready. The same source added bStocks to several trading bots, but
automation is not a new payoff. Canonical trigger triage:
`docs/model-research/action-value/binance-aug28-public-structural-trigger-triage-v1-2026-08-29.json`,
SHA-256
`bca11d612042f9a859f53b71e425cd320cca5d4a5d7695cd1f0a0de539b0eea1`.

The announced TradFi perpetual mark-price update effective
`2026-08-31T08:15:00Z` changes Price 2's basis moving-average window from 30 to
60 one-second observations. It changes mark-path and liquidation behavior, but
publishes no fee, funding cash flow, payout, atomic execution, principal
subsidy, or conversion right. Do not rerun terminal TradFi parity or XAU/PAXG
carry screens for this smoothing alone. Reopen only if an effective change
alters a frozen cash-flow, fee, margin, executable-conversion, or quantitatively
binding liquidation gate. The current Binance Trading Bots guide likewise adds
no Smart Arbitrage subsidy, so that terminal family remains closed.

The older native-stock/TradFi-perpetual parity screen remains an incomplete
result for its frozen 14-symbol population. Thirteen rows completed and zero
survived the 30 bps stress; NBIS led gross at 27.5414636448 bps but failed
one-share perpetual capacity, while SNDK was the best capacity-valid row at
14.6402997861 bps. Current official all-symbol Stocks stream behavior
materially expanded the discoverable population, invalidating the old
KLAC-only completeness assumption. Do not run the KLAC-only recovery. One
exploratory expanded public screen found no after-public-fee positive row but
did not retain a canonical raw population, so it cannot terminalize the
current universe or support profit. Reopen only after a material fee, basis,
or stream-architecture change, with a preregistered exhaustive population
boundary and raw quotes retained before every calculation. The old canonical
contract and result remain source authority only for their frozen population:
`docs/model-research/action-value/binance-native-stock-perpetual-parity-contract-v1.json`,
SHA-256
`ec5d4855c69d3afa461838b674530936a07e646394540a1a2b30ae3ddaf77db1`,
and
`docs/model-research/action-value/binance-native-stock-perpetual-parity-v1-2026-08-27.json`,
SHA-256
`2776ff86fddf78e7e87860c6b9500cb237fce5af908a4840d351ae0cc2eff930`.

The official 2026-08-28 launches of TEMUSDT, MRKUSDT, IONQUSDT, MARAUSDT,
and PDDUSDT materially triggered one separately frozen delta rather than a
repeat of the old population. Exactly five no-retry public native-stock quote
streams were opened; all five timed out without a valid exact-ticker quote, so
the contract correctly made zero futures or USDCUSDT requests. There is no
matching native-stock/perpetual row, price screen, profit evidence, or account
access from this delta. Do not rerun it. Reopen only for a new official listing
that creates a previously unscreened exact ticker match, or a material native
stock stream-architecture change. Canonical contract and result:
`docs/model-research/action-value/binance-native-stock-new-tradfi-perpetual-contract-v1-2026-08-29.json`,
SHA-256
`37a3424645103a351c232ec7bf7c6e2cb4912be1e60bf136bc8cc170644f9adf`,
and
`docs/model-research/action-value/binance-native-stock-new-tradfi-perpetual-result-v1-2026-08-29.json`,
SHA-256
`d8b87863ea750386f1074daef988443a12390f0a36cfecc538765e00bded9a9f`.

A new primary paper, *Taker vs. Maker Arbitrage*, distinguishes the execution
mechanism in which an arbitrageur passively supplies one payoff-equivalent leg
and, after that leg fills, aggressively completes the bundle. This materially
extends the existing paired crypto maker-rebate family, but it does not prove a
stable edge. A reproducible DuckDB diagnostic conservatively deduplicated one
exact 2026-04-27 Polymarket-v1 daily partition and required a public actor to
have exactly two condition participations: first an outcome buy as maker, then
an exact-equal-quantity opposite-outcome buy as taker 1-60 seconds later and at
least 10 seconds before close. Across 3,111,951 scoped BTC/ETH/SOL fills, 159
strict sequences in 105 conditions survived. Applying the current crypto taker
fee curve as a sensitivity left only 75 positive sequences (`47.1698113208%`),
a median complete-set cost of `1.012037`, and aggregate sensitivity P&L of
`-9.33095786` pUSD on 1,305.82 matched shares. BTC and ETH were negative in
aggregate; SOL had zero sequences despite 203,085 scoped fills. Twelve of 24
UTC hour bins were aggregate-negative and four contained zero positive
sequences.

This rejects market-situation independence for the observed sequence. Without
the opposite executable ask at the maker fill, a later profitable completion
requires the opposite leg to become cheaper during an unhedged interval. The
dataset explicitly omits order placements, cancellations, books, and queue
position, is historical v1 rather than current v2, and cannot establish that
the hedge was lockable at order creation. Do not download more historical days
or treat the paper headline, the 75 favorable paths, or current-fee sensitivity
as realized profit. After the protected boundary and only with explicit
authenticated paper authority, a minimum prospective study may post one leg
only when its price plus a synchronized exact-quantity opposite ask and every
cost is strictly below one. On an owned maker fill it must immediately submit a
frozen-cost FOK/FAK hedge, then reconcile acceptance, latency, partial or failed
hedges, cancellations, orphan P&L, fees, rebates, merge/redemption, capital, and
all cross-regime slices. Canonical candidate:
`docs/model-research/action-value/polymarket-maker-first-taker-hedge-complete-set-candidate-v1-2026-08-27.json`,
result SHA-256
`4fe308ddeb6fd080bbd8548347a095762d8fc67eb5820fb0c7b3c2d6b7430d69`.

A second primary paper, *Settlement Manipulation in Prediction Markets*,
rejects treating even ordinary spread capture as an all-situation BTC
five-minute edge. Its P3 sample classifies 227 market makers from 243,155 public
wallets. In 1,613 top-decile manipulation-pressure cycles the cohort lost
`0.62M` USD (`381` USD per cycle) and was negative in `58.6%`; across 14,460
normal cycles it made `3.11M` USD (`215` USD per cycle) and was negative in
`37.7%`. The sign reversal is a regime failure, and the paper explicitly leaves
net-of-changing-fee P&L to future work. Its manipulation label divides final-
ten-second Binance order flow by the completed cycle's median body flow, making
it ex-post and inadmissible as a live filter. The paper also finds that, when a
favored side traded from `0.90` to `1.00` ten seconds before close, a push
reversed resolution in `34.2%` of identified push cycles versus `1.0%` without
a push. This evidence is used only to avoid toxic settlement exposure; never
attempt, facilitate, or simulate trading the underlying to influence resolution.

The paper's P2 fifteen-minute BTC test has 229 near-the-money and 5,870 far-
from-money classified cycles and finds the manipulation footprint largely
absent, without proving market-maker profit. Consequently, the first future
maker cohort is fifteen-minute only and must still satisfy the full creation-
time complete-set lock, owned-fill, hedge, fee, rebate, inventory, orphan,
capital, and cross-regime gates. Five-minute remains excluded until a separate
current source-continuous latency-stress preflight proves every owned maker
order cancel-confirmed before a source-bound settlement-risk window without
using future PushIntensity or reversal. Canonical regime gate:
`docs/model-research/action-value/polymarket-maker-execution-manipulation-regime-gate-v1-2026-08-27.json`,
result SHA-256
`7d3387289a7e82b33fa52c03b2bc134864259a001c3d28524745026bb83db387`.

Polymarket's official changelog creates a current execution-regime break:
crypto taker delay fell from 250 milliseconds to 50 milliseconds at
`2026-08-17T11:00:00Z`. A marketable taker order therefore waits 200 ms less,
or only 20% of the old interval, before matching. This mechanically reduces the
time in which a resting maker might cancel a stale quote; it does not quantify
current adverse selection, queue behavior, or maker PnL. Treat every 250 ms
result as historical-only before the effective timestamp and never use 250 ms
as a current forward assumption. Do not resample books on this change alone.
The existing protected-boundary and explicit authenticated paper-authority
retry trigger remains unchanged. Any future authorized execution contract must
source 50 ms or the exact current market delay and fail closed on absence or
conflict. Canonical regime artifact:
`docs/model-research/action-value/polymarket-crypto-taker-delay-regime-change-v1-2026-08-27.json`,
result SHA-256
`c7b785a1fbf4d6380033810338b2cf2845399f2a7464688c8ac36427b375a777`.

Current official sports execution rules add a separate fail-closed constraint
to the live-NBA implication candidate. A marketable sports order enters the
market's configured delay and cannot be cancelled while pending; after that
delay it is revalidated and may match, reject, or become unmatched. The help
center lists a three-second sports delay and a one-second NBA/MLB test, but the
compact public CLOB market-info response does not expose a numeric sports delay:
its `itode` boolean identifies a separate crypto/finance delay path without a
numeric duration, and `oas` is only documented as minimum order age. The
current crypto duration is 50 ms from the changelog; no finance duration is
inferred. Never reuse a crypto or finance constant for sports.

The delay is not free maker protection. Public rules do not prove that a maker
can observe incoming delayed taker intent or win a cancellation race. After a
maker fill, the marketable hedge itself can instead become an uncancellable
pending orphan; two all-taker legs can be delayed and revalidated independently,
so synchronized books or batch submission do not prove atomic paired fills.
The current Trading Fees and Maker Rebates help pages also conflict between 15%
and 20% sports maker rebates. Credit zero until effective-date or owned payout
evidence resolves the conflict. A future public NBA recurrence screen must use
independent one- and three-second causal delay sensitivities and no reward
credit; authenticated paper execution additionally requires exact runtime
delay phase, every owned state transition, full orphan P&L, and explicit
authority. Canonical gate:
`docs/model-research/action-value/polymarket-sports-taker-delay-maker-protection-gate-v1-2026-08-27.json`,
result SHA-256
`4847ec7828e598950da9a455170b66a529d9a5d671bfb4c37a57a36f608b9627`.

## Broad Crypto Funding And Options RFQ R&D

The prior BTC/ETH/SOL-only funding-carry boundary was a research omission, not
evidence about the broader cross-section. A one-use frozen public preflight
corrected it without changing execution scope. It selected exact Binance
Spot/USDT and USD-M USDT `PERPETUAL`/`COIN` pairs by the smaller current 24-hour
quote volume, required at least 25 million USDT on each leg, and retained 23
raw public responses plus a durable response journal before parsing. Seventeen
symbols qualified. Zero passed the frozen training, validation, and test gate.
Every role was negative after 32 bps of round-trip execution stress plus a 10%
annual opportunity hurdle on each of two gross capital legs, and every
family-adjusted moving-block bootstrap lower bound was negative. PYTH was the
least-negative mature holdout (`-82.35/-77.02` bps validation/test); TUT had
`-24.69/-86.83` bps. Neither approached promotion, and neither passed the
required direction, volatility, path, drawdown, and concentration gates.

This closes the current-liquidity-selected broad funding-only preflight; do not
resample, change the universe cutoff, or relax costs. The selection is
historically biased by construction and therefore could never accept an edge,
but the all-negative frozen result does not justify a more expensive
point-in-time basis study. Canonical contract/result:
`docs/model-research/action-value/binance-broad-crypto-funding-carry-preflight-contract-v1.json`,
contract SHA-256
`7a2c34c4f9c0d44bcd6bed4564d33cda89831ab9a60341ac7ca23cc452231ed1`;
`docs/model-research/action-value/binance-broad-crypto-funding-carry-preflight-v1-2026-08-27.json`,
result SHA-256
`095009a36a5c6a8a5a2dfdfb3e57ebe6183721bb84600518552ccf6d463617c8`.

A distinct Binance-Hyperliquid direction-neutral funding-spread hypothesis was
then reproduced from the exact Zenodo package and extended with one frozen
70-day public interval. BTC, ETH, SOL, and DOGE each had complete 1,680-hour
cross-venue coverage. After the frozen 20-bip round trip and exact entry/exit
premium-basis drift, their APRs were 0.23645%, 2.16957%, 1.27718%, and 1.17155%;
the primary BTC/ETH equal-weight basket was 1.20301%. All failed the
same-timestamp 3.86% DGS3MO opportunity-cost hurdle. Do not refit or resample
this interval. Reopen only after a material venue-fee, funding, basis, or
hurdle change. Canonical result:
`docs/model-research/action-value/binance-hyperliquid-cross-venue-funding-spread-extension-v1-2026-08-27.json`,
result SHA-256
`23eb54dfd19890d984d73156ef05950f7362f8fffe081b93cc5d471f59f62755`.

A distinct Binance Options execution architecture is materially reopened.
Current official Options RFQ sources prove that predefined same-expiry
two-leg call spreads and put spreads bypass the public order book and execute
all predefined legs together. They do not prove arbitrary custom legs, a
four-leg box template, a documented RFQ API, minimum size, Last Look behavior,
account eligibility, exact commission/margin/capital cost, or a positive
quote. The official developer corpus contains standard `/eapi/v1/order` and
`/eapi/v1/batchOrders` paths plus signed `GET /eapi/v1/commission`, but zero
literal Options RFQ references and no documented RFQ endpoint. Therefore the
old displayed-book vertical and box results remain terminal; only the two-leg
vertical execution architecture is reopened.

Do not request a quote without explicit quote-request-only authority. When the
trigger exists, preregister one minimum nonaccepted call-spread and one
minimum nonaccepted put-spread RFQ for a fixed payoff identity, stop without
confirmation, and advance only if each exact account quote clears commission,
settlement, margin, capital, and cross-regime stress. Confirmation or execution
requires separate explicit authority. Canonical triage:
`docs/model-research/action-value/binance-options-rfq-fixed-payoff-execution-triage-v1-2026-08-27.json`,
result SHA-256
`64943efe0c6ad16f8d02f78548afef38f919448d2da87c7573e825a2eeefd6b9`.
That checkpoint left seventeen accepted scoped edges. The current registry now
has twenty and result SHA-256
`e712a9086d31944b42f93270256c393c6d8ab38997c20b7f8638cd4aa9088a34`.

The complete Binance XAU/XAG Commodity Options versus matching TradFi
perpetual lower-bound screen is terminal for the active 2026-08-27 and
2026-08-28 expiries. The market-direction-independent construction was long a
call at its ask plus an equal short perpetual at its bid, or long a put at its
ask plus an equal long perpetual at its ask. Across all 92 active commodity
options, 73 published positive asks, but zero had positive indicative gross
terminal-payoff headroom. The best row, `XAU-260827-4580-C`, was already
`-11.1005015702` bps before the frozen 33.5-bps option, exercise, futures, and
exit-basis stress and before adverse funding. The all-options ticker publishes
ask prices but no ask quantities, so zero rows could satisfy executable
top-level capacity; this does not affect the negative-gross rejection.

Do not resample those 92 symbols. Reopen only for a newly listed XAU/XAG
expiry or a material option-fee, funding, basis, book, or product-access
change. Freeze an all-ticker prefilter first; fetch per-symbol option depth
only for rows that are strictly positive after the same conservative stress.
The retained capture made exactly seven public GETs with zero authenticated or
trading actions. A population-hash serialization mismatch stopped after the
first response; the receipt-bound response was reused and requests 2-7 were
made without retrying request 1. A later futures label correction was processed
entirely offline from the seven retained, hash-verified responses. Canonical
contract SHA-256:
`a1ecde2ac379d40fba81840cc9adf10dd731f29bd8b4eba030a6e71521158b94`.
Canonical result:
`docs/model-research/action-value/binance-commodity-option-perpetual-lower-bound-v1-2026-08-27.json`,
result SHA-256
`3cbc79050473b456e4175239b687b0329bc1c7a66d3530842e524ac4200a0905`.

A frozen 2026-08-29 retry-trigger observation then found zero active
`TRADFI_OPTIONS`/`COMMODITY` XAU or XAG rows. Both retained 2026-08-27 and
2026-08-28 expiries had disappeared and no new expiry existed, so the runner
stopped after its single public `exchangeInfo` GET. It did not request option
tickers, futures metadata, books, premium index, or funding. Do not poll the
empty inventory; reopen only after a new official XAU/XAG Commodity Options
listing or relisting announcement, or a material fee, funding, basis, book, or
access change. Result SHA-256:
`42e7fc2fb8e999f948b2c219d61462ee2af7dbc1477cc4acc68e324d4dea5f1d`.

The contract's literal `frozen_at_utc` was manually anticipated and lands
9.218 seconds after the retained request start. The original contract and raw
capture remain unchanged: runner control flow read and hash-bound the exact
contract before HTTP, so a separate adjudication preserves the metadata error
without rewriting or rerunning the consumed observation. The runner and
`AGENTS.md` now reject missing, malformed, offset-free, or future freeze times
before HTTP. Never type a rounded or anticipated freeze time. Adjudication file
SHA-256:
`072d6b83c90a71a50bcf36cb310f0c933f4dfdc1b5e50bb3b7672a97f960f5ba`.

The distinct same-venue `XAUUSDT` versus `PAXGUSDT` perpetual funding-and-
basis spread is also terminal under the retained current architecture. The
frozen two-request public history screen received 500 rows per symbol. Its
original exact-millisecond join retained only 83 rows because the venue's
independently published settlement timestamps differed by as much as 13 ms.
The original contract and failed result remain immutable. A no-refetch
adjudication aligned all 500 corresponding ordinal settlement slots within the
frozen one-second tolerance, retained each leg's own funding time, rate, and
mark, and evaluated both directions in training, validation, and test.

The training-selected `long PAXG / short XAU` direction was `-296.9983`,
`-151.9946`, and `-104.9276` bps in training, validation, and test after 40 bps
round-trip execution stress and a 10% annual opportunity-cost hurdle on each
capital leg. The opposite direction also failed every role. Do not paginate,
rerun, or refit this tail; reopen only after a material funding, index, fee,
margin, or product-architecture change. This corrected a reusable methodology
error without spending another request and produced no accepted edge or trading
authority. Contract SHA-256:
`39367c3544711a6c206e8d9a3b98f1832ed2c912d8f113ad35872a1fb11e6f36`;
original result SHA-256:
`4cf430a7c5b6ce6ab57fd71979d705732e737cbcb116658705454b80daa025a9`;
adjudication SHA-256:
`46bf134d1be8b645d7f6272d651be8d3c0b6a8e5b2e7d2b4540f3609d6997a96`.

The newly listed `DJTBUSDT` bStock and `DJTUSDT` TradFi perpetual were tested
once as a symbol-only material universe addition to the existing bStock carry
family. Official metadata binds DJTB to DJT with multiplier exactly one. All
100, 1,000, and 5,000 USDT targets had same-quantity depth, but their gross
entry headroom was already `-45.5622`, `-56.5107`, and `-91.6986` bps. Six
funding events existed; the short hedge paid `26.3045` bps across them, and no
positive funding was credited. Zero target survived the frozen 50-bps fixed
round-trip and exit-basis stress. Do not resample this listing state. A future
new exact-multiplier bStock with a matching previously unscreened TradFi
perpetual may receive one symbol-only public prefilter; all account, scope, and
execution gates remain separate. Canonical result:
`docs/model-research/action-value/binance-djtb-new-listing-spot-perpetual-v1-2026-08-27.json`,
SHA-256
`2b85a6eca339799a6eb07ba48069e3a2943d97116a9320ce20400d260227e1be`.

The frozen 2026-08-29 inventory-delta observation then tested only whether a
new exact-multiplier listing trigger existed after DJTB. Its single public
bStock inventory response was byte-identical to the retained DJTB baseline:
68 rows, SHA-256
`87aa11d459f9babcba9837743ab616fef4c066b20e209524b43ee383429cde3d`,
with zero added or removed tickers. The conditional full futures metadata
request therefore did not run; no funding or books were requested. Do not poll
this unchanged registry. Reopen public listing research only after a new
official bStock listing announcement names a previously unscreened matching
TradFi perpetual. Canonical delta result:
`docs/model-research/action-value/binance-bstock-inventory-delta-result-v1-2026-08-29.json`,
result SHA-256
`c343614b061e19ba32813b911d984630d8260cb3a46a1216389a63609a75925c`.

The DJT contract's manually entered `frozen_at_utc` was incorrectly
future-dated. The original contract hash is preserved; correction artifact
`docs/model-research/action-value/binance-djtb-new-listing-contract-timestamp-correction-v1.json`
records the local `03:42:18Z` final-write evidence and the later first request
at `03:44:38.213Z`, with no economic or decision-rule amendment. Freeze
timestamps must henceforth be read directly from the system clock when the
file is created and checked against the first request timestamp; never type or
estimate them. Correction SHA-256:
`692ee1f9a0374726b0adeccbe4fcd710c9c3a86d49b4b33210a8c34b2c79c4d3`.

## Lite Loan Stablecoin Yield-Curve R&D

The current Binance Lite Loan promotion creates one narrow, time-limited
public candidate when its 0.3% upfront fee is paired with the fixed 7% USD1
Simple Earn bonus. A frozen one-request screen used the current public
USD1/USDT book and the official zero-fee USD1/USDT row. At loan amounts of
100, 500, and 1,000 USDT, current fixed-bonus-only net headroom was 24.9497,
25.2827, and 25.3382 bps. After the retained worst 30-day USD1 close decline,
only 1.1079, 1.2964, and 1.3278 bps remained, equal to 0.0111, 0.0648, and
0.1328 USDT. The U and plain-USDT routes failed the same gate.

This is not an accepted, stable, profitable, or deployment-ready edge. The
historical close stress is not a future depeg or issuer-loss bound; account
region, product access, quota, exact repayment behavior, reward rounding,
redemption, and every custody, tax, liquidity, and operating cost remain
unproved. The BTC collateral must already be independently held and idle for
the full 30-day term. Variable Real-Time APR and the random voucher are both
credited as zero.

The loan promotion ends at `2026-08-27T23:59:59Z`. Before then, only if both
designated credentials and explicit signed GET-only authority exist, an exact
account eligibility/product/quota prequalification may run. It must not borrow,
convert, subscribe, or repay. Every such funded action requires separate
explicit authority. Without that trigger, let this candidate expire; do not
resample the unchanged book or reprice it after expiry. Reopen only for a
materially new loan, yield, fee, or eligibility term.

Canonical contract/result:
`docs/model-research/action-value/binance-lite-loan-stablecoin-yield-curve-contract-v1.json`,
SHA-256
`73f6a0362ca88db393b723119b21e603f202345f570b59a70204fa3779349d41`;
`docs/model-research/action-value/binance-lite-loan-stablecoin-yield-curve-v1-2026-08-27.json`,
result SHA-256
`65f223a245fa1bb65a8fd791275da0dbd71d3c52ee2d232ac1420feb198b129d`.

## bStock Spot LP All-Symbol Rebate R&D

The 2026-08-24 through 2026-10-18 bStock Spot Liquidity Provider promotion
adds a separate bStock maker-share route into Binance's existing Spot LP tiers.
The bStock weekly maker-share thresholds are 0.05%, 0.10%, 0.30%, and 0.60%
for tiers 1 through 4. Their maker fees are 0, -0.0040%, -0.0060%, and
-0.0080%, respectively. If the resulting bStock tier exceeds the participant's
original category tier, Binance says the higher rebate applies across all
symbols. The first review week ends `2026-08-30T23:59:00Z`; its tier is first
effective from `2026-09-01T00:00:00Z`, so no positive rebate is credited now.

This is a conditional all-symbol fee-overlay candidate only for independently
existing, legitimate organic maker flow. The public source does not expose the
same-account enrollment, original tier, bStock denominator, owned volumes,
fills, adverse selection, inventory hedges, final rebate rows, or after-cost
persistence. The manual trial requires more than 20 million USD of 30-day Spot
volume but grants tier 1, whose positive rebate floor is zero. Never manufacture
volume, self-deal, wash trade, or apply on the user's behalf. A maximum 0.8-bp
rebate is not proof that bStock qualification or hedging is profitable.

Do not fetch books. Not before the first effective week, and only with both
designated credentials plus explicit signed GET-only authority, query the
frozen liquidity-program overview, performance, weekly result, and rebate
history. Join those rows only to already-existing owned organic maker flow;
orders, applications, and volume generation remain forbidden. Canonical result:
`docs/model-research/action-value/binance-bstock-spot-lp-all-symbol-rebate-overlay-candidate-v1-2026-08-27.json`,
result SHA-256
`d279f8ab88875c812e6691fa500fdfde741f2e2fbca19ee240b4c0d4a579d607`.

## Existing-Stock Transfer Reward R&D

Binance's 2026-08-11 through 2026-09-30 first U.S.-stock transfer-in program is
the widest public structural reward lead in the current pass. For first
successful transfer values of 2,000, 10,000, 30,000, 150,000, 400,000, and
1,000,000 USD, the fixed transfer bonuses are 50, 150, 200, 600, 1,000, and
2,000 USDC: 250, 150, 66.67, 40, 25, and 20 bps at the thresholds. After an
illustrative 10% annualized 21-day liquidity hurdle, the first three thresholds
still retain 192.47, 92.47, and 9.13 bps; the larger three fail that sensitivity.
Fee-reimbursement caps are excluded from profit and may only offset actual
eligible owned fees after voucher receipt. IBKR-origin transfers do not qualify
for fee reimbursement.

This is a high-margin candidate, not an accepted or guaranteed edge. The
300,000-USDC pool is allocated strictly first-come by successful credit time and
may end early, so the public forward reward floor is zero. The program excludes
the United States, United Kingdom, EEA, and other restricted jurisdictions;
host timezone does not prove account region. Exact KYC, Stocks access,
first-transfer status, supported symbols, same legal owner, pool availability,
transfer fees, tax, custody, corporate actions, price-snapshot tier buffer,
voucher conversion, and operational costs remain unproved. Transfers generally
require at least 14 business days, and the bonus is cancelled by transferring
out activity stocks or selling and withdrawing the equivalent within 21 days
after credit.

Do not buy or transfer stock to chase the reward. Before any external action,
explicit read-only account authority must first prove exact eligibility and a
live unallocated or account-reserved reward. Then present the exact existing
inventory and cost decision to the user. Submitting a Binance or external-broker
transfer is a separate high-impact action requiring explicit authority.
Canonical result:
`docs/model-research/action-value/binance-existing-stock-transfer-reward-overlay-candidate-v1-2026-08-27.json`,
result SHA-256
`3ecb4f39848719f788b6853bd90120d1809379b8d81b5419da4b1bbc957fec3d`.

## Binance Stocks FPSL Existing-Inventory Yield R&D

Binance Stocks Fully Paid Securities Lending is a materially distinct
direction-independent incremental-income candidate. The official FAQ, updated
2026-08-14, says eligible fully settled U.S.-listed stocks and ETFs already held
in the account may be lent by Alpaca to institutional borrowers. Interest
accrues daily only while shares are actually on loan, is paid at the end of the
following month, and selling remains available because a sale automatically
recalls the loan. The cashflow is market value times annualized lending rate
times the account's fee-share percentage times days on loan divided by 365.

This is not an accepted, stable, or publicly proved profitable edge. Lending is
not guaranteed, public sources provide neither current symbol rates nor the
Binance account's fee-share percentage, and loan records may lag by two business
days. The public forward income floor is zero. Do not import Alpaca's direct
retail or Elite share percentages into the separate Binance account contract.
Shares on loan lose SIPC coverage and voting rights; dividends are paid as
cash-in-lieu, so incremental tax, custody, counterparty, and support costs can
exceed small interest distributions.

Apply this candidate only to stock inventory already independently owned and
intended to remain held. Never buy, transfer, or retain stock to manufacture the
overlay. Exact account region, Stocks and FPSL eligibility, settled holdings,
actual loan days, rates, share percentage, monthly distributions, and after-cost
economics remain unproved. Account inspection requires explicit read-only
authority, and enabling FPSL or answering its suitability questionnaire is a
separate account-state change requiring explicit user authority. Promote only
after at least three independent owned monthly distribution cycles remain
positive after every incremental cost. The expired July leaderboard has zero
forward value and is only discovery evidence for the continuing product.
Canonical result:
`docs/model-research/action-value/binance-stocks-fpsl-existing-inventory-yield-overlay-candidate-v1-2026-08-27.json`,
result SHA-256
`3fe1801a6cbf442ab1ce79d1f3bd4586542d97414aea954b0bbd9a55a85453e1`.

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

The 2026-08-29 exact-one-NO NegRisk V2 event-log observation consumed its
frozen one-use public request after the prior adapter-address evidence
materially opened that blocker. The preliminary `eth_blockNumber` receipt
retained block `92874137`, but the exact `eth_getLogs` request failed HTTP 400.
The frozen runner did not retain the HTTP error body or completed receipt, so
the cause is unknown and the outcome proves neither presence nor absence of an
exact one-NO conversion. Do not retry that interval or switch providers.
Canonical failure adjudication:
`docs/model-research/action-value/polymarket-negrisk-one-no-v2-conversion-failure-adjudication-v1-2026-08-29.json`,
result SHA-256
`7c976cd84795718b63463ea4e32ebeddaf51e807fc5ebe9aa8cb49b476541e19`.
Every future one-use HTTP runner must journal method, URL, and exact request-
body hash before access, then retain HTTP error status and body before raising.

The distinct Binance BLVT primary-market NAV-versus-spot parity hypothesis is
terminal for the current inventory. One frozen unauthenticated public
`exchangeInfo` request returned 3,685 symbols and 40 legacy symbols carrying
the `LEVERAGED` permission, but zero had `TRADING` status. No current BLVT
therefore exists for subscription, redemption, or spot parity; no API key,
`tokenInfo`, book, account, order, or mutation was used. Do not poll the
unchanged inventory. Reopen only after a new official BLVT listing or relisting,
then require explicit GET-only authority for the API-key-classified
`tokenInfo` prefilter before any separately frozen book study. Canonical result:
`docs/model-research/action-value/binance-blvt-primary-market-nav-parity-public-gate-v1-2026-08-29.json`,
result SHA-256
`85c8ef364b03fb2fbf0aeebddec10d51abbdd608f56ff9c0dccb1835cacc2179`.

The exact non-crypto Polymarket/Binance TradFi-perpetual funding family is
terminal for the frozen current top five. A retained zero-request instrument
join found 36 exact current matches. One frozen four-request public prefilter
advanced SKHYNIX, CRWV, ARM, HOOD, and MSTR from seven rows above the
conservative 1.8656229615-bip per-eight-hour history threshold. The next frozen
contract spent exactly 15 public unauthenticated GETs on only those five:
Polymarket hourly funding, Binance Regular-plus-Special funding, and Binance
one-hour klines. It requested no books, conversion, credentials, account state,
orders, funds, or protected-capture assets.

SKHYNIX returned 26 normalized settlements and was negative in training and
validation. The other four returned 11 rather than the frozen 12-row minimum,
so the consumed result correctly rejected them as insufficient rather than
pretending to prove economic failure. A separately frozen zero-network
adjudication then reused only the hash-bound saved responses and could strengthen
rejection but never repair or promote the sample shortfall. Every fixed
orientation failed training, validation, and test after the unchanged 20-bip
round-trip hurdle and two 500-bip annual capital hurdles. ARM was least negative:
`-24.6296621005` bips full-window, with training, validation, and test netting
`-23.1729100457`, `-20.7754260274`, and `-20.6813260274` bips. Do not request
books or repeat the selected population and window. Reopen only after a material
funding cash-flow, fee, market-session, instrument, conversion, or execution-
architecture change capable of clearing the retained role deficits. Prefilter,
history, and retained adjudication result SHA-256 values are respectively
`9e2d5843a986f02f757aa56641c5cb38c35500e62cb2a1804984dea7793a1859`,
`ad896a698edd65b42b039f84d1b037cf67302c7b2bb7ae59e9008f45328939bb`,
and `5e67277ad30b9f0164a3987804162ed2d1cdabb820e7e258d6a0b79748cf7d06`.

The structural registry now contains 43 ranked hypotheses and 20
narrowly accepted scoped overlays, and has canonical result SHA-256
`e712a9086d31944b42f93270256c393c6d8ab38997c20b7f8638cd4aa9088a34`.

The Round 21 sidecar worktree
`C:\trader\simple_ai_trading-round21-sidecar-v2` remains protected through
`2026-08-29T23:40:00Z`. Its process IDs are ephemeral. Do not touch it until a
contract-defined terminal audit proves the boundary has passed and the process,
lease, state, database, and WAL agree.
This is a narrow process-and-file protection boundary, not a Polymarket research
moratorium. Separate unauthenticated, read-only public source, market-data, and
blockchain investigations may proceed iteratively in the main worktree. They do
not grant account, funding, order, or transaction authority.

## Next Work

1. Advance only the highest-ranked structural-edge hypothesis whose exact retry
   trigger is currently satisfied. Keep R&D on source-bound mechanism and
   executable after-cost evidence; defer broad CI, alert, and hosted-workflow
   rechecks until a coherent code or research checkpoint is ready to publish.
2. Keep the completed model-dev three-way audit frozen. Do not bulk-integrate
   stale or divergent files; reevaluate a specific path only when a current task
   requires it.
3. Do not rerun the consumed exact-one-NO NegRisk V2 log interval: its HTTP 400
   lacks a retained provider body and therefore cannot be diagnosed or treated
   as absence.
   Reopen only on materially new exact one-NO access, cost, or causal unwind
   evidence under a separately frozen contract. Do not poll Binance BLVT
   inventory: zero current `TRADING` leveraged-token symbols makes the NAV
   parity mechanism unavailable. Reopen it only after a new official BLVT
   listing or relisting, then stop after the API-key `tokenInfo` prefilter unless
   explicit GET-only authority exists and the exact NAV/fee gap is positive.
   Do not repeat the Polymarket/Binance TradFi-perpetual current top-five
   funding screen: every fixed orientation failed the unchanged economic role
   gates. Reopen only on its material funding, fee, session, instrument,
   conversion, or execution-architecture trigger.
   Do not rerun rejected Binance elevated-funding spot/perpetual carry or the
   terminal one-use broad-current-liquidity-selected crypto funding preflight,
   the terminal 70-day Binance-Hyperliquid cross-venue funding-spread extension
   without a material venue-fee funding basis or hurdle change,
   USDT/USDC perpetual funding
   differential, quarterly carry, two-sided touch making,
   Polymarket binary complete-set taking, negative-risk parity, logical
   threshold/deadline implication parity, Binance spot triangles, Binance
   option vertical/convexity parity, option box parity, or Polymarket exact
   cross-condition duplicate discovery, the incomplete Combo catalog contract,
   the terminal Polymarket/Binance Perps OI-carry contract, or paired-maker
   reward snapshots as if repetition could create an edge.
   Do not resample the rejected pUSD external parity pools or request finite-size
   quotes unless a materially lower-fee same-asset route appears or a
   source-bound deviation first clears fee, conservative gas, and price impact.
   A repeat is justified only by a frozen prospective sampling contract or
   materially new fee/execution evidence.
   The live NBA moneyline/spread candidate is the one separate implication
   family materially reopened by new primary recurrence evidence. Do not sample
   unrelated sports or the off-season repeatedly. When future active NBA
   full-game markets exist, first bind exact same-game settlement and per-market
   fee schedules, then run only the preregistered synchronized public all-taker
   recurrence contract. Any uncovered push, overtime, cancellation, or forfeit
   state rejects the pair.
   The general two-market dependent-subset candidate is also materially
   reopened, but only for exact multi-outcome subset-indicator equality. Do not
   replay the historical VWAP study or search semantically guessed pairs. A
   future pair first needs complete source rules and an exhaustive deterministic
   joint payoff table; only then may the frozen one-batch current-fee recurrence
   contract run. Keep specialized NBA observations out of the general count.
   For Binance Launchpool, do not poll the WAF-empty current page or reuse the
   historical OPN campaign. A new official active or upcoming project with its
   own stablecoin pool, exact window, allocation, eligibility, and distribution
   terms is the only public retry trigger. Account GETs and every lock, unlock,
   transfer, conversion, or reward sale retain their separate authority gates.
   For the Polymarket Combo Boolean identity, do not repaginate the terminal
   broad catalog or treat catalog prices as quotes. Advance only when approved-
   builder access and explicit quote-request-only authority exist. Request one
   minimum nonaccepted BUY and SELL RFQ first; fetch synchronized underlying
   books only if the exact RFQ totals leave positive conservative headroom.
   Quote acceptance, account mutation, and any order remain separately gated.
   For Binance bStock dividends, do not repeat direct pre-adjustment long-bStock
   short-perpetual capture: historical AMAT and MSFT prove the short pays the
   gross dividend while bStock receives only net reinvestment. The frozen GLW
   pre-snapshot history observation is now consumed: eight observed rows were
   all `Regular`, so no conditional books ran. Do not poll it or attempt a 2026
   GLW book capture. After the snapshot, freeze at most one terminal history
   reconciliation for mechanism evidence only; any executable recurrence must
   wait for a future independent weekend event. No account access, order,
   gross-dividend credit, or profit claim is authorized.
   The separate bStock listing inventory refresh is also consumed and matched
   the 68-row DJTB baseline byte-for-byte. Do not poll it. A new official bStock
   listing announcement plus a previously unscreened matching TradFi perpetual
   is required before another one-symbol public prefilter.
   For the NOK under-debit exception, do not infer the net historical payout
   from the `0.0462` gross amount, current multiplier, or `lastCashAmount`, and
   do not poll the preliminary 2026-10-27 record date. Reopen only when Nokia's
   Board resolves an exact new amount and Binance publishes a matching NOKB
   announcement; close the historical multiplier event lineage before any
   prospective event-time capture.
   Do not generalize NOK to TSM: the source-selected 2026-09-16 TSM event is
   terminal because the prior exact Special short debit exceeded both the
   prior gross and net ADR dividend before all other costs. No TSM book or
   funding refresh is allowed for that event.
   For Ondo/bStock/stock-perpetual wrapper parity, Binance Alpha public full
   depth has now closed the missing executable-looking ask question. Do not
   repeat either the old point screen or the frozen four-contract Alpha book
   screen: zero of CRCL, TSLA, COIN, and MSTR survived the 20 bps pre-account
   stress, with MSTR best at only 7.2132724213 bps gross. Reopen only after a
   material Alpha fee, execution, or book-architecture change capable of
   clearing that gate; freeze a new complete-population contract before any
   request. Installing or signing into a wallet, approvals, transfers, swaps,
   and orders retain separate authority gates.
   For native Stocks Trading versus TradFi perpetual parity, the old 14-symbol
   population remains incomplete but the current all-symbol stream architecture
   has invalidated its KLAC-only recovery instruction. Do not run that recovery,
   treat the old discovery set as current, or promote the unretained exploratory
   expanded screen. The separately frozen TEM/MRK/IONQ/MARA/PDD delta is also
   consumed: all five native quote streams timed out and correctly caused zero
   downstream GETs. Do not rerun it. Reopen only after a new official listing
   creates a previously unscreened exact ticker match, or after a material
   stream-architecture change; retain each raw native quote before pairing it
   with immediate perpetual and USDCUSDT books, and charge stock, perpetual,
   conversion, exit, funding, orphan, and opportunity costs. Stock disclaimer,
   account access, and every order remain separately gated.
   VIP Earn is a distinct conditional idle-inventory overlay, not a reason to
   manufacture trading volume, borrow, buy BNB, or retain a volatile asset.
   Its public BTC/ETH/SOL listing is now exact enough to close the displayed-
   maximum screen: zero row beats the best simultaneously visible non-VIP
   maximum. Do not repeat it. Reopen only after a material current rate,
   duration, quota, redemption, or comparator change, or when both designated
   credentials and explicit GET-only account evidence authority exist. Reject
   undated or pre-launch sheets unless a current official page adopts them.
   Join exact same-product terms, existing VIP status, independently idle
   eligible balance, and every cost; stop before every TRADE endpoint.
   For BFUSD/RWUSD, wait until both designated ephemeral credential variables
   exist, then source-bind and hosted-verify one six-GET rate, quota, and
   flexible-alternative prequalification before its single run. Do not build
   credential-dependent collection code while that trigger is false.
   The same credential trigger applies to WBETH/ETH and BNSOL/SOL conversion
   parity. Source-bind conversion history, quota, account eligibility, exact
   fees, redemption delay, and an equal-base hedge before any public book
   sampling; an unhedged redemption is not market-direction independent.
   It also applies to the delta-hedged BNB Simple Earn and airdrop reward stack.
   Source-bind eligible BNB principal, realized Simple Earn and dividend rows,
   conservative executable reward sale values, and exact account costs against
   the frozen hedge history. Do not refresh BNB funding or books.
   For the idle-native-token family, do not poll unchanged public pages. The
   public liquid-staking rates are now known but are only gross sensitivities.
   When both designated ephemeral credentials exist and explicit read-only
   authority is given, freeze one signed ETH/SOL liquid-staking quota,
   conversion-rate, reward, and operation-history prequalification alongside
   the Soft Staking product-list/reward history and BTC Simple Earn product-
   list, position, and reward-history prequalification.
   Never call the state-changing activation GET under read-only authority, and
   never subscribe, redeem, or credit pending-order, Auto-Subscribe, or prompt-
   liquidity inventory without the separately required authority and evidence.
   For USDe, refresh the public page only after a published APR, eligibility,
   account-scope, snapshot, distribution, or campaign-term change. Exact KYC,
   jurisdiction, eligible balance, completed 24-hour duration, and owned weekly
   distributions require both ephemeral credentials and explicit read-only
   authority. Never acquire, deposit, convert, borrow, or retain USDe for the
   reward, and never double-count collateral use or a separate Ethena yield.
   Do not resample the Binance-option/Polymarket-threshold model wedge unless
   an exact strike-expiry-settlement match appears or a two-sided same-date
   model gap exceeds the frozen 4.27-percentage-point escalation threshold.
4. Do not run Binance Round 76 or Polymarket Round 29 from their failed source
   campaigns. After the protected Round 21 sidecar reaches its terminal boundary,
   use the source-continuity recovery design to freeze separate Round 77 Binance
   and Round 30 Polymarket activation contracts. Each must bind an exact fixed
   schedule, unique per-slot storage, role capacities, host supervision, and a
   pre-market activation receipt. Do not share schedules or storage between
   venues.
   The August 15-minute/4-hour crypto-TWAP liquidity-reward source screen is
   terminal after its exact BTC reward identity failure. Do not retry that
   window or fetch books. A materially new endpoint, exact configuration, or
   program allocation is required to reopen it, and every future one-use source
   collector must journal raw responses before validation.
   The later 550,000 dollar five-minute allocation did materially reopen the
   excluded duration once. That one seven-asset attempt journaled both sources
   and then stopped before books because raw Gamma omitted `clobRewards` for all
   seven exact markets. A distinct complete unsigned sponsored current-rewards
   list then returned 54 terminal-cursor rows and zero exact matches for the
   same seven conditions, also stopping before books. Do not retry either
   source configuration or divide the monthly cap into an invented daily rate.
   Reopen only for a genuinely new public exact per-market dated allocation
   response or another program change.
   The distinct Polymarket maker-rebate study now includes the maker-first,
   exact-quantity taker-hedge mechanism. Its one-day historical diagnostic is
   negative in aggregate and unstable across assets and hours; do not expand
   the public historical tape. It may activate only after the protected
   sidecar boundary and only under a frozen prospective contract that admits a
   maker order when a synchronized opposite executable ask is already positive
   after every cost, then measures authenticated owned fills, immediate FOK/FAK
   hedge success, queue and cancellation latency, adverse selection, realized
   rebate payment, merge/redemption, and complete orphan P&L. Nominal rebate
   algebra and a favorable future hedge fill are not substitutes. Start with a
   fifteen-minute-only cohort. Keep five-minute excluded until a separate
   current preflight proves every owned order cancel-confirmed before the
   settlement-risk window under latency stress; completed-cycle PushIntensity
   and post-settlement reversal are forbidden live inputs. Never trade the
   underlying to influence resolution.
   The finalized-winner-before-adapter-close lead now outranks that broader
   maker study. Do not expand its public history. After the protected boundary
   and only with explicit authenticated paper authority, freeze one minimum-size
   post-only order created after exact on-chain finality and reconcile its full
   acceptance, fill or clean rejection/cancellation, ownership, and redemption
   path without any pre-finality resting exposure.
   Public and on-chain monitoring of the accepted scoped complete-set holding-
   yield edge may continue now without touching the protected sidecar. Monitor
   official terms, current market eligibility, daily `YIELD` continuity, and
   distributor receipts. Deployment still requires an owned split-to-merge
   cycle, every cost outside the documented gasless relayer path, alternative
   cash yield, and capacity. Do not fund, authenticate, or transact before the
   protected boundary and explicit authority.
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

## Resolved-Leg Duplicate-Payoff Checkpoint

Two independent active Polymarket combined events now establish the same
direction-independent payoff mechanism against the standalone 2026 House
control markets. A final-Yes shutdown leg makes each `Shutdown & Party` outcome
identical to the matching standalone party outcome; a final-No ACA-extension
leg does the same for each `Not Extended & Party` outcome.

The economics did not recur. In the shutdown instance, the retained batch had
35.484 seconds source timestamp skew. Offline maker-first sensitivity found
three positive roles after a two-tick hedge, but the most practical current role
retained only `0.03092` pUSD on five shares with 500 visible shares ahead and no
owned queue-censored fill or causal hedge. The independent ACA batch used one
eight-token request, made no fee requests because every package was already
gross-negative, and had 11.695 seconds total source skew. Every all-taker package
was negative after fees and stress; the best lost `0.29046` pUSD. All eight
maker-first roles were also negative, and the best one-tick-improved zero-queue
sensitivity lost `0.11638` pUSD with 11.173 seconds pair skew.

This closes the current escalation efficiently: a repeated payoff identity does
not establish repeated positive economics. Do not spend another source-
continuous capture, credentials, authenticated paper probe, or order-capable
work on this family until a material price, fee, rule, or market change. The
accepted-edge count remains 19 and no trading authority was opened. Canonical
ACA all-taker result SHA-256
`2c224485b3ee4647000e9cfc016a421c08fdb03c158716ca2b56e95fcca2a90b`;
maker-first result SHA-256
`900246f7bf066a8d310c6dcc6e9318edc1c6b83d4779f9b2f628c668e7d258e6`.

## Current WNBA Monotone-Payoff Checkpoint

The first exact active WNBA extension of the NBA moneyline/spread mechanism is
terminal for Toronto Tempo versus Phoenix Mercury. One public Gamma event GET
proved the exact moneyline, Phoenix -9.5, and Phoenix -8.5 rules, including
overtime and 50-50 cancellation treatment. Exhaustive states prove three
two-leg bundles with a minimum one-pUSD payout per share without forecasting
the winner.

The executable economics failed decisively. One frozen four-token CLOB batch
had 2.132 seconds maximum book timestamp skew and zero positive packages. The
best bundle, Phoenix -8.5 plus Toronto +9.5, lost `0.42436` pUSD at actual
five-share depth, `0.62356` pUSD under the frozen one-second sensitivity, and
`1.02076` pUSD under the three-second sensitivity. No fee endpoint was called:
every package was already gross-negative. No credential, account, fund, order,
transaction, or protected-capture asset was used.

Do not resample this event. For every future NBA or WNBA exact pair, first
prove the payoff identity, then use retained Gamma `outcomePrices` only as a
rejection-only optimistic gate. If every displayed package sum is at or above
its guaranteed payout floor, stop before CLOB books. Gamma prices can never
accept or promote an edge. Only a package below the floor may advance to one
frozen exact-depth batch, and escalation still requires an independently
positive after-cost recurrence. Canonical event result SHA-256
`6851d26788abfd175b75649e573d341696e570ce76bc235b0c5a6070bdd72167`;
book result SHA-256
`cc657982abd9ede0f0f7b18787df32e62c69b7c3b3e547ade3f6f3ccb734ed46`.

The next distinct future WNBA event proves that gate saves resources. One
reusable contract-driven metadata runner captured Minnesota Lynx versus Atlanta
Dream with one exact public GET. Its rules prove Minnesota moneyline plus
Atlanta +2.5 pays at least one pUSD per share in every terminal state, but the
retained Gamma prices sum to `1.080`. That is already an optimistic loss of
`0.080` pUSD per share and `0.400` pUSD at five shares before depth, fees, and
delay. The event is terminal without a CLOB or fee request. Do not resample it
unless price, fee, delay, or resolution rules change materially. Canonical
metadata result SHA-256
`c7629f0869bf7b1b9b6622cde42b0822f35e63386c9cb3e2e4364423fa4f7156`;
adjudication SHA-256
`61b3436b3367ba3442ebe777c8a506948243c6d3b6d6a4cb9346d2db3aaf335f`.

A PowerShell preflight used an unparenthesized `Test-Path ... -or Test-Path`
expression, which emitted a non-terminating parser error before continuing.
The runner's independent one-use output guard prevented duplicate access and
the contract still consumed exactly one request. Future shell preconditions
must parenthesize both operands and enable terminating errors; the runner must
continue to enforce the same boundary independently.

## Current MLB Rejection-Only Monotone Checkpoint

The distinct Los Angeles Dodgers versus Detroit Tigers event validates the
same rejection-first workflow beyond WNBA. One frozen public unauthenticated
event-by-slug GET retained 17 active accepting markets and exact rules. No
credential, account, protected-capture asset, book, fee, order, transaction, or
fund was accessed.

The reusable offline adjudicator parsed the moneyline plus full-game and
first-five spread and total ladders. It proved 24 within-time-scope two-leg
packages whose minimum terminal payout is one pUSD per share, including 50-50
cancellation. All 24 Gamma sums were at or above one; the closest was `1.010`.
Preserve that consumed result, but its completeness wording was wrong: it did
not enumerate deterministic implications across compatible time scopes.

A separately frozen retained-data correction proved the missed identity.
Cumulative full-game runs cannot be below cumulative first-five runs. Therefore
full-game Over 6.5 plus first-five Under 6.5 pays at least one pUSD per share in
every terminal state, including cancellation. Retained Gamma prices summed to
`0.965`, an optimistic `0.035` pUSD per-share lead, so the strict gate allowed
one exact two-token book request. The books had 1.511 seconds timestamp skew.
Five-share asks were `0.44` and `0.96`, costing seven pUSD against the five-pUSD
floor: an optimistic zero-fee loss of two pUSD. The one-second sensitivity lost
`2.2` pUSD and the six-tick sensitivity lacked complete depth. Because the
candidate was already gross-negative, zero fee requests were made.

Do not resample LAD/DET. For every future exact sports event, enumerate both
within-family and cross-period deterministic subsets before calling the payoff
lattice complete. Stop before books unless at least one displayed sum is
strictly below its guaranteed payout floor. Gamma may reject but never accept
or promote. Exact metadata result SHA-256
`7d31545dfb4195b8ecc3fd19e8f2711e4634dd4cc259aa3a8d22f64402852593`;
cross-period result SHA-256
`fc01c54e9c04117067aa3b43ae194649b93efc12a5265fce508e64f082f320b2`.

The next rejection-first extension used one bounded active-MLB catalog request
instead of polling events individually. It ordered the frozen first page by
start time and filtered events to `startTime` strictly after the retained
request completion. The response equaled the 100-event ceiling, so coverage is
explicitly partial and cannot support a complete-current-population claim.
Within that page, 92 events were future-dated and only five had exact shared
full-game and first-five total thresholds with complete rules. None cleared the
strict Gamma gate. Displayed package sums were `1.260`, `1.31`, `1.370`,
`1.375`, and `1.405`; the best already lost an optimistic `0.260` pUSD per
share before depth, fees, or delay. Zero books and fees were requested.

The consumed catalog result retained aggregate counts and candidate rows but
not the five rejected relation rows. Preserve it. A zero-network, zero-refetch
adjudication reconstructed all five exact identities and sums from the saved
raw page. Future rejection screens must retain every tested relation, not only
candidates. Do not adaptively request offset 100 under this consumed design.
Canonical catalog adjudication SHA-256
`d3ba85e995753d781178fdf6144ac0cb7520d2b1830525cd4be1aad1a5b5b598`.

## Binance Same-Expiry CLOB Box Checkpoint

The same-expiry four-option fixed-payoff box mechanism was already terminal in
the registry from the August 25 exact-depth screen. A later audit initially
misclassified it as novel because the terminal registry aliases were not
searched before implementation. The repository rule now requires searching
both prioritized and terminal families plus mechanism and payoff aliases before
building a new collector; an existing terminal family advances only on its
literal retry trigger.

The retained audit still contributes a later, zero-network rejection. It reused
the hash-bound August 27 Binance option catalog and all-ticker payload, covered
1,410 eligible BTC/ETH/SOL option symbols across 22 underlying-expiry groups,
and evaluated 10,382 complete long- and reverse-box directions. Zero was gross-
positive even before fees, margin, four-leg execution risk, settlement, or
capital costs, so no current endpoint, book, credential, account, or order was
accessed. Do not repeat this family absent a material price, fee, book, or
product change.

The first frozen offline implementation failed before writing a result because
397 retained ticker rows lacked `closeTime`. Preserve its contract SHA-256
`01d5a8880556406883414fe53cf19189f3dc6cf1443090726015d3d487e5a754`
and implementation hash. The separately frozen v2 repair treated a missing
timestamp as unconditionally unsynchronized without changing the economic or
60-second gate and reused the same retained bytes. V2 contract SHA-256
`806b99257dd081fddef2fcaa5657776e9dfecee65dd52da1bba351052a062e81`;
canonical result SHA-256
`a9b0e7a2aba9bda7f83b9515be587a17e6da69fa0bc987191a21f9d37e912d3b`.

## Binance Option/Perpetual Conversion Checkpoint

A distinct zero-network retained screen tested every BTC, ETH, and SOL same-
strike call/put pair against its matching linear USDT perpetual. The conversion
and reversal payoff algebra is market-direction-forecast independent, but a
perpetual must still be closed at option expiry; funding, expiry basis, margin,
and three-leg execution therefore remain economic risks rather than a fixed
payoff identity.

The first frozen implementation made a timestamp-semantics mistake: it treated
the Options ticker `closeTime` as best-bid/ask update time. Binance's official
Options schema instead labels the corresponding ticker timestamp as transaction
time and does not document it as quote-update provenance. Preserve v1 and its
canonical result SHA-256
`c7dfb805da6d55bb3fcccb48cb45baa7bb3044f0d304f26dfc1dd67b3bc7529f`;
do not reinterpret its zero synchronized count. A separately frozen v2 changed
only synchronization provenance. It used the retained HTTP request intervals as
observation windows, kept `closeTime` diagnostic-only, and measured 1.639
seconds maximum possible observation skew. V2 found 71 synchronized nominal
gross-positive directions among all 1,410 directions: 35 BTC reversals, 34 ETH
reversals, and two SOL conversions. V2 contract SHA-256
`5cbcc2f302b78da934d63801a58c56e640670e4b31935a6e2542536360502ee5`;
canonical result SHA-256
`1adf3c51cd008d40744c8ae91e1e4865e9e19e7dbb1ae9a0995e2b2676b3bd58`.

None advanced to current depth. A third frozen offline gate covered all 71
rows using the exact common quantity lattice, two adverse ticks on each of the
three legs, 4 bips for two option taker fees, 1.5 bips for one settlement fee,
10 bips for the perpetual round trip, 20 bips expiry-basis stress,
direction-specific worst retained 500-event BTC/ETH/SOL funding stress at 1.25
times entry notional, and a two-notional 10 percent annual capital hurdle. Zero
rows survived. The best BTC, SOL, and ETH rows ended at respectively
`-37.4697460101`, `-44.0814249716`, and `-59.4016381237` bips before option
depth, account-bound commissions, or owned fills. No current endpoint,
credential, account, order, or fund was used. Do not repeat this retained
snapshot or request current depth absent a material price, fee, book, funding,
margin, or product change. Stress contract SHA-256
`5ac091035b9eeadda23292fa28631dcc7c8bb0b64e001faa34c94ffad5b6ecc5`;
canonical result SHA-256
`c09d62e98cd0df88622d4b98d9d8f01247121ccd786fffb580bc72429ef6bf30`.

## Current NFL Tie-State Monotone Checkpoint

Packers versus Vikings is the first exact NFL extension of the same-game
moneyline/spread monotone-payoff family. One frozen public unauthenticated
event-by-slug request retained 34 active accepting markets: one moneyline, nine
spreads, and 24 totals. No credential, account, protected-capture asset, order,
transaction, or fund was accessed.

A separately frozen zero-network adjudication modeled final score margin as an
integer, retained actual-game tie and cancellation as explicit states, and
enumerated all 321 ordered threshold relations across the compatible full-game
margin and total families. Every relation pays at least one pUSD per share.
Four retained Gamma sums were below that floor. The strongest was the Packers
outcome in Vikings -0.5 (equivalent to Packers +0.5) plus Vikings moneyline:
Gamma displayed `0.375 + 0.52 = 0.895`, an optimistic `0.105` pUSD per-share
floor before execution costs. In an actual tie, that package pays `1.5`; in a
Packers win, Vikings win, or cancellation it pays at least one.

One frozen exact two-token book batch tested that strongest candidate. Exact
five-share asks were `0.55` and `0.53`, costing `5.400` pUSD against the
five-pUSD floor: a `0.400` pUSD zero-fee loss. The frozen two-tick sensitivity
lost `0.600` pUSD, and the books' source timestamps were 10,049,940 ms apart.
The package was already gross-negative and outside the five-second skew gate,
so zero fee requests were made. This is not an accepted or deployment-ready
edge.

Do not resample Packers/Vikings. The reusable exact-depth runner now supports
nullable moneyline lines without changing prior frozen implementations. For a
distinct future NBA, WNBA, or NFL event, first prove every payoff relation and
use Gamma only as a rejection gate. Request one exact-depth batch only for a
strictly sub-floor displayed package, then require synchronized after-fee
positive recurrence before any order-capable work. Canonical metadata,
adjudication, and exact-depth result SHA-256 values are respectively
`8ebf70181290234c1c05f4659245d2c8c1502fd4a02eaa93dff9a4f60e375c6e`,
`c387e389d852ab5571056a9f2e80f91c63ae6f1c124ca55291b0fc787b5faeae`,
and `731ca32a06f8f1a42aaae9e326c2bd89379657e338231dd906b749790c15ddfa`.
The structural-edge registry retains 43 hypotheses and 20 narrowly accepted
scoped overlays; its new canonical SHA-256 is
`e712a9086d31944b42f93270256c393c6d8ab38997c20b7f8638cd4aa9088a34`.

## Future NFL Catalog Checkpoint

One frozen public unauthenticated keyset request screened the complete NFL
window from `2026-09-13T20:25:01Z` through `2026-09-21T23:59:59Z`, beginning
one second after the already-consumed Packers/Vikings event. The response
returned 17 events with no next cursor. The v1 runner exactly parsed 16 events,
retained all 4,621 monotone-payoff relations, and found 674 rows below their
guaranteed payout floor using Gamma prices solely as a rejection gate. The
precommitted strongest row was Commanders/Cowboys Over 56.5 plus Under 58.5:
Gamma displayed `0.285 + 0.500 = 0.785` against a one-pUSD floor.

One separately frozen exact two-token book batch tested only that row. The
five-share asks were `0.54` and `0.97`, costing `7.55` pUSD against the five-pUSD
floor, a `2.55` pUSD zero-fee loss. The two-tick sensitivity lost `2.75` pUSD,
the six-tick path lacked depth, and source timestamps were 25,189,367 ms apart.
Because the package was gross-negative and outside the five-second skew gate,
zero fee requests were made. No candidate was accepted or promoted.

The excluded Cowboys/Giants event exposed a v1 modeling inefficiency: its
half-half-tie moneyline and Cowboys -0.5 market share integer win threshold one
but differ at an actual tie. A frozen zero-network correction ordered the
moneyline as the statewise superset, retained 268 complete relations, and found
four additional Gamma-only candidates. Outcome-aware correction did not grant
adaptive depth authority, so no book or fee request followed. Do not resample
this window. Retry this family only on a distinct event outside it or a material
price, fee, delay, or resolution-rule change, and precommit the single strongest
package before exact depth.

Catalog contract/result SHA-256 values are
`3dc5413c76517eaf14c62d23b42fcd040c8f6f9f53b78c6e75f8a9f7e59de608` and
`7c4472e0a77cde09f5643a06a1326fbfc2cc1e5ec37641314d875a346e1a7754`.
Commanders/Cowboys contract/result SHA-256 values are
`0d6fba26dc1656c90e2cf78a0224e215c525364f201b8995c36439d391834292` and
`729d482f9a15b60b5345ba6c52ee75941a1f0751db2453e307c30f8872bbac35`.
Cowboys/Giants correction contract/result SHA-256 values are
`d481f24cd43703c4ed094631ebdbae8daa2588d92b7ecca93dc2aee4cd3195f0` and
`37f79cc8a4f5f96fa395a729e85a793e12c2127e2124591db693c92b1b459928`.

## Near-Expiry Fixed NegRisk Complete-Set Checkpoint

The old BTC/ETH/SOL structural-parity runner was not reused because it requests
books for every discovered event and predates durable preaccess journaling. A
new rejection-first contract instead froze one all-category keyset request for
events ending from `2026-08-29T18:41:26Z` through `2026-09-05T23:59:59Z`.
It requested the documented maximum `limit=500`, retained the raw response and
two-phase journal, and prohibited on-chain, book, fee, credential, account,
order, fund, or protected-capture access.

The service returned 100 events plus a cursor. Therefore the population is
explicitly incomplete even though the official contract documents 500 as the
maximum accepted limit. The partial page retained classifications for all 100
events, exactly screened 49 fixed non-augmented NegRisk events, and found 24
Gamma all-YES sums below the provisional one-pUSD complete-set floor. The
lowest displayed sum was `0.9450` across 17 Olympique Lyonnais/Le Havre exact-
score outcomes, an optimistic `0.0550` pUSD before execution costs. This is not
executable evidence: exact on-chain question-count and conversion-fee proof was
not requested, and Gamma prices remain rejection-only.

The frozen contract required `proof_candidate=null` whenever a cursor is
present. No adaptive pagination, on-chain query, CLOB book, or fee request was
made, and no edge was accepted. Do not continue this consumed cursor or depth-
test its outcome-aware best row. A future distinct study must prefreeze either a
narrow window that returns no cursor or a cursor budget independent of results,
then prove on-chain exhaustiveness before one precommitted all-token book batch.
Contract/result SHA-256 values are
`d5b81adb03fd4fe322d9a54fbacbe15aa8a6a7e55512aa71e9aa361617f2c6e6` and
`96610d7cba90a2dc97489bd70c95b7d03568d5b89017ace1e8c92829c70cee14`.

A prospectively distinct follow-up narrowed the population to the single
`2026-09-06` UTC end-date window, wholly outside the consumed range. Even that
one-day window returned exactly 100 events plus a cursor. It retained 57 fixed
NegRisk events and two Gamma-only sub-floor indications: Dallas/Sporting Kansas
City at `0.985` and Fluminense/Vasco da Gama at `0.990`. The contract again set
`proof_candidate=null`; zero on-chain, book, or fee requests were made. This
confirms that an all-category daily window is not prospectively complete under
the observed effective page size. Do not narrow repeatedly in response to page
results. The next distinct retry must precommit an hourly or series-specific
window, or an outcome-independent cursor budget, before its first request.
Follow-up contract/result SHA-256 values are
`18d513c0b54c6155897ae435cf9f4b8a0ef327f6072d39b122ebd4579b7f0972` and
`3e3ae8fd8c98c93c3e2194425db5992f06aed412e07d32333525601c2b34bc52`.

## RWUSD Reward-Retaining Futures Margin Reconciliation

A source-first audit reused the retained `2026-08-25` USD-M exchange-information
response instead of spending another market-data request. That response still
marks RWUSD `marginAvailable=true`. One exact public unauthenticated request then
retained Binance's official RWUSD Futures-margin announcement and a durable
journal. The announcement explicitly says RWUSD remains reward-bearing on the
margin balance in the USD-M Futures account, supports Multi-Assets and Portfolio
Margin modes, and published a 99.9% standard collateral value ratio plus
Portfolio Margin Pro tiers. It also says liquidation may trigger auto-exchange
and redemption requires transfer back to Spot for USDC.

The historical article's 4% rate is excluded. The separately source-bound
current public base APR is 3.36%, or 20.25205479 bips over 22 days. At the
published 99.9% ratio, assigning a labeled 10% annual alternative yield only to
the extra 0.1001001% principal leaves 3.34998998999% annual base yield. This is
accepted only as an extension of the existing reward-bearing Futures-margin
family: RWUSD must already be independently held, and the margin must already be
required by a separately justified after-cost USD-M Futures strategy. It does
not increase the accepted-edge count. Never subscribe, retain, transfer, redeem,
or trade to obtain this overlay; never double-count the existing VIP bonus; and
never treat the historical rate or haircut as exact current account evidence.

The frozen contract contains an invalid manually entered `19:58:00Z` timestamp.
It is preserved rather than rewritten. Filesystem evidence shows the contract
was created at `19:56:32Z` and hash-bound at `19:56:43Z`, before the single
request started at `19:57:18.202Z`; the result explicitly records this
provenance defect and treats the pass as deterministic primary-source semantics,
not a statistical outcome-sensitive fit. Canonical contract/result SHA-256
values are `9f2539c76c1f0f16619b0967fe26ae3cd1988447ee44a869c1d68e7eb3cf9f62`
and `e4f455511516babe956f4aa459648a032fb77f86c3253fd47d4f15317da72063`.
The structural-edge registry remains at 43 ranked hypotheses and 20 accepted
scoped families; its canonical SHA-256 is
`659904cc23e3d91c5d8622c9a8274e0227818d506724a72cce071df285eb681e`.

## Binance Indirect Organic-Conversion Route Savings

Two primary sources materially reopened a mechanism distinct from ordinary
closed-cycle arbitrage. The 2020 Binance empirical study found that traders
rebalancing from asset A to B through one intermediary obtained an exchange
ratio 14.4 bips better than the direct pair on average, while ordinary
triangular arbitrage remained fee-constrained. The 2026 working-paper metadata
reports network path optimization under conservative all-taker fees and
slippage across 64 Binance trading-pair groups during 2025, but proves only
comparative risk-adjusted outperformance rather than current absolute profit.

One frozen public unauthenticated screen therefore enumerated every current
TRADING Spot edge and all direct-versus-one-intermediary routes before viewing
books. Sixty complete all-symbol responses over 118.591 seconds covered 2,716
directed edges, 10,440 routes, and 20,568 route-size evaluations at 100 and
1,000 USDT. It charged a conservative 10-bip taker fee on every leg, exact lot
rounding, displayed top-level capacity, source and intermediary residual value,
and a further 3-bip extra-leg stress. Twenty route-size rows passed the frozen
recurrence gate.

The strongest headline rows were mostly stale fiat-quote displays. A separately
frozen one-request activity gate required every direct and indirect symbol to
change top-of-book state at least five times in the retained capture and report
at least 100 trades in the current rolling 24-hour response. It rejected 17 of
20 rows. Three diagnostics survived: USDC-to-PNUT through USDT at a 6.5881697492
bip median stressed saving, and SHIB-to-EUR through USDT or USDC at about 2.97
bips, all at 1,000 USDT displayed size. These are not static accepted routes or
fill claims.

The accepted edge is the general fail-closed cost-saving overlay only: when a
legitimate same-account Spot conversion is independently required, compare the
direct route with every one-intermediary route immediately before execution and
use an indirect route only when exact account fees, filters, fresh finite-size
depth, residuals, extra-leg latency stress, partial-completion unwind, and all
other costs remain strictly cheaper. Otherwise use the direct route or abstain.
Never create, split, reroute, self-match, or wash volume for this overlay, and
never treat public VIP-0 fees as account evidence. No account, order, balance,
credential, or protected capture was accessed.

The consumed runner omitted its run-level elapsed and request-latency gates from
the per-route candidate Boolean and sorted infeasible gross rows ahead of
feasible candidates in a diagnostic top-100 table. Both run-level gates passed
independently, so the omission did not change any candidate or decision. The
runner and raw evidence remain preserved without a repair capture; `AGENTS.md`
now requires every candidate Boolean to conjoin run- and row-level gates and
requires wide screens to rank only after feasibility gates.

A frozen zero-network extension then exhaustively compared every direct route
with exactly two intermediaries against the already accepted best direct-or-one-
intermediary comparator on the same 60 complete books. The population contained
1,064,216 routes and 2,128,432 route-size rows. A mathematically optimistic,
lossless rejection bound reduced exact evaluation to 16,051 rows; exact lot
rounding, displayed capacity, residual valuation, 10-bip-per-leg fees, 6-bip
three-leg stress, and the unchanged recurrence gates left 253 apparent
incremental candidates.

A separately frozen retained activity adjudication applied the existing five
top-book-change and 100-current-24-hour-trade floors to every required direct and
three-leg symbol. All 253 candidates failed the quote-change floor; 18 also
failed the trade-count floor. The 174-symbol population is therefore terminal
with zero survivors. Most large nominal gains depended on stale TRY books. Do
not build a live three-leg collector, treat any row as a static route, or spend
account or order access on this extension. Reopen exactly two intermediaries
only after a material fee, filter, activity, batching, or execution-architecture
change. The accepted direct-versus-exactly-one-intermediary overlay is unchanged.

The first extension implementation was intentionally stopped before output after
its one-object-per-route-size summaries reached about 847 MB. The second used the
lossless bound but failed closed before output because its survivor mask omitted
the already-required finite direct-to-USDT valuation for both intermediary
residuals. Both consumed attempts are preserved and may not be rerun. The third
added only that frozen source gate and completed. `AGENTS.md` now requires
topology cardinality and worst-case retained-output sizing before exhaustive path
work, followed by a frozen lossless bound and bounded diagnostics when large.
No network request, credential, account, order, mutation, or protected-capture
access occurred in any of these retained-data stages.

A subsequent one-request arXiv metadata delta was complete for papers published
from `2026-08-01T00:00:00Z`: eight papers appeared, one already known paper was
excluded, and all seven novel papers were adjudicated. Zero was an actionable
market-direction-independent edge. The Binance archive paper reported negative
net Sharpe; crypto mean reversion peaked near 1.3 bips gross against a 5-bip
round-trip benchmark and is directional; the other five covered security, AML,
cross-chain tracing, optimal-stopping theory, or consensus diagnostics without
an executable positive payoff.

The query also surfaced the older but previously unreconciled primary paper
`A Truckload of Satoshis` (`arXiv:2607.09491v1`). One frozen PDF request proved
that its OWA path is exactly two separate, non-atomic trades through one non-
anchor intermediary. The paper inferred 402 million Binance sequences from
anonymized settled trades ending in May 2023. Its 73.3 million USD gross and
31.2 million USD fee-adjusted estimates use a two-second pre-trade VWAP rather
than executable depth and assumed 1.2-bip maker and 2.4-bip taker fees. Mean
fee-adjusted profit was only 0.15 USD maker/taker and 0.04 USD taker/taker; the
paper could not identify exact fees, books, slippage, failed attempts, fixed
infrastructure, ownership, or unwind costs and found declining margins.

This source strengthens historical prevalence for the accepted fail-closed
direct-versus-one-intermediary organic-conversion overlay, but it is not a new
mechanism, a current profitability result, or evidence for the rejected exactly-
two-intermediary three-trade extension. It does not satisfy the literal collector
retry trigger. The literature query and PDF used two public read-only requests
total and no venue data, credential, account, order, mutation, or protected state.
Canonical literature result and paper adjudication SHA-256 values are
`21c830177ae1e17f18c941a5630df56a2c3dec5c0f26acd14ff740275fe29b06`
and `3f9684ed1986cd6cf676482069cda53846e336a15bc4b35141193b8e43406e65`.

Canonical contract, screen result, activity contract, and adjudication SHA-256
values are `dcd959d9d7ec0e1dffec47b910cece585c106b76b69feaaa9ca03ddcc7caa83a`,
`b3e7f724e2b1ce2cc7a8444d00632466196af6d89f139259164549e91d1f84bc`,
`b6331c1d7987b3a6093df2d4846368ccba634c27bf4967cd85f9c10e707fbd30`,
and `0307a9dbfb26ca62e94ae01e5b5d40316340b686a60829e85f258c07e565678c`.
The exactly-two-intermediary v1 resource adjudication, v2 source-gate
adjudication, v3 exact result, and final activity result SHA-256 values are
`be1bfdb40200f0e0acb26fcd1413a9c39e3ab34d61161550794fc429fe50cddd`,
`d23724ee0d088ac1965e96612557039a576f73724a1b630d2e434536ef0b2079`,
`0a5e37f2fb48c639334256e3118e3eeb2f17a548572faaf13d3849204404b45e`,
and `cde72e05b1760d9fe23eb65e5bd5f59377230ac91095354936c2a84a9a3758ae`.
The structural-edge registry now contains 44 ranked hypotheses and 21 accepted
scoped overlays; its canonical SHA-256 is
`ff5b41b572833ff0eed459098a2f93d1d62fed03891616b9a1fa71bc832f887e`.

## Future WNBA Complete-Catalog Rejection Screen

The rank-30 exact-payoff sports family had a literal distinct-event trigger, so
one frozen public keyset request covered WNBA starts strictly after the already
consumed Minnesota/Atlanta event: `2026-08-30T19:00:01Z` through
`2026-09-02T23:59:59Z`. The documented maximum 500-event limit returned three
events with no cursor, proving completeness under the frozen series and time
filters. The request was public and unauthenticated; it touched no books, fees,
credentials, accounts, orders, funds, or protected capture state.

Golden State/Portland and Connecticut/Dallas each exposed one machine-proved
moneyline/spread package with a one-pUSD payout floor over every integer final
margin and cancellation. Their rejection-only Gamma sums were `1.165` and
`1.365`; Los Angeles/Seattle had no exact monotone relation. Thus zero of two
relations was even optimistically sub-floor, and the frozen gate stopped before
all book and fee requests. Do not resample, narrow, or follow this population.
Only a future distinct NBA, WNBA, or NFL event outside every consumed window
with a machine-proved implication and a strict Gamma sub-floor package may
advance to one separately frozen depth batch.

Canonical contract and result SHA-256 values are
`36faeee7464832f335739ec8d1fc5609c98e1cdc9b6f267901934fbd8277f831`
and `fd0a9e844a7ad7d1a6eb5372c961ff82ea52d3c72a8c558ba191a53bace02cef`.
The raw response and two-phase journal SHA-256 values are
`1b989cf0b2edf79445c73ce616149c2696cff96c6f6dd7d89c310f770d4004e3`
and `40a7edc61fcf5c3a42a8ab7aef65cf04703cde30989b7b23f23a915291620368`.
The structural registry remains at 44 hypotheses and 21 accepted scoped edges;
its canonical SHA-256 is
`ff5b41b572833ff0eed459098a2f93d1d62fed03891616b9a1fa71bc832f887e`.

## Sep 7 Hour-00 Fixed NegRisk Complete Rejection Screen

The rank-31 retry condition prescribed a prospectively fixed hourly or
series-specific population, so one non-overlapping hour was frozen before
access: `2026-09-07T00:00:00Z` through `00:59:59Z`. One public unauthenticated
keyset request with the documented 500-event limit returned six events and no
cursor, proving completeness under the frozen time filter. Five events were
fixed non-augmented NegRisk sets; the Leagues Cup winner event was excluded by
the frozen classification rules.

Zero fixed sets had a Gamma all-YES sum below the one-pUSD payout floor. The
best visible upper-bound row, Nacional Potosi/Blooming match winner, summed to
`1.090` before any execution cost. The rejection-first gate therefore stopped
with zero onchain, book, fee, account, credential, order, transaction, fund, or
protected-capture requests. This is a terminal rejection for this population,
not an edge claim. Do not sweep adjacent hours, narrow around the observed
event, or adaptively continue any consumed page. Reopen only for a future
source-selected nonadjacent or series-specific population whose selection is
independent of these outcomes.

Canonical contract and result SHA-256 values are
`32dfa5e282f43204f73117fa2dba198c69171944bc41a85bf661e05789089439`
and `92734472ed41bccdc1d88c947b218e05fa35827cad6b1711ec192c06cf60cc64`.
The raw response and two-phase journal SHA-256 values are
`4b0322846de9fd229591c460eaf1cb22d4b0ac4c9e6b52ac2f35c8a0bec99442`
and `9349af5f5aee892fe5e16d710b55fefabef9ba63f36299b00bff4e1d90de8627`.

## Price-Blind Crypto Series Deployment Gate

The next rank-31 pass used the official series endpoint with
`exclude_events=true`, so selection saw no event, market, or price outcome. The
first recurring BTC/ETH/SOL row in server `volume24hr` order was series 10684,
`BTC Up or Down 5m`. Its prospectively frozen September 14 through September
21 keyset request returned zero events and no cursor. This was a clean null
deployment observation, not a market or edge rejection.

Retained price-blind metadata also identified the first structurally relevant
multi-strike row: series 45, `BTC Multi Strikes Weekly`. One separately frozen
series-specific request covered September 7 01:00 through September 21 and
also returned zero events without a cursor. Both stages stopped with zero
onchain, book, fee, credential, account, order, transaction, fund, or protected
capture access. No profitability or deployment claim exists.

The important correction is now executable policy: recurrence labels, titles,
update timestamps, and volume do not prove forward event deployment. Do not
repeat either empty query or immediately refine another undeployed horizon.
Reopen only if an official event deployment is independently visible, or not
before `2026-09-20T00:00:00Z` for one separately frozen nonoverlapping series-45
window covering September 22 through September 28. Canonical contract/result
SHA-256 pairs are
`a13f554d2b6b40dce3ea83236bff5cd99993f41aa1c988fe38eb99dc28d43f5e` /
`73ae75ccf391a30ef592f649f63ef535e29865e1234bbff03021c806fd75268b`
and
`ca2f7b556207a0d2d04c006ed502bf35a673a87642e0e5dd443227af8aba82d2` /
`f032753b45c82b2e0945d1a8c0e0d5fc01f8fb1727cdad34e73064c7590417ba`.

## Binance All-Symbol Triangular-Cycle Terminal Adjudication

The existing retained complete Binance Spot graph was reused in one frozen
zero-network rejection pass, avoiding another public capture. It enumerated
all 3,480 unique directed three-asset cycles from 2,716 directed edges; cyclic
rotations were deduplicated while reverse directions remained distinct. The
frozen quote-change and 24-hour trade-count gates retained 1,442 cycles. Across
60 complete book responses, 100 and 1,000 USDT starting sizes, and both a
zero-fee upper bound and conservative VIP-0 all-taker fees, 5,648 evaluable
cycle-size-fee rows received exact sequential lot rounding, displayed top-book
capacity checks, residual USDT valuation, and 3 bips of operational stress.

Zero rows passed the recurring candidate gate even with zero fees; VIP-0 also
had zero candidates. This is therefore terminal for the current retained
population, not a profitable edge. Do not repeat these books or spend account
queries on this family absent a material Binance Spot fee, filter, order-batch,
atomic-execution, or market-structure change that first reopens the gross upper
bound. This strengthens the accepted indirect-routing result's boundary: the
cost-saving organic-conversion overlay remains accepted, while standalone
closed-cycle arbitrage does not.

The consumed runner's `top_100_feasible_first` diagnostic label is inaccurate:
it sorted candidates and then median economics but did not separately put
capacity-feasible non-candidates ahead of infeasible rows. Candidate booleans
and both zero counts were computed before sorting, so the defect changes no
decision. The frozen runner and source result remain preserved, and no adaptive
rerun was performed. Canonical contract, source-result, and adjudication hashes
are `cc19bdd97167265e0831f84325624a623f83f2dc2d939491ed1c51859c0a38bb`,
`30c5e00aa955ea3777f9b096b1fa1ae44d51318665561e4b6922f797f45706cc`,
and `2fffd2044e72d1712ecdaa0c4e24cb829057ea2005c07e12129c443478b07902`.

## Binance RPI Maker-Hedge Source Gate

A distinct USD-M Retail Price Improvement execution lead was source-gated
before any book request. The retained native endpoint index identifies
`GET /fapi/v1/commissionRate` as `USER_DATA` and names
`GET /fapi/v1/rpiDepth` as the RPI order book, but it does not provide the
account's exact RPI commission, an owned fill, or after-cost economics. The
frozen current primary documentation GET then returned HTTP 202 with an empty
body. Its required rendered semantics were therefore not source-bound, the
second documentation request was not sent, and zero RPI-depth, ordinary-book,
commission, credential, account, order, mutation, or protected-capture requests
were made.

Do not retry that dynamic page or switch aliases. RPI remains an unaccepted
maker-first hedge lead with a zero public profit floor. Reopen only on a
material official byte-retainable RPI fee or execution contract, or when both
designated credentials and explicit signed GET-only commission authority exist
for an independently positive organic equal-base hedge question. Every RPI or
hedge order still requires separate trade authority. Canonical failure
adjudication:
`docs/model-research/action-value/binance-rpi-maker-hedge-source-failure-adjudication-v1-2026-08-30.json`,
result SHA-256
`82245f341e23ab2e8c8e9e3bd4d47805e88aebc3b39f6ac1a6360067491be7ef`.

The methodology correction is durable: a browser-renderable dynamic docs route
is not presumed byte-retainable. Use an already hash-bound native index or
preflight exact response bytes before freezing a source contract, and preserve
HTTP 202 with zero bytes as a consumed null response.

The structural registry remains at 44 ranked hypotheses and 21 accepted scoped
overlays; its terminal source-capture entries now total 52.

## Accepted Market-Independent Yield Frontier

A zero-network portfolio audit now ranks all nine accepted yield and capital-
efficiency overlays separately from the thirteen accepted fee, referral,
creator, financing-cost, and organic-flow overlays. It does not add an edge or
loosen any gate. All nine
remain existing-balance or already-required-capital overlays; none authorizes
asset acquisition, borrowing, retention, subscription, transfer, conversion,
or trading, and none is deployment-ready.

The strongest stable edge remains Polymarket complete-set holding yield because
it is the only row with owned recurrent cash payments, 42 positive daily
payments out of 42 possible across BTC, ETH, and SOL, a demonstrated 1,039 pUSD
principal, direct split/merge cost identity, and positive retained economics
through a 3% alternative annual yield. Its realized portfolio rate was
3.1820191118%; the retained economics turn negative at a 3.25% alternative
before external friction. The strongest long-history Binance overlay remains
LDUSDT yield on collateral already required by an independently justified
futures strategy: 505 daily closes over 504 days, 2.0109050595% compound
annualized appreciation, and positive retained haircut sensitivities. The
strongest source-bound current fixed-bonus allocation is the 7% USD1 Simple
Earn bonus on at most the first 1,500 already-held idle USD1 through the
published 2026-09-25 end, subject to exact account eligibility and capacity.

The fixed-bonus account-prequalification urgency order, if designated
credentials and explicit read-only authority later exist, is USDT through
September 7, U through September 14, RWUSD VIP bonus through September 17, and
USD1 Simple Earn through September 25. This is an evidence-routing order, not
an instruction to acquire, subscribe, or move principal. Canonical frontier:
`docs/model-research/action-value/accepted-market-independent-yield-frontier-v1-2026-08-30.json`,
result SHA-256
`7a7fa5ed15ab63bfd0c4d5d2ce65888391a72c4e73eea69e7f7c1fcf01a13fb8`.

## pUSD-to-USDT Fixed-Bonus Opportunity-Cost Rejection

One frozen public sequence used the hash-bound official native Markdown index,
retained the current quote and supported-assets contracts, made one current
supported-assets GET, and then one exact 500 pUSD to Polygon USDT quote. All
four requests were public and unauthenticated. The quote estimated 492.459811
USDT output: a 7.540189-USDT or 150.80378-bip optimistic one-way loss.

The maximally favorable remaining 4% fixed bonus through
`2026-09-07T23:59:59Z` is only 0.4856127333 USDT or 9.712254665 bips. Charging
the realized 3.1820191118% pUSD holding-yield opportunity cost leaves only
0.0993054837 or 1.986109674 bips of incremental headroom. The one-way quote
loss is 15.527 times the entire fixed bonus and 75.929 times the incremental
advantage before any Binance deposit, return conversion, eligibility, capacity,
custody, tax, or operating cost. The cross-venue acquisition route is rejected
before account access.

This does not reject the accepted USDT bonus overlay for independently already-
held idle eligible USDT. Do not move pUSD for this promotion or repeat the quote
without a material bridge-quote or bonus-term change. No credential, account,
deposit or withdrawal address, order, subscription, approval, transfer,
transaction, fund, or protected-capture state was accessed. Contract and result
SHA-256 values are
`1098e4af17f5269ae1e233b3bcf1fa339f0da45eb8b87a2b3fa327e6ca6fd011`
and
`98b74abfcb213a8d1bd554fc1bfec9044d6a6a3990abd800bef92f29437533b3`.
The accepted-edge count remains 21, ranked count remains 44, and registry
SHA-256 is
`8ef8e033e169084a57237321d13467f05486152c2b713cc478023372efc6b877`.

## Binance Spot Amend-Keep-Priority Candidate

One frozen official FAQ GET returned HTTP 200 with 4,705 bytes and was durably
retained. The run failed because two required raw-Markdown phrases did not
predeclare formatting normalization: the source inserted bold delimiters around
one clause and backticks around `amendAllowed`. The request is consumed and was
not retried. A separate zero-network adjudication removes only those exact
formatting delimiters and preserves the failed contract and journal.

The retained source proves that reducing an existing order quantity in place
keeps its same-price time priority while cancel-replace loses priority and
executes behind existing same-price orders. A failed amendment leaves the order
unchanged; the amend adds zero to the unfilled-order count and has request
weight four. The already retained complete production exchangeInfo snapshot
contains 3,685 symbols and independently has BTCUSDT, ETHUSDT, and SOLUSDT in
`TRADING` state with `amendAllowed=true` and a ten-amend filter for each. No new
exchangeInfo or book request was made.

This is a material direction-independent execution candidate, not an accepted
or deployment-ready edge. Queue priority is only a weak dominance identity for
the same existing order, price, and reduced quantity. Public evidence proves no
owned organic reduction, success acknowledgement, counterfactual incremental
fill, adverse selection, latency, commission, or after-cost cash value; the
public profit floor is zero. The endpoint is `TRADE`. Advance only with an
independently required existing simple scoped maker-order reduction and
separate explicit testnet or paper order authority, or after a material source
or configuration change. Contract and adjudication SHA-256 values are
`020d741a6d19bc4c6038a3fedcbea5fdb0cbd01c433c62127453c7892a144c38`
and
`c17217ff011d6ff48b4c0cf48cc6c8e49c27319c811be407cadebdd2e8d7faeb`.
Accepted edges remain 21, ranked hypotheses remain 44, and registry SHA-256 is
`e9e25be7ae77d25c2f98b30b734c64f85259e73059405e938be77c776bd0a066`.

## Verification Scope

The amend-keep-priority contract, consumed one-request failure journal, exact
raw source, formatting-normalized offline adjudication, retained 17.5-MB
production configuration identity, three exact scoped symbol rows, rank-five
lineage, and reusable Markdown-gate correction pass three direct tests. The
prior RPI family test now checks exact artifact membership rather than mutable
tail position. All 326 registry-coupled tests pass in one focused run. Ruff is
clean for the changed Python scope. The staged-blob audit also reconstructs the
one CRLF worktree source validated before access from its durable LF Git blob;
no economic field changes. No FAQ or exchangeInfo request was repeated, and no
book, credential, account, amendment, cancellation, replacement, order, fund,
or protected-capture workflow ran.

The pUSD-to-USDT contract, four pre-request journal rows, two native Markdown
sources, exact current supported-assets population, quote response, conversion
and reward arithmetic, rank-one lineage, and terminal no-repeat entry pass the
two direct source-bound tests. All 323 registry-coupled tests pass in one
focused run. Ruff and Python compilation are clean for the changed Python
scope, the current Polymarket publication manifest still reconstructs, and
`git diff --check` is clean. No quote was repeated, and no broad model, CI,
hosted, release, credential, account, address-generation, order, subscription,
transfer, transaction, fund, or protected-capture workflow ran.

The price-blind series selector, both empty event catalogs, four canonical
contract/result hashes, three retained raw-response hashes, journals, updated
rank-31 routing, and the new deployment-evidence instruction pass four direct
source-bound tests. The registry hash change covered 270 focused tests across
70 registry-coupled files in one pass: 269 passed and one stale six-artifact
tail expectation failed; after extending that exact expectation to the four new
artifacts, the corrected test plus the four direct tests pass. Ruff is clean
for the new tool and two directly affected tests. No market request was repeated
for verification, and no broad model, CI, hosted, release, account, order, fund,
or protected-capture workflow ran.

The future-WNBA and Sep 7 hour-00 contracts, compact complete-page results,
two-phase request journals, raw responses, exact payoff and fixed-NegRisk
screens, rank-30/rank-31 lineage, and all registry-hash-coupled checks pass 270
focused tests across 70 files in one combined pass. Ruff and Python compilation
are clean for the two directly affected tests. The hourly result reconstructs
its contract, canonical result, raw response, and journal hashes exactly. No
onchain, book, fee, credential, account, order, fund, protected-capture, broad
model, CI, or release workflow was used.

The frozen zero-network all-symbol triangular-cycle contract, deterministic
retained-data runner, source result, defect-disclosing terminal adjudication,
updated rank-16 lineage, and all registry-hash-coupled checks pass 267 focused
tests across 69 files; one stale terminal-family count was updated and its
exact test rerun passed. Ruff and Python compilation are clean for the changed
Python scope. No market request, credential, account, order, protected capture,
broad model suite, CI, or release workflow was used.

The two-request holding-yield continuity monitor, its frozen contract, exact
retained-row selection, preaccess request-body journals, two raw Polygon
receipts, distributor-transfer reconciliation, and updated rank-one registry
lineage pass their focused tests. All registry-hash-coupled tests pass with
Ruff and Python compilation clean. The accepted-edge count remains 19 and no
current-rate, account, funded, or deployment gate was opened.

The two current WNBA metadata captures, payoff proofs, synchronized Toronto-
Phoenix book screen, Lynx-Dream rejection-only stop, request journals, raw-
response hashes, reusable exact-event runner, and registry lineage pass their
focused tests with Ruff and Python compilation clean. The rejection-only
prefilter is now a hard efficiency gate for this family.

The Packers-Vikings exact capture, future NFL catalog, Commanders/Cowboys depth
screen, Cowboys/Giants tie-state correction, near-expiry fixed NegRisk partial
catalog, complete retained payoff proofs, raw journals, and registry lineage
pass the focused structural checks. The registry-hash-coupled suite passes all
252 focused tests across 66 files. Ruff is clean for the changed tools and
directly affected tests. No broad model, CI, or release suite was repeated.

The resolved-leg House duplicate-payoff checkpoint, raw journals, source hashes,
and directly affected registry lineage pass 12 focused tests with Ruff and
Python compilation clean. The registry still contains 43 ranked hypotheses and
20 accepted scoped overlays; its result SHA-256 is
`e712a9086d31944b42f93270256c393c6d8ab38997c20b7f8638cd4aa9088a34`.

The exact-one-NO failure preservation and Binance BLVT current-inventory gate,
including every registry-hash-coupled test, pass 199 focused tests with Ruff
and Python compilation clean. Both new result hashes, both frozen contracts,
their implementation hashes, retained raw responses, and request journals
reconstruct exactly; the accepted-edge count remains 19.

The Binance-option/Polymarket-threshold gate and registry propagation pass 23
focused tests with Ruff clean; both changed JSON result hashes reconstruct.
The Binance option/perpetual conversion v1 preservation, v2 timestamp-semantics
repair, complete 71-row retained stress, exact quantity lattice, source hashes,
and terminal registry routing pass their focused tests without network access.
The focused Round 75 closeout passes 20 tests. The cross-regime promotion change
passes 62 affected tests before final publication checks. Run the smallest
affected checks during development, then full CI once before publication. Do
not repeat unchanged expensive suites between adjacent edits.

The Polymarket/Binance TradFi-perpetual prefilter, bounded history, retained
shortfall adjudication, all six canonical contracts/results, fixed orientation,
Regular-plus-Special cash-flow aggregation, source and implementation hashes,
terminal registry routing, and protected-capture non-access pass 11 directly
focused tests. The single registry-hash-coupled pass covers 260 tests across 67
files with Ruff and Python compilation clean. No broad model, CI, hosted,
release, or repetitive market-data suite was run.

The RWUSD deterministic primary-source reconciliation, retained raw response and
journal hashes, current margin-inventory reuse, current-rate isolation,
nonduplicative reward-bearing-margin family routing, and explicit provenance
correction pass the one directly affected test plus the registry-hash-coupled
suite: 261 tests across the same 67 files. Ruff and Python compilation are clean
for the directly changed test, every canonical JSON hash reconstructs, and the
retained raw response reconstructs its journal-bound payload hash. No broad
model, CI, hosted, release, or repetitive market-data suite was run.

The indirect-conversion screen, activity gate, source hashes, exact rounding,
fee, capacity, residual, recurrence, and registry lineage cover 264 focused
tests across 68 registry-coupled files. The combined pass found only four stale
registry-length expectations; after their exact metadata correction, those four
failures pass without repeating the already-passing 260 tests. Ruff is clean
across all 70 changed Python files, Python compilation is clean for the frozen
runner, all five canonical JSON hashes reconstruct, and the current activity
response hash matches its adjudication. No broad model, CI, hosted, release, or
repetitive market-data suite was run.

The previous verbose handoff and chronology are preserved byte-for-byte in:

- `docs/archive/agent-history/AGENT_START-before-2026-08-23-closeout.md.txt`
  (SHA-256 `2ba0ee28f38a9f5d2a177cf4b270fe924517e88f6a9511dd7acb3507ab7907c5`)
- `docs/archive/agent-history/CONTINUATION-before-2026-08-23-closeout.md.txt`
  (SHA-256 `2170f14bcfdf49674c576b8fd7d42aa02dc4569c48ba1f643ec6ad43c8d30b18`)

## 2026-08-29 Binance Spot SOR configuration result

The novel native-SOR liquidity-overlay hypothesis was frozen before outcome
access, then rejected at its cheapest live gate. The single public production
`exchangeInfo` response was HTTP 200 and contained 3,685 symbols, but omitted
the optional `sors` configuration field documented by Binance. The runner
stopped immediately: zero scoped BTC/ETH/SOL groups, zero `bookTicker`
requests, zero candidates, and no credential, signed, account, order, fund, or
protected-capture access.

The exact result is terminal until a material official or live SOR
configuration change; documented feature availability alone is not a retry
trigger. Git retains the 17,532,885-byte raw response as deterministic lossless
gzip rather than a wasteful uncompressed blob. The decompressed SHA-256
`658d03279eea9a2384171eb56e151541407ae7407c123c700c9902d5e9f56c9d`
reconstructs the journal receipt; the gzip SHA-256 is
`9f3f83fe1efcec7a0230dea646d2038fe5c0a32ae72de5c7e4f5ee7fd850304b`.
The accepted-edge count remains 21 and the ranked-hypothesis count remains 44.
The updated registry result SHA-256 is
`ff5b41b572833ff0eed459098a2f93d1d62fed03891616b9a1fa71bc832f887e`.
The direct SOR evidence test and the one registry-coupled pass are clean; no
broad CI, model, release, or repeated market request was run.

## 2026-08-29 Polymarket official-documentation novelty gate

One frozen public unauthenticated GET retained the current official
`https://docs.polymarket.com/llms.txt` index and a pre-request intent journal.
The response was HTTP 200, 13,746 bytes, and contained 80 top-level English
Markdown pages. The gate spent exactly its one allowed request and stopped
before every linked page, market-data endpoint, credential, account, order,
mutation, fund, and protected-capture path.

The offline registry diff found zero distinct structural mechanism. Current
reward, fee, resolution, negative-risk, position, Combo, and Perps pages map to
existing ranked families. Session keys and matching-engine restart state are
execution controls. Bridge routes add quoted conversion fees and delay but no
documented discrepancy, rebate, yield, or settlement right. Mark price, index
price, margin, liquidation, and market sessions are valuation and risk
constraints. Combo collateral return is not novel: its exact source was already
retained in rank 33's canonical Combo evidence.

This snapshot is terminal and authorizes zero linked-page or market-data tests.
Do not repeat it until the official index SHA changes from
`68256fa9849e72626806cbc7373f726421fd6d62dddecc0ae3a8009595bd2b8d`
and an offline title or description diff identifies a genuinely new cash flow,
cost reduction, settlement right, or executable package. Contract and result
SHA-256 values are
`74932c7a4a14217d2dbf419203ae252cd8666cfae52b58f8ea59613d4adb8d3d`
and `e56ce8f2a491d6da3f66b0d085381894ec0e7c078e4f4cfaf238d8e044fa281a`.
The accepted-edge count remains 21, the ranked-hypothesis count remains 44,
and the updated registry SHA-256 is
`ff5b41b572833ff0eed459098a2f93d1d62fed03891616b9a1fa71bc832f887e`.

## 2026-08-29 Polymarket Combo collateral-release overlay

One frozen public source-only GET retained the official Combo collateral-return
page and a pre-request journal. The 44,910-byte response exactly matches the
payload SHA retained on 2026-08-27, so no changed documentation or adaptive
second request was needed. No Combo catalog, quote, book, fee, account, plan,
credential, order, transaction, fund, or protected-capture path was accessed.

The source proves the mechanism's exact economic boundary: compatible
offsetting Combo positions are decomposed, complementary exposure merges into
pUSD, and unmatched exposure remains in residual positions that preserve the
wallet's remaining economic exposure. This makes earlier collateral release a
direction-independent capital-efficiency candidate for positions that already
exist independently. It is routed into rank 33 with the existing idle-pUSD or
other exact opportunity-cost comparator, not added as a duplicate family.

The candidate is not accepted or deployment-ready. Released pUSD is principal,
not profit. The public source does not prove owned compatible inventory, exact
`net_pusd_out`, the monetary unit or payer of `estimated_cost`, a universal
zero-fee or gasless guarantee, execution success, or an independently available
positive use of the released collateral. The zero-network gate is:
`net_pusd_out * after_cost_annual_rate * usable_days / 365` must exceed every
approval, execution, retry, truncation, residual-position, and operating cost.
At a labeled 3.25% sensitivity, the maximum total cost is 2.6712328767 pUSD per
1,000 released for 30 days and 8.0136986301 pUSD for 90 days.

Reopen only with independently existing owned compatible positions, explicit
account-specific plan-request-only authority, and an exact positive use of
released pUSD. Request one plan without execution and fail closed unless it has
positive `net_pusd_out`, zero `required_pusd_input`, complete residual lineage,
bounded chunks, exact monetary costs, and strict positive conservative value.
Every approval, signature, submission, transaction, poll, or retry requires
separate authority. Contract/result SHA-256 values are
`3dcc72146293b5b76f078173392a7ec6dd14c6c3b4af2fef8110e1f310c55a59`
and `5514bd931557b350579a07448db9c4e1f2664919efff48145861c8841f0bc7ea`.
The accepted count remains 21, ranked count remains 44, and registry SHA-256 is
`ff5b41b572833ff0eed459098a2f93d1d62fed03891616b9a1fa71bc832f887e`.

## 2026-08-29 Polymarket organic Relayer gas-subsidy overlay

One frozen public unauthenticated GET retained the current official Builder-tier
page with a durable pre-request journal. The source proves that gas fees are
subsidized for supported smart-wallet Relayer operations. The Unverified tier
currently permits 100 Relayer transactions per day, Verified permits 10,000,
Partner is unlimited, and the FAQ describes an unlimited own-wallet Relayer-key
route. No market data, credential, account, order, profile, key, signature,
transaction, fund, or protected-capture state was accessed.

The narrow candidate value is only the exact avoided user-paid gas cost for an
independently required, otherwise-positive supported Deposit, Safe, or Proxy
wallet operation within the active account tier or own-wallet limit. Zero
activity has zero value. Never create, enlarge, split, retry, or reroute an
operation to consume sponsored gas, and never double-count the same gas line
with builder fees, holding yield, rebates, or another mechanism. The candidate is
not accepted, a standalone-profit claim, or deployment-ready: the frozen source
contract forbids source-only promotion, and exact active-key state,
wallet support, remaining capacity, an owned successful receipt, counterfactual
same-action gas, and all setup and operating costs remain account-gated.

Weekly USDC rewards and grants remain unaccepted with a zero public forward
floor. Both are subject to approval and the page publishes no deterministic
rate, threshold, allocation formula, cap, timing, or current owned award. The
same applies to rate limits, support, marketing, and priority access: they are
not cash edges. Reopen those candidates only on a deterministic published
formula or an account-confirmed approved award under explicit read-only evidence
authority. Contract/result SHA-256 values are
`87ff005cd29184501aa9bb17d450a1112f0f8324fe9ad7bd248c375f30b7698f`
and `1c33e778d217ec5e7ef817e83af3186df7da0e5dd0cc75cb72464bfd97d18d49`.
The accepted count remains 21, ranked count remains 44, and registry SHA-256 is
`0a34d7289331515f8e7b3f09e856fbc331ecbc3a91130fea20542a39ef211f60`.

## 2026-08-29 Binance Stocks fee-extension source failure

The rank-5 official term-change trigger fired when Binance published a Stocks
fee-promotion extension through `2026-09-30T23:59:00Z`. The frozen one-request
CMS capture was consumed, but curl exited `23` because its raw-output parent did
not exist. No response body or HTTP status was durably retained, so the existing
accepted overlay's earlier duration remains unchanged and the accepted-edge
count remains 21. Do not retry the exact CMS request or repair it through an
endpoint alias. Reopen only on a materially distinct official primary source or
a later official fee change. Contract/result SHA-256 values are
`7747533e1014272e7e95a252c0f1d3dac76af0bc12cf27b52ea7c3c16f73fc2d`
and `498003b3f593cda600570099f9089bcb0db0a189e5c92c87e9395bd2afeb3ed8`.

The failure exposed a reusable efficiency defect. Every one-use file-backed
runner must now create and verify all raw and journal parents before HTTP; a
post-response local write error consumes the request and may not be hidden by a
second filename or endpoint alias.

## 2026-08-29 Round 21 Binance sidecar terminal failure

The protected boundary passed and the former process IDs are absent. The
campaign lock accepted an exclusive read; the actual 17,620,807,680-byte
database was located at the legacy sidecar path, had no WAL, and was not opened.
All 17 segment receipts were reconciled from metadata: 16 are interrupted and
the final segment failed in `finish_run` with a DuckDB memory-limit allocation
error. Across them, 882,811,373 raw messages were recorded, but there are zero
complete or degraded eligible segments. The contract-defined terminal-manifest
command correctly rejected the campaign with `no eligible segment`.

The campaign is terminally failed, source continuity did not pass, and no
payload, outcome, model, profitability, account, credential, or order authority
was admitted. Do not rerun or reuse its database, schedule, or failed lineage.
Proceed only through the already frozen venue-separated prospective
source-continuity recovery design. Canonical terminal-failure SHA-256 is
`9e6790644a566dcfd6e786442a8da3a63c8837f991e766b418fea0df90d0cc8e`.

## 2026-08-30 Binance Spot Block Matching cost overlay

Current official sources prove a materially distinct direction-independent
execution path for an independently required large bilateral Spot trade.
Whitelisted master accounts can trade directly off the public order book;
settlement is immediate to the master Spot account. The current CMS FAQ, last
updated `2026-07-20T07:50:00Z`, charges both maker and taker `0.025%` (`2.5`
bips), provides no market-maker rebate, gives a block request a default
30-minute validity, and permits creation up to 10% above or below the current
market price. The current Agent Native API index classifies symbols and order
history as signed `USER_DATA`, while place, take, cancel, and extend are
`TRADE` operations.

The fetched general-information page embedded credential-shaped authentication
examples, so its raw body is intentionally excluded from version control under
the repository's stricter credential-hygiene rule. Its original response hash
remains in the request journal and candidate retention exception, but none of
its example values are reproduced or source-bound. The current FAQ,
introduction, and Agent Native index independently support every admitted term.

This is a material rank-5 cost-reduction candidate, not an accepted or stable
edge. At 100,000 quote notional, the fee-only saving versus counterfactual Spot
fees of 10, 7.5, and 5 bips is 75, 50, and 25 quote units; versus 2.5 bips it is
zero, and versus a zero-fee pair it loses 25. Those are rejection-first
sensitivities only. The permitted price band is much wider than every fee
saving, and public sources prove no whitelist, supported pair or minimum,
authentic counterparty, negotiated price, same-time finite-size book,
account-specific Spot fee, fee rounding, failure cost, or owned fill. The
public forward profit floor remains zero.

Do not contact VIP coverage or a counterparty, request whitelisting, relay a
settlement key, or place, take, cancel, or extend an order. Reopen only when an
independently planned legitimate large bilateral master-account Spot trade,
existing whitelist, authentic counterparty, exact pair/quantity/side/time, both
designated ephemeral credentials, and explicit signed read-only authority all
exist, or after a material official term change. Prequalify symbols, minimum,
and exact counterfactual account fee once, then reject before any order unless
the same-quantity block path is strictly cheaper than fresh finite-size public
book execution after every cost. Every order action requires separate explicit
trade authority. Canonical candidate SHA-256 is
`2d9c4872a6ecd707716cd8d769eb20cb715c1ed7feb61e7e560a5efdb169dc57`.

The earlier low-value-asset documentation request was separately consumed by a
CloudFront WAF challenge (`HTTP 202`, zero body bytes). It source-binds no
target, fee, balance, quote, or positive economics and may not be retried or
repaired through an alias. Canonical failure SHA-256 is
`e2fc005dda76af1a6aad7eb29ca09db19023f0aaa5ae64eff944cc0ef75ee48a`.
The accepted-edge count remains 21, the ranked-hypothesis count remains 44, and
the updated registry SHA-256 is
`0416a75158adf12ca08e6dc2d529efa29db53300715ba06f7c04378dcfa2a396`.

## 2026-08-30 Binance public Spot block-trade price preflight

Official Binance Spot documentation proves a production, real-time
`<symbol>@blockTrade` stream with event time, trade time, symbol, exact price,
quantity, block-trade ID, and buyer-maker identity. The official
market-data-only guide explicitly makes those streams available without
authentication on `data-stream.binance.vision`; the changelog scheduled the
production rollout for `2026-05-12T07:00:00Z`.

A frozen one-use connection observed only `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`
block-trade and ticker streams for 1,205.0309999999954 seconds, with no
reconnects. The complete raw response retained 1,188 BTC, 1,122 ETH, and 1,058
SOL ticker messages, proving the connection and ordinary market-data flow were
live, but it contained zero block-trade events. Therefore there was no exact
price, quantity, side, causal pre-trade quote join, or economic row. The frozen
gate requiring three recurrent same-symbol-and-side observations with a
positive price-concession lower bound failed.

This result neither proves the stream has no future capacity nor disproves the
account-gated Block Matching mechanism. It rejects only an immediate repeated
public capture under unchanged conditions. Do not repeat or expand the public
preflight before `2026-08-31T00:28:14.5427794Z`, unless a material official
stream, fee, pair, whitelist, settlement, or block-volume change occurs. The
public forward profit floor remains zero; every authenticated or trade-capable
path retains its existing authority gates. Canonical preflight SHA-256 is
`b7d60e0d9f3e30b2a62663ff1290be77e6309ac33a7d48776b6f5ea1c8dcfe68`.
The accepted-edge count remains 21, the ranked-hypothesis count remains 44, and
the registry SHA-256 is
`6e0c9d33e909ec980af5fa65d8ed2cdaebd8dd3fa576671165fe0a331f7af817`.

## 2026-08-30 Binance Convert Limit Order Simple Earn overlay

The current official FAQ proves a distinct direction-independent
capital-efficiency mechanism. For an independently required Convert Limit
Order, the user may elect to place the source asset in Simple Earn Flexible
instead of freezing it without rewards in Spot. Successful enrollment leaves
the funds in Simple Earn until execution or a separate redemption after
cancellation or expiry. Cancellation alone does not redeem them.

This benefit carries a directly coupled execution risk. Personal subscription
and daily redemption quotas apply; delayed or failed redemption prevents the
order from executing, and an unavailable or already redeemed asset causes
automatic cancellation. Reaching the displayed limit price does not guarantee
a fill because Convert quote price, depth, liquidity, and immediate asset
availability still govern execution. Regional eligibility also varies.

The candidate remains unaccepted with a zero public forward reward floor. The
FAQ publishes no exact eligible BTC/ETH/SOL rate, tier, cap, minimum, accrual
start, distribution timing, account quota, or guaranteed redemption. At the
retained BTC `0.27% Max` sensitivity, one day earns only `0.0739726` bips and
seven days `0.5178082` bips before costs; one bip of missed-fill shortfall takes
`13.5185` days to recover. At the displayed `0.02%` base component, recovery
takes `182.5` days. These are rejection-only sensitivities, not account rates.

Advance only when an exact BTC, ETH, or SOL Convert Limit Order is independently
required and its pair, source amount, limit price, maximum wait, and missed-fill
stress are frozen, both designated credentials exist, and signed read-only
authority is explicit. Reject unless the exact conservative account reward
strictly exceeds every subscription, redemption, delay, missed-fill, tax,
custody, alternative-yield, and operating cost. Creating the order, checking
the Earn box, subscribing, redeeming, canceling, transferring, or trading each
requires separate explicit authority. Canonical candidate SHA-256 is
`3b33ca4ef8c03a609bef1665ccfc2104a3f6585033770f0bb99ec3c5699949f8`.
The accepted-edge count remains 21, the ranked-hypothesis count remains 44, and
the registry SHA-256 is
`546904123e0985aa23d7f3c58567dd2b8e877681b48e560512e4b15b9082721b`.

## 2026-08-30 Exact Polymarket Reward Candidate Rejection

The rank-17 distinct exact-market retry trigger fired for the active Elon Musk
40-64 posts market. The first frozen attempt retained one exact Gamma response
and one exact `sponsored=true` reward response, but its discovery-value equality
gate was wrong: the exact sources agreed on 50 shares and 5.5 cents, while the
104 pUSD/20-share/4-cent search tuple was stale or misattributed discovery.
A separately frozen correction reused those retained bytes and made only the
previously unrequested book call. The response failed freshness at 6,408 ms;
one-tick 0.48 and 0.53 bids were both marketable, summed to 1.01, and would lose
0.50 pUSD if both filled before reward uncertainty. No retry is admissible.

No account, credential, order, cancellation, fund, or protected-capture state
was accessed. Publicly proven payout remains zero and the accepted-edge count
remains 21. Continue with the highest-ranked hypothesis whose literal trigger
is satisfied; do not repeat this market. The zero-network best-bid alternative
also failed: 0.50 pUSD both-fill gross versus 26 pUSD orphan loss needed 43.554
reward days under observed competition and 4,306.850 under 100-times stress,
against 3.693 days remaining. Canonical rejection SHA-256 is
`facecfaa3b92d905c700083c7b8afe153adc495403ceabc91e417bdb248d059b`;
registry SHA-256 is
`de28d80cc4b0b9cd1bd3f9954cb840dcaefe46fcc0fdf9ac3fd53218169370cb`.

## 2026-08-30 Binance public Spot block-trade follow-up

Rank 5's literal nonoverlapping public retry boundary passed. One separately
frozen five-minute connection reused the existing hash-bound official stream,
market-data-only, changelog, and 2.5-bip fee evidence. It made no new
documentation GET, signed request, account query, book request, order, contact,
settlement action, or fund movement.

The transport completed with no reconnects after 305.0 elapsed seconds and
retained 292 BTCUSDT, 233 ETHUSDT, and 207 SOLUSDT ticker messages. It contained
zero `blockTrade` events, zero analyzable rows, and zero positive price-
concession lower bounds. Cumulatively, both complete nonoverlapping observations
span 1,510.0309999999954 seconds and 4,100 healthy ticker messages with zero
public block events. That evidence does not prove private Block Matching
capacity or future activity is zero, but it cannot support recurrence, access,
price concession, profitability, or stability. The public forward profit floor
remains zero.

Time-only daily polling is now rejected as an inefficient unchanged question.
Do not repeat before `2026-09-06T03:47:16.3134381Z` unless a material official
stream, fee, pair, whitelist, settlement, block-volume, or observed public-event
change occurs. The account-gated parent may still advance only under its full
independently planned trade, whitelist, counterparty, credential, and explicit
signed read-only authority trigger; every trade-capable action still requires
separate authority.

The initial local invocation used an unverified global Python interpreter and
failed to import `websockets` before runner entry or network access. The journal
preserves that unconsumed local failure; the locked `uv` project runtime then
passed. `AGENTS.md` now requires the exact locked runtime and transport imports
to be preflighted before freezing a one-use contract. Canonical follow-up
SHA-256 is
`9fa2c8893d73ea7b1bf0efb70c284a20d606866798c121defca131985e84c056`.
The accepted-edge count remains 21, the ranked-hypothesis count remains 44, and
the updated registry SHA-256 is
`a375476e54a0a2949e6954d04384f72f11157f73238af61209a393c9362725c8`.

## 2026-08-30 Polymarket Elon fixed-NegRisk exact parity rejection

Current official discovery exposed a newly deployed, previously unconsumed
August 31–September 2 Elon post-count event. This fired rank 31's literal new
fixed-NegRisk deployment trigger. One frozen exact Gamma GET retained all ten
active compatible outcome markets. Their displayed all-YES sum was 1.0130 pUSD,
rejecting the guaranteed-payout purchase path, but each one-NO-to-other-YES
identity showed a 0.0130 pUSD displayed gap. That was a source-only lead, never
executable or profitable evidence.

A separately frozen single POST retained all 20 exact CLOB books. The response
passed every run-level gate: 121 ms request elapsed, 297 ms oldest-book age,
129 ms timestamp skew, exact token/condition identities, matching minimums and
ticks, and complete five-share depth evaluation. The best path lost 0.075 pUSD
before fees. Current Gamma taker fees widened the loss to 0.23977 pUSD, and one
adverse tick on every leg widened it to 0.61792 pUSD. All evaluated views had
zero profitable paths, so the screen stopped before fee-rate, adapter, Polygon,
account, credential, order, transaction, or fund access.

The exact event is terminal. Do not repeat either source request, reorder or
subset the books, select favorable bins, or reinterpret the 1.3-cent displayed
gap as executable. Reopen rank 31 only on another literal distinct deployment
or its other existing material triggers.

The frozen v1 book contract was not consumed: full local preflight failed on an
ASCII-only reader before a request journal or network access existed. The raw
Gamma source contains valid UTF-8 punctuation. The failure is preserved with
canonical SHA-256
`fb562b32287caee9842e0dac48aad3bd16f8cf6e2c45d78dca11ce2dde0f9078`.
The corrected v2 loader parses JSON bytes and passed the complete retained-input
and contract path before freezing. `AGENTS.md` now requires that full path, not
only `--help`, before any retained-input one-use contract. The original frozen
runner remains mechanically reconstructable from the corrected runner with one
exact replacement; canonical lineage SHA-256 is
`dbfa67537e141344d5d0b15c62944eec3ef72da2c7cb945c61303085b4b40bc5`.

Canonical prefilter and exact-book result SHA-256 values are
`63fb913d9f56034879ccee6bc43d531d4a5e805550db99ec6331e31051c680aa`
and `4601f3980f14ccb4130fbdc36862def5abdd47f46e9f48c7da25113c72fe33a2`.
Accepted edges remain 21, ranked hypotheses remain 44, terminal families become
53, and registry SHA-256 is
`5e34a52a6e0eebf48d5c4ae397bcb1893c10389116425d057b580a2a05013c40`.

## 2026-08-30 Polymarket NYC Mayor fixed-NegRisk exact parity rejection

Polymarket's current official event page exposed a previously unconsumed NYC
Mayor September 1-8 post-count event, opened August 29 with eleven mutually
exclusive outcomes. This fired rank 31's literal distinct fixed-NegRisk
deployment trigger. One frozen exact Gamma GET confirmed the fixed event and a
displayed all-YES sum of 0.9855 pUSD, leaving 0.0145 pUSD of source-only gross
headroom. Gamma remained rejection-only and did not establish executable depth.

A separately frozen complete 22-token CLOB batch then eliminated the lead. The
request itself took 121 ms, but the oldest book was 92,065 ms old and cross-book
timestamp skew was 91,385 ms, so the frozen freshness gate failed. More
decisively, the best five-share all-YES path lost 0.45 pUSD even before fees,
0.54928 pUSD after current Gamma taker fees, and 0.69638 pUSD after one adverse
tick per leg. Every view had zero profitable paths. No adapter, fee-rate,
on-chain, account, credential, order, transaction, or fund request was needed.

The exact event is terminal. Do not repeat either request, selectively refresh
books, or reinterpret the displayed 1.45-cent gap as executable. The reusable
prefilter now validates an exact contract-bound outcome count instead of a
ten-bin special case, and the book runner derives the complete token population
instead of assuming twenty tokens. Historical frozen implementations remain
bound to their original commit hashes.

Canonical prefilter and exact-book result SHA-256 values are
`c4d9b8ce130881d65a7711dc4cc9e48d9d56085d9e07bd15736d44a63ab13bf1`
and `2dcaa72b8a9643b3f6652691f7395ac5405cd6f15ee88e57ce45af1f69b0dc6b`.
Accepted edges remain 21, ranked hypotheses remain 44, terminal families become
54, and registry SHA-256 is
`f6b73019910d57daf98764d78a80e487421ae73525499ad1c4e5600ab6018d4f`.

## Polymarket BTC September 5 paired-maker reward rejection

Rank 17's distinct source-selected exact-allocation trigger fired for `Will the
price of Bitcoin be above $72,000 on September 5?`. The source-only prefilter
correctly excluded all discovery-page values from economics. Exact Gamma and
the exact sponsored-condition endpoint reconciled the same two tokens, 50-share
reward minimum, 4.5-cent maximum spread, zero maker fee, and one active dated
configuration paying 1.99972 pUSD per day. No book was requested until that
reconciliation passed.

The separately frozen two-token book request completed in 240 ms, but the two
internally agreeing timestamps were 30,871 ms old, so current executable
economics remain unqualified. The retained best-bid join nevertheless provides
a terminal optimistic rejection for that exact observation: 2.05 pUSD gross if
both 50-share legs filled, versus 46.80 pUSD maximum one-sided settlement loss.
Even the impossible assumption that one maker captured 100% of every remaining
reward produced only 12.923014 pUSD, covering 27.61% of the orphan loss before
competition, adverse selection, queueing, cancellation latency, or hedge cost.

Do not refetch this exact market or treat its stale book as current. The new
generic source prefilter prevents the prior discovery-value hard-gate error,
and the retained-source book runner uses the full remaining pool as a cheap
rejection bound before any repetitive scoring or paper capture. A zero-request
audit caught one implementation omission: the consumed runner had not compared
the reward row's ordered token IDs to Gamma. Both retained exact sources do
contain the same ordered YES/NO IDs, so the omitted gate passed; the reusable
runner now enforces it before future book access, without refetching or changing
the rejection. Correction SHA-256 is
`61df67471329b0e4a1273deea0fbbba9d918d3e11559b3dc26f6f462f69691a4`.
No credentials, accounts, orders, cancellations, funds, or protected captures
were accessed.
Accepted edges remain 21, ranked hypotheses remain 44, and terminal families
become 55. Canonical retained adjudication SHA-256 is
`1153ef2f90345be8ebfda5b0c2fd3f02a56dc0dad854edf251e21370a9677743`;
registry SHA-256 is
`d82fe12b7ec4fb7765bdbad781ac7fc6ef1e6bf26da84882ab42f139411bb6fd`.

## Binance Spot PRIMARY_PEG execution overlay candidate

Official current Binance documentation and the retained production
`exchangeInfo` snapshot jointly establish `PRIMARY_PEG LIMIT_MAKER` support on
BTCUSDT, ETHUSDT, and SOLUSDT. The order derives its price from the same-side
best quote at matching-engine arrival and joins behind existing orders at that
price.

The efficient offline screen made zero new market requests and reused the two
retained public ticker windows. At a frozen one-second lag, deterministic
fixed-price crossing/rejection counterfactuals occurred in 416 of 3,365
discovery comparisons (12.36%) and 31 of 729 validation comparisons (4.25%),
with at least one observation in every symbol and window. This supports a
recurrent direction-independent order-acceptance overlay candidate only. It
does not establish subsecond state, acknowledgement, queue priority, fills,
spread capture, adverse selection, after-cost profit, or deployment readiness.

No credential, account, order, cancellation, fund, or protected Polymarket
state was accessed. Advance only with explicit separate Spot testnet or paper
authority for one minimum-size `PRIMARY_PEG LIMIT_MAKER` acknowledgement and
cancel comparison against a frozen fixed-price counterfactual, or after a
material official pegged-order semantics, filter, fee, or production-
configuration change. No mainnet authority exists. Canonical candidate
SHA-256 is
`605d5b195f43bbb9976a5bd3d239388aa918110a6860669da480fa7949b789a2`.
Accepted edges remain 21, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`4bf91c297d3c41583874fe77c3b1c456736bc2809c4bfdac320edd8124f62005`.

## Binance Spot STP TRANSFER internal-reallocation preflight

One frozen public official-FAQ request exposed a distinct direction-independent
mechanism. For different accounts sharing a `tradeGroupId`, when both maker and
taker specify `TRANSFER`, Binance applies `DECREMENT` and transfers the last
prevented base quantity and notional between the accounts. The already-retained
production `exchangeInfo` allows `TRANSFER` on BTCUSDT, ETHUSDT, and SOLUSDT,
while defaulting all three to `EXPIRE_MAKER`.

The retained example conserves aggregate inventory exactly: maker changes are
+0.2 BTC and -0.04 USDT, taker changes are -0.2 BTC and +0.04 USDT, and both
aggregate changes are zero. Both executed quantities are zero, the taker has no
fills, and the FAQ says its commission and price examples are fictional. The
maker remains open with reduced available quantity, so this is an internal
reallocation and queue-continuity candidate, not an accepted or profitable
edge. The official API index proves ordinary Universal Transfer exists, but its
exact fee, latency, atomicity, limits, and safety comparator remains unbound.

No credentials, account request, order, cancellation, or funds were used. Do
not manufacture self-crosses. Advance only with designated credentials,
explicit signed GET-only `tradeGroupId` authority, and an independently
existing legitimate BTC/ETH/SOL cross-account rebalance whose exact ordinary
internal-transfer comparator leaves positive incremental value. A testnet or
paper `TRANSFER` order requires separate explicit authority; mainnet remains
unauthorized. Canonical preflight SHA-256 is
`59da62dccab340bcae63b2ad34697f49caa7e381a7a06755e50adafba2d8d118`.
Accepted edges remain 21, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`8813fbc12ea5fe5ad5eb38630c7faba9c1a740389266ad27bb0c5db41759c52a`.

## Binance Institutional Loan realized interest-rebate overlay

The retained current official API index contains previously unregistered signed
`USER_DATA` endpoints for Institutional Loan interest history, interest-rebate
balance, and rebate records. The Binance-issued release retained through one
public provenance request says that, effective June 1, 2026, KYB VIP borrowers
may qualify for full monthly interest rebates on USDT, USDC, BTC, or U borrowing
up to 10 million USD by meeting performance targets tied to incremental trading-
volume share, Open Interest, or Net Asset Value.

This is accepted only as a narrow, direction-independent realized-credit edge
on an independently required existing eligible loan. Before exact account
evidence, the public forward floor is zero because target thresholds, enrollment,
eligibility, charged interest, credit calculation, distribution timing, and
successful payment are absent. The rebate may never justify borrowing, leverage,
collateral retention, target-chasing volume, Open Interest, or Net Asset Value,
and it cannot rescue an unprofitable loan or trading strategy. The issuer warns
that the entire Institutional Lending Account balance may be liquidated.

No credentials, account request, loan, collateral action, order, or fund access
occurred. Advance only with both designated credentials, explicit signed
GET-only authority, and an independently existing legitimate Institutional Loan
question. Reconcile the exact active risk unit, month, charged interest, rebate
balance, and successful rebate record after all incremental costs. Every state
change and account-manager contact requires separate authority. Canonical result
SHA-256 is
`e8e17c66a238878e722aa635f1517b685c00fcc9b288c72df3d934a8c235e59c`.
Accepted edges become 22, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`e4ec302a1cc32a57ec1d68cf00ab5d99cbc22d74b80fbc3c68da242485899fd4`.

## Binance CAAS realized organic-client commission markup

A zero-new-venue-data audit of the retained current official API index exposed
a registry-absent mechanism: Binance Crypto-as-a-Service has signed `USER_DATA`
queries for fee groups, members, markup-trade commission aggregations, and exact
markup-trade details. Fee-group creation or deletion, commission updates, and
member assignment or removal are separately classified `TRADE` operations.

The edge is accepted only as exact realized positive markup commission from
independently existing bona fide external client trades under an already active
disclosed configuration after every incremental platform, disclosure, consent,
compliance, support, demand-elasticity, tax, custody, settlement, and operating
cost. It is not a standalone strategy or public forward-profit claim. Public
floor remains zero because partner eligibility, active fee group, exact rate,
member lineage, client flow, realization, payout, and all costs are absent.
Creating, rerouting, splitting, churning, self-matching, soliciting, or assigning
activity for commission is prohibited.

The one frozen public dynamic-document preflight returned HTTP 200 but contained
the already known generic security document, zero required CAAS or commission
terms, and credential-shaped examples. It was consumed and failed closed; the
response hash is retained, the raw body is excluded from Git, and no alias or
retry is permitted. No credential, account, client, fee-group, member, order,
trade, fund, or mutation was accessed. Signed reporting may advance only when
both designated credentials, explicit signed GET-only authority, an independently
existing legitimate CAAS reporting question, an already active disclosed fee
group, and bona fide external client flow all exist. Every mutation, onboarding,
contact, order, trade, transfer, or withdrawal requires separate authority.

Canonical result SHA-256 is
`d1656eeccbcf780a2e71190d5f969db07a33548f58de940b55867d494dddebd2`.
Accepted edges become 23, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`a1fc3c10482909d2c77dbb20ca13dae3eb73e465fa96dcfb20dd0ede17337573`.

The affected integrity run also exposed two avoidable global-churn defects in
older Polymarket tests. One family test hard-pinned a historical whole-registry
hash even though it only needed the current registry self-hash and one terminal
family. A consumed pUSD-to-USDT contract pointed at the mutable current yield-
frontier path, so later legitimate accepted-edge updates broke reconstruction.
The historical contract and result remain untouched; the exact consumed
frontier bytes were recovered from Git into an immutable hash-bound sidecar,
and the test now resolves only that consumed source through the sidecar while
the live frontier remains current. `AGENTS.md` now requires immutable snapshots
before any outcome-sensitive contract binds a registry, frontier, manifest, or
rolling artifact. Snapshot adjudication result SHA-256 is
`8105343e417aea4084debb50a7ffe3a90b1c7c7bedf83da920961601ba152b23`.

## Binance USD-M Futures BNB fee-reduction source gate

The retained current official API index exposed a registry-absent, direction-
independent direct-cost lead: signed `GET /fapi/v1/feeBurn` reads the USD-M
Futures BNB-burn status, while `POST /fapi/v1/feeBurn` is separately classified
`TRADE`. A cached official publication indicated a 10% Futures trading-fee
reduction when BNB fee payment is enabled, so one exact public source request
was frozen before access to bind that rate.

The exact request failed closed: it returned HTTP 202, 2,035 bytes, and zero of
the three preregistered terms. The raw response and durable journal are retained.
The cached extraction is a lead, not admissible current-rate or account evidence,
and the public forward floor remains zero. Do not repeat or alias the consumed
page, acquire or retain standing BNB, create volume, toggle fee burn, or use the
discount to rescue an otherwise unprofitable trade.

Reopen only after a materially new byte-retainable current official source, or
when both designated credentials, explicit signed GET-only authority, and an
independently planned legitimate organic BTC/ETH/SOL USD-M positive-commission
question all exist. Exact status, positive standard commission, BNB acquisition,
full prompt consumption, residual, owned realized fee ledger, and every
incremental cost must then reconcile. Any toggle still requires separate
explicit state-change authority. No credential, account request, order, trade,
fund, or mutation was used. Canonical adjudication SHA-256 is
`3a2c7358757491c8e0e9d737a76583756596db59346c5877a7bcfc1ca9e300b4`.
Accepted edges remain 23, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 becomes
`9e3ceff615c795156ec151f209d9b6c391a9ea50184baa1bee78108f6eb2cb7b`.

## Binance BNSOL Boost APR airdrop existing-holding lead

The current official web-rendered SOL Staking schema exposes a separate Boost
APR airdrop mechanism: rate history contains boost reward fields, boost history
distinguishes `CLAIM` and `DISTRIBUTE` and returns reward token, amount, BNSOL
holding, and status, while unclaimed rewards return amount and reward asset.
The claim endpoint is separately signed `TRADE`. This is economically distinct
from the BNSOL conversion ratio and base staking reward, but no public example
is current account entitlement or owned reward evidence.

The one frozen raw source-retention request returned HTTP 202, 2,038 bytes, and
zero of eight required terms. It is consumed and must not be retried through an
alias. The raw response and journal are retained, the rendered extraction is
discovery-only, and the public forward floor is zero. No credentials, account
request, staking, claim, order, trade, fund, or mutation was used.

Reopen only on materially new byte-retainable official terms, or when both
designated credentials, explicit signed GET-only authority, and independently
existing BNSOL held for unrelated reasons all exist. Read exact rate history,
`CLAIM` and `DISTRIBUTE` history, unclaimed rewards, base rewards, and holding;
reject unless positive value is non-double-counted and survives every cost.
Claiming remains a separate state change requiring separate authority.
Canonical adjudication SHA-256 is
`f0b6b4df8632b1cc302bdb24189c4336968be1fa9b94d1f7792205f46450c466`.
Accepted edges remain 23, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 becomes
`38450f4fa1720247e0b1717ac1d6022e1a1a3f95e473c65ae5b39deda17cc720`.

## Binance Link-and-Trade realized client kickback overlay

The retained current Agent Native index and one predeclared official rendered
source extraction exposed a registry-absent direct-cost mechanism. The signed
client-side `/sapi/v1/apiReferral/kickback/recentRecord` route returns `income`,
`asset`, `symbol`, and `time` for the latest seven days. It is distinct from the
explicitly named partner-rebate route, which accepts customer selection and
returns customer, order, trade, distribution, and commission lineage. The
adjacent signed client-status route returns `rebateWorking` and referrer lineage.

The edge is accepted only as exact realized positive own-account kickback income
from independently justified legitimate organic Spot trading on an already
linked rebate-working account, after exact owned balance reconciliation and
every incremental fee, tax, support, compliance, settlement, and operating cost.
The underlying trade must remain positive without the kickback. Documentation
examples are not rate, eligibility, income, or profitability evidence; the
public forward floor is zero and this is not deployment-ready.

No credentials, signed request, account state, linking, customization, order,
trade, transfer, fund, or mutation was used. Advance only when both designated
credentials, explicit signed GET-only authority, the already linked account and
known agent code, and an exact latest-seven-day organic kickback question all
exist. Never open, relink, customize, reroute, split, churn, self-match, or trade
for kickback, and never double count partner rebate, referral commission, fee
discounts, or promotions. Canonical result SHA-256 is
`ada551b385d9040e4126ee0e73e1dd1f417b103e6c5c5f7c567411ab913ff065`.
Accepted edges become 24, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 becomes
`8888314ada1411b74b2b6417201f6cee13733d713527f22c07abfa4ab1900864`.
The accepted yield frontier still contains nine yield rows and now excludes 15
non-yield organic-flow, fee, referral, creator, and financing overlays; its
canonical SHA-256 becomes
`5f638281d9df79456a5a909f4da07e6375ff46404a8a63fceb7f96899a5a74c2`.

## Binance Prediction Trading yield-bearing collateral lead

The current official rendered Prediction Trading schema exposes a previously
unregistered direction-independent carry lead. Market list and detail include a
per-market `isYieldBearing` boolean alongside vendor, chain, collateral,
`feeRateBps`, and `slippageBps`. Both endpoints explicitly require a signed
timestamp. This metadata is structurally relevant because a genuinely credited
yield on independently justified prediction collateral could offset part of
capital lock-up without forecasting market direction.

The lead is not an edge. The schema does not define the yield recipient, rate,
base, accrual clock, distribution, redemption, current true market, account
entitlement, or non-double-counted after-cost value. The exact public
machine-readable schema URL was frozen before access, but returned HTTP 202 with
zero bytes and zero required terms. That one-use source request is consumed and
must not be retried through an alias. No Binance Prediction market, account,
wallet, quote, order book, order, position, transfer, redemption, OTC, fund, or
protected Polymarket request occurred.

Reopen only when a materially new byte-retainable current official source binds
the complete yield economics, or when both designated credentials and explicit
signed GET-only Prediction Trading market-metadata authority exist. Even then,
never treat `isYieldBearing=true` as user-owned positive yield without exact
account evidence and every fee, basis, conversion, custody, settlement, tax,
and redemption cost. Canonical failure adjudication SHA-256 is
`e0049982adfdb631bb71bc7ebaf957d0a96336b42f269783ec99d4812e12bafc`.
Accepted edges remain 24, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 becomes
`aec184ae571d27d933003fc9dac056241d11fe01bb40ed16dac9a86c77cddd3b`.

## Polymarket Ball State vs. Ohio State exact CFB monotone rejection

A newly indexed September 5 college-football event materially extended the
direction-independent sports monotone-payoff family beyond its consumed
NBA/WNBA/NFL populations. One frozen exact Gamma event request returned an
active Ball State vs. Ohio State event with three accepting markets: moneyline,
Ohio State -50.5 spread, and 56.5 total. All three currently use the same
fee-enabled 5% taker-only curve, and the source shows a one-second sports delay.

The complete offline payoff reduction retained nine feasible rule classes:
Ball State win, Ohio State win by 1-50, Ohio State win by at least 51, and an
actual tie, each split by total under/over 57, plus cancellation. A nonnegative
linear superhedge over all six outcome tokens found a minimum Gamma displayed
cost of exactly 1.0000 pUSD; one optimum is simply one Over plus one Under
share. The nontrivial monotone package of Ohio State moneyline plus Ball State
+50.5 cost 1.5015 pUSD. Therefore no package was strictly below its guaranteed
payout floor even before taker fees, spread, slippage, latency, or partial-fill
risk. Gamma remained rejection-only and no CLOB book was requested.

The exact event-specific capture implementation had temporarily occupied an
existing reusable runner path. Its consumed bytes are now retained in an
immutable hash-bound sidecar, the reusable runner is restored byte-for-byte,
and the terminal artifact binds both. The request and frozen contract were not
rewritten or repeated. `AGENTS.md` now requires a tracked-path ownership check
before adding any one-use runner.

The exact event is terminal. Do not refetch it or generalize one null event into
a family rejection. Reopen only for a distinct active NBA, WNBA, NFL, or CFB
event outside every consumed population after exact rules, tie/cancellation
states, and fee schedules are bound and Gamma exposes at least one package
strictly below its guaranteed payout floor. Canonical terminal SHA-256 is
`13eb2d843260b2e05d693f82ddc6f3efcbae51e83d61f5fe06693be56c2c30a5`.
Accepted edges remain 24, ranked hypotheses remain 44, and terminal families
become 56. Registry SHA-256 becomes
`2d2e24457ea117c094c82f2259ed75970e29666391a1575d285b5e4275d83080`.

## Polymarket Elon August 28-September 4 partial-resolution rejection

Polymarket's official rendered page exposed a distinct 26-outcome Elon post-
count event and fired rank 31's literal new fixed-NegRisk deployment trigger.
One frozen exact Gamma GET retained all 26 markets. The event remained active,
but `<20` and `20-39` were already closed, nonaccepting, and displayed at zero
YES/one NO; the other 24 markets were active and accepting orders.

The consumed reusable runner retained the raw response and journal, then failed
closed because it required every market to be simultaneously available. A
zero-request retained audit assigned the two closed losing bins their optimistic
zero YES cost and summed all 24 active YES prices to 1.0055 pUSD. This is already
0.0055 pUSD above the maximum one-pUSD complete-set payout before fees, spread,
slippage, latency, partial fills, or adapter constraints. No book, fee, on-chain,
account, credential, order, transaction, or fund request was justified.

Do not refetch, alias, or selectively refresh this event. The exact consumed
runner bytes are retained in an immutable sidecar. The current reusable runner
now converts unavailable or incompatible exact-event populations into a normal
fail-closed terminal result rather than crashing after a future capture. The
first direct-file `--help` preflight also failed locally before network because
the runner imports `tools.*`; that error is journaled and the mandated locked
module invocation passed before access.

Canonical terminal SHA-256 is
`d22954b0611b7bd8210dbc8d59bf1f9a645c118331da057d80585b364ed0c0ac`.
Accepted edges remain 24, ranked hypotheses remain 44, and terminal families
become 57. Registry SHA-256 becomes
`c03acc97b09440a4def55447f2fc1d62e410bfc9f77b6aae97b58d7cb2eee155`.

## Binance Exchange Link organic commission-rebate candidate

A zero-new-venue-request audit of the retained current official Agent Native
index found a previously unregistered mechanism under rank 24. Binance lists
Exchange Link as a separate account, asset, and commission-data product and
names exact Spot and Futures commission-rebate record endpoints. These are not
the adjacent Link-and-Trade `apiReferral` client-kickback and partner-rebate
routes, CAAS markup reports, Referral Pro commissions, market-maker rebates, or
Institutional Loan interest rebates.

This is a direction-independent candidate only. The retained index does not
bind either exact endpoint's security classification, parameters, response
fields, time-window semantics, eligibility, rate, trade lineage, distribution
asset or timing, reversal or clawback state, owned payout, or incremental cost.
Bounded current primary-source searches exposed no separately indexable exact
schema. The public forward floor is therefore zero, the accepted count remains
24, and no profitability or deployment claim is admitted.

No venue API, credential, signed request, account, client, subaccount, API key,
permission, commission change, order, trade, transfer, withdrawal, or fund was
accessed. Do not infer that either `GET` is public, read-only, `USER_DATA`, or
`TRADE` from its verb, path, neighboring endpoints, or product name. Reopen
first on a materially new byte-retainable current primary contract binding the
exact endpoint security, schema, and semantics. Any later signed reconciliation
also requires both designated credentials, explicit signed GET-only authority,
an independently existing legitimate Exchange Link reporting question, an
already active disclosed relationship, and bona fide external flow; every
mutation or trading action remains separately unauthorized.

Canonical candidate SHA-256 is
`245ef96228dccf51194f0e10176ffa39676ec4e5e07f78a2daefa1205b2fde3a`.
Accepted edges remain 24, ranked hypotheses remain 44, and terminal families
remain 57. Registry SHA-256 becomes
`e71ec77c3a06164f35249879f22300f9c8d3b46a1020d6b96f271c4c6fb5b661`.

## Binance USD1 Simple Earn remaining-horizon correction

The activation-day source-bound stress credited 28 potential USD1 bonus days
and left only 0.9683445745 bips after its historical USD1USDT basis and spread
stress. A distinct zero-request recalculation preserved every frozen economic
input and changed only the mechanically elapsed accrual days and the competing
USDT fixed-bonus days remaining under their respective published end dates.

For a subscription dated 2026-08-30, bonus accrual begins on August 31 and has
26 USD1 days versus eight remaining USDT fixed-bonus days. The incremental
reward is 25.5707762557 bips. After the exact retained 23.9544864757-bip worst
30-day close decline and 0.1000095009-bip displayed round-trip spread, only
1.5162802791 bips or 0.2274420419 quote units at the 1,500 cap remain before all
unknown costs. The unchanged case falls to 0.0550930645 bips for a September 1
subscription and becomes negative on September 2 even before those costs.

The accepted same-principal fixed-bonus scope is unchanged, but stable profit is
not proved and the public after-all-cost floor remains zero. No network,
credential, account, conversion, subscription, redemption, transfer, order, or
fund was used. Do not refresh the book, kline history, or this calculation each
day. Before September 2, only both designated credentials plus explicit signed
GET-only account-evidence authority could justify one eligibility, capacity,
exact-rate, fee, and sold-out-state prequalification. Every funded action still
requires separate authority.

Canonical stress SHA-256 is
`669fd50772087cb81a4d1e9439e5666a75b5c1ed9de68b1e8b27cb360f2d5934`.
Accepted edges remain 24, ranked hypotheses remain 44, and terminal families
remain 57. Current frontier SHA-256 becomes
`53afed572c779113de6e9760f319ad7fb7a2d1e1958997139793142cf241fbe7`;
registry SHA-256 becomes
`9d4902c9608c358d212e329d9f2aa74e726d7187a218fd4123ed0cec0fe2d123`.

## Binance U Flexible remaining-horizon correction

The activation artifact's 10.3422-bip acquisition sensitivity used a
2026-08-26 subscription and nineteen U reward days. A distinct zero-request
correction preserved its promotion terms, displayed UUSDT spread, and worst
retained nineteen-day close move, then applied only the elapsed accrual days and
the competing USDT promotion's own published end date.

For a 2026-08-30 subscription, accrual begins August 31 and leaves fifteen U
fixed-bonus days versus eight USDT fixed-bonus days. The primary case credits
only U's fixed 8% bonus, not its approximate variable 0.5% component. Its
incremental reward is 19.6712328767 bips before the retained 16.97354665-bip
spread-plus-basis sensitivity. Only 2.6976862267 bips or 1.3488431134 quote
units at the 5,000 U cap remain before every unproved account, issuer,
redemption, custody, tax, settlement, and operating cost. The same case leaves
0.1771382815 bips for a September 1 subscription and becomes negative on
September 2 before those costs.

This does not change the accepted existing-holding scope and does not accept a
USDT-to-U purchase. Stable acquisition profit is not proved and the public
after-all-cost floor remains zero. No network, credential, account, conversion,
subscription, redemption, transfer, order, or fund was used. Do not roll this
calculation or refresh its retained market inputs daily. Reopen only on the
literal published term, rate, quota, fee, reserve-attestation, redemption-
contract, or post-campaign trigger.

Canonical stress SHA-256 is
`d9be584383bdbf4e45f570987103e9d380358b2ba08aafc865ccecfc4b2c225e`.
Accepted edges remain 24, ranked hypotheses remain 44, and terminal families
remain 57. Current frontier SHA-256 becomes
`e3e6941790079587e3a21bf0894e165963cc871b9165821fd7b00600fc3c4dec`;
registry SHA-256 becomes
`beaa24adc3a5550185f0b387818565d7a0bb5bbddea4520ca71a101afce4baaf`.

## Polymarket realized organic crypto maker-rebate overlay

The retained official contract gives BTC/ETH/SOL crypto makers a zero maker fee
and a daily nominal rebate equal to 20% of the taker-fee-equivalent formula,
subject to a one-pUSD accrued-payment minimum and a discretionary rate. The
retained recurrence study then observed positive `MAKER_REBATE` receipts for
all ten public top-volume wallets, eight on all fourteen UTC dates, totaling
234,881.8839 pUSD. One exhaustively joined wallet-day contained 668 BTC/ETH/SOL
rows and 7,017.331032 pUSD of rebate cash.

The earlier adjudication correctly rejected a standalone maker strategy because
public payments do not prove owned fills, queue position, quote duration,
inventory, adverse selection, latency, orphan risk, or complete P&L. It was too
broad in rejecting the incremental cash overlay as well. The repository now
accepts only an exact realized positive owned rebate on an independently
justified legitimate organic BTC/ETH/SOL maker fill after every incremental
cost. A nominal estimate, another wallet's receipt, manufactured volume, or a
rebate used to rescue negative base economics does not qualify.

The public forward floor remains zero, the standalone market-making strategy
remains rejected, and deployment readiness remains false. No credential,
account, owned wallet, venue API, order, cancel, transfer, withdrawal, or fund
was accessed. Every order-capable action still requires separate authority.

Canonical overlay SHA-256 is
`a8db5f3c823c8b1caffa6b0032282647dbb1e7fb014f8923821c1b1fe97d1c81`.
Accepted edges become 25, ranked hypotheses remain 44, and terminal families
remain 57. Current frontier SHA-256 becomes
`05d423fdfa461f22a88ff1d3887804c2dc40080d7a18f4fa63e6b91381ee140d`;
registry SHA-256 becomes
`6f1ead1fd1609f6da6c0ab762a236974b86c967ca31e5d9b7bdc4e4b776687c7`.

## Binance realized organic Spot Liquidity Program rebate overlay

The retained current public program gate proves zero maker fees after
enrollment and higher tiers displaying 0.4, 0.6, and 0.8-bip rebates. It also
binds account-specific overview, performance, weekly final-rebate, and Spot
rebate-history endpoints. The symbol commission endpoint explicitly excludes
both the Liquidity Program rebate and BNB discount effects.

The earlier adjudication correctly refused to accept a standalone market-making
strategy without account enrollment, owned fills, queue, adverse-selection,
inventory, unwind, and complete P&L evidence. It was too broad in also rejecting
an exact final positive cash rebate on independently justified organic maker
flow. The repository now accepts only that realized incremental receipt after
every enrollment, capital-turnover, hedge, slippage, inventory, opportunity,
tax, compliance, and operating cost.

The public schedule and 0.8-bip maximum are not owned income. Zero maker fees,
BNB discounts, bStocks promotions, symbol-commission savings, and a final
rebate may each be counted at most once. No application, account, credential,
signed request, venue API, order, cancel, amend, transfer, withdrawal, or fund
was used. The public forward floor is zero, the standalone strategy remains
unaccepted, and deployment readiness remains false.

Canonical overlay SHA-256 is
`13c0a9468e439f9163ede0b5824c9b065737078ece0cba5f2b348fb402ef01d4`.
Accepted edges become 26, ranked hypotheses remain 44, and terminal families
remain 57. Current frontier SHA-256 becomes
`53c5a99573dcc97d85caf07f42e84818d649107b2ff4fa18af52c0b9eb505c6a`;
registry SHA-256 becomes
`e9b79af6bd9ad29564aff297daa2ea7375f71f42df3757761d72fd4e20d09995`.

## Binance Cross Margin BNB interest discount overlay

Current official Binance Academy material, updated 2026-06-26 and retained
byte-for-byte on 2026-08-30, states that holding BNB in the margin account and
enabling the discount provides a 5% reduction on Cross Margin interest. The
same source separately states a 25% margin trading-fee discount; the two values
must never be conflated or double counted.

Current official API contracts expose signed read-only
`GET /sapi/v1/bnbBurn` status with the `interestBNBBurn` field and signed
read-only `GET /sapi/v1/margin/interestHistory`. Omitting `isolatedSymbol`
selects Cross Margin history, while `PERIODIC_CONVERTED` and
`ON_BORROW_CONVERTED` identify interest converted into BNB. The current toggle
is `POST /sapi/v1/bnbBurn`, a mutation that was not called and remains behind
separate explicit authority.

The accepted scope is only the exact incremental saving on an independently
existing legitimate Cross Margin borrow when enough BNB was already held in
that same account for unrelated reasons. Credit requires `interestBNBBurn=true`,
exact converted-interest rows, a source-bound undiscounted counterfactual, the
actual BNB debit, and subtraction of conversion, transfer, spread, price,
opportunity, tax, compliance, rounding, and operating costs. Isolated Margin,
Portfolio Margin, new borrowing, trading, BNB acquisition or retention, and
using the discount to claim the underlying leveraged strategy profitable are
excluded. The public forward monetary floor is zero and deployment readiness
remains false.

No credential, account, venue API, order, borrow, repay, transfer, toggle, or
fund was accessed. Canonical overlay SHA-256 is
`38aa0313cdd71a2613f3850267e71acd9d44006dd2699e4c00f801ffca8772f8`.
Accepted edges become 27, ranked hypotheses remain 44, and terminal families
remain 57. Current frontier SHA-256 becomes
`aed37a0d9527c4c02b63cf1b7bffb7a061e4236e246d3098ab27328ee51f8a58`;
registry SHA-256 becomes
`59bd2c2b7b0c5a0faff86ef76bebcbbaa5b3965d93cb779a8ce5de8abab3fbd1`.

## Evidence-Quality Correction

The current market-independent yield frontier now separates scoped mechanism
acceptance from owned profitable execution. Of its nine accepted yield and
capital-efficiency scopes, one has owned recurrent cash evidence after
source-bound direct costs, eight have public historical or fixed-term gross
evidence without owned account reconciliation, zero have a current owned
positive floor after every incremental cost, and zero are deployment-ready.
This classification does not revoke any narrow accepted mechanism; it prevents
the accepted-edge count from being read as nine currently executable profits.

No immediate venue request is justified. The WNBA window is consumed and must
not be repeated. Under unchanged evidence, the next exact public triggers are
the terminal GLWUSDT history reconciliation after `2026-08-31T00:00:00Z`
without books, followed by at most one distinct rank-1 Polymarket single-wallet
continuity pulse not before `2026-08-31T02:15:30Z`. Do not roll the USD1 or U
remaining-horizon stresses daily.

## Binance direct-versus-indirect standalone spread triage

The paper metadata for SSRN `6453880` was already retained under rank 44, but
its abstract did not distinguish an organic conversion saving from a standalone
statistical spread. One frozen canonical-page request retained an independent
replication that resolves that classification without requesting Binance data.
The standalone strategy longs direct-route value and shorts indirect-route
value; it is direction-independent in identity but is not a current scalable
edge.

The replication reports that capacity-constrained PnL fell from about 4,500 USD
in 2025 to about 250 USD in 2026-H1, a roughly 90% decline. Every one of the 42
profitable 2025 paths supported at most 25 USD per trade under its five-percent
volume rule, the median was about 6 USD, and 78% of intended trade minutes had
no executable volume on the thinnest leg. Zero of 86 profitable routes had a
perpetual on every required leg, while the only scalable examples, BTC-FDUSD and
ETH-BTC, both lost money. The estimated edge also required all-in cost below
about seven bips per leg.

This secondary review supplies no public code, exact route-level output, books,
owned fees, or fills. It therefore justifies neither a current profitability
claim nor a broad permanent rejection, but it does reject spending resources on
another minute-bar collector under unchanged evidence. The accepted rank-44
scope remains only the fail-closed saving on an independently required organic
conversion. Reopen the standalone spread only on reproducible route-level audit
evidence, a material fee or perpetual-leg listing change, or current executable
capacity sufficient to clear fixed costs. Canonical triage SHA-256 is
`ba9d063f78b027f6aab5e45723f5dc4ea2e9df1303de4f493e780ee07d4425b7`.
Accepted edges remain 27, ranked hypotheses remain 44, and terminal families
become 58. Registry SHA-256 becomes
`46eee6375a04b6226db08e6ee5ddc59c530b8c991840feefc6e145b48059c0fd`.

## Binance Portfolio Margin capital-netting sensitivity

Current official Binance material now makes one important correction to prior
spot-perpetual carry screens: Portfolio Margin provides unified margin across
Spot, Futures, and Options and uses cross-margining and position netting to
reduce required margin. The current account schema also distinguishes
collateral-rate-adjusted `accountEquity` from unadjusted `actualEquity` and
reports `accountInitialMargin`. A matched owned-spot long and USD-M perpetual
short therefore should not be rejected merely by assuming two fully separate
cash collateral pools.

This is a capital-efficiency mechanism, not a profitability result. The exact
current asset collateral rate, tier, leverage, eligibility, negative-balance
interest, liquidation buffer, and owned account fees remain unproved. Binance
labels `GET /sapi/v1/portfolio/collateralRate` as `MARKET_DATA`, but its current
official schema still requires an `X-MBX-APIKEY`; the tiered endpoint is signed
`USER_DATA`. No credential, account, Portfolio Margin enrollment, transfer,
borrow, order, trade, or fund was accessed.

A zero-network sensitivity reused the frozen 17-symbol broad funding result and
optimistically removed one of its two 10%-annual opportunity-cost legs while
preserving the 32-bip execution stress and every observed funding path. Only
one of 51 role rows became nominally positive: TUTUSDT validation at
`20.5190204528` bips, but its family-adjusted bootstrap lower bound remained
`-39.7220795472` bips. The best training row remained TUTUSDT at
`-22.6368803653` bips and the best test row remained PYTHUSDT at
`-31.3614004566` bips. Zero of 17 symbols was positive in training,
validation, and test; zero of 51 family-adjusted bootstrap lower bounds was
positive. Portfolio Margin therefore does not rescue this retained population.

One frozen CMS request was consumed but its response was lost after access when
an ephemeral orchestration decoder was unavailable; it was not retried or
repaired through an alias. Two separately prejournaled direct documentation
captures returned empty HTTP 202 responses and support no content claim. The
reusable correction is now explicit in `AGENTS.md`: never consume one-use HTTP
inside an ephemeral memory-only callback; preflight the full byte-to-durable-
file path and let the client atomically retain the raw body and receipt.

Do not resample, refit, paginate, or request books for this retained 17-symbol
population. Reopen only on a prospectively complete point-in-time universe, a
material funding/fee/collateral/margin/execution change capable of clearing the
retained training and test deficits, or exact same-account read-only Portfolio
Margin evidence after both designated ephemeral credentials and explicit
signed GET-only authority exist. Canonical sensitivity SHA-256 is
`b31cc92f4fad9dad7d8d0ea98c3275605b16069afc0d6d5882e75501025f7d14`.
Accepted edges remain 27, ranked hypotheses remain 44, and terminal families
become 59. Registry SHA-256 becomes
`a4d7119d665b2410939a07f1306091b396592aa98e9635abd57b4ac3809aa165`.

## Polymarket Clemson vs. LSU exact CFB monotone rejection

The next bounded official-source sweep found no new Binance bStock, TradFi,
Launchpool stablecoin-pool, commodity-option, or leveraged-token listing trigger.
Rank 30 then advanced one distinct CFB event outside every consumed event and
catalog window: Clemson vs. LSU on 2026-09-05. The event was selected as the
first exact result in a bounded public search without using discovery prices as
economic inputs. One frozen public Gamma GET retained 17,203 bytes, four active
accepting markets (moneyline, two spreads, and one total), and exact tie and
cancellation rules. No credential, account, protected capture, book, fee,
order, transaction, or fund access occurred.

The first zero-network football adjudication failed on a parser defect: it
required the unobserved phrase `if the LSU win`, while the retained CFB rule says
`if LSU win`. The failed contract remains immutable. A separately frozen CFB
grammar correction reused the exact retained bytes with zero refetches and now
binds both teams' win text, each exact spread threshold, its complementary
outcome, and cancellation semantics before producing any payoff relation.

The corrected complete lattice proved three exact relations and found zero
Gamma displayed packages strictly below the 1 pUSD payout floor. The best
displayed package cost 1.045 pUSD, or negative 0.045 pUSD optimistic headroom
before taker fees, spread, slippage, latency, and partial-fill risk. Therefore no
book or fee request is justified, the exact event is terminal, and the sports
family remains unaccepted. Do not repeat Clemson-LSU; advance only a future
distinct event whose rejection-only Gamma package is already strictly sub-floor
or a material sports price, fee, delay, or resolution-rule change.

Canonical corrected result SHA-256 is
`a32e77230d1ec6b48e69e500183c6400d3acf796efbd17acb8853183977d6da2`.
Accepted edges remain 27, ranked hypotheses remain 44, and terminal families
become 60. Registry SHA-256 becomes
`6edc1fe45b6bb6a156dffbec4ff5de2b05f5e0a221fca8c862612cf0b07c1ac1`.

## Polymarket complete September 3-4 CFB monotone catalog rejection

Rank 30 advanced with one complete CFB catalog instead of another hand-picked
event. The frozen start-time window was 2026-09-03T00:00:00Z through
2026-09-04T23:59:59Z, ending before the consumed Ball State-Ohio State and
Clemson-LSU September 5 events. One public unauthenticated keyset GET used the
documented maximum 500-event page size, retained 240,237 raw bytes, and returned
18 events with no continuation cursor. No adaptive pagination or narrowing was
performed.

Thirteen events had complete rule-compatible moneyline, spread, or total
lattices. Five were excluded because they had no exact monotone relation. The
screen retained all 19 proved relations and found zero Gamma displayed packages
strictly below the 1 pUSD guaranteed payout floor. The best rejection-only row
was the San Jose State-Eastern Michigan full-game total package at 1.02 pUSD,
or negative 0.02 pUSD optimistic headroom before taker fees, spread, slippage,
latency, and partial-fill risk. Therefore the window is terminal before any
book, fee, credential, account, order, fund, or protected-capture access.

Do not repeat, paginate, narrow, or depth-test this exact CFB window. Reopen the
sports family only for a future distinct event outside every consumed event and
window whose rejection-only Gamma package is already strictly sub-floor, or a
material price, fee, delay, or resolution-rule change. Canonical result SHA-256
is `fd6373bba5d18b07d4286e9b96f643741afede5c960d490a91438bf77cf67d3d`.
Accepted edges remain 27, ranked hypotheses remain 44, and terminal families
become 61. Registry SHA-256 becomes
`90508047f15e556c7a59ea37628c3127103e7e76f1e4af68a51639d0cf5a6073`.

## Polymarket Elon September 1-8 fixed-NegRisk depth rejection

Rank 31's literal new-deployment trigger fired for the distinct 26-bin `Elon
Musk # tweets September 1 - September 8, 2026?` event. The discovery page was
used only to select the exact event; none of its displayed values entered the
economics. One frozen public Gamma GET retained 104,604 bytes and confirmed all
26 active accepting fixed-NegRisk markets. The complete displayed YES sum was
1.0305 pUSD. Although all 26 displayed one-NO conversion identities remained
source-only leads, that prefilter could not establish executable economics.

One separately frozen public 52-token CLOB batch then retained 110,476 bytes in
507 ms. It failed the frozen freshness gate because the oldest book timestamp
was 59,170 ms old and cross-book timestamp skew was 52,902 ms. The original
generic adjudicator exposed an efficiency defect after the response was already
durable: it attempted all `2^26 - 1` conversion subsets three times. The owned
processes were stopped without refetching. The completed raw response and exact
two-row request journal remain immutable.

A separately frozen zero-network adjudication replaced exhaustive enumeration
with the exact additive best-subset identity and bounded meet-in-the-middle
profitable-path counting. On the retained batch, only the complete all-YES path
was executable. At five shares its best net was negative 0.425 pUSD before fees,
negative 0.66440 pUSD after current Gamma taker fees, and negative 1.12694 pUSD
after one adverse tick per leg. The event is therefore terminal on both
freshness and economics before any adapter, on-chain, credential, account,
order, transaction, or fund request.

The reusable optimizer now evaluates the exact best path without exponential
enumeration, enforces a 32-variable pre-request ceiling for exact path counting,
and documents the required module invocation. Do not repeat this event's Gamma
or book requests, restart the exhaustive runner, or select favorable bins.
Canonical retained adjudication result SHA-256 is
`8391ecd524d0db9c312f51e627f391d95797cf9f5d32f569fd31210fe064bc83`.
Accepted edges remain 27, ranked hypotheses remain 44, and terminal families
become 62. Registry SHA-256 becomes
`163cfbca723f1b84620e09853d698dd194e61d416c34fa71b2079a9e927c2313`.

## Binance Algo Trading execution-cost source gate

A zero-network comparison of the retained current official Binance API index
against every ranked and terminal mechanism found one distinct structural lead:
exchange-native Spot TWAP plus USD-M Futures TWAP and Volume Participation.
These algorithms could reduce market impact only for an independently required
legitimate BTC, ETH, or SOL execution; they do not create profitable flow and
are not an accepted edge.

One frozen public unauthenticated GET targeted the exact current official Algo
Trading OpenAPI schema. The request returned HTTP 202 with zero bytes. The raw
empty response and two-row intent/completion journal are retained, and none of
the six preregistered endpoint or child-order evidence terms passed. The exact
request is consumed: do not retry it, switch to an alias, or use rendered
discovery values to override the failed source gate.

The public forward saving floor is zero. Public documentation preserves only a
candidate architecture; it does not prove savings, maker/taker mix, extra algo
fees, fill quality, testnet support, account eligibility, or positive after-cost
value. Reopen only on a materially new byte-retainable official execution or fee
source. A historical signed child-order read additionally requires both
designated credentials, explicit GET-only authority, and an independently
existing legitimate algo order. Any prospective comparison requires confirmed
testnet or paper support, separate explicit order authority, independently
required flow, and a precommitted identical-quantity direct-execution benchmark.
No credential, account, order, transaction, fund, or protected-capture access
occurred.

Canonical failure adjudication SHA-256 is
`621f03ae812f57e9c8994e6073adc4153ca6eb447e9db61bad3bebc36e0b242f`.
Accepted edges remain 27, ranked hypotheses remain 44, and terminal families
become 63. Registry SHA-256 becomes
`49e909d549f015918690c14c1062021ca39c10c1ddeadce2ecacf0db8571bb37`.

## Polymarket complete September 5 CFB catalog and depth rejection

Rank 30 advanced with one complete day catalog instead of another hand-picked
event. The one public unauthenticated Gamma keyset request retained 1,142,294
bytes and returned 89 events with no continuation cursor. Both previously
consumed September 5 games were excluded before ranking. Fifty-eight other
events had rule-complete full-game moneyline, spread, or total lattices; the
screen retained all 88 exact payoff relations and found six displayed packages
strictly below their guaranteed 1 pUSD payout floor.

The deterministic frozen ordering selected only Fordham vs. North Dakota State
Over 56.5 plus Under 57.5, displayed at 0.960 pUSD per paired share. Its rules
prove a minimum 1 pUSD payout: totals through 56 pay Under 57.5, exactly 57 pays
both legs, totals from 58 pay Over 56.5, and cancellation pays one-half on each
leg. The other five observed candidates were not selected and may not be
cherry-picked after the depth outcome.

One separately frozen two-token CLOB batch rejected the selected package. At
five shares, both executable asks were 0.92, so the package cost 9.2 pUSD for a
5 pUSD guaranteed floor and lost 4.2 pUSD before fees. The two books were also
152,285 to 300,344 ms old with 148,059 ms timestamp skew. The stress path was
already gross-negative, so no fee request was made. No credential, account,
order, transaction, fund, or protected-capture access occurred.

Do not repeat, paginate, narrow, or refresh the September 5 catalog; do not
refetch Fordham-North Dakota State or try any runner-up from the consumed
catalog. The reusable exact sports runner now conjoins oldest-book age with
cross-book skew, and `AGENTS.md` records that a complete catalog gets one
precommitted deterministic depth escalation. Reopen only for a distinct event
outside every consumed population or a material price, fee, delay, or
resolution-rule change.

Canonical catalog adjudication SHA-256 is
`32c0e75914e651b6fc8da933628e39a90f6d83cb57563e2295d14bb279cf740b`;
canonical exact-depth result SHA-256 is
`8d2f9e8a5f00fa84c4291822692712151cb51bc7bdc7a659d12aa13f788361a1`.
Accepted edges remain 27, ranked hypotheses remain 44, and terminal families
become 64. Registry SHA-256 becomes
`b55f6a20ac311bca6ff68facae3015a2aa5da53fa6e6ae74208eaf4c3c613732`.

## Binance Link-and-Trade realized partner-rebate overlay

An efficient zero-new-venue-request adjudication reused the already retained
current official Link-and-Trade Spot rendered extraction and Agent Native API
index. The exact signed partner route is
`GET /sapi/v1/apiReferral/rebate/recentRecord`; unlike the separate own-client
kickback route, it carries customer, order, trade, commission, and distribution
lineage together with `income` and `asset`.

This establishes the twenty-eighth narrow direction-independent structural
edge only as exact realized positive partner-rebate income from independently
existing bona fide external Spot client flow under an already active disclosed
relationship after exact owned-payout reconciliation and every incremental
cost. The public forward floor remains zero. No account entitlement, active
relationship, client flow, rate, owned income, after-cost current profit, or
deployment readiness was proved; no credentials, signed request, account,
order, trade, fund, or mutation was accessed.

Never create, customize, solicit, relink, reroute, split, churn, self-match, or
manufacture customer or trade flow, and never double count a fee, trade,
customer, distribution, or payout across Link-and-Trade kickback, Referral Pro,
CAAS, Exchange Link, Square, Builder, maker rebate, fee discount, or promotion.
Advance only when both designated credentials, explicit signed GET-only
authority, an already active disclosed Link-and-Trade partner relationship,
exact legitimate customer identity, and an independently existing bona fide
external Spot latest-seven-day reporting question all coexist. All mutations
remain separately unauthorized. The separate Exchange Link candidate remains
unaccepted.

One exploratory direct Exchange Link documentation-root GET returned HTTP 202
with zero bytes without durable request-bound retention. It is excluded from
every claim, recorded as an efficiency/process miss, and must not be retried or
used to alter the Exchange Link decision. Canonical partner result SHA-256 is
`b859a815f0243285d7a01e2f002f77655fbcb25b07f46873e5de68b3ebfa8dd0`;
accepted edges become 28, ranked hypotheses remain 44, and terminal families
remain 64. Frontier SHA-256 becomes
`f84ac384ef3f4edfaa9f3a98e3588223ba3b4c6dd0b1389bbc11e4ece09a8b26`;
registry SHA-256 becomes
`e3cb85b3bf7920d32fd8a521690b2bffdb475d540225adf13031795249176b9a`.

## Binance Copy Trading realized Lead Trader profit-share overlay

The next official material-trigger sweep found no current Binance or Polymarket
rate, listing, fee, reward, or product change that satisfied a terminal
family's literal reopen condition. A zero-request novelty audit of the retained
current Binance Agent Native API index instead found an unregistered Copy
Trading mechanism. One frozen public official rendered read then proved that
Copy Trading covers Spot and Futures, Lead Traders set their profit-share rate,
and an authentic follower pays a share when closing a profitable copied
position. Qualification, performance, activity, verification, and regional
availability remain conditional, and the source explicitly rejects any future
performance guarantee.

This is the twenty-ninth narrow direction-independent structural edge only as
exact realized positive owned Lead Trader profit share from independently
existing authentic followers' profitable copied closes on an independently
cross-regime accepted legitimate BTC, ETH, or SOL Spot or Futures strategy in
an already active lead portfolio, after every incremental fee, slippage, tax,
compliance, support, disclosure, settlement, operating, and strategy-capacity
cost. Its public forward floor is zero. No account, portfolio, follower, rate,
copied close, payout, current after-cost profit, or deployment readiness was
accessed or proved; no credentials, account request, order, trade, fund, or
mutation occurred.

Do not use profit share to rescue an unprofitable or unsupported strategy;
apply, enroll, publish, or change a lead portfolio; solicit, incentivize, churn,
self-match, wash, or manufacture followers or copied activity; or place/copy
trades. The retained current API index classifies
`GET /sapi/v1/copyTrading/futures/userStatus` as `TRADE`, so never call it under
read-only authority. Advance only when an already active lead portfolio, an
independently accepted strategy, an authentic existing follower, an exact
profitable copied close, and explicit account-specific payout-evidence
authority all coexist. Every state change remains separately unauthorized.

Focused verification exposed and removed one older family test's coupling to
the mutable global accepted-edge count. New and prior family tests now verify
their artifact-local count transition and canonical registry binding without
forcing unrelated edits on the next valid discovery.

Canonical contract SHA-256 is
`90d61004b2f8c4f81a179a0180f3d9afa7c0872df6b9f5cbf80c92aa478e852b`;
rendered evidence SHA-256 is
`32a61a3f5fb00df8ca9b998d39878d5687492d35a454e219dd0104e7a4f9e692`;
edge SHA-256 is
`6a5acab5c5b9561fa08fedae5b782198db87e8aa8a5c6172f0c4e0fadc3ef7c0`.
Accepted edges become 29, ranked hypotheses remain 44, and terminal families
remain 64. Frontier SHA-256 becomes
`bc2db7e81a2e14fee68dc9f57041d226843fefabc0cc47e81db10985e04d84d3`;
registry SHA-256 becomes
`20f029bbdc6fc31a496f47f74ff3cf59c81b8cf89c522b41e80d72823861cb1a`.

## Binance BFUSD BTC/ETH/SOL funding-carry dominance bound

The next actual-trading audit stopped before new market data because a stronger
retained upper bound already exists. The broad funding contract charges 10%
annual opportunity cost per capital leg. The later Portfolio Margin sensitivity
optimistically deleted one complete leg, so it grants exactly the same economic
improvement as a 10% BFUSD APR and dominates every lower BFUSD rate before
conversion, redemption, eligibility, quota, collateral, tax, custody, or
operating costs.

An exact zero-request recomputation of the retained BTC, ETH, and SOL funding
roles found that both the last hash-bound 5.12% BFUSD last-day APR and an
optimistic 10% APR leave zero of nine training/validation/test role nets, zero
family-adjusted bootstrap lower bounds, and zero of 72 required bullish,
bearish, sideways, directional, choppy, and volatility slices positive.
Necessary—not sufficient—BFUSD APR thresholds are strictly above
22.04372931374745% for BTC, 22.18277961587428% for ETH, and
26.55978689762650% for SOL. Existing drawdown and positive-week-concentration
failures plus all execution and account gates still remain above those rates.

One current official rendered BFUSD page read confirmed that holding rewards do
not disqualify the holder from funding fees on their own hedging strategy and
that Futures, Spot, Funding, Portfolio Margin, Trading Bots, and Margin holdings
enter the reward snapshot. Its current numeric APR and collateral fields were
placeholders, so no current rate was admitted and no trigger fired. No venue
API, credential, account, book, order, trade, transfer, subscription,
redemption, fund, or mutation was accessed.

Do not resample the retained 17-symbol population or request books. Reopen only
with a new source-bound BFUSD APR strictly above the applicable necessary
threshold plus same-period reward history and distribution semantics capable of
recomputing drawdown and concentration, or a material funding, fee, execution,
basis, margin, or capital-cost change capable of clearing the retained deficit.
Canonical result SHA-256 is
`477a4db3c7f9c594ea8c351ce8f0a766280f062437c8089758d6137fbdd54d86`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
64, and registry SHA-256 becomes
`fe3116d77f82ef88f2ab929b8e71ec67ee029a7399a526e4b98edd9e45c81ef7`.

## Polymarket all-category realized maker-rebate scope extension

The current official Maker Rebates program is materially broader than the
retained crypto-only accepted overlay, firing rank 17's literal program-change
trigger. A frozen one-use public unauthenticated GET of the exact official
Markdown source retained 5,945 bytes. The source gate failed and must not be
retried: discovery exposed stale or differently rendered USDC and 25% Sports
values, while retained current primary bytes say pUSD and 15% Sports; Markdown
table padding, bold text, and escaping caused the other literal phrase misses.
Every discovery value is excluded from economics.

The retained bytes mechanically prove daily pUSD rebates, a 1 pUSD accrued
payout threshold, zero maker fees, per-market competition, a discretionary
pool fraction, and these current eligible-category terms: Crypto 20% rebate on
a 0.07 taker-fee rate; Sports 15% on 0.05; Finance, Politics, Mentions, and Tech
25% on 0.04; Economics, Culture, Weather, and Other / General 25% on 0.05.
Geopolitics is fee-free and excluded.

This is a scope extension of the already accepted exact-realized maker-rebate
identity, not a thirtieth edge. Credit only an exact positive owned daily pUSD
rebate from independently justified legitimate organic maker fills after every
inventory, adverse-selection, hedge, execution, latency, cancellation,
compliance, tax, custody, operating, and capacity cost. The public forward
floor is zero; no market-making strategy, fill quality, account eligibility,
owned income, profitability beyond the realized increment, or deployment
readiness is proved. No credentials, account, wallet, order, cancel, hedge,
trade, fund, or mutation was accessed.

Never use discovery values, create quotes or volume for rebates, infer queue or
fill quality, rescue a negative base strategy, double-count another reward, or
repeat or alias the consumed source request. Reopen public research only on a
material official fee, rebate, distribution, minimum-payout, category,
currency, or execution-architecture change. Account reconciliation requires an
independently existing owned legitimate maker fill and explicit read-only
evidence authority; every state change remains separately unauthorized.

Canonical failed contract SHA-256 is
`d59f9f93359ff82add45272ab43c02f095b87da334ada6ddcc240786a72a1bb0`;
retained Markdown SHA-256 is
`8d2c6562bd1b3376bc3fc1557a60efef5aa3c1d856c7f8dcc405139a07e9ba2a`;
canonical scope-extension result SHA-256 is
`d37aeac00dca154bbf0d676c3696a688bc7ee6cef9e8118730ffb5be05fb2550`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
64, and registry SHA-256 becomes
`44a63556d0a8d680661a82743fbdc0eb0bed2a50d75f1679689d72258af6ce41`.

## Binance Agentic Wallet and Alpha-versus-Spot rejection-first triage

The unregistered Agentic Wallet organic-conversion lead was tested only at its
source boundary. A frozen public official `market-order.md` GET retained 8,816
bytes and matched four of five preregistered phrases, including the exact
non-executing quote command. It failed closed because Binance's retained primary
failure wording differed from the discovery rendering. Do not rewrite, alias,
or retry the source contract, and do not admit discovery fee values. Reopen only
on a material byte-retainable official fee or quote change, or with an
independently required exact conversion, an active Wallet session, and explicit
quote-only authority; stop before swap.

The first local direct-path capture invocation failed during import, before any
request and before any output existed. The preserved corrected entry mode is
`uv run python -m tools.capture_public_source_contract` from the repository
root. This repeats the already-correct `AGENTS.md` package-import rule so the
same operator error is not repeated.

The next genuinely distinct lead was Binance Alpha-versus-Spot same-token
parity. One frozen complete public Alpha token-list GET retained 665 tokens.
Deterministic zero-book prequalification found 398 live Alpha rows, only five
live rows marked `listingCex` with a nonempty `cexCoinName`, and zero active
Spot pairs for those five across USDT, USDC, FDUSD, USD1, BTC, or BNB in the
retained 2026-08-29 Spot inventory. No price, book, account, credential, order,
transfer, fund, protected capture, or mutation was accessed. The current family
is terminal before books. Reopen only when a material token-list, Spot-listing,
or transfer-architecture change creates a source-proved exact transferable
overlap; then apply an optimistic cost bound before any depth request.

Canonical Wallet contract SHA-256 is
`3fe1995cd5a03b7df5e8e60cd5e66bfc35ca908e6367f8e392f782ccf514537b`;
Wallet source result SHA-256 is
`610c9c3c92add0ea5ebd02b6c0a2ab48005da3de312fe7b499a9aa211f022d86`.
Canonical Alpha token-list contract SHA-256 is
`38cc393602c78f1935fb5097ddb158da73bc6e5d0f01f9eb30e5522ec947d14c`;
token-list source result SHA-256 is
`a738d4e628a7e66d705e7f1d3d527bb64fc07ee988f7bd1527d1d4fddd746233`;
canonical parity prefilter SHA-256 is
`ee7653904a54848775236b72c8319e5fc6889b34753ea4ed2d964f085c5a6d85`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
67, and registry SHA-256 becomes
`5ec42c3de1890f6c9f5e4ecc027b892b3c6f7323eeb813a4db821c1f2b67d2cd`.

## Cross-sectional funding capital-sensitivity correction

The original unlevered two-leg capital treatment was explicitly audited rather
than treated as an untouchable rule. With zero new market requests, the retained
validation and test roles were combined into one continuous hold and credited
with interval-by-interval perfect foresight, zero switching, only 32 bips for
the two-leg entry and exit, and optimistic equal 5x leverage. Even this
impossible upper bound is -18.840831506849315068493150692 bips.

The necessary break-even leverage is greater than
12.9536779944653097611862238x before relative-price and basis PnL, liquidation
risk, and every omitted cost. Because any training-selected fixed orientation
is dominated by that oracle, no fixed-orientation implementation or price
request is justified. Reopen only on a source-bound material funding, fee,
execution, portfolio-margin, netting, or capital-treatment change capable of
clearing the full risk-adjusted deficit; leverage by itself is not a trigger.

Canonical sensitivity result SHA-256 is
`61a65f1f81b7109a6f959a53f8b780e88582b1338ec0cc512ef6784020da029f`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
67, and registry SHA-256 becomes
`d840ca77bf9f250a87fbc4d5a8423f98d94f1494a18d7863eebfc83812bc030f`.

## Binance fixed-value Gift Card false-discount rejection

The current official catalog exposed a previously unregistered,
direction-independent-looking dual-token fixed-value Gift Card mechanism. One
frozen public complete-reference GET received 8,000,456 bytes and passed all six
source phrases. A bounded hash-bound extraction proved that its Gift Card
family contains exactly six documented endpoints and no non-mutating exact
positive-discount quote.

Do not treat the endpoint title's parenthetical "discount feature" as an edge.
The operative example debits 100 USDT plus a minting fee and later redeems only
BTC equivalent to exactly 100 USDT. Its optimistic gross value difference is
zero and its net upper bound is the negative minting fee before every remaining
cost. The only creation operation that could reveal realized economics is the
state-changing `TRADE` `buyCode`; it also requires KYB, Funding Wallet balance,
and withdrawal-enabled API authority. No credential, account, token-limit,
verify, purchase, redemption, order, fund, or protected-capture request is
justified.

Reopen only after a material official Gift Card architecture or terms change
documents a non-mutating exact quote whose face value strictly exceeds payment,
minting fee, and every eligibility, transfer, expiry, conversion, custody, tax,
failure, and opportunity cost. Credentials or purchase authority alone are not
retry triggers.

The unrelated full response contained public illustrative API-key values and
private-key blocks and therefore was removed before staging. The request journal
and capture result preserve its exact byte count and response hash; the exact
2,716-byte Gift Card section had zero secret-pattern matches and is retained
byte-for-byte. Search the retained Agent Native index and safe sections before
another large documentation request, and secret-scan the response before staging.

Canonical source contract SHA-256 is
`49fd6dd8121e9132335669eb75f33d28f799f2c7471c5ba0a0042313a6d812ec`;
source result SHA-256 is
`631e9ec81519a1adbe970e0820ebcf53aad289c9b5965876d9c7b273bd3c2180`;
raw response SHA-256 is
`c785b773eb2f36e87fd077891461320e60cb1aeedc8cec42e268e134e1b68d8a`;
canonical terminal result SHA-256 is
`316e3182ce6a33287463d1c9c6d32a9bd3066bb49740c68d83e5ad717bf36868`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
68, and registry SHA-256 becomes
`4814e73d89db659d602ad3ed5901f5b94add2b70c8daa51a635ff275ffebddb5`.

## Polymarket Bitcoin September 4 fixed-NegRisk depth rejection

The newly deployed eleven-bin "Bitcoin price on September 4?" event fired rank
31's literal new fixed-NegRisk trigger. A frozen exact Gamma GET retained the
complete active event and exposed a 3.0075 pUSD displayed all-YES sum plus eleven
positive rejection-only one-NO conversion identities. This justified exactly
one separately frozen public 22-token CLOB batch; no discovery price entered
the calculation.

The batch completed and is durable. The original runner then stopped before
writing a result because it had reused an older event's exact 5% fee schedule,
while all eleven retained Crypto markets use a 7% taker rate and 20% rebate
fraction. No request was repeated. A frozen zero-network adjudication against
the immutable books found the best five-share path at -3.730 pUSD under zero
fees, -3.81628 pUSD after the exact retained current taker fees, and -4.07770
pUSD after one adverse tick on every leg, with zero profitable paths in every
view. The batch also failed freshness at 141,591 ms age and 129,032 ms skew, but
freshness cannot reverse the already-negative zero-fee optimistic bound.

The reusable book runner and retained adjudicator now accept a contract-bound
exact event fee schedule, preserving historical defaults only for already
consumed contracts. Do not repeat this event's Gamma or book request, refetch to
repair the consumed fee/freshness failure, or request adapters, chain state,
accounts, credentials, orders, or funds. Advance another literal trigger.

Prefilter contract SHA-256 is
`4f931ca8c7c94d68c6fc36ea77868bf6e285f4f33d062aed491d140cc048b5c8`;
prefilter result SHA-256 is
`a8fa32e0ec3cc51c35670df5c2dc1c5b0fa1dc02ba281958c56ded4953922cbf`;
book contract SHA-256 is
`96fabc2319bb25cf9ee000edd8c9dbe72f3d20ce82ee891ffe4c97993e6917a6`;
retained adjudication contract SHA-256 is
`197b4b8ebe2c4fb1f8c440aa3e08b9aa22aa60424471a677e581688f6031c0ed`;
canonical terminal result SHA-256 is
`426b53310b6f46ea39312b4d06f404453ceac2b98863efeee91aa9d592120208`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
69, and registry SHA-256 becomes
`087468040a17520bde05b3dd2e8bf2df94a3608176d9aad6b190382e043d858e`.

## Binance Spot Price Range Execution Rule dominance rejection

A current official-source sweep exposed an unregistered, direction-independent-
looking Spot execution mechanism. One frozen public official FAQ GET retained
7,999 bytes and passed all seven preregistered semantics phrases. No Binance
venue endpoint, live configuration, credential, account, order, trade, fund, or
protected Polymarket capture was accessed.

The rule constrains taker execution around an exchange reference price and
expires out-of-range residual quantity. It is a safety control, not a positive
standalone edge. For a buy with worst acceptable price `P`, an otherwise
identical marketable `LIMIT IOC` or `LIMIT FOK` with limit `P` cannot fill above
`P`; the sell identity is symmetric. The exchange rule can impose only an
additional, possibly looser cap. Its incremental payoff upper bound versus that
user-bounded comparator is therefore zero for every market direction and every
live multiplier or reference price.

Do not request or poll `GET /api/v3/executionRules` or `referencePrice` merely
to compare against an avoidably unbounded `MARKET` order. Query exact current
configuration only inside a separately frozen candidate whose economic decision
materially relies on unbounded taker execution or exact residual-expiry
semantics, and then model `EXECUTION_RULE_PRICE_RANGE_EXCEEDED`. Reopen this
family only if an official rule or order-type change adds positive cash
consideration, a strictly better executable price, or protection unavailable
through a user-bounded order.

Canonical source contract SHA-256 is
`53381cf5bc5e8328283dcf06efdcdb7630466b8becb25451405f219588ba7569`;
source result SHA-256 is
`2866f68ea2dd0b7fb460fd132da19fc278cab30e6b8cb1db55f92f857a3281a3`;
raw FAQ SHA-256 is
`ec6fa180dc99ea1f1846f8e310caa958c8f0ff33fec65a1b94160113f25259d7`;
canonical terminal result SHA-256 is
`6716c320effd97f20ebe84536366e0308ca7089b1ef15d4c6f601c232182a10d`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
70, and registry SHA-256 becomes
`d2d24730180c4b4c9182a4f694a906cab819e1f93633ac04ddc834af5f6d3d31`.

## Polymarket Ethereum exact monotone threshold-ladder rejection

The current official event surface exposed a previously unregistered exact
within-event crypto payoff identity. In a ladder whose common rules resolve YES
only when the same Binance close is strictly above its threshold, buying
`YES(L)` and `NO(H)` for `L < H` pays 1 pUSD below or at `L`, 2 pUSD strictly
above `L` through `H`, and 1 pUSD above `H` when every independent condition
applies the common source rule consistently. This is an optimistic one-pUSD
rule-consistent floor without predicting market direction; independent disputes,
cancellations, inconsistent resolutions, and settlement delay are additional
downside and were not proved away.

A frozen one-use Gamma GET of "Ethereum above ___ on September 4?" retained all
11 exact active thresholds from 2,000 through 3,000 and exhaustively tested all
55 lower-higher packages. The best rejection-only pair was `YES(2,900) +
NO(3,000)` at 1.0015 pUSD, already 0.0015 pUSD above its floor before fees. Zero
packages cleared the strict gross gate, so no book, fee, on-chain, account,
credential, order, fund, or protected Polymarket capture request was justified.

Do not repeat, narrow, or refetch this event, and do not cherry-pick the current
BTC or SOL sibling ladders after the complete ETH population failed. Reopen the
family only for a distinct active BTC, ETH, or SOL ladder outside every consumed
event with complete exact common rules and at least one rejection-only Gamma
displayed `YES(L) + NO(H)` sum strictly below the optimistic 1 pUSD floor. Gamma
remains rejection-only; any candidate still requires source-bound exceptional
settlement and independent-condition resolution risk plus a separately frozen
exact two-token batch, contract-bound fee schedule, minimum size, freshness,
adverse-tick stress, and all costs.

Canonical contract SHA-256 is
`0737aa5e76be4151213f1a6174eca525e32ec7c46e7e4347842e6ac41c8a7331`;
canonical result SHA-256 is
`42c122e54bd9a7299cc9e739724fabd4cd76716dd3bdfe76c039a1bab8014d2a`;
raw Gamma SHA-256 is
`84a0536e067b5f72a5a4c9fc1ac4a215316b51a742a5f7e781ab37e7fbe5b1be`;
request journal SHA-256 is
`b06553b207ad6c276a08262c08319836b36ec7523f9aa8f9c6c902f693414944`.
Canonical zero-network terminal adjudication SHA-256 is
`25f90c75b9d8657e44b27ebab8dd4c26fb3434a306a6dbd9e35fdc2fdd53419d`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
71, and registry SHA-256 becomes
`fdfaac8e3bc873bd07e44e89efb290ebc0bbd7766157d11b6b614875345adead`.

## Polymarket Bitcoin cross-event range/threshold boundary rejection

The newly deployed official "Bitcoin above ___ on September 4?" surface and
the retained "Bitcoin price on September 4?" fixed-NegRisk event refer to the
same Binance BTC/USDT one-minute close at noon ET. The apparently exact mapping
between a strict-above threshold and cumulative upper range bins is false at an
observable boundary: at `x = T`, threshold YES is zero while the higher range
bin is one. Rank 31's deterministic subset-indicator equality trigger therefore
does not fire.

A weaker direction-independent coverage package survives under consistent
rules: `NO(T)` plus every range YES beginning at `T` or higher pays one pUSD
below `T`, two at exact equality, and one above `T`. Independent condition
disputes, cancellations, resolution divergence, and delay remain additional
downside. Current economics were deliberately not manufactured: the exact BTC
range Gamma request is consumed, the official-page values were discovery only,
and no venue endpoint, book, fee, on-chain, account, credential, order, fund, or
protected-capture request occurred.

Do not refetch either September 4 event or use discovery prices. Reopen this
coverage relation only on a distinct nonconsumed BTC, ETH, or SOL range-plus-
threshold pair with the same exact source observation, complete boundary and
exceptional-settlement rules, contemporaneous frozen complete populations, and
at least one rejection-only displayed `NO(T)` plus cumulative upper-range YES
sum strictly below the optimistic one-pUSD floor. Any survivor still requires
exact simultaneous depth, fee, size, freshness, adverse-tick, cancellation,
resolution, and all-cost proof.

Canonical boundary adjudication SHA-256 is
`e7290745dac6aac63a0363a98ac9596280548de7093e4ff7261f34cc95eb3ca8`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
72, and registry SHA-256 becomes
`f0f70eb60bc1ca13d899a3c01930bdc37838c874e875636722c5bd2039f1eec1`.

## Polymarket Solana September 4 cross-event coverage rejection

The official September 4 Solana range and strict-above events were a distinct,
nonconsumed same-source pair and fired rank 31's new coverage trigger. A frozen
two-request Gamma screen retained both complete eleven-market populations and
tested both valid directions at every one of the ten shared boundaries. Of all
20 packages, only `NO(above 150)` plus `YES(higher range 150)` cleared the
rejection-only displayed gate, costing 0.9855 pUSD versus its optimistic
common-rule one-pUSD floor.

The precommitted exact two-token book batch rejected the apparent 145-bp
headroom. Five shares per leg cost 5.04 pUSD against a 5 pUSD floor at actual
asks, a 0.04 pUSD zero-fee loss. One adverse tick per leg increased cost to
5.05 pUSD; the five-tick package was not fillable below one pUSD per share. The
book timestamps were also 4,655,213 ms old and 7,949 ms apart, failing both
five-second gates. Since actual zero-fee depth was already negative, no fee
endpoint was requested.

Do not repeat either Gamma event, the exact book batch, or repair the stale and
skewed snapshot; do not request fees or cherry-pick the now-consumed ETH/BTC
siblings. Reopen only for a literal distinct nonconsumed same-observation pair
whose complete frozen population contains a strict displayed sub-floor package,
then require fresh synchronized exact depth, fees, resolution risk, and every
cost before any account or order-capable work.

Prefilter contract SHA-256 is
`38b02e051962f979c90564318928bcb216c13aca5fe6847eceb8d0398a9abe23`;
prefilter result SHA-256 is
`8efba9e824fb125bd4a5be654704c6f47d62ddc1fd8b38006fad06ac52247417`;
book contract SHA-256 is
`5f0ff60c030abef0a5cc68b0926090de29dc3cbc3f9c55bb9ccd9cc20f68d331`;
book result SHA-256 is
`5bc1e557a85af2588d7b319476e7ef9d4f2afe2c9103100f2a56a41864a9ef81`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
73, and registry SHA-256 becomes
`932ff844edcda13efc17bd48569c1cd4b5f25abd0f2c0c3747dae696d1b19649`.

## Polymarket Solana September 4 sponsored-reward overlay rejection

The retained `NO(above 150)` plus `YES(higher range 150)` package was reused
only as immutable input to a distinct rank-17 liquidity-reward question; its
stale books were not refreshed. At the exact 50-share minimum, retained
one-tick maker quotes reconstruct a 48.65 pUSD combined cost, 1.35 pUSD
optimistic both-fill gross, and 48.20 pUSD maximum one-leg orphan loss.

The frozen runner made exactly two public unauthenticated read-only requests:
one `sponsored=true` reward GET for each exact condition. Both returned HTTP
200, terminal `LTE=` cursors, and zero rows. Thus the maximum public remaining
pool is 0 pUSD. Gamma's minimum-size and spread fields are eligibility metadata,
not funding evidence. No Gamma, book, fee, on-chain, account, credential,
order, fund, or protected Polymarket capture request was made.

Do not repeat either condition or refresh these books. Reopen only on a material
exact funded-program change. For every future maker-reward overlay, first prove
from retained economics and exact sponsored sources that an impossible 100
percent share of all remaining pools strictly exceeds maximum minimum-size
orphan loss; otherwise stop before books and account work. Invoke the reusable
runner as a module (`python -m tools.screen_polymarket_retained_cross_event_rewards`),
not as a file path.

Contract SHA-256 is
`6064ab6bb733f82dab2ef3fc8f9ea3e4ffeebb3cc2f005f91d41275e6aa1a2ae`;
result SHA-256 is
`97471d6fe9148ba2e4fd818902e0fb33e4f796c7115cd6494c2a35be0bbebeaf`;
both raw reward payloads have SHA-256
`cb1463591af370d3e3eb39e1dc5821bb1ae64d7dde15ba0748011273b32e9148`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
74, and registry SHA-256 becomes
`67a955835c85b8d4d7f11d34bf67e6ed4932a09ab8d0ad1c54e52941cf36bb40`.

## Polymarket September 6 BTC/ETH/SOL range-threshold delta rejection

At `2026-08-30T16:08Z`, ranks 1 through 30 had no in-scope public retry trigger
and the protected holding-yield and GLWUSDT boundaries had not arrived. One
documented public unauthenticated Gamma keyset GET therefore tested whether a
new rank-31 crypto surface had appeared. The newest-first response retained 100
open crypto events from `16:07:16Z` through `15:37:42Z`; 85 were newer than the
prior `15:46:52Z` checkpoint. Because the retained page crossed that cutoff,
the new-event delta is complete even though the API returned a cursor.

The delta exposed six simultaneous September 6 events: one complete range and
one complete strict-above event for each of BTC, ETH, and SOL. Before accessing
their displayed prices, a frozen hash-bound offline contract preregistered all
labels, exact Binance close rules, shared boundaries, both valid coverage
directions, and global tie-breaking. It evaluated 52 packages: 20 BTC, 20 ETH,
and 12 SOL. Zero were strictly below their optimistic common-rule one-pUSD
floor. The global best was BTC `NO(above 88,000) + YES(>88,000)` at exactly
1.0 pUSD; ETH's best was 1.095 and SOL's was 2.0.

No exact-event refetch, cursor continuation, book, fee, on-chain, account,
credential, order, fund, or protected Polymarket request occurred. Do not
repeat this delta or its September 6 events, follow the retained cursor, or
request books for an exactly-at-floor row. A later simultaneous sibling set
must be screened completely from one retained population and only its global
strictly sub-floor row may authorize a separately frozen depth batch.

Contract SHA-256 is
`f2642e0577b422e62e7c4df30eb16ff85c09b741017146dae233c782963b928b`;
result SHA-256 is
`cc4cb32adcafba2da3d48cc8325d7af8a2bfd39c0892ceeb1ec1def939e173f9`;
raw delta SHA-256 is
`4f9aadb6a95bdf2612845b3e3bc96146cc1ea5f23b3cb6bf8815ccb43c8ce087`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
75, and registry SHA-256 becomes
`bb1b929151474ac48c146e335ce0533f459fb09705ab4feae4c3e5bb76dd81e2`.

## Polymarket September 6 BTC/ETH/SOL threshold-ladder delta rejection

The same immutable September 6 crypto delta contained a second distinct exact
payoff family that had not yet consumed its trigger: within each strict-above
ladder, `YES(L) + NO(H)` for `L < H` has an optimistic common-rule one-pUSD
floor without requiring a market-direction forecast. A separately frozen,
hash-bound, zero-network contract reused only the retained raw bytes and
exhausted all 55 pairs in each complete 11-market BTC, ETH, and SOL ladder.

Zero of 165 displayed packages were strictly below one pUSD. The deterministic
global best was BTC `YES(68,000) + NO(70,000)` at exactly 1.0 pUSD, leaving no
gross headroom before fees, spread, latency, independent-condition risk, or
capital cost. Therefore no book, fee, on-chain, account, credential, order,
fund, or protected-capture request was justified.

Do not repeat these three ladders or request books for the at-floor row. Reopen
only on a literal distinct nonconsumed complete BTC, ETH, or SOL ladder whose
frozen rejection-only screen contains a strict sub-floor package. Before any
new market request, enumerate distinct payoff families already testable from a
retained complete population and freeze each one independently.

Contract SHA-256 is
`207e04f6c773adb04b73ee55417ad15cbfaa00e0ef2bf64e05a9ff94ab89b73f`;
result SHA-256 is
`c505584dafa3391fc17647fb897d03a402ec4523fcfc5205960383e4b3967fdf`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
76, and registry SHA-256 becomes
`dd82fabad42807a602c020f43d58dc7c7c37bad67b41eda47004dc4f64afd5a2`.

## Polymarket BTC/ETH/SOL exact interval-composition rejection

The retained delta exposed an unconsumed direction-independent identity across
three adjacent 5-minute TWAP markets and their exactly covering 15-minute
market for each of BTC, ETH, and SOL. Prior hash-bound prospective evidence
proves source-continuous opening and closing TWAP values. Therefore three `Up`
intervals imply covering `Up`, and three strict `Down` intervals imply covering
`Down`. Buying the complement of each premise plus the implied covering outcome
creates two exact rule-consistent four-leg packages per asset with a one-pUSD
payout floor.

The v1 offline wrapper stopped before economic rows because Gamma's nested list
fields were JSON strings rather than native arrays. Zero requests occurred and
no output was created. The failure is preserved rather than hidden. After an
exact representation preflight, a new implementation-hash-bound v2 contract
changed only the field decoder and reused the same immutable raw bytes.

The corrected zero-request screen exhausted all six packages. BTC, ETH, and SOL
up-chain costs were each 1.990 pUSD; down-chain costs were each 2.010 pUSD. Zero
were strictly below the one-pUSD floor. No book, fee, on-chain, credential,
account, order, fund, or protected-capture access was justified. Do not repeat
the 11:45–12:00 ET set. Reopen only for a distinct exactly aligned interval
partition with source-proved value continuity, complete equality semantics, and
a strict rejection-only sub-floor package.

V1 contract SHA-256 is
`925045d42cac0ba8b4ff0c7cdb6c0c07c70e02fbbf4c3d8ec6389559850152ba`;
failure SHA-256 is
`c7d758f352be3d2ae9d1f4c2957f82fa3926a7a7837f4db0ae315397182cdc83`;
v2 contract SHA-256 is
`ceaf67b3de430e41369470188a466d30f1dc6f0879ab94ab9361acc59488f449`;
result SHA-256 is
`6ef1b3acc9c4a234bc7826395bca02397351c9e168d7d010145752dad33b7747`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
77, and registry SHA-256 becomes
`869416f212c475915411f2aa4b70ae08b61d89fc0c1e8e54843c941b67b82241`.

### Interval-composition persistence and settlement follow-up

A separately frozen one-use historical persistence test selected 12
non-overlapping BTC/ETH/SOL aligned sets from retained August 26 markets: 48
exact conditions and 24 packages. Its only public unauthenticated Data API
trade request returned HTTP 408 after about seven seconds and exposed zero
trade rows. The raw 48-byte error and request journal are retained. Never retry,
split, narrow, paginate, reorder, or alias this consumed population.

A distinct zero-network contract then audited every complete aligned set across
the six immutable retained market sources. All 50 packages in 25 sets and 100
terminal markets paid at least the one-pUSD floor; observed payouts ranged from
one to three pUSD. This materially supports the payoff identity but proves
neither sub-floor acquisition, atomic execution, fees, capacity, owned fills,
profit, nor deployment readiness. Reopen only for a future distinct aligned
population that first passes a strict rejection-only sub-floor gate, then freeze
one prospective exact live CLOB package capture.

Historical-trade contract SHA-256 is
`799d310c2fd56098fb8cd208e79dc88a338462dfb6ee0ee4f2c9db28d951c65d`;
failure SHA-256 is
`5dcacfa8c9f1b953a9fa28e380e43cfb8c0107361426f67d5b4adeb83dabdfb1`;
settlement contract SHA-256 is
`268301993f1ace29a9bf99d936be5f74101e26e72b61b25e2a272d7ee7146747`;
settlement result SHA-256 is
`81a86ae5a71708516b9f23fbbae0b51cbf337691c64c655e4eb03508a032d84a`.
The exact pre-format runner bytes remain reconstructable through lineage
artifact SHA-256
`c8fde55efcd61e8fa376d3b5184ca6013f816cf8d74f1f706aad11032e0c297b`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
78, and registry SHA-256 becomes
`809c82d3233fb76b8fa41c1ba1cd7e7cb168ee8932a71098870d5f5b7e2ab04a`.

### Exact GLW time-gate instruction correction

The accepted market-independent yield frontier had a stale early summary of
the GLW terminal reconciliation gate. The frozen executable contract and full
rank-34 registry row are authoritative: the one-use request is prohibited
before `2026-08-31T00:10:00Z`, not merely before midnight. The frontier is now
corrected to the exact later instant.

Two new discovery-only search batches found no material official September
2026 bStock dividend or Special-funding episode. Do not convert that null search
into adaptive requests. Immediate research spend remains zero until an exact
trigger fires. Corrected frontier SHA-256 is
`1c346600a0bc2a439aa868fba51ed0bf939a48011dbc021ef107bcb5c9771040`.

### Binance funding-estimate known-at-entry rejection

A primary-literature lead, *Funding Timing and No-Arbitrage Bounds in
Decentralized Perpetual Markets*
(`https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6805838`), suggested testing
whether displayed perpetual funding could be treated as a fixed transfer at
entry. A frozen one-use public reconciliation used three Binance funding-history
GETs and the retained
2026-08-25, 2026-08-27, and 2026-08-29 BTCUSDT/ETHUSDT/SOLUSDT snapshots. Across
nine observations at 16,264-23,801 seconds before funding, six displayed values
changed before settlement. Maximum absolute change was 0.6266 basis points.
The three exact matches were all the standard positive `0.00010000`
interest/clamp plateau; they do not prove a general lock. That value is not the
funding-rate cap.

The hours-ahead estimate-lock premise is terminal and must not be promoted or
retested on this population. No books, accounts, credentials, orders, funds, or
testnet endpoints were touched. Reopen only on an official fixed-at-entry rule
or a separately preregistered near-finality executable study whose conservative
guaranteed transfer clears all costs.

Contract SHA-256 is
`46d0bbf9e48b090332653d8b5cfe38b2350fc8ce3d8b2fd0100addc10916c8df`;
result SHA-256 is
`d1a75d29bc7d48154f2006a335a401d7451a6ad58c36b12127b387581f3c3ac3`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
80, and registry SHA-256 becomes
`d49940b750e9fc4d8416840185136d1fcefbc734d12473ad4c00fe76ea2d8f89`.

### Near-finality funding capture prefilter

A zero-network, hash-bound audit reused the retained 500 realized settlements
per BTCUSDT, ETHUSDT, and SOLUSDT from 2026-03-13 16:00 UTC through 2026-08-27
00:00 UTC. Maximum absolute funding was 1.2276, 2.2976, and 3.9810 basis
points, respectively. Zero of 1,500 rows exceeded even a 4-basis-point gross
diagnostic, and none approached the already-frozen 32-basis-point two-leg
round-trip stress.

Do not spend a near-finality stream or book capture on the current evidence.
This is a research-spend rejection, not a universal statement about future
extreme funding. Reopen only when a public displayed absolute scoped rate first
exceeds 32 basis points, or when material exact account-fee and execution
evidence supports a separately frozen lower all-in gate. No network, account,
credential, testnet, order, fund, or protected-capture access occurred.

Canonical result SHA-256 is
`894cf7c6903a90a4225ddb8a264df039fffaafa1370885a298163a055ce950ff`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
81, and registry SHA-256 becomes
`61e32583121d421bb15294fd2c092443b611a79a2f8e799c3292395e9b8a0bbd`.

### Binance Delta Mode alias reconciliation and request-efficiency correction

The current official Agent Native index exposes Portfolio Margin Pro Delta Mode
through a `USER_DATA` status GET and a `TRADE`-class switch POST. This is not a
new capital-efficiency trigger. The retained Portfolio Margin sensitivity had
already granted the strictly stronger optimistic counterfactual of deleting one
complete capital-opportunity-cost leg; zero symbols remained positive in
training, validation, and test, and every family-adjusted bootstrap lower bound
remained negative. Endpoint presence supplies no quantitative offset, collateral,
eligibility, liquidation, or cost semantics. No credentials, status request,
mode switch, books, account state, orders, transfers, or funds were used.

Do not create a separate Delta Mode hypothesis or query account status from the
name alone. Reopen the existing family only on a material official semantic or
quantitative margin-offset change capable of clearing its retained deficits, or
after exact public semantics plus explicit signed GET-only authority support an
independently required account-specific comparison. A redundant same-day GET of
the exact Agent Native index was detected before staging and removed; future
public research must deduplicate the canonical method and URL against retained
same-day successful captures before access.

### Polymarket Sports maker-rebate source-conflict adjudication

The literal official-source conflict trigger fired. One exact refresh of the
canonical Maker Rebates Markdown was byte-identical to the retained same-day
source and states a 15 percent Sports maker rebate paid in pUSD. One separately
frozen exact developer Fees Markdown request also states 15 percent for Sports,
but describes fees in USDC. Its frozen source gate failed only because two
expected phrases had changed or disappeared; the raw response remains retained
for offline conflict adjudication and the consumed request was not retried. The
previously documented Help Center surface reports 20 percent for Sports.

Therefore the existing all-category maker-rebate overlay remains accepted only
as the same exact-realized-cash edge: credit an exact positive owned payment in
the asset actually received, joined to independently justified legitimate
organic maker fills, exact effective market parameters at match time, conversion
basis, and every incremental cost. No public 15, 20, or 25 percent Sports rate
and no pUSD or USDC label may be used for forward profit. The public forward
floor remains zero; this is not a new edge, a market-making strategy, or a
deployment-ready claim.

Maker Rebates refresh contract SHA-256 is
`90157b82de207c7a03b704b2dd6a86a7d04ca8798bcaa5588c1d71a7aaf23b5e`;
its result SHA-256 is
`fe8e63e5fb766614eca5af040492a1ff646d32c1ca9ceab645688fd295554824`.
Fees cross-source contract SHA-256 is
`fe8bce8a7ea95f5a2aacd9126b74a4a957fcc725667496dd7f5f9d822c5a9eb1`;
its source-result SHA-256 is
`eb03bc4614095eca44ed73e21181beec2a438851c8dc61cb5538ada56cfe7dd1`.
Canonical adjudication SHA-256 is
`3313a90b2257207c292eab289cc4199db7def8ad35361ad21fcabc435c0bc6a2`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
82, and registry SHA-256 becomes
`2dd5aeb4e2649c12e6f62105dfb0e6472539fc080a5fb41507f02ee3072b189e`.
No account, credential, signed request, book, order, funds, testnet endpoint, or
protected capture was touched.

### September 6 CFB exact monotone-payoff screen

The first distinct active CFB population outside the consumed September 3-5
windows fired rank 30. One frozen public Gamma keyset request for the complete
`2026-09-06T00:00:00Z` through `2026-09-06T23:59:59Z` start-time population
returned 24 events with no cursor. Fifteen rule-complete events produced 31
exact full-game margin or total monotone relations and two rejection-only Gamma
packages below their one-pUSD payoff floors.

The deterministic best was Mercyhurst +29.5 plus New Mexico State -28.5 at a
displayed 0.995 pUSD per paired share. Its exact five-share two-token batch then
cost 9.8 pUSD for a guaranteed 5 pUSD floor, losing 4.8 pUSD before fees. The
books were also 1,763,317 ms old and 337,138 ms skewed. The two-tick stress had
insufficient complete depth, so no fee request was justified. Do not refetch the
event, repeat or narrow the September 6 population, or select the other observed
candidate after the precommitted winner failed.

Catalog contract SHA-256 is
`91954283c2cfb0ce9647e3c108ea9bbaa2c06a4abd55e10c8fc67ddb7e14e6ba`;
catalog result SHA-256 is
`9ca30ccdce00d7a0928fea85eab35e6da2dbdbb9b4c9b339d3a9f5ee1151bc02`.
Metadata contract and result SHA-256 are
`23bc4465eda396bdc56fb14319af93f080051f7c5dc67b926ff223fbac73a1f0`
and `e8bf6417b0c296bffb04a998d18959a175f5c7f3a0961c54e82c2a6da1ddb97b`.
Package contract and result SHA-256 are
`28756a8e168ef943a78886cb0e30cf2943e92db7fb82bcc13c050c08120b6240`
and `3cb7ae40ad0397d13bb2dfd5cae43ff8d4a815a18f076fe343e144ac8f7df1e2`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
83, and registry SHA-256 becomes
`1aad977a7ca40e38e040b24e99518aea23b4f29d7c961aa0b9143f0ad38b8a54`.
No credentials, accounts, orders, funds, fee endpoints, testnet endpoints, or
protected captures were touched.

### Cross-sectional funding literature alias correction

A new SSRN discovery lead, *Risk Control as the Durable Edge*, describes a
cross-sectional dollar-neutral funding-carry sleeve. The rendered official
abstract is discovery only and was not used as a source-bound economic input:
it reports three losing calendar years out of six, says the steady funding
coupon is only 6-15 percent annually while the crowded-name price spread makes
and loses the rest, and explicitly disclaims reliable profitability. Its exact
symbols, selection, turnover, fees, and price-spread rules were not retained.

This does not create or reopen a strategy family. The existing BTC/ETH/SOL
Binance funding-dispersion dominance bound remains stronger for the scoped
funding-only question: impossible interval-perfect-foresight orientation, zero
switching cost, only 32 bips entry and exit, and optimistic equal 5x leverage
still lost 18.840831506849315068493150692 bips across validation plus test
before relative-price, liquidation, and operating costs. No market request,
model, backtest, collector, credentials, account, order, funds, testnet endpoint,
or protected capture was used. Search the new triage before implementing any
literature-described cross-sectional or dollar-neutral funding sleeve.

Triage result SHA-256 is
`919f4183dc70edd28158f9729872de1701a081bd579cd31f80f3761029fd1c18`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
83, and registry SHA-256 becomes
`3aa7aef92c7ed41622aef16d0cea9f45f82925813d590a27d278f85a516eff32`.

## Binance Yield Arena discovery routing correction

Yield Arena is not a distinct cash-flow mechanism or a comparable headline APR.
The repository already retained the current August 26 article; a subsequent
search rediscovered it because the surface was not explicitly routed. It mixes
Simple Earn, ETH/SOL staking, Dual Investment, and other products whose exact
payoff, cap, term, eligibility, redemption, and risk must be adjudicated in
their existing families. Search the exact article code, product, asset, and
payoff identity before another web or signed request.

The official Agent Native index classifies
`GET /sapi/v1/earn/arena/activities` as USER_DATA. Do not call it to browse for
an edge. It may be used only for an exact account/product reconciliation after
explicit GET-only authority exists. Do not build an Arena-level collector or
reopen a family because the Arena brand or `Up to` headline changes. Canonical
routing addendum SHA-256 is
`b01792aabe04989b4e65fb5ae00719249fa6186f7c5bd345c59a3ea9b4d8ff66`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
83, and registry SHA-256 becomes
`8262b29c4ba9b6eb322beb91d18a9c44756355b451374292904aa569e7c81941`.

## Polymarket September 7 CFB monotone rejection

The current primary SMU vs. Florida State event page supplied the exact
nonoverlapping rank-30 trigger: a September 7 CFB game with moneyline, spread,
total, active purchase surfaces, and complete resolution rules. One frozen
public September 7 keyset GET returned exactly that event, no cursor, and one
machine-proved moneyline/spread monotone relation.

Its rejection-only Gamma sum was 1.09 pUSD—SMU 0.59 plus Florida State 0.50—
against a 1 pUSD minimum payout. The optimistic floor was therefore negative
0.09 pUSD before depth, fees, or any external cost. Zero candidates cleared the
strict sub-floor gate, so no book or fee endpoint was called. Do not repeat,
paginate, narrow, refetch, or request books for September 7. Contract SHA-256 is
`391356fc6d94e3e6c5407502afc94f81be4d4a3ea6192c1368dace613644bf19`;
result SHA-256 is
`5cba2d835c600d8eec6b8f27a7010a535b72292680d5a558fb44f29316a3c796`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
84, and registry SHA-256 becomes
`7c4c4ecc062a4cda012958571fcc477e304d38e0f00f8c67aecd37ae4a031645`.

## Polymarket soccer exact-score implication rejection and prefilter correction

A zero-network retained audit tested all 16 explicit exact-score implications
in the Los Angeles Galaxy versus New England Revolution event pair. Under the
aligned regulation, source, postponement, and cancellation rules, buying NO on
an explicit score plus YES on its implied full-game result has a common-rule
one-pUSD floor. The first representation contract stopped before economic rows
because its exact cancellation literal omitted the retained word `to`; the
failure is preserved and v2 changed only that literal.

V2 found ten apparent strict candidates from Gamma `outcomePrices` and selected
NO 0-0 plus YES draw at 0.750 pUSD. The one frozen current two-token book batch
rejected it decisively: five shares cost 6.20 pUSD for a 5.00-pUSD floor before
fees. The oldest book was 4,361,793 ms old and cross-book skew was 4,104,742 ms.
No fee request was justified, and the exact event may not be refetched, retried,
or replaced by another retained candidate.

The consumed result exposed an avoidable methodology defect. The retained
exact-score object already reported a YES best bid of 0.01, implying a 0.99 NO
ask rejection proxy, while the draw market reported a 0.27 YES best ask. Their
1.26-pUSD side-specific proxy already rejected the package; the 0.505
`outcomePrices` NO value was midpoint-like diagnostics, not acquisition cost.
The corrected v3 zero-network screen uses YES `bestAsk` plus `1 - YES bestBid`
for the NO rejection proxy and rejects all 16 relations; the best is 1.26 pUSD.
The generic two-leg verifier now also enforces oldest-book age in its final
candidate boolean, supports NegRisk legs, and supports markets with null lines.
`AGENTS.md` now prohibits using `outcomePrices` alone to authorize books.

V1 contract and failure SHA-256 values are
`5b53f0298a5eaaae7441a23f26c3dd22174d1d9657894fef503079142c5876bb`
and `9bc9712f78c0553dc30a02a0b6bde2d8cc88c269b833611207dd61a21865e0a3`.
V2 contract and result SHA-256 values are
`2dd021d474e3af9b987bde8163b78658d8c09cf92120d1553205fdeabc99703b`
and `f37cca192e911922e102c6284353694f9d7569f78f53ed2466ac2264817d8138`.
Book contract and result SHA-256 values are
`0b23820f6438e78e93ff0234f400f35c88a506d38630a4a5ff03cbd594ff70cc`
and `ba775fe53af1d69c9db3239a834e7e86a3f4e08837d6967e0da7eb32f19da8fa`.
The retained journal-lineage SHA-256 is
`3aa77d400d2e5822707e20f23775de0a40419da210f79aa86d8ead9115eb43a1`;
future two-leg results bind the final journal hash directly.
V3 contract and result SHA-256 values are
`66ceb9889b95dce357614a9bca36bdfbb2895b4b112066ad4edc149d1293469f`
and `aeb52c0bcd375fb0c31282cf8be92b9a6e6b93de0d8e7a0ac2c01bb579091386`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
85, and registry SHA-256 becomes
`7c4d62403212f9826a8bf9f972647a3c56df6671d78cab99af35b9781e8da115`.

### Retained soccer structural graph rejection

A distinct zero-network graph then exhausted every supported soccer relation
in the same immutable retained page: 11 main/exact-score pairs, 10
main/first-to-score pairs, and 7 main/more-markets pairs. The graph covered 305
rule-proved packages across exact score to full-game result, Neither-first to
draw, monotone full-game totals, BTTS to Over 1.5, and Under 0.5 to draw.
Cancellation was explicit in every payoff family, including 50-50 totals and
BTTS settlements.

Side-specific `bestAsk` or complement-proxy evidence was complete for 197
relations; zero were strictly below their one-pUSD floor. The deterministic
best was NO exact FC Dallas 1-3 plus YES Sporting Kansas City at 1.15 pUSD.
The other 108 relations lacked at least one side-specific field and remain
fail-closed; do not fill them with `outcomePrices` or adaptive requests. No
Gamma, book, fee, account, credential, order, fund, or protected request was
made.

Contract SHA-256 is
`ebd6309bc16e9bbb140f77f5c31854c5ab4ac2f553c2864da77b50282bee7774`;
result SHA-256 is
`6c54fcb7e8031a1c6cf43c969b445fa12e955368283922edf4d1b7782fb0c60b`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
86, and registry SHA-256 becomes
`616c1991be031641954464c4099b662afc75f835188847f087748a7544fa356a`.

## Polymarket fee-rounding fragmentation dominance rejection

The official retained maker-rebate source exposes a potentially adversarial
fee discontinuity: fees are rounded to five decimals and amounts below the
0.00001-pUSD quantum round to zero. A zero-network exhaustive audit bound that
mechanism against every retained current book rather than placing or sampling
orders. The 15 book files contain 180 token rows and 96 distinct conditions;
every condition has a five-share minimum order and a 0.001 or 0.01 tick.

At the lowest published nonzero 0.04 fee rate, five shares at the most favorable
observed extreme tick of 0.001 still produce a 0.0001998-pUSD fee, or 19.98
quanta. A zero-fee fragment would have to be smaller than
0.2502502502502502502502502503 shares. Such a fragment is below the observed
minimum order, and current sources do not bind user control over partial-fill
partitioning or fee aggregation. Even under the most favorable interpretation,
one zeroed fee assessment saves strictly less than 0.00001 pUSD, so more than
100,000 independently zeroed assessments are required for one pUSD of gross
savings before spread, adverse selection, latency, and operating costs.

Fee rounding therefore has a zero standalone profit floor. Never split, churn,
self-match, reroute, or manufacture volume for it. Reopen only after a material
fee-precision, minimum-order, tick, aggregation, or partial-fill semantics
change, or exact owned organic fills prove recurring positive after-cost
savings. No network, credential, account, order, fund, transaction, or protected
capture was accessed. Canonical result SHA-256 is
`24a471a56b10e67ce20350f7680e4cd67b54ca911564141dcfe229a1fe21edbd`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
87, and registry SHA-256 becomes
`2dcff4f6fcf1c3d40ac78eb3a09dd963303349203fc290381046111e762afa83`.

## Binance Spot OPO and OPOCO received-quantity execution candidate

Pinned official Binance Spot terms prove a structural buy-then-sell execution
mechanism that does not require market-direction prediction. OPO and OPOCO need
only the working BUY balance, lock the resulting received funds, account for
commission, adjust the contingent SELL quantity to lot filters, and unlock any
unused residual. Relative to the frozen manual sequential comparator, this
removes one post-fill client order submission and the need to pre-fund the
pending sell.

One frozen public unauthenticated Testnet `exchangeInfo` request returned 5,891
bytes and confirmed `TRADING`, `otoAllowed=true`, `opoAllowed=true`, and
`ocoAllowed=true` for BTCUSDT, ETHUSDT, and SOLUSDT. Their minimum quantities,
step sizes, five-USDT minimum notionals, 200-order limits, and 20-order-list
limits match the retained production snapshot. Do not repeat, reorder, narrow,
expand, or alias that exact Testnet request.

This is not accepted profit. The contingent sell activates only after the
working buy fully fills, so partial fills remain unprotected. The mechanism is
same-symbol BUY-then-SELL only and proves no exit fill, price improvement, fee
reduction, millisecond advantage, or monetary profit floor. Its standalone
profit claim is terminal while the rank-5 execution overlay remains a candidate.
Advance it only with separate explicit Spot testnet order authority and an
independently required minimum-size BTCUSDT, ETHUSDT, or SOLUSDT organic
buy-then-sell question against a precommitted identical-quantity manual
sequential comparator. Canonical result SHA-256 is
`4680faf2b4b4e36466cbc7ace4de2a2214430cbe4a5d537a3997e3f43bc47cc8`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
88, and registry SHA-256 becomes
`0e01f791af46253ebfc7cd82f86f145fef738e9d4a4bbf46454f2c1d66fbb48e`.

## Binance Spot SBE depth-freshness candidate

The current pinned official SBE market-data contract documents a real nominal
depth cadence advantage: SBE Diff Depth publishes every 20 ms, while the
retained JSON Diff Depth contract's fastest stream is 100 ms. SBE partial top-20
depth is 50 ms, all SBE timestamps are in microseconds, and one connection can
carry 1,024 streams. Access requires an Ed25519 API key in the connection header
but no extra API permission, timestamp, or signature.

The one-use documentation capture retained 4,981 bytes but failed its literal
source gate because the contract required `An API Key is necessary for access.`
while the Markdown bytes contain `**An API Key is necessary for access**.`.
Preserve that consumed failure; do not refetch, alias, or loosen it. A prior
local invocation also failed before any request because its rounded freeze time
was still in the future; that unconsumed preflight error is retained separately,
and the corrected contract used the exact observed UTC clock.

This remains unaccepted. A 20 ms publication interval is not an 80 ms measured
arrival lead, and both SBE best bid/ask and JSON book ticker are documented
real-time. No Ed25519 credential, decoder proof, source continuity, simultaneous
same-host comparator, owned or paper fill, or after-cost monetary value exists.
The documentation-only profit claim is terminal. Advance the rank-5 candidate
only with a designated ephemeral Ed25519 key, explicit read-only market-data
authority, and one precommitted same-symbol SBE-versus-JSON capture with no
orders. Canonical adjudication SHA-256 is
`6b01c825831657d8dac8a33efb196bbd63d64698288e15cf9b1ff27be9b4aa77`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
89, and registry SHA-256 becomes
`46f24a32204c23ba68189cfa2cae8d0b55e17b5665e3c616708cba26867827c5`.

## Binance Spot FIX unordered execution-risk candidate

The frozen one-use current official FIX source retained 168,282 bytes and passed
all six neutral identity/authentication/lifecycle phrases. Its receipt hash is
preserved, but repository policy prohibited committing an unrelated illustrative
private-key block in the full document. A byte-exact 15,222-byte excerpt from
four preidentified relevant line ranges is hash-bound by a secret-free extraction
manifest; the full payload was removed and must not be refetched. The excerpt
establishes a real
execution architecture: `UNORDERED` message handling should offer better
performance with multiple client messages in flight; FIX ExecutionReport push
should offer better performance; order entry permits 10,000 messages per 10
seconds per connection and up to 10 concurrent connections; and one symbol mass
cancel covers all account orders, including orders from other connections.

The adversarial shortcut failed: the source never documents automatic
cancel-on-disconnect. It instead requires the client to handle maintenance News,
heartbeat failures, Logout, reconnection, and unknown execution status after a
10-second backend timeout. Unfilled-order counts remain account-wide, Drop Copy
is delayed one second, order entry requires an Ed25519 key with `FIX_API`, and
the retained contract exposes only production hostnames. Better performance is
qualitative, not a measured latency, fill, stale-loss, fee, or profit floor.

Do not repeat or alias the source. Reopen only after confirmed Spot testnet or
Demo FIX order-entry support or a material official semantics change; execution
work additionally needs a designated ephemeral Ed25519 `FIX_API` key, explicit
separate session and order authority, an independently required BTCUSDT,
ETHUSDT, or SOLUSDT multi-flight flow, and a precommitted same-host identical
non-FIX comparator. The HMAC-style testnet credential supplied in chat is not
compatible and was not used. Canonical result SHA-256 is
`177d9d119c8a57c86df03b136bdfc2c880b0220ac50c51bce20e84cbe21f1755`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
90, and registry SHA-256 becomes
`0f43e2b26b9c339dde0db00401122bd1339ad77713df076444e3bcd5b3f9246b`.

## Accepted-edge profitability and durability audit

A zero-network audit corrected the main R&D routing metric. The registry's 29
accepted scopes are not 29 equally profitable strategies: they comprise one
source-demonstrated recurring direct-cash edge, eight same-principal yield or
capital overlays, twelve savings or rebate overlays that require independently
existing internal activity, and eight external-user or client revenue overlays.
Only eight scopes can add cash without an independently required trade, borrow,
loan, conversion, or external user; all eight still require already-held idle
principal or another pre-existing balance condition.

Polymarket complete-set holding yield remains the strongest direct-cash result:
42 of 42 positive daily payouts across BTC, ETH, and SOL on 1,039 pUSD produced
1.2681 pUSD after direct split/merge principal loss and user gas, a
principal-weighted 3.1820191118 percent realized annualized rate. It is not a
stable deployment claim: the current rate remains fail-closed unqualified,
public-wallet evidence is not owned-account qualification, external friction is
not fully bound, and the public new-capital after-all-cost floor is zero.

Binance LDUSDT is the strongest historical persistence overlay, with 505 aligned
daily closes over 504 days and a 2.0109050595 percent compound annualized index
increment. It remains non-standalone because it applies only to collateral
already required by a separately profitable USD-M Futures strategy. After the
retained ten-percent annual opportunity-cost sensitivity on only the extra
principal required by the 99.9-percent collateral ratio, the historical
increment is 2.0008950495 percent; exact account cash flows and external costs
remain unproved.

The strict current stable, account-qualified, after-all-cost edge count is zero.
Future work must seek Tier-A recurring direct cash or materially improve Tier-A
account, cost, capacity, and persistence evidence. Do not spend captures or tests
merely accumulating organic-flow discounts, acceptance predicates, or referral
programs. The Polymarket rank-one not-before remains
`2026-08-31T02:15:30Z`; unchanged Binance triangle, stablecoin-cycle, and official
documentation inventory screens remain consumed. Canonical audit SHA-256 is
`c5c4f9c9751f7b434be1c66cece2bee41ceb06aac3b0c0e3a70a7f76342f0618`.

## Binance EURI issuer-redemption parity terminal

A distinct primary-source lead established a real direction-independent payoff
identity: Banking Circle states that EURI holders have a right to redeem at any
time at par, one EUR per EURI, and identifies Binance availability. One frozen
public unauthenticated source request retained that exact issuer statement. It
does not prove this repository's access, onboarding, eligibility, limits, fees,
transfer support, timing, capacity, continuity, or completed redemptions.

No new market request was needed. The hash-bound August 29 Binance population
already contains 60 EUREURI top-of-book observations; all 60 show the same
1.00010000 bid and 1.00030000 ask with zero quote changes, despite 3,675
reported 24-hour trades. Selling 1,000 EUR at the retained bid yields only
0.1000 EUR, or one basis point, before redemption costs. The route loses 0.2000
EUR after the frozen three-basis-point operational stress and loses 0.9001 EUR
under a VIP0 0.1-percent fee before that stress. Transfer, redemption, banking,
delay, failure, compliance, custody, tax, infrastructure, and alternative-yield
costs remain adverse and unproved.

The exact retained population is terminal, not an edge. Do not poll EUREURI or
repeat the issuer page. Reopen only on a material issuer or Binance access,
fee, timing, or continuity change, or an independently observed event-driven
discount above 25 basis points; then freeze one finite-size book and bind every
account and redemption cost. No credential, account, order, fund, transaction,
or protected capture was used. Canonical adjudication SHA-256 is
`1b344ebc26d348d00d32663e7d365e1812dd09d2010656ab53338d4c01fd2bd9`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
91, registry SHA-256 is
`0f43e2b26b9c339dde0db00401122bd1339ad77713df076444e3bcd5b3f9246b`,
and the source-bound durability-audit SHA-256 is
`c5c4f9c9751f7b434be1c66cece2bee41ceb06aac3b0c0e3a70a7f76342f0618`.

## Binance TUSD issuer-redemption event-dislocation candidate

A stronger distinct issuer-parity lead was selected from retained data before
any new market request. All 60 August 29 TUSDUSDT top-of-book observations had
the same 0.99810000 bid and 0.99830000 ask; top ask quantity ranged from 2,246
to 2,258 TUSD. The retained 24-hour activity row reported 2,399 trades and
48,598 TUSD of volume. This is a persistent historical level in the frozen
population, not evidence of quote-change or event-time recurrence.

One frozen public unauthenticated official TrueUSD request retained current
terms stating no issuer mint or redemption fee, a 1,000-TUSD minimum redemption,
KYC and AML requirements, a typical one-business-day bank wire after receipt,
and Binance availability. Those terms do not establish repository-account
approval, exact limits, continuity, or completed redemption.

At the retained ask, 1,000 USDT purchases 1,001.7028949214 TUSD before fees.
Under the explicitly unproved one-USDT-equals-one-bank-USD sensitivity, this is
17.0289492137 basis points gross. A VIP0 taker fee leaves 1,000.7011920264 TUSD,
or 7.0119202644 basis points, and the frozen three-basis-point operational stress
leaves only 4.0119202644 basis points. At the conservative retained top ask size,
the remaining buffer after VIP0 and stress is only 0.89954546 USD.

This is not accepted, stable, after-all-cost profit, or deployment-ready. The
input is USDT while redemption pays bank USD; the same-unit executable bridge,
exact Binance account fee, withdrawal network and fee, receiving-bank and
correspondent wire costs, delay, failure, compliance, custody, tax,
infrastructure, and displaced-yield costs are unproved and cannot be credited as
zero. Do not repeat the issuer page, repeat the exact population, or poll the
pair. Reopen only on a material issuer or Binance term change, complete exact
account-and-same-unit cost evidence, or an independently observed event-driven
TUSD ask discount above 25 basis points; then freeze one finite-size book before
any separately authorized action.

No credential, account, order, withdrawal, redemption, fund, transaction, or
protected capture was used. Canonical candidate SHA-256 is
`6403c2d269e8cc682b6f6d63a612e5e7de00fed50902b2797c4c1e999ed37eb5`.
Accepted edges remain 29, ranked hypotheses become 45, terminal families become
92, registry SHA-256 is
`b96faacc247e021a4d1775a0e412da2c59fc3adc5a081e68fc042d54d2474637`,
and the source-bound durability-audit SHA-256 is
`7d322edf9c9ca6c3d47909ad375c789fc9d1e2029468d0874cbd147069c904d8`.

## Binance retained stablecoin issuer-parity frontier

A zero-network frontier now prevents sequential issuer-by-issuer research on
weaker retained quotes. It exhaustively inspected all 20 trading pairs whose
base and quote assets are in the retained USD-stable, EURI, or EUR set across
the same 60 August 29 all-symbol book snapshots. Sixteen same-unit pairs were
ranked under an explicitly non-economic par sensitivity; four EUR/USD pairs
remain cross-unit and unranked. Nineteen pairs had identical bid and ask prices
in all 60 snapshots; only UUSDT changed once, so this is historical persistence
rather than event-time recurrence.

TUSDUSDT is the strongest source-admitted retained lead at 17.0289492137 basis
points gross. The next diagnostics are FDUSDUSDT at 13.0169219986 basis points,
FDUSDUSDC at 12.0144173008 basis points, and RLUSDU at 10 basis points, but their
issuer identities and complete costs were deliberately not source-bound. None
can dominate the already-admitted TUSD candidate from retained pricing alone.
TUSD still falls to 7.0119 basis points after the public VIP0 sensitivity and
4.0119 basis points after the frozen three-basis-point operational stress before
every external cost and its unproved USDT-to-bank-USD bridge.

Do not spend issuer requests one row at a time. Reopen the existing rank-45
issuer-dislocation family only if a candidate can materially dominate the
frontier through a non-polling independently observed event above 25 basis
points, complete exact account-and-cost evidence, or a material official term
change. The new artifact SHA-256 is
`d9e327b427adf37ea0ee6a0ac8bdfc8e91966a5802926cfe0d0113ed10856e1f`.
Accepted edges remain 29, ranked hypotheses remain 45, and terminal families
remain 92. Registry SHA-256 becomes
`4c9d7ec1eae56de0ba4301f5f8072a39907887fc0822580c298efbf18edd9dd2`;
the rebound durability-audit SHA-256 is
`398db39349738346ffeac68333cd092ddcc6b2f67d0cb8cf9e00ef495b315c31`.

## Polymarket soccer half-result/full-result superhedge graph

A distinct zero-network audit found and exhausted two previously untested
direction-independent soccer payoff families in the immutable August 29
near-expiry page. Ten match families contained a complete base match,
halftime-result, and second-half-result event triple. The audit proved 70
three-leg conjunction superhedges: when the two half results jointly imply the
full result, buying NO on both half outcomes plus YES on the implied full result
pays at least one pUSD. It also proved 20 reverse-union superhedges: a full-game
team win requires that team to win at least one half, so NO full-game win plus
YES first-half win plus YES second-half win pays at least one pUSD.

The one-pUSD floor survives cancellation because team-win markets resolve NO
and draw markets resolve YES. It also survives the half-market no-data fallback
because both half legs resolve 50-50. The v1 representation contract required a
literal 48-hour fallback and stopped before output on two retained Belgian
families that instead say 24 hours. That failure is preserved. V2 changed only
the literal gate to accept the exact retained 24-or-48-hour variants; both have
the same 50-50 economics and no source was refetched.

All 90 relations had complete side-specific rejection prices and zero were
strictly below the floor. The best was NO Parma full-game win plus YES Parma
halftime lead plus YES Parma second-half win at 1.12 pUSD for a one-pUSD floor,
an optimistic 0.12-pUSD loss before books, fees, synchronization, or external
cost. Therefore no current book, fee, account, credential, order, fund, or
protected request was justified. Do not repeat this retained population.

The v1 contract/failure SHA-256 values are
`c46aed6f2003ae1e6222ca774c0a0040b0f2fa3a14ba59462937f7aa2a7d08f4`
and `5aa4e2491a913a7bbe1509b4665d60bd7c32a5e0c53b88faf0fff38ebec1c12c`.
The v2 contract/result SHA-256 values are
`f778b8f774beea52ff1860d89c28412aae15e8692198612b075ee426460df5a3`
and `1db8d0bd47e4141bff551421df2d923bfe63545d807779e702a95ceeddb17c3f`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
93, registry SHA-256 becomes
`5725f87d7ec56260da65504449f5b4431b72f1d2e6d2f1637fb30f1736eeb85e`,
and the rebound durability-audit SHA-256 is
`dd2d50abedb7b10a79a0b0e941b46b74b41e386fcb704b71afbf97c5a3873d0f`.

## Polymarket soccer exact-score cross-family graph

A frozen zero-network audit closed the remaining exact-score logical-arbitrage
surface in the immutable August 29 soccer page. Seven match families had a
complete base event, exact-score event, first-to-score event, and more-markets
event. The audit exhaustively generated 1,428 distinct rule-proved packages:
14 exact 0-0/Neither equivalences, 7 Under 0.5-to-exact-0-0 implications,
42 one-sided exact-score-to-first-scorer implications, 105 exact-score-to-BTTS
implications, 630 exact-score-to-full-total implications, and 630
exact-score-to-team-total implications.

All 1,428 packages had complete side-specific `bestAsk` or conservative
`1 - bestBid` rejection evidence. Zero were strictly below the one-pUSD payoff
floor. The best, NO Neither-first-to-score plus YES exact 0-0 for
Oud-Heverlee Leuven versus Standard Liege, equaled exactly 1.000 pUSD before a
current book, fees, synchronization, or external costs. The games have ended,
so no historical or current book request was allowed regardless of the result.

Do not rebuild, reprice, or refetch this population. A future soccer event can
advance only when its exact common horizon, postponement, cancellation, and
resolution rules preserve the payoff floor and at least one side-specific
rejection-only package is strictly sub-floor; then freeze one exact live book
batch. No network, credential, account, order, fund, transaction, or protected
capture was accessed.

The contract SHA-256 is
`f9f88d1d76172e0ebbc8c00a9593b25f3b75673f89612b9c1a3d968e6e291482`;
the result SHA-256 is
`a0d76c05979cde7b30f1ffc912e554f6405beb82e534eaa9b5410f073ef7d42f`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
94, registry SHA-256 becomes
`3ee674e0dbc81c965ca1afba65c6b3a032f1cdeb271b856b3b7b268510d6af1f`,
and the rebound durability-audit SHA-256 is
`c2a6a0d5d0dcc17fb0827ee41631c9e654d9c07b6bd733d46087d0a576e20637`.

## Polymarket soccer goalscorer-to-total superhedges

A distinct frozen zero-network audit exhausted every retained same-match
anytime-goalscorer implication into full-game Over 0.5. If a named player is
credited with a goal, the match total must be positive; on full cancellation,
both markets resolve 50-50. Therefore NO anytime goalscorer plus YES Over 0.5
has a one-pUSD rule floor independent of match direction.

Five complete match families produced 43 packages. All 43 had complete
side-specific `bestAsk` or conservative `1 - bestBid` rejection prices, zero
were strictly below the floor, and the best—NO Tosin Aiyegun anytime scorer plus
YES FC Lorient/ES Troyes Over 0.5—cost 1.90 pUSD. Do not rebuild or reprice this
population. Reopen only on a future distinct active match whose exact rules
preserve the floor and whose rejection-only sum is strictly below one pUSD.

The contract SHA-256 is
`a1c7e244f196d7531bf40950dcb368b5cb71ae0578e71495282ae9b6e933d6ae`;
the result SHA-256 is
`6b5fa74bc6f9e4022472969155960ab6fa676d472f570b1e9a56ae0286ba5a03`.

## Polymarket soccer corner-count structural graph

A second distinct frozen zero-network audit exhausted full-game, first-half,
second-half, and team corner-count ladders; the exact first-half plus
second-half and home plus away additive partitions; and parity implied by
adjacent full-total thresholds. Every tested market counts corners taken rather
than awarded and uses the same official-statistics fallback chain. Two-leg
packages retain a one-pUSD cancellation/no-data floor and three-leg packages
retain 1.5 pUSD under those 50-50 states, while their ordinary floor is one.

The V1 literal gate stopped on a Belgian 24-hour no-data fallback instead of
the expected 48 hours, even though both resolve 50-50. That failure is
preserved. V2 accepted only the exact 24-or-48-hour variants but stopped on an
incorrect strict zip of an adjacent ladder and its one-element-shorter tail;
that failure is also preserved. V3 changed only that mechanical pairing.

Seven complete 23-market corner events produced 1,820 distinct packages: 273
within-ladder monotone packages, 560 half-partition packages, 945 team-partition
packages, and 42 adjacent-interval parity packages. All 1,820 had complete
side-specific rejection prices and zero were strictly sub-floor. The best,
FC Lorient/ES Troyes Over 12.5 plus Under 13.5 total corners, cost 1.10 pUSD for
a one-pUSD floor before books, fees, synchronization, or external costs.

Do not rebuild, reprice, refetch, or request books for this population. Reopen
only on a future distinct active corner event with the same exact counting,
partition, cancellation, no-data, and source rules plus at least one
rejection-only package strictly below the floor. No network, credential,
account, order, fund, transaction, or protected capture was accessed.

The V1 contract/failure SHA-256 values are
`179c50e5e73c9743bb9c325796d1f9cb47cc0ad25e5f5163cd009bbffa3b0dc4`
and `ac8bad645ffa37146e8e5456f399840c73ac5aa9749890840459a07b13a43fb8`.
The V2 contract/failure SHA-256 values are
`66400e1bdf3efea98b517a4edf53db3e306e63ffebe742c27d726e89b916edf6`
and `ad491cd62a0b0a8fbe3937b590335baa5444225db90f272ee4aa190ffd1b9a88`.
The V3 contract/result SHA-256 values are
`0f057fd21f536afd27f4e83b131a1036354a9231e79d863053aeeadde3ca37d6`
and `3fcbb162ac70c5cd607d390d7e240614d9dd5b47eb07450522397b7c5e1dabb4`.

Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
96, registry SHA-256 becomes
`038f86e6b9c9765e44d7ba6f64454cab78946deeec6c602b6899197122fdd661`,
and the rebound durability-audit SHA-256 is
`7254a0504730471143e5449df532bcbe0e36aa207474afb77b048909e55afd4b`.

## Polymarket holding-yield V11 continuity pulse and frozen V12 receipt

The rank-one post-gate trigger was satisfied. The frozen V11 runner made its
one permitted public unauthenticated BTC activity GET and retained an HTTP 200,
8,038-byte response. It found exactly one new YIELD row: 0.0133 pUSD at
2026-08-31 00:10:10 UTC, 86,190 seconds after the prior selected row, with
transaction
`0xfb4022e56f217004e89c9a3be838222da491b73abcc0633c1eab55457cc61d5a`.
The raw response SHA-256 is
`63d2cbc9b33baee0a9773ecce72d13a38f5654ed08e6863e697d17dda01732c0`.

The V11 contract and result SHA-256 values are
`267040fa9fcdca77ff20e124203de4b128e4f1e9e53a819cd6dd98416c3cc323`
and `0ed8af56f9488757c9351c01b711ece0fbe579ea408cdfc28873863a91555cc9`.
The protected V7 partial payloads were not read, repaired, rerun, or touched.
The runner stopped before any receipt request as contracted.

A separate transaction-specific V12 receipt contract is frozen at SHA-256
`f7682f246682bcbff842c6e66776b58990446848a69040fcd1248867e405bb10`.
It binds exactly one public `eth_getTransactionReceipt` call for the transaction
above and the exact distributor-token-wallet-amount transfer. It has not been
executed. The next continuation must run only this frozen request once, without
retry, alias, parameter change, or any other receipt or activity request. Do
not repeat V7, V9, V10, or V11. Only after V12 resolves should the already-due
rank-34 GLW terminal history reconciliation advance.

This observation extends single-wallet payout continuity but does not qualify
the current three-wallet rate, owned-account eligibility, external costs,
deployment readiness, future persistence, or a profit floor for new capital.
Stable current account-qualified after-all-cost edges remain zero. Accepted
scoped edges remain 29, ranked hypotheses remain 45, and terminal families
remain 96. Registry SHA-256 is
`1797b2c3bc3c7698c5f9ed799277e37cfc7ee7f8ab711608a6a06eb15701446a`,
and the rebound durability-audit SHA-256 is
`c7b116f8786575044287a74d6c550bb771708c671d8e7b3c52217ece7b1f3108`.

## Polymarket V12 exact payout receipt reconciliation

The frozen V12 transaction-specific contract was executed exactly once. Its
single public unauthenticated Polygon receipt request returned HTTP 200 and
retained 256,395 bytes at SHA-256
`a102c37099fd5a04bb28ab3705088579d4b053c2bd42a48caf69921448faaebd`.
Transaction
`0xfb4022e56f217004e89c9a3be838222da491b73abcc0633c1eab55457cc61d5a`
was successful in block 92953322 and contained the exact 0.0133 pUSD transfer
from the frozen distributor to the frozen BTC wallet. Result SHA-256 is
`5befa8d4ed1d93459632537a47a534bf1650d3d36b7ee1967ed1bce012b58309`.

This extends public single-wallet continuity to 19 positive rows. It does not
qualify the current three-wallet rate, owned-account eligibility, alternative
yield, every external cost, future persistence, deployment readiness, or a
new-capital profit floor. Never repeat V7, V9, V10, V11, or V12. Reopen only on
a material official rate, program, cross-asset payout, or economic-comparator
change; authenticated or funded work still requires separate authority.

## Binance GLW terminal special-funding timing reconciliation

After V12 resolved, the already-due rank-34 contract made its one permitted
public unauthenticated GLWUSDT funding-history GET. The HTTP 200 response was
1,243 bytes at SHA-256
`60f998c7d593cfdad2ccbd17f911f849ca0b92f0e72a16cad13b7004099908d5`
and contained ten complete rows below the frozen 100-row limit. Exactly one
negative Special row appeared at 2026-08-31 00:00:01.003 UTC. Its per-unit
debit was 0.279999893 USDT, matching the declared 0.28 USD gross dividend
within 0.000000107 USDT, but the row arrived 1.003 seconds after the bStock
snapshot.

The hypothesized pre-snapshot adjustment gap therefore did not exist in the
2026 GLW episode. No book, filter, premium-index, account, credential, order,
fund, or transaction request was made. The result is terminal, not an edge;
accepted-edge and deployment-ready flags are false and the public profit floor
is zero. Never repeat, retry, paginate, alias, extend, repair, or book-capture
this episode. Reopen only for a future independent weekend or holiday dividend
event under a new prospective contract with a positive source-bound
conservative net-distribution floor. Result SHA-256 is
`315b2ba2a4f30caba2a7be1181dff3a8d93bc0e28adb2ad9738623a90342bd4b`.

Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
97, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 is
`5e67d7e5da543ac85d364fcb23dd09b42dce28c538a447c716393427d5e76b49`,
and the rebound durability-audit SHA-256 is
`ead5a00c80c621738d71dab382ef8cae9fb2f6ab7ec967c2d8ed7e6664eada06`.

## Binance Direct Stocks current fee-source conflict

The rank-five material fee-term trigger was investigated without credentials,
account state, orders, mutations, or protected capture access. The August 28
official announcement says the unchanged Direct Stocks discount is extended
through `2026-09-30T23:59:00Z`: 0.05% instead of 0.10% for orders of 340 USD
and above, and 0.17 USD instead of 0.35 USD for orders under 340 USD. That
announcement was already discovered by the consumed August 29 CMS contract;
its failed raw retention remains terminal and its exact request was not retried
or aliased.

One materially distinct logged-out read of the current Direct Equities fee
schedule still displayed `Promotion Until 2026-08-31 00:00 (UTC)` for both
fee tiers at `2026-08-31T10:31:24.588Z`, after that time had passed. The
announcement and live schedule therefore conflict on the effective interval.
The public current discount floor is zero: the historical accepted overlay is
preserved only through its original end, the extension is not credited, and
the accepted-edge count does not change. Do not repeat or alias the consumed
CMS request and do not poll the unchanged fee page. Reopen only on an explicit
official correction, withdrawal, replacement, or materially updated live fee
schedule that resolves one effective end time; even then, amend duration only
and require an independently positive organic order, exact account eligibility,
preview, and owned realized fee reconciliation.

The rendered semantic snapshot SHA-256 is
`bf77ad3565b90c3cb04c00e4ce1ef41268f72579e3b40603496ac72c478da14b`;
the request journal SHA-256 is
`ab950ac70f2938256ba67ba401c8e3e871b7a0f73ed6739cfdab93c426d504af`;
and the canonical conflict adjudication SHA-256 is
`eb6dcd919b73a033e1de48f6e8a806ab66f19ef657de75fa99fb98d0e1a56341`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families remain
97, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 is
`5f54a935d93a3e3b0380b178f8812f05880056f4477a28f4f672eabeaa7590f7`,
and the rebound durability-audit SHA-256 is
`89b94461381341ef497f27c79f7ceaed1626d0309dcf81d14a7fe33f112a0e47`.

## Binance GOOGL holiday-adjusted dividend timing rejection

The apparent September GOOGL timing gap was a false lead and is now corrected.
Alphabet's SEC filing states a September 7 record date, but Nasdaq is closed
for Labor Day and the applicable ex-dividend rule moves the ex-date to September
4. Binance's GOOGLB snapshot is also September 4, so the apparent three-day
record-date gap is not a pre-ex-date entitlement window.

The one-use Binance announcement capture retained an HTTP 200 response of
84,176 bytes at SHA-256
`10195102bfcff57204be744b365e15a5da3cfcd4f296f7fab9d61b73facde9f2`.
Its frozen gate failed four of five only because it expected generic trading
wording; the retained response instead says exactly that GOOGLB/USDT trading
will not be affected. Preserve that failure and adjudicate the immutable bytes
offline; do not refetch or alias it. The separate SEC capture failed before a
response on DNS resolution and is also consumed without retry or alias.

Retained exact GOOGLUSDT history contains one prior Special row at
`2026-06-08T00:00:00.007Z`. Its 0.2201324931 USDT per-unit short debit slightly
exceeded the issuer's 0.22 USD gross dividend. A public-network attempt to enter
or close between snapshot and a similar future debit would therefore be an
unproved millisecond race, not a structural edge. Current Binance terms also
permit withholding taxes, fees, costs, and other deductions, leaving the
conservative net distribution floor at zero. No current book, funding, account,
credential, order, fund, or protected-capture request was made.

Never infer ex-date from record date alone or repeat either source capture.
Reopen rank 34 only for a future independent episode where the bStock snapshot
materially precedes the official exchange ex-dividend adjustment and current
primary terms bind a strictly positive conservative net-distribution floor
before any precommitted adjustment and book sequence.

Binance source contract SHA-256 is
`a0ad60e0ad7a8662c7337ae98c8a8f759c93ef559761789dabbc996269854e86`;
source result SHA-256 is
`dd7c7bb3172b567a27afa5ec1df811e5143d5f38f23980a811b2df7a3726f154`;
SEC capture contract SHA-256 is
`638afff3be67a9b8b551246e4b39a7f8984762e5cc9889fe36be0fd53ec0a09a`;
canonical adjudication SHA-256 is
`9afb3aea93660ceabc34630e0cc6f5d562e094c96d7a21f21438a6daac6b2ce6`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
98, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 is
`dfe854e92f8de1af0b514f54a8c7ceeb49cb93fd168cad99bd81338b40d184c2`,
and the rebound durability-audit SHA-256 is
`e90253b1682b720e9e1e7ad9ad4c388e1aed415f04c30a0cc1ac5fcce6fe0dbd`.

## Polymarket Patriots-Seahawks monotone total rejection

The current official NFL surface supplied the first source-selected event
outside the consumed September 13-21 NFL window: Patriots vs. Seahawks on
September 9. One exact public Gamma event GET retained 110 active accepting
markets. Offline rule proof exhaustively evaluated 291 full-game margin and
total monotone packages and found two rejection-only displayed sums below their
one-pUSD floor. The precommitted best was Over 24.5 plus Under 26.5 at 0.995
pUSD per share, an optimistic 0.005 pUSD gross lead.

One separately frozen two-token book batch rejected it decisively. Five shares
per leg cost 5.55 pUSD against a 5 pUSD floor even at zero fees, and 5.65 pUSD
after one adverse tick per leg. The books were 11,031 ms old with 2,055 ms
timestamp skew. Because current exact asks were already negative by 0.55 pUSD
at zero fee, no fee request was made. Do not refresh this event, repair its age
gate, or select the already observed 49ers-Rams sibling after outcome access.

Exact-event contract SHA-256 is
`7f15f1c84d443301cf845abb126caaa091c688934ca3bad6633c964d1b67d25a`;
metadata result SHA-256 is
`e62d7b8a93a2a11197058d631e2fea88a408cfa30bd2dda6a9df54998b524d36`;
offline prefilter contract SHA-256 is
`c79bfa78b125e657003ef85b0b5d47f4b7a367230da10be5dbe551c5f6cbbf4e`;
prefilter result SHA-256 is
`2658f410b44f2a689c528b82f94f71c6459f5f52729824cd5f58fb22e920cc38`;
book contract SHA-256 is
`cff83822356354609fe1debed3d260405a80e57bb95a9f9cce933ee41d8bef71`;
terminal book result SHA-256 is
`2658c04330fdeaa7f39b4a9dc842a079c7e81ef1feeeba75ce74384e328338ae`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
99, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 is
`d8ca22c91a98d42fcd1ed63b532eeb1eeacf83f15d97be7c0198ea5b13639e5e`,
and the rebound durability-audit SHA-256 is
`e84cb6c9da66aed936e5bbab58ea87f48e6c7f1cb29640a0b37233e6fb393b72`.

A distinct zero-request audit then exhausted the retained NFL team-total to
full-game-total implication graph. When team threshold `T >= G`, team Over `T`
implies game Over `G`, so team Under `T` plus game Over `G` has a common-rule
one-pUSD floor, including cancellation because both legs resolve 50-50. All
nine valid packages were above floor; the best, Seahawks Under 24.5 plus Game
Over 24.5, cost 1.465 pUSD before books or fees. First-half, second-half, and
quarter markets were excluded because their retained horizon and overtime
semantics differ. No request was made. Canonical result SHA-256 is
`51d0f5bda02fe124a7f02eb6f2365c358f8ec6a99d75b8f87bd8de8eec1d669c`.
Terminal families become 100; accepted edges remain 29 and stable current
account-qualified after-all-cost edges remain zero. Registry SHA-256 is
`da888279bbf565413faadd9036547e4452b10bff91ed7ff1735117eb946c6f15`,
and the rebound durability-audit SHA-256 is
`dab81b0b99a64203392a4b7f684bf1b7c18417c2961e990a9db12cb5bb16dc96`.

## Polymarket Patriots-Seahawks V2 proof and pricing correction

The prior V1 team-ladder/additive artifact is preserved but superseded. It
fixed an initial lower-bound-only error by requiring
`A+B-1 <= G <= A+B`, but that remained over-restrictive. For Patriots Under
`A` plus Seahawks Under `B` plus Game Over `G`, all legs pay zero exactly when
both teams reach their thresholds and their combined score remains below `G`.
The necessary and sufficient one-pUSD-floor condition is therefore only
`G <= A+B`; the Over leg already pays in every game-over state.

V1 also treated midpoint-like Gamma `outcomePrices` as displayed acquisition
prices. V2 uses only retained `bestAsk` for the first outcome and conservative
`1 - bestBid` for the second. The corrected graph contains 325 full-team
additive covers, including 292 omitted by V1, and 20 full-team ladders. Zero are
side-specific sub-floor candidates; their best costs are 1.78 and 1.34 pUSD for
a one-pUSD floor, rather than the V1 diagnostics of 1.370 and 1.055 pUSD.

The same deterministic zero-network runner also exhausted 8,508 additional
period structures: 113 half/quarter/team-period/spread ladders, 36 same-half
team covers, 128 half-to-full covers, 48 team-half-to-full covers, 97
quarter-to-half covers, and 8,086 four-quarter-to-full covers. Across all 8,853
relations, both the side-specific and diagnostic strict-sub-floor counts are
zero. The best remaining period structure costs 1.44 pUSD for a one-pUSD floor.

No network, credential, account, order, fund, transaction, or protected capture
was accessed. The V2 contract SHA-256 is
`38df25f415434cb1ae823959e6fd81e0d777f52432bb3f5578122e407ed584f6`;
the V2 result SHA-256 is
`9c80f1a188c059890c682f16f367cc472667bda595482d5d0db26ebda2d014bb`.
Do not rebuild, reprice, or request books for this retained graph.

Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
102, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 is
`c1ff9777a1d8b0782f708e228c0a921a95ebecc0f38c2a745625357df162a5ce`,
and the rebound durability-audit SHA-256 is
`d86ef74ca37f66f414e0916dbaaefc851a461726359e7e16007f8e71386d12a9`.

## Binance QCOM/PYPL shared dividend net-floor prefilter

The already-consumed current QCOMB/PYPLB/GOOGLB announcement was exhaustively
reused offline instead of treating its two unadjudicated sibling tickers as new
source requests. It binds a September 3 QCOMB snapshot and September 4 PYPLB
snapshot, while retained exact histories already bind matching QCOMUSDT and
PYPLUSDT perpetuals. The shared distribution term is decisive: Binance
reinvests only the net cash dividend after applicable withholding taxes, fees,
costs, and other deductions, with no retained ceiling or complete formula.

The conservative public net-distribution floor is therefore zero for both
exact episodes. No official ex-dividend date, current funding, book, account,
credential, order, fund, transaction, or protected capture was requested. A
favorable calendar outcome cannot repair an independently zero guaranteed cash
flow, so both episodes are terminal before issuer-calendar or market access.
Do not repeat or alias the shared announcement or research QCOM/PYPL dates,
funding, or books for these episodes.

This corrects the workflow: when one primary source covers multiple events,
enumerate every sibling and apply the cheapest independent decisive gate across
the entire set before investigating siblings serially. A different ticker does
not make shared terms new.

The canonical result SHA-256 is
`ddb99b6d56e18e82218c057ee180eacdd7d0b055daab6701c762120a7a0d64df`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
104, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`6cd4204e8b3a4a75af1047c7a6d9ff36b0fb32b30322622ae06eb5d8ea5ebc29`,
and the rebound durability-audit SHA-256 becomes
`c9a25482f9984581a775ab2a02a73d4be35e0566e60ae8217d2eb6edd6e4f9f4`.

## Polymarket complete NFL catalog side-specific correction

The complete retained September 13-21 NFL catalog is now exhaustively corrected
without network access. Its payoff proof remains valid, but its 674 apparent
sub-floor candidates were produced by midpoint-like `outcomePrices`, not
side-specific acquisition bounds. The deterministic correction repriced all
4,621 relations using first-outcome `bestAsk` and conservative second-outcome
`1 - bestBid`. All 4,621 relations were price-complete, zero cost strictly less
than their one-pUSD floor, and the best was a Denver-Kansas City package at
1.05 pUSD.

This makes the prior Commanders-Cowboys book request avoidable in hindsight;
its 2.55-pUSD zero-fee loss and skew evidence remain preserved, not rewritten.
Do not repeat, reprice, or book-capture any event in the consumed NFL window.
For every future sports catalog, enumerate payoff relations once and require a
strict side-specific Gamma rejection result before any exact book or fee access.

No network, credential, account, order, fund, transaction, fee, book, or
protected capture was accessed. The offline contract SHA-256 is
`d37c0924ab5c1d66ee95d4ad06956b0cbb24c031db18b5c733033e463d69611b`;
the canonical correction SHA-256 is
`61fb2010d57b3295dd0ca859345c54404372dbd8b0f7ac4bca42d3fb0e40ddfd`;
and the exhaustive corrected-relation digest is
`4f66205743538c5491fd52246edb40f5a033bc27865195f20ebb5e842339d6fc`.

Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
105, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`549dfc93d72d056b3e8b72f96dc93b571c80bcdcc73f0d6ad36af3b1c7b0c67e`,
and the rebound durability-audit SHA-256 becomes
`95baa8bd82e7d92af36f1f39464c2961fe595f5040f378d6f8aeb67f705d48ab`.

## Post-August-29 primary-literature delta

The exact *Taker vs. Maker Arbitrage* title and SSRN identifier `7269858`
were already source-bound in the August 27 maker-first candidate. The paper is
not new primary evidence and does not reopen rank 2. Its existing historical
diagnostic remains aggregate-negative, unstable across assets and hours, and
unable to bind creation-time hedge executability or queue access.

A separately frozen one-request ArXiv delta then tested only for primary papers
strictly after `2026-08-29T22:37:17.664866Z`. The HTTP 200 Atom response retained
50 newest-first entries and 124,492 bytes at SHA-256
`dc5014da0761c6a317bbd1c3c3b185f700d2574aed94871c44d11de4b35bdeb6`.
The newest returned paper was dated `2026-08-26T10:21:10Z`, so the exact
post-cutoff paper count is zero and the post-cutoff population is complete.
No paper download, venue market-data, book, fee, credential, account, order,
fund, transaction, or protected-capture request was justified.

The source gate failed only because it froze one literal serialized XML root
tag and the response used an equivalent namespace representation. That failure
is preserved. The retained bytes parsed as the exact namespace-expanded Atom
`feed` with 50 entries and were adjudicated offline without refetch. Future
paper work must search exact titles, DOI, arXiv ID, SSRN ID, author-title pairs,
and mechanism aliases before browsing; future XML gates must bind semantic
namespace-expanded elements rather than attribute layout.

Do not repeat, paginate, alter keywords, alias, or repair this query. Reopen
only for a later primary-publication delta or an independently discovered exact
paper absent from retained identifiers and mechanisms. Contract SHA-256 is
`b809d79d24678212dd90c8e61b75d502463f24aa66cfc52a9ae262aebe4dba0f`;
capture-result SHA-256 is
`986a79fec025a0d27862c59cdf1e5e865b39ec6754b31bcc9ae5ea090c10f8af`;
canonical adjudication SHA-256 is
`17a2d6b537a05f1b4a4e273db682d8b611cd23f6c0906bcd4120be332d855101`.

Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
106, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`5a6ce51d4fe290ec5121e56914896980108d1b4bcc9e162fb1b4769c79406e59`,
and the rebound durability-audit SHA-256 becomes
`e127ae7f13413599822c67970cdd8e6ada506d65eb1a61c0ecb0f8acf6da0a4a`.

## Polymarket binary reverse-complete-set retained audit

The standard binary mint-one-complete-set then sell-YES-and-NO hypothesis is
now terminal on the exhaustive retained repository population without another
market request. The offline audit covered all 16 retained Polymarket book
arrays: 182 books, 82 complementary-pair observations, and 80 unique condition
IDs at five shares per leg. Seventy-nine observations had complete displayed
bid depth and all 79 were gross-negative before fees; three lacked sufficient
depth. The closest gross result was `-0.005` pUSD and the median was `-0.05`
pUSD. Fees and every external cost were intentionally omitted only as an
optimistic rejection bound.

The retained official CTF Exchange V2 README, SHA-256
`41def0727a8adbaccefb3c25bce4e50166915f98ea3e9588323304c2851fac7c`,
defines complementary both-buy matching as a MINT and both-sell matching as a
MERGE. Persistent crossed complementary books are therefore structurally
self-clearing rather than a stable harvestable overround. The initial
zero-network exploratory pass exposed outcomes before the canonical runner and
artifact were written, so the result is explicitly outcome-aware,
rejection-only, and forever ineligible to promote an edge. This method error
did not create a favorable claim and is preserved rather than hidden.

Do not rerun the same retained files or poll binary books for this unchanged
identity. Reopen only on a material official matching or settlement
architecture change, or independently frozen source-continuous evidence of a
persistent finite-size combined bid above one after every fee and external
cost. No credential, account, order, fund, transaction, new book, or protected
capture was accessed. Canonical audit SHA-256 is
`83d1e6b35f79d3b2542b4de11f86ca859d34b279b94395469585085ac7cd9671`.

Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
107, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`3e3938a581501a8f937accfdab920187f22a1870bc69628c632397a95590387f`,
and the rebound durability-audit SHA-256 becomes
`e21ea8f29f6732333689964e4f41f673f2ae9ac32738c6c1d8cbbb7032c326ac`.

## Binance Lite Loan USD1 current-horizon closure

The former narrow Lite Loan plus USD1 fixed-bonus lead is no longer an
available candidate. Its retained official enrollment window ended at
`2026-08-27T23:59:59Z`. A zero-network hypothetical late-entry adjudication
also preserved every originally favorable input except elapsed reward time and
recomputed the route at `2026-08-31T14:31:15.606103Z`. Only
25.3949466886 bonus days remained.

At the frozen 100, 500, and 1,000 USDT loan sizes, elapsed time removed
6.8446628624, 6.8861456677, and 6.8930594685 bips of reward. The original
same-stress margins therefore became `-5.7367793816`, `-5.5897295597`, and
`-5.5652212560` bips, respectively. Under those unchanged inputs, break-even
fixed APR is now 7.8328724815%, 7.8066349163%, and 7.8022926913%, all above
the retained 7% offer before account, collateral, tax, custody, and operating
costs. This is rejection-only time decay, not a current book observation and
not permission to reprice the expired offer.

Do not account-prequalify, borrow, subscribe, reprice, or poll this expired
episode. Reopen only for a new official Lite Loan offer with a fresh enrollment
window and complete service-fee and stablecoin-reward terms, or an exact
account voucher or rate change that restores positive after-every-cost
headroom. No network request, credential, account, order, conversion,
subscription, borrow, repayment, fund, or protected capture was used for this
adjudication. Canonical result SHA-256 is
`b48f1a9fb2d2858e872205e2d00ded6c50bed93eb4ef639cb0133cec9dff5f3f`.

Accepted edges remain 29, ranked hypotheses remain 45, terminal families remain
107, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`e6f4afd492fb2ea9f9bb5f36cc844db5268f19918480bea868ff440043155e6d`,
and the rebound durability-audit SHA-256 becomes
`c5ec5512240cb91d6e10a8080afd59659eb91bca22bd242ff05c9cf0ac7fce62`.

## Polymarket Chicago tornado reward-source rejection

The current Polymarket Rewards discovery surface supplied a materially larger
liquidity-reward lead: Chicago tornado risk on August 31 displayed 50 pUSD per
day, a 20-share minimum, a 4-cent maximum spread, and zero competition. Those
page values were frozen as discovery only and excluded from exact economics.

The one-use exact Gamma request reconciled the active accepting market,
condition, binary tokens, five-share order minimum, zero maker fee, and a
20-share reward minimum, but reported a 4.5-cent reward spread rather than the
displayed 4 cents. The conditional exact `sponsored=true` reward request then
returned HTTP 200 with a terminal cursor and zero rows. The sequence stopped
before books. The maximum publicly proved forward reward is zero, and the
candidate is terminal without a profitability claim.

Do not repeat either request, change the sponsored filter, substitute another
observed tornado city after outcome access, or use the Rewards-page amount or
competition display as funded economics. Reopen the family only for a distinct
source-selected exact market whose Gamma condition and exact sponsored response
reconcile to one active dated positive allocation before any book access, or a
documented material aggregation change.

Contract SHA-256 is
`25ed7ffeb6ab89366721e95a0ca5d11574a08e2df4052659632ada309017f091`;
canonical adjudication SHA-256 is
`1e514db9565c9aacb6a9d0c9e11a2bfa371b892ee0c133dc43da02c64815b66a`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
108, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`4c79d044f1cde823b1848fa13f47ea7fb0ebd0dda98209672158ff29d4f10fa3`,
and the rebound durability-audit SHA-256 becomes
`d9838da2a03c04fb0826736ce01228b3bbc940c1dc1db90981c825390e7963fe`.

## Binance delta-neutral dual-grid paper source failure

An exact-title, DOI, SSRN-ID, author-title, registry, and artifact alias check
found no retained adjudication for *Delta-Neutral Grid Market Making with
Adaptive Hedging and Risk Management for Cryptocurrency Spot-Futures Markets*.
The lead was therefore distinct enough for one primary-paper request, but not
for venue data or testing.

The frozen exact SSRN `Delivery.cfm` PDF GET returned HTTP 403 with 5,757 bytes
of Cloudflare challenge HTML at SHA-256
`7cb68acb61069555e8672266bfc294785fc72df37410027a151c5c59b1d1542f`.
The request and raw failure are retained and may not be retried through a URL
variant or mirror.

The indexed abstract remains discovery only. It reports a 52-day November-
December 2025 Gate.io BTCUSDT tick-data backtest, near-zero rather than exact
delta, trend-skewed quoting, and active inventory management. Even if those
claims are accurate, they do not establish current Binance fees, fills, queue,
slippage, hedge costs, funding, rebates, basis, capital, liquidation, or bull,
bear, sideways, choppy, volatility, liquidity, and latency durability. Spread
capture, funding carry, and maker rebates already map to retained families; no
unique subsidy, atomic hedge, principal guarantee, or capital credit was
source-proved. No collector, venue request, credential, account, book, order,
fund, transaction, or protected capture was used.

Contract SHA-256 is
`be6bb998493e6ab3c816c30ef991574729f8ac9888c5217ab9beef98f7dc775a`;
canonical adjudication SHA-256 is
`f07468a381362d3551c2455344035ac6adc84c317d0e7851f9791d73351eaa17`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
109, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`16a69108fdd3b8f2b20dacf4d193427f988f4438a4fa285667062e2fda694fc3`,
and the rebound durability-audit SHA-256 becomes
`93bf922b12fd3fef040503660f1a04f70abe842a07be5c6d8ecc83cb164d9740`.

## Polymarket NYC precipitation reward date-gate rejection

The current first page of the official Rewards surface supplied a distinct
month-long lead: NYC September precipitation above six inches had the largest
displayed daily reward on that page. All displayed rate, spread, size, price,
volume, and competition values remained discovery only.

The frozen source contract incorrectly bound the rendered market page's
October 2 `End Date` as if it were Gamma `endDate`. The one exact Gamma request
returned an active accepting binary market ending September 30 at 23:59 UTC,
with a 20-share reward minimum, 4.5-cent reward spread, 0.01 tick, and zero
maker fee. The prospective mismatch gate terminated the sequence before the
exact sponsored reward request or any book request. The public forward reward
floor under this contract is zero and no edge is claimed.

Do not repeat or repair the consumed Gamma request, request rewards or books
for this candidate, or substitute another observed precipitation bracket. A
rendered page date can be a user-facing resolution date rather than Gamma's
market end. Future exact-reward contracts must bind the Gamma end from Gamma or
omit rendered-to-Gamma equality instead of inventing timestamp precision.

Contract SHA-256 is
`101f5cacd9dc7a4acf2282f373e25f8abdaee5207d85f36812b0e74e98ddc877`;
canonical adjudication SHA-256 is
`41d36380efda98913a149156cbb5aacc4425e0d7be8c4372c73be9dbba316d5c`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
110, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`bc6d360c1107bb50b5ec3dfbb845a05dbeef6b2187f9f111144abf964cf53e32`,
and the rebound durability-audit SHA-256 becomes
`cb6014f6bdca23b3146a32844770b22badfaee2b2614954b942feff53da1cb4a`.

## Polymarket Ontario leadership paired-maker reward rejection

The live official Rewards surface supplied a distinct long-horizon,
non-weather lead: the exact Navdeep Bains Ontario Liberal leadership binary.
Its displayed reward, spread, size, prices, and competition were selection
evidence only and were excluded from exact economics. An alias check found no
retained use of this market or event before any venue API request.

The one-use source prefilter reconciled the active accepting Gamma identity,
condition, ordered binary tokens, November 21 market end, five-share order
minimum, zero maker fee, and the exact sponsored reward row. Exact funding is
40 pUSD per day, 20 shares, and 5.5 cents, with about 82.34 market days left.
The reusable source prefilter was corrected prospectively so Gamma can supply
the authoritative end without a guessed rendered-page equality; its existing
exact-date gates remain available when a genuine authoritative date is frozen.

The separately frozen two-token book request returned HTTP 200 in 281 ms with
zero cross-token timestamp skew. The retained top levels were YES 0.39/0.44
and NO 0.56/0.61. Joining both best bids at 20 shares had a 1.00 pUSD
both-fill gross and 11.20 pUSD maximum one-leg settlement loss; improving both
by one tick remained non-marketable with 0.60 pUSD both-fill gross and 11.40
pUSD maximum orphan loss. These are diagnostics, not executable or realized
profit.

The official book-snapshot timestamp was 174,712 ms old against the frozen
10,000 ms ceiling, so the freshness conjunction failed even though both token
timestamps matched. The current official API schema labels this field the
order-book snapshot timestamp; an HTTP receipt proves observation time, not a
new internal book update. The consumed gate therefore remains fail-closed and
was not weakened after the favorable-looking rows appeared. Publicly proved
reward payout remains zero, no edge is accepted, and no refetch is permitted.

Do not repeat the exact source or book requests, reprice, substitute another
Ontario candidate after observing this outcome, or treat the optimistic full
reward pool as owned payout. Reopen rank 17 only for a genuinely distinct
source-selected market under a newly frozen exact contract or a documented
material program or timestamp-semantics change. No credential, account, order,
fund, transaction, or protected capture was used.

Source-contract SHA-256 is
`ad5931984f16a3f5f4f900bf7878266bd15bd205641076056529a0cb7ee8b4ab`;
source-prefilter SHA-256 is
`85cef790f285cd732d0f3ce1ae71077d20d685681c008f0ee843cf2cd49af3e5`;
book-contract SHA-256 is
`207c62797736a1aefbe1284fd2c191557a92aef65c35cbdbd173297f654ac8b3`;
canonical terminal book-screen SHA-256 is
`03b4cdc0f577028e51f1f6bee6c3e2f0426d501d36d381c6d3d14036b974b294`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
111, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`f86d8ddc878b21744ba40fa3d88f025d030eea3778016a04922cfadb3c907a27`,
and the rebound durability-audit SHA-256 becomes
`3fc2e1175dec783936f3a469a19f86d0942c91d11b4b3295bed278614df5c75b`.

## Polymarket GTA VI paired-maker reward rejection

The official Rewards table was sorted by descending daily reward to avoid
paging or serial low-value screens. Current five-, fifteen-minute, and four-hour
crypto rows were excluded under consumed rank-17 boundaries. The September Fed
rows were then excluded before access because exact filename-only alias checks
showed that event was already captured in the holding-yield population. Ontario
siblings were excluded after the prior Navdeep outcome. The first remaining
zero-alias event was GTA VI Extended Look under 20 million week-one views.

One early alias command mistakenly allowed `rg` to print a large canonical JSON
line. It made no network request and changed no decision, but wasted output.
Subsequent checks used filename-only matching as `AGENTS.md` already requires;
do not repeat line-printing searches on minified JSON or retained HTML.

The frozen exact-source sequence reconciled an active accepting binary market,
ordered tokens, September 3 23:59 UTC end, five-share order minimum, zero maker
fee, and one exact sponsored reward row. Exact funding was 536.99616 pUSD/day,
200 shares, and 4.5 cents with about 3.327 days left. The absolute 200-pUSD
one-leg settlement-loss ceiling was below the roughly 1,786-pUSD impossible
100%-of-pool bound, so one separately frozen book request was permitted.

The two-token books returned in 278 ms with zero timestamp skew. Top levels
were YES 0.68/0.69 and NO 0.31/0.32. Joining both best bids at 200 shares cost
0.99 per complete pair and had 2 pUSD both-fill gross against a 136-pUSD maximum
one-leg settlement loss. Improving both legs by one tick was immediately
marketable, summed to 1.01, and lost 2 pUSD on both fills before every external
cost. The snapshot timestamp was 17,830 ms old, exceeding the frozen 10,000-ms
ceiling by 7,830 ms, so executable freshness failed and no refetch is allowed.

An offline retained adjudication then applied the official liquidity-reward
formula. Covering one maximum orphan would require 7.6157644569% of the entire
optimistic remaining pool or 25.3260656464% of one daily pool. The formula
normalizes each maker against all makers at every sample and normalizes again
across the epoch. Public books do not identify future makers or queue access,
and current API documentation exposes `market_competitiveness` without a
conservative mapping to this hypothetical maker's final share. The owned public
reward-share and payout floors therefore remain zero.

Do not repeat the source or book requests, reprice, weaken freshness, or select
another GTA bracket after observing this outcome. Reopen only for a genuinely
distinct source-selected market or documented material program, fee, identity,
or timestamp-semantics change. No credential, account, order, fund, transaction,
or protected capture was used.

Source-contract SHA-256 is
`5bab95ee364650762d7ec87db04f4ba6b88c91adb44d5608946c59f4308b9755`;
source-result SHA-256 is
`d0c2fba4bd24c97b4c6745059b7b8beb964a7d7f3b56027ee98f6bf15f2a2c60`;
book-contract SHA-256 is
`e06bfe5b10ae6cc8386b67469d605a27d20e58c4c8a34fb6fd0b059648d91eb2`;
book-result SHA-256 is
`dc61d9375638de915b4f16546e9a9fc876815d5a493cb53b117011f7b10ada1d`;
canonical adjudication SHA-256 is
`f25854cc7ec978cea4c5357bc8438093cfb818b0435f34773de78096af8ef051`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
112, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`10a3ec81ad74ea57d0e7051dfe3a4296db4be6da18f3d8b3438145ae9d4e1693`,
and the rebound durability-audit SHA-256 becomes
`8b1dde039a56cda32769aa289955404dde578f48396dc6ff4a1ad1cd7bdf855d`.

## Polymarket CFB empty-window and retained multi-event parity closure

Rank 30 advanced through one nonoverlapping, prospectively frozen CFB catalog
window rather than another single-event book test. The September 8 through 12
window began after every consumed CFB event. Its retained densest-day bound was
89 events, so five days projected to 445 against the source-bound 500-row
ceiling. The one public unauthenticated keyset request returned HTTP 200 with
zero events, no cursor, and a complete empty population. Zero relations, books,
or fee requests followed. Do not poll, narrow, refine, repeat, or immediately
move to another unproved forward horizon; reopen only after an actual distinct
deployment is source-proved or a material rule, fee, or price change occurs.

The contract SHA-256 is
`dc73c6b56d2ae89a4eaab93666f7b5b6bf5856daed1663ef5b5fef4e76f8c178`;
the result SHA-256 is
`5511d2446585d02577c91249f454e458b1b7551e656cd4f68abd8dae0ac526be`;
the 97-byte raw response SHA-256 is
`1caf48c002786edb458302862e4e33fbff3b2afecb9ec796169d7a3881ad7e9e`.

Focused verification then exposed that the consumed CFB runner delegated its
price gate to midpoint-like `outcomePrices`, despite the new contract requiring
side-specific rejection prices. The September 8–12 decision is invariant: its
complete population contained zero events and zero price rows. The exact
11,535-byte consumed runner was preserved under an immutable sidecar at
SHA-256 `d26ef14f...`; the reusable runner was corrected prospectively to use
first-outcome `bestAsk` or conservative second-outcome `1 - bestBid` before
ranking. The empty-population method adjudication SHA-256 is
`29d8342af4060c5e75d9e4cfca88db38e59e67f32a332847b046d0b1f900d5a0`,
and the runner-lineage SHA-256 is
`e55df3c23a28a49cc346f3d382a04b0fa9410612ae519be6e59547a85c3b4627`.

A zero-network correction then repriced all 139 retained CFB relations across
the complete September 3–7 populations. The old midpoint-like method reported
eight strict sub-floor candidates; the correct side-specific method reports
zero. Corrected best sums are 1.07, 1.08, 1.07, and 1.62 pUSD respectively for
1 pUSD floors. All earlier payoff identities remain valid, but their
`outcomePrices` economic gates are superseded. The consolidated correction
SHA-256 is
`cc247e76896eea386ae252474099af3c6872393e77ea81b9d4262671f17ac9e0`.

Before spending another market request, a distinct zero-network rank-31 audit
reused the nine hash-bound August 26 market rows already retained by the
holding-yield study. It exhausted seven exact direction-independent relations
across four events: two mutually exclusive Fed brackets, two largest-company
outcomes, three best-AI outcomes including the all-NO triple, and the nested
Anthropic 1.5T/2.0T valuation thresholds. Rejection prices used YES `bestAsk`
or conservative `1 - YES bestBid`; Gamma remained rejection-only.

Zero of seven sums were strictly below their optimistic common-rule payout
floor. The best was NO no-change plus NO 25-bps-increase at 1.01 pUSD for a
1 pUSD floor. The all-NO best-AI triple cost 2.14 pUSD for a 2 pUSD floor, and
the nested Anthropic package cost 1.26 pUSD for a 1 pUSD floor. No current book,
fee, credential, account, order, fund, transaction, or protected capture was
justified. Do not rebuild, reprice, or book-capture these retained rows.

The canonical retained-parity SHA-256 is
`b3f5480fea15bb73990a278dba93356cb34c49f91d82e2624c79fef8191e09f4`.
Accepted edges remain 29, ranked hypotheses remain 45, terminal families become
114, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`f6f2e93e2706b1db421c53fc07e5066cb2084367e8deee98b44b3fd62606180c`,
and the rebound durability-audit SHA-256 becomes
`a45b7f745ea564fc859a7c034f8739c96716df8e43bda8f74b6f814b8459d7eb`.

Focused verification exposed one older source-binding defect: 32 consumed
contracts or results expected SHA-256 `ba7ebaa1...` at the reusable
`screen_polymarket_exact_two_leg_package.py` path, which was later legitimately
changed. The consumed artifacts and current tool were not rewritten. The exact
17,547 historical bytes were recovered from commit `7e9bffa7...` into an
immutable binary sidecar and tests now route only that historical hash to the
sidecar. The lineage artifact SHA-256 is
`a584e3665300bb405183370a66f47e2fc83ddc43764251ec236c2341f7a4d1ef`.

## Binance stock-option plus opposite-perpetual deployment rejection

A retained official API catalog comparison exposed one genuinely distinct
structural lead not previously registered. For a European option and exactly
matching underlying unit, a long call plus an equal short perpetual has the
terminal gross lower bound `perpetual entry - strike - call ask`; a long put
plus an equal long perpetual has `strike - perpetual entry - put ask`. This is
market-direction independent, but it matters only if stock options are actually
deployed and their settlement identity matches a tradable perpetual.

Current official Binance developer documentation classifies
`GET /eapi/v1/exchangeInfo` as public market data and exposes `contractType`,
`underlyingType`, unit, strike, expiry, and status. TradFi Options contract
acceptance is a separate `USER_DATA` `POST` and was not called. The prospective
contract froze the complete active `TRADFI_OPTIONS` plus `EQUITY` filter and one
request maximum before access.

The exact public inventory GET returned HTTP 200 with 1,290,567 retained bytes
and zero active stock option rows. The empty population SHA-256 is the standard
empty-byte hash `e3b0c442...`. The sequence stopped immediately: zero option
tickers, futures metadata, books, premium-index rows, funding rows, credentials,
account reads, contract acceptances, orders, or funds were used. This rejects
current deployment, not the payoff identity. Do not poll or repeat the current
population. Reopen only after an official Binance TradFi equity-option listing
or a material stock-option settlement, fee, access, unit, or matching-perpetual
architecture change, then freeze a separate economics contract.

Contract file SHA-256 is
`08bcd2fd86082e9c4d03b4408c38aabfb5f84248ace9748d6898ebcef3114624`;
canonical result SHA-256 is
`b72592efad26563aacc4e6d8611f15f3172039ffc220153ab824bf57231afcb3`;
raw response SHA-256 is
`88db31aba07967eb9fcd6dd3c93b409e97b835365df9fd001590f53c6f3d3e23`.
Accepted edges remain 29, ranked hypotheses become 46, terminal families become
115, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`06ca44a66364f8bcd1d78b76d4e75f13f1431ce9437a5e7f7ce8c1f6af5ffd04`,
and durability-audit SHA-256 becomes
`a0896ba8a9782234b2735e67553583118694715e8f65886416d6f73481d4ea1a`.

## Binance BTC ETH SOL single-option terminal-floor retained screen

The stock-option deployment review exposed a stronger zero-request extension:
the retained August 27 option/futures capture already contained 1,410 active
BTC, ETH, and SOL crypto options plus synchronized executable-side perpetual
books and 500 funding rows per asset. Earlier work screened two-option
conversion/reversal parity, but not the simpler long-option terminal floor.

For a unit European call bought at ask plus an equal short perpetual entered at
bid, the terminal gross lower bound is `F_bid - K - call_ask` if the perpetual
is closed at the option settlement underlying. For a put plus equal long
perpetual it is `K - F_ask - put_ask`. Neither construction writes an option or
requires a price forecast, but basis, funding, fees, ticks, capital, depth, and
non-atomic risk remain real costs.

The implementation was formatted, import-preflighted, and then exercised in a
separate zero-economics preflight against every hash-bound source before the
one-use contract was frozen. The production screen made zero network requests
and exhaustively evaluated all 1,410 options. Exactly 1,115 had positive entry
sides and exactly two had a positive displayed gross floor:

- `SOL-260828-90-C`: 0.17 USDT per unit, 16.71090141 bips gross.
- `SOL-260828-92-C`: 0.10 USDT per unit, 9.82994200 bips gross.

The best row already lost against the frozen 33.5-bip option-entry,
option-settlement, futures-round-trip, and expiry-basis stress before funding,
capital, or two adverse ticks. Complete stress reduced the two rows to
-41.09843077 and -47.97939017 bips. Futures top-level capacity passed at the
0.05-unit common minimum, but the retained option ticker has no ask quantity;
because zero rows survived the prior cost gate, no exact option depth or current
market request was justified.

Do not rebuild, reprice, or depth-capture this retained population. Reopen only
for a distinct active option population, material fee/settlement/tick/depth/
funding/basis/capital change, or an independently observed non-polling displayed
terminal floor strictly above every applicable cost. No credentials, account
state, orders, quotes, transfers, funds, paper trades, or live trades were used.

Contract SHA-256 is
`b31c691c728d7d2c7a5d7e13151c57139c6aad1c5fc22fb2be913edb6e5b9a60`;
result SHA-256 is
`90c05ed35db00da7e5b4a2d8ec6ac0a51367a1a768dc58a39ef479510d5aa745`;
implementation SHA-256 is
`04780a528d5708b9f6a7cdec3d38f29292254a2a7d7bea647287f08ef2f8363a`.
Accepted edges remain 29, ranked hypotheses become 47, terminal families become
116, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`7eeba81c9b93404372bd4833002c67c5a958d25e5b7de258f86c95968f70b247`,
and durability-audit SHA-256 becomes
`9fe9eff62c4d11e94d7791bbba3edfbb89d8bc26b0f38da204b1ef38cf6a0485`.

## Binance yield-asset isolated-margin removal rejection

One exact current primary Binance announcement exposed a novel but terminal
forced-settlement lead. Isolated `WBETH/ETH`, `BNSOL/SOL`, and `BFUSD/USDT`
borrowing is suspended at `2026-09-01T06:00:00Z`; automatic settlement, order
cancellation, and pair removal are scheduled for
`2026-09-03T06:00:00Z`. The source contains no aggregate long/short position
direction or size, settlement-price formula, LST conversion/redemption ratio,
fee, capacity, delay, BFUSD face-value redemption contract, observed discount,
reward, or recurrent cash flow.

The announcement therefore proves event timing but not a direction-independent
payoff. Any trade based only on presumed forced buying or selling would be
directional speculation. The sequence stopped after the single source request:
zero books, conversion quotes, account reads, credentials, borrows, orders,
funds, or transactions were used. Do not poll or pre-position around this exact
removal. Reopen only if an independent non-polling observation shows a
finite-size discount strictly above a separately source-bound exact same-account
conversion or redemption floor after every fee, delay, basis, hedge, unwind,
capital, and operational cost.

Contract SHA-256 is
`8ae3039fd20c9daa8108a964ba0beff8419204f9b119cd0e582a6d7f6ccd9767`;
canonical terminal SHA-256 is
`b0367360207044b8b70cc76a72de288f698c563c0e52133836e1b6402b15f392`;
raw response SHA-256 is
`5f1cec111cb8aa171d1d23a1a6222b8daec3ce6be45cb2c8b99c5effe35bb4e8`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
117, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`87d53a5b052600a788db81d19cfdcac75b8f1c9fe616b42d116ce789f48b9a1a`,
and durability-audit SHA-256 becomes
`067a2014b02bafde283841940cbb3f1299b8e1fe9108f06cf238816bf2e0d014`.

## Polymarket Prešov paired-maker reward rejection

The current official Rewards page supplied a distinct first-page zero-alias
lead: František Oľha for the 2026 Prešov mayoral election. Page values were
selection evidence only. The frozen exact-source sequence reconciled the active
accepting Gamma identity, ordered binary tokens, October 25 market end,
five-share order minimum, zero maker fee, and the exact sponsored reward row.
Exact funding is 55 pUSD/day, 20 shares, and 5.5 cents with about 54.22 days
left. The impossible 100%-of-pool bound therefore exceeded the absolute
20-pUSD minimum-size orphan ceiling and permitted one separately frozen book
request.

The two-token response returned HTTP 200 in 250 ms but had zero YES bids and
zero NO asks. The only displayed executable sides were a 0.87 YES ask and a
0.13 NO bid, so no complete paired best-bid state or both-fill gross exists.
Both official book timestamps were also 221,182 ms old against the frozen
10,000-ms ceiling. The runner retained the exact response and failed closed on
the empty side before producing a normal book result; the zero-network
adjudication preserves that failure rather than weakening the validator or
refetching. The reusable runner is corrected prospectively to emit a terminal
`rejected_incomplete_paired_book` result with null paired economics when an
exact side is empty, so future valid empty-book outcomes no longer require an
exception-path adjudication. The consumed runner bytes remain bound to commit
`5ab5aa98e0f2be7d5400c147b9a39fd97565564a` and were not rerun.

Publicly proved reward payout remains zero: the relative formula does not map
`market_competitiveness` to a conservative owned share, and an empty-sided stale
book supplies no fresh executable paired quote. Do not repeat the source or
book requests, substitute a Prešov sibling after observing the result, or use
Rewards-page values as economics. No credential, account, order, fund,
transaction, or protected capture was used.

Source-contract SHA-256 is
`e1b09fb8ca8287b93a15b51d00859703767dd313b76023dca376547084e2a31d`;
source-result SHA-256 is
`1ca9fb12c928d6ca5f507f4aa118261c78df096f96edd81f8248fc568f9ed319`;
book-contract SHA-256 is
`2eea2c6e44bdc6ce407cc1ac140147dc9b3f033d6f9e2da66428a7f8f3b6136a`;
canonical adjudication SHA-256 is
`efa25a2905029b44e96684660a291d12d8d049f49204136da1840099e550669d`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
118, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`0214595c90112a546ef535c26ff6a750dbfe6d975eae3fe8240008502fc694fe`,
and durability-audit SHA-256 becomes
`6c40ae46fc8247db33b1d6b14ab96e3c5a11b2eda8be1d327f63189b8fab75d2`.

## Polymarket public reward-share semantics gate

The strongest unresolved rank-17 shortcut is now source-bound and rejected.
The exact public Prešov reward row exposes `market_competitiveness: 0`, but the
retained official liquidity-reward formula allocates the pool by each maker's
relative normalized score and supplies no account-independent owned-share
floor. A zero metric therefore cannot be interpreted as zero competitors,
100% owned share, or any positive payout.

One commit-pinned public source retention captured the current official
Polymarket Python v2 client at
`215fc63a8fd6ec3a10c7edb73997c9772d8686d3`. Both exposed owned-reward routes,
`get_reward_percentages` and `get_user_earnings_and_markets_config`, call
`assert_level_2_auth`; the latter is also date-specific. This proves that the
SDK's owned percentage and user-market context is authenticated and
account-specific. It does not create authority to access an account, does not
make the observations a forward guarantee, and does not define the public
`market_competitiveness` field.

The adjudication therefore retains a public owned-share and payout floor of
zero. Do not multiply a public reward pool by an inferred positive share, and
do not request another book merely because `market_competitiveness` equals
zero. Reopen only if a current primary source explicitly defines a bounded
account-independent owned-share lower bound, or if separately authorized owned
eligible orders later provide exact authenticated scoring percentage, realized
earnings, and every cost. No credential, account request, market-data request,
order, fund, transaction, or protected capture was used by the adjudication;
the only new request retained immutable public SDK source.

Source-contract SHA-256 is
`e66201f597c4016fd211115dd180aacdc2869b20f4d1ab3c09f67ba3cc07d884`;
source-result SHA-256 is
`c57cbe963a3cf3caa04beef0df72f62fbcd6d300f8909b057e4fa62cb85207fd`;
canonical adjudication SHA-256 is
`7b58e5317a212c1e50f91bb7742f64f18915e7a3d477cdfc884c2c9938e3ee5b`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families
remain 118, and stable current account-qualified after-all-cost edges remain
zero. Registry SHA-256 becomes
`e9dcc345636a26a3fc43c3152087d7ca33f53744674d291b500f16f0fff00776`,
and durability-audit SHA-256 becomes
`87d5e74c6b7feefc95c3a047980527679763668e34e234aa0fee4abd98526256`.

## Polymarket 2026 Senate-House projection-cover rejection

A distinct joint-to-marginal projection cover passed the rendered and exact
metadata fee gates. Buy `NO` on the standalone Republican Senate seat outcomes
50, 51, and 52, then buy `YES` on every joint-event row whose Senate projection
is 50-52: House seats at least 223, 208-222, 193-207, and at most 192. When the
Senate result is outside 50-52, all three NO legs pay. When it is 50, 51, or
52, two NO legs plus exactly one exhaustive House row pay. The seven-leg
package therefore has a three-pUSD terminal floor independent of market
direction.

Two frozen one-use public Gamma GETs retained 77,675 and 65,510 bytes at
SHA-256 `0c2e183ff2d8681095b12635a4b88dc714eede64a36f778da1d35b994f790a3f`
and `cff3146fd7f73bd341a2e716c27d130569ddde3deacc0a3c76a08978d5ce5158`.
The exact joint event contains 13 active open accepting-order markets and the
standalone Senate event contains 11. All four required 50-52 House rows are
present and exhaustive. The Senate election universe, October 31 cutoff,
runoff, vacancy, party attribution, independent-caucus, January 4 fallback,
and resolution-source rules align. The differing nominal Gamma end fields are
retained as additional timing friction and were not used to improve economics.

Conservative side-specific metadata prices were 0.88, 0.90, and 0.927 pUSD
for the three NO legs using `1 - bestBid`, and 0.035, 0.107, 0.084, and 0.028
pUSD for the four direct YES asks. The 2.961-pUSD per-share package left
0.195 pUSD gross at five shares. All seven current 0.04 exponent-1 taker fees
totaled 0.09938 pUSD, leaving 0.09562 pUSD before depth.

The cheapest remaining decisive gate then rejected it without wasting a book
request. One exact adverse tick per leg raised the per-share cost to 2.986
pUSD; recomputed taker fees were 0.09675 pUSD at five shares, making the
after-fee floor -0.02675 pUSD before depth, time value, seven-leg non-atomicity,
partial-fill unwind, latency, capital, or operational cost. No fourteen-token
book batch, fee endpoint, credential, account, order, fund, transaction, or
protected capture was requested.

Do not refresh, reprice, alias, omit a House row, or book-capture this exact
two-event population. Reopen only for a distinct rule-complete joint projection
whose conservative package remains positive after current fees and at least
one adverse tick per leg before depth, or a material fee, tick, rule, or market-
architecture change. Contract SHA-256 values are
`0cb91fb5c8b0ec763c581b91817e01b0cea15209163f73cc497a7a8bc72b7834`
and `38faaa6e90618a75fe68edd99444e37e9644c93ed38cf45c77af935979e65bf0`;
canonical adjudication SHA-256 is
`2fdcd4595af5b5a7e50a0e97fa0559694bcaddd03f844c201894441afc46adbd`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
133, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`93ea1175161da3c536af13d0ac86cc0c08ec1e729cd56b72eafa32c4654c4125`,
and durability-audit SHA-256 becomes
`bd2f8e5bb16a9b9dc64d2f79f626b051147fc20679f72c06d89563909dca8c1e`.

## Polymarket OR-06 hidden-election-sibling rejection

The rendered OR-06 House Election Winner card exposed only Democratic and
Republican outcomes at an apparent combined 0.962 pUSD. The candidate was
distinct from prior retained families, so a one-use public unauthenticated
Gamma metadata contract was frozen before access. Its exact GET returned HTTP
200 with 33,714 retained bytes at SHA-256
`82323cd9184939d9f3d630fb3c26c8c1eedd6af68b1724eb6faa5cd5da5e7829`.

Exact metadata rejected the apparent edge. Event `191602` is a NegRisk event
with eight markets, not two. Democratic and Republican are active with YES
asks 0.964 and 0.007 pUSD. Six hidden inactive siblings remain accepting
orders: `Other`, `A`, `B`, `C`, `D`, and `E`, each with a direct YES ask of 1
pUSD; `Other` is explicitly `negRiskOther`. The two active asks sum to 0.971
pUSD and would retain 0.1266578 pUSD at the five-share minimum after current
fees and one adverse tick per leg, but this is diagnostic only: it is not a
complete payoff package because the hidden siblings are not source-proved
impossible or safely omittable. Every returned YES ask sums to 6.971 pUSD per
share for a one-pUSD event payout, or a -29.855 pUSD gross floor at five
shares.

The exact event is terminal before books. Do not retry, refresh, alias, omit
hidden siblings, or request OR-06 depth. Reopen only for a distinct exact event
whose complete Gamma market set is active, acquisition-capable,
rule-exhaustive, and positive after current fees and at least one adverse tick
per leg before depth. No book, fee endpoint, credential, account, order, fund,
transaction, or protected capture was touched.

Contract, capture-result, raw, journal, and adjudication SHA-256 values are
`35dfa8a68ec3856e6f8ff32bed471ff5378c19a9997bea9b5345e4af061e592b`,
`c60b91d086e1246e87eb39b5845fa9fd3ca60864102351477d583f7c5b96d804`,
`82323cd9184939d9f3d630fb3c26c8c1eedd6af68b1724eb6faa5cd5da5e7829`,
`e6a2bb7fea570f9d506c1af58c20e2dc95fdd8b72047f5b85b57f2be0baa814c`,
and `b8ab67f254bf6278e6acb993c8a73412e3a4465fe3c7d99dc34b4c603d0455e5`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families
become 134, and stable current account-qualified after-all-cost edges remain
zero. Registry SHA-256 becomes
`bafdead76f0ee1aca2c414674764a7ebca7510fd74478f797fa41735007e80dc`,
and durability-audit SHA-256 becomes
`7bfabbc199967018ce0cfde0787850d48f52960b146064e846455899b754dbf4`.

## Polymarket Senate-majority-leader generated-placeholder rejection

Rendered discovery found a distinct exact-payoff lead in `Next Senate Majority
Leader?`: ten displayed open candidate YES asks summed to 0.950 pUSD, the FAQ
said eleven outcomes, and the rules explicitly routed a no-majority or missing-
announcement state to `Other`. Repository aliases and retained journals had no
prior request for the exact event. A one-use public unauthenticated Gamma
metadata contract was frozen before access.

The contract preflight first rejected a manually anticipated `frozen_at_utc`
that was nine minutes in the future. The validator failed before output-path
creation and before any network access, so the one-use request remained
unconsumed. Only that pre-access timestamp was corrected to an already observed
host UTC instant, the canonical hash was recomputed, and resolved-path offline
validation passed. `AGENTS.md` now requires capturing the host UTC clock before
writing, followed by canonical hashing and absolute-path offline validation.

The single permitted GET returned HTTP 200 with 218,775 retained bytes at
SHA-256 `374fa8c998ef05a8c13b321b602651e6ac355d91b2663646dad0844218c47d38`.
Exact metadata rejected both the apparent complete set and a possible party-
projection cover. Event `289908` contains 64 NegRisk markets. Only ten are
open, active, accepting orders; Lindsey Graham is active but closed; and 53
are inactive. The inactive population contains 49 identity-free `Person X`
placeholders plus Dick Durbin, John Cornyn, Rick Scott, and `Other`. Every
inactive market accepts orders in metadata but has a direct YES ask of one
pUSD, and `Other` is explicitly `negRiskOther`.

Every returned YES ask sums to 53.931 pUSD per share, costing 269.655 pUSD at
five shares for a five-pUSD event payout and a -264.655 pUSD gross floor. The
ten open-active asks sum to 0.930 pUSD and retain 0.09003 pUSD after current
fees and one adverse tick per leg, but that diagnostic omits 53 inactive
siblings, the closed Lindsey Graham market, and the explicit `Other` state. It
is not a complete payoff package. The 49 identity-free placeholders also
cannot be assigned to a political party, so no complete Senate-control
projection is proved.

Do not retry, refresh, alias, omit, classify, or party-map hidden siblings, and
do not request this event's books. Reopen rank 31 only for another distinct
exact event whose complete returned market set is open, active, acquisition-
capable, rule-exhaustive, and positive after current fees and at least one
adverse tick per leg before depth. No book, fee endpoint, credential, account,
order, fund, transaction, or protected capture was touched.

Contract, capture-result, raw, journal, and adjudication SHA-256 values are
`5f69284d64c20444343221b592558dfeb731c95217b12760f8ff1e4c11f76491`,
`ff5ddefed63331563e69fd528755dfef2176f0a693268211a42b1dbf699811e0`,
`374fa8c998ef05a8c13b321b602651e6ac355d91b2663646dad0844218c47d38`,
`a661de1ca0749ec5efaab367eb22d4c2cec179bbf3a6d500a83222609f391ea9`,
and `bf4605a90bd9a9389615cef7618b6ce04fbe379ca121bad9db7dbdc65a956772`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families
become 135, and stable current account-qualified after-all-cost edges remain
zero. Registry SHA-256 becomes
`002bbb75244f3333b2e70464b8835922fa7787fd5ccc01a857ed8becb459ee1a`,
and durability-audit SHA-256 becomes
`773581acc22ef94cf350c3d423c4b601514f17701f4ce57a3cdb623e8bc5af7e`.

## Binance SHEIN quanto to Polymarket Perps exact-match rejection

The September 1 official Binance announcement list exposed a genuinely new
instrument trigger: an announced `HK0625USDT` SHEIN quanto TradFi perpetual.
Rank 43 permits a fresh check only when a material instrument change can create
a new exact cross-venue population. The cheapest decisive gate was therefore
one current Polymarket Perps instrument inventory, frozen before access; no
funding, ticker, price, fee, or book request was authorized.

The one permitted public unauthenticated GET returned HTTP 200 with 44,614
retained bytes. The current inventory still contains exactly 67 instruments,
with identical symbol membership to the retained August 29 population: zero
additions, zero removals, and zero `SHEIN` or `HK0625` symbol or base-asset
matches. A company-related prediction market is not a fungible perpetual hedge,
so no fixed-orientation cross-venue funding population exists for this
announcement. The retained primary page proved the announcement title and
date, not completed Binance deployment; deployment was not assumed or needed
for the negative counterpart gate.

The related SHEIN IPO closing-market-cap threshold event was separately
screened only through rendered side-specific discovery values. Its two
unambiguous adjacent guaranteed packages cost 1.81 and 1.548 pUSD against a
one-pUSD floor. Blank or `0.0` higher-threshold `No` buttons were unavailable,
never free; no exact interval sibling was source-proved. That stopped the
threshold branch before Gamma or CLOB.

Do not refresh, alias, or substitute this exact 67-instrument population and do
not request SHEIN funding, prices, fees, or books. Reopen rank 43 only after a
later exact common-underlying instrument appears or another literal material
funding, fee, session, unit, conversion, or execution-architecture trigger is
satisfied. A new listing on only one venue must now pass an exact counterpart,
share-class, contract-unit, quote-unit, and settlement-mapping gate before any
economic request.

Contract, capture-result, raw, journal, and adjudication SHA-256 values are
`74952fcaa3bd83f826c9113051776aecedc46bd7d4add3f522ff517432e672de`,
`360159707942d24e08cca729c390317aac62b3cdbe8ccbfd2fc833155d78f46a`,
`427f7f9578285dc3c4200c3ad5c185f6cbb2a343f91e249184aae652805e9083`,
`42dcb2b4c6cd1249e9aba4565cd47c800c25aef525c43b4f04e16395e7a8f6ac`,
and `bce940e94a6846e2524f01d77f18b7e921bc273b1f027d9dc7cab4a5b99faf50`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
136, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`e077098bcd22df64153ecbd2bc859980acdeaa2eabe4b13925728ea735ab248c`,
and durability-audit SHA-256 becomes
`40491937fb0a4ffd930a190f2ec4ca6c38d25d4b843bd308ad94c1d48ec62169`.

No account, credential, order, fund, transaction, or protected capture was
touched.

## Binance Copy Trading Lead Trader VIP fee overlay

A bounded official Binance announcement-catalog capture exposed a previously
unregistered, market-direction-independent cost mechanism: eligible Futures Copy
Trading Lead Traders can receive a higher VIP fee tier. The catalog itself is
category-specific and not a complete current Binance feed; its one retained page
contained no row newer than the retained September 1 SHEIN discovery cutoff. It
was used only to discover the historical program article, never to claim that no
other Binance category has changed.

Two separately frozen one-use public CMS GETs retained the exact announcement and
its linked program FAQ. The announcement response was 33,055 bytes at SHA-256
`7d1beba2d82f072bfd56c4e55db344b10b5242d0bce2d1209fb823b0a2419327`;
the FAQ response was 30,162 bytes at SHA-256
`5b6ef45a26fc304d4c3e86aacbdabc28350cdc1c3f760c6e7e068b88b39a5c8e`.
The FAQ was published July 16 and updated August 18. Existing lead portfolios
need at least 200 public or signal copiers, or 50 private copiers, at least 60%
of the corresponding 500,000, 1,000,000, or 3,000,000 USDT AUM threshold, and
at least 5,000,000 USDT of authentic copier-generated Futures volume over the
prior 30 days. Exclusive tiers receive VIP4, VIP5, or VIP6 or a two-tier upgrade;
non-exclusive portfolios with more than 60% of assets on Binance receive VIP3,
VIP4, or VIP5 or a one-tier upgrade. VIP5 and VIP6 require at least 25 BNB.
Reviews occur during the first week of each month, with the first round starting
August 3; the current FAQ publishes no program end.

The retained August 27 USD-M USDT fee table supplies sensitivity rather than a
forward guarantee. Adjacent VIP3-to-VIP4, VIP4-to-VIP5, and VIP5-to-VIP6 examples
save 20/20, 20/30, and 20/20 USD per million USDT of maker/taker notional before
the BNB discount, or 18/18, 18/27, and 18/18 USD after that discount. A VIP1-to-
VIP6 example saves 120 maker or 250 taker USD per million before BNB, and 108 or
225 USD after it. At the program's five-million-USDT volume floor, even the
smallest retained positive example is 90 USD, but this is diagnostic only.
Owned eligibility, manual approval, applied tier, current commission, authentic
external copier flow, any preexisting 25 BNB, exclusivity opportunity cost, and
realized fees remain unproved.

The scoped exact-realized fee reduction is accepted as the thirtieth structural
edge only on independently existing legitimate organic BTC, ETH, or SOL Futures
lead flow after owned reconciliation and every incremental cost. It does not make
the underlying strategy profitable and remains non-deployment-ready with a zero
public and account-qualified forward floor. The quota-limited three-month new-top-
trader/KOL experience is excluded from the stable edge. Never manufacture copiers,
followers, AUM, volume, PnL, exclusivity, or evidence; never acquire BNB or abandon
another venue merely to chase a tier; and never double-count the reduction with
BNB discount, profit share, referral, rebate, or another promotion.

Reopen only for a material official eligibility, tier, fee, review, duration,
exclusivity, or product-scope change, or with both designated mainnet credentials,
explicit signed GET-only authority, an independently existing qualifying lead
portfolio, and an exact owned organic fee-reconciliation question. Testnet keys
are not mainnet eligibility evidence and were not used. Applications, manual
review, Binance contact, exclusivity agreements, evidence disclosure, BNB
acquisition, portfolio creation, trading, transfers, and account changes each
require separate authority.

Canonical adjudication SHA-256 is
`542980565432c5160cf0970b3a7e909ea2d028cbc9a0917ebbf7b21fded3f6c5`.
Accepted edges become 30, ranked hypotheses remain 47, terminal families remain
142, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`5689c8bb60c5a36d7832a4a74cb723e10de94cf96a8f90535e7b15256e50cb2f`.
Durability-audit SHA-256 becomes
`082b02117a9c15d7f1e70ea84e344791af7f2a638b0755760a3ebcd1de9ef9ab`.

No account, credential, order, follower, position, fund, transaction, or
protected capture was touched.

## Predict.fun public books, maker rebates, and false cross-venue identity

Public logged-out Predict.fun pages now provide a useful rejection-first route:
the rendered order book and a Polymarket comparator can be inspected without a
Predict API key. This changes the old assumption that every Predict.fun price
question requires authenticated API access. It does not weaken acceptance
gates. A comparator probability is not an executable ask, a missing ask is not
free, and rendered browser observations remain rejection-only unless exact
rules, side-specific depth, fees, and synchronized timestamps are durably bound
under a prospective contract. Predict.fun is a separate BNB Chain venue, not
Binance Exchange; Binance mainnet or testnet credentials do not authorize it.

The first GPT-6 page exposed economic values before a clean book contract was
frozen. That exact population was marked promotion-ineligible rather than
retrofitting the freeze. A disjoint OpenAI-acquisition pair was then frozen for
rejection only. Its rendered resolution rules matched materially, but
Predict.fun showed zero total volume, no last trade, no spread, YES asks starting
at 0.92 for 625 shares, and no NO ask. Polymarket showed executable YES and NO
asks of 0.042 and 0.959. Predict YES plus Polymarket NO cost 1.879 pUSD before
fees; the opposite package was unavailable. The pair is terminal without a book
retry or profit claim.

Four exact current official Predict.fun documentation pages were retained.
Their raw SHA-256 values are:

- Predict Points: `adfef78827f9715a381f6767e1d5e39259089eb39e9338bee04bf9eff16b16ec`
- Maker Rebates: `452224e0e84d7616170f89e18cb16ddf034d06d5fe16b562356fd4f90c1ae560`
- Fees and Limits: `ca8bea438416438f5ba14363c414ebcd65eef143e2529fe19176e7218e1d9e11`
- Chainlink Price Markets: `8b8632103bae76066ed4a67547a313e72b808ad562b1de0befbddc182bf070bf`

Predict Points are terminal as a current monetary edge. The official terms say
PP allocations depend on frequently sampled executable liquidity and other
weekly factors, can be zeroed for non-executable orders, balance manipulation,
or cancellation abuse, arrive only after a weekly calculation period, and will
continue to evolve. They define no deterministic cash value, transferable token,
redemption formula, or account-independent floor. Credit zero and never create
volume, self-trade, wash-trade, manipulate balances, abuse cancellation, or post
non-executable orders to farm PP.

The Maker Rebates source defines a real but narrow direction-independent income
overlay. Makers pay no fee; on each eligible UP/DOWN crypto fill the maker
receives 25% of the taker fee automatically, with no signup, claim step, or
minimum payout. The trial is published to end September 16, 2026, but no exact
end time or inclusivity is stated, so require the live eligible badge and active
program state. The taker fee is
`base_rate * min(price, 1-price) * shares`, with a 2% base and possible 10%
discount. The maximum nominal maker rebate is therefore only 0.0025 USDT
equivalent per share at price 0.5, or 0.00225 with the discount, before every
cost. The page says portfolio history may record either USDT Distribution or
Shares Distribution, so bind the exact received asset and conservative
liquidation or redemption basis.

This exact realized positive rebate is accepted as the thirty-first scoped edge
only on an independently justified legitimate organic eligible maker fill after
adverse selection, inventory, hedge, gas, latency, capital, custody, tax,
failure, and every other incremental cost. It does not establish profitable
market making, account eligibility, deployment readiness, or a positive public
forward floor.

The apparent Predict.fun-to-Polymarket short-horizon crypto hedge is also
terminal before books. Current Predict.fun terms use BTC/USDT Chainlink Data
Streams v3 point reports, separate Up/Down/Flat states, up to a five-second
buffer, and a paused emergency manual-close path. A prospectively selected
current Polymarket round, `btc-updown-5m-1788249000`, was retained at raw SHA-256
`1e320cfa14d8190fe36875e137fd54c9361553eb475c092ae2506f6a4bf644df`.
It uses BTC/USD 60-second TWAP and resolves equality to Up, otherwise Down.
Different pair, averaging, equality, flat, buffer, and emergency branches are
source-proved payoff counterexamples; matching asset, interval, and timestamp
strings are not a deterministic hedge.

Canonical adjudication SHA-256 is
`b054ef3388dd9e97b065120d35792699ef3bbdc40efbd72e2034e77db0efe1fd`.
Accepted edges become 31, ranked hypotheses become 48, terminal families become
148, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`5805fc3f11e3a8f9c4db2a365d7dd1108e0653b3b2a8147135bdeb4de0d456a8`.
Durability-audit SHA-256 becomes
`400296f66e5807219ce6f8b5825a3b4a863c7383a94c766ba9ca546bc3a55343`.

Reopen before the published trial end only for one distinct currently badged
eligible market plus a new official public fill-quality source or explicit
separate Predict.fun read-only paper-study authority, with chronological queue,
fill, cancellation, inventory, hedge, and regime gates frozen first. A Predict
API key is distinct from Binance keys. Orders, funds, wallets, claims, accounts,
and state changes each require separate explicit authority.

No credential, account, wallet, order, position, fund, transaction, claim, or
protected capture was touched.

## Predict.fun public-chain gate and post-September-2 WNBA trigger

The prospectively frozen Polymarket WNBA catalog for September 3 through 9
returned a complete empty population: zero events, relations, or candidates.
The exact window is terminal without a book or fee request and must not be
shifted, aliased, or repeated.

Two current official Predict.fun sources were retained. The technical overview
confirms off-chain matching with on-chain execution and settlement, CTFExchange
`matchOrders`, FeeModuleV2, ConditionalTokens/NegRisk, and fill-transaction
rebate distribution. Neither it nor the audits page publishes the deployed
exchange, fee-module, conditional-token, NegRisk, or eligible-market contract
address needed to enumerate public fills and rebates. The linked 33-page CRE
audit identifies a BSC-mainnet admin multisig, two Chainlink infrastructure
addresses, and a BSC-testnet adapter; none is identified as the required trading
or rebate contract. Never substitute testnet for mainnet or guess deployments
from selectors, bytecode, admin neighbors, or audit examples.

The technical source also states that yield-bearing contracts place collateral
into Venus and that yield managers can claim accumulated yield. It does not
define traders as beneficiaries or publish a trader rate, base, distribution,
or redemption contract. Trader-owned Predict.fun collateral yield is therefore
terminal with zero credited value.

Canonical adjudication SHA-256 is
`aa89b82912335b288506e731382e8ac87fa8e60afb614986f224a1b88d730c22`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
151, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`57d12349bb20e5ed5c58e1bed6db6dbfccbe3ea6c11868b56f16dc056bf0d480`.
Durability-audit SHA-256 becomes
`ec2c84a4fd19dded954e5a632c9b58dc7b3b0ceb3c2f71617c41aac756448894`.

Reopen Predict.fun public-chain fill/rebate work only on an exact official
deployed-contract or explorer reference, a new public fill-quality source, or
explicit separate Predict.fun read-only paper-study authority. Any account,
wallet, order, funded, claim, or state-changing action still requires separate
explicit authority. No credential, protected capture, account, order, fund, or
transaction was touched.

## September 30 ET complete sets and Gemini projection

One prospectively frozen public Gamma request covered the exact two-minute UTC
endpoint band around September 30, 2026, 11:59 PM ET. The service returned its
effective 100-event page with a cursor, so the population is incomplete and
terminal without narrowing or pagination. Its three visible fixed NegRisk
events all failed the rejection-only all-YES floor: the best sum was 1.0085
pUSD. No on-chain, book, fee, account, credential, order, or fund request was
justified.

The retained page did expose a distinct exact-payoff idea: buy YES on every one
of the 30 Gemini Pro release-date outcomes except `No release by September 30`,
plus NO on the separately retained cumulative September 30 deadline. Rendered
economics had leaked before the zero-network contract, so the population is
promotion-ineligible. More importantly, the cumulative event began 6,654,892
seconds before the exact-date partition; without source proof of no qualifying
release in that gap, the one-pUSD floor is not established.

Even granting that optimistic floor for rejection only, exact side-specific
metadata decisively fails: 31 legs cost 1.264 pUSD per share, losing 1.320 pUSD
at the five-share minimum before tick stress and 1.64295 pUSD after one adverse
tick per leg plus exact taker fees. Search snippets and rendered headline odds
were stale or non-synchronous and were not used as executable economics.

Catalog contract/result SHA-256 values are
`1c7b0d2778a6cd46a676af45fa7974834d28105e09278be82ba0654bb77f76bf`
and `c77380085ca24a759d28d1f0a95211b0f3c2c0b391784653a05a421a66c22678`.
Projection contract/result SHA-256 values are
`c5e0e413700266a6d4d6aefe990d8ef5bd46d9af78e05a9b3c2119204be91285`
and `233475bd3a302ba7a905b5cd8eea4d53939513aea8dbdc9fcb5ddefcc8f053ce`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
153, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`5a5040f33e7a14803dd066e565626279c1e3d9e6bfc4c39803237b5010822abf`
and `6b264230457ad0c159d9632c0f395ea7c326b7cd1c94f4d4048dbad13ce2749e`.

Do not repair the cursor, refetch either Gemini event, omit partition legs,
weaken the prehistory gate, or use this leaked population for promotion. No
credential, protected capture, account, book, fee, order, fund, or transaction
was touched.

## Retained U.S. measles scalar threshold ladder

The untouched exact nested U.S. measles September 30 event supplied a distinct
zero-request scalar-ladder test. Before viewing economics, the contract froze
all four thresholds (2900, 3000, 3100, and 3200), all six ordered lower-higher
pairs, the common CDC total-case counter, September 30 cutoff, identical source
fallback, at-least boundary, side-specific `bestAsk`/`1-bestBid` rejection gate,
and deterministic ranking. The incomplete parent catalog was not treated as a
complete event catalog; only this exact nested event was adjudicated.

Two pre-economic local preflights exposed reusable implementation assumptions:
the retained raw response is UTF-8 rather than ASCII, and its numeric display
labels use a trailing `+`. Both failures were preserved, no economics were
decoded or printed, and the final implementation-bound contract was refrozen
after UTF-8 decoding and trailing-display-suffix normalization passed against
the unchanged hash-bound bytes.

All six direction-independent `YES(lower) + NO(higher)` packages had complete
side-specific rejection prices. Zero cost strictly below their one-pUSD floor;
the best cost 1.030 pUSD before fees, ticks, depth, time value, non-atomicity,
latency, unwind, or operating cost. The exact event is terminal without any
book or fee request. Do not refetch, reprice, select one sibling pair, or request
depth; reopen only on a material price, fee, tick, rule, or market-architecture
change.

Contract/result SHA-256 values are
`feb1a65dcbf3e0a3af2e7b33fea7431fc5eebc3439ca295c4a4bf63058265e5b`
and `990207a430a388ed85c74924eceb53b0e68ba7d31c3b3827a6d19a4d04376a50`.
The preserved pre-economic failure SHA-256 is
`c393ac6b0fbb49c487c7b71214fe7348783420be171fc19212c58c5bfab981ae`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
154, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`96982b13beec4577cfa9863b4efa9678fa63983a19842d014f81847ed64567d3`
and `d3c0205a671bb920061b566004dec16b913de0e439804cdbb09c657534331cb8`.
No network, account, credential, book, fee, order, fund, transaction, on-chain,
or protected capture was touched.

## AI Lab second/third rank mutual exclusion

Eight untouched exact nested events exposed four distinct common Labs-view
rankings: Code Arena WebDev, Chinese Text Arena Overall, Text Arena Overall, and
Text Arena Math. Their second- and third-place rules use the same source view,
check instant, ambiguity fallback, deterministic tie sequence, permanent-source
fallback, and candidate population within each family. One named company cannot
occupy both positions, so `NO(second) + NO(third)` has a one-pUSD floor.

The contract prospectively excluded every best event because it uses the Models
view rather than the Labs view. It also excluded `Other`, every identity-free
Company placeholder, and ByteDance/Bytedance across the Math pair because those
labels do not source-prove one cross-event identity. It froze all remaining 67
named-company packages before economics and ranked them globally.

The first preflight failed before economics because the implementation required
inactive excluded placeholders to be executable. That failure was preserved;
the correction kept full population and representation validation but applied
the active executable gate only to frozen named package rows. The refrozen
preflight then passed against the unchanged retained bytes.

Forty-eight packages had complete side-specific `1-bestBid` NO acquisition
proxies. Google in Text Arena Overall was the only strict metadata candidate:
0.18 pUSD for second-rank NO plus 0.81 pUSD for third-rank NO, costing 0.99 pUSD
per share. At the five-share common minimum the 0.05 pUSD gross headroom did not
survive the frozen execution hurdle. One adverse 0.01-pUSD tick on each leg plus
0.06030 pUSD of exact taker fees produced a -0.11030 pUSD floor before depth,
time value, non-atomicity, latency, unwind, or operating cost. No book request
was authorized.

Contract/result/preflight-failure SHA-256 values are
`3fd13b0fb651614e1dbce6f9e85f1baf51a5cb6f020a150efa23e0378e8aaa1a`,
`268aa6e00d349b164f1727e5d47ae34afe05787627e6552307aa04b59cdbe1e6`,
and `32495c463d2748df8b4afbcfc579887d0ee01bb61ea2b933da981bac8a53c151`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
155, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`384d75c8fcec15cc2caecb249171b01ff00dad24c4a5209d94d0880364f97992`
and `4723a56809cf8d6f9ba082a02e17a45f90b326bdfffd226ef53dc80a96a2f0e5`.
Do not refetch, reprice, alias labels, or request books. Reopen only on a material
price, fee, tick, rule, or market-architecture change that makes the exact
package positive after the same fee-and-one-tick stress. No network, account,
credential, book, fee endpoint, order, fund, transaction, on-chain, or protected
capture was touched.

## Creation-window-safe retained deadline implications

A prospectively frozen zero-network screen inventoried 14 active-looking
cumulative-deadline families in the retained September 30 catalog. Ten were
excluded before economics because the later-deadline market started later or
the rules differed. In those cases, an event in the creation gap can make the
earlier market YES while the later market remains NO, so the proposed package
has no source-proved floor.

Only four pairs had identical rules, a strictly later deadline, and a
later-market start no later than the earlier-market start: Discord August 31 to
September 30, Meta Muse Spark September 15 to September 30, and Qwen Plus
August 31 or September 15 to September 30. Two had complete side-specific
prices. Zero cost strictly below the one-pUSD floor. The best Qwen Plus
September 15 NO plus September 30 YES package cost 1.19 pUSD per share and lost
1.10590 pUSD at five shares after one adverse tick per leg plus exact fees.

Contract/result SHA-256 values are
`d85ff8d51b720b2b88d41952d25b003b5d5240771499e37d19579204180adde8`
and `1b3dcf4167ef6bd5dbec85b1f8d5e23f23b0ad7bcac2cfe4e32b58d65afedb23`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
156, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`c5e22c02abcb89ce9ed73b515c32c1fad6a730842c8ff858b5863197f98d14bf`
and `77b63ce226f366adc047bb756407322828aa856bf562d771a3a4e3fd4b21b6fb`.

Do not refetch, reprice, weaken the exact creation-window gate, or request books.
Reopen only on a material price, fee, tick, rule, or market-architecture change.
No network, account, credential, book, fee endpoint, order, fund, transaction,
on-chain, or protected capture was touched.

## NYC September 1 high-temperature fixed-NegRisk prefilter

The exact eleven-bin NYC September 1 high-temperature event fired rank 31's
distinct fixed-NegRisk trigger. The official rendered discovery page exposed
all eleven displayed Buy-YES prices before the contract freeze. That mistake is
preserved: this exact event is permanently promotion-ineligible and cannot
select a book candidate or justify downstream access.

One frozen exact Gamma GET was still retained for rejection and lineage. It
reconciled all eleven markets. Buying every YES cost 1.0485 pUSD for a one-pUSD
floor, so the complete-set route failed before books. Every displayed
one-NO-to-other-YES identity was gross-positive, with a best 0.0485-pUSD gap
from the `73 F or below` source row, but Gamma displayed prices are not
executable output bids and the pre-freeze economic leak bars promotion.

Contract/result SHA-256 values are
`f870a6417968fa86dc3b194905a7d3d25dca59b4969fd69768290256c48e5336`
and `c3dc478bc582ec5f60ace208fab5cb28e9d1cf593059894443d2783c12f114ac`.
Raw-response and request-journal SHA-256 values are
`28314d59c66cf50b81232ba85ae4017248f2ad73e498535e28b6526cfe03b9d9`
and `201bdcb74190a36fe9f37de795bda9fc527fe502039279a9d808f44d7b5daefb`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
158, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`fd3948dffc72413f18f54fc966278fae617d3050f2fe6fd682c8308874e231f5`
and `c7e4f16beee4a9bade1d359774849b22f53155db908e0901842d61e2b59db636`.

Do not refetch, alias, reprice, or request books for this event. For the next
distinct daily weather partition, freeze deterministic event selection and the
complete expected population before any outcome-aware rendered page or search
access. One public unauthenticated request was made; no account, credential,
book, fee, on-chain, order, fund, transaction, or protected capture was touched.

## Polymarket US versus Global exact-payoff identity gate

The first cross-venue parity study used Polymarket US's documented public API
as a source-first gate, not a price-discovery shortcut. Before requesting the
U.S. venue, the contract selected one objectively resolvable exact Global title
from the hash-bound retained September 30 catalog: whether the U.S. House will
pass a bill restricting military aid to Israel by September 30. Candidate
selection used titles and rule objectivity only; no candidate economics were
inspected.

One frozen unauthenticated `GET /v1/search` retained two U.S. events and zero
exact-title matches. The payoff-identity gate therefore failed before prices:
there is no source-proved same contract to arbitrage across the two venues.
Matching themes or approximate titles are not substitutes. No price, BBO,
order-book, fee, account, credential, order, fund, or transaction request was
made or justified.

Contract/result SHA-256 values are
`fb9f342d39d672c41c33c08bb11685bf43403d1c9650c2ff05e981ef3226c960`
and `2ffbcad5a046dca1008bfc8b11693e7621ed23848fde8c0ddf24f03453375f6e`.
Raw-response and request-journal SHA-256 values are
`4f216f2c962498d149c846ba5f5044d11328bbdfb1c45855a33014ea156e8742`
and `4357311cb699c3458e46b25faa69b05e4bd55f51d9c80b2c6ce414b2261197cc`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
159, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`979ba80e079e03afb3a6991fcc186a5e5e61749e5485467317e47ceab8efffdd`
and `1513efdc5d7c34a7ee14c45fa2725b0fa373eeb1ab1691c6bd4a041df6cead8c`.

Do not retry, alias, paginate, or substitute a similar House or Israel question.
Reopen this family only for a distinct exact contract selected and frozen before
any Polymarket US search, or after a material listing or venue-architecture
change. The protected holding-yield partial capture remains untouched.

## Polymarket US resting-order liquidity incentive source gate

Official Polymarket US documentation exposes a more promising direction-neutral
mechanism than approximate cross-venue title matching. The open Liquidity
Incentive Program scores resting orders every second whether or not they fill,
allocates fixed USD pools proportionally by discounted price-distance and size,
and can stack with a separately documented 25 percent maker share of taker fees
when an order fills. The same sources also make the missing economics explicit:
score share can be zero, Target Size must be met, sub-one-dollar rewards are not
paid, parameters can change, and canceled or postponed games receive no reward.

Before current program access, a one-use contract froze the complete-page gate,
active-interval checks, liquidity-program type, positive finite reward pool and
Target Size, valid Discount Factor, and a deterministic ranking by reward pool
per hour per target contract. The official active-program GET returned HTTP 503
with 33 retained bytes. The failure is source-side and was durably journaled;
the exact request was not retried and the frozen adjudicator was not run.

The mechanism remains a high-quality research lead, but no current positive pool,
owned score share, public book candidate, jurisdiction or account eligibility,
deployment readiness, or profit floor is proved. Accepted edges therefore remain
31, ranked hypotheses remain 48, terminal families become 160, and stable current
account-qualified after-all-cost edges remain zero.

Contract/failure/raw/journal SHA-256 values are
`3e6d4fdbe0feb3584439b8d6976d751dabb86209c6484d0a0885e8b51984521f`,
`30707da50f9d52740ecb59376a0e060edcf5d88cace334a8bdea169c703a6047`,
`997a01746aa1c0f5cd00f437f9422e133b8b86244aabdf8516c22c92d03ea935`,
and `db5c8be67102dad3b868204badab43347dad0b73ef2c780f44fc62c1e78dc0af`.
Registry and durability-audit SHA-256 values become
`84967248716813aca5de3dc68c154ccbedb01ef74c2f3cf6fbd56ccd234f0952`
and `34f82a9403b62b9c96ac0c8d9f6305c07b60fdedc7cf9d02fc1354b8dad07e95`.

Do not retry, alias, paginate, request authenticated earnings, select a stale
documentation example, or request books. Reopen only after a material official
endpoint-availability, documented-host, or schema change with a fresh contract
frozen before access. Any account, credential, order, deposit, fund movement,
or transaction requires explicit separate authority. Never manufacture volume,
score, fills, liquidity, or rebates through self-trading, wash trading, spoofing,
fake orders, related accounts, manipulation, or program abuse.

## AI Arena Overall creation-safe threshold ladder

The retained September 30 catalog exposed five AI Arena Overall score
thresholds with one common leaderboard, score column, style-control setting,
cutoff, at-least boundary, unavailable-source wait, and permanent-unavailability
fallback. A prospectively frozen zero-network screen validated the complete
five-market event before economics. The 1510 market was closed and could not be
an acquisition leg.

Among the active 1520, 1530, 1540, and 1550 markets, four of six ordered
`YES(lower) + NO(higher)` pairs had a source-proved one-pUSD floor. The
1530-to-1540 and 1530-to-1550 pairs were excluded because the lower market
started 4.588726 seconds and 0.253017 seconds after the higher market; a higher
threshold reached in those gaps can leave both proposed legs at zero. All four
safe pairs had side-specific prices. Zero cost strictly below one pUSD. The
best 1540 YES plus 1550 NO cost 1.035 pUSD per share and lost 0.19670 pUSD at
five shares after one adverse tick per leg plus exact taker fees.

The first preflight stopped before economic access because the validator
textually compared a retained millisecond timestamp with its semantically equal
microsecond serialization. The failure was preserved and the refrozen runner
compares timezone-aware UTC instants without weakening exact equality.
Contract/result/preflight-failure SHA-256 values are
`8246a5ebc592cdcdd847e1717af26cbf57db1b51d63edd7712b69e1870f617d9`,
`cec42898652936f8eeea78a956f8a9f6e55916a57cd1adc56485f1eb9636387f`,
and `36475331926c94e259959ec71d38015665bc1222389b024c2a9a225a345b4da2`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
157, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`2dc83e4141aabd5d35a2e3a34f0857443579f684ebed7250d52aa7ccd87283c7`
and `f37ee3effc4f76c737d9f6a5b33bf463fb908bbd3e1cf8b57f3e254aca9eb9a7`.

Do not refetch, reprice, ignore exact creation instants, or request books.
Reopen only on a material price, fee, tick, rule, or market-architecture change.
No network, account, credential, book, fee endpoint, order, fund, transaction,
on-chain, or protected capture was touched.

## Polymarket US maker-rebate cent-rounding identity

The next distinct direction-neutral mechanism used the current official
exchange-wide Polymarket US fee schedule rather than another market-price
screen. Before the one permitted public documentation GET, the contract froze
the exact source, required maker coefficient `0.0125`, taker coefficient
`0.05`, banker's cent rounding, and a complete 99-price by 1000-quantity grid.
The retained 371114-byte source passed all gates.

The exhaustive 99000-combination adjudication found 98641 positive rounded
maker rebates and 48949 positive maker-rounding uplifts. The maximum uplift
over the unrounded rebate is exactly `0.00500000` USD per trade, the maximum
effective maker share is `0.5` of the rounded taker fee, zero positive maker
rebates coincide with a zero rounded taker fee, and zero maker rebates exceed
the rounded taker fee. This proves a bounded accounting overlay, not a
standalone arbitrage or a profitable market-making strategy.

The thirty-second accepted scope is therefore only the exact positive USD
maker rebate actually credited on an independently justified legitimate
organic Polymarket US maker fill after every incremental cost. Do not split,
combine, or time orders to optimize rounding; manufacture volume or liquidity;
self-trade, wash-trade, spoof, abuse cancellation; assume partial-fill grouping;
or credit an advertised amount before an exact owned receipt. Reopen only with
explicit separate read-only Polymarket US account authority plus independently
existing bona fide organic maker fills, or after a material official fee,
rounding, fill-grouping, payment, or exchange-architecture change. Any order,
fund, account, or state change still requires separate explicit authority.

Contract/source-result/adjudication/raw/journal SHA-256 values are
`f3a9e18049344c0f1f02109b3adbc09628eec5e91766cfeae2eb8cab6e838101`,
`42d098f39c3ba01bb8210982a56d65cf5fe3e54772bfef7f3f2872ed47037f33`,
`d13ecf94825c27143a36af243e8fa008b6b99334e3651343189a55b823c6b21d`,
`58ba04e9d5da110d16dce0f8f9dff6ee4e6f2ad1c362d920e2ccb44b3733de`,
and `0d64f9857b51437605aa57de569dd23426040d908f54d337ed572e8adfd7891a`.
Registry and durability-audit SHA-256 values become
`c91039e95182af5c40207facfb3d511923ddb3e2e01fa5e8a601ff2d488df422`
and `cc629da0b30ed3652426825cb45e1128a30ffe13cd71ef8e33de1546633f9026`.
Accepted edges become 32, ranked hypotheses become 49, terminal families become
161, and stable current account-qualified after-all-cost edges remain zero.
No account, credential, order, fund, transaction, on-chain action, or protected
capture was touched.

## Polymarket US complete catalog and open incentive overlays

A prospectively frozen one-use public Polymarket US Events API request returned
one active, nonclosed, nonarchived event, strictly below the requested 500-row
limit. An offline zero-economic-field join compared its exact title and exact
source terms with all 100 events in the retained September 30 Global catalog.
There were zero exact title matches and zero exact source identities, so the
cross-venue route stopped before any price, BBO, book, fee, account, credential,
order, fund, or transaction request. Do not retry with fuzzy titles, aliases,
pagination, or repricing. Reopen only after a material listing or venue
architecture change, or for a distinct population frozen before US access.

Catalog contract/source-result/adjudication/raw/journal SHA-256 values are
`a327468de94837db6644abe15878fe5af2881965ffd765f17fdcec19a08c85fc`,
`2cae855c1d22f68da0fed37406af5086dcad7713e94e4236c1bb1cd9937db956`,
`d4ef63463e60b52cf97bbc5044e679b6c8f529813c8e843b9864ffef87700988`,
`c94c2f2e8d54d036100e7f2add7957f9d9596a26274d7da6b551305d5d20dad2`,
and `d1c453f96068bfc739003508c4a3605c7125aea83818593f59972e9b58024ea0`.

The next prospectively frozen one-use current official source reconciled the
complete open Polymarket US incentive page jointly rather than selecting one
program after access. Volume rewards are proportional to eligible contract
volume, with only trades described as between 3 and 97 cents counted; the
source does not specify endpoint inclusivity. Fill rewards
are proportional to filled resting-order volume, paid weekly, and capped at
50,000 USD or 10 percent of the pool per trader; the displayed March 14 table is
historical, not a current pool. The open first-deposit program advertises a
20 USD account credit within 30 days for a genuinely new participant onboarded
through an ISV who successfully onboards and deposits at least 1 USD.

The thirty-third accepted scope is only an exact positive owned Volume or Fill
reward actually credited on independently justified legitimate organic
eligible taker volume or resting maker fills after every incremental cost. The
thirty-fourth is only exact positive owned first-deposit credit actually usable
or withdrawable after independently intended genuine onboarding and deposit and
every incremental cost. Both public forward profit floors are zero; neither is
deployment-ready or stable/current/account-qualified. Never manufacture volume,
fills, orders, deposits, accounts, or eligibility; self-trade, wash-trade,
spoof, cycle related accounts, abuse cancellation, or misrepresent new-participant
status. Any account, order, deposit, fund, or state change requires separate
explicit authority.

Incentive contract/source-result/adjudication/raw/journal SHA-256 values are
`4eb9d5db0fc15ef5ee8941dbed781207eb4dde089ab32438862819b2e59dfcb8`,
`d3c1630c24c2f5156c9ed75d04742b112f928ddd6c7e3b8672f85ec8a4469ac3`,
`d57ce7474a91cce5ab9e07b7ed5d01bf970b1969c2cc6921c506e956e50dc990`,
`abc2ffaabd69844cf323c6b85cf334418ff12e7401b778c489a2aa36d8fe6333`,
and `86de752574f1bb5d2f1818204325d28231060f8a87d606599c52b8c9279a0ff9`.
Registry and durability-audit SHA-256 values become
`c74173fe11f09db70393b67c5f37a4495bdfc4a78cd6ced41f4872fe825ebb69`
and `ff1c9aae493f8acf533464ad90352a486d1dfe23c73511ba1f7ba9a628441922`.
Accepted edges become 34, ranked hypotheses become 51, terminal families become
163, and stable current account-qualified after-all-cost edges remain zero.
No credentials were used and the protected Polymarket capture was untouched.

## Polymarket US promotional-credit withdrawal conflict and edge revocation

The official documentation index exposed a previously unconsumed source titled
`Why can't I withdraw my promotional credit?`. Before opening it, a one-use
contract froze the exact official Markdown URL and the decision rule that an
advertised credit is not cash. The retained 1,528-byte source proves that promo
credits are trading credits, not cash, and cannot be withdrawn even after use,
settlement, or liquidation. Only resulting trading proceeds may possibly become
withdrawal-eligible after the full credit is used as collateral and the related
positions settle or are liquidated. Withdrawal also requires a linked payment
method, an initial deposit, and its clearing delay.

That source linked the complete User Incentive Programs terms. A second
prospectively frozen one-use 4,959-byte official Markdown capture proved
additional conflicts and controls:

- the general open-incentive page says a new ISV-onboarded participant deposits
  at least 1 USD, while the full terms say the participant qualifies only for
  the campaign presented before deposit and Campaign A requires at least 10 USD;
- the specific FAQ says the original credit always remains non-withdrawable,
  while the full terms say incentive funds may become withdrawable after
  collateral use and release through settlement or liquidation;
- Polymarket may withhold, cancel, or reverse incentives for suspected fraud,
  abuse, manipulation, self-dealing, self-referral, coordinated activity, or
  activity inconsistent with program terms or exchange rules;
- Daily Trading and Deposit and Trading program details are not yet live.

The contemplated direction-independent free-bet-style hedge therefore stops
before economics. No public source expressly permits the exact offsetting hedge
or defines state-complete credit consumption, cash-account mixing, fees,
whole-contract rounding, fill atomicity, liquidation, withdrawal, or reversal
economics. The face amount is non-withdrawable and positive resulting proceeds
otherwise require a favorable trading outcome. The former thirty-fourth
accepted edge is revoked rather than preserved as a tautological
exact-positive-proceeds predicate. It remains rank 51 as a zero-floor
hypothesis and may reopen only on exact account campaign terms, explicit
separate read-only authority, and current official permission plus complete
economics for the exact non-abusive hedge, or on a material program change.

The shared public-source validator also now rejects a missing `request_name`,
nonexact or empty output paths, and a missing, empty, or malformed implementation
list before creating output directories or making a request. Four focused
assertions cover the valid contract and each observed pre-side-effect failure.

Usability contract/source-result/raw/journal SHA-256 values are
`f2328021f2f937178ef645a32b8730dac8ee19a4b9bf716a6dd3c3fa28ba5012`,
`0e127bf1c2abc044ebc5e26815e331ed840e9440026a79f33b1ec45075e92106`,
`75585a94e745d315da0ab0ef55a4e7efc17b41e5aed48f0bcb6539bb458f7174`,
and `b3b1b51621b054f3857b7db2505577846412e64c831a3932a057becc161bc874`.
Full-terms contract/source-result/raw/journal SHA-256 values are
`9f237e7b0092d60553364c396d20822d2f7fe805535d451c4114a38c94839c92`,
`8bf6ca023bf62623b5c0334a4866f3df4d5cbe4f9460e333b8c620d46c99caad`,
`3a1987bbff88abf718c43b190db8b40d040ab5ba9c42141920e250e186f9d1ec`,
and `c22e4a529233a4fb2dbf2c80008c86b1073a36b5e9b0191e3eb7a1e42992ac0c`.
Adjudication, capture implementation, registry, and durability-audit SHA-256
values are
`14411b0378171e180f2e4423bb89aa491522315082d216ed219f6345b803883b`,
`ba0e76bd6643eedc072cc1fcff2aa363c1dc14a5b8e14956e874c7843637b376`,
`a788b2d7d26e95b8719e3943338b27bb7500453dc149bc9f66fc2e9617b1594a`,
and `7d9449c50cadd7ce8987f4d80733b4fa887ddc2e4342c902f51dcb877a4e2a21`.
Accepted edges become 33, ranked hypotheses remain 51, terminal families become
164, and stable current account-qualified after-all-cost edges remain zero.
No BBO, book, fee, account, credential, deposit, order, fund, transaction, or
protected capture was touched.
