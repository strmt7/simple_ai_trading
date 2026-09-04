# Agent Start

Read `AGENTS.md`, then `docs/CONTINUATION.md`. Those two files are the current
operating contract. Historical handoff text is archived under
`docs/archive/agent-history/`; use it only for provenance.

## Current Truth

Paradex fee-route terms are now source-bound: lower nominal fees require
qualification of the actual order route and incremental execution costs.
See `review/2026-09-04/paradex-fee-routes.md`. This source-only addition leaves
all acceptance counts, consumed market results and retry triggers unchanged.

The new Paradex/Binance funding-index population is consumed with no survivor.
Its 84 public responses reconstruct 78 intervals; gross spread fails even one
20-bp execution allowance over the whole window. See
`review/2026-09-04/paradex-index-review.md`. Canonical terminal observations
are now 188; counts in older checkpoint paragraphs are historical.

Maker execution review: full-fill support flags do not establish flat inventory
or zero partial exposure. Read `review/2026-09-04/maker-execution-semantics.md`
before reusing Round 57/58 support infrastructure for economic labels. Preserve
old code/results; no new capture or training trigger follows from this audit.

Review update, September 4: use the current registry and the top of
`docs/CONTINUATION.md` for counts and routing. The dated sections below are
historical checkpoints and can contain superseded counts or plans.

The exact NYC September 5 long-only frontier is consumed: configured fees
eliminate every evaluated row even without adverse ticks. See
`review/2026-09-04/nyc-sep5-frontier/review.md`; do not refresh the event or
request missing sides/books. Canonical terminal observations are now 187.

`docs/REVIEW_2026_09_04.md` records the ongoing comprehensive review. The new
September USD1 campaign is documented, but its first distribution rates remain
unpublished; its next scheduled public reconciliation is September 11 at
18:00 UTC. The expired August campaign is a separate observation.

| Boundary | State |
| --- | --- |
| Release | `0.1.0-beta.1`; experimental |
| Development branch | `main` only |
| Binance | BTC, ETH, and SOL; paper or testnet/Demo only |
| Polymarket | Order-capable BTC, ETH, and SOL research is disabled by default; public unauthenticated structural discovery may cover other markets, with no live authority. |
| Accepted edges | Thirty-seven scoped structural edges. The canonical complete scopes, counts, and retry gates are in `docs/model-research/structural-edge-priority-registry-v1.json`; none is deployment-ready or fully account-and-external-cost-qualified. |
| Live-money authority | None |
| Historical cutoff | `2026-08-14T00:00:00Z` |

No model, AI component, backtest, capture, or paper result may be described as
profitable without reproducible source-bound after-cost evidence. AI may veto
or reduce risk only; it never creates positions, selects leverage, overrides a
safety gate, blocks Stop, or submits an order.

### Latest WDCB/NVDAB dividend source-floor gate

The distinct September 4 WDCB/NVDAB official dividend announcement is
consumed. One frozen public CMS GET retained 69,675 bytes and proved that only
the net cash amount after applicable withholding taxes, fees, costs, and other
deductions is reinvested. It publishes no positive gross amount, no finite
upper bound or complete formula for every deduction, and no deterministic
positive final units or multiplier increment. Both exact episodes therefore
have a zero source-bound conservative net-distribution floor and do not satisfy
rank 34's full retry trigger.

Do not refetch or alias the article and do not request WDC or NVDA issuer dates,
books, funding, accounts, credentials, orders, or funds for these episodes.
Reopen only when a future independent episode first binds a strictly positive
net floor and its bStock snapshot materially precedes the official exchange
ex-dividend adjustment. Accepted edges remain 37, ranked hypotheses remain 65,
terminal families become 185, and stable current account-qualified after-all-
cost edges remain zero. Registry and durability-audit SHA-256 values are
`2b0a4ec1734c8e34362e978fd7d6fbccf2ad49db435b1d1a83c80f16c1cd52c6`
and `4d956c98e5ef90a425f5dcd0c40017049c057e3d6d1fa6633a6020285c7ff4dd`.

### Latest Binance CXMT four-hour funding trigger

An exact official notice proves that Binance changed `CXMTUSDT` from eight-hour
to four-hour funding effective 2026-09-04T08:15:00Z and halved the
per-settlement cap from 2 to 1 percent. Retained inventories prove an exact
active `CXMTUSDT` to Polymarket Perps `CXMT-USD` match. The arithmetic absolute
daily cap remains 6 percent, so the change creates a history test, not an edge
or a favorable-rate claim.

Do not sample early. At or after 2026-09-06T08:10:00Z, freeze the receiving-leg
orientation before history access and run one twelve-settlement post-change
test split 6/3/3 across training, validation, and test. No books, fees,
accounts, credentials, orders, or funds are permitted before a persistent
after-hurdle history survivor. Trigger result SHA-256 is
`99d0fded6f7378d0b398b33cd5221f515704cd8c31c3b360e9773eac784f6402`.

### Prior Binance scheduled-yield publication gate

The exact September 4 trigger ran after the promised 18:00 UTC deadline under
prospectively frozen one-use rules. Both official public articles still expose
placeholders instead of the promised USD1 final and RLUSD third completed-week
rates, and both retain their August 28 update timestamps. The capture therefore
failed closed without extrapolation or adaptive retry.

USD1 holding-airdrop forward yield is now zero because the campaign ended. Its
prior positive weeks remain historical scoped evidence only. RLUSD remains an
unaccepted candidate with zero public forward floor. The separate USD1 Simple
Earn candidate retains all prior account, capacity, horizon, transition, and
all-cost gates. Accepted edges remain 37, ranked hypotheses remain 65, terminal
families remain 182, and stable current account-qualified after-all-cost edges
remain zero. Registry, durability-audit, and adjudication SHA-256 values are
`4ceb5de5af67264d50129ae2b44ee33e4ded59cdb3a32eef4ffe85459002e3ee`,
`d0736b1f3da7af118516993839f271d10dcadc3a3edefaef9202a0d05f6ee2f7`,
and `5728de0226526a070c44179f3dc4eb673098349b38e62e646722a6f548f126bb`.
Do not poll or retry the consumed September 4 pages. Reopen only on a separately
observed newer official article timestamp, a material economics change, or the
distinct RLUSD fourth-distribution trigger after 2026-09-11T18:00:00Z.

### Prior Binance Options block-trade atomicity rejection

One frozen exact official public GET retained a 40,487-byte four-page Binance
Options Block Trade EAPI PDF. The text-only capture runner failed closed after
durable retention because the extensionless source was PDF rather than UTF-8;
the request was not retried. Complete offline extraction and visual review of
the hash-bound bytes proves `POST eapi/v1/block/order/create` accepts a
mandatory `legs` list but explicitly caps it at `Max 1 (only single leg
supported)`.

The documented block-trade API therefore cannot atomically execute either a
two-leg vertical or four-leg box and cannot reopen fixed-payoff box parity.
Rank 37's separate account-gated RFQ UI two-leg route is unchanged and still
requires explicit quote-request-only authority plus eligible account access.
Accepted edges remain 37, ranked hypotheses remain 65, terminal families become
182, and stable current account-qualified after-all-cost edges remain zero.
Registry, durability-audit, and adjudication-result SHA-256 values are
`5e7c835dffe9314325a95724de35875637177a3a17aa1886506aad0b6ca6883c`,
`1ec056ab4102eeea8631449e83444b7e3692d176c6016081651c786f5ecbb4c2`,
and `31f0149202f997f0ba7015629d6c3df1e9f86b6580c5f98a34c024d2f3da7b6d`.
Do not refetch, alias, use batch orders as atomic execution, or request quotes,
books, commissions, accounts, credentials, orders, funds, or transactions for
this workaround. The public-source capture helper now records a clean
fail-closed format result for future non-UTF-8 responses instead of raising
after retention; it does not weaken any economic gate.

### Latest Binance Futures Bonus Voucher downside rejection

One frozen current official Binance FAQ says a Futures Bonus Voucher supplies
same-cryptocurrency collateral, can offset trading losses, and lets generated
profits be withdrawn or transferred. It also says the FAQ is not legal terms
and may be outdated. It omits loss ordering, fees, funding, liquidation,
deficits, clawbacks, mixed-balance and other-asset protection, additional
margin, exact depletion, and transfer or withdrawal effects.

Both the fully nonrecourse free-option gate and the narrower exact organic-cost
overlay gate fail. Accepted edges remain 37, ranked hypotheses become 65,
terminal families become 181, and stable current account-qualified after-all-
cost edges remain zero. Registry, durability-audit, and terminal-result SHA-256
values are
`cc6d24ba7c3485e31f18896d8fe67be4eaeaed29660fc29b0543f406f6c13516`,
`d001f3b28b65e9e440cd1b1510de01ed853523b36e518155bd93526d78d8dc73`,
and `57822cbd67822d8c8d61e6d353db99b42dc9895ff19e51e6825beee0a4f66423`.
Do not refetch, follow linked terms, open Rewards Hub, use testnet credentials
as mainnet evidence, claim, activate, deposit, trade, or mutate state. Reopen
only on rank 65's complete controlling-term and independently owned-voucher
trigger; every state change needs separate authority.

### Prior Binance Trading Fee Rebate Voucher scoped edge

One frozen current official Binance FAQ proves that an activated Trading Fee
Rebate Voucher refunds eligible net-paid Spot, Margin, or Futures fees at its
voucher-specific trade-back percentage. The daily refund reaches Spot Wallet in
USDT, USDC, or BNB the next day, after VIP and BNB discounts, subject to the
remaining-balance cap, expiry, product settings, and exclusions.

The accepted scope is only exact positive owned rebate actually credited on
independently justified organic eligible fee-bearing activity after every
incremental cost. It must never justify generating or churning volume, and no
other fee reduction may be double-counted. No owned voucher, exact terms,
eligible organic fee, receipt, account qualification, persistence, deployment
readiness, or positive public forward floor is proved.

Accepted edges become 37, ranked hypotheses become 64, terminal families remain
180, and stable current account-qualified after-all-cost edges remain zero.
Registry, durability-audit, and adjudication-result SHA-256 values are
`8a9c3282758f348a7009af934a17ceffdcd5501b17292717555684ff6686063a`,
`530f25f5b07815e08ce94a214f5b319da394bc19ba8ced83e5368a434399ef4c`,
and `f2d8c368459ac59d440e388c0a43e976ab33aa0404132455327ad7001fce17aa`.
Do not refetch, open Rewards Hub, use testnet credentials as mainnet evidence,
claim, activate, spend Points, trade, or mutate state. Reopen only on rank 64's
material official change or explicit signed GET-only mainnet authority plus an
independently existing voucher. Every state change needs separate authority.

### Prior Binance Futures Free Position downside rejection

One frozen current official Binance FAQ proves that a Futures Free Position
uses given trading margin, fixes voucher parameters, lets the user choose Long
or Short, permits closing at any time, and makes generated profits withdrawable
or transferable. It does not say that losses, fees, funding, liquidation,
deficits, clawbacks, or other liabilities are capped at the supplied margin or
cannot reach the user's other balances.

The exact zero-personal-capital free-option hypothesis is therefore terminal.
Supplied margin and withdrawable upside alone do not prove a nonnegative payoff,
positive expected value, stable profit, or deployment readiness. The run stopped
before prices, books, accounts, credentials, Rewards Hub, voucher activation,
positions, orders, or funds.

Accepted edges remain 36, ranked hypotheses become 63, terminal families become
180, and stable current account-qualified after-all-cost edges remain zero.
Registry, durability-audit, and terminal-result SHA-256 values are
`3b48e46b34d06e443d8a52d33554b0c7ff6db0ea35e9fb73aba870b5d250c3cc`,
`99bab9069d94f690de7781310f8124a1dd9cde1c35380185cde9b4c907c8d329`,
and `a258fff5e8a96f8a36de8aca28d72d15d7e8d29c26a262d9df2c673a5b750fff`.
Do not refetch, alias, follow the adaptively discovered terms link, open Rewards
Hub, or use testnet credentials as mainnet evidence. Reopen only on rank 63's
exact complete nonrecourse-terms and independently owned-voucher trigger.

### Prior Binance Token Voucher source-gate rejection

One frozen exact official Binance FAQ capture returned HTTP 200 and retained the
substantive ordinary Token Voucher mechanics. The source says redeemed free
tokens reach Spot Wallet and may be used, transferred, or withdrawn `with no
restrictions`. The contract, however, guessed the unobserved exact phrase
`without any restrictions`. Because every exact phrase was precommitted as
mandatory, the source gate failed and the exact hypothesis is terminal.

The retained mechanics are hypothesis-generation evidence only; they do not
rescue the failed contract or establish an owned voucher, acquisition cost,
token, amount, expiry, eligibility, realized Spot credit, profit, or deployment
readiness. Do not rewrite, refetch, alias, open Rewards Hub, or use credentials.
The repository rule now requires exact source-observed phrase forms, minimal
observed substrings, or prospectively frozen alternatives before access.

Accepted edges remain 36, ranked hypotheses become 62, terminal families become
179, and stable current account-qualified after-all-cost edges remain zero.
Registry, durability-audit, and terminal-result SHA-256 values are
`4bfa33cf6f44af9876a0a500efdd704499f802a7d57927736c144906d850ad1d`,
`8ace9a89c7b36136ccec36ada332955828cc37eab10fa199c89dddd759bc6d23`,
and `1776c7bcd318edd3c703446f7e2ca746922efc307c2c295075dd0b58eb03c752`.
Reopen only on rank 62's exact material-change or independently owned-voucher
read-only trigger. Any claim, redemption, transfer, withdrawal, conversion,
order, or state change still requires separate explicit authority.

### Prior Binance Simple Earn Trial Fund scoped edge

One frozen exact official Binance FAQ proves that Simple Earn Trial Fund face
value cannot be withdrawn or transferred, is used only for APR reward
calculation, and never reaches Spot. A used voucher creates a locked position;
its APR rewards reach Spot daily for the voucher period, and early redemption
is prohibited. Token, amount, APR, and validity are voucher-specific.

The accepted scope is only exact positive Spot APR rewards from an independently
awarded zero-cost voucher after every incremental cost. Face value is not cash
or profit, and Points-funded or otherwise acquired vouchers require full
opportunity-cost subtraction. No owned voucher, positive APR, eligibility,
receipt, account qualification, persistence, deployment readiness, or positive
public forward floor is proved. Testnet credentials are irrelevant to mainnet
Simple Earn state and remained unused.

Accepted edges become 36, ranked hypotheses become 61, terminal families remain
178, and stable current account-qualified after-all-cost edges remain zero.
Registry, durability-audit, and adjudication-result SHA-256 values are
`5c8474e2446e3bf8e6adb7ed5aad3e75ff0be1c72eaf513fd70445ae691b2798`,
`cf6e6cfdfd15472e1506153780cedcf5bc3969afe530c1119d30ba394acf476e`,
and `d1738d58015b2ee1e34d8dc3bc24eee15cceeeec8730b2ef4c1b8d35a5f30f70`.
Do not refetch the FAQ, open Rewards Hub, claim, redeem, subscribe, spend Points,
or mutate state. Explicit signed GET-only Binance mainnet authority plus an
independently existing voucher is required even for account reconciliation.

### Latest Binance-Backpack funding rejection

The distinct prospectively frozen Binance-USDT versus Backpack-USDC BTC, ETH,
and SOL perpetual funding screen aligned 270 Binance eight-hour buckets to
2,160 Backpack hourly intervals per asset. Training selected short Backpack
and long Binance for every asset. Gross carry was frequently positive, but zero
asset survived validation and test after 20 bips execution, a 10 percent annual
two-leg capital hurdle, 25 bips quote-unit stress, and 25 bips custody, latency,
and failure stress. BTC, ETH, and SOL test nets were respectively about
`-118.97`, `-113.62`, and `-110.22` bips.

A v1 preflight failure on Backpack's timezone-naive whole-hour interval label
is preserved. Official Backpack terms source-bound the sole v2 parser
correction by anchoring hourly perpetual funding at 08:00 UTC and payment at
each interval end. No population, role, orientation, alignment, cost, gate, or
raw input changed, and no economic value was emitted before v2 was frozen.

Do not refetch, resample, realign, refit, weaken costs, or request premiums,
basis, books, accounts, credentials, orders, or funds for this exact
population. Reopen only on a material venue funding, fee, quote-unit, custody,
latency, capital, or execution-architecture change, or a distinct prospectively
frozen nonoverlapping population.

Accepted edges remain 35, ranked hypotheses become 60, terminal families
become 178, and stable current account-qualified after-all-cost edges remain
zero. Registry, durability-audit, and adjudication-result SHA-256 values are
`d82874659b6cd34e6713253d6322af9833cf79a6a4ad71809fc9ce9a1dd5619c`,
`6f8d7cc5838bf64251fb6e40e0ff4847f49e79454874f0eaf7a4ec7117843f87`,
and `ef665320c7cb3537d5f1a9b296fb95ca545def685fe7bb25abbda52c1193f133`.

### Latest Binance-Paradex funding source-gate rejection

The distinct BTC/ETH/SOL Binance-versus-Paradex direction-neutral funding
family was prospectively frozen before any funding value was viewed. Official
Paradex terms proved USDC settlement, continuous pro-rata accrual, an eight-hour
reference rate, and positive funding as long-pays/short-receives. They also
proved that Paradex funding aggregates Binance and other venues, so the only
possible edge was a timing or smoothing spread rather than an independent
signal.

Four exact public GETs retained one current market inventory and one bounded
history per asset. The inventory exceeded its frozen two-MB ceiling. Each
history exceeded one MB, returned exactly 5,000 rows, and carried a continuation
cursor. The frozen contract prohibited pagination or cadence adaptation, so the
population was incomplete and the run stopped before any funding value, basis,
book, account, credential, order, or fund access. Do not refetch, alias,
paginate, or consume the exact cursors.

`AGENTS.md` now requires uncertain-cadence cursor endpoints to freeze traversal,
maximum pages, total rows and bytes, deduplication, and stop conditions before
the first economic page, with page ceilings derived from documented maximum row
count rather than an arbitrary guess.

Accepted edges remain 35, ranked hypotheses become 59, terminal families become
177, and stable current account-qualified after-all-cost edges remain zero.
Registry, durability-audit, and terminal result SHA-256 values are
`2dc19765f6a9a3a0cb688991828f7624d12f10917f0bba02533281193ff9689c`,
`99c6e83c348f6f3b8b3e95c238db5a66efbf6fcc8af1d3f7e1db6066e7527646`,
and `98c5441b5f9828664c387b9f9b8a05e4eef919a3857054c8bbd73faa4c3ea30a`.

### Latest prospective NYC September 2 long-only basis rejection

Before any price-bearing source was opened, the next deterministic NYC daily
temperature slug and its expected eleven-outcome count were frozen. One exact
public Gamma GET reconciled event 940515. The screen covered the all-YES
complete set, every same-market `YES + NO` straddle, and every optimal `k`-NO
cardinality frontier. Zero packages were strictly subfloor and zero survived
current fees plus one adverse tick per leg.

The best package was the `69°F or below` same-market binary straddle. It cost
`1.001` pUSD per share, lost `0.005` pUSD gross at five shares, and lost
`0.01625` pUSD after stress. The strict source gate therefore stopped before
books. Do not refetch, alias, reprice, request books, or select a sibling.

The preceding retained cardinality audit proved that 2,148,007,910 possible
`k`-NO subsets collapse to 41 exact cheapest frontiers. None passed; its best
metadata and stressed floors were `-0.060` and `-0.150` pUSD at five shares.
That post-price retained role is hypothesis generation only.

Accepted edges remain 35, ranked hypotheses remain 58, terminal families are
176, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values are
`ba86b0e4b333407df56e81d9e1c9af3d7384622c4e625f13ac3267b4ba805881`
and `f61327d608cc3f0d2baacf0f29623348ffdba3eee7c2ad8442bba4a7a15da19f`.
The prospective result SHA-256 is
`68232f3fa8f8ba9a6a747ce1b1c06428bbd1ad7e6b1a190b5120610b97ab9c8b`.
No book, fee endpoint, account, credential, order, fund, transaction, or
protected capture was used.

### Latest retained fixed-NegRisk pairwise-NO rejection

The hash-bound retained September 30 catalog contains three complete fixed-
NegRisk events. Any two distinct outcomes within one event are mutually
exclusive, so `NO(A) + NO(B)` has a one-pUSD floor. A frozen zero-network
screen exhausted all 646 unordered pairs: 481 had complete side-specific
prices, 165 remained price-incomplete, zero cost strictly below the floor, and
zero survived current fees plus one adverse tick per leg. The best complete
pair cost `1.084` pUSD per share and lost `0.475` pUSD at five shares after
stress.

The v1 run stopped before output because one fee-disabled market lacked a fee
schedule. That failure is preserved. The prospectively frozen v2 mechanical
correction treats an absent schedule as zero only when `feesEnabled` is false
and `feeType` is empty; enabled markets still require an exact supported fee
schedule. Do not repeat, refetch, reprice, request books, or adaptively fill the
165 incomplete rows. Reopen only for a distinct complete event or material
price, fee, tick, rule, or architecture change.

Accepted edges remain 35, ranked hypotheses remain 58, terminal families are
174, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values are
`d035ed692935222ec415702231ff2d30c6d8f07dbfb6eea57d801199916ebba9`
and `3144e5eeadbef632dddc427eaf71adef1318e57ca919be68e73e8f580c0d74d0`.
The v2 result SHA-256 is
`951bde07b04cd1257d6be4e4d6e163a3b6a8adc3198bf653eb2504f791d23520`.
No network, credential, account, order, fund, transaction, or protected capture
was used.

### Latest retained Iran-island OR implication rejection

The retained September 30 catalog exposed a new direction-independent Boolean
family: each individual Farsi, Hengam, or Hormuz condition optimistically
implies the four-island Farsi/Hengam/Hormuz/Kharg OR condition. Every
`NO(individual) + YES(composite)` package cost `1.018` pUSD for a one-pUSD
floor. A complete OR replication already cost at least `1.041` pUSD before the
missing nonnegative Kharg leg, so no Kharg metadata or book request was useful.

This exact family is terminal without a network refresh. Do not refetch, alias,
request Kharg metadata, reprice, or request books. Reopen only for a distinct
rule-complete family with a prospectively observed strict side-specific
sub-floor package. Accepted edges remain 35, ranked hypotheses remain 58,
terminal families become 173, and stable current account-qualified after-all-
cost edges remain zero. Registry and durability-audit SHA-256 values are
`0a9cb63f157cd654cab3a3f49cb8910aa2d3f4a64d8fff9dc2368953e23db8c4`
and `610d9a68b1121155660c0f4bb7a9e1b38a2992e0f71d6b0d453a54aaa91f3cc7`.
No network, credential, account, order, fund, transaction, or protected capture
was used.

### Latest Polymarket US participant-program exact-URL deduplication

Rank 58 selected `https://docs.polymarket.us/incentives/user-programs.md`, but
that exact URL had already been prospectively captured under rank 51 earlier
the same day. The retained 4,959-byte hash-bound source says Daily Trading
details will be published once that program is live and says the same for the
Deposit and Trading program. Its overview names only generic deposits, daily
trading requirements, and possible credits; it does not publish deterministic
live eligibility, tasks, rewards, caps, timing, expiry, payment, withdrawal,
reversal, or after-cost economics.

No duplicate request was made. Rank 58 is terminal on the retained current
source with public floor zero and is not accepted, account-qualified, stable,
profitable, or deployment-ready. The mechanism-index novelty inventory had
deduplicated at family level but missed exact selected-URL reuse. `AGENTS.md`
now requires exact URL checks against retained contracts, source results, and
journals before every index-selected fetch; family novelty cannot override a
one-use URL boundary.

Do not refetch or alias this source without an independently known material
change. Never manufacture deposits, trades, volume, days, accounts, referrals,
or eligibility; self-trading, wash trading, spoofing, coordinated activity,
manipulation, and abuse are prohibited. Any onboarding, deposit, trade, order,
fund, or state change requires separate explicit authority. All five families
from the retained Polymarket US mechanism index are now adjudicated; resume only
an existing registry hypothesis whose exact retry trigger is independently
satisfied.

Rank 58 adjudication SHA-256 is
`cba93ed78972affef86459fba0ad070431b803087d404bb7b9fe92eb284e17dd`.
Its reused retained contract/source-result/raw/journal hashes are
`9f237e7b0092d60553364c396d20822d2f7fe805535d451c4114a38c94839c92`,
`8bf6ca023bf62623b5c0334a4866f3df4d5cbe4f9460e333b8c620d46c99caad`,
`3a1987bbff88abf718c43b190db8b40d040ab5ba9c42141920e250e186f9d1ec`,
and `c22e4a529233a4fb2dbf2c80008c86b1073a36b5e9b0191e3eb7a1e42992ac0c`.
Registry and durability-audit hashes become
`c038dde89dc80e2712d8adbbcd90056b27b625be062467f34b592e233e5e7eb8`
and `bcf28988454195e319db22ba3f31ef1ad1e13d157d332ff84b2c7a7ad4981556`.
Accepted edges remain 35, ranked hypotheses remain 58, terminal families become
172, and stable current account-qualified after-all-cost edges remain zero. No
network, account, credential, signed request, deposit, trade, order, fund,
transaction, or protected capture was touched by the rank 58 adjudication.

### Prior Polymarket US combo RFQ source gate

One prospectively frozen exact official Polymarket US RFQ overview retained
4,071 bytes. Retail RFQ access is authenticated beta for explicitly enabled
users. An RFQ requests two-sided liquidity for an exact combo symbol; a taker
can accept one quote side and the maker can confirm during last look. The
private stream is best effort with no replay or separate subscription
acknowledgement, so reconnect requires REST reconciliation.

Crucially, `QUOTE_STATUS_EXECUTED` and `quoteExecuted` mean only that paired
exchange orders were submitted and their order IDs recorded—not that they
filled. RFQ orders enter the normal combo book, may trade with other resting
liquidity, and can leave unfilled quantity resting when either side enables
`restRemainder`. The public source therefore proves no atomic fill, guaranteed
price improvement, authentic quote, fee, last-look rejection rate, or positive
after-cost value. Rank 57 is terminal on public mechanics and is not accepted,
account-qualified, stable, profitable, or deployment-ready.

Do not refetch RFQ, combo, endpoint, schema, streaming, institutional, trader-
guide, HTML, FAQ, or search aliases. Do not authenticate or request a quote.
Every authenticated read, RFQ or quote creation, acceptance, confirmation,
cancellation, order, fill, transfer, fund, or state change requires separate
explicit authority. Never manufacture RFQs, quotes, orders, fills, volume, or
counterparties; self-trading, wash trading, spoofing, reference-price
manipulation, information abuse, and manipulative cancellation are prohibited.
Rank 58 participant Daily Trading / Deposit and Trading programs was the next
source-qualified hypothesis at that checkpoint and is adjudicated above through
exact-URL retained-source reuse.

Combo RFQ contract/source-result/raw/journal/adjudication hashes are
`dca10fd4fb7ff27d8c9e88670cbc57e79d5a12e683f4ea18d7dbc22f5415fd96`,
`210cb590e3bbdd50ab653a80f5265d461237d7f5a9e162afa2165e320472d43e`,
`ab97a5110e8f5ee7aae0996cb6931fdea52a813c78512a7a3a01106bb4414c4b`,
`bee6c4768ac645746def16ae8be8d1cd0ff0ff3a41a72991012079304af289c7`,
and `d1a4075cde9f3f3bd830067a8b70ae1f1c3a19a7de97e8788f16d6536c42bc20`.
Registry and durability-audit hashes become
`7a6dd4d5cb7aed3b1dc15062ce017387b873ea64da9c7f3a5feb62c894c3d077`
and `e6b22e14da4128fa3e1b37c5214776360dcb41651e77f6cef47348abe5416e02`.
Accepted edges remain 35, ranked hypotheses remain 58, terminal families become
171, and stable current account-qualified after-all-cost edges remain zero. No
account, credential, signed request, authenticated access, RFQ, quote, order,
fund, transaction, or protected capture was touched.

### Prior Polymarket US market-maker source gate

One prospectively frozen exact official Polymarket US source retained 488 bytes.
It says the program rewards approved market makers for providing liquidity
across a wide range of contracts in given categories and gives an institutional
contact route to learn more or apply. It publishes no deterministic approval,
eligibility, category, quote uptime, size, spread, depth, performance, reward
formula, payment asset or cadence, cap, discretion, withholding, clawback,
suspension, termination, renewal, or change economics.

The public after-all-cost floor is therefore zero. Rank 56 is not accepted,
account-qualified, stable, profitable, or deployment-ready. An exact-positive
future owned payment predicate would be tautological and cannot replace
predictive terms. Do not refetch or alias the source, contact or apply, or
manufacture quotes, orders, fills, liquidity, volume, accounts, or eligibility.
Self-trading, wash trading, spoofing, manipulative cancellation, and non-
executable quoting are prohibited. Every contact, application, negotiation,
onboarding, configuration, quote, order, cancellation, fill, fund, or state
change requires separate explicit authority. Rank 57 combo RFQ was the next
exact unconsumed source-qualified hypothesis at that checkpoint and is
adjudicated above.

Market-maker contract/source-result/raw/journal/adjudication hashes are
`d3fb2c4ac6975acc612760624fc126ae382ff232e8f8e2e1f8e67577f37038bb`,
`4acef72e1c83f2861df3a099e189355b30d6e7f15a5d54728a120f6299aa39b1`,
`d4b568df596d93c7e7d1b7da7d74602349449b59535644abafe2909dc7245d42`,
`07d538a24afc46dd5d1d4adac8d34a6a3890dc1fed5db392536154f3caaac20e`,
and `af80c16faa5666735cc74f17d5212ec95d226bc0ed58e8d247811eab06ded381`.
Registry and durability-audit hashes become
`afdc6b4f6d82f4011ce723076a9cff22e1cf21c45a4995a28ca9bfca83a86b4e`
and `b6e2d132f31b0879214ef2e81658bd48fa781bfaaea89536a8c1eb2c13c98f18`.
Accepted edges remain 35, ranked hypotheses remain 58, terminal families become
170, and stable current account-qualified after-all-cost edges remain zero. No
account, credential, signed request, contact, application, quote, order, fund,
transaction, or protected capture was touched.

### Prior Polymarket US vendor-fee adjudication

One frozen exact current official Polymarket US partner source defines a fixed
USD vendor fee computed by the partner under its participant agreement. The fee
is declared against an order, accrues as an unsecured receivable, requires
partner shadow-balance and report reconciliation, and can be collected at most
once per participant account per day through one net `VENDOR_FEES` transfer.
No fee cash moves at order time; the declaration is not automatically adjusted
for partial fill or cancellation; Polymarket does not reserve or underwrite the
receivable; participant trading losses can cause insufficient-funds rejection;
no API exposes accrued balance; the daily report delivery method is TBD; and
the capability is beta and subject to change.

The thirty-fifth accepted scope is only exact positive owned USD vendor-fee
cash actually collected on bona fide independently existing external
participant orders under an already approved integration and disclosed,
consented agreement, after uncollectible receivables, waivers, refunds,
reversals, acquisition, support, compliance, tax, opportunity, and every
incremental cost. It is market-direction-independent external-user revenue,
not autonomous trading profit. Public, account-qualified, stable-current, and
deployment-ready profit floors remain zero. Do not refetch the consumed source
or treat a declaration, accrual, report row, or receivable as cash. Never create
users, orders, volume, fees, or activity, and never charge undisclosed or
unconsented fees. Contact, application, onboarding, configuration, order, fee
declaration, transfer, collection, fund, or state changes require separate
explicit authority. Rank 56 approved market-maker contracts was the next exact
unconsumed source-qualified hypothesis at that checkpoint and is adjudicated
above.

Vendor-fee contract/source-result/raw/journal/adjudication hashes are
`10f643129214fe6d104ba04b119884e1880a0478a61df20c1c062907b61a82b7`,
`6fc0c7918d461f21fb352fdf2af95e7ba6eaa77ddaa0f3d1ac0760507559fb71`,
`5ee5063318cf5c5ad4991fa1c0a52430a03d91df2981021e5bcc96f83c18eae8`,
`6e774dcc56c1a658e021656bd133988f02f9eacf42fccb31744cb464898fb3d3`,
and `92b881d5b426b6e271550dbd3f3fa6b73b1c8164d326d2885528dc4524f57b25`.
Registry and durability-audit hashes become
`d2f93c88d50905be28cefe0f215e72137c8b9bbb7f9d73741b87925691413b44`
and `8fb854400727c2b40666047a8df5ac42b183082db8c8ce2f3c77ce16a95828af`.
Accepted edges become 35, ranked hypotheses remain 58, terminal families become
169, and stable current account-qualified after-all-cost edges remain zero.
No account, credential, signed request, order, fund, transaction, or protected
capture was touched.

### Prior structural source inventory

One frozen exact Polymarket US `llms.txt` GET retained a complete 350-line
index with 342 link rows and 300 unique absolute HTTP URLs. Offline novelty
deduplication identified five previously unrepresented economic families:
approved-affiliate referral rewards, partner vendor fees, approved market-maker
contracts, combo RFQ execution, and participant daily/deposit-trading programs.
The precommitted one-linked-source cap advanced only Referral Incentive and
deferred the other four without opening their pages at that checkpoint.

The exact referral page says approved affiliates receive rewards for referring
new Participants, but publishes no approval, eligibility, attribution,
qualifying activity, payout, cap, duration, payment, anti-abuse, reversal, or
change economics. It is rank 54 with public floor zero, not an accepted edge.
Do not inflate the edge count with a tautological exact-positive-payment
predicate, refetch the page, or open the Refer-A-Friend FAQ alias. Rank 55
partner vendor fees was subsequently consumed and adjudicated above. Every
later family must receive its own fresh prospective source contract; no
contact, application, referral, account, KYC, deposit, quote, order, fund, or
state change is authorized.

At that checkpoint accepted edges remained 34, ranked hypotheses became 58,
terminal families became 168, and stable current account-qualified after-all-
cost edges remained zero.
Index contract/source-result/adjudication hashes are
`7625c1acceffc154492429b20b5218ab32f5f9e77f2e180e67377c8f6eb4866f`,
`43eb515fb75c5a1891c78c998e261d8f8f2c60707c5034db0d7aec9e70314c52`,
and `214f9807ed7d9120478be6e75daf889ee8aeb30f8bea194c3af8d5eb4e8c251f`.
Referral contract/source-result/adjudication hashes are
`cb8dca1749f6c2d58e60d9a72d993072a45be22308c7b13771481c25b98d3d11`,
`62494e77c882afe747e87056d64661f2abc1f80e58bcee4a1725cc4f8e43d935`,
and `87cce3f6175f12b2b5604db57d0f35fcfec1868577d0e0d92a99a82e189bf6d6`.
Registry and durability-audit hashes become
`8041cee138044411dc4c8357c7dc8e0e8b8a4cc5b4493f3ddc475db4aac1303f`
and `454c2c0e2f3f3eea07a58c6d8ec92be3dd85aac39d48b26d836b415c3d29c3bb`.

### Prior structural novelty gate

Binance Web3 Wallet Prediction Trading OTC block trades are a genuinely
distinct direction-independent price-concession candidate, not an alias of
Binance Exchange Spot block matching or Polymarket's public CLOB. The exact
official developer source was frozen before access, but its one-use GET returned
HTTP 202 with an empty body. Search-rendered endpoint and `secretToken` examples
are discovery-only and cannot prove public access, quote authenticity, minimum
size, fees, settlement, fillability, or positive after-cost value.

Do not retry a locale or rendered alias, use cached snippets, guess a private
token, or request quotes. Reopen rank 53 only after a material byte-retainable
official documentation/API change, or with explicit separate read-only
authority, an independently required legitimate large prediction trade, and an
exact same-payoff public-book counterfactual. Quote acceptance, cancellation,
negotiation, orders, funds, wallets, and account mutations always require
separate explicit authority. Accepted edges remain 34, ranked hypotheses become
53, terminal families become 166, and stable current account-qualified
after-all-cost edges remain zero. Contract, failure-adjudication, registry, and
durability-audit hashes are
`2441d38840ceb9a442a649b5631c2ef0c8ef73c58afdc61f7b015cbc91f92b50`,
`02a2085182a0ff292aaf5b7f4544a3d4d3c5a813bab9e7adca9638fac8500525`,
`9b2236283e5ec3c877be2fa0cabb423699d3708bd2cb1a6d656e26dace4aaefb`,
and `dfb85ebfcc78b7d05da92036860701dbdae54953c04911a1fc6e47cad3dbd762`.
No account, credential, signed request, quote, order, fund,
wallet, transaction, or protected capture was touched.

### Latest efficient structural screen

The distinct Binance-Bybit BTC, ETH, and SOL USDT perpetual funding population
is terminal. A clean preregistration froze exact 90-bucket alignment,
training-only orientation, chronological 45/22/23 roles, 20-bip round-trip
execution, a 10% annual two-leg capital hurdle, and 25-bip cross-venue stress
before any funding value was viewed. Six bounded public GETs proved the exact
live linear USDT-settled instruments and retained 90 eight-hour funding rows per
asset. All three selected long Bybit and short Binance from training; zero
survived every role. BTC gross was positive but economically dominated in every
role, ETH and SOL validation gross were negative, and every test net was below
-62 bips.

Do not paginate, resample, reverse after observation, alter roles, weaken
costs, or request basis or books. Reopen only on a material Bybit or Binance
funding, fee, quote-unit, custody, transfer, latency, capital, or execution-
architecture change. Accepted edges remain 30, ranked hypotheses remain 47,
terminal families are 145, and stable current account-qualified after-all-cost
edges remain zero. Preregistration, adjudication-contract, result, registry,
and durability-audit hashes are
`6cbf711127949566a3ad8e7dc50f3bd38700e96987ca06e6d1a3f75abee77480`,
`00cfb669c0a0300ba1b1de919e03910d0e1aa1c33c1955b696625973a2da0963`,
`eb8d9badab2dd2d4869b2d52154a49815a2bb9b7034098bdf475163ac8b9303f`,
`a3e770993bdc1705703a38ccbaf01c14ec90ceed4c948b04032a275ee43e5eef`,
and `b3050b131b33588f5722302e1d4926bf56e6ae6589a44b5c6e91bd79133df005`.
No account, credential, signed request, order, fund, transaction, basis, book,
or protected capture was touched.

### Prior efficient structural screen

The exact largest, second-largest, and third-largest company events are
terminal at the source gate. Three prospectively frozen one-use public Gamma
GETs retained 25, 29, and 29 markets. Their market-cap metric, September 30
market-close instant, and reporting source align, but the independently
resolved NegRisk rules do not specify cross-rank tie or corporate-action
handling. Hidden Company-letter populations differ by rank and have no
source-proved identities, while any-other-company can denote different
companies at different ranks. An all-different matching floor is therefore not
source-proved.

No price field was used for the decision and no CLOB book or fee request was
made. Do not retry, alias-map placeholders, or assume ordinal tie semantics.
Reopen rank 31 only if exact rules add deterministic cross-rank tie and identity
semantics or a distinct family proves them before economics. Accepted edges
remain 29, ranked hypotheses remain 47, terminal families are 141, and stable
current account-qualified after-all-cost edges remain zero. The three contract
hashes are
`c60773469228f8b94f43e62fae1b8fcacf87c69f2a97cb2ed075656bd2a1555d`,
`77806e15cd1326385ecdf3860c760a11f24df8612b77d570dab30e6b86fdc862`,
and `c89545cb8fe40944bff0f44b978573f9fec215cdaa9758847a8ca5c8dc9341f8`;
capture hashes are
`e338dfd907f90c758aee7c04aa5cabd4ef7391eff34bb303d6289c22c370869c`,
`59cc6a074cbc29088490d7208481a90fa4750861e616a824b9f8790f2457c97b`,
and `c945347331b14bbb5ace8c68660fa3169e99b4e62fae78a0347ea74fd726f077`;
the adjudication, registry, and durability-audit hashes are
`47d7c9f000b9e8445d227a4b9b35ebeaf0681e8b6bfbdaaf2ebcb359e054839e`,
`9c793173647d9a3f50e0be1f3696a16614517fb6d921dc9efb9f2151532b2939`,
and `0d25d77c6d84cad2bb356d15469cea7f6d0c3032e38ab18e0040388ed29466c4`.
No account, credential, signed request, order, fund, transaction, or protected
capture was touched.

### Earlier efficient structural screen

The exact Mythos release-date and cumulative-deadline population is terminal.
Two prospectively frozen one-use public Gamma GETs retained 38 and 10 markets;
a separately frozen zero-network graph exhausted 208 valid implication,
mutual-exclusion, and no-release-exclusion packages across 30 future exact dates
and eight deadlines. Zero conservative side-specific packages cost strictly
below their one-pUSD floor, and zero survived current taker fees plus one
adverse tick per leg. The best metadata package cost 1.17 pUSD per share and
lost 0.99878 pUSD at the five-share common minimum after fee-and-tick stress.
No CLOB book or fee-endpoint request was justified.

Do not retry, refresh, alias, reprice, or book-capture this exact two-event
population. Reopen rank 31 only for a distinct unconsumed source-proved release
family or a material rule, fee, tick, or market-architecture change. Accepted
edges remain 29, ranked hypotheses remain 47, terminal families are 140, and
stable current account-qualified after-all-cost edges remain zero. The metadata
contract hashes are
`bcaf6ca43afab29fa4d8dcd614e36522e93bfbf9ec09761a19e7efbe63052765`
and `482f3afab76c71f60ce049cfb310d64decfcadea7294d5f9f0218860d32cca3e`;
capture hashes are
`db3f62ddb586c898b3a8b9e2258cb6db005c3c7c66253093afe4b73738a3e13c`
and `2b51a74a707a29029ba7d1392adc56115f4caf08b1a5b4406008fb9dbfca6ab8`;
the graph contract, adjudication, registry, and durability-audit hashes are
`fa9dc2f724d03cf869865ed1bd0f40b46c2274c11e1c121d0290757c2446f65b`,
`f515939e699129b50b7570fc7157cc9d9f70951a3b8a3127555152603ddf782e`,
`98a39ba8c2a565104cd49df26b41e711fc770c57433a26fdb6d90ba20d6f6300`,
and `fa699d282419535be266c71a5e0ff465cde3ecddbf7fbb4ce15a0c63f30c86ff`.
No account, credential, signed request, order, fund, transaction, or protected
capture was touched.

### Earlier structural screen

The Gemini Pro October-31 deadline to November-1 interval pair is terminal at
the exact metadata gate. Two prospectively frozen one-use public Gamma GETs
reconciled 11 interval and 21 deadline markets. The substantive public-release
rules align and closed No siblings exclude a qualifying release during the
selected markets' creation gap. However, the conservative `NO + NO`
acquisition proxies are 0.67 and 0.33 pUSD, exactly 1.00 pUSD against a
one-pUSD floor. Zero gross headroom remains before fees, ticks, depth, time
value, non-atomicity, latency, unwind, capital, or operating cost, so no CLOB
book or fee endpoint was requested.

Do not retry, refresh, alias, reprice, or book-capture this exact pair. Reopen
rank 31 only for a distinct unconsumed same-target sibling whose conservative
side-specific package is strictly sub-floor, or a material rule, fee, tick, or
architecture change. Accepted edges remain 29, ranked hypotheses remain 47,
terminal families are 139, and stable current account-qualified after-all-cost
edges remain zero. The interval contract/capture hashes are
`56fdf5aa9445190d1e2683f68a1f77fa9b5590e9535f05c31e7457bd5aeb68b9`
and `748cdb55075ff118848ea71608622571e89912763a33a271eef0674a84ad4309`;
the deadline contract/capture hashes are
`d9c3633af7c754ba0aef5fdcc319eb3cf03b8a083ecb19ff2c6056007983cb8e`
and `7748a2c988adf2cc4efebed0686ebc8bd49ad6f1f23166b2096de99ffbf14df5`;
the adjudication, registry, and durability-audit hashes are
`a1e7562a56408bf3634f60585a6b64be89e401b7f1a25875762eed42c6908ac6`,
`097e92b82ef74899f08c972747d268fd4554beb92f1fdbcb23c5316767a19703`,
and `62ebb5280c1d3ded0a0298c97ac1cad0189d733be57918e9bf3a1da4bbb4884f`.
No account, credential, signed request, order, fund, transaction, or protected
capture was touched.

Contract hashing must use the repository's Python sorted-key compact-ASCII
canonicalizer, never PowerShell `ConvertTo-Json`. This correction was made
before validation, output creation, or network access and changed no frozen
question, population, gate, or request boundary.

### Earlier structural screens

Two new literal triggers were closed at the cheapest decisive gate.

First, rank 38's native-stock parity branch cannot use the newly announced
`HK0625USDT` SHEIN quanto instrument. Current official Binance Stocks
documentation limits the native product to US-listed NYSE or NASDAQ common
shares and US-listed ETFs, so the exact HK0625 Hong Kong share has no documented
native counterpart. A frozen documentation GET returned HTTP 202 with an empty
body and is preserved without alias or retry; the separately rendered official
page is rejection-only. Never substitute an ADR, wrapper, tokenized stock,
bStock, Ondo, xStock, company-name match, or different share class. Zero native
streams, futures, FX, funding, fee, book, account, or credential requests were
justified.

Second, the September private-company pairwise-to-five-way Polymarket
projection has a tie-semantics hole. Pairwise equal growth pays 50-50, while the
five-way market awards one full winner by highest final valuation. The naive
`NO(Anthropic wins five-way) + YES(Anthropic beats OpenAI)` floor is therefore
0.50 pUSD, not one pUSD. Its required five-way `NO` leg alone displayed 0.97
pUSD, so the gross upper bound is at most -0.47 pUSD before the second leg or
any cost. No Gamma, CLOB, or fee request was justified. `AGENTS.md` now requires
explicit tie-payout alignment for every pairwise-to-multiway projection.

Accepted edges remain 29, ranked hypotheses remain 47, terminal families are
138, and stable current account-qualified after-all-cost edges remain zero.
The Binance contract, failure, and adjudication hashes are
`3fd064e94d8e591d874a816c495c34b878fb0f6979662a7dcca03625ab43c9d0`,
`4baf2f1b3a5bce69cc23575234c4ee27405136c222f8d60904adc0ba4e2bec73`,
and `6ff02e16bc0c8e8f6e833da8a221eee532868b103b6ef078ca16b72495ecf139`.
The Polymarket adjudication hash is
`bcb49e2e1bdba407fe8121a37ff56dcddd770b1957e1add2c95f225fdb76fded`.
Registry and durability-audit hashes are
`06e9c43228f09c23d6ad26dde21216eed7d350fbfe19453380a7bf10247565b9`
and `a6c3d63d971bf16aacfec64492dc13660c203d93e740c4fade4122f2fc50a5d8`.

No signed request, account, credential, order, fund, transaction, or protected
capture was touched.

Binance's new September 1 announcement for an `HK0625USDT` SHEIN quanto TradFi
perpetual supplied a literal instrument-change trigger for rank 43. The
cheapest decisive gate was one frozen current Polymarket Perps instrument
inventory, not funding or books. The HTTP 200 response retained 44,614 bytes
and contained the same 67-symbol membership as the August 29 baseline: zero
additions, zero removals, and zero `SHEIN` or `HK0625` symbol or base-asset
matches. No new cross-venue funding population exists, and Binance deployment
was not inferred from the announcement alone.

The related Polymarket SHEIN IPO market-cap threshold page was screened only at
the rendered discovery layer. Its two unambiguous adjacent guaranteed packages
cost 1.81 and 1.548 pUSD for a one-pUSD floor; blank higher-threshold `No`
buttons were unavailable, never free. No strict sub-floor package or exact
bracket sibling was source-proved, so zero Gamma or CLOB requests followed.

Do not refresh the exact 67-instrument population, substitute an IPO prediction
market as a perpetual hedge, or request SHEIN funding, prices, fees, or books.
Reopen rank 43 only after a later exact common underlying exists or another
literal material funding, fee, session, unit, conversion, or execution trigger
is satisfied. Contract, capture-result, and adjudication SHA-256 values are
`74952fcaa3bd83f826c9113051776aecedc46bd7d4add3f522ff517432e672de`,
`360159707942d24e08cca729c390317aac62b3cdbe8ccbfd2fc833155d78f46a`,
and `bce940e94a6846e2524f01d77f18b7e921bc273b1f027d9dc7cab4a5b99faf50`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
136, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`e077098bcd22df64153ecbd2bc859980acdeaa2eabe4b13925728ea735ab248c`
and `40491937fb0a4ffd930a190f2ec4ca6c38d25d4b843bd308ad94c1d48ec62169`.

No funding value, ticker, price, book, fee endpoint, credential, account, order,
fund, transaction, or protected capture was touched.

The distinct `Next Senate Majority Leader?` event initially exposed a stronger
complete-set lead: ten rendered open candidate YES asks summed to 0.950 pUSD,
while the FAQ said eleven outcomes and the rules explicitly named `Other`.
One frozen public unauthenticated Gamma GET was therefore justified before
depth. Exact metadata returned 64 NegRisk markets, not eleven. Only ten were
open, active, and accepting orders; Lindsey Graham was active but closed; and
53 markets were inactive. The inactive set included 49 identity-free `Person X`
placeholders, Dick Durbin, John Cornyn, Rick Scott, and `Other`, each at a
one-pUSD YES ask.

Every returned YES ask sums to 53.931 pUSD per share for a one-pUSD payout, or
a -264.655 pUSD gross floor at five shares. The open-active-only diagnostic
retains 0.09003 pUSD after current fees and one adverse tick per leg, but it is
not a complete package. The placeholders also cannot be party-mapped into a
safe projection against the Senate-control marginal. The exact event is
terminal before books: do not refresh, alias, omit, classify, or party-map
hidden siblings.

No credential, account, fee endpoint, order, fund, transaction, book, or
protected capture was touched. Accepted edges remain 29, ranked hypotheses
remain 47, terminal families become 135, and stable current account-qualified
after-all-cost edges remain zero. Contract, capture-result, and adjudication
SHA-256 values are
`5f69284d64c20444343221b592558dfeb731c95217b12760f8ff1e4c11f76491`,
`ff5ddefed63331563e69fd528755dfef2176f0a693268211a42b1dbf699811e0`,
and `bf4605a90bd9a9389615cef7618b6ce04fbe379ca121bad9db7dbdc65a956772`.
Registry and durability-audit SHA-256 values are
`002bbb75244f3333b2e70464b8835922fa7787fd5ccc01a857ed8becb459ee1a`
and `773581acc22ef94cf350c3d423c4b601514f17701f4ce57a3cdb623e8bc5af7e`.

The contract preflight initially rejected a manually anticipated
`frozen_at_utc` that was nine minutes in the future. This happened before any
output path or network request existed. The timestamp was replaced with the
already observed host UTC instant, the canonical hash was recomputed, and the
resolved-path offline validator passed before the one permitted request. The
repository instructions now require that exact sequence.

The rendered OR-06 House Election Winner card exposed only Democratic and
Republican outcomes whose visible prices appeared to sum to 0.962 pUSD. A
frozen one-use public Gamma GET was therefore justified before any depth
request. Exact metadata instead returned eight NegRisk markets: the two active
party outcomes plus six hidden inactive but accepting-order siblings named
`Other`, `A`, `B`, `C`, `D`, and `E`. The exact active-only ask sum was 0.971
pUSD, but it is not a complete payoff package because none of the hidden
siblings is source-proved impossible or safely omittable.

Buying every returned YES side costs 6.971 pUSD per share for a one-pUSD event
payout. The apparent structural edge is therefore terminal before books. Do
not refresh, alias, omit hidden siblings, or request OR-06 depth. Reopen only
for a distinct exact event whose complete market set is rule-exhaustive and
remains positive after current fees and at least one adverse tick per leg
before depth. No credential, account, fee endpoint, order, fund, transaction,
book, or protected capture was touched.

Accepted edges remain 29, ranked hypotheses remain 47, terminal families
become 134, and stable current account-qualified after-all-cost edges remain
zero. Contract, capture-result, and adjudication SHA-256 values are
`35dfa8a68ec3856e6f8ff32bed471ff5378c19a9997bea9b5345e4af061e592b`,
`c60b91d086e1246e87eb39b5845fa9fd3ca60864102351477d583f7c5b96d804`,
and `b8ab67f254bf6278e6acb993c8a73412e3a4465fe3c7d99dc34b4c603d0455e5`.
Registry and durability-audit SHA-256 values are
`bafdead76f0ee1aca2c414674764a7ebca7510fd74478f797fa41735007e80dc`
and `7bfabbc199967018ce0cfde0787850d48f52960b146064e846455899b754dbf4`.

A distinct joint-projection cover advanced past rendered discovery and exact
fees: buy `NO` on standalone Republican Senate seat outcomes 50, 51, and 52,
then buy `YES` on all four joint-event rows whose Senate projection is 50-52.
For a Senate result outside 50-52 all three NO legs pay; inside 50-52 two NO
legs plus exactly one House row pay. The package therefore has a three-pUSD
floor without a market-direction forecast.

Two frozen one-use public Gamma GETs reconciled all 13 joint siblings and all
11 standalone Senate siblings. Every market was active, open, and accepting
orders; the four required House ranges were complete; and the Senate election,
cutoff, runoff, vacancy, attribution, independent-caucus, fallback, and source
rules aligned. Conservative exact prices cost 2.961 pUSD per share. At five
shares the current 0.04 exponent-1 taker fees left a positive 0.09562 pUSD,
but one exact adverse tick per leg reduced the after-fee floor to -0.02675
pUSD before depth, time value, seven-leg non-atomicity, unwind, latency,
capital, or operational cost. The exact population is terminal before books.

Do not refresh, reprice, alias, omit a House row, or request the fourteen-token
book batch. Reopen only for a distinct rule-complete joint projection whose
conservative package remains positive after current fees and at least one
adverse tick per leg before depth, or a material fee/tick/rule/architecture
change. No credential, account, order, fund, transaction, fee endpoint, book,
or protected capture was touched. Accepted edges remain 29, ranked hypotheses
remain 47, and terminal families become 133. Joint/marginal contracts and
adjudication SHA-256 values are
`0cb91fb5c8b0ec763c581b91817e01b0cea15209163f73cc497a7a8bc72b7834`,
`38faaa6e90618a75fe68edd99444e37e9644c93ed38cf45c77af935979e65bf0`,
and `2fdcd4595af5b5a7e50a0e97fa0559694bcaddd03f844c201894441afc46adbd`.
Registry and durability-audit SHA-256 values are
`93ea1175161da3c536af13d0ac86cc0c08ec1e729cd56b72eafa32c4654c4125`
and `bd2f8e5bb16a9b9dc64d2f79f626b051147fc20679f72c06d89563909dca8c1e`.

A distinct joint-to-marginal Boolean state cover passed rendered discovery at
0.983 pUSD: `NO(Democratic House) + YES(DD) + YES(RD) + YES(joint Other)`.
Two frozen one-use public Gamma GETs proved a complete active five-state joint
partition and a rule-compatible House marginal. The marginal event contains
nine markets, including seven hidden inactive siblings; none was treated as a
free leg, and joint `Other` remained mandatory to cover every unmatched state.

Conservative exact metadata prices summed to 0.993 pUSD, only 0.035 pUSD gross
headroom at five shares. The current 0.04 exponent-1 taker-only schedules charge
0.11778 pUSD, producing a -0.08278 pUSD after-fee floor before depth or every
other cost. The exact population is terminal before books. Do not refresh,
reprice, alias, omit `Other`, or request depth. Reopen only for a distinct
rule-complete joint-to-marginal package that remains strictly positive after
current fees before depth, or a material fee/rule/architecture change.

No credential, account, order, fund, transaction, or protected capture was
touched. Accepted edges remain 29, ranked hypotheses remain 47, and terminal
families become 132. Joint/marginal contracts and adjudication SHA-256 values
are `8201e7535d07b7340a612f10473a43718ed1efa350b6d804dd5762a57cab64b7`,
`1ce1e4450da6cc36e7b2c64702bad00ce52fc0b33b9937353ae8ecb16bdbf634`,
and `738a9d89f9334e2cfc91f4c21b474528d5222cab222455856cb8289c055ac3ff`.
Registry and durability-audit SHA-256 values are
`264076c9ab6c42043689bf0d53043c2520236c493859cfd1f159acaea78e0cac`
and `a0c062913d17787d027b2973c947a5d8fb4d2be81d018ee90f9ca354034d6285`.

A bounded rendered-only pass screened eight distinct, exhaustive September
central-bank decision partitions. The direct known `Buy Yes` complete-set sums
were Bank of Canada 1.005 pUSD, Bank of Japan 1.024, Reserve Bank of Australia
1.006, Bank of Russia 1.018, Fed 1.016, and ECB 1.009. Bank of England already
exceeded 1.024 before one missing displayed side; Reserve Bank of New Zealand
already exceeded 1.003 before its missing displayed `Decrease` ask. Every page
therefore rejected at the discovery layer before Gamma, CLOB, fees, accounts,
credentials, orders, funds, transactions, or protected captures. Missing or
zero-rendered sides are unavailable, never free.

Do not refetch these exact September decision pages merely to chase mill
changes. Reopen rank 31 only for a distinct rule-complete partition with every
required direct acquisition side displayed strictly below its payoff floor,
then freeze exact metadata before depth. This rendered screen changes no
accepted, ranked, or terminal-family count and does not require registry or
publication-hash regeneration.

A distinct Polymarket complete-set candidate initially cleared the rendered
discovery gate: eight visible streaming-service YES asks totaled 0.99 pUSD for
a nominal one-pUSD floor. Because the rules named an unlisted `Other` fallback,
one prospectively frozen exact public Gamma GET reconciled the event before any
book access. The immutable 44,700-byte response contains nine markets. The
hidden ninth `Other` market is inactive and has direct YES `bestAsk=1`, making
the complete rejection sum 1.99 pUSD before every fee and execution cost.

This exact event is terminal. Do not retry, refresh, alias, omit `Other`, or
request books. No credential, account, order, fund, transaction, or protected
capture was touched. Accepted edges remain 29, ranked hypotheses remain 47,
and terminal families become 131. Contract and adjudication SHA-256 values are
`90525ae26b9d336473ad3e676bc8a229fc90bbabb0b8d7ceeb6230e15c1e5f30`
and `9ffdd9272bb4b18644aad2c42e713a270689c2fed2145cd91079ed6b7ba82c05`.
Registry and durability-audit SHA-256 values are
`89cd6f271e75fb716682976870b5d74d4fd06b8a27c8933f88cb0019c993a533`
and `7232a04530aff3090453f32f6f7819fa29e102975b6134775a6219a3a64beeba`.

A distinct, cleanly preregistered Binance-OKX direction-neutral funding family
was rejected with four public unauthenticated requests and zero basis or book
access. One complete OKX swap inventory fixed exactly one live, linear,
USDT-settled BTC, ETH, and SOL perpetual. Three one-use public funding responses
contained 296 rows each and aligned the exact 90 retained Binance eight-hour
buckets, avoiding any Binance refresh.

Training alone selected long OKX / short Binance for every asset, fixed through
22-row validation and 23-row test after a 45-row training role. Zero assets
survived the frozen 20-bip round trip, 10% annual capital hurdle, and 25-bip
custody/transfer/latency/failure stress. BTC gross train/validation/test carry
was 3.66193/-1.34312/0.73452 bips; ETH was 5.85134/-2.04475/-0.80671;
SOL was 2.97859/-2.46280/1.08712. Validation was negative for all three, ETH
test was negative, and every role was dominated by the frozen hurdles.

This exact 90-bin population is terminal. Do not paginate, resample, change
alignment, refit orientation, weaken costs, or request basis/books. Reopen only
on a material OKX or Binance funding, fee, quote-unit, custody, transfer,
latency, capital, or execution-architecture change. No account, credential,
order, fund, or protected capture was touched. Preregistration and adjudication
result SHA-256 values are
`68de39a4ff7db0b01a204f410b637f513fcdb5058705b8a34949991acc8a585b`
and `3bd57553ea5f40ac1141a48ee3095b02e3c6e6c41012de93d96d95ebe13b006e`.
Accepted edges remain 29, ranked hypotheses remain 47, and terminal families
become 130. Registry SHA-256 is
`eb915168778e0824dc80bb405c2e1c5657916c0750e6f12bc50ea687024b88a4`;
durability-audit SHA-256 is
`159f9940502865da19cdcc6df40f32c385497b723e31988725da5951286d061f`.

A distinct direction-neutral Binance-Lighter funding family was rejected with
only four public unauthenticated requests and zero basis or book access. One
complete Lighter perpetual inventory fixed active BTC, ETH, and SOL market IDs;
three one-use funding requests returned 718 hourly rows each. Reusing retained
Binance histories avoided three redundant source calls. Lighter's exclusive
time boundaries mechanically left 88 complete eight-hour aligned buckets.

Training alone selected each asset's orientation, which stayed fixed for
validation and test. Zero assets survived every chronological role after the
frozen 20-bip round trip, 10% annual capital hurdle, 25-bip USDC/USDT stress,
and 25-bip custody/bridge/latency stress. BTC gross spread was 3.1217, 2.5119,
and -3.4251 bips; ETH was 10.0548, 0.4170, and 3.1349; SOL was 4.4760, 3.8258,
and 15.1961. Those totals are far below capital cost, while BTC reversed in
test, ETH failed validation persistence, and SOL failed training persistence.

This exact population is terminal. Do not resample, change alignment, refit
orientation, weaken hurdles, or request basis/books. Reopen only on a material
Lighter or Binance funding, fee, quote-unit, custody, bridge, latency, capital,
or execution-architecture change. A bounded schema inspection accidentally
printed first/last economic rows before the offline contract was finalized; the
contract therefore made the population promotion-ineligible even though the
terminal rejection is invariant. `AGENTS.md` now forbids value-bearing schema
prints before the full funding adjudication freeze. No account, credential,
order, fund, or protected capture was touched. Accepted edges remain 29, ranked
hypotheses remain 47, and terminal families become 129. Registry SHA-256 is
`333a9118e9339a7b1e0d3ce1ddbfc5f382c1d77f006cb81254b160bcd0f221a8`;
durability-audit SHA-256 is
`fe8c01b5389fdef3b4a2acd6bc87fec85ac1da65f3111ce018b6ba0da29bb3a9`.

A distinct current Rewards-page lead advanced rank 17 without sampling its
siblings: Anduril valuation at or above 122.5B by September 30. One frozen exact
Gamma plus sponsored-condition reconciliation proved an active binary market,
zero maker fee, 20-share reward minimum, 4.5-cent scoring band, 50 pUSD/day,
and 30.7062 remaining reward days. Its exact one-use two-token book showed a
0.97 combined best-bid join, 0.60 pUSD both-fill gross, and 10 pUSD maximum
one-leg settlement loss for 20 shares. The configured remaining pool was
1,535.26 pUSD under an intentionally impossible 100% owned-share bound.

This is a serious paired-maker lead but not an accepted or profitable edge.
The official book snapshot was 79,200 ms old against the frozen 10,000 ms gate,
and public evidence still proves a zero owned reward-share and payout floor.
Official docs define the book field as the snapshot timestamp, so its age cannot
be waived as a last-change time. Do not retry, refresh, or select a September
valuation sibling. No credential, account, order, cancellation, fund, on-chain,
or protected-capture action occurred. Accepted edges remain 29, ranked
hypotheses remain 47, and terminal families become 128. Registry SHA-256 is
`8953013ab62dd510724c702ade0480234c95e6139f3ff57054e4cc12cbd352a3`;
durability-audit SHA-256 is
`14f3fdfcabaccf344e6ac7573412425ccecd5132e7e84428a1a40bd7c564bf11`.

A new Polymarket event card appeared to offer a direction-independent deadline
ladder: buy `NO` on Qwen Flash (3.9+) by October 31 and `YES` by December 31.
The exact common rule gives that package a one-pUSD floor in every release-time
state. However, the rendered card's apparent 0.99 sum combined separate market
probabilities rather than labeled acquisition sides. One prospectively frozen,
public, unauthenticated exact Gamma reconciliation returned October `NO` 0.50
plus December `YES` 0.52 = 1.02 pUSD before enabled fees, depth, latency,
failure unwind, and capital-time cost. It was rejected immediately; no book,
account, credential, order, fund, on-chain, or protected-capture request was
made.

The exact Qwen event is consumed and terminal. Do not refresh, reprice, or book
capture it. A multi-deadline card may reopen rank 31 only when its displayed
values are explicitly labeled side-specific acquisition prices and a frozen
exact metadata reconciliation remains strictly below the proved payoff floor.
Accepted edges remain 29, ranked hypotheses remain 47, and terminal families
become 127. Registry SHA-256 is
`4dfb9c701d88737992999978a77bdb8554ea2f6e47f6194c95a6db0112e8b70b`;
durability-audit SHA-256 is
`1c90bd52fbc61e8b3fba8c521170b87710ab7640696a17388dbec4ad59a23b8d`.

A bounded current-page pass screened eight distinct, previously unrecorded
end-of-September monotone equity ladders: OPEN, NFLX, PLTR, META, NVDA, TSLA,
MU, and SPCX. For thresholds `L < H`, the common-rule package `YES(above L) +
NO(above H)` has a one-pUSD terminal floor independent of the final stock
price. None of the displayed side-specific acquisition packages was strictly
below that floor. The closest was META `YES(above $680) + NO(above $700)` at
0.13 + 0.98 = 1.11 pUSD before fees, depth, latency, failure unwind, and
capital-time cost.

These rendered buttons are discovery-only, not executable economics. No Gamma,
CLOB, Binance market, account, credential, order, fund, or protected-capture
request followed. Do not repeat these exact eight September pages. The current
official Binance announcement/developer delta supplied no new Launchpool,
stock/perpetual, stock-option, or funding-cash-flow trigger. Public Polymarket
Combo catalog/configuration also does not reopen rank 33: the retained candidate
already proves RFQ quoting is approved-builder/authenticated and forbids using
catalog prices or `rfqEnabled` as executable quotes. The posted Binance Testnet
credentials were neither used nor persisted and cannot satisfy the frozen
Mainnet fee-evidence trigger for quarterly carry.

A bounded public rendered-page pass screened eight distinct, previously
unrecorded week-of-August-31 scalar ladders: DXY, EWY, Natural Gas, PLTR,
SpaceX, WTI, Gold, and Silver. Each exact event used a common within-event
high/low scalar, observation window, boundary convention, and fallback, so its
ordered monotone packages have a one-pUSD terminal floor. None of the displayed
side-specific discovery packages was strictly below that floor. The closest
was WTI `YES(above $110) + NO(above $115)` at 0.005 + 0.999 = 1.004 pUSD
before fees, depth, latency, failure unwind, and capital-time cost.

Rendered buttons are discovery-only and this near-miss is not an executable,
accepted, stable, or profitable edge. No Gamma, CLOB, Binance market, account,
credential, order, fund, or protected-capture request followed. Do not refetch
these exact pages merely to chase a four-mill gap. Reopen rank 31 only when a
distinct nonconsumed same-rule package is already strictly below its floor on
side-specific discovery asks, then freeze one exact primary and depth sequence.
The posted Binance Testnet credentials were neither used nor persisted and do
not satisfy Mainnet-only account-evidence triggers.

A later retained Binance all-options ticker exposed exactly two BTC option
symbols absent from the complete 1,576-symbol inventory captured five hours
earlier. One prospectively frozen zero-network rank-47 screen reused that ticker
and its synchronized USD-M Futures book. The 94,000 call had a gross terminal
floor of `-17,970.60` USDT per BTC before the 33.5-bip fixed stress; the 94,000
put had no positive displayed ask. Zero rows had a complete positive gross
floor, so no exchange-info refresh, ticker, book, depth, funding, account,
credential, order, fund, or protected-capture request was made.

This exact two-symbol snapshot is terminal and price-incomplete, not an edge.
Do not refetch or poll rank 47 again within the same UTC day. Reopen only after
`2026-09-01T22:44:19Z` for a later distinct population or another literal
material trigger. Accepted edges remain 29, ranked hypotheses remain 47,
terminal families become 126, registry SHA-256 is
`6ad94f7f192f61f7f47f146139985debe5da1252a6a1b7470e53581c5b9d9bf2`,
and durability-audit SHA-256 is
`4f7196ba69e68d8e28bfc02f3f60c1f063456a310a456a9bcbb8a0d39868aed8`.

A previously unconsumed early September 13 NFL window materially satisfied rank
30's public trigger. One frozen complete Gamma keyset request returned 12
events; 10 rule-complete active events proved 2,978 exact full-game margin and
total monotone relations. The 120 midpoint-like `outcomePrices` candidates were
not credited. An exhaustive zero-network correction found zero strict
side-specific sub-floor packages across 2,965 price-complete relations, with a
best rejection sum of 1.04 pUSD for a one-pUSD floor.

Thirteen Atlanta-Pittsburgh relations lacked the `bestBid` required for a
conservative second-outcome ask, and two events contained duplicate logical
thresholds. The exact population is therefore price-incomplete rather than an
exhaustive economic rejection, but it authorizes no refetch, sibling selection,
book, fee, account, credential, order, fund, or protected-capture access. Two
consumed offline failures were preserved: the first unnecessarily required an
unused side field; the second exposed the genuinely missing selected-side
price. The shared adjudicator now validates only the selected side, retains
price-incomplete relations, preserves historical complete-result hashes, and
fails closed before depth when a population is incomplete.

Catalog contract/result SHA-256 values are
`821e45f53f134dbc14e5f1d94bc657680ac047a74d188365b5c0d29154cab8bf`
and `126b1dc61fa379458aaa88a8edef899be437d33fda6cba1a7a29e3536cbe856f`;
final offline contract/result SHA-256 values are
`714f4688f00111be75d9cb39de668bed7101b3e54debb454f85ffdd2775d9c76`
and `1965d997ba11fdeb51cf5bac40e9a13569640d724b4f58624a59fa230e9d69f9`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
125, registry SHA-256 is
`1647539495270c7e732dd9fe8d421cc69456f6d7ccbb53e7655d34cfc7316f85`,
and durability-audit SHA-256 is
`0195a2498172aec1a92e86a3e553e6e344e2c87980761a2fa4c59ab6bead33d7`.

A newly indexed exact official Polymarket Perps fee source materially narrowed
rank 19's BTC short-Polymarket-Perps/long-Binance-spot carry near-miss. One
frozen public unauthenticated Markdown GET proved the zero-volume maker rate is
1.25 bips per fill, or 2.50 bips round trip. The retained carry had only
25.7920876712 bips available after funding, its conditional 6% OI reward, and
the two-leg 5% annual capital hurdle, so only 23.2920876712 bips remain before
every Binance fee, basis, quote conversion, transfer, custody, and failure cost.
The retained standard Binance maker sensitivity leaves only 3.2920876712 bips;
the conditional zero-maker quote sensitivity leaves 23.2920876712 bips but is
not exact current account or same-unit route evidence.

This is a tighter candidate budget, not an accepted or profitable edge. Do not
refresh funding or request books. Advance only after explicit read-only
authority proves the exact account's one-million-dollar daily-average gross OI
reward eligibility, Polymarket tier, Binance fee and eligible quote route, and
every same-unit external cost; only a positive residual may authorize one
separately frozen synchronized finite-size basis/depth study. Contract, capture,
and adjudication SHA-256 values are
`d262e39f404530dbc5c7451fa930f3c9a6180f32ba334e842c5fa467fa2565d1`,
`864efb46a06dda7a7af76e1dda766fe25870f80c78a1bfb322c83033acee7529`,
and `69fc78d24c28106e7c8f96a6bca32e92f4591582698261398bcdb77a6fc2fa78`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families remain
124, registry SHA-256 is
`d4a134ac80079622c76e0f39bcf30dba4a5d0ae539635dde93f5f802b3005cbb`,
and durability-audit SHA-256 is
`a520e2e0a41bb053cfeefe6752acecfe95d471c21fffe02b9512c06c46be7e96`.

A distinct Binance Wallet routed Aster USD1-versus-Binance USDT perpetual
funding lead has been rejected before any basis, book, account, credential,
order, or fund request. One prospectively frozen public Aster inventory capture
proved active BTCUSD1, ETHUSD1, and SOLUSD1 contracts. Three sibling funding
captures then aligned 209 eight-hour rows per asset with retained Binance
BTCUSDT, ETHUSDT, and SOLUSDT histories. Orientation was selected on the first
104 rows only and held fixed for 52-row validation and 53-row test roles.

All three assets failed the frozen 20-bip round trip, 10% annual two-leg capital
hurdle, and 23.954486-bip USD1/USDT stress. BTC and ETH had negative validation
and test gross funding; SOL had positive validation gross but negative test
gross. No asset passed even the training role after all costs, so the contract
forbade deeper requests. This exact 209-row population is terminal. Reopen only
on a material Aster or Binance funding, fee, basis, USD1 conversion, custody,
capital, or execution-architecture change. Prefilter contract and result
SHA-256 values are
`20ef24c12e5b18960ef491e07e51855868c18355adfb1e9fb3c3c465f28b307e`
and `c8983f0668fdd157add672940769f5ceb3db91c2cbe495ce27b8034dd491da91`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
124, registry SHA-256 is
`400313770bcd2cbf542e97f5a1d1fcf59b66dbf76121ecbd784e107bed4767c8`,
and durability-audit SHA-256 is
`13910f293d085c63f8ac6fb001e06b26c23dae7481af8852af1f1e7165fcc73f`.

A 2026-09-01 zero-request discovery pass stopped three tempting Polymarket
shortcuts before Gamma or CLOB access. Current official rendered pages show
that the new hourly BTC contract compares Binance one-hour candle open and
close, a five-minute SOL contract uses Chainlink start/end values, and the
daily ETH contract compares Binance noon-to-noon closes with a 50/50 equality
case. Those observation functions are not interchangeable, so cross-horizon
interval composition has no proved payoff identity. Separately, rendered Buy
quotes for the new September 3 and September 5 BTC range/threshold pairs and
the August 31-September 6 ETH hit/daily-threshold implications were all at or
above their optimistic guaranteed floors. These rendered values are discovery
only: they justify spending no deeper request, but do not terminalize a
population, establish executable economics, or change the registry.

The distinct active Polymarket `Ethereum above ___ on September 2?` threshold
ladder satisfied rank 31's literal trigger. A prospectively frozen corrected v2
screen made one exact public unauthenticated Gamma event GET and excluded
midpoint-like `outcomePrices` from economics. All 11 active markets and all 55
direction-independent lower-YES plus higher-NO packages had side-specific
rejection prices: lower YES used `bestAsk`, while higher NO used the conservative
`1 - bestBid` proxy. Zero packages were strictly below their guaranteed one-pUSD
terminal floor. The cheapest was 2000 YES plus 2100 NO at 1.004 pUSD, so this
exact event fails before fees, ticks, latency, failure unwind, or depth.

This September 2 event is terminal with zero book, fee, account, credential,
order, fund, on-chain, or protected-capture requests. Do not repeat or reprice
it, and do not select a BTC, SOL, or adjacent-date sibling after observing the
result. Reopen only on rank 31's registered literal trigger for a distinct
nonconsumed exact population whose side-specific rejection proxy is already
strictly sub-floor. Contract and result SHA-256 values are
`1ef97e13565e958fa67c380abb6fb9e519d4207e1a66a843ce2f9c2fc46bf8b7`
and `a83c568656fe15000624d0ae872abc75955bd42581714a1a926b114bc4206f33`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
123, registry SHA-256 is
`77a777ed40570ebd4773b431c20d32057fb36ce2f648011e503870fe58700352`,
and durability-audit SHA-256 is
`58109b3122be2876d4a6f8acd2a028bd13cc06255d6d379a8f2b4063802897ff`.

Rank 47's literal new-population trigger was satisfied and consumed efficiently.
A frozen zero-network audit compared the exhausted 1,410-symbol August 27
BTC/ETH/SOL crypto-option population with the already retained complete August
31 Binance Options `exchangeInfo`. The current population had 1,576 eligible
symbols: 508 exact additions and 342 removals. A separately frozen two-request
public prefilter then captured the complete Options ticker and USD-M Futures
book ticker once and evaluated only those 508 additions. Four hundred thirteen
had positive option-ask and executable perpetual-entry sides, but zero had even
a positive gross long-option plus opposite-perpetual terminal floor. They
therefore fail before the frozen 33.5-bip option fee, settlement fee, futures
round-trip, and expiry-basis stress, and before funding, ticks, capital, depth,
or account costs.

This exact August 31 delta is terminal with zero option-depth, funding, account,
credential, order, or fund requests. Do not refresh, subset, reprice, or rebuild
it. Reopen rank 47 only for a later distinct active BTC/ETH/SOL option population
or another literal material fee, settlement, tick, depth, funding, basis,
capital, or independently observed above-all-cost trigger. The population-delta
contract and result SHA-256 values are
`15604a7006b324bdd873481c79d5ac4ec34551a1d39d80904b546eb50ea441bc`
and `001abaada3b352235cbc38228dec6b6176a26cdfb33e208f0c3467f858cf9446`.
The price-prefilter contract and result SHA-256 values are
`5e2d0c36588530a4c4bf176bfe874a43af6f0063b636670fd4d599d3356bd5af`
and `93d2ed3c9b6041f9ffcc7f9579f184687113049051a421f9fc048d2d4e309eee`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
122, registry SHA-256 is
`fc026e6e2c76d6d6589f964dbe52fe140e126d87ee005c96e786289f40f9664c`,
and durability-audit SHA-256 is
`6d72a1f54af7e18351133f9bd329acfc9f2800f429274f0d564228efdd9e1d0e`.

A serious Polymarket existing-inventory opposite-lock candidate has passed one
frozen out-of-sample historical validation. The rule was selected only on the
retained August 25 public-wallet day, then one preregistered public
unauthenticated August 30 wallet-day GET produced 1,964 rows and passed with 66
causal 1-60 second locks across 27 BTC, ETH, and SOL conditions, 1,486.108451
matched shares, and 217.6302921908223351705204892211 pUSD of locked historical
cashflow after the full 0.07 fee on both legs and one adverse 0.01 tick on the
later hedge. A zero-network fixed-lock-set correction then conservatively
ceiled each leg of each matched fragment to the published 0.00001 pUSD fee
quantum. All 66 remained positive; total drag was only
0.0005909365952300705204892211 pUSD and corrected locked cashflow was
217.6297012542271051 pUSD. Both time halves and all three assets were positive.
A separately preregistered zero-network robustness audit then kept all 66 locks
fixed. At five total adverse ticks plus 0.05 pUSD fixed cost on every lock,
aggregate cashflow remained 154.9035632142271051 pUSD and each asset aggregate
remained positive, but only 43 locks, or 65.15%, stayed individually positive.
That fails the frozen 80% cross-lock gate. The median lock absorbed eight
additional whole adverse ticks above baseline, while the minimum and p10
absorbed zero. Treat the candidate as serious but fragile, not stable.

This is not an accepted edge, current executable profit, or a first-leg entry
strategy. It applies only when one outcome is already held for an independently
justified reason; the later exact-quantity opposite fill makes the matched
inventory direction-neutral. The public wallet is not owned or reproducible,
and current owned per-lot basis, book, merge or redemption, external costs, and
failure unwind remain unproved. Do not repeat, narrow, paginate, alias,
resample, or cherry-pick historical survivors. Advance only with explicit
read-only account authority plus independently preexisting eligible inventory;
reject each lot unless its exact completion is positive after every cost. Any
order requires separate explicit authority. Candidate SHA-256 is
`4ee6b1d3a54b6b112f9f031dc5cb91cb2abc2943119f512dfb26c79fe6c93a01`;
validation SHA-256 is
`b81af57f094f1ff75bcb77f9938ec7c84791af4e1cecb44b3402dac17d4dc1df`.
Fee-rounding correction SHA-256 is
`b423b44e57bfd329220256facf4b9eabe45371b267e7a91ea08aa11a666be204`.
Robustness contract and result SHA-256 are
`cc52cb4cafd7d36c432e39b1610e352bf513f405d2244432f7bcbc26dca2ea6f`
and `3f2bc8f2ea70345700062f43766bd1110299f7636367385daf4ce944f129046e`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
121, registry SHA-256 is
`669324eaae8533fb51fae63078f561499b74666a511847fe99f7c8b48eba085e`,
and durability-audit SHA-256 is
`3b4cbdacb890c13ce7f91ed1eb31feee65dc468503afbe0281f23f4760678467`.

The exact-title literature delta for `Arbitrage in Perpetual Contracts` (SSRN
5262988) is terminal as a source failure, not a Binance edge. The one frozen
primary PDF GET returned HTTP 403 and a 5,625-byte Cloudflare HTML challenge.
The indexed abstract is discovery only and cannot authorize a clamp-bound
collector or retained-data screen. Do not retry a query variant, alias, mirror,
or locale route. Reopen only on a materially revised complete primary version
from the authors or a public reproducible repository exposing the exact formula,
code, data, and after-cost execution method. Canonical adjudication SHA-256 is
`dd5058136c809425b833266f9a1bce568bab6e692301a2adbffec8b18b017275`.

The apparent August 31 Polymarket Sports maker-rebate change is terminal as a
source-quality event, not a market edge. A search-index rendering reported a
25% Sports rebate, 0.03 taker-fee rate, and USDC payout wording. One frozen
canonical `.md` GET returned exactly the prior 5,945 bytes and SHA-256
`8d2c6562bd1b3376bc3fc1557a60efef5aa3c1d856c7f8dcc405139a07e9ba2a`,
still stating 15%, 0.05, and pUSD. Search snippets are discovery only and may
never enter economics. The prior official cross-surface conflict and zero
forward Sports rebate credit remain. Do not retry or alias this drift while the
canonical hash is unchanged. Canonical adjudication SHA-256 is
`0874302d3acb6e641f38f693eafaa94d2a3b86167c0f796691efc5707b5cf64a`.
Accepted edges remain 29, ranked hypotheses remain 47, terminal families become
120, registry SHA-256 is
`6a7d560501abdd37f3a88db28a708ef69dd0e83668639ea5d4bac9836cee6dc3`,
and durability-audit SHA-256 is
`bd0f48ebd0b56e14ffdc988c0a72ba9c016e6a4c8db7d0b743242e26003efac5`.

Binance COIN-M inverse versus USD-M linear same-asset perpetual funding is now
source-bound and terminal for the retained BTC/ETH/SOL population. One frozen
public eight-request sequence retained current exchange configuration and 499
aligned funding rows per pair. Training selected short COIN-M and long USD-M
for all three assets, but combined validation plus test gross was only 3.9516,
11.8749, and 25.2915 bips versus one 32-bip two-leg entry-exit hurdle. BTC and
SOL also had a negative August, while ETH had a negative July. The zero-request
causal lagged-sign rescue required 116 to 172 turnover units per out-of-sample
role and lost 220.1847 to 330.5917 bips at only two bips one-way. Do not repeat,
resample, request books, or use credentials for this family. Reopen only after
a material COIN-M or USD-M funding cash-flow, collateral, settlement, fee, or
contract-architecture change. Canonical adjudication SHA-256 is
`560c42b8bbebf9c10ac1292c44043001fe4d26ccf85217f71b93095e72bf9230`.

The current first-page zero-alias Polymarket Rewards lead, František Oľha for
the 2026 Prešov mayoral election, is source-bound and terminal. Exact Gamma and
the exact `sponsored=true` condition row reconciled at 55 pUSD/day, 20 shares,
5.5 cents, zero maker fee, and about 54.22 remaining days. The separately
frozen two-token book then returned zero YES bids and zero NO asks; both book
timestamps were also 221,182 ms old against the frozen 10,000-ms ceiling. No
paired best-bid state or both-fill gross therefore exists, and the relative
reward formula still supplies no conservative owned payout share. Do not retry,
substitute a Prešov sibling, or use the Rewards-page display values as exact
economics. Canonical adjudication SHA-256 is
`efa25a2905029b44e96684660a291d12d8d049f49204136da1840099e550669d`.

The follow-on rank-17 semantics gate is also terminal. The public
`market_competitiveness: 0` value is not zero competitors or a 100% owned
reward share. The retained official formula is relative across makers, and the
commit-pinned official v2 SDK requires Level 2 authentication for both owned
reward-percentage routes. No retained primary source maps the public metric to
an account-independent positive share, so the proved owned-share and payout
floors remain zero and no further book request is justified by that field.
Reopen only on an explicit primary-source lower-bound definition or separately
authorized owned eligible-order scoring and realized-earnings evidence with all
costs. Canonical adjudication SHA-256 is
`7b58e5317a212c1e50f91bb7742f64f18915e7a3d477cdfc884c2c9938e3ee5b`.

The August 31 Binance margin-pair removal is source-bound and terminal as a
market-independent forced-settlement edge. The exact current primary article
removes isolated `WBETH/ETH`, `BNSOL/SOL`, and `BFUSD/USDT`, suspends borrowing
at `2026-09-01T06:00:00Z`, and schedules automatic settlement, order
cancellation, and pair removal at `2026-09-03T06:00:00Z`. It publishes no
aggregate position direction or size, settlement-price formula, conversion or
redemption ratio, fee, capacity, discount, reward, or recurrent cash flow.
Predictable timing therefore does not determine forced-flow direction and is
not structural carry. Do not poll books or pre-position around this episode.
Reopen only after an independent non-polling finite-size discount strictly
exceeds a separately source-bound exact same-account conversion or redemption
floor after every cost. Canonical terminal SHA-256 is
`b0367360207044b8b70cc76a72de288f698c563c0e52133836e1b6402b15f392`.

The non-aliased Binance-style delta-neutral dual-grid paper is terminal as a
research trigger. Its exact SSRN PDF request returned a retained Cloudflare
HTTP 403 page and may not be retried through a mirror. The indexed abstract is
discovery only and reports a 52-day Gate.io window, near-zero rather than exact
delta, trend-skewed quoting, and active inventory management. It proves neither
current Binance fills and costs nor cross-regime stability, and it supplies no
unique subsidy or atomic hedge. Do not build a collector or order test from it.

The source-selected Chicago August 31 tornado liquidity-reward lead is
terminal before books. The public Rewards surface displayed 50 pUSD/day,
20 shares, a 4-cent band, and zero competition, but exact Gamma reported a
4.5-cent band and the exact `sponsored=true` condition response was an empty
terminal population. The public funded-reward floor is therefore zero. Never
repeat either request, change the filter, substitute another observed city, or
use Rewards-page values as economics. This is the intended rejection-first
workflow: exact condition and exact sponsored allocation must reconcile before
spending a book request.

The next source-selected month-long reward lead, NYC September precipitation
over six inches, stopped even earlier. Its prospectively frozen rendered-page
October 2 date did not equal Gamma's exact September 30 23:59 UTC market end,
so the one-use sequence terminated after one public Gamma request. Sponsored
rewards and books were never requested and the exact candidate is terminal.
This exposed an avoidable source-semantics mistake: a rendered `End Date` can
be a user-facing resolution date, not Gamma `endDate`. Future contracts must
bind Gamma time from Gamma or omit rendered-to-Gamma equality; never repair or
repeat this consumed market or substitute another observed precipitation bin.

Two later source-selected reward candidates are also terminal. Navdeep Bains
reconciled at 40 pUSD/day and 20 shares, but its only book snapshot was 174.712
seconds old. A descending-reward pass then excluded consumed crypto rows, the
already retained Fed event, and Ontario siblings before selecting the first
zero-alias event: GTA VI Extended Look under 20 million views. Exact sources
reconciled at 536.99616 pUSD/day, 200 shares, 4.5 cents, zero maker fee, and
3.327 remaining days. Its books had zero cross-token skew and best bids summing
to 0.99 for 2 pUSD both-fill gross, but the snapshot was 17.830 seconds old
against the frozen 10-second ceiling; improving both legs one tick crossed to
1.01 and negative 2 pUSD gross. The official relative-score formula and the
undocumented mapping of `market_competitiveness` to any owned share leave the
public payout floor at zero. Do not refetch, reprice, weaken freshness, or
substitute another GTA outcome after observing this result.

The next complete CFB payoff-lattice discovery window, September 8 through 12,
returned zero events and no cursor. It is a consumed complete empty population:
do not poll, refine, or move immediately to another unproved forward horizon.
Verification exposed that its consumed runner still used midpoint-like
`outcomePrices`; zero events made that decision invariant. The exact old runner
is now immutable and the reusable runner is corrected prospectively. A
zero-network correction repriced all 139 retained September 3–7 CFB relations
with `bestAsk` or `1 - bestBid`: eight apparent candidates became zero, with
best sums of 1.07, 1.08, 1.07, and 1.62 pUSD for 1 pUSD floors. Never use the
superseded CFB `outcomePrices` gates.
A separate zero-network audit then exhausted every exact relation exposed by
the retained August 26 holding-yield market rows: seven mutual-exclusion,
triple-NO, and nested-threshold packages across four events. None was strictly
below its optimistic common-rule floor; the best Fed pair cost 1.01 pUSD for a
1 pUSD floor. Do not reprice or book-capture those retained rows.

The retained Los Angeles Galaxy versus New England Revolution event pair proved
16 exact-score-to-full-game-result payoff implications, but also exposed a
request-efficiency bug: Gamma `outcomePrices` made the best package appear to
cost 0.750 pUSD while side-specific fields already implied 1.260 pUSD. The one
consumed current book confirmed a 6.20-pUSD five-share cost for a 5-pUSD floor
before fees and was stale and skewed. Never use `outcomePrices` alone to
authorize a book. Use YES `bestAsk` and a direct NO ask or conservative
`1 - YES bestBid` rejection proxy first. The exact soccer pair is terminal.

Polymarket fee rounding is not a hidden standalone edge. Across 96 distinct
conditions in all 15 retained current book files, every minimum order was five
shares and ticks were 0.001 or 0.01. At the lowest published nonzero fee rate,
the smallest whole-order extreme-tick fee is 0.0001998 pUSD, or 19.98 fee
quanta. A zeroed fragment would have to be below 0.2502502503 shares, while fee
aggregation and fragment control are unproved; every zeroed assessment saves
strictly less than 0.00001 pUSD. Never split, churn, self-match, or manufacture
volume for rounding.

Binance Spot OPO and OPOCO are a real direction-independent execution candidate,
not standalone profit. Pinned official terms prove the pending sell uses the
working buy's received quantity with commission and lot adjustment, needs no
pre-funded pending sell, and removes one post-fill client submission from the
frozen sequential comparator. Retained production and one exact frozen public
Testnet response both expose OTO, OPO, and OCO support on BTCUSDT, ETHUSDT, and
SOLUSDT. The pending order activates only after full fill, so partial fills are
unprotected, and no monetary latency, fill, fee, or profit floor is proved.
Do not repeat the Testnet configuration request. Any order experiment requires
separate explicit Spot testnet order authority and an independently required
minimum-size organic buy-then-sell comparator.

Binance Spot SBE Diff Depth is a distinct direction-independent freshness
candidate: the retained current official source documents 20 ms updates versus
the retained JSON Diff Depth source's fastest 100 ms cadence. Do not call that
an 80 ms arrival lead or profit. Both SBE best bid/ask and JSON book ticker are
documented real-time, SBE access requires an Ed25519 API key, and decoder,
continuity, receive-time, fill, and cost evidence are absent. The consumed
official source gate failed only because Markdown bold delimiters wrapped one
required sentence; preserve and adjudicate those retained bytes, never refetch
or alias them. Advance only with a designated ephemeral Ed25519 key, explicit
read-only market-data authority, and a precommitted same-host dual-feed capture.

Binance Spot FIX is a separate direction-independent execution-risk candidate.
The retained current official contract says `UNORDERED` should perform better
with multiple messages in flight, FIX ExecutionReport push should perform
better, and one FIX mass-cancel message covers every account order on one symbol
across connections. It does not document automatic cancel-on-disconnect, any
non-live order-entry endpoint, or a measured latency, fill, or profit floor.
Do not repeat the source. Reopen only under the literal non-live Ed25519
`FIX_API` session-and-order comparator trigger; the supplied HMAC-style testnet
credential is incompatible and must not be used.

The Polymarket BTC/ETH/SOL interval-composition family has a mechanically
verified one-pUSD payoff floor across all 50 packages in 25 retained aligned
settlements, with zero violations. It remains unaccepted: the current displayed
best package cost 1.990 pUSD and no atomic executable sub-floor acquisition is
proved. One separately frozen 48-condition historical trade request returned
HTTP 408 before exposing any row. Never retry, split, narrow, paginate, reorder,
or alias that consumed population. Reopen only for a future distinct aligned
population that passes a strict rejection-only sub-floor gate, followed by a
prospectively frozen exact live CLOB package capture.

The GLWUSDT terminal special-funding reconciliation is consumed. Its complete
ten-row delta contained one negative Special row matching the 0.28 USD gross
dividend within 0.000000107 USDT, but it arrived 1.003 seconds after the bStock
snapshot. The hypothesized pre-snapshot timing gap did not exist. Never repeat,
retry, paginate, alias, extend, repair, or book-capture this 2026 episode.

The current GOOGL holiday-gap screen is also terminal. September 7 is a Nasdaq
holiday, so the applicable ex-dividend rule moves the ex-date to September 4,
the same date as Binance's GOOGLB snapshot; the apparent three-day record-date
gap is not a pre-ex-date gap. Retained exact GOOGLUSDT history shows the prior
gross-matching Special debit at `00:00:00.007Z`, only 7 ms after midnight, and
current terms leave fees, costs, and other deductions unbounded, so the public
conservative net-distribution floor is zero. Do not request current GOOGL books
or funding, repeat either consumed source capture, or attempt a millisecond
race. Reopen rank 34 only for a future episode whose bStock snapshot materially
precedes the official exchange ex-dividend adjustment and whose current primary
terms bind a strictly positive conservative net-distribution floor.

The same retained QCOMB/PYPLB/GOOGLB announcement is now exhausted at its
cheapest shared gate. QCOMB snapshots on September 3 and PYPLB on September 4,
and matching QCOMUSDT/PYPLUSDT histories were already source-bound, but Binance
reinvests only the net dividend after withholding taxes, fees, costs, and other
deductions without a retained ceiling or complete formula. The conservative
public net-distribution floor is therefore zero for both exact episodes before
issuer-calendar, funding, or book work. No new request was made. Do not research
QCOM or PYPL issuer dates or markets for these episodes, and do not repeat or
alias the consumed announcement.

The first source-selected nonconsumed NFL rank-30 event, Patriots vs. Seahawks,
is terminal. Its retained rules proved 291 full-game margin and total monotone
packages; two rejection-only Gamma sums were 0.995 pUSD, but the precommitted
Over 24.5 plus Under 26.5 package cost 5.55 pUSD at exact asks for a 5 pUSD
floor before fees and 5.65 pUSD after one adverse tick per leg. The oldest book
was 11.031 seconds old, but freshness cannot rescue the independently negative
zero-fee upper bound. Do not refresh this event or select the already observed
49ers-Rams sibling after outcome access. A distinct zero-request audit of the
same retained bytes also proved all nine valid team-total to full-game-total
implications were above floor; the best cost 1.465 pUSD for a 1 pUSD floor.
Do not rebuild, reprice, or book-capture that graph.

The V1 retained team-ladder/additive audit is superseded. Its 33 additive rows
were valid but not exhaustive: `Under A + Under B + Over G` has a one-pUSD floor
whenever `G <= A+B`; no lower bound is needed because the Over leg already pays
in every game-over state. V1 also reported midpoint-like `outcomePrices` rather
than side-specific rejection prices. The deterministic V2 correction proved
325 full-team additive covers and 20 full-team ladders, with zero side-specific
sub-floor candidates and best sums of 1.78 and 1.34 pUSD. It also exhausted
8,508 half, quarter, team-period, and spread ladders/partitions. Across all 8,853
relations, zero were sub-floor; the best remaining period package cost 1.44
pUSD. The correction used the same retained bytes and zero network requests.
Never rebuild, reprice, or book-capture this graph.

The older complete September 13-21 NFL catalog is now corrected too. Its 674
apparent Gamma candidates came from midpoint-like `outcomePrices`. A zero-network
audit repriced every one of its 4,621 proved monotone packages with side-specific
`bestAsk` or conservative `1 - bestBid`; all fields were complete, zero packages
were strictly below the one-pUSD floor, and the best cost 1.05 pUSD. Preserve the
original catalog and Commanders-Cowboys depth loss as superseded methodology
evidence. Never repeat, reprice, or book-capture that consumed NFL window.

The one-use primary-literature delta after the retained August 29 checkpoint is
terminal. Its 50-entry newest-first ArXiv response contained zero papers after
`2026-08-29T22:37:17.664866Z`; the newest returned paper was August 26, so no
paper download, venue data, or ranked retry was justified. The exact *Taker vs.
Maker Arbitrage* title and SSRN `7269858` were already adjudicated on August 27
and do not reopen rank 2. Preserve the consumed XML root-tag formatting failure
and its offline semantic adjudication; never repeat, paginate, change keywords,
or alias this query. Future literature work must alias-check exact titles and
identifiers before browsing and use namespace-aware XML gates.

The historical Binance Direct Stocks fee overlay ended at
`2026-08-31T00:00:00Z`. Binance's August 28 extension announcement says the
same 0.05% spread and 0.17-USD fee tiers continue through September 30, but the
logged-out live Direct Equities fee schedule still publishes the August 31 end.
This current primary-source conflict makes the public current discount floor
zero. Do not retry the consumed CMS article request or poll the unchanged fee
page. Reopen only when an explicit correction or materially updated live fee
schedule resolves one effective end time; this can amend duration only and
cannot create a new accepted edge.

The consumed Polymarket created-event delta after
`2026-08-30T16:07:16.021321Z` hit the endpoint's effective 100-row cap before
crossing its cutoff. It screened zero packages and authorized no books. Never
paginate, narrow, refresh, alias, or repeat it; a future delta must first prove
its worst-case arrivals fit below the observed cap.

The retained August 29 soccer population is now exhausted across the remaining
exact-score cross-family surface. Seven complete match families produced 1,428
rule-proved exact-score links to first scorer, BTTS, full-game totals, and team
totals. Every relation had side-specific rejection prices and zero were strictly
below the one-pUSD floor; the best only equaled one pUSD before books, fees, or
execution. Do not rebuild or reprice this population. Reopen only on the rank-31
future-distinct-event trigger.

The same retained page is also exhausted across player-prop and corner-count
identities. All 43 NO-anytime-goalscorer plus YES-Over-0.5 packages were
price-complete, zero were sub-floor, and the best cost 1.90 pUSD for a one-pUSD
floor. Seven complete corner events produced 1,820 full/half/team monotone,
additive-partition, and adjacent-interval parity packages; all were
price-complete, zero were sub-floor, and the best cost 1.10 pUSD. Preserve the
corner V1 literal-rule and V2 strict-zip failures. Do not rebuild, reprice, or
request books for either population.

The rank-one Polymarket holding-yield gate is consumed through V12. The V11
public BTC activity request found exactly one new 0.0133 pUSD YIELD row at
2026-08-31 00:10:10 UTC, 86,190 seconds after the prior row. The protected V7
partial payloads were not read, repaired, or touched. The exact V12 public
receipt request reconciled a successful distributor pUSD transfer in Polygon
block 92953322, extending single-wallet continuity to 19 positive rows. Never
repeat V7, V9, V10, V11, or V12. Current three-wallet rate, account
qualification, deployment readiness, and new-capital profit remain fail-closed.

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
  The consumed activity payloads also contained one later daily BTC and ETH
  `YIELD` row. A distinct two-request monitor reused those exact rows and
  retained equal mergeable positions, then reconciled both transactions to
  successful exact pUSD transfers from the holding-yield distributor. The
  intervals were 86,474 and 86,724 seconds and the amounts repeated at 0.0133
  and 0.0391 pUSD. This strengthens payout continuity but does not repair v7,
  qualify the current three-wallet rate, authorize new capital, or establish
  deployment readiness. Canonical continuity result SHA-256
  `2eb7b434170afb195cc4f4faef8260ac4ec30b655c20fc07ee1bc9acbdfe090d`.
  After the next 25-hour payout window closed, a frozen one-request BTC pulse
  found exactly one later `YIELD` row: 0.0133 pUSD at
  `2026-08-30T00:13:40Z`, 86,290 seconds after the prior row. A separately
  frozen one-request Polygon receipt check reconciled its exact successful
  pUSD transfer from the holding-yield distributor. This extends observed BTC
  continuity to 18 positive rows without repairing v7, qualifying the
  three-wallet current rate, or opening a forward new-capital profit floor.
  Do not repeat either consumed check; the next distinct single-wallet pulse is
  prohibited before `2026-08-31T02:15:30Z` absent a material terms change.
  Canonical receipt result SHA-256
  `4e57d6c0216886144fb89f8ae69b11a2eee4db37149ce6c956adecf293b7b927`.
- The complete one-request September 5 CFB catalog returned 89 events, excluded
  both consumed games, retained 88 exact payoff relations across 58 rule-complete
  events, and found six Gamma sub-floor candidates. Its precommitted best
  Fordham-North Dakota State Over 56.5 plus Under 57.5 package failed the only
  authorized depth screen: two 0.92 asks cost 9.2 pUSD for a 5 pUSD floor, while
  book age was 152,285 to 300,344 ms and skew was 148,059 ms. No fee request was
  needed. Do not repeat, narrow, paginate, refetch the event, or cherry-pick the
  other five observed candidates. Canonical depth result SHA-256 is
  `8d2f9e8a5f00fa84c4291822692712151cb51bc7bdc7a659d12aa13f788361a1`.
- A distinct USD-M Futures BNB fee-reduction lead remains unaccepted. The
  retained current official API index proves a signed `GET /fapi/v1/feeBurn`
  status query and a separately state-changing `POST /fapi/v1/feeBurn` toggle,
  but the frozen exact current public fee-guide request returned HTTP 202 with
  2,035 bytes and zero of three required discount terms. Do not repeat or alias
  that request, infer the cached 10% publication text as source-bound current
  account economics, acquire standing BNB, or toggle fee burn. Reopen only on
  a materially new byte-retainable official source or both designated
  credentials, explicit signed GET-only authority, and an independently planned
  legitimate organic BTC/ETH/SOL USD-M positive-commission question. Every
  toggle remains separately unauthorized. Canonical adjudication SHA-256 is
  `3a2c7358757491c8e0e9d737a76583756596db59346c5877a7bcfc1ca9e300b4`.
- BNSOL Boost APR airdrop rewards are a distinct existing-holding lead, not an
  accepted edge. The current official web-rendered schema separates base BNSOL
  rewards from boost rate history, `CLAIM`/`DISTRIBUTE` history, unclaimed
  rewards, and the `TRADE` claim action. The frozen raw page request returned
  HTTP 202 with 2,038 bytes and zero of eight required terms, so rendered schema
  examples are discovery only and the public floor is zero. Do not repeat or
  alias that page, buy or retain BNSOL, double count rewards, call signed reads,
  or claim. Reopen only on a new byte-retainable official source or both
  designated credentials, explicit signed GET-only authority, and independently
  existing BNSOL; claiming still requires separate authority. Canonical
  adjudication SHA-256 is
  `f0b6b4df8632b1cc302bdb24189c4336968be1fa9b94d1f7792205f46450c466`.
- Binance Link-and-Trade client kickbacks are accepted only as exact realized
  positive own-account income from independently justified legitimate organic
  Spot trades on an already linked account with `rebateWorking` true. The
  current official rendered contract distinguishes the client kickback route
  from the explicitly named partner-rebate route and binds client income,
  asset, symbol, and time. Public rate, account eligibility, owned income, and
  forward profit remain unproved, so the public floor is zero. Never open,
  relink, customize, churn, reroute, or trade for kickback. Signed reconciliation
  requires both designated credentials and explicit GET-only authority; every
  link, customization, order, trade, transfer, or mutation remains separately
  unauthorized. Canonical result SHA-256 is
  `ada551b385d9040e4126ee0e73e1dd1f417b103e6c5c5f7c567411ab913ff065`.
- Binance Exchange Link commission rebates are a distinct direction-independent
  candidate under the existing organic platform-fee family. The retained
  current official index names separate Spot and Futures commission-rebate
  record endpoints, but it does not bind either endpoint's exact security
  class, parameters, response schema, eligibility, rate, payout, or costs. The
  public floor is zero and the candidate is not accepted. Do not infer access
  semantics from `GET`, create or change subaccounts, API keys, permissions,
  commissions, transfers, or trading flow. Reopen first on a byte-retainable
  current primary exact endpoint contract; any later signed reconciliation also
  requires explicit GET-only authority and independently existing bona fide
  external flow. Canonical candidate SHA-256 is
  `245ef96228dccf51194f0e10176ffa39676ec4e5e07f78a2daefa1205b2fde3a`.
- Binance Prediction Trading exposes a distinct direction-independent
  collateral-carry lead through per-market `isYieldBearing` metadata alongside
  vendor, chain, collateral, fee, and slippage fields. It is not an accepted
  edge: the current detailed official schema requires signed timestamps for
  market list and detail, defines none of the yield recipient, rate, base,
  accrual, distribution, or redemption economics, and both designated
  credentials are absent. The frozen machine-readable schema retention request
  returned HTTP 202 with zero bytes and is consumed; do not retry its URL or an
  alias. Reopen only with a materially new byte-retainable complete official
  yield contract, or both designated credentials plus explicit signed GET-only
  Prediction Trading metadata authority. Canonical failure adjudication SHA-256
  is `e0049982adfdb631bb71bc7ebaf957d0a96336b42f269783ec99d4812e12bafc`.
- A materially new Ball State vs. Ohio State college-football deployment
  extended the sports monotone-payoff family beyond its consumed NBA/WNBA/NFL
  windows. Exact Gamma exposed active moneyline, Ohio State -50.5 spread, and
  56.5 total markets. A complete nine-state score/tie/cancellation-aware
  nonnegative superhedge screen found a minimum displayed cost of exactly
  1.0000 pUSD, leaving zero strict gross headroom before fees or execution
  costs. No book was requested. Do not repeat this event; advance another sports
  event only when exact rules and current fees are source-bound and a Gamma
  displayed package is strictly below its guaranteed payout floor. Canonical
  terminal SHA-256 is
  `13eb2d843260b2e05d693f82ddc6f3efcbae51e83d61f5fe06693be56c2c30a5`.
- The distinct 26-bin Elon August 28-September 4 fixed-NegRisk event fired rank
  31, but its one exact Gamma response already had two closed nonaccepting
  losing bins. The remaining 24 accepting YES prices summed to 1.0055 pUSD, so
  even the most optimistic retained complete-set interpretation lost 0.0055
  pUSD before fees and no book was requested. The consumed runner failed on the
  partial resolution after retaining raw evidence; its exact bytes are in an
  immutable sidecar and the reusable runner now returns a fail-closed diagnostic
  for that state. Do not refetch this event. Canonical terminal SHA-256 is
  `d22954b0611b7bd8210dbc8d59bf1f9a645c118331da057d80585b364ed0c0ac`.
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
  Its terminal file is now audited: the connection closed after only
  `19635.343` seconds, or `22.726091435%`, so source continuity and the frozen
  24-hour duration both failed. The only 0.82-equivalent execution exceeded the
  initially visible 98.74-share queue by `1.246314` shares, below both the five-
  and twenty-share frozen orders. No queue-censored input fill or causal output
  unwind is admissible. Do not restart the consumed contract. Canonical terminal
  adjudication:
  `polymarket-negrisk-maker-input-prospective-terminal-v1-2026-08-29.json`,
  result SHA-256
  `613453649f84407d6216e72228bdb16005b0a5c290c6bd58fa522007de5317e5`.
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
  A zero-request scope correction now accepts only an exact realized positive
  final Liquidity Program rebate on independently justified legitimate organic
  owned maker fills after every incremental cost. It does not credit the public
  tier schedule, accept a standalone market-making strategy, authorize volume,
  or double count zero maker fees, BNB discounts, bStocks promotions, or symbol
  commission savings. The public forward floor remains zero. Canonical overlay
  SHA-256 is
  `13c0a9468e439f9163ede0b5824c9b065737078ece0cba5f2b348fb402ef01d4`.
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
  A distinct current Cross Margin overlay accepts only the exact 5% interest
  reduction on an independently existing legitimate borrow when enough BNB is
  already held in the same Cross Margin account for unrelated reasons and every
  incremental cost is lower than the saving. Current official account evidence
  exposes signed read-only `interestBNBBurn` status and converted-interest rows;
  the toggle itself is a signed POST and remains unauthorized. Isolated and
  Portfolio Margin, new borrowing, BNB acquisition or retention, and double
  counting with trading-fee discounts are excluded. The public monetary floor
  is zero and deployment readiness remains false. Canonical overlay SHA-256 is
  `38aa0313cdd71a2613f3850267e71acd9d44006dd2699e4c00f801ffca8772f8`.
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
  edge. Binance published 8.07% and then 5.78% effective APR for the first two
  completed weeks of its RLUSD/XRP campaign. It has no stated individual cap, but requires an
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
  Observe the fixed September 4 and September 11 RLUSD updates; do not extrapolate
  completed weeks, illustrative examples, or system timezone into profitability or
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
  The current bStocks zero-maker promotion still cannot reopen the family: even
  erasing the entire 60-bp two-leg execution stress leaves the basket negative
  by `21.8046206700` bps in validation and `48.1975271954` bps in test, with
  negative bootstrap lower bounds and zero of 12 test symbols positive. No
  funding or book refresh is allowed for this promotion. Canonical retained
  counterfactual:
  `binance-bstocks-zero-maker-carry-retained-counterfactual-v1-2026-08-29.json`,
  SHA `2af1504748f51ad36c18c76162a91e82803395311932d73f1809d2781dfc4fb7`.
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
  A 2026-08-30 discovery signal exposed new dated rows on the official rewards
  page, but its one-use source-contract check failed closed before market data.
  The current official v2 client omits `sponsored=true`; four frozen default
  pages each returned the maximum 500 rows and the cursor remained nonterminal
  at offset 2,000. That incomplete default population is not aggregation-
  comparable with the prior complete 54-row `sponsored=true` baseline. Do not
  follow the cursor, select a partial-page winner, or call the 1,996 retained
  nonbaseline identities a population delta. Reopen only after a current
  primary source explicitly defines bounded comparable aggregation/filter
  semantics or source-selects a distinct exact per-market dated endpoint.
  Canonical terminal adjudication:
  `current-rewards-population-delta-terminal-v1-2026-08-30.json`, SHA-256
  `0a885b96eadbc0109ec3da70a8670680a23813a5f492d61abf89e49f3e892481`.
  A distinct source-selected exact-market retry then retained the active Elon
  Musk 40-64 posts Gamma and sponsored reward responses. Its initial gate
  incorrectly treated a stale or misattributed discovery tuple as an economic
  input even though the exact sources agreed on 50 shares and 5.5 cents. A
  separately frozen correction reused those exact bytes and made only one book
  request. It rejected at 6,408 ms age; one-tick 0.48 and 0.53 bids were both
  marketable, summed to 1.01, and lost 0.50 pUSD if both filled before reward
  uncertainty. A precommitted zero-network best-bid join then earned 0.50 pUSD
  both-fill gross but risked 26 pUSD orphan loss; observed displayed competition
  required 43.554 reward days versus 3.693 remaining, and 100-times competition
  required 4,306.850 days. No fresh capture is justified. Canonical rejection:
  `facecfaa3b92d905c700083c7b8afe153adc495403ceabc91e417bdb248d059b`.
  The accepted count remains 21; registry SHA-256 is
  `de28d80cc4b0b9cd1bd3f9954cb840dcaefe46fcc0fdf9ac3fd53218169370cb`.
- The separate official crypto maker-rebate schedule has exact conditional
  filled-order arithmetic, not an accepted standalone market-making edge. At 50 shares bid on each side
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
  payout floor therefore remains zero and no standalone strategy is accepted. Canonical
  recurrence:
  `crypto-maker-rebate-public-recurrence-v2-2026-08-26.json`, result SHA-256
  `c992e0e1febc1a9789289cb129c166280ee0192cab203d3a6935a8c40e949612`.
  A zero-request scope correction now accepts only the exact realized positive
  pUSD rebate on independently justified legitimate organic owned BTC/ETH/SOL
  maker fills after every incremental cost. It does not credit a nominal
  rebate, authorize a quote, accept the underlying strategy, or let the rebate
  rescue negative base economics. The public forward floor remains zero.
  Canonical overlay SHA-256 is
  `a8db5f3c823c8b1caffa6b0032282647dbb1e7fb014f8923821c1b1fe97d1c81`.
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
  APR is 5.27% after three positive completed weeks; the current 6.33% boost is
  creditable only when at least 300 USD1 of
  lowest-hourly daily Futures open interest already exists for an independently
  justified organic strategy. Simple Earn is not in the closed eligible-account
  list, so the same principal cannot receive both yields under the published
  contract. Credentials, region eligibility, exact account rates, reward-sale
  costs, and USD1 principal risk remain unresolved. Do not acquire USD1 or open
  Futures exposure for this promotion. The next and final public refresh is the
  fixed 2026-09-04 trigger. Canonical gate:
  `binance-usd1-wlfi-holding-airdrop-gate-v1-2026-08-26.json`, result SHA-256
  `c67367932b440d6f4a23330a17c405c0e15b0021b0484575a0b0efcc6e9238a6`.
- The mutually exclusive USD1 Simple Earn route is now separately accepted only
  for at most the first 1,500 USD1 that is independently already held idle. The
  logged-out public product page displayed `8.62% Max`, decomposed as a variable
  `1.62%` Real-Time APR plus a fixed `7%` promotion bonus. Against the latest
  completed 5.27% holding-airdrop base, the fixed bonus has 173 annualized bps
  of gross uplift and recovers one forfeited airdrop day in 3.0463 days. At a
  1,500 USD1 balance, its same-week fixed-bonus sensitivity exceeds the latest
  base airdrop by only 0.4977 USD1 before account and transition costs. Never
  count both rewards on the same principal or infer that the
  displayed USD1/U/USDC/USDT rate ordering justifies a conversion. Exact account
  eligibility, capacity, liquidity need, transition cost, tax, and redemption
  timing must all be proved before any subscription, which requires separate
  authority. Canonical adjudication:
  `binance-usd1-simple-earn-versus-holding-airdrop-allocation-edge-v1-2026-08-27.json`,
  result SHA-256
  `a4158bf059f4f5ad839b2f504c08c4afc65615260b4171533866f4c2337494e0`.
  A zero-network remaining-horizon correction now prevents the activation-day
  28-day stress margin from being treated as current. For a 2026-08-30 new
  allocation, the unchanged fixed case leaves only 1.5162802791 bips, or
  0.2274420419 quote units at the 1,500 cap, before every unproved account,
  conversion, redemption, issuer, tax, custody, and operating cost. It leaves
  only 0.0550930645 bips for a 2026-09-01 subscription and turns negative on
  2026-09-02 even before those costs. Do not roll this calculation or refresh
  books daily. Before the sign change, only both designated credentials plus
  explicit signed GET-only authority could justify account prequalification;
  every subscription or conversion remains separately unauthorized. Canonical
  stress SHA-256 is
  `669fd50772087cb81a4d1e9439e5666a75b5c1ed9de68b1e8b27cb360f2d5934`.
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
  8.5% APR on the first 5,000 U through 2026-09-14. A zero-request current-
  horizon correction does not credit the approximate variable 0.5% component:
  for a 2026-08-30 subscription, the fixed 8% bonus leaves only 2.6976862267
  bips or 1.3488431134 quote units after the retained USDT alternative,
  displayed spread, and worst observed 19-day close move, before every unproved
  cost. The unchanged sensitivity turns negative for a 2026-09-02 subscription.
  Buying U for the promotion remains rejected. The issuer's terms give ordinary secondary
  holders no direct redemption right or reserve claim, and its homepage lists
  only a December 2025 attestation. Do not infer region from the host timezone,
  acquire U for the reward, roll this stress daily, or call the promotion stable
  or deployment-ready.
  Canonical gate: `binance-u-flexible-idle-holding-yield-gate-v1-2026-08-26.json`,
  result SHA-256
  `6f44b65e5aa85d33cc02e8611a372162cf00f4162fdff99828a31cf498ced6f9`.
  Canonical current-horizon stress SHA-256 is
  `d9be584383bdbf4e45f570987103e9d380358b2ba08aafc865ccecfc4b2c225e`.
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
  continues earning. The direct direction-independent same-asset loop fails its
  first product gate: the live official page explicitly says the collateral and
  loan cryptocurrency cannot be the same, while the newer 2026 overview does
  not reverse that restriction. Two reciprocal cross-asset isolated loans are
  not an equivalent workaround because either position can liquidate under an
  opposing price path. All current asset, rate, LTV, position, income, and reward
  inputs remain signed USER_DATA. The designated credentials are absent; no
  signed request was sent and the public after-cost floor is zero. Do not borrow,
  subscribe, repay, adjust LTV, acquire collateral, use leverage, or double-count
  idle yield. Canonical gate and same-asset adjudication:
  `binance-flexible-loan-simple-earn-collateral-yield-gate-v1-2026-08-26.json`,
  result SHA-256 `ac010265c5236152907ac7b3c12ce13104f473b4cc61c5db43fb8b28c6678182`;
  `binance-flexible-loan-same-asset-loop-adjudication-v1-2026-08-30.json`,
  result SHA-256
  `7106bb072533327d3154c773ba2c1969ff7891df6ec16d314f2d8f1b410f48e6`.
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
`bb2d030b9465ded6cc4ce0ba894719d60ecd812d673432a10941db1779d0d758`.
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
- The one-use exact current sports title lead `Colorado Rockies vs. Washington
  Nationals` returned zero events from the public Gamma keyset endpoint and
  stopped before prices or books. Do not adapt or rerun that title. It closes
  only that displayed lead; the broader exact-subset family still waits for a
  future active complete-rules pair. Contract SHA-256
  `99559dd57d8ba1520fd4f607c4e4e56cea1070a2798536941af10134e4376aed`;
  result SHA-256
  `e5ce48b6b0521a5ba2fe58ae17316e703ab2155934a126e603eeadf81e219d9c`.
- The distinct current postponed BOS/NYY MLB event proved 16 active accepting
  markets and 37 exact monotone moneyline, spread, and total-run relations. A
  single public 30-token batch found zero positive five-share guaranteed
  packages after the exact current `0.03` fee curves, displayed depth, and two
  adverse ticks per leg; the best still lost `0.53262` pUSD on a five-pUSD
  floor. All 30 ask arrays were strictly descending contrary to the documented
  ascending order, so the original runner failed and a no-refetch outcome-aware
  adjudication preserved the consumed evidence without promotion. Do not retry
  this event. Adjudication SHA-256
  `1e75e049abb116955294d878830f940491fe4044f09c7e3564ad2761c0129178`.
- A fresh Los Angeles Dodgers versus Detroit Tigers event initially proved 24
  within-time-scope full-game and first-five margin and total-run packages.
  Every one failed the Gamma rejection gate; the best displayed sum was
  `1.010`. Preserve that narrower adjudication, but do not call it a complete
  event lattice: it omitted cross-period implications. A separately frozen
  retained-data correction proved that first-five runs cannot exceed full-game
  runs, so full-game Over 6.5 plus first-five Under 6.5 pays at least one pUSD
  per share. Its Gamma sum was `0.965`, an optimistic `0.035` pUSD per-share
  lead that correctly triggered one exact two-token book batch. Current asks
  were `0.44` and `0.96`: seven pUSD for five shares against a five-pUSD floor,
  a two-pUSD zero-fee loss with 1.511 seconds book skew. Six-tick stress lacked
  complete depth, so zero fee requests were needed. Do not resample this event.
  Canonical cross-period result SHA-256
  `fc01c54e9c04117067aa3b43ae194649b93efc12a5265fce508e64f082f320b2`.
- One bounded current MLB catalog request then screened this cross-period
  identity without event-by-event polling. The response hit the frozen
  100-event limit, so it is explicitly partial, not complete current coverage.
  Ninety-two events were future-dated at receipt time; only five exposed exact
  shared full-game/first-five total thresholds. Zero had a displayed sum below
  one. The best was Arizona-San Francisco at `1.260`, already an optimistic
  `0.260` pUSD per-share loss before books, fees, or delay. The consumed result
  retained counts and candidates but omitted rejected rows; a zero-refetch
  offline adjudication preserved all five sums. Do not adaptively paginate this
  page or request books. Canonical adjudication SHA-256
  `d3ba85e995753d781178fdf6144ac0cb7520d2b1830525cd4be1aad1a5b5b598`.
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
  has a closed direct family and terminal GLW and GOOGL timing episodes. Historical
  AMAT and MSFT special negative funding debits matched their declared gross
  dividends within three micro-USDT per matched unit. Because bStock
  receives only the net dividend after deductions, direct pre-adjustment long
  bStock plus short perpetual contributes `N-D=-F<0` before every other cost;
  do not repeat it. GLW's exact Special debit arrived 1.003 seconds after its
  snapshot, so its pre-snapshot premise failed. GOOGL's apparent holiday gap
  also fails because its holiday-adjusted ex-date equals its snapshot date; its
  prior same-underlying Special debit arrived 7 ms after midnight at slightly
  more than gross, and its current public net-distribution floor is zero. Never
  infer ex-date from record date alone, attempt a millisecond race, or request
  books for either terminal episode. Canonical candidate:
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
- The independent TSM recurrence comparator rejects the next declared non-US
  event before any current funding or book request. TSMC's official pages set
  the prior ADR dividend at 0.939325 USD gross and 0.742067 USD after
  withholding, and the upcoming ex-date at `2026-09-16` with an estimated
  1.11 USD gross ADR dividend. Retained TSMUSDT history contains one exact
  `Special` row on the prior ex-date: -0.233910% at a 407.23 mark, debiting a
  matched short 0.952551693 USDT. That already exceeds both prior dividend
  amounts before ordinary funding, execution, basis, fees, and capital costs.
  Do not capture TSM books or funding for the September event. Canonical result
  SHA-256
  `82acc3529620f1d9c728eac24ea0fb256f228e4065650c766dff057d198a5e60`.
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
- A distinct bStocks overlay now extends zero maker fees on all supported pairs
  through `2026-09-30T23:59:00Z`. Credit only an exact otherwise applicable
  account maker fee on an owned qualifying maker fill from independently
  justified organic flow. Bot routing and post-only submission do not prove the
  fill role. Do not create volume, rescue a negative trade, or double-count BNB
  discounts or LP rebates. The announced TradFi mark Price 2 basis-window
  change from 30 to 60 seconds is not income and does not reopen terminal carry
  or parity families by itself; the current Trading Bots guide likewise adds no
  unique Smart Arbitrage subsidy. Canonical triage:
  `binance-aug28-public-structural-trigger-triage-v1-2026-08-29.json`, SHA-256
  `bca11d612042f9a859f53b71e425cd320cca5d4a5d7695cd1f0a0de539b0eea1`.
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
- The independent Round 21 sidecar boundary passed at
  `2026-08-29T23:40:00Z`; no capture process remains and its terminal failure is
  documented below. Do not reopen, reuse, or modify its failed database, WAL,
  lineage, schedule, or capture assets. This preserved failed evidence does not
  block research, staging, commits, or pushes in the main worktree.
- The same-venue `XAUUSDT`/`PAXGUSDT` perpetual funding-basis spread is
  terminal under the retained architecture. The original exact-millisecond
  join kept only 83 rows, but a no-refetch audit proved 500 corresponding
  settlement slots with at most 13 ms skew. Both directions failed training,
  validation, and test after 40 bps execution stress and two 10% annual capital
  hurdles; the training-selected long-PAXG/short-XAU direction netted
  `-296.9983`, `-151.9946`, and `-104.9276` bps. Do not paginate, rerun, or
  refit; reopen only after material funding, index, fee, margin, or product-
  architecture change. Adjudication SHA-256
  `46bf134d1be8b645d7f6272d651be8d3c0b6a8e5b2e7d2b4540f3609d6997a96`.

- Two distinct resolved-source combined events now prove a recurring exact
  duplicate-payoff identity against the standalone 2026 House-control markets.
  The shutdown-final-Yes instance exposed three positive maker-first
  sensitivities after a two-tick hedge, led by `0.03092` pUSD on five shares for
  a combined-Democratic maker role with 500 visible shares ahead, but its pair
  timestamps were 35.484 seconds apart and no owned fill or subsequent hedge is
  proved. The independent ACA-final-No instance stopped the escalation: all
  four all-taker packages and all eight maker-first roles were negative after
  the same stress; its best one-tick-improved role lost `0.11638` pUSD and had
  11.173 seconds pair skew. The payoff identity recurs; positive economics do
  not. Do not spend a source-continuous capture, credentials, or order-capable
  work on this family until a material price, fee, rule, or market change.
  Canonical ACA maker-first result SHA-256
  `900246f7bf066a8d310c6dcc6e9318edc1c6b83d4779f9b2f628c668e7d258e6`.
- The exact-one-NO NegRisk V2 event-log observation consumed its frozen public
  interval and failed HTTP 400 after retaining only the preliminary current-
  block receipt. Because the runner failed to save the HTTP error body or
  completed receipt, the provider cause is unknown and neither conversion
  presence nor absence is proved. Do not retry the interval or substitute a
  provider. Every future one-use HTTP request must durably prejournal method,
  URL, and exact body hash, then retain HTTP error status and body before
  raising. Canonical failure result SHA-256
  `7c976cd84795718b63463ea4e32ebeddaf51e807fc5ebe9aa8cb49b476541e19`.
- Binance BLVT primary-market NAV/spot parity is terminal for the current
  inventory. One public `exchangeInfo` response contained 3,685 symbols and 40
  legacy `LEVERAGED`-permission symbols, but zero with `TRADING` status. Do not
  poll it. Reopen only after a new official BLVT listing or relisting, and then
  require explicit GET-only authority for API-key `tokenInfo` before any book
  study. No credential, account, book, order, subscription, redemption, or
  mutation was used. Canonical result SHA-256
  `85c8ef364b03fb2fbf0aeebddec10d51abbdd608f56ff9c0dccb1835cacc2179`.
- The first exact active WNBA extension of the NBA moneyline/spread monotone-
  payoff mechanism is terminal for Toronto Tempo versus Phoenix Mercury. Three
  exhaustive two-leg packages guaranteed at least one pUSD per share, but one
  synchronized four-token batch found zero positive packages. The best lost
  `0.42436` pUSD at actual five-share depth, `0.62356` pUSD under the frozen
  one-second sensitivity, and `1.02076` pUSD under the three-second
  sensitivity. No fee request was needed because every package was already
  gross-negative. Do not resample this event. For future NBA/WNBA pairs, use
  retained Gamma `outcomePrices` only as a rejection-only optimistic gate and
  stop before books unless at least one displayed package sum is below its
  guaranteed payout. Never accept or promote from Gamma. Canonical book result
  SHA-256
  `cc657982abd9ede0f0f7b18787df32e62c69b7c3b3e547ade3f6f3ccb734ed46`.
- The rejection-only gate then stopped Minnesota Lynx versus Atlanta Dream
  without a CLOB or fee request. Exact retained rules prove Minnesota
  moneyline plus Atlanta +2.5 has a one-pUSD floor, but Gamma's displayed sum
  was `1.080`: an optimistic loss of `0.080` per share and `0.400` at five
  shares before execution costs. This second distinct active WNBA extension is
  terminal absent a material price, fee, delay, or resolution-rule change.
  Reuse the contract-driven exact-event runner rather than creating another
  event-specific metadata collector. Canonical adjudication SHA-256
  `61b3436b3367ba3442ebe777c8a506948243c6d3b6d6a4cb9346d2db3aaf335f`.
- Binance same-expiry four-option CLOB box parity remains terminal. A later
  zero-request audit reused the retained August 27 option catalog and all-
  ticker payload, evaluated 10,382 complete BTC/ETH/SOL box directions, and
  found zero gross-positive rows before fees, margin, legging, settlement, or
  capital costs. The first frozen offline runner failed on 397 ticker rows
  without `closeTime`; its immutable contract and error are preserved. The v2
  repair maps a missing timestamp to an always-unsynchronized sentinel and
  completed without network access. This mechanism was already in the terminal
  registry from the August 25 depth screen, so no new ranked hypothesis was
  created. Search prioritized and terminal registry families before building a
  purportedly novel collector. Do not repeat box parity absent a material
  price, fee, book, or product change. Canonical v2 result SHA-256
  `a9b0e7a2aba9bda7f83b9515be587a17e6da69fa0bc987191a21f9d37e912d3b`.
- Binance BTC/ETH/SOL option/perpetual conversion and reversal parity is
  terminal for the retained August 27 snapshot. Preserve the consumed v1
  timestamp-semantics error: Options ticker `closeTime` was incorrectly treated
  as quote-update time. A separately frozen zero-network v2 used the 1.639-
  second HTTP observation-window skew and found 71 synchronized nominal rows
  across all 1,410 directions. A complete offline stress then applied exact
  quantity lattices, two adverse ticks per leg, 35.5 bips fee/basis cost,
  direction-specific worst retained funding, and a two-notional capital hurdle;
  zero rows survived and no current depth was requested. Do not repeat absent a
  material price, fee, book, funding, margin, or product change. Canonical
  stress result SHA-256
  `c09d62e98cd0df88622d4b98d9d8f01247121ccd786fffb580bc72429ef6bf30`.
- The first exact NFL extension of the sports moneyline/spread monotone-payoff
  family is terminal for Packers versus Vikings. One exact public Gamma event
  response retained 34 active markets. A zero-network adjudication proved all
  321 within-family threshold relations, including NFL tie and cancellation
  states; four passed the rejection-only Gamma gate. The strongest package was
  Packers +0.5 plus Vikings moneyline at a displayed `0.895` sum for a one-pUSD
  floor. One frozen exact two-token book batch then found asks of `0.55` and
  `0.53`: five shares cost `5.400` pUSD, a `0.400` pUSD loss before fees, while
  book timestamps were 10,049,940 ms apart. Zero fee requests were spent. Do
  not resample this game. Gamma may reject but never accept or promote; require
  a distinct event to survive exact synchronized depth and current fees before
  escalation. Canonical exact-depth result SHA-256
  `731ca32a06f8f1a42aaae9e326c2bd89379657e338231dd906b749790c15ddfa`.
- One complete future-NFL keyset request then covered the distinct
  `2026-09-13T20:25:01Z` through `2026-09-21T23:59:59Z` window: 17 events
  returned, 16 exactly parsed, 4,621 relations retained, and 674 Gamma-only
  rejection-gate candidates. The precommitted strongest Commanders/Cowboys
  Over 56.5 plus Under 58.5 row displayed `0.785`, but exact asks cost `7.55`
  pUSD for five shares against a five-pUSD floor, a `2.55` pUSD zero-fee loss;
  book timestamps were 25,189,367 ms apart and zero fee requests were spent.
  A frozen offline Cowboys/Giants correction handled the shared-threshold
  moneyline/minus-0.5 actual-tie distinction, retained 268 relations and four
  more Gamma candidates, but granted no adaptive depth. Do not resample this
  window. Canonical catalog, depth, and correction result SHA-256 values are
  `7c4472e0a77cde09f5643a06a1326fbfc2cc1e5ec37641314d875a346e1a7754`,
  `729d482f9a15b60b5345ba6c52ee75941a1f0751db2453e307c30f8872bbac35`,
  and `37f79cc8a4f5f96fa395a729e85a793e12c2127e2124591db693c92b1b459928`.
- A distinct all-category near-expiry fixed NegRisk screen requested the
  documented 500-event keyset maximum but received 100 events plus a cursor,
  so it is explicitly incomplete. It retained all 100 classifications, 49
  fixed event screens, and 24 Gamma-only sub-floor all-YES rows. The lowest was
  the 17-outcome Lyon/Le Havre exact-score event at `0.9450`, but the frozen
  contract set `proof_candidate=null` and spent zero on-chain, book, and fee
  requests. Do not continue the consumed cursor or depth-test its outcome-aware
  winner. Canonical result SHA-256
  `96610d7cba90a2dc97489bd70c95b7d03568d5b89017ace1e8c92829c70cee14`.
  A distinct single-day `2026-09-06` UTC window also returned 100 plus cursor,
  with 57 fixed events and only two Gamma candidates led by Dallas/Sporting
  Kansas City at `0.985`; it likewise spent zero proof requests. Daily all-
  category windows are still incomplete. Retry only with a prospectively fixed
  hourly or series-specific window, never repeated outcome-aware narrowing.
  Canonical follow-up result SHA-256
  `3e3ae8fd8c98c93c3e2194425db5992f06aed412e07d32333525601c2b34bc52`.
- The exact non-crypto Polymarket/Binance TradFi-perpetual funding family is
  terminal for the frozen current top five. A four-request rejection prefilter
  joined 36 exact current instruments and advanced SKHYNIX, CRWV, ARM, HOOD,
  and MSTR without books. The 15-request bounded history plus a separately
  frozen zero-network shortfall adjudication then found every fixed orientation
  failed training, validation, and test after 20 bips execution and two 500-bip
  annual capital hurdles. ARM was least negative at `-24.6296621005` bips over
  the full retained window and below `-20.68` bips in every role. No conversion,
  book, credential, account, order, or protected-capture asset was accessed.
  Do not repeat absent a material funding cash-flow, fee, session, instrument,
  conversion, or execution-architecture change. Canonical result SHA-256
  `5e67277ad30b9f0164a3987804162ed2d1cdabb820e7e258d6a0b79748cf7d06`.
- Binance indirect internal conversion routing is now accepted only as a
  fail-closed cost-saving overlay for an independently required legitimate
  same-account organic Spot conversion. The complete current public graph
  evaluated 10,440 direct-versus-one-intermediary routes at 100 and 1,000 USDT.
  Twenty route-size rows passed conservative 10-bip-per-leg fees, exact lot
  rounding, displayed capacity, residual valuation, 3-bip extra-leg stress,
  and 60-sample recurrence; a separately frozen quote-change and 24-hour
  activity gate retained only USDC-to-PNUT through USDT and two SHIB-to-EUR
  diagnostics. No static route is accepted. Never create volume, use public
  VIP-0 fees as account fees, or treat the two-minute screen as a fill claim.
  Canonical adjudication SHA-256
  `0307a9dbfb26ca62e94ae01e5b5d40316340b686a60829e85f258c07e565678c`.
- The zero-network exactly-two-intermediary extension is terminal for the same
  retained population. It exhaustively covered 1,064,216 routes and 2,128,432
  route-size rows. A lossless optimistic bound reduced exact work to 16,051
  rows; exact rounding, capacity, residual, fee, extra-leg stress, and recurrence
  left 253 apparent incremental candidates versus the best direct-or-one-
  intermediary comparator. Every candidate failed the retained five-change
  activity gate, and 18 also failed the 100-trade gate across 174 required
  symbols. Do not build a three-leg live collector or promote the stale TRY-
  heavy rows. Reopen only after a material fee, filter, activity, batching, or
  execution-architecture change. The first exhaustive implementation was
  stopped before output because its per-row summaries consumed about 847 MB;
  the second failed closed on a missing finite residual-valuation source gate.
  Both are preserved, were not rerun, and caused the topology/output-size rule
  in `AGENTS.md`. Canonical final exact and activity result SHA-256 values are
  `0a5e37f2fb48c639334256e3118e3eeb2f17a548572faaf13d3849204404b45e`
  and `cde72e05b1760d9fe23eb65e5bd5f59377230ac91095354936c2a84a9a3758ae`.
- A complete-for-cutoff primary-literature delta reviewed seven novel August
  papers and found zero actionable structural leads. A separately frozen source
  audit of arXiv `2607.09491v1` strengthens the existing one-intermediary
  mechanism but does not reopen a collector. The paper inferred 402 million
  two-trade Binance sequences from anonymized 2017-2023 fills and estimated
  31.2 million USD after assumed 1.2/2.4-bip fees, but mean sequence profit was
  only 0.15 USD maker/taker and 0.04 USD taker/taker. Its comparator was a two-
  second VWAP, not synchronized depth, and exact fees, books, failed attempts,
  fixed infrastructure, ownership, and unwind costs were unavailable. This is
  the already-covered direct-versus-one-intermediary identity, not evidence for
  the rejected three-trade extension. Canonical source adjudication SHA-256 is
  `3f9684ed1986cd6cf676482069cda53846e336a15bc4b35141193b8e43406e65`.
- The standalone statistical-spread interpretation of SSRN `6453880` is now
  separately triaged from the accepted organic-conversion saving. A retained
  independent replication reports that 2026-H1 capacity-constrained PnL fell
  about 90% from 2025, every one of the 42 profitable 2025 paths supported at
  most 25 USD per trade under its five-percent volume rule, 78% of intended
  trade minutes had no executable volume on the thinnest leg, and zero of 86
  profitable routes had perpetuals on every required leg. Its only scalable
  BTC-FDUSD and ETH-BTC examples both lost money. The review exposes no code,
  exact route outputs, books, owned fees, or fills, so it proves neither current
  profit nor a complete falsification. Do not build another minute-bar route-
  spread collector without reproducible route-level evidence or a material fee,
  instrument, or executable-capacity change. Canonical triage SHA-256 is
  `ba9d063f78b027f6aab5e45723f5dc4ea2e9df1303de4f493e780ee07d4425b7`.
- Binance closed three-asset Spot cycles are terminal for the current retained
  complete public graph. One frozen zero-network pass covered all 3,480 unique
  directed cycles, 1,442 activity-qualified cycles, 60 complete books, 100 and
  1,000 USDT sizes, zero-fee and VIP-0 scenarios, exact lot rounding, displayed
  capacity, residual value, and 3-bip stress. Zero rows passed even the recurring
  zero-fee upper-bound gate. Do not resample or query account fees absent a
  material fee, filter, batching, atomic-execution, or market-structure change.
  The consumed diagnostic top-100 label overstated its feasibility ordering;
  the sort occurred after candidate counting and did not change the terminal
  decision. Canonical adjudication SHA-256
  `2fffd2044e72d1712ecdaa0c4e24cb829057ea2005c07e12129c443478b07902`.
- One distinct complete WNBA window from `2026-08-30T19:00:01Z` through
  `2026-09-02T23:59:59Z` returned three events. Two exact moneyline/spread
  packages were provable, but Golden State/Portland displayed `1.165` and
  Connecticut/Dallas `1.365` against a one-pUSD payout floor. The third event
  had no exact relation. The rejection-first gate therefore spent zero book and
  fee requests. Do not resample this window or narrow around its outcomes.
  Canonical result SHA-256
  `fd0a9e844a7ad7d1a6eb5372c961ff82ea52d3c72a8c558ba191a53bace02cef`.
- One prospectively fixed non-overlapping NegRisk hour from
  `2026-09-07T00:00:00Z` through `00:59:59Z` returned six events without a
  cursor and therefore a complete frozen population. Five were fixed,
  non-augmented NegRisk sets; none had a Gamma all-YES sum below one pUSD.
  The best, Nacional Potosi/Blooming match winner, summed to `1.090` before
  execution costs. The rejection-first gate therefore spent zero onchain,
  book, fee, account, order, or fund requests. Do not sweep adjacent hours or
  narrow around these outcomes. Canonical result SHA-256
  `92734472ed41bccdc1d88c947b218e05fa35827cad6b1711ec192c06cf60cc64`.
- A price-blind official series catalog selected `BTC Up or Down 5m` without
  embedded events or prices, but its fixed September 14 through September 21
  event window returned zero rows. A distinct retained-metadata selection of
  series 45, `BTC Multi Strikes Weekly`, also returned zero events in its fixed
  September 7 01:00 through September 21 window. Both catalogs were complete
  and used zero onchain, book, fee, credential, account, order, fund, or
  protected-capture requests. Series recurrence, title, update time, and volume
  are not forward-deployment evidence. Do not repeat either empty query. Reopen
  only on an explicit new fixed-NegRisk event deployment or not before
  `2026-09-20T00:00:00Z` for one distinct September 22 through September 28
  series-45 discovery. Canonical result SHA-256 values are
  `73ae75ccf391a30ef592f649f63ef535e29865e1234bbff03021c806fd75268b`
  and `f032753b45c82b2e0945d1a8c0e0d5fc01f8fb1727cdad34e73064c7590417ba`.
- The structural-edge registry now has 44 ranked hypotheses and 21
  narrowly accepted scoped overlays, and result SHA-256
  `0a34d7289331515f8e7b3f09e856fbc331ecbc3a91130fea20542a39ef211f60`.

## Binance RPI maker-hedge source gate

The distinct USD-M RPI execution architecture is not an accepted edge. The
retained native index proves that `commissionRate` is account `USER_DATA` and
that an RPI order-book endpoint exists, but the frozen primary documentation
GET returned HTTP 202 with an empty body before any market data. Do not retry
the page, switch aliases, or request `rpiDepth` without exact account RPI
commission and an independently positive organic equal-base hedge question.
Visible depth never proves an owned fill, and every RPI or hedge order requires
separate authority. Canonical failure result SHA-256 is
`82245f341e23ab2e8c8e9e3bd4d47805e88aebc3b39f6ac1a6360067491be7ef`.

For dynamic documentation, never assume that browser rendering means a direct
HTTP client will receive the rendered contract. Prefer a hash-bound native
index or preflight byte-retainability before freezing an outcome-sensitive
source capture; HTTP 202 with zero bytes is a consumed null response.

## Accepted market-independent yield frontier

The canonical zero-network frontier for all nine accepted yield and capital-
efficiency overlays is
`docs/model-research/action-value/accepted-market-independent-yield-frontier-v1-2026-08-30.json`,
result SHA-256
`7a7fa5ed15ab63bfd0c4d5d2ce65888391a72c4e73eea69e7f7c1fcf01a13fb8`.
It excludes the thirteen accepted organic-flow fee, referral, creator, and
financing-cost overlays and adds no accepted edge.

Polymarket complete-set holding yield remains strongest on realized stability:
42 of 42 positive daily payments across BTC, ETH, and SOL, 1,039 pUSD
demonstrated principal, and 3.1820191118% realized annualized portfolio yield.
It remains positive through a 3% alternative yield and negative at 3.25%
before external friction. Binance LDUSDT is the strongest long-history
incremental overlay, but only for collateral already required by an independent
futures strategy. USD1 Simple Earn is the strongest fixed current bonus
allocation only for at most 1,500 already-held idle eligible USD1. None of the
nine rows is an acquisition edge or deployment-ready.

## pUSD-to-USDT Fixed-Bonus Opportunity-Cost Rejection

One frozen four-request public sequence retained the native Markdown contracts,
current supported-assets population, and one exact 500 pUSD to Polygon USDT
quote. The quote estimated 492.459811 USDT output: a 7.540189-USDT or
150.80378-bip optimistic one-way loss. Through the fixed bonus end, the entire
remaining 4% bonus can return at most 0.4856127333 USDT or 9.712254665 bips;
after the realized 3.1820191118% pUSD holding-yield opportunity cost, its
incremental headroom is only 0.0993054837 or 1.986109674 bips. The quoted loss
is therefore 15.527 times the full bonus and 75.929 times its incremental
advantage before Binance deposit, return conversion, eligibility, capacity,
custody, tax, or operating costs.

This rejects only acquiring USDT from pUSD for the capped promotion. It does
not reject the accepted bonus overlay on independently already-held idle
eligible USDT. Do not repeat the quote without a material bridge-quote or bonus-
term change. No credential, account, address generation, order, subscription,
approval, transfer, transaction, fund, or protected capture was accessed.
Canonical result SHA-256 is
`98b74abfcb213a8d1bd554fc1bfec9044d6a6a3990abd800bef92f29437533b3`;
the registry SHA-256 is
`8ef8e033e169084a57237321d13467f05486152c2b713cc478023372efc6b877`.

## Binance Spot Amend-Keep-Priority Candidate

Official retained evidence now proves a direction-independent execution
candidate for an independently required reduction of an existing simple Spot
maker order. A successful in-place quantity reduction keeps the same-price time
priority; cancel-replace loses that priority and moves behind existing orders.
Failed amendments leave the order unchanged, each amendment adds zero to the
unfilled-order count, and its request weight is four. The retained production
exchangeInfo snapshot has BTCUSDT, ETHUSDT, and SOLUSDT trading with
`amendAllowed=true` and `MAX_NUM_ORDER_AMENDS=10` for each.

This is not accepted, cash-valued, or deployment-ready. Public evidence proves
no independently required owned reduction, queue counterfactual, incremental
fill, success acknowledgement, adverse selection, latency, commission, or
after-cost value, so its public forward profit floor is zero. The amendment
endpoint is `TRADE`; do not call it without separate explicit testnet or paper
order authority. Reopen only for an existing scoped maker order that already
needs a same-price quantity reduction, or after a material semantics, weight,
filter, or production-configuration change.

The one source request was retained but its frozen raw-Markdown phrase gate
failed on bold and inline-code delimiters. It was not retried; the exact bytes
were adjudicated offline and the reusable Markdown-gate rule was corrected.
Canonical adjudication SHA-256 is
`c17217ff011d6ff48b4c0cf48cc6c8e49c27319c811be407cadebdd2e8d7faeb`;
the current registry SHA-256 is
`e9e25be7ae77d25c2f98b30b734c64f85259e73059405e938be77c776bd0a066`.

## Task Routing

| Work | Read first |
| --- | --- |
| Current plan or handoff | `docs/CONTINUATION.md` |
| Binance model/backtest | `docs/model-research/action-value/latest/README.md` |
| Polymarket model | `docs/model-research/polymarket/latest/README.md` |
| Structural parity | `structural_parity.py`, `logical_parity.py`, and the three 2026-08-25 snapshots |
| Structural edge priorities | `docs/model-research/structural-edge-priority-registry-v1.json` |
| Binance all-symbol triangular-cycle terminal adjudication | `docs/model-research/action-value/binance-all-symbol-triangular-cycle-retained-adjudication-v1-2026-08-29.json` |
| Binance indirect organic-conversion route savings | `docs/model-research/action-value/binance-indirect-internal-conversion-activity-adjudication-v1-2026-08-29.json` |
| Live NBA moneyline/spread implication candidate | `docs/model-research/action-value/polymarket-live-nba-moneyline-spread-combinatorial-parity-reopen-v1-2026-08-26.json` |
| Cross-market exact dependent-subset parity candidate | `docs/model-research/action-value/polymarket-cross-market-dependent-subset-parity-reopen-v1-2026-08-26.json` |
| Current sports exact-title discovery terminal result | `docs/model-research/action-value/polymarket-current-sports-monotone-pair-discovery-result-v1-2026-08-29.json` |
| Current BOS/NYY monotone-parity terminal adjudication | `docs/model-research/action-value/polymarket-current-mlb-monotone-parity-failure-adjudication-v1-2026-08-29.json` |
| Current LAD/DET rejection-only monotone adjudication | `docs/model-research/action-value/polymarket-lad-det-monotone-prefilter-adjudication-v1-2026-08-29.json` |
| Current LAD/DET cross-period total terminal result | `docs/model-research/action-value/polymarket-lad-det-cross-period-total-result-v1-2026-08-29.json` |
| Future MLB cross-period partial-page adjudication | `docs/model-research/action-value/polymarket-future-mlb-cross-period-catalog-adjudication-v1-2026-08-29.json` |
| Current Toronto/Phoenix WNBA monotone-parity terminal result | `docs/model-research/action-value/polymarket-current-wnba-monotone-parity-result-v1-2026-08-29.json` |
| Current Lynx/Dream WNBA rejection-only terminal adjudication | `docs/model-research/action-value/polymarket-lynx-dream-monotone-prefilter-adjudication-v1-2026-08-29.json` |
| Future WNBA complete catalog rejection screen | `docs/model-research/action-value/polymarket-future-wnba-monotone-catalog-result-v1-2026-08-29.json` |
| Current Packers/Vikings NFL exact-depth terminal result | `docs/model-research/action-value/polymarket-packers-vikings-tie-state-package-result-v1-2026-08-29.json` |
| Future NFL complete catalog rejection screen | `docs/model-research/action-value/polymarket-future-nfl-monotone-catalog-result-v1-2026-08-29.json` |
| Commanders/Cowboys NFL exact-depth terminal result | `docs/model-research/action-value/polymarket-commanders-cowboys-total-package-result-v1-2026-08-29.json` |
| Cowboys/Giants NFL tie-collision correction | `docs/model-research/action-value/polymarket-cowboys-giants-tie-collision-correction-v1-2026-08-29.json` |
| Near-expiry fixed NegRisk incomplete complete-set catalog | `docs/model-research/action-value/polymarket-near-expiry-negrisk-complete-set-catalog-result-v1-2026-08-29.json` |
| Sep 6 fixed NegRisk incomplete daily catalog | `docs/model-research/action-value/polymarket-sep6-negrisk-complete-set-catalog-result-v1-2026-08-29.json` |
| Sep 7 hour-00 fixed NegRisk complete rejection catalog | `docs/model-research/action-value/polymarket-sep7-hour00-negrisk-complete-set-catalog-result-v1-2026-08-29.json` |
| Binance retained CLOB box-parity terminal prefilter | `docs/model-research/action-value/binance-options-clob-box-retained-prefilter-v2-2026-08-29.json` |
| Binance retained option/perpetual conversion terminal stress | `docs/model-research/action-value/binance-options-perpetual-conversion-retained-stress-v1-2026-08-29.json` |
| Polymarket holding-yield latest retained receipt continuity | `docs/model-research/polymarket/complete-set-holding-yield-continuity-receipts-v8-2026-08-29.json` |
| Resolved-leg House duplicate-payoff recurrence | `docs/model-research/action-value/polymarket-aca-house-maker-first-candidate-v1-2026-08-29.json` |
| Binance XAU/PAXG funding-basis terminal adjudication | `docs/model-research/action-value/binance-xau-paxg-perpetual-funding-spread-failure-adjudication-v1-2026-08-29.json` |
| Polymarket exact-one-NO V2 conversion failure adjudication | `docs/model-research/action-value/polymarket-negrisk-one-no-v2-conversion-failure-adjudication-v1-2026-08-29.json` |
| Binance BLVT current primary-market NAV-parity terminal gate | `docs/model-research/action-value/binance-blvt-primary-market-nav-parity-public-gate-v1-2026-08-29.json` |
| Polymarket/Binance TradFi-perpetual funding terminal adjudication | `docs/model-research/action-value/polymarket-binance-tradfi-perps-funding-history-shortfall-adjudication-v1-2026-08-29.json` |
| Binance idle-stablecoin Launchpool reward candidate | `docs/model-research/action-value/binance-stablecoin-launchpool-idle-inventory-reward-candidate-v1-2026-08-26.json` |
| Polymarket Combo RFQ versus CLOB Boolean parity candidate | `docs/model-research/action-value/polymarket-combo-rfq-boolean-parity-candidate-v1-2026-08-27.json` |
| Polymarket Combo exposure-preserving collateral-release overlay | `docs/model-research/action-value/polymarket-combo-collateral-release-overlay-candidate-v1-2026-08-29.json` |
| Polymarket terminal broad sports Combo requester-overround validation | `docs/model-research/action-value/polymarket-combo-maker-overround-validation-v1-2026-08-27.json` |
| Binance bStock dividend/perpetual funding timing-gap candidate | `docs/model-research/action-value/binance-bstock-dividend-perp-funding-timing-gap-candidate-v1-2026-08-27.json` |
| Binance bStock Spot LP all-symbol rebate overlay | `docs/model-research/action-value/binance-bstock-spot-lp-all-symbol-rebate-overlay-candidate-v1-2026-08-27.json` |
| Binance bStocks zero-maker carry retained counterfactual | `docs/model-research/action-value/binance-bstocks-zero-maker-carry-retained-counterfactual-v1-2026-08-29.json` |
| Binance existing-stock transfer reward overlay | `docs/model-research/action-value/binance-existing-stock-transfer-reward-overlay-candidate-v1-2026-08-27.json` |
| Binance Stocks FPSL existing-inventory yield overlay | `docs/model-research/action-value/binance-stocks-fpsl-existing-inventory-yield-overlay-candidate-v1-2026-08-27.json` |
| Binance NOK bStock dividend/perpetual under-debit candidate | `docs/model-research/action-value/binance-nok-bstock-dividend-perpetual-underdebit-candidate-v1-2026-08-27.json` |
| Binance TSM bStock dividend/perpetual recurrence rejection | `docs/model-research/action-value/binance-tsm-bstock-dividend-underdebit-v1-2026-08-30.json` |
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
| Binance August 28 structural-trigger triage and accepted bStocks zero-maker overlay | `docs/model-research/action-value/binance-aug28-public-structural-trigger-triage-v1-2026-08-29.json` |
| Binance VIP 6 for Six organic-fee overlay candidate | `docs/model-research/action-value/binance-vip6-for-six-organic-fee-overlay-candidate-v1-2026-08-27.json` |
| Lite Loan and fixed-bonus stablecoin yield curve | `docs/model-research/action-value/binance-lite-loan-stablecoin-yield-curve-v1-2026-08-27.json` |
| U Flexible idle-holding yield | `docs/model-research/action-value/binance-u-flexible-idle-holding-yield-gate-v1-2026-08-26.json` |
| Existing RWUSD VIP bonus overlay | `docs/model-research/action-value/binance-rwusd-existing-vip-bonus-overlay-gate-v1-2026-08-26.json` |
| Current USDT Flexible bonus overlay | `docs/model-research/action-value/binance-usdt-flexible-current-bonus-overlay-v1-2026-08-26.json` |
| Existing USDe automatic holding reward | `docs/model-research/action-value/binance-usde-existing-holding-reward-edge-v1-2026-08-26.json` |
| Organic third-party Polymarket builder-fee overlay | `docs/model-research/action-value/polymarket-organic-third-party-builder-fee-overlay-v1-2026-08-26.json` |
| Polymarket organic Relayer gas subsidy and Builder-tier reward gate | `docs/model-research/action-value/polymarket-organic-relayer-gas-subsidy-and-builder-tier-reward-v1-2026-08-29.json` |
| BFUSD reward-unit conflict gate | `docs/model-research/action-value/binance-bfusd-existing-holding-reward-unit-conflict-gate-v1-2026-08-26.json` |
| Binance Smart Arbitrage terminal adjudication | `docs/model-research/action-value/binance-smart-arbitrage-terminal-family-adjudication-v1-2026-08-26.json` |
| Organic Polymarket referral net-fee overlay | `docs/model-research/action-value/polymarket-organic-referral-net-fee-overlay-v1-2026-08-26.json` |
| Binance Flexible Loan collateral-yield gate | `docs/model-research/action-value/binance-flexible-loan-simple-earn-collateral-yield-gate-v1-2026-08-26.json` |
| Binance Flexible Loan same-asset loop adjudication | `docs/model-research/action-value/binance-flexible-loan-same-asset-loop-adjudication-v1-2026-08-30.json` |
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

## Binance Spot SOR production-configuration terminal screen

The frozen public SOR screen consumed one unauthenticated production
`exchangeInfo` request. Binance's current documentation defines Smart Order
Routing and names the optional `sors` field as its live configuration source,
but the retained 17,532,885-byte response covering 3,685 symbols omitted that
field. The runner therefore stopped with zero scoped groups before the optional
all-symbol book request. It used no credentials, signed endpoint, account,
order, funds, or protected Polymarket state.

This exact production snapshot has no public gross SOR candidate. Do not poll
or proceed to a signed test merely because the feature is documented. Retry
only after a material official or live `exchangeInfo` SOR-configuration change.
The raw response is retained losslessly as deterministic gzip; its decompressed
SHA-256 is bound to the response journal and result receipt. The contract and
result SHA-256 values are
`93b9f984ff4ae347cc6ca15b9a79e5f9baf60ce3287e3e81ce5694613be0790a`
and `895dc0eba4f72b9b08b19dbba245b20434e4db905fd4ded3ea70779733db6d47`.
The registry remains at 44 hypotheses and 21 accepted scoped edges; its updated
result SHA-256 is
`ff5b41b572833ff0eed459098a2f93d1d62fed03891616b9a1fa71bc832f887e`.

## Polymarket official-documentation novelty gate

One frozen public unauthenticated request retained the complete current
`llms.txt` index before any linked page or market-data access. Its 80 top-level
English Markdown pages add zero distinct economic mechanism outside the
registry. Reward, fee, resolution, position, Combo, and Perps pages map to
existing families. Session keys, matching-engine restarts, bridge routes, mark
price, index price, and market sessions are operational or risk constraints,
not standalone cash flows. In particular, Combo collateral return was already
retained in rank 33 and cannot be promoted again under a new name.

Do not re-fetch this exact index or open linked pages from it. Retry only if the
official `llms.txt` SHA changes from
`68256fa9849e72626806cbc7373f726421fd6d62dddecc0ae3a8009595bd2b8d`
and an offline title or description diff identifies a genuinely new cash flow,
cost reduction, settlement right, or executable package. Canonical result
SHA-256 is
`e56ce8f2a491d6da3f66b0d085381894ec0e7c078e4f4cfaf238d8e044fa281a`.
The registry remains at 44 hypotheses and 21 accepted scoped edges; its result
SHA-256 is
`ff5b41b572833ff0eed459098a2f93d1d62fed03891616b9a1fa71bc832f887e`.

## Polymarket Combo collateral-release overlay

Current official documentation proves a distinct direction-independent
capital-efficiency mechanism for independently existing compatible Combo
positions. Collateral return decomposes offsetting positions, merges only their
complementary exposure into pUSD, and leaves unmatched economic exposure in
residual positions. The retained page is byte-identical to the source hashed on
2026-08-27, but this checkpoint asks the previously unadjudicated capital-release
question rather than treating the page only as RFQ plumbing.

This is a material rank-33 candidate, not an accepted edge. `net_pusd_out` is
released principal, not profit. Public evidence proves no owned compatible
positions, exact plan output, monetary meaning of `estimated_cost`, universal
gasless guarantee, successful transition, or positive after-cost use of the
released pUSD. At an illustrative 3.25% annual comparator, 1,000 pUSD released
for 30 days can justify at most 2.6712328767 pUSD of total execution and
operating cost; for 90 days the ceiling is 8.0136986301 pUSD. These are
sensitivities, not qualified current returns.

Only when independently existing owned compatible positions, explicit
account-specific plan-request-only authority, and an exact positive use of
released pUSD all exist may one plan be requested without execution. Require
positive `net_pusd_out`, zero `required_pusd_input`, complete consumed-created
residual lineage, bounded chunks, exact monetary costs, and a strictly positive
conservative inequality. Every approval, signature, submission, transaction,
poll, or retry needs separate explicit authority. Canonical result SHA-256 is
`5514bd931557b350579a07448db9c4e1f2664919efff48145861c8841f0bc7ea`.

## Polymarket organic Relayer gas-subsidy overlay

One frozen, public, unauthenticated request retained the current official
Builder-tiers page. It proves that Polymarket subsidizes gas for supported
smart-wallet Relayer operations: the Unverified tier currently allows 100
transactions per day, Verified allows 10,000, Partner is unlimited, and the
own-wallet FAQ describes an unlimited Relayer-key route. This is a material
direction-independent cost-reduction candidate only on an independently required,
otherwise-positive Deposit, Safe, or Proxy wallet operation. It cannot justify
creating, splitting, retrying, or rerouting transactions, and its value is zero
when no independently required activity exists.

The public page also lists weekly USDC rewards and grants for Verified builders,
but both are subject to approval and publish no rate, threshold, formula, cap,
timing, or guaranteed award. Their public forward floor therefore remains zero;
rate limits, support, marketing, and priority access are not cash edges. No
credential, account, order, profile, key, signature, transaction, fund, or
protected-capture state was accessed. The candidate is not accepted or
deployment-ready: the frozen source contract forbids source-only promotion, and an
exact active key, wallet type, remaining tier capacity, owned successful receipt,
counterfactual same-action gas cost, and every setup and operating cost remain
account-gated. Contract/result SHA-256 values are
`87ff005cd29184501aa9bb17d450a1112f0f8324fe9ad7bd248c375f30b7698f`
and `1c33e778d217ec5e7ef817e83af3186df7da0e5dd0cc75cb72464bfd97d18d49`.
The registry remains at 44 hypotheses and 21 accepted scoped edges; its
canonical SHA-256 is
`0a34d7289331515f8e7b3f09e856fbc331ecbc3a91130fea20542a39ef211f60`.

## Binance Stocks extension capture and Round 21 sidecar terminal state

The official Binance Stocks fee-extension trigger was observed, but its frozen
one-use CMS capture failed locally with curl exit `23` before any response body
or HTTP status was retained. Do not retry that request or use an endpoint alias
as a repair. The accepted overlay's prior duration remains unchanged; canonical
failure SHA-256 is
`498003b3f593cda600570099f9089bcb0db0a189e5c92c87e9395bd2afeb3ed8`.
All one-use file-backed runners must verify writable raw and journal parent
directories before network access.

The Round 21 Binance sidecar is no longer protected or running; its scheduled
boundary passed. Its terminal metadata contains 17 segments: 16 interrupted and
one failed during `finish_run` at the frozen DuckDB memory limit, leaving zero
eligible segments. The database exists at the legacy sidecar path, has no WAL,
and was not opened during the audit. The terminal-manifest command correctly
rejected the campaign, so source continuity, model eligibility, and profitability
all remain false. Never reuse or rerun this campaign. Continue only through the
venue-separated prospective source-continuity recovery design. Canonical
terminal-failure SHA-256 is
`9e6790644a566dcfd6e786442a8da3a63c8837f991e766b418fea0df90d0cc8e`.

## Binance Spot Block Matching cost overlay

Current official sources prove a direction-independent off-book execution
candidate for an independently required large bilateral Spot trade. The
current CMS FAQ charges both maker and taker 2.5 bips, offers no market-maker
rebate, permits the creation price within 10% of market, and settles immediately
to a whitelisted master Spot account. Current API documentation makes symbols
and order-history operations signed `USER_DATA`; place, take, cancel, and extend
are `TRADE` operations.

The fetched general-information page embedded credential-shaped authentication
examples. Its raw body is deliberately excluded from version control; its
original response hash is preserved only in the journal and candidate retention
exception. No example value is reproduced or used as evidence, and the FAQ,
introduction, and Agent Native index cover the admitted terms.

This is not accepted or deployment-ready. The exact negotiated price dominates
the small fee saving, and no whitelist, supported pair/minimum, authentic
counterparty, account Spot fee, same-time finite-size book, failure cost, or
owned fill is proved. At 100,000 quote notional, fee-only savings versus 10,
7.5, and 5-bip Spot fees are 75, 50, and 25 quote units; the saving is zero at
2.5 bips and negative on a zero-fee pair. Never contact VIP coverage or a
counterparty, request whitelisting, relay a settlement key, or act on an order
without separate explicit authority. Reopen only under rank 5's exact
read-only prequalification trigger, and require separate trade authority for
every order operation. Canonical candidate SHA-256 is
`2d9c4872a6ecd707716cd8d769eb20cb715c1ed7feb61e7e560a5efdb169dc57`.

The separate low-value-asset documentation request was consumed by a WAF
challenge with `HTTP 202` and zero body bytes. It proves no edge and may not be
retried or repaired through an alias. Canonical failure SHA-256 is
`e2fc005dda76af1a6aad7eb29ca09db19023f0aaa5ae64eff944cc0ef75ee48a`.
The registry remains at 44 hypotheses and 21 accepted scoped edges; its
canonical SHA-256 is
`0416a75158adf12ca08e6dc2d529efa29db53300715ba06f7c04378dcfa2a396`.

## Binance public Spot block-trade price preflight

Official current sources prove the production `<symbol>@blockTrade` market
stream is real-time, unauthenticated on the market-data-only domain, and exposes
exact price, quantity, trade time, event time, block ID, and buyer-maker
identity. The one-use public BTCUSDT/ETHUSDT/SOLUSDT capture completed for
1,205.0309999999954 seconds with zero reconnects and 3,368 ordinary ticker
events, but zero block-trade events. Consequently it produced zero causal
price-concession rows and failed the frozen recurrent-observation gate.

Do not repeat or expand this public preflight before
`2026-08-31T00:28:14.5427794Z` unless a material official stream, fee, pair,
whitelist, settlement, or block-volume change occurs. This exact zero-event
window does not prove the stream has no future activity and does not reject the
parent account-gated mechanism, but it leaves the public forward profit floor
at zero. Canonical preflight SHA-256 is
`b7d60e0d9f3e30b2a62663ff1290be77e6309ac33a7d48776b6f5ea1c8dcfe68`;
registry SHA-256 is
`6e0c9d33e909ec980af5fa65d8ed2cdaebd8dd3fa576671165fe0a331f7af817`.

## Binance Convert Limit Order Simple Earn overlay

Current official terms prove that an independently required Convert Limit Order
can optionally keep its source asset in Simple Earn Flexible while waiting,
instead of freezing the same funds without rewards in Spot. This is a material
direction-independent rank-3 capital-efficiency candidate, but it is not
accepted. Subscription quota applies, cancellation does not redeem the asset,
and delayed or failed redemption prevents execution; a limit-price touch also
does not guarantee a fill.

The public forward floor is zero because the feature FAQ binds no exact
BTC/ETH/SOL rate, tier, cap, accrual start, distribution schedule, account quota,
or guaranteed redemption. At the retained BTC `0.27% Max`, seven days amount to
only `0.5178082` bips before costs and one bip of missed-fill loss needs
`13.5185` days to recover; the `0.02%` base component needs `182.5` days. Do not
create or extend a conversion to chase this reward. Advance only under the exact
rank-3 read-only trigger, and require separate authority for every order,
subscription, redemption, cancellation, transfer, or trade. Canonical candidate
SHA-256 is
`3b33ca4ef8c03a609bef1665ccfc2104a3f6585033770f0bb99ec3c5699949f8`;
registry SHA-256 is
`546904123e0985aa23d7f3c58567dd2b8e877681b48e560512e4b15b9082721b`.

## Exact Polymarket reward candidate rejection

The latest distinct rank-17 retry exposed and corrected a methodology error:
the initial contract used a stale or misattributed search discovery tuple as a
hard gate even though exact Gamma and sponsored-condition sources agreed on
50 shares and 5.5 cents. A frozen retained-source correction made only the
previously unrequested book call. It failed the 5-second freshness gate at
6,408 ms, and one-tick bids of 0.48 plus 0.53 were both marketable, summed to
1.01, and lost 0.50 pUSD if both filled. A frozen offline best-bid join retained
positive 0.50 pUSD both-fill gross but 26 pUSD orphan risk; even observed book
competition needed 43.554 reward days versus 3.693 remaining, while 100-times
competition needed 4,306.850 days. Do not retry this market. Accepted edges
remain 21. Canonical rejection SHA-256:
`facecfaa3b92d905c700083c7b8afe153adc495403ceabc91e417bdb248d059b`;
registry SHA-256:
`de28d80cc4b0b9cd1bd3f9954cb840dcaefe46fcc0fdf9ac3fd53218169370cb`.

## Binance public Spot block-trade follow-up

Rank 5's literal nonoverlapping time trigger fired. A second frozen public
BTCUSDT/ETHUSDT/SOLUSDT observation reused the existing hash-bound official
stream and fee sources, made no documentation or account requests, and ran for
305.0 elapsed seconds with zero reconnects. It retained 732 healthy ticker
messages but zero `blockTrade` events, analyzable rows, or price concessions.
Together, the two complete windows now cover 1,510.0309999999954 seconds and
4,100 ticker messages with zero public block events.

This does not disprove private Block Matching activity or the parent
account-gated cost candidate. It leaves the public profit floor at zero and
rejects daily time-only polling as wasteful. Do not repeat before
`2026-09-06T03:47:16.3134381Z` unless a material official stream, fee, pair,
whitelist, settlement, block-volume, or observed public-event change occurs.
The first local invocation used an unverified global interpreter and failed on
the `websockets` import before runner entry or network access; the locked `uv`
runtime succeeded, and the durable workflow now requires that exact runtime
preflight before freezing one-use evidence. No credential, order, account,
fund, or protected-capture boundary changed. Canonical follow-up SHA-256 is
`9fa2c8893d73ea7b1bf0efb70c284a20d606866798c121defca131985e84c056`;
registry SHA-256 is
`a375476e54a0a2949e6954d04384f72f11157f73238af61209a393c9362725c8`.

## Polymarket Elon fixed-NegRisk exact parity rejection

The newly deployed August 31–September 2 Elon post-count event fired rank 31's
literal distinct fixed-NegRisk trigger. One exact Gamma GET confirmed ten active
compatible bins. The displayed all-YES sum was 1.0130 pUSD, so that guaranteed-
payout route failed, while every one-NO-to-other-YES identity showed a 0.0130
pUSD source-only gap. Gamma remained rejection-only and did not promote it.

One separately frozen complete 20-token CLOB batch then passed freshness at
121 ms request time, 297 ms oldest-book age, and 129 ms cross-book skew. At the
five-share minimum, the best path lost 0.075 pUSD even before fees, 0.23977 pUSD
after current Gamma taker fees, and 0.61792 pUSD after one adverse tick per leg.
No fee-rate, on-chain, account, credential, order, or fund request was needed.
The exact event is terminal and must not be retried or selectively resampled.

The first frozen book contract never reached the network because its full local
preflight exposed an ASCII-only retained-JSON reader. Its unconsumed failure is
preserved; v2 parsed the UTF-8 source from bytes and passed the full contract
validator before access. `AGENTS.md` now requires full retained-input parsing,
not just imports or `--help`, before a one-use freeze. The frozen v1 runner is
mechanically reconstructable by one exact replacement, bound by lineage SHA-256
`dbfa67537e141344d5d0b15c62944eec3ef72da2c7cb945c61303085b4b40bc5`.
Canonical exact-book
result SHA-256 is
`4601f3980f14ccb4130fbdc36862def5abdd47f46e9f48c7da25113c72fe33a2`;
registry SHA-256 is
`5e34a52a6e0eebf48d5c4ae397bcb1893c10389116425d057b580a2a05013c40`.

## Polymarket NYC Mayor fixed-NegRisk exact parity rejection

The newly deployed NYC Mayor September 1-8 post-count event fired rank 31's
literal distinct fixed-NegRisk trigger. Its exact eleven-outcome Gamma payload
displayed an all-YES sum of 0.9855 pUSD, a 1.45-cent source-only lead.

One separately frozen complete 22-token CLOB batch rejected it. Although the
request completed in 121 ms, the oldest book was 92,065 ms old and timestamp
skew was 91,385 ms. The best five-share all-YES path lost 0.45 pUSD before
fees, 0.54928 pUSD after current Gamma taker fees, and 0.69638 pUSD after one
adverse tick per leg. No on-chain, account, credential, order, transaction, or
fund access occurred. Do not repeat or selectively refresh this exact event.

The reusable runners now accept the contract-bound outcome count and derive the
complete token population instead of carrying ten-market and twenty-token
special cases. Canonical exact-book result SHA-256 is
`2dcaa72b8a9643b3f6652691f7395ac5405cd6f15ee88e57ce45af1f69b0dc6b`;
registry SHA-256 is
`f6b73019910d57daf98764d78a80e487421ae73525499ad1c4e5600ab6018d4f`.

## Polymarket BTC September 5 paired-maker reward rejection

Rank 17's distinct source-selected exact-allocation trigger fired for the BTC
above 72,000 on September 5 market. Exact Gamma and the sponsored-condition
endpoint reconciled two tokens, a 50-share minimum, 4.5-cent spread, zero maker
fee, and 1.99972 pUSD/day without using discovery-page values. The separately
frozen book was 30,871 ms old, so it does not qualify current execution. Its
retained best-bid observation still rejected the exact snapshot: 2.05 pUSD
both-fill gross, 46.80 pUSD maximum orphan loss, and only 12.923014 pUSD even
under the impossible assumption of capturing the entire remaining reward pool.
Do not refetch this exact market. A zero-request correction verified that the
retained Gamma and reward rows contain the same ordered YES/NO token IDs and
updated the reusable runner to enforce that previously omitted cross-source
gate. Correction SHA-256 is
`61df67471329b0e4a1273deea0fbbba9d918d3e11559b3dc26f6f462f69691a4`.
No credential, account, order, fund, or protected-capture boundary changed.
Canonical adjudication SHA-256 is
`1153ef2f90345be8ebfda5b0c2fd3f02a56dc0dad854edf251e21370a9677743`;
accepted edges remain 21 and terminal families become 55. Registry SHA-256 is
`d82fe12b7ec4fb7765bdbad781ac7fc6ef1e6bf26da84882ab42f139411bb6fd`.

## Binance Spot PRIMARY_PEG execution overlay candidate

Current official Binance Spot documentation proves that `PRIMARY_PEG` derives
price from the same-side best book price at matching-engine arrival and that a
`LIMIT_MAKER` pegged order queues after existing orders at that price. The
already-retained production `exchangeInfo` snapshot independently proves
`pegInstructionsAllowed=true`, `LIMIT_MAKER`, and `TRADING` for BTCUSDT,
ETHUSDT, and SOLUSDT.

A zero-new-market-request audit reused the two retained public ticker windows.
At a frozen one-second observation lag, a fixed-price `LIMIT_MAKER` would have
crossed the later opposite quote in 416 of 3,365 discovery comparisons (12.36%)
and 31 of 729 validation comparisons (4.25%). Every symbol in both windows had
at least one such deterministic rejection counterfactual. This is a recurrent,
direction-independent order-acceptance candidate, not profit evidence: the
public samples do not prove subsecond state, acknowledgements, queue position,
fills, spread capture, adverse selection, or PnL.

Do not place or cancel an order on this evidence. Advance only with explicit
separate Binance Spot testnet or paper authority for one minimum-size
`PRIMARY_PEG LIMIT_MAKER` acknowledgement-and-cancel comparison against a
frozen fixed-price counterfactual, or after a material official pegged-order
semantics, filter, fee, or production-configuration change. Mainnet remains
unauthorized. Canonical candidate SHA-256 is
`605d5b195f43bbb9976a5bd3d239388aa918110a6860669da480fa7949b789a2`.
Accepted edges remain 21, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`4bf91c297d3c41583874fe77c3b1c456736bc2809c4bfdac320edd8124f62005`.

## Binance Spot STP TRANSFER internal-reallocation preflight

The current official STP FAQ proves a distinct direction-independent mechanism:
when two different accounts in the same `tradeGroupId` would self-match and
both orders specify `TRANSFER`, Binance applies `DECREMENT` and transfers the
last prevented base quantity and notional between the accounts. The retained
production configuration allows `TRANSFER` on BTCUSDT, ETHUSDT, and SOLUSDT;
their default remains `EXPIRE_MAKER`.

The official example is cash-conserving, not profit: the maker gains 0.2 BTC
and loses 0.04 USDT while the taker loses 0.2 BTC and gains 0.04 USDT. Aggregate
BTC and USDT changes are both zero, both executed quantities are zero, and the
FAQ warns that its commission and price examples are fictional. The maker
remains open after a 0.2 prevented quantity, which is an execution-architecture
lead, but no cash edge exists without an independently required cross-account
rebalance and a strictly inferior ordinary internal-transfer comparator.

Do not manufacture crossing orders, count the transfer as volume or a fill, or
query account state. Advance only with both designated credentials, explicit
signed GET-only `tradeGroupId` authority, and an independently existing
legitimate BTC/ETH/SOL cross-account rebalance whose exact internal-transfer
comparator leaves positive incremental value. Any testnet or paper order needs
separate explicit order authority; mainnet remains unauthorized. Canonical
preflight SHA-256 is
`59da62dccab340bcae63b2ad34697f49caa7e381a7a06755e50adafba2d8d118`.
Accepted edges remain 21, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`8813fbc12ea5fe5ad5eb38630c7faba9c1a740389266ad27bb0c5db41759c52a`.

## Binance Institutional Loan realized interest-rebate overlay

A zero-request offline audit of the retained current official API index exposed
an unregistered economic mechanism: Institutional Loan has signed `USER_DATA`
endpoints for interest history, interest-rebate balance, and rebate records. A
single provenance-only public capture then retained an issuer-distributed
Binance release. Effective `2026-06-01T00:00:00Z`, qualifying KYB VIP borrowers
may receive full monthly interest rebates for USDT, USDC, BTC, or U borrowing up
to 10 million USD by meeting targets tied to incremental trading-volume share,
Open Interest, or Net Asset Value.

This is the twenty-second accepted scoped direction-independent edge, but only
as exact realized cash credited against an independently required existing
eligible loan after its performance target was independently satisfied. The
public forward floor is zero: thresholds, enrollment, account eligibility,
charged interest, payment timing, and realized credit are unproved. Never
borrow, increase leverage, retain collateral, manufacture volume or Open
Interest, or move assets to chase the rebate. The underlying loan can liquidate
the complete Institutional Lending Account balance and remains economically
separate from this incremental credit.

Advance only when both designated credentials, explicit signed GET-only
authority, and an independently existing legitimate Institutional Loan question
all exist. Reconcile one active risk unit, exact monthly interest, rebate
balance, and successful rebate record; deduct every incremental cost. Any
application, enrollment, borrowing, repayment, transfer, collateral, order,
trade, or account-manager contact requires separate authority. Canonical result
SHA-256 is
`e8e17c66a238878e722aa635f1517b685c00fcc9b288c72df3d934a8c235e59c`.
Accepted edges become 22, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`e4ec302a1cc32a57ec1d68cf00ab5d99cbc22d74b80fbc3c68da242485899fd4`.

## Binance CAAS realized organic-client commission markup

The retained current official API index exposes a previously unregistered
Binance Crypto-as-a-Service commission-markup mechanism. It classifies fee-
group and member queries plus markup-trade aggregation and detail reports as
signed `USER_DATA`; creating or deleting a fee group, changing commission, and
assigning or removing members are `TRADE` operations.

This is the twenty-third accepted scoped direction-independent edge, only as an
exact realized positive markup commission from independently existing bona fide
external client trades under an already active disclosed configuration after
every incremental platform, disclosure, consent, compliance, support, demand-
elasticity, tax, custody, settlement, and operating cost. The public forward
floor is zero because partner eligibility, active configuration, rate, client
flow, payout, and costs are unproved. Never create, reroute, split, churn,
self-match, solicit, or assign activity to chase the commission, and never use
it to rescue an otherwise unprofitable platform or trade.

One frozen public dynamic-document preflight returned the already known generic
security page, with zero required CAAS or commission terms and credential-shaped
examples. Its hash and failed decision are retained, its raw body is excluded
from Git, and no alias or retry is allowed. No credential, account, client,
fee-group, member, order, trade, fund, or mutation was accessed. Advance only
with both designated credentials, explicit signed GET-only authority, an
independently existing legitimate CAAS reporting question, an already active
disclosed fee group, and bona fide external client flow. Canonical result
SHA-256 is
`d1656eeccbcf780a2e71190d5f969db07a33548f58de940b55867d494dddebd2`.
Accepted edges become 23, ranked hypotheses remain 44, and terminal families
remain 55. Registry SHA-256 is
`a1fc3c10482909d2c77dbb20ca13dae3eb73e465fa96dcfb20dd0ede17337573`.

## Binance Link-and-Trade realized partner-rebate overlay

The retained current official Link-and-Trade Spot API evidence contains a
second ledger distinct from the already accepted own-client kickback route.
Signed `GET /sapi/v1/apiReferral/rebate/recentRecord` is explicitly the partner
rebate record and binds `income` and `asset` to customer, symbol, order, trade,
commission, and distribution lineage. No documentation example value is owned
income or a forward rate.

The twenty-eighth accepted scoped edge is only exact realized positive partner
rebate income from independently existing bona fide external Spot client flow
under an already active disclosed relationship, after owned payout
reconciliation and every incremental platform, disclosure, consent, compliance,
support, demand-elasticity, tax, custody, settlement, conversion, and operating
cost. Its public forward floor is zero; it is not deployment-ready and proves
no current account eligibility, flow, rate, payout, or profit.

Do not create, customize, solicit, relink, reroute, or manufacture customer or
trade flow. Advance only when both designated credentials, explicit signed
GET-only authority, an already active disclosed partner relationship, exact
legitimate customer identity, and an independently existing bona fide external
Spot latest-seven-day reporting question all coexist. Every relationship,
customer, referral, account, subaccount, API-key, permission, commission,
transfer, order, trade, withdrawal, or other mutation remains separately
unauthorized. This does not accept the separate Exchange Link broker-rebate
candidate.

One discovery-only direct GET to the Exchange Link documentation root returned
HTTP 202 with zero bytes but was not durably request-bound. It is excluded from
all evidence and must not be retried or used. Canonical partner result SHA-256
is `b859a815f0243285d7a01e2f002f77655fbcb25b07f46873e5de68b3ebfa8dd0`;
frontier SHA-256 is
`f84ac384ef3f4edfaa9f3a98e3588223ba3b4c6dd0b1389bbc11e4ece09a8b26`;
registry SHA-256 is
`e3cb85b3bf7920d32fd8a521690b2bffdb475d540225adf13031795249176b9a`.

## Binance Copy Trading realized Lead Trader profit-share overlay

A bounded current official-source sweep found no material Binance or Polymarket
rate, listing, fee, reward, or product trigger worth reopening. The retained
current Binance Agent Native API index then exposed an unregistered Copy
Trading family, and one frozen official rendered product read bound its exact
economic mechanism: Binance supports Spot and Futures Lead Traders; an
experienced trader can apply, set a profit-share rate, and receive that share
when an authentic follower closes a profitable copied position. The same source
states that qualification and regional access are conditional, market risk
remains, and past performance does not guarantee future performance.

The twenty-ninth accepted scoped edge is only exact realized positive owned
Lead Trader profit share from independently existing authentic followers'
profitable copied closes on an independently cross-regime accepted legitimate
BTC, ETH, or SOL Spot or Futures strategy in an already active lead portfolio,
after every incremental fee, slippage, tax, compliance, support, disclosure,
settlement, operating, and strategy-capacity cost. The public forward floor is
zero. No account eligibility, portfolio, follower, configured rate, copied
close, owned payout, current profit, or deployment readiness was proved.

Never use the share to rescue an unprofitable or unsupported strategy, create
or solicit followers, manufacture related-party or wash activity, apply,
enroll, publish, or alter a lead portfolio, or place or copy trades. Binance's
current API index classifies the Futures Lead Trader status `GET` as `TRADE`, so
read-only authority is insufficient. Advance only from an already active
portfolio plus an independently accepted strategy, authentic existing follower,
exact profitable copied close, and explicit account-specific payout-evidence
authority; every state change remains separately unauthorized.

Canonical contract SHA-256 is
`90d61004b2f8c4f81a179a0180f3d9afa7c0872df6b9f5cbf80c92aa478e852b`;
rendered evidence SHA-256 is
`32a61a3f5fb00df8ca9b998d39878d5687492d35a454e219dd0104e7a4f9e692`;
edge SHA-256 is
`6a5acab5c5b9561fa08fedae5b782198db87e8aa8a5c6172f0c4e0fadc3ef7c0`;
frontier SHA-256 is
`bc2db7e81a2e14fee68dc9f57041d226843fefabc0cc47e81db10985e04d84d3`;
registry SHA-256 is
`20f029bbdc6fc31a496f47f74ff3cf59c81b8cf89c522b41e80d72823861cb1a`.

## Binance BFUSD BTC/ETH/SOL funding-carry dominance bound

Do not run another BFUSD-funded spot-perpetual carry backtest on the retained
population. A zero-new-market-request audit proved that the existing optimistic
Portfolio Margin sensitivity is a strict upper bound for every BFUSD APR at or
below 10%: the frozen carry contract charges 10% annual opportunity cost per
capital leg, and that sensitivity already deleted one complete leg.

At both the last hash-bound 5.12% BFUSD last-day APR and an optimistic 10% APR,
zero of nine BTC/ETH/SOL training, validation, and test roles has positive net
economics, zero family-adjusted bootstrap lower bounds are positive, and zero of
72 mandatory regime slices are positive. Necessary but not sufficient BFUSD
APRs are strictly above 22.04372931374745% for BTC, 22.18277961587428% for ETH,
and 26.55978689762650% for SOL. Drawdown, positive-week concentration, basis,
fees, slippage, liquidation, account eligibility, conversion, redemption, and
all external costs remain additional gates.

The current official BFUSD page confirms daily rewards can stack with the
holder's own hedging funding fees and that several account types enter reward
snapshots, but its current numerical APR and collateral fields rendered as
placeholders. No current rate is admitted. Reopen only after a new source-bound
rate strictly clears the applicable necessary threshold together with
source-bound same-period reward history and distribution semantics, or after a
material funding, execution, fee, basis, margin, or capital-cost change capable
of clearing the retained deficit. Never resample this population or request
books merely to test BFUSD.

Canonical result SHA-256 is
`477a4db3c7f9c594ea8c351ce8f0a766280f062437c8089758d6137fbdd54d86`;
accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
64, and registry SHA-256 is
`fe3116d77f82ef88f2ab929b8e71ec67ee029a7399a526e4b98edd9e45c81ef7`.

## Polymarket all-category realized maker-rebate scope extension

Rank 17's literal program-change trigger fired because the current official
Maker Rebates source covers ten fee-enabled categories, materially broader than
the retained crypto-only accepted overlay. One frozen public unauthenticated
Markdown GET retained 5,945 bytes. Its phrase gate failed and is consumed: the
discovery surface said USDC and a 25% Sports rebate, while the retained current
primary bytes say pUSD and 15%; Markdown padding and emphasis caused additional
literal misses. Do not refetch, alias, or rewrite that contract.

A zero-network retained-byte adjudication excludes every discovery value and
extends the existing edge—without increasing the accepted count—to exact
realized positive owned pUSD maker rebates from independently justified
legitimate organic fills in Crypto, Sports, Finance, Politics, Economics,
Culture, Weather, Other / General, Mentions, and Tech markets after every
incremental cost. Current source rates are 20% Crypto, 15% Sports, and 25% for
the other eligible categories; maker fees are zero. Geopolitics is fee-free and
excluded. Allocation is per market, rates are discretionary, and the public
forward floor remains zero. This does not accept market making, fill quality,
account eligibility, owned income, or deployment readiness, and authorizes no
order, cancel, hedge, wallet, account, or funded action.

Canonical failed contract SHA-256 is
`d59f9f93359ff82add45272ab43c02f095b87da334ada6ddcc240786a72a1bb0`;
the retained current Markdown SHA-256 is
`8d2c6562bd1b3376bc3fc1557a60efef5aa3c1d856c7f8dcc405139a07e9ba2a`;
the canonical scope-extension result SHA-256 is
`d37aeac00dca154bbf0d676c3696a688bc7ee6cef9e8118730ffb5be05fb2550`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
64, and registry SHA-256 is
`44a63556d0a8d680661a82743fbdc0eb0bed2a50d75f1679689d72258af6ce41`.

## Binance Agentic Wallet and Alpha-versus-Spot rejection-first triage

Do not register Agentic Wallet conversion savings from the consumed current
source. The frozen official `market-order.md` GET retained 8,816 bytes and
proved that `market-order quote` is non-executing, but the exact five-phrase
gate failed 4/5 because the primary failure wording had changed from the
discovery rendering. Preserve the failed gate, exclude every discovery fee
value, and do not retry an alias. Reopen only on a material byte-retainable fee
or quote-contract change, or when an independently required exact conversion,
active Wallet session, and explicit quote-only authority coexist; stop before
every swap.

One local direct-path invocation failed at import before any request or output.
The corrected and required entry mode is
`uv run python -m tools.capture_public_source_contract` from the repository
root, matching the existing `AGENTS.md` rule. Do not repeat the direct-path
form.

A distinct Binance Alpha-versus-Spot same-token parity prefilter then used one
frozen public token-list GET and the retained current Spot inventory. The
complete Alpha response contained 665 tokens, 398 live rows, and only five live
`listingCex` rows with nonempty `cexCoinName`; none had an active Spot pair in
USDT, USDC, FDUSD, USD1, BTC, or BNB. Therefore zero exact candidates and zero
Alpha or Spot book requests were justified. Do not poll those five rows or
infer asset identity from ticker text. Reopen only on a material Alpha token
list, Spot listing, or transfer-architecture change that creates an exact
transferable overlap.

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
67, and registry SHA-256 is
`5ec42c3de1890f6c9f5e4ecc027b892b3c6f7323eeb813a4db821c1f2b67d2cd`.

## Cross-sectional funding capital-sensitivity correction

The BTC/ETH/SOL same-venue funding-dispersion rejection did not merely defend
its original unlevered two-leg capital rule. A zero-request sensitivity audit
gave a continuous validation-plus-test hold every favorable assumption: the
sum of interval-by-interval perfect-foresight funding maxima, zero switching,
only the frozen 32 bips two-leg entry and exit, and optimistic equal 5x
leverage. That impossible upper bound still loses
18.840831506849315068493150692 bips.

Break-even requires more than 12.9536779944653097611862238x equal leverage
before relative-price and basis PnL, liquidation risk, or any omitted cost. A
fixed orientation is dominated by the oracle, so do not request price history
or build a fixed-orientation rescue. Reopen only on a source-bound material
funding, fee, execution, portfolio-margin, netting, or capital-treatment change
that clears the full risk-adjusted deficit; leverage alone is not a trigger.

Canonical sensitivity result SHA-256 is
`61a65f1f81b7109a6f959a53f8b780e88582b1338ec0cc512ef6784020da029f`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
67, and registry SHA-256 is
`d840ca77bf9f250a87fbc4d5a8423f98d94f1494a18d7863eebfc83812bc030f`.

## Binance fixed-value Gift Card false-discount rejection

A novel direction-independent-looking Gift Card lead was frozen against the
current complete official Binance API reference. The 8,000,456-byte capture
passed all six architecture phrases, and its retained exact 72-line Gift Card
section contains exactly six documented endpoints. None provides a
non-mutating exact positive-discount quote.

The title's parenthetical "discount feature" is not positive buyer economics.
The operative example says a fixed-value BTC card costs 100 USDT plus a minting
fee and redeems into BTC equivalent to exactly 100 USDT. Thus the most
optimistic gross value difference is zero and net value is at most the negative
minting fee before every other cost. `buyCode` is a `TRADE` operation requiring
KYB, sufficient Funding Wallet balance, and an API key with withdrawals enabled.
Do not request credentials, token limits, verification, purchase, or redemption.

Reopen only if an official architecture or terms change documents a
non-mutating exact quote whose face redemption value strictly exceeds payment,
minting fee, and every remaining cost. Credentials or purchase authority alone
are not triggers.

The unrelated full reference contained public illustrative API-key values and
private-key blocks, so it was not committed. Its receipt hash remains durable;
the exact 2,716-byte secret-free Gift Card section was mechanically extracted
and retained byte-for-byte. Search the retained Agent Native index and safe
sections before any future full-reference request, and secret-scan every large
documentation capture before staging.

Canonical source contract SHA-256 is
`49fd6dd8121e9132335669eb75f33d28f799f2c7471c5ba0a0042313a6d812ec`;
source result SHA-256 is
`631e9ec81519a1adbe970e0820ebcf53aad289c9b5965876d9c7b273bd3c2180`;
raw response SHA-256 is
`c785b773eb2f36e87fd077891461320e60cb1aeedc8cec42e268e134e1b68d8a`;
canonical terminal result SHA-256 is
`316e3182ce6a33287463d1c9c6d32a9bd3066bb49740c68d83e5ad717bf36868`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
68, and registry SHA-256 is
`4814e73d89db659d602ad3ed5901f5b94add2b70c8daa51a635ff275ffebddb5`.

## Polymarket Bitcoin September 4 fixed-NegRisk depth rejection

Polymarket's newly deployed eleven-bin "Bitcoin price on September 4?" event
fired rank 31's literal new fixed-NegRisk deployment trigger. One frozen exact
Gamma GET confirmed a complete active fixed-NegRisk population. Its displayed
all-YES sum was 3.0075 pUSD and all eleven rejection-only one-NO conversion
identities were positive, so one separately frozen 22-token public CLOB batch
was justified. Discovery values were not used as economics.

The batch completed and was retained, but the original runner stopped before a
result because it incorrectly assumed the 5% taker schedule from an older event;
the exact retained Crypto schedule is 7% with a 20% maker-rebate fraction. Do
not refetch. A frozen zero-network adjudication of the immutable batch found the
best five-share path lost 3.730 pUSD even with zero fees, 3.81628 pUSD with the
exact retained taker schedule, and 4.07770 pUSD after one adverse tick per leg.
All three views had zero profitable paths. The books were 141,591 ms old with
129,032 ms timestamp skew, but freshness cannot rescue the independently
negative zero-fee upper bound.

The reusable runner now contract-binds each exact event's fee schedule instead
of carrying the older event's schedule across categories. Do not repeat this
event's Gamma GET or book batch, refetch to repair the consumed fee or freshness
failure, or treat displayed Gamma prices as executable. Reopen only on rank
31's remaining literal trigger; require the same strict exact-depth and fee gate
before any on-chain, account, credential, order, or funded work.

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
69, and registry SHA-256 is
`087468040a17520bde05b3dd2e8bf2df94a3608176d9aad6b190382e043d858e`.

## Binance Spot Price Range Execution Rule terminal screen

One frozen current official FAQ GET retained 7,999 bytes and passed all seven
source phrases. The rule is an exchange-set taker price cap, not a standalone
edge: a marketable `LIMIT IOC` or `LIMIT FOK` with the user's worst acceptable
price provides the same or tighter protection for either side, so the rule's
incremental payoff upper bound is zero in every market direction. No live
`executionRules`, `referencePrice`, account, credential, order, fund, or
protected capture request was made.

Do not poll the public rule endpoints merely to compare with an avoidably
unbounded `MARKET` order. Source-bind current rule state only for a separately
frozen candidate that materially depends on unbounded taker execution or exact
`EXECUTION_RULE_PRICE_RANGE_EXCEEDED` residual behavior. Canonical terminal
result SHA-256 is
`6716c320effd97f20ebe84536366e0308ca7089b1ef15d4c6f601c232182a10d`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
70, and registry SHA-256 is
`d2d24730180c4b4c9182a4f694a906cab819e1f93633ac04ddc834af5f6d3d31`.

## Polymarket Ethereum threshold-ladder exact payoff rejection

The current eleven-market "Ethereum above ___ on September 4?" event exposed a
new exact direction-independent payoff family. For thresholds `L < H`, one
`YES(L)` plus one `NO(H)` share has an optimistic one-pUSD floor when every
independent condition applies the common rule consistently, and pays 2 pUSD
only when the close is strictly above `L` and at or below `H`. Independent
condition disputes, cancellations, or inconsistent resolutions are additional
downside and remain unbound.

One frozen exact Gamma GET retained the complete active event and tested all 55
lower-higher packages. The best was `YES(2,900) + NO(3,000)` at a displayed
1.0015 pUSD, already 0.0015 pUSD above its floor before fees. No book, fee,
on-chain, account, credential, order, fund, or protected-capture request was
made. Do not repeat this event or cherry-pick its current BTC or SOL siblings.
Reopen only for a distinct active BTC, ETH, or SOL ladder outside every consumed
event whose complete exact rules and rejection-only Gamma screen contain at
least one displayed package strictly below the optimistic 1 pUSD floor; any
later candidate must also source-bind independent-condition resolution risk.

Canonical contract SHA-256 is
`0737aa5e76be4151213f1a6174eca525e32ec7c46e7e4347842e6ac41c8a7331`;
canonical result SHA-256 is
`42c122e54bd9a7299cc9e739724fabd4cd76716dd3bdfe76c039a1bab8014d2a`;
raw Gamma SHA-256 is
`84a0536e067b5f72a5a4c9fc1ac4a215316b51a742a5f7e781ab37e7fbe5b1be`.
Canonical zero-network terminal adjudication SHA-256 is
`25f90c75b9d8657e44b27ebab8dd4c26fb3434a306a6dbd9e35fdc2fdd53419d`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
71, and registry SHA-256 is
`fdfaac8e3bc873bd07e44e89efb290ebc0bbd7766157d11b6b614875345adead`.

## Polymarket Bitcoin cross-event range/threshold boundary gate

The current official surface deploys a Bitcoin September 4 strict-above
threshold event over the same Binance BTC/USDT noon-ET close used by the
retained fixed-NegRisk range event. It does not create the tempting exact
subset equality. At a close exactly equal to threshold `T`, threshold YES is
zero while the higher range bin is one. The exact equality gate therefore
fails before any outcome-sensitive request.

The weaker package `NO(T)` plus every range-bin YES at or above `T` has an
optimistic common-rule one-pUSD floor and pays two only at exact equality. That
is a new structural research lead, not current economics: the exact range Gamma
request is consumed, discovery prices are excluded, and no refetch is permitted
merely to synchronize it. Reopen only on a distinct nonconsumed same-source,
same-instant BTC/ETH/SOL range-plus-threshold pair with complete rules,
contemporaneous frozen populations, and a displayed package strictly below one
pUSD before exact depth, fees, resolution risk, and every cost.

Canonical boundary adjudication SHA-256 is
`e7290745dac6aac63a0363a98ac9596280548de7093e4ff7261f34cc95eb3ca8`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
72, and registry SHA-256 is
`f0f70eb60bc1ca13d899a3c01930bdc37838c874e875636722c5bd2039f1eec1`.

## Polymarket Solana cross-event coverage depth rejection

The distinct nonconsumed September 4 Solana range and strict-above events fired
rank 31's exact cross-event coverage trigger. Two frozen Gamma GETs retained
both complete eleven-market populations and exhaustively screened the two valid
coverage directions at all ten shared boundaries. Exactly one source-only row
cleared the displayed floor: threshold `NO(150)` plus range `YES(>150)` at
0.9855 pUSD versus a one-pUSD optimistic common-rule floor.

One separately frozen two-token public book batch rejected it. Five shares per
leg cost 5.04 pUSD against a 5 pUSD floor even at zero fee, and 5.05 pUSD after
one adverse tick per leg. The books were 4,655,213 ms old and 7,949 ms skewed.
Because actual zero-fee depth was already negative, zero fee-rate requests were
made. Do not refetch either event or the book, repair freshness, request fees,
or select a consumed ETH/BTC sibling.

Prefilter contract SHA-256 is
`38b02e051962f979c90564318928bcb216c13aca5fe6847eceb8d0398a9abe23`;
prefilter result SHA-256 is
`8efba9e824fb125bd4a5be654704c6f47d62ddc1fd8b38006fad06ac52247417`;
book contract SHA-256 is
`5f0ff60c030abef0a5cc68b0926090de29dc3cbc3f9c55bb9ccd9cc20f68d331`;
book result SHA-256 is
`5bc1e557a85af2588d7b319476e7ef9d4f2afe2c9103100f2a56a41864a9ef81`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
73, and registry SHA-256 is
`932ff844edcda13efc17bd48569c1cd4b5f25abd0f2c0c3747dae696d1b19649`.

## Polymarket Solana cross-event sponsored-reward rejection

The retained September 4 Solana `NO(above 150)` plus `YES(higher range 150)`
package was also screened as a distinct rank-17 maker-reward overlay without
refreshing its stale books. At the exact 50-share reward minimum, one-tick
maker quotes reconstructed a 1.35 pUSD optimistic both-fill gross but a 48.20
pUSD maximum one-leg orphan settlement loss.

Two frozen unauthenticated `sponsored=true` condition GETs each returned a
complete empty population. The maximum publicly proven remaining reward pool
is therefore zero, not the Gamma `rewardsMinSize=50` and
`rewardsMaxSpread=4.5` metadata. No Gamma or book refresh, fee request, account,
credential, order, fund, or protected-capture access occurred. Do not repeat
either condition or refresh the books. Reopen only after a material exact
funded-program change; any survivor must first prove that even its optimistic
full remaining pool strictly exceeds maximum orphan loss.

Contract SHA-256 is
`6064ab6bb733f82dab2ef3fc8f9ea3e4ffeebb3cc2f005f91d41275e6aa1a2ae`;
result SHA-256 is
`97471d6fe9148ba2e4fd818902e0fb33e4f796c7115cd6494c2a35be0bbebeaf`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
74, and registry SHA-256 is
`67a955835c85b8d4d7f11d34bf67e6ed4932a09ab8d0ad1c54e52941cf36bb40`.

## Polymarket September 6 crypto range/threshold delta rejection

One documented newest-first crypto keyset GET retained 100 open events. Its
oldest row preceded the prior rank-31 checkpoint, so the 85 newer events form a
complete delta despite the server returning a nonterminal cursor. That delta
contained all six newly deployed September 6 BTC, ETH, and SOL range and
strict-above events.

A frozen zero-network screen evaluated both valid coverage directions at every
shared boundary: 20 BTC, 20 ETH, and 12 SOL packages. Zero of 52 displayed sums
were strictly below the optimistic one-pUSD common-rule floor. The global best,
BTC `NO(above 88,000) + YES(>88,000)`, was exactly 1.0 pUSD before fees. No
book, fee, on-chain, account, credential, order, fund, or protected-capture
request was justified. Do not follow the cursor, repeat this delta, screen only
one sibling asset, or request books for an exactly-at-floor row.

Contract SHA-256 is
`f2642e0577b422e62e7c4df30eb16ff85c09b741017146dae233c782963b928b`;
result SHA-256 is
`cc4cb32adcafba2da3d48cc8325d7af8a2bfd39c0892ceeb1ec1def939e173f9`;
raw delta SHA-256 is
`4f9aadb6a95bdf2612845b3e3bc96146cc1ea5f23b3cb6bf8815ccb43c8ce087`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
75, and registry SHA-256 is
`bb1b929151474ac48c146e335ce0533f459fb09705ab4feae4c3e5bb76dd81e2`.

## Polymarket September 6 crypto threshold-ladder delta rejection

The immutable September 6 crypto delta also fired the distinct within-event
threshold-ladder trigger. A separately frozen zero-network screen exhausted all
165 `YES(L) + NO(H)` pairs across the complete BTC, ETH, and SOL ladders. Zero
were strictly below their optimistic common-rule one-pUSD floor; the global
best, BTC `YES(68,000) + NO(70,000)`, cost exactly 1.0 pUSD before fees.

No book, fee, on-chain, account, credential, order, fund, or protected-capture
request was justified. Do not repeat these ladders or request books. Reopen only
for a literal distinct nonconsumed complete ladder with a strict displayed
sub-floor package. Enumerate and independently freeze all exact-payoff families
already testable from retained complete bytes before spending another request.

Contract SHA-256 is
`207e04f6c773adb04b73ee55417ad15cbfaa00e0ef2bf64e05a9ff94ab89b73f`;
result SHA-256 is
`c505584dafa3391fc17647fb897d03a402ec4523fcfc5205960383e4b3967fdf`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
76, and registry SHA-256 is
`dd82fabad42807a602c020f43d58dc7c7c37bad67b41eda47004dc4f64afd5a2`.

## Polymarket exact TWAP interval-composition rejection

The retained crypto delta also exposed three adjacent 5-minute markets and one
exactly covering 15-minute market for BTC, ETH, and SOL. Source-bound opening
and closing TWAP continuity proves two forecast-free transitive four-leg payout
floors per asset. The first offline wrapper failed before prices because Gamma
nested lists were JSON strings; that zero-request failure is preserved. A
preflighted v2 decoder then screened the same immutable bytes.

All six packages failed: each up-chain cost 1.990 pUSD and each down-chain cost
2.010 pUSD for a one-pUSD floor. No book, fee, account, credential, order, fund,
or protected-capture request was justified. Do not repeat the 11:45–12:00 ET
set. Reopen only for a distinct exact partition with source/value continuity,
complete tie semantics, and a strict displayed sub-floor package.

A distinct one-use historical execution-persistence contract then selected 12
non-overlapping BTC/ETH/SOL aligned sets, 48 exact conditions, and 24 packages
from retained August 26 markets. Its only Data API trade request returned HTTP
408 after about seven seconds and exposed zero rows. The raw 48-byte error and
request journal are retained. This exact query is terminal: do not retry, split,
narrow, paginate, reorder, or alias it.

Without another request, a separately frozen retained-settlement audit tested
all 25 complete aligned sets available across the six immutable sources: 100
terminal markets and 50 packages. All 50 realized payouts met the one-pUSD
floor; the realized range was one to three pUSD. This strengthens the structural
payoff identity but proves neither sub-floor acquisition, atomic execution,
fees, capacity, owned fills, profit, nor deployment readiness. Only a future
distinct aligned population with strict rejection-only sub-floor economics may
trigger a prospective exact live CLOB package capture.

V1 contract SHA-256 is
`925045d42cac0ba8b4ff0c7cdb6c0c07c70e02fbbf4c3d8ec6389559850152ba`;
failure SHA-256 is
`c7d758f352be3d2ae9d1f4c2957f82fa3926a7a7837f4db0ae315397182cdc83`;
v2 contract SHA-256 is
`ceaf67b3de430e41369470188a466d30f1dc6f0879ab94ab9361acc59488f449`;
result SHA-256 is
`6ef1b3acc9c4a234bc7826395bca02397351c9e168d7d010145752dad33b7747`.
Historical-trade contract SHA-256 is
`799d310c2fd56098fb8cd208e79dc88a338462dfb6ee0ee4f2c9db28d951c65d`;
failure SHA-256 is
`5dcacfa8c9f1b953a9fa28e380e43cfb8c0107361426f67d5b4adeb83dabdfb1`;
settlement contract SHA-256 is
`268301993f1ace29a9bf99d936be5f74101e26e72b61b25e2a272d7ee7146747`;
settlement result SHA-256 is
`81a86ae5a71708516b9f23fbbae0b51cbf337691c64c655e4eb03508a032d84a`.
Consumed runner byte lineage SHA-256 is
`c8fde55efcd61e8fa376d3b5184ca6013f816cf8d74f1f706aad11032e0c297b`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
78, and registry SHA-256 is
`809c82d3233fb76b8fa41c1ba1cd7e7cb168ee8932a71098870d5f5b7e2ab04a`.

## Exact GLW time-gate instruction correction

The accepted market-independent yield frontier incorrectly summarized the
frozen GLW terminal reconciliation as available after `2026-08-31T00:00:00Z`.
The executable contract and complete rank-34 registry row both require
`2026-08-31T00:10:00Z`. The frontier now carries the exact later gate; never
run this request from the superseded earlier prose time.

Two distinct discovery-only searches for a new September 2026 Binance bStock
dividend or Special-funding announcement found no material forward episode, so
they did not authorize another request or alter economics. Current immediate
research spend remains zero until an exact trigger fires. Canonical corrected
frontier SHA-256 is
`1c346600a0bc2a439aa868fba51ed0bf939a48011dbc021ef107bcb5c9771040`.

## Binance displayed funding estimate is not locked hours before funding

A primary-literature lead, *Funding Timing and No-Arbitrage Bounds in
Decentralized Perpetual Markets*
(`https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6805838`), suggested a
strict structural distinction: funding can support a known-at-entry carry only
when the applicable transfer is fixed at entry. One frozen, one-use public
Binance reconciliation compared nine
retained BTCUSDT, ETHUSDT, and SOLUSDT displayed estimates with their eventual
funding rows. At lead times from 16,264 to 23,801 seconds, six of nine estimates
changed; the largest absolute change was 0.6266 basis points. The three exact
matches were all the standard positive `0.00010000` interest/clamp plateau and
do not establish a general lock. That value is not the funding-rate cap.

This is a terminal rejection of the observed hours-ahead estimate-lock premise,
not an edge and not authorization to trade. No book, fee, account, credential,
order, fund, or testnet request was made. Reopen only for an official
fixed-at-entry rule or a separately preregistered near-finality executable study
with positive conservative after-cost guaranteed headroom.

Contract SHA-256 is
`46d0bbf9e48b090332653d8b5cfe38b2350fc8ce3d8b2fd0100addc10916c8df`;
result SHA-256 is
`d1a75d29bc7d48154f2006a335a401d7451a6ad58c36b12127b387581f3c3ac3`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
80, and registry SHA-256 becomes
`d49940b750e9fc4d8416840185136d1fcefbc734d12473ad4c00fe76ea2d8f89`.

## Near-finality funding capture rejected before access

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

## Binance Yield Arena discovery routing correction

Do not model, rank, or query Binance Yield Arena as a distinct edge. The repo
already retained and adjudicated the current August 26 article; a later search
rediscovered that same surface because its product-routing role was not explicit.
Yield Arena aggregates Simple Earn, ETH/SOL staking, Dual Investment, and other
products with incompatible payoff and lockup semantics. Route every exact offer
to its existing product family by article code, asset, and payoff identity.

The signed `GET /sapi/v1/earn/arena/activities` endpoint is USER_DATA and is
only justified for an exact product/account reconciliation under explicit
GET-only authority. It is not an authenticated browsing shortcut. Arena
branding, a headline `Up to` APR, or a changed headline does not trigger a new
collector, backtest, or hypothesis. Canonical routing addendum SHA-256 is
`b01792aabe04989b4e65fb5ae00719249fa6186f7c5bd345c59a3ea9b4d8ff66`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families remain
83, and registry SHA-256 becomes
`8262b29c4ba9b6eb322beb91d18a9c44756355b451374292904aa569e7c81941`.

## Polymarket September 7 CFB monotone rejection

A current primary Polymarket event page exposed SMU vs. Florida State on
September 7 with moneyline, spread, total, and complete rules, satisfying rank
30's literal first-distinct-event trigger outside the consumed September 3-6
CFB window. One frozen exact September 7 keyset GET returned a complete one-event
population and machine-proved one valid moneyline/spread monotone relation.

The only relation cost 1.09 pUSD at rejection-only Gamma prices for a 1 pUSD
minimum payout, already negative by 0.09 pUSD before execution costs. There were
zero strict sub-floor candidates, so no book or fee request was permitted. Do
not repeat, paginate, narrow, refetch, or request books for this population.
Reopen rank 30 only for its next literal distinct-event trigger outside every
consumed window. Contract SHA-256 is
`391356fc6d94e3e6c5407502afc94f81be4d4a3ea6192c1368dace613644bf19`;
result SHA-256 is
`5cba2d835c600d8eec6b8f27a7010a535b72292680d5a558fb44f29316a3c796`.
Accepted edges remain 29, ranked hypotheses remain 44, terminal families become
84, and registry SHA-256 becomes
`7c4c4ecc062a4cda012958571fcc477e304d38e0f00f8c67aecd37ae4a031645`.

## Binance stock-option deployment gate

The current official Binance developer contract exposes a public unauthenticated
Options `exchangeInfo` population with `contractType` and `underlyingType`, and
separately classifies TradFi Options contract acceptance as `USER_DATA` and a
state-changing `POST`. A distinct direction-independent payoff lead therefore
tested only whether active `TRADFI_OPTIONS` plus `EQUITY` symbols are actually
deployed before spending any economic request.

The one frozen public inventory GET returned HTTP 200 and zero active stock
option rows. Stop there: no option tickers, futures metadata, books, premium
index, funding, account, credential, contract acceptance, order, or fund request
is permitted for this population. API feature support is not deployment.
Reopen rank 46 only after an official stock-option listing or a material option
settlement, fee, access, unit, or matching-perpetual architecture change.

Contract file SHA-256 is
`08bcd2fd86082e9c4d03b4408c38aabfb5f84248ace9748d6898ebcef3114624`;
canonical result SHA-256 is
`b72592efad26563aacc4e6d8611f15f3172039ffc220153ab824bf57231afcb3`.
Accepted edges remain 29, ranked hypotheses become 46, terminal families become
115, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`06ca44a66364f8bcd1d78b76d4e75f13f1431ce9437a5e7f7ce8c1f6af5ffd04`,
and durability-audit SHA-256 becomes
`a0896ba8a9782234b2735e67553583118694715e8f65886416d6f73481d4ea1a`.

## Binance retained crypto single-option terminal-floor rejection

Before another venue request, reuse the hash-bound August 27 BTC, ETH, and SOL
option ticker, futures book, exchange-info, and 500-row funding histories. A
distinct one-option payoff identity was not covered by the earlier two-option
conversion/reversal screen: long call plus equal short perpetual has gross floor
`F_bid - K - call_ask`; long put plus equal long perpetual has gross floor
`K - F_ask - put_ask` before expiry basis, funding, and costs.

The zero-network screen exhausted all 1,410 eligible active options. Of 1,115
positive-entry rows, only `SOL-260828-90-C` and `SOL-260828-92-C` had positive
gross floors. The best was 0.17 USDT per unit, or 16.71090141 bips. The frozen
33.5-bip option/futures fee plus expiry-basis stress alone exceeded that gross
floor; two adverse ticks, adverse short funding, and capital cost reduced it to
-41.09843077 bips. Zero rows survived, so no option depth or current market
request is permitted. Do not rebuild or reprice this retained population.

Contract SHA-256 is
`b31c691c728d7d2c7a5d7e13151c57139c6aad1c5fc22fb2be913edb6e5b9a60`;
result SHA-256 is
`90c05ed35db00da7e5b4a2d8ec6ac0a51367a1a768dc58a39ef479510d5aa745`.
Accepted edges remain 29, ranked hypotheses become 47, terminal families become
116, and stable current account-qualified after-all-cost edges remain zero.
Registry SHA-256 becomes
`7eeba81c9b93404372bd4833002c67c5a958d25e5b7de258f86c95968f70b247`,
and durability-audit SHA-256 becomes
`9fe9eff62c4d11e94d7791bbba3edfbb89d8bc26b0f38da204b1ef38cf6a0485`.

## Current Predict.fun structural-edge checkpoint (2026-09-01)

Logged-out Predict.fun pages expose rendered public books and a Polymarket
comparator without a Predict API key. Use this only for rejection-first public
discovery: comparator probabilities are not asks, missing asks are not free, and
acceptance still requires prospectively source-bound exact rules, side-specific
depth, fees, and synchronized timestamps. Predict.fun is a separate BNB Chain
venue; Binance Exchange and testnet credentials do not authorize it.

The disjoint OpenAI-acquisition pair is terminal: materially matching rules,
Predict zero volume/no NO ask, Predict YES ask 0.92, and Polymarket YES/NO asks
0.042/0.959 produced one 1.879-pUSD package and one unavailable package. The
current short-horizon crypto cross-venue identity is also terminal before books:
Predict uses BTC/USDT point reports with Up/Down/Flat, up to a five-second
buffer, and emergency manual close; the exact frozen Polymarket round uses
BTC/USD 60-second TWAP with equality assigned to Up.

Predict Points have zero admitted monetary value. The current official maker-
rebate program does define a thirty-first scoped edge: makers pay zero fee and
receive 25% of the taker fee automatically on eligible badged UP/DOWN crypto
fills during a trial published to end September 16, 2026. The maximum nominal
rebate is only 0.0025 USDT equivalent per share at price 0.5 before costs. Credit
only exact value actually distributed on an independently justified legitimate
organic fill after the received-asset basis and every incremental cost; this is
not an accepted market-making strategy, stable account-qualified edge, or
deployment-ready claim. Never create volume, self-trade, wash-trade, manipulate
balances, abuse cancellation, or post non-executable orders to farm rewards.

Reopen before the trial end only for a distinct active badged market plus a new
official public fill-quality source or explicit separate Predict.fun read-only
paper-study authority, freezing chronological queue, fill, cancellation,
inventory, hedge, and regime gates first. Orders and funded actions require
separate explicit authority.

Canonical adjudication SHA-256 is
`b054ef3388dd9e97b065120d35792699ef3bbdc40efbd72e2034e77db0efe1fd`.
Accepted edges are 31, ranked hypotheses are 48, terminal families are 148, and
stable current account-qualified after-all-cost edges remain zero. Registry and
durability-audit SHA-256 values are
`5805fc3f11e3a8f9c4db2a365d7dd1108e0653b3b2a8147135bdeb4de0d456a8`
and `400296f66e5807219ce6f8b5825a3b4a863c7383a94c766ba9ca546bc3a55343`.

## Latest public-chain and deployment-trigger checkpoint (2026-09-01)

The frozen September 3-9 Polymarket WNBA catalog returned a complete empty
population, so that exact window is terminal before books and fees. Current
official Predict.fun technical and audit sources confirm on-chain execution and
fill-transaction rebates but publish no deployed exchange, fee, token, NegRisk,
or eligible-market contract address. The linked CRE-audit addresses are an admin
multisig, Chainlink infrastructure, and a BSC-testnet adapter, not source-bound
fill/rebate lineage. Do not guess deployments or substitute testnet for mainnet.

The technical source assigns accumulated Venus collateral-yield claims to yield
managers and defines no trader beneficiary, rate, base, distribution, or
redemption. Credit zero trader-owned yield.

Canonical adjudication SHA-256 is
`aa89b82912335b288506e731382e8ac87fa8e60afb614986f224a1b88d730c22`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
151, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values are
`57d12349bb20e5ed5c58e1bed6db6dbfccbe3ea6c11868b56f16dc056bf0d480`
and `ec2c84a4fd19dded954e5a632c9b58dc7b3b0ceb3c2f71617c41aac756448894`.

## Latest exact-payoff checkpoint (2026-09-01)

The frozen September 30 ET two-minute endpoint band returned 100 Gamma events
with a cursor. The population is incomplete and terminal without narrowing or
pagination; its three visible fixed NegRisk events all had all-YES sums above
one pUSD, best 1.0085.

A zero-network retained-input audit then tested cumulative Gemini Pro September
30 NO plus all 30 exact-date release YES legs. The deadline event began
6,654,892 seconds earlier, so the floor is not source-proved. Even under the
optimistic rejection-only floor, exact side-specific cost was 1.264 pUSD per
share: -1.320 pUSD at five shares before ticks and -1.64295 pUSD after one
adverse tick per leg plus exact fees. Do not refetch, omit legs, repair the
cursor, or promote this discovery-leaked population.

Projection adjudication SHA-256 is
`233475bd3a302ba7a905b5cd8eea4d53939513aea8dbdc9fcb5ddefcc8f053ce`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
153, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values are
`5a5040f33e7a14803dd066e565626279c1e3d9e6bfc4c39803237b5010822abf`
and `6b264230457ad0c159d9632c0f395ea7c326b7cd1c94f4d4048dbad13ce2749e`.

The next frozen zero-network retained-input screen reused only the untouched
exact U.S. measles September 30 event. Its 2900, 3000, 3100, and 3200 markets
share one CDC total-case counter, cutoff, fallback, and at-least rule, so every
ordered `YES(lower) + NO(higher)` pair has a one-pUSD floor. All six pairs had
side-specific rejection prices; zero cost strictly below the floor and the
best cost 1.030 pUSD before any fee, tick, depth, time-value, non-atomicity,
latency, unwind, or operating cost. No new request was made and no book, fee,
account, credential, order, fund, on-chain, or protected state was touched.

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

A distinct four-family zero-network rank screen then tested named companies in
the exact second- and third-place Labs-view events. It excluded the different
Models-view best events, `Other`, identity-free Company placeholders, and the
unproved ByteDance/Bytedance spelling alias. The frozen population contained 67
`NO(second) + NO(third)` one-pUSD-floor packages; 48 had complete side-specific
prices. Google in Text Arena Overall was the only strict metadata candidate at
0.99 pUSD per share, but at five shares one adverse tick per leg plus exact
taker fees changed 0.05 pUSD gross headroom into -0.11030 pUSD. No books were
authorized.

Contract/result/preflight-failure SHA-256 values are
`3fd13b0fb651614e1dbce6f9e85f1baf51a5cb6f020a150efa23e0378e8aaa1a`,
`268aa6e00d349b164f1727e5d47ae34afe05787627e6552307aa04b59cdbe1e6`,
and `32495c463d2748df8b4afbcfc579887d0ee01bb61ea2b933da981bac8a53c151`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
155, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`384d75c8fcec15cc2caecb249171b01ff00dad24c4a5209d94d0880364f97992`
and `4723a56809cf8d6f9ba082a02e17a45f90b326bdfffd226ef53dc80a96a2f0e5`.
No network, account, credential, book, fee endpoint, order, fund, on-chain, or
protected capture was touched.

The following zero-network retained-input screen inventoried 14 active-looking
cumulative-deadline families. Ten failed before economics because the
later-deadline market started later or the rules differed, leaving an uncovered
state where earlier YES and later NO can coexist. Only four pairs had identical
rules, a strictly later deadline, and a later-market start no later than the
earlier-market start. Two had complete side-specific prices. Zero cost strictly
below their one-pUSD floor; the best was Qwen Plus September 15 NO plus
September 30 YES at 1.19 pUSD per share. At five shares it was -1.10590 pUSD
after one adverse tick per leg and exact fees. No books were authorized.

Contract/result SHA-256 values are
`d85ff8d51b720b2b88d41952d25b003b5d5240771499e37d19579204180adde8`
and `1b3dcf4167ef6bd5dbec85b1f8d5e23f23b0ad7bcac2cfe4e32b58d65afedb23`.
Accepted edges remain 31, ranked hypotheses remain 48, terminal families become
156, and stable current account-qualified after-all-cost edges remain zero.
Registry and durability-audit SHA-256 values become
`c5e22c02abcb89ce9ed73b515c32c1fad6a730842c8ff858b5863197f98d14bf`
and `77b63ce226f366adc047bb756407322828aa856bf562d771a3a4e3fd4b21b6fb`.
Do not refetch, reprice, weaken the creation-window gate, or request books.
Reopen only on a material price, fee, tick, rule, or market-architecture change.
No network, account, credential, book, fee endpoint, order, fund, transaction,
on-chain, or protected capture was touched.

The next distinct fixed-NegRisk discovery was the exact eleven-bin NYC
September 1 high-temperature event. The official rendered discovery page
exposed all eleven displayed Buy-YES prices before a contract was frozen, so
the exact event is permanently promotion-ineligible. One frozen exact Gamma
GET was retained only for rejection and lineage. It reconciled all eleven
markets. The all-YES acquisition sum was 1.0485 pUSD for a one-pUSD floor and
failed before books. All eleven displayed one-NO-to-other-YES identities were
gross-positive, with a best 0.0485-pUSD displayed gap, but Gamma prices are not
executable output bids and the pre-freeze leak forbids candidate selection,
depth, or promotion.

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

Do not refetch, alias, reprice, or request books for this exact event. For the
next distinct daily weather partition, freeze deterministic event selection and
the complete expected population before any outcome-aware page or search
access. No account, credential, book, fee, on-chain, order, fund, transaction,
or protected capture was touched.

The next frozen zero-network screen validated all five returned AI Arena
Overall score thresholds. The 1510 market was closed and non-acquisition-
capable. Four of six pairs among the active 1520, 1530, 1540, and 1550 markets
were creation-window-safe; two were excluded because the lower-threshold market
started 4.588726 seconds or 0.253017 seconds after the higher-threshold market.
All four safe `YES(lower) + NO(higher)` packages had side-specific prices. Zero
cost strictly below their one-pUSD floor. The best 1540 YES plus 1550 NO cost
1.035 pUSD per share and lost 0.19670 pUSD at five shares after one adverse tick
per leg plus exact fees. No books were authorized.

The first preflight stopped before economics because a millisecond timestamp
was reserialized at microsecond precision. The preserved correction compares
timezone-aware UTC instants without weakening exact-time equality.
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
