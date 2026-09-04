# Paradex execution-route qualification

Outcome: documented lower-fee routes merit qualification, not promotion.
The completed funding study failed a frozen cost allowance, not measured
commissions. This source review neither changes that outcome nor proves a
new effective-dated fee change. No old market window was accessed again.

## Retained primary evidence

Two exact public GETs completed September 4, 2026, 23:28:58–23:29:00 UTC:
9,844 bytes, no redirects, retries, accounts, tokens or orders. The
[frozen plan](paradex-fee-source-plan.json),
[journal](paradex-fee-source/requests.jsonl) and
[reconstructible review](paradex-fee-source/review.json) bind the exact sources
and implementations. Current documentation does not establish historical
fees; the July 3 maker-fee notice supplies no year.

| Documented route | Perpetual trading fee | Qualification issue |
| --- | --- | --- |
| Retail | Maker/taker 0 bp | Interactive order classification; 300-ms submission and cancellation delays |
| Pro | Maker 0.3 bp; base taker 3.5–4.5 bp | Standard API default; volume tier and applicable discounts must be evidenced |

The taker discount floor is 1.75 bp, not zero. Delisted spot/perpetual
positions incur a separate 1.5-bp settlement fee. Discounts requiring token
holdings, staking or qualifying passive-Retail fills cannot be credited
without their exact conditions. See the retained official
[fee terms](https://docs.paradex.trade/trading/trading-fees.md).

Classification is per order, not permanent user type. Retail caps are
3 orders/second, 30/minute, 300/hour and 1,000/24 hours; exceeding them
switches to Pro behavior and removes RPI access. UI/interactive API intent
does not establish that this platform's proposed automation qualifies.
See the retained official
[order-classification terms](https://docs.paradex.trade/trading/trader-profiles.md).

## Financial interpretation and next decision

Actual fee cash is the sum of absolute executed base quantity times execution
price times applicable fee fraction, retaining each payment asset. Entry and
exit notionals need not match. A lower nominal fee improves execution only
if it exceeds incremental spread, hedge delay, adverse selection, partial-fill
inventory, cancellation-race and other costs. Zero trading fee is not zero
total cost or guaranteed positive carry.

For a future independently justified study, qualify a permitted order route
before choosing its cost model. Bind both venues' execution fees, basis and
currency conversion, funding cash, capital and venue risks; compare identical
base exposure and horizons. Model uncertain fill/latency costs as scenarios,
not observations. No artificial volume, rate-cap evasion, account mutation,
token acquisition or staking is authorized by this review.

The immediate missing evidence is workflow/account eligibility and realized
execution quality, not another funding sample. Current terms alone do not
satisfy a material-change retry trigger. Preserve the exact existing triggers;
do not relabel a retrospective cheaper-fee scenario as untouched validation.
Rank 59 gains this source artifact only. Counts remain 37 scoped acceptances,
65 hypotheses, 188 terminal observations and zero current account-qualified
after-all-cost edges. Prior ledger bytes remain in Git at
`88a90521bed217a038e283d70232ba56a384d9bd`; old experiments are unchanged.
