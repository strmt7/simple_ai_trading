# Existing-inventory completion: the missing economic comparator

Exploratory retrospective source review, September 4, 2026. Base:
`bbc1e943a515537e83a5664faa45617cc3685c23`. No venue request, account access,
order, new training run, protected-data access or historical-result change.

## What the retained evidence actually contains

The August 30 validation result has 66 selected historical lock rows across
27 conditions: 58 BTC, 6 ETH and 2 SOL. Its canonical self-hash was rechecked:
`b81af57f094f1ff75bcb77f9938ec7c84791af4e1cecb44b3402dac17d4dc1df`.
The file is
`docs/model-research/action-value/polymarket-wallet-opposite-lock-validation-result-v1-2026-08-31.json`;
its exact file SHA-256 is
`b90553bc3701ee53d55f308ce58d5597c1904fee4ec66529655a73528c384ccc`.

All row keys were inspected: asset, completion_timestamp, condition_id,
duration, event_slug, first_leg_all_in_cost_per_share, hedge_observed_price,
hedge_stressed_all_in_cost_per_share, lag_seconds, locked_pnl,
locked_pnl_per_share and matched_shares. None supplies the contemporaneous
executable original-token sell bid, sell depth or sell fee. This is a finding
about these retained lock rows, not proof that no such data exists anywhere.

The fee-rounding correction self-hash also reconstructs:
`b423b44e57bfd329220256facf4b9eabe45371b267e7a91ea08aa11a666be204`.
The existing positive historical cash-flow totals and failed robustness gate
remain unchanged. Do not select only the 43 stress survivors and call them
new validation.

## Financial comparison

For the same quantity of legitimately preexisting inventory, compare two
feasible routes at the same decision time and valuation horizon:

- Completion value: conservative realizable matched-pair proceeds, less the
  all-in cost of obtaining the required *net* opposite shares, completion or
  redemption costs, and incremental capital/time/failure costs.
- Liquidation value: executable net sale proceeds from the existing token,
  after depth, fees and the route's other costs.

Their difference is the completion route's incremental economic value over
selling. The inventory's historical acquisition cost cancels from this
comparison. It remains essential for accounting, ownership and total-strategy
P&L; cancellation is not permission to ignore those controls.

Illustrative arithmetic, **not market data**: 100 existing shares cost 35.
Buying 100 opposite shares costs 40, and the matched pair realizes 100 with
no other costs in this example. The pair locks a historical profit of 25.
But if selling the original shares immediately nets 65, sale profit is 30:
completion is worse by 5. Positive historical lock P&L alone does not prove
incremental execution advantage. Missing sale proceeds must remain unknown,
not zero. Future redemption cannot be equated to immediate cash without a
bound on delay, capital cost and redemption feasibility.

## Consequence for training and research

Before training a completion model, establish a source-bound comparator with
decision-time original-token bids and opposite-token asks, matched depth,
net-delivered quantities, side-specific fees, exact condition/payoff identity,
source receipt times, legitimate inventory and feasible completion mechanics.
Record failed and unavailable routes as well as apparent winners. Avoid
training on later observed hedge fills as though they were available quotes.

The first benchmark should be a fixed economic break-even rule versus direct
sale and abstention. Only then compare a small causal model of completion
probability, adverse selection and incremental cost on an independent,
prospectively frozen population. A classifier predicting whether the original
pair made money targets a different question and can reward selection luck.

The Binance analogue is a whole-lifecycle matched-carry comparison: integrated
funding and basis cash flows, both entry/exit legs, financing, margin reserve,
liquidation risk and all committed capital. Direction-neutral notional alone
does not establish stable net returns or positive incremental value over exit.

## Authority and next trigger

This review does not satisfy the registry's existing inventory-overlay retry
trigger. That still requires separately authorized read-only account evidence
plus independently preexisting eligible inventory, or a material current fee,
merge, redemption or execution-architecture change. No new capture is launched
from this memo. The holding-yield post-conflict captures also remain consumed.
This is a useful new comparator requirement, not a family-wide rejection,
repaired holdout, accepted edge, current opportunity or profitability claim.
