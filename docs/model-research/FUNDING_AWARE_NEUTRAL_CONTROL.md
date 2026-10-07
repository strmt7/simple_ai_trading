# Funding-aware control: cash truth before an optimizer

The [October 7 primary-source audit](../review/2026-10-07/funding-aware-maker-source/review.md)
supports funding as a useful inventory-control state, not a direction-neutral
profitable strategy. The paper leaves hedge accounting off and uses proxy fills.
Its gains, simulation-seed intervals and selected stress windows cannot qualify
Binance or Polymarket execution. No paper parameter or return is adopted.

## Cash state and settlement clock

For fixed base quantity `q`, a linear perpetual's discrete funding cash is
`-q * settlement_mark * fractional_rate` in that instrument's payment asset.
Normalize by the actual entry notional only when reporting an entry-relative
return. Summing raw rates assumes a different, constant-notional exposure and
omits the trades/costs needed to maintain it. A last-trade candle is not proof of
the settlement mark. Retain individual cash flows and currencies; conversion at
par is a named diagnostic, not an observed conversion.

Binance point settlements, Paradex index accrual and Polymarket payoff
settlement require separate clock/entitlement rules. Do not divide a realized
Binance payment across a nominal holding period or import an hourly Hyperliquid
model as Binance's payment law. Predicted funding, the last settled rate and a
native receipt are different evidence. Features require actual availability
before decisions; future realized prices/rates belong only in labels.

Price scaling is also a model assumption. If fractional funding `F` is modeled
as OU and cash funding is `f = S * F`, then `df = F dS + S dF + d[S,F]`.
An independent, constant-parameter cash OU does not follow automatically.
Retain joint state or qualify a local price-frozen approximation and test its
error. A small reported sample correlation is not independence or tail safety.

## Neutral objective and execution

Match actual base quantities across hedge legs and retain the residual exposure
after every partial fill. A dealer with no predictive price drift can still
carry directional inventory. Optimize conservative incremental net cash versus
feasible liquidation/abstention, including every leg's fees, funding, basis,
cancel race, latency, financing, margin reserves and orphan unwind. Funding
income alone is not the objective. Separate quote-currency cash and reserved
capital; linear funding sign matching does not remove independent liquidation
or custody paths.

An HJB/value table is conditional on its arrival and state-transition model.
Require quote-size units, fee-adjusted rewards and event-clock consistency;
value differences per quote unit must not be confused with price offsets.
Jump/tail funding and fill-conditioned adverse selection belong in the
economic stress, not just a likelihood appendix. A blocked quote side must be
absent from submission, not represented as an arbitrary enormous price.
Deterministic ownership, Stop and model-health controls remain outside the
optimizer, with no learned override.

Use existing cash/completion and forward make/take components where their
contracts apply. Polymarket's state-specific payout/oracle model is not an OU
perpetual-funding model. Do not build a second evaluator to bypass current
source, queue, partial-fill or admission restrictions.

## Evidence-efficient evaluation

One hundred simulated fill seeds on one price/funding path quantify conditional
simulation uncertainty, not one hundred independent economic observations.
Pair policies with event-keyed innovations, identical feasible capacity and
costs; equal random seed numbers alone do not establish shared innovations when
policy branches consume randomness differently. Compare risk-matched baselines
with quantity-sensitive fill support, not a larger inventory budget under an
unchanged fill curve. Holdout-selected extreme windows are diagnostic slices,
not decision-time regime labels or untouched confirmations.

Before a solver or large CPU/GPU fit, qualify the label cash and clock, explicit
counterfactual, complete failure/partial-fill population and chronological
roles. Then freeze a small paired economic experiment with an untouched period
and failure consequence. Increase data/compute only for a measured information
or runtime benefit. Public trades/volume hits do not prove our queue fills.

## Confirmed current label defect and repair sequence

`derivatives_hurdle_data._funding_in_holding_window` and the corresponding
`barrier_payoff_data` helper currently sum funding rates without mark/entry-price
weighting, although price returns use fixed-quantity entry normalization.
`FundingState` and the stored `funding_rates` rows have no settlement marks.
The source-bound [counterexample](../review/2026-10-07/funding-aware-maker-source/reproduce_funding_cash_units.py)
confirms a 50-versus-51-bip debit and opposite signs near a zero-return boundary.
It is synthetic algebra, not an observed edge or fee-qualified full replay.

The cash computation remains unfixed. Before retraining affected targets:

1. Add source-bound settlement marks with exact symbol, product, currency and
   timestamps; preserve existing rate rows and all old results. Conflicting,
   missing or substituted candle marks must not silently qualify cash.
2. Centralize fixed-quantity cash weighting and declared entitlement boundaries.
   For intrabar barrier exits, retain uncertain settlement eligibility or use a
   justified adverse cash bound; a minute's end is not the known stop-fill time.
3. Route both builders through that cash layer, bind mark coverage/hash into new
   dataset provenance and test paired long/short, signed-rate, price-change,
   missing-evidence and timing cases. Preserve old reports as rate-only proxies;
   generate distinct cash-qualified targets rather than rewriting them.
4. Audit affected source-bound model/replay consumers, then perform the smallest
   chronological economic comparison. No repaired-label or profitability claim
   follows from correcting the AI prompt or the type-only import.

This is the next model-data deliverable, not permission to repair consumed
captures, refresh blocked populations or train on fabricated marks. Unaffected
public research and capital-safety work may continue. Current qualified stable
profitable edge count remains zero.
