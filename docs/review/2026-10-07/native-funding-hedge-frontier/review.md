# Native funding: complete-hedge break-even, not a profitable strategy

The [one-use offline result](result.json) advances the existing spot/perpetual
funding family using the separately captured September 5-October 6 native data.
No new market request, account access, credential, order, fit or campaign was
used. It does not repair or repeat the old failed history population.

## Financial result

For a matched fixed-base long Spot/short USDT-linear perpetual hedge, supplied
cash reconciles as:

`net = q*((F_entry-S_entry)-(F_exit-S_exit)) + funding_cash - total_cost`.

Common price direction cancels; hedge-basis change, actual costs and independently
margined liquidation do not. The funding component uses the existing exact cash
core, not a sum of unweighted rates. The first native event in each panel is only
a normalization reference and its payment is excluded. Subsequent entitlement is
hypothetical; UNKNOWN-entitlement bounds are retained separately. No execution
at the reference mark is assumed.

All three assets and four calendar-only eight-day panels plus the full period
were fixed before cash access in [the contract](contract.json). These panels
overlap the full period; they are not independent observations or holdouts.

| Full-period reference-scale sensitivity | BTC | ETH | SOL |
|---|---:|---:|---:|
| Funding cash, basis points | 43.0289 | 43.5837 | 34.3168 |
| Basis-deterioration budget after 32-bp reserve, zero capital cost | 11.0289 | 11.5837 | 2.3168 |
| Same budget with 3.25% annual cost on two reference capital units | -45.9574 | -45.4026 | -54.6695 |
| Annual capital break-even, zero basis change, 32-bp reserve, two units | 0.6290% | 0.6606% | 0.1321% |

The 32-bp reserve, two capital units and 3.25% annual cost are deliberately
explicit sensitivities, not native commission, financing or collateral facts.
The budget is a strict-positive-net boundary, not observed net PnL. Negative
budgets mean basis must improve by more than the deficit under that scenario;
they do not prove negative expected value for every implementation.

All twelve eight-day asset panels have funding below the 32-bp reserve even
before capital cost. Repeatedly restarting that hedge would consume the thin
cash component under the declared reserve. Longer holding can spread entry/exit
cost, but introduces further basis, regime and collateral risk. This supports
studying low-turnover, economically selective hedges rather than churning or
optimizing prediction accuracy alone. It does not establish a stable edge or
authorize a new quote hunt.

## Qualification and next useful work

Actual paired execution cash/fee assets, equal net quantities, independent event
coverage, owned entitlement, continuous margin/ADL/custody exposure, basis
behavior and untouched cross-regime capacity remain unqualified. Funding-only
prefund and drawdown are not hedge drawdown or account solvency. UNKNOWN payment
bounds include negative cash for all three full-period panels, so even gross
receipts cannot be credited without entitlement evidence.

Reuse the retained frontier to design a qualified all-in-cost/basis and
collateral comparison before another capture or model fit. A fee-efficient
maker entry must include partial fills, queue-origin uncertainty and adverse
selection; advertised maker fees alone cannot erase those costs. Do not choose
the cheapest sensitivity after seeing outcomes as a promotion rule. Existing
family retry triggers remain unchanged. No cost rate is hardcoded into the
runtime trading engine.

Verification: 96 affected cash/frontier tests and Ruff pass. They cover exact
cash weighting, excluded reference entitlement, paired hedge/basis cancellation,
strict break-even reconciliation, invalid inputs, source/implementation tamper,
terminal failure, durable one-use journals and CLI status. Raw/source hashes,
panel arithmetic and journals are verified separately. The runner was preflighted
on all retained input schemas before freezing; no consumed runner is changed.
Research counts remain 201 observations / 65 hypotheses / 37 scopes / zero
qualified stable profitable edges. No broad CI, GPU benchmark or training ran.
