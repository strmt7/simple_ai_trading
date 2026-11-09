# Paired inventory cash before neutral-model training

The previous cash replay modeled one derivative leg. A positive funding payment
or equal quote budgets cannot establish a neutral strategy's net economics.
This checkpoint composes that existing exact replay with explicit incremental
spot fills, net base inventory, spot cash, financing, opportunity and other
declared quote debits. It changes no historical result or funding-payment law.

The [frozen scope](intent.json) permits zero network requests, market-dataset
reads, account access, orders or training. The [generated controls](controls.json)
bind exact input quantities, current source bytes and the reproduction tool.
Their placeholder source digests are explicitly synthetic declarations, not
native evidence. The eight cases are accounting controls, not observations,
independent evaluations or prospective profitability tests.

## Financial behavior

`SpotInventoryCashPath` tracks individual gross base/quote fills and explicit
base/quote commissions. BUY base fees reduce received inventory; quote fees
reduce cash. Cumulative order acknowledgements must not be submitted as a series
of incremental fills. Fill identities are unique and ordering is explicit.
Overselling net received quantity, unsupported SELL base commissions and a
missing full terminal unwind reject. Other fee assets require separate evidence
and valuation; the helper does not silently convert them.

For supplied reference price `P` and net quantity change `dq`, each trade's
equity adjustment is `quote_cash_change + dq * P`. Holding contributes
`q * (P_next - P)`. This retains base fees and execution-price differences once.
At complete liquidation, the sum must exactly equal recorded quote cash minus
initial marked incumbent inventory. Sunk pre-path acquisition cash is excluded;
the initial mark is not proof of an executable immediate-liquidation comparator.

`HedgedInventoryCashPath` requires the same supported underlying, USDT unit and
boundary clocks on both legs. It reuses `replay_fixed_base_inventory` for the
short linear perpetual, including exact settlement-mark funding and uncertain
boundary entitlement. Perpetual executions/costs remain its proportional
supplied model. Financing, opportunity and other quote debits are declared
separately and deducted once. They must exclude already recognized fees and
avoid counting financing and opportunity costs twice on the same capital.
Equal initial wealth, feasible alternatives, funding sources and collateral
eligibility must be independently qualified; supplied cost numbers do not prove
them. This is a wealth-change calculation, not available margin or wallet cash.

Each interval retains `spot quantity + perpetual quantity` as residual base
exposure. Each boundary also encloses the possible unmatched quantity across
the spot fill sequence and the derivative's monotone transition. A matched
interval can still have a nonzero entry/exit orphan range. Those independent
enclosures do not claim jointly attainable extrema, executable fill likelihoods,
cash-loss bounds, atomic entry/exit or continuous solvency.

## Exact constructed examples

All following numbers come from the generated controls and share a declared
quote unit. They are not actual Binance prices, commissions or funding returns.

| Construction | Exact consequence |
| --- | --- |
| One net spot unit versus one short perpetual unit; constant one-unit basis | Common price changes cancel at exits 80 and 120 before costs. Entry/exit orphan ranges still enclose minus one to plus one base unit. |
| 100 quote units on each leg, with entry prices 100 and 101 | Perpetual quantity is `100/101`, leaving `1/101` directional base exposure. A common 20-unit move produces `20/101` profit or loss. |
| Gross spot BUY one unit with 0.01 base commission; hedge one full unit | Received inventory is 0.99. At the supplied exit the combined change is `-6/5`, with a `-1/100` base residual. |
| The same spot fee, hedge reduced to 0.99 | Common price changes cancel; the original base fee still costs exactly one quote unit. |
| Supplied funding credit 0.11 | Spot quote fees 0.22, modeled perpetual costs 0.1332 and external debits 0.09 leave `-833/2500`, or -0.3332. Positive funding alone is insufficient. |

Partial fills, resizing, incumbent inventory, flat abstention, boundary funding,
intragrid orphan exposure, invalid inputs and source changes have direct checks.
The cash/funding domain has 290 distinct passing cases, including 45 new cases;
five artifact checks separately verify exact reproduction, source/intent binding,
exclusive creation and output privacy. Tests were staged; no full trading suite,
coverage benchmark, GPU fit or market collector was rerun.

Reproduce only when source changes invalidate the receipt, with a new output:

```powershell
uv run --no-sync python -m tools.reproduce_hedged_inventory_cash --output <new-controls.json>
```

## Remaining qualification

The result keeps financial qualification, venue-margin qualification, training
admission and accepted-edge flags false. It is not an authenticated fill adapter,
admitted native dataset, equal-wealth funding model, independent supervisor,
continuous optimizer or newly trained policy. It is not wired into inference or
order submission. Native paired coverage, complete fee-asset and execution
semantics, causal roles, independently justified costs/alternatives, basis exits,
orphan unwind and continuous solvency remain required before fits or deployment.

Current registry rows and trigger conditions were reviewed. The holding-yield
family has no newly established material official change; its consumed refresh
is not reopened. The quarterly fee-account gate is not satisfied by testnet
credentials. No market population, endpoint, protected partial evidence or
promotion was reopened to produce this checkpoint. Counts remain 203/65/37/0.
The registry is unchanged because constructed accounting controls are not new
market observations or accepted mechanisms. Continue with native paired
cash/cost qualification and transition-aware forecasting, not funding-only fits.
