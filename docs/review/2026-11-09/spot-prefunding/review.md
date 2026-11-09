# Spot prefunding before portfolio wealth

The existing spot/perpetual replay correctly reports incremental wealth, not
available wallet cash. Its output alone did not expose the peak quote cash needed
to execute the supplied spot fills. The new `spot_prefunding` analysis preserves
that separation and supplies a necessary cash constraint for future carry
evaluation. Existing replay modules, input contracts and historical result bytes
are unchanged.

## Financial calculation

For initial quote cash `C`, supplied buffer `B`, and ordered incremental fill
cash deltas `d_i`, let `D_j` be every cumulative fill prefix, including `D_0 = 0`.
The cash model is:

```text
cash_j = C + D_j
minimum_initial_quote_cash = B - min_j(D_j)
additional_quote_cash_required = max(0, minimum_initial_quote_cash - C)
```

Including the zero prefix keeps the required cash at least the buffer. Every fill
is checked before boundary aggregation, so a buy and subsequent sale cannot hide
temporary cash demand in their net boundary cash. The implementation uses exact
Fractions, the existing fee-aware quote deltas and the existing inventory-validity
replay. It checks terminal quote conservation. Its traversal is linear in fills
and boundaries; no market dataset, numerical benchmark or GPU fit is needed.

Initial marked inventory remains part of initial wealth. Its sale can supply
quote cash under the explicit immediate-proceeds model; its unrealized price
appreciation cannot pay an acquisition. A base commission reduces inventory
through the existing accounting law, without an invented second quote debit.
Quote commissions consume cash at the fill. A supplied buffer applies to initial
cash and every prefix. Equality is cash-feasible under this model; it is not a
strict safety margin or native margin qualification.

`evaluate_hedged_spot_prefunding` composes this constraint with the existing
paired replay. Derivative wealth and funding receipts never become spot-wallet
credit through an assumed transfer. Opportunity charges in a wealth comparator
are likewise not silently treated as wallet payments. Native financing/fee
debits still need their actual payment wallet and settlement clock.

## Constructed consequences

All values below are synthetic same-unit cash identities, not market returns.

| Supplied fills | Necessary initial quote cash, zero buffer |
| --- | ---: |
| Buy for 100, later sell for 120 | 100, despite ending profit 20 |
| Buy for 100, sell for 100 at the same boundary | 100, despite zero net boundary cash |
| Own one marked unit; sell for 100 before repurchasing for 100, then unwind | 0; initial marked inventory still contributes 100 to initial wealth |
| The same incumbent inventory, but repurchase before the first sale | 100, despite identical ending cash |
| Gross BUY for 100, quote commission 1 and base commission 0.01 | 101; base fee is not charged twice |
| Abstain with no fills or inventory | Only the separately supplied buffer |

A paired control earns 11 supplied funding units before a 5-unit opportunity
charge, giving positive wealth change 6. With no initial spot quote cash, its
necessary 100-unit acquisition budget still fails. Providing 100 fixes only this
spot cash constraint; derivative margin and continuous solvency remain unproved.

The [frozen intent](intent.json) permits zero network/data/account/order/fit
access. [Seven retained controls](controls.json) bind exact rational inputs and
LF-normalized source digests. Source identifiers in those fixtures are explicit
synthetic declarations, not native evidence. Reproduce after source changes
invalidate the receipt, using a new destination:

```powershell
uv run --no-sync python -m tools.reproduce_spot_prefunding --output <new-controls.json>
```

## Verification and boundaries

All **70 focused checks pass**, including 20 new cash-prefix checks, 45 existing
paired-cash checks and five historical artifact controls. The latter reproduce
the prior cash report exactly; no old result was regenerated or requalified.
Authored Python passes Ruff. The tests cover order-sensitive demand, partial
fills, both fee units, initial-buffer equality, abstention, invalid cash types,
inventory validation, input binding and exclusion of derivative wallet credit.
This is focused accounting verification, not a green full suite or a market
experiment.

The cash model assumes immediate fill-proceeds settlement and no outside credit.
Unfilled-order locks, transfer delays, native cash-source/borrow provenance,
additional wallet debits, venue reserves, derivative collateral and continuous
liquidation remain separate requirements. Minimum prefunding is a necessary
lower bound, not sufficient total strategy capital. All capital-source, margin,
training and accepted-edge qualification flags remain false.

Future paired-carry research must report this ordered-fill constraint alongside
source-qualified derivative collateral and an equal-initial-wealth alternative;
do not train or rank a strategy as deployable merely because terminal wealth is
positive. Financial counts remain 203/65/37/0. No source retry, protected partial
data, credentials, order, persistent process or new market observation occurred.
The full R&D/refactor goal remains active.
