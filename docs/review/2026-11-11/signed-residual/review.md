# Preserve signed partial-close residuals

A signed futures partial close could remove its positive remaining lot from
the active ledger. At base `709d77692269efc4d609f0c359f7df131420c1f8`, the
durable gateway's synthetic long and short cases each submitted quantity 1 and
received cumulative quantity 0.999999999 with `PARTIALLY_FILLED`. After reopen,
the active ledger was empty even though the close remained UNKNOWN. Both
regressions failed before the repair.

The cause was `PositionsStore.record_close_result`: positions without original
native Spot receipt fields used an epsilon-based full-close decision. That
included futures and legacy signed Spot. The tolerance could erase a positive
residual, depriving later recovery of its owned lot. UNKNOWN still blocked new
exposure; this was lost inventory visibility, not a reproduced unauthorized
opening or proven loss of exchange funds.

## Resulting behavior

Every signed close now subtracts the complete supplied decimal quantity
projections using exact rational arithmetic. Only zero remainder removes the
lot. Positive remainders survive in the paired store; oversells reject before
mutation. Nonpositive, nonfinite, Boolean and unsupported quantities also reject.
If the result cannot be stored exactly through the current decimal/float
projection, accounting rejects instead of rounding away a movement. Original
UNKNOWN obligations remain available for reconciliation.

Exact arithmetic does not depend on caller decimal precision, exponent limits
or subnormal traps. Existing native Spot receipt identity checks remain, and
paper-only quantity tolerance retains its prior behavior. No new dependency,
venue request, order retry, authority, Stop completion or rearm was added.

The final gateway cases retain quantity 0.000000001 after restart for both long
and short futures. A second close is rejected before another synthetic order
while the original close is unresolved. Direct paired-store cases cover signed
Spot and both futures directions at two scales, exact full closes, invalid
quantities, oversells, unrepresentable residuals and paper compatibility.

These are synthetic admitted-input tests, not evidence that those particular
quantities satisfy a current instrument's filters or that the remainder can be
traded. Minimum-quantity, dust disposition and original wire-precision evidence
remain separate obligations. Exact supplied projections cannot recover quantity
digits already lost before this accounting boundary. Native cash, commissions,
account availability and fee valuation are not inferred from this repair.

## Verification and research disposition

The final **446 affected checks passed** across positions, paired transactions,
durable closing, terminal closing evidence, closing inventory, owned runtime,
autonomous behavior, Spot inventory/cash and capital valuation. The initial
two-case failure and intermediate source phases remain retained separately.
A final arithmetic improvement replaced decimal-context-dependent subtraction
with rational subtraction; the affected domain was rerun for that change, not
for documentation. Ruff passed; range formatting avoided rewriting unrelated
legacy code. The [sealed evidence](evidence.json) binds exact source, archives,
logs and the semantic-search receipt.

The fresh source index and actual CLI semantic search routed recovery code.
The registered MCP transport had closed; the reviewed CLI adapter succeeded
against the same current index. Its raw and fused receipts remain private.
That source snapshot predates this patch and is not claimed current afterward.

Financial discovery did not establish a new Auto Order Routing or SOR retry
trigger. Search results were not admitted as page content. No web source
extraction, market capture, credential/account access, order, fit or benchmark
was performed. The financial frontier remains 206 observations / 65 hypotheses /
37 historical scoped mechanisms / zero qualified stable edges. No protected
partial capture or historical result was changed.

This repairs one prerequisite for safe recovery. Applying retained terminal
movements exactly once, native fee/account qualification, explicit rearm,
independent supervision and full application qualification remain active work.
