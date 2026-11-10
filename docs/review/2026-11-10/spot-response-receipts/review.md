# Complete Spot receipts after response loss

An in-flight ordinary Spot order could be accepted while its response was lost.
The gateway queried that exact order, but a status response without inline fills
could not supply commission-aware opening inventory or closing cash. Two retained
synthetic reproductions show one native write and one status query, zero trade
queries and an unresolved obligation. The [evidence](evidence.json) binds that
original source phase. Existing terminal recovery collectors were found and
preserved; this repair does not duplicate or replace them.

## Repair

For a FILLED opening or FILLED/PARTIALLY_FILLED closing whose queried status lacks
inline fills, ordinary owned dispatch now collects one exact-order trade page.
It uses the same bound observation helper as paired Spot recovery. The helper
verifies request semantics, current execution scope, order ID, symbol, side,
unique trade IDs, quantities, cash and commission assets/amounts. Its 1,000-row
limit is unchanged from the existing native order-trade interface. Cumulative
quantity and cash must reconcile; a page that advances beyond the status snapshot
is unresolved, not silently truncated or retried until it fits.

Opening accounts for net received base after commission; closing retains the
original entry receipt and commission-aware cash. Scope changes after the status
query reject before the trade query. Unsupported ordinary order states spend no
additional trade query. The component makes one collection pass and never
resubmits an order, acquires authority, creates storage or releases a journal
barrier itself. Existing native GET transport retry/deadline policy still applies.
The existing admission or closing guard remains held by the owning gateway.

Direct acknowledgements and already complete inline recovery receipts retain
their existing validation. Paired recovery still rejects unexpected inline
query fills and preserves its explicit separate completion phase. Partial owned
closing records only the observed fill and keeps its reconciliation barrier.
Cancellation, missing/foreign fees, ambiguous pages and transport failure do not
manufacture a position or a completed close. Futures execution is unchanged.

## Verification

The new tests exercise the actual Binance client methods with synthetic
dictionary/list transports and a real, explicitly enrolled local controller and
journal. Real HTTP is forbidden. They verify exact status/trade selectors, one
order submission, base/quote commissions, partial closing, advancing trade pages,
scope rotation, cancellation and malformed/unbounded/incomplete evidence.

The first focused lane passed 176 cases. The expanded source-bound lane had
451 cases with one failure in an older close mock that omitted request fields
and an exact trade interface. The mock now supplies the full native-shaped
query and trade page, retains its original assertions and additionally checks
the trade-query count. Its complete 38-case module passes. Evidence replaces
that module's earlier source phase rather than adding overlapping counts or
rerunning unaffected cases. The resulting 451 distinct affected cases pass;
the full application suite was not run. Ruff passes for changed Python files.

The helper borrows the caller's HTTP client and allocates bounded local data;
it creates no session, process, thread, service or restart mechanism. Test clients
close their sessions, and all bounded verification helpers returned. No real
venue request/order, model fit or financial observation occurred. This is
synthetic execution evidence, not native account or profitability acceptance.

## Remaining boundaries

This repairs in-flight receipt recovery only. Existing retained terminal
observations and inventory diagnostics do not by themselves apply or rearm
arbitrary historical obligations. Independent session supervision, generation
and credential fencing, complete native-writer coverage, verified migration,
rollback/host failure acceptance and native capital qualification remain open.
The separate [funding review](../funding-fixing/review.md) preserves the economic
target and source gaps. No stable edge or release is qualified by this batch.
