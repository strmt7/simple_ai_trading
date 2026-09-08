# Durable scoped Binance closing obligations

September 8, 2026. [Source-bound checkpoint](binance-durable-closing.json).

## Demonstrated defect and repair

At baseline `de8c0ad28a1c991c79bb1368461e113fc4e550aa`, the existing partial-close
retry test succeeds with two different close client IDs and two ledger records.
The first acknowledgement is PARTIALLY_FILLED; no evidence establishes that
order's terminal state before the second submission. Its remaining quantity
could still execute. The baseline module and test were executed from their Git
objects in isolated temporary state, without network or user-ledger access.

Both active callers, operator `close_tracked_open_positions` and autonomous
`run_loop`, now use one durable close gateway. The existing scoped SQLite intent
database stores UNKNOWN before transmission. The paired position transaction
serializes admission against the exact current owned lot. A pending close blocks
another close for that lot and both lifecycle/direct-intent new entry; it does
not globally block independently verified closes of other owned lots.

The write and response-loss exact-client-ID query carry the frozen execution
scope. Acknowledgement checks require matching client IDs, symbol and closing
side, positive finite interpreted quantity/price, and FILLED or PARTIALLY_FILLED.
Futures also require one-way BOTH and reduceOnly true. Full interpreted quantity
must match the adapter's eight-decimal transmitted quantity. This is not a full
native raw order/trade reconciliation: the existing float fill interpreter and
modeled-fee calculation remain. They must not establish exact native balances.

Both partial and full close persistence compare the original lot with the
current stored lot before replacement. A full close can move to RECORDED only
after its lot is absent and its exact trade uniquely present in the paired
ledger. Partial closes stay UNKNOWN even after their first fill is recorded.
Crash or accounting/output failure leaves the obligation unresolved, never
permission for an automatic replacement order. A normal full completion is not
an explicit recovery/rearm workflow.

## Remaining work and operating limits

No automatic adoption of legacy/unbound inventory or a different configured
account is allowed. Scope binds the existing journal to origin, product and
API-key identity; it is not independent authenticated account/lot provenance.
Verified migration, rotation and current account reconciliation remain required.

Next implement exact terminal close evidence and incremental cumulative-fill
application with native commissions, using the same authoritative persistence.
Opening and closing evidence need distinct side/reduce-only semantics; do not
masquerade a closing order as an opening to reuse its validator. Join this with
native opening inventory application and explicit policy/account/process-fenced
rearm. Do not clear a pending close merely because a ledger row exists.

Generic `record_close` calls outside the active gateway are not deduplicated.
Database/enrollment deletion, coordinated rollback, out-of-band/old writers,
power-loss or OS-reboot recovery, independent process supervision and complete
model-health controls remain unfinished. No actual user ledger was migrated.
This checkpoint neither establishes profitability nor completes whole-repository
line review, comprehensive bug hunting or enterprise readiness.

## Verification and efficiency

The final twelve-file affected-domain suite passed **422 cases**, including
**38 new close cases**: Spot long, futures long/short, exact-ID response-loss
queries, partial restart, uncertain transport, abrupt child `os._exit(73)`,
competing callers, wrong scope, malformed identity/status, futures reduction,
accounting interruptions, changed inventory and malformed close schema.
All venue interactions use offline fakes. No credentials or protected capture.

The earlier broad suite output was lost during output/context truncation; it is
not counted as passed. One final run retained a local JUnit summary. Old pytest
failure-cache entries referred to removed test names and were not current failure
evidence; a targeted invocation could not collect them. Future checks must resolve
current test names, not infer them from that cache. No full repository suite,
training, GPU benchmark or hosted test campaign was launched for this change.

Pinned Ruff 0.16.5 panicked on range formatting the existing autonomous test file.
Whole-file formatting completed; the fallback caused mechanical changes outside
the behavioral hunk. Do not retry the same crashing ranges or change semantics to
satisfy the formatter. Final source bindings follow formatting; historical source
bindings and research results remain untouched. Test runtime is not a benchmark.
Ten source bindings, changed-line formatting and formatter AST equivalence were
verified separately. Use explicit UTF-8 for subprocess text pipes on Windows;
the default console encoding rejected one initial formatter-stdin check, while
the corrected encoding passed without changing source or repeating the suite.
