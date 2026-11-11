# Apply retained Spot terminal closes without replay

Receipt-backed Spot closing recovery now has a programmatic application boundary.
It reconciles the existing paired position store and resolves the particular
close intent in one recoverable transaction. It sends no order, collects no
account data and does not rearm. The reviewed base is
`171bf3a1b771de30f742328a54e3412940456498`.

## Application and authority

`closing_recovery_sha256` inspects the exact retained closing record through a
noncreating connection. `apply_spot_closing_recovery` requires that inspected
digest, the original execution scope, ready execution enrollment and the matching
Binance controller. It rejects implicit setup, missing/replaced storage, active
ownership, unsupported products and conflicting records.

The new stopped-recovery guard holds both the dispatch and owned-close interlocks.
Control must be ownerless, paused and Stop-latched, with no lease/start/heartbeat
ownership. Existing account, process-generation and lost-owner recovery requirements
are not replaced by this precondition. No exposure-closed declaration, automatic
Stop completion or new lease is used by the applier.

Inside the journal writer transaction, the applier revalidates the original
request, terminal order, exact fills and native commissions. The current owned
lot must equal its original template less any already recorded partial fill.
A previous partial cash receipt must have matching native trade IDs, prices,
quantities, commissions and assets in the retained terminal fill set. Unknown
prefix lineage rejects instead of guessing incremental fills.

Terminal evidence is cumulative. The applier replaces the matching partial
closed-trade projection with the actual terminal cumulative projection, rather
than appending the cumulative quantity again. The audit receipt retains the
previous partial record and source digest. Unrelated ledger values are preserved;
the resulting closed-trade projection is ordered by native completion time and
stable identity. Quantity arithmetic shares the ordinary close remainder helper.

FILLED can remove the lot. CANCELED, EXPIRED and EXPIRED_IN_MATCH can retain an
exact positive residual. A zero-fill terminal result, including REJECTED, resolves
only that order and leaves the owned lot intact. Native quote/base entry fees and
quote exit fees feed the existing receipt-derived cash projection. Unvalued
third-asset fees, SELL base fees, unsupported rebates and legacy openings without
native cash evidence remain rejected. No missing fee is assigned zero value.

The checksummed application receipt, close-intent transition and paired redo
record commit together before either JSON projection is published. The existing
paired recovery finishes an interrupted publication. Repeated application validates
the same source and cumulative projection and does not add another trade. Derived
feedback is refreshed afterward; its failure is explicitly reported as potentially
post-commit, with inspection/retry instead of order replay.

## Two integration defects repaired

Review found that shared runtime `release("")` could clear ownerless Stop-latched
control. Both Binance and Polymarket reproduced the defect before repair. Release
now requires an exact nonempty lease and uses the dispatch interlock. Its existing
valid-owner behavior is retained. A competing valid release cannot pass a held
submission guard; an unowned release cannot clear the Stop latch.

Zero-fill closes create no filled ledger row. The old next-close ID calculation
could therefore reuse the canceled order's identity. A separate reproduction
failed after successful zero-fill application. The generator now checks the
consumed close-intent identities, including other lots' identities, before selecting
the next ID. Preparing a new close consumes a different ID; replaying the older
application does not resolve the newer UNKNOWN obligation.

## Evidence and limits

The [sealed evidence](evidence.json) binds final affected verification and separate
source phases. The final **682 affected checks passed** across application,
shared/venue controls, storage, ordinary execution, native cash, positions and
autonomous accounting. Local tests use actual enrolled SQLite/JSON storage and synthetic
venue responses. They exercise full/partial/zero terminal outcomes, two competing
appliers, both interlocks, corrupt evidence/application records, changed inventory,
lost storage, post-commit feedback failure and repeated-call resource bounds.

Two owned child processes exited after the database commit and after the first
JSON replacement respectively. Reopening finished the intended pair exactly once;
the original Stop remained latched. These tests establish those process-exit
paths, not physical power-loss, filesystem rollback or arbitrary host-failure
acceptance. The resource check concerns this test process, not a universal leak
or performance claim. No other PC task was stopped or reconfigured.

An initial fixture used an unsupported completion argument and was corrected
before behavior qualification. A later missing-file test wrongly assumed an empty
ledger already existed; it now checks absence preservation. Both failed source
phases remain retained. The empty-release and consumed-ID failures are separately
preserved as reproduced production defects.

This API is not automatically invoked by CLI/native workflows. Operator inspection
and application integration, native futures/legacy recovery, current account UID,
filter/dust qualification, explicit rearm and independent supervision remain open.
The user decides release. No actual user ledger, credential, account or order was
accessed; no scheduled/background launcher was installed.

No new financial source or market capture occurred. Historical results and
protected partials remain untouched. Counts remain 206 observations, 65 hypotheses,
37 historical scoped mechanisms and zero qualified stable edges. Full product
and profitable-edge work remains active.
