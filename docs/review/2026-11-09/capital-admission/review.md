# Bind capital approval to the state actually reserved

Three offline regressions exposed gaps remaining after the
[unresolved-order admission repair](../order-admission/review.md).

| Change during signing | Previous behavior | Repaired behavior |
| --- | --- | --- |
| Another opening completes its fill and becomes terminal | The pending-order gate no longer sees it; the stale approval can exceed aggregate risk | The changed ledger revision rejects the old opening decision before reservation or dispatch |
| Order limits become stricter | The previously approved order can still dispatch | Captured limits must still match; limit updates and opening dispatch are ordered by a coordinator lock |
| Another close reserves the same parent lot | Both closes can use the same stale available quantity | The second close fails the current parent/lot check inside the writer transaction |

The completed-fill scenario uses a 0.65-unit order ceiling and 0.70-unit aggregate
ceiling: each 0.61-unit declared order is individually valid, but the first
confirmed 0.51-unit acquisition leaves insufficient aggregate capacity for the
second order. The correction rejects the stale decision rather than silently
resizing, reapproving or retrying it.

Openings capture the immutable ledger audit tip before preflight, then preserve
that `PolymarketLedgerRevision` through the required `expected_revision` admission
argument. The existing `BEGIN IMMEDIATE` transaction checks pending orders and
compares the actual tip before inserting the prepared record. An intervening
completed fill, redemption or other audited ledger mutation cannot bypass the
fence merely by leaving no currently pending order. The normal coordinator is
the production caller; low-level evidence recording still grants no submission
authority. This fence depends on the existing audited-writer/integrity boundary.

The coordinator uses one captured immutable risk-limit value for the opening
calculation. A changed value rejects the approval. Its setter and the final
opening dispatch share an `RLock`, ordering concurrent in-process policy changes
against that dispatch. This is per-coordinator synchronization, not persistent
or cross-process policy governance. Closing does not acquire this opening lock.

Closing admission instead rechecks the exact parent identity, terminal status,
confirmed quantity and existing outstanding close reservations in the same
SQLite writer transaction as insertion. It does not use a global ledger-revision
veto. A separately added unrelated opening reservation therefore leaves a valid
owned close available. Two independent connections competing for the same lot
admit exactly one close. Runtime Stop and ownership rules remain controlling.

The existing lot arithmetic is reused through connection-bound readers. Public
lot reads now establish one SQLite snapshot for fills, orders and redemptions;
transactional admission uses its existing writer connection without nested
connections or another initialization. A real second-writer control inserts a
close between reader queries: the first snapshot consistently reports zero
reserved quantity, while a fresh read reports the new one-unit reservation.
Syntax-tree comparison confirms unchanged lot arithmetic apart from redemption
reader plumbing, and unchanged order/fill/redemption validation and condition
accounting methods. Read connections close on every exit.

**248 focused checks pass**, covering live orders, user-stream subscription,
settlement, runtime control and implementation manifests. The three corrected
regressions fail on the retained original module and pass on the repair. Before
snapshots, logs, XML and current source bindings are retained in
[verification](verification.json). The first completed-fill fixture had invalid
inconsistent ceilings and failed before exercising its target; its original log
is preserved separately and excluded from proof. The corrected before-run loads
the retained original module under its real package identity in an isolated
process, with unchanged dependencies. No full-suite or timing claim follows.

No market capture, credential use, account request, venue order, training or
persistent worker occurred. Prior financial results are untouched. Native
balance/allowance changes, independently changing venue state, cross-process
policy updates, full crash recovery and independent supervision remain outside
this acceptance scope. Qualified stable edges remain **zero**; the full trading
R&D, application revamp and final-review goal remains active.
