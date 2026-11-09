# Atomic admission after preflight and signing

An offline regression reproduced a stale-decision race in Polymarket submission.
Order A passed preflight with no unresolved orders. During its signing step,
order B passed preflight and became live. A then resumed and reserved/submitted
without rechecking that newly unresolved order. The previous global opening
policy existed, but its preflight decision was separated from reservation.

Execution now uses `PolymarketLiveOrderLedger.reserve_submission`. Its existing
`BEGIN IMMEDIATE` transaction checks the canonical unresolved states before
inserting a new opening reservation. A pending record is hash-verified and rejects
the second opening before its reservation or network dispatch. SQLite writer
serialization protects separate ledger connections and processes sharing this
database. An index on order state supports this check without an additional full
terminal-history scan. No latency improvement is claimed or benchmarked.

The lower-level `reserve` remains an evidence-recording operation, preserving
historical/recovery fixtures; it does not grant submission authority. The sole
production Polymarket coordinator call now uses `reserve_submission`, and agent
instructions require that path. Exact repeated prepared records remain idempotent;
the coordinator's state transition still prevents duplicate dispatch. The new
admission check and row insert commit or roll back together.

Closing-only requests keep their separate ownership, lot quantity, runtime and
reconciliation checks. An unrelated unresolved order does not acquire a new power
to prevent an owned reduction. This change does not assert atomic revalidation
of every capital limit, remote balance, fill, redemption or close quantity.
Races involving completed fills or changed risk limits, independent supervision,
crash/reboot recovery and complete capital protection remain open work.

The regression first failed on the published implementation because the expected
block did not occur. That failure log and XML are retained. After the repair, a
deterministic signing interleave permits one simulated submission and retains
only its admitted record. A concurrent two-connection control permits exactly
one prepared opening and blocks the other; replaying the admitted declaration
does not add a row. The source-bound before/after evidence and its scope limits
are in [verification](verification.json).

**240 focused checks pass** across live-order behavior, user-stream subscription,
settlement, runtime control and implementation manifests, including the existing
close/cancel/Stop paths. Ruff passes for changed Python. This is focused software
verification, not full repository/security acceptance or native account execution.
Tests use isolated synthetic venues and owned temporary ledgers. No credentials,
account calls, venue orders, protected capture, model fit or persistent worker was
used. Historical numerical results are unchanged. Qualified stable edges remain
zero; the whole research, application and final-review goal remains active.
