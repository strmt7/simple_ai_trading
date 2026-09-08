# Recoverable paired position transactions

September 8, 2026. [Source-bound evidence](position-paired-transactions.json).

The active position store now commits a paired redo record before publishing
either JSON file. This repairs a reproduced failure at baseline `0631ed85`:
interrupting the second close write left the same position in both the closed
ledger and open inventory. Native opening application was deliberately not
wired onto that inconsistent persistence foundation.

## Authority and recovery

The existing `binance_open_intents.sqlite3` database holds one checksummed
`position_replacement` record, with exact before/after bytes for both JSON files.
`BEGIN IMMEDIATE` serializes participating writers and readers; synchronous FULL
commits PREPARED before either flushed replacement. Recovery first checks both
files against the retained before/after pair, then finishes only that committed
operation. APPLIED subsequently requires exact current bytes. Unexpected edits,
deleted enrolled files, malformed records and unavailable locks reject access
without overwriting conflicting evidence. The checksum detects inconsistent
bytes, not an attacker who can rewrite both data and checksum.

The transaction releases its SQLite lock when committing PREPARED. Another
participant can finish it and transact again before the original writer resumes.
The original writer therefore recovers the **current** database record, never
republishes its stale snapshot. This matters for avoiding lost concurrent writes.

`record_open`, `remove_open`, full/partial close and normal reads use this boundary.
Partial closes also reject a changed/absent source lot. Statistics read both
ledgers in one snapshot. No Binance UNKNOWN state, scope binding or rearm changes.
Learning feedback remains a derived post-commit output, not part of the money
ledger transaction; an output failure must not be treated as permission to replay
the close. Exact repeated-close acknowledgement deduplication remains required.

## Compatibility and limits

An initial read does not create storage or rewrite legacy files. A first valid
mutation enrolls the pair. Thereafter do not manually edit its JSON projections
or run an older writer against the same directory. External direct file readers
do not receive atomic paired visibility; use `PositionsStore.load_snapshot`.
Separate calls to the two individual loaders can still straddle a later valid
transaction. Copy the complete quiescent storage set for backup; moving only one
projection is not a supported migration or recovery operation.

This is recoverable transactional publication for participating access, not an
atomic two-file filesystem primitive. It does not detect deletion of the entire
database/enrollment table, prove power-loss durability, qualify account balances,
deduplicate exchange closes, or implement native-fee application and rearm.
It retains the existing whole-ledger JSON rewrite cost; no speed claim is made.
Native fees, explicit scope/account/policy/process-fenced recovery, and independent
supervision remain next integration work. Historical evidence remains unchanged.

## Verification

384 distinct affected checks across stages, including 29 new transaction cases.
Coverage includes before/between/after projection interruption, actual child
`os._exit(72)`, four simultaneous writers adding twelve distinct positions,
corruption/missing-file preservation, SQL rejection before publication, lock
contention, stale partial closes and initial enrollment. The 361-case domain run
covered autonomous entry/close, execution lifecycle, scope, terminal/native
opening retention and position behavior; later focused stages covered checksum
and enrollment changes, including rejection of a first mutation without leaving
an unusable empty database. Ruff passed. No real ledger, exchange, credential,
account, protected research capture or unrelated process was touched.
