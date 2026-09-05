# Durable Binance execution scope

New autonomous opening intents bind an official HTTPS venue origin, Spot/Futures
product and non-secret API-key fingerprint before submission. The expected scope
is passed explicitly through orders, futures leverage/bracket calls and exact-ID
queries, then compared to the actual transport snapshot before network access.
See [source bindings and staged verification](binance-execution-scope.json).

## Reasoning and implemented boundary

A matching client-order ID is not sufficient when the configured account or
environment has changed. A key fingerprint identifies a credential scope, **not
a permanent Binance account UID**. A rotated key may belong to the same account,
but that relationship is not inferred or authorized by this implementation.
No API key or secret is persisted in the journal.

Per-intent binding alone was insufficient: after a successful fill, a later key
could otherwise start mixing account histories. Schema 2 therefore has a
transactionally initialized singleton journal scope. New intents and completion
transitions must match it. This is a constant-size metadata check, not a scan of
all prior order payloads on every submission. Missing metadata with existing
history cannot be reconstructed from current settings. Completion updates also
bind the stored position ID and cannot create a missing database.

Readback returns only the exact canonical pending position template after checking
journal scope, payload version, per-intent scope and row identities. It neither
queries a venue nor clears an obligation. Legacy schema 1 remains readable by the
admission barrier, but new writes/recovery need explicit verified migration;
legacy data is not silently attributed to the current credential. No migration
command or automatic rearm is implemented yet.

## Evidence and review

Offline cases change key, origin or product immediately after intent persistence
and prove zero transport calls plus a retained UNKNOWN obligation. A futures case
changes the key after leverage configuration and proves that neither order nor
recovery query reaches the new key. Successful Spot/Futures flows preserve binding
through every signed operation. Other cases cover key rotation after a recorded
fill, invalid/tampered records, legacy preservation, missing metadata, and deleted
storage. The API's unscoped legacy call contract remains available to callers not
yet migrated; autonomous durable openings always supply a scope.

The first affected run passed 392 checks. Semantic review then identified the
journal-history fence described above. After implementing it, all 201 affected
journal/execution/lifecycle checks passed; 194 unchanged API/transport cases from
the earlier run remain applicable, for **395 distinct checks across stages**, not
a claimed single final whole-repo run. One test initially kept its own SQLite
connection open while deleting its temporary database on Windows; explicit
connection closing corrected the fixture. Production connections already use
deterministic closing. Ruff and diff checks pass.

The Python and regression skills guided explicit typed scope propagation and
paired failure/success cases; the documentation skill kept new evidence separate
from prior source-bound results. No actual credentials, venue requests, orders,
protected captures, market-result changes or unrelated process interference.

## Next obligations

Implement exact authenticated terminal-order recovery and explicit rearm using
these boundaries, including partial fills and zero-fill terminal statuses.
Keep unresolved legacy data blocked until verified migration. Finish remaining
gateway callers, close intents, inventory/account scope, persistent policy and
supervisor generations. A deleted whole journal can still look like a fresh path
to admission; comprehensive ledger rollback/deletion detection is not qualified.
Neither process containment nor reboot/power-loss safety is complete. This is
execution-risk reduction, not profitable-edge or enterprise-readiness evidence.
