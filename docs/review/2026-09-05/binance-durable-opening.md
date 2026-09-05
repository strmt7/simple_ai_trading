# Durable autonomous opening obligations

The previously observed crash window is now guarded: the autonomous Binance
opening path commits an UNKNOWN intent before invoking the venue adapter.
The initial regression reached the actual loop's stubbed submission with no
pending intent; it now proves the pending record exists before that boundary.
[Source bindings and verification](binance-durable-opening.json) distinguish
this engineering result from profitability or complete capital protection.

The per-position-store `binance_open_intents.sqlite3` journal uses a SQLite
transaction with `synchronous=FULL`. It retains the exact client/position
identity and original position/risk template, not credentials or signed HTTP
requests. Position and closed-trade accounting remain in their existing ledgers;
this journal records unresolved execution obligations, not a second inventory.
Duplicate identities are rejected even after completion, and concurrent
preparations cannot admit two unresolved openings through this journal.

An acknowledgement must explicitly match client identity, symbol and supported
fill status. Contradictory returned identities or side are rejected. Local fill
identity, finite values and transmitted-quantity consistency are checked before
persistence. A full fill releases
admission only after the position write; a partial fill leaves UNKNOWN active.
Position/closed-ledger writes now flush bytes before atomic replacement and
reject nonfinite JSON instead of writing nonstandard numeric values.

The shared lifecycle plan blocks new non-paper exposure on unresolved or
unreadable intent state. That check does not itself block separately verified
owned closes. Recorded intent history also blocks entry when its position file
is missing. A fresh legacy store without this journal remains compatible; that
does not establish detection of a deleted or rolled-back journal.

## Verification and limits

The final affected-domain run passes 216 tests. Nine selected CLI integration
checks also pass: 225 distinct affected checks, not a whole-repository audit.
The new journal file contains 40 cases covering normal Spot/Futures submission,
duplicate prevention, malformed storage, invalid/mismatched responses, partial
fills, ledger-write failure, interruption before and after position persistence,
concurrent preparation and restart. A task-owned subprocess exits abruptly with
code 71 inside a stubbed submission; a new reader still finds its UNKNOWN intent.
No venue request, order, real credential or unrelated process was involved.
After the domain run, the partial-fill fixture was strengthened to supply half
the requested quantity; that exact test passed again without another broad run.

The restart fixture was tightened after review found that missing credentials
and a zero synthetic risk budget could independently block startup. Its final
assertion proves the pending intent is the only blocking lifecycle check before
exercising startup. A range-format invocation on the legacy autonomous test file
hit Ruff's `start.raw <= end.raw` panic; the small formatting change was applied
directly, and changed-file lint passes. Expanding the retained recovery template
also exposed a serialization error-type inconsistency; the focused regression
now verifies the common intent error without weakening rejection.

## Next work, not implied completion

UNKNOWN has no automatic reset in this checkpoint. Do not delete or edit the
journal to resume, infer rejection from a timeout/not-found response, or replay
an order. Implement explicit exact-order recovery next, retaining account and
endpoint identity, full/partial/zero-fill terminal evidence and operator rearm.
Existing verified close/reconciliation gates remain controlling meanwhile.

Only autonomous openings now create these intents. Lower-level APIs, CLI entry
and roundtrip paths, and closes still need complete gateway integration.
The preparation transaction is not end-to-end multi-worker fencing or a capital
reservation system. Generation/policy binding, independent supervisor/process
containment, and transactional consolidation of open/closed accounting remain.
SQLite FULL and flushed file replacement are not a completed power-loss,
filesystem-directory durability, rollback/deletion or OS-reboot qualification.
No model training, market retry, accepted-edge count or historical result changed.
