# Position mutation integrity repair

Reproduced at `f63f27e6`: a temporary open-position file containing `{broken`
was silently replaced when `record_open` used the legacy forgiving loader.
Non-object and incomplete rows could likewise disappear during mutation.
This could erase unresolved ownership evidence; an atomic replacement alone
does not make a read/modify/write operation safe.

The shared store now uses strict, lossless admission for open, remove, full-close
and partial-close mutations. Invalid JSON/UTF-8, wrong payload shape, incomplete
rows, duplicate keys, ambiguous open IDs and nonfinite JSON numbers reject.
Both retained files are loaded before the first close write. No corrupt file is
automatically repaired, deleted or overwritten. Valid multiple partial-close
records may still share a position ID. Legacy read-only projections remain
compatible but are not evidence that a ledger is empty or safe to mutate.

Verification: 241 affected checks passed, followed by 43 final focused checks
after preserving the legacy invalid-encoding exception (242 distinct checks
across stages). Ruff passed. Tests assert corrupt bytes and the other ledger
remain unchanged. Source hashes and exact limits are in the [evidence record](position-mutation-integrity.json).
No network, account, real ledger or protected research data was accessed.

This repair is not a multi-file transaction, concurrent-writer fence,
missing-file detector, complete semantic validator or exactly-once close ledger.
An OS crash between two successful writes is still a separate recovery problem.
Next work must integrate transactional inventory/native fees and recovery with
the existing scoped intent journal, then explicit rearm and process fencing;
do not create a competing source of truth or rerun this unchanged suite.

The documentation-maintenance skill keeps this scoped result separate from
the still-unfinished enterprise-readiness and final whole-codebase review gates.
