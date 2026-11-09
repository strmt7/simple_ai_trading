# Missing runtime control state fails closed

The original initializer treated an existing control table with a missing record
as a fresh application. An isolated SQLite reproduction latched Stop at epoch 1,
deleted only that record, and then observed `stopped`, epoch 0 and an admitted
replacement lease. A read could erase evidence of the missing Stop state.

The repaired initializer creates the table and initial record together under one
writer transaction only when the namespace is absent. It rechecks after acquiring
the writer, so a stale absence observation cannot overwrite another initializer's
committed state. An existing namespace requires its valid record. Missing current
or legacy records raise an error without manufacturing state or advancing a
partial migration.

Legacy migration verifies a concurrently completed valid migration instead of
trying to migrate it twice. Cancellation during bootstrap or migration rolls
back its transaction and closes the SQLite connection. Existing checksums,
schema version, state transitions and risk budgets are preserved. The separate
close guard remains usable without recreating a missing record; this does not
replace the ledger's ownership and native-close qualification gates.

## Verification

Seven targeted regressions fail against the exact source before this storage
repair: missing-record reads and acquisition, missing legacy state, cancelled
bootstrap/migration, non-atomic first creation and a stale competing migrator.
All **61 current runtime-control, interlock and autonomous-runtime checks pass**,
with no skip. They also cover record loss around migration, stale first-use
observations, actual concurrent readers, rollback/connection closure and the
independent close guard. Existing test functions are AST-identical; the new tests
extend their coverage. Ruff lint and formatting checks pass.

The [evidence](evidence.json) binds exact implementation/test hashes, source
navigation receipt, original reproduction and private logs. Production changes
are limited to connection cleanup and initialization/migration helpers. The
previous interlock cleanup and all other production definitions are AST-identical
to this phase's baseline. The [earlier phase reconstruction](../../2026-11-09/interlock-cleanup/source-reconstruction.json)
preserves the exact module used for its original 29-check result; the current
combined verification is the 61-check result here.

## Remaining protection boundary

This detects a missing record in an existing control namespace. Whole database
or complete namespace deletion/replacement still needs account-scoped storage
enrollment, a durable identity witness and controlled bootstrap/recovery. No
physical power-loss, reboot, rollback resistance or adversarial rewrite proof
is claimed. A checksum alone does not supply those guarantees.

The independent trading supervisor, credential-separated gateway, complete
order-caller inventory, bounded native calls, persistent capital policy and
model-health quarantine remain open in the [capital architecture](../../../CAPITAL_PROTECTION_ARCHITECTURE.md).
This repair came first because it closes a demonstrated state-reset defect.
Semantic navigation supplied candidate process boundaries, not proof that every
order-capable path has been reviewed. No native market/account request, credential,
order, real user database, protected capture, service or persistent worker was
used. Financial counts stay **204 / 65 / 37 / 0 qualified stable edges**.
