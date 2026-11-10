# Runtime control storage and recovery

Normal runtime-control access never creates a database, control namespace or
interlock. It requires a ready enrollment file, matching resource identities and
the same committed store UUID. Lost, replaced or malformed storage blocks
opening instead of becoming fresh stopped state. The independent enrollment
contains no credentials or paths; its local filesystem identities are private
installation state and must not be committed or embedded in application defaults.

The default enrollment filename is derived from the selected database and
controller namespace. Configure `--runtime-enrollment` consistently when keeping
the policy file in a separate owned directory. SQLite's database, WAL, SHM,
journal and dispatch-lock paths cannot be used as the enrollment file. Software
paths remain configurable; a filesystem without stable ordinary-file identities
is rejected rather than silently weakening the guard.

## First setup and legacy adoption

The local `polymarket-live` actions `inspect-control`, `enroll-control` and
`recover-control-enrollment` do not construct an authenticated venue or request
credentials. They create no trading authority. Use the normal configured
`--ledger`, and the same optional `--runtime-enrollment`, for every related action.

1. Stop the prior application through its normal control flow before adopting
   storage. Preserve the original ledger, policy and unresolved obligations.
   A control checksum alone does not establish that accounts or positions are
   reconciled, and a stale PID is not authority to kill another process.
2. Use `--action inspect-control --json` to inspect an existing local namespace.
   It reports the exact validated `control_record_sha256`, enrollment state and
   store ID, with `new_exposure_authorized` false. Missing or corrupt data is an
   error, not evidence of flatness. Inspection does not create or adopt storage.
3. For a deliberately new namespace, use `--action enroll-control` without an
   expected record digest. Existing control rows reject that shortcut. For a
   legacy namespace, supply its exact inspected value through
   `--expected-control-record-sha256`. Adoption preserves the record, including
   Stop history and any current lease; supported legacy pause migration retains
   its original hash checks. An already enrolled namespace cannot be rebound.
4. A newly provisioned namespace begins `stop_requested`, paused, at Stop epoch
   1. Provisioning and enrollment recovery do not clear it. Rearm requires the
   separately authorized owned-exposure reconciliation/Stop completion flow and
   all existing account, model and lifecycle admission. Do not derive a flat
   declaration from a new file, empty control record or matching hash.

Binance uses the same core through `BinanceRuntimeControl.provision_storage`,
`inspect_storage` and `recover_storage_enrollment`. Its independent namespace
does not inherit Polymarket authority. Native Binance roundtrip integration is
still outstanding; this API is not proof that every Binance writer is fenced.

## Interrupted enrollment

Enrollment writes a prepared marker, commits its UUID and control namespace in
one SQLite transaction, then publishes the ready marker atomically. These are
not one cross-file atomic transaction: incomplete setup deliberately stays
blocked. Never delete a prepared marker or automatically restart provisioning.

If inspection confirms a committed prepared store, retain its
`enrollment_store_id` and `control_record_sha256`. Use
`--action recover-control-enrollment` with both
`--expected-control-store-id` and `--expected-control-record-sha256`. Recovery
requires the same enrolled resources, committed UUID and exact current record.
It only finishes enrollment; it preserves Stop, current ownership and all other
application obligations. The operator may inspect and recover this metadata
locally without account access.

An exit before database commit leaves no qualified store to finalize. Missing
resources, an uncommitted UUID or a changed record reject recovery. Restore the
original enrolled resources where possible and keep opening blocked. Verified
relocation/replacement migration remains an explicit unfinished requirement;
editing file identities, removing guards or manufacturing a fresh namespace is
not a supported migration procedure.

## Scope and remaining work

Owned closing retains its separate enrolled lock and can serialize even when
the control database or opening record is unavailable. A configured custom
enrollment is reused through the coordinator's actual runtime interlock. Missing
or replaced close serialization is rejected rather than split into a new lock
domain. The guard supplies neither missing owned-position data nor authority to
cancel foreign orders. Unenrolled legacy storage needs explicit adoption.

The guard detects enrolled file/namespace loss and ordinary replacement. It does
not detect all in-place valid rollback, authenticate against an actor able to
rewrite every local authority file, preempt a frozen native process, or prove
reboot/power-loss acceptance. Independent process/generation fencing, credential
isolation, verified data migration and complete caller coverage remain required.
See the [source-bound review](review/2026-11-10/runtime-storage-enrollment/review.md).
