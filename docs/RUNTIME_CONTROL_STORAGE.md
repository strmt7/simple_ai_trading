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

## Ordinary owned Binance orders

The shared `submit_durable_open` gateway, used by ordinary CLI and autonomous
owned-position entry, requires an already enrolled controller, a current running
lease and an independently enrolled execution journal. It never acquires a
lease, clears Stop or provisions missing storage. Its default controller is
`binance_runtime_control.sqlite3` beside the position store; the caller supplies
the existing lease through `SIMPLE_AI_TRADING_BINANCE_RUNTIME_LEASE`. Keep that
value private and never put it in logs, commands committed to Git, or evidence.

Programmatic callers can instead pass a `RuntimeLeaseInterlock` with their
configured `BinanceRuntimeControl`, plus `execution_enrollment_path` when using a
custom execution marker. Set up or adopt that exact journal with
`BinanceExecutionStorage` while the controller is ownerless, paused and Stop is
latched. The [Spot storage actions](SPOT_ROUNDTRIP.md#execution-storage-setup)
provide the Spot operator flow; they do not provision a futures account. A
supervisor must reconcile real obligations before completing Stop and must own
lease heartbeat and teardown. The gateway borrows that authority; it does not
create an independent supervisor or infer flat exposure from local metadata.

Opening admission holds the existing runtime submission guard through UNKNOWN
precommit, dispatch, acknowledgement validation and durable inventory completion.
It rechecks controller and journal authority before dispatch and never recreates
a missing execution journal. Interrupted or ambiguous execution retains the
existing reconciliation barrier. Signed uses of the legacy nondurable position
helpers are rejected; paper uses keep their existing behavior.

Noncreating access continues through native position accounting and close
completion, including after a valid acknowledgement. Loss during or after
dispatch must not bootstrap an empty replacement journal. The generic position
and journal APIs retain creation for setup/paper callers; the native gateway
explicitly passes `create=False`. If accounting already committed before
completion fails, its receipt can remain in the JSON pair. Preserve available
original obligations and reconcile the actual venue state; failure is not proof
that an order was absent. Never re-enroll simply to make the next order proceed.

`submit_durable_close` uses the separate enrolled closing lock around the entire
owned close attempt. Persisted Stop, loss of the opening marker, or loss of the
controller database does not itself disable this reduction path. Lost closing
serialization still rejects the attempt; closing still needs the original owned
inventory, scope and exact acknowledgement. These are local synthetic-path
verification results, not authority to trade or proof of venue execution.

After an ambiguous ordinary Spot submission, a usable exact status response
without inline fills now receives one bound order-trade-page collection. The
same helper serves paired Spot recovery. Scope, order/side, cumulative cash and
quantity, and native commissions must reconcile; incomplete or advancing pages
retain uncertainty. The order is never resubmitted. This is in-flight receipt
recovery, not automatic application/rearm of historical terminal observations.
See the [source-bound repair](review/2026-11-10/spot-response-receipts/review.md).

Signed close accounting [preserves positive residuals](review/2026-11-11/signed-residual/review.md)
without a full-close epsilon, including futures and legacy Spot. An oversell or
remainder that cannot be represented exactly rejects before paired-store mutation.
This preserves the original UNKNOWN obligation; it does not apply retained terminal
observations or establish that a residual meets native filters or available cash.

## Retained Spot closing application

The [terminal application checkpoint](review/2026-11-11/terminal-application/review.md)
adds a programmatic API for existing receipt-backed Spot closes. It does not
construct a client, request credentials or query an account. Independently
authorized read-only collection must already have retained the exact terminal
order and complete native fills in the original scoped journal.

`closing_recovery_sha256(store, scope=scope, client_id=client_id)` inspects that
record without mutating ledgers. Supply its digest as `expected_recovery_sha256`
to `apply_spot_closing_recovery`, together with the same store, scope, client ID,
Binance control and any explicit `execution_enrollment_path`. These are configured
objects; never substitute a guessed current account for the original scope.

Application requires ready original execution enrollment and ownerless, paused,
Stop-latched control. End the owned session normally before recovery. If its
owner or lease remains unresolved, preserve that state; do not declare exposure
flat, erase ownership, delete guards or provision another namespace to pass this
gate. Latching Stop on an already ownerless controller is local control, not an
account-flatness or rearm declaration.

The applier replaces a matching partial ledger projection with its verified
cumulative terminal projection and atomically records its application and resolved
intent through the existing paired redo. Zero execution leaves the owned lot
unchanged. Prior fill IDs and native cash/fee evidence must reconcile. Unsupported
fees, products, missing lineage, changed inventory and unrepresentable residuals
reject. Both ordinary dispatch and owned-close serialization remain held during
application. No Stop completion, lease acquisition, account mutation or new order
occurs. Later owned closes use unconsumed journal identities, including after a
zero-fill cancellation.

A post-commit feedback error does not mean the application was rolled back.
Inspect the retained receipt and retry the same application; never resubmit its
order. Repeated application is local and idempotent. Its result describes this
particular application, not current account balances or global flatness. CLI/native
integration and explicit rearm remain separate unfinished work.

### Local operator command

The [operator workflow](review/2026-11-11/recovery-operator/review.md) exposes
retained Spot evidence without loading credentials or requesting account data:

```text
simple-ai-trading recover-close --data-dir "<existing-owned-directory>" --client-order-id "<exact-owned-close-id>" --json
```

Inspection is the default and does not publish pending redo. Its
`execution_scope_sha256` identifies the original stored endpoint/product/key scope;
`recovery_sha256` identifies the retained closing evidence. It also distinguishes
local intent state from a committed application receipt. Neither is current-account
or flat-exposure proof, and application admission is explicitly unchecked.

To apply, use the same directory/client selection with `--action apply` and pass
those exact values through `--expected-scope-sha256` and
`--expected-recovery-sha256`. Optional `--runtime-control`, `--runtime-enrollment`
and `--execution-enrollment` select existing configured resources. There is no
prompt, implicit setup, account call or rearm. Error exit 2 after application
entry can mean a committed operation; inspect/retry this application rather than
submitting its order again.

The generated native expert dropdown registers `recover-close` under Risk's
Owned close recovery group. Its required selection and apply hashes use Extra
Flags. Missing arguments fail parsing; the default does not apply anything.
Guided forms and collection selection are separate remaining operator work.

### Bounded programmatic collection

The [stopped collection boundary](review/2026-11-11/close-collection/review.md)
provides a separate authenticated read API. Inspect the original unresolved
request with `closing_collection_sha256(store, scope=scope, client_id=client_id)`.
Pass its exact digest to `collect_stopped_closing_recovery` through
`expected_request_sha256`, together with that original scope, store, client ID,
matching Binance controller and explicitly supplied HMAC credentials. An optional
`execution_enrollment_path` selects the existing enrollment. It never retrieves
credentials from logs/history or grants account authority merely because a local
request exists. Do not place keys or secrets in command arguments.

The existing execution enrollment must be ready, and control must be ownerless,
paused and Stop-latched. One scoped order read and at most one matching trade
page share a bounded response deadline in a hidden owned worker. Exact terminal
fills and native cash must reconcile before retention. Cached evidence causes no
account refresh. Missing prerequisites or ambiguous responses preserve the
unresolved obligation. No order, application, Stop completion or rearm follows.
This API requires the existing locked optional HTTP runtime; missing dependencies
produce an actionable isolated-setup error. Current CLI/native `recover-close`
continues to inspect/apply already retained evidence and does not collect it.

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
