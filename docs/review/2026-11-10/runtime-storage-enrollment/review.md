# Independent runtime storage enrollment

The original controller could recreate a lost database as fresh stopped state,
erase its Stop epoch and issue a new lease. The retained original reproduction
is preserved. The shared core now rejects normal access unless an independent
ready enrollment, database UUID and ordinary-file identities agree. Neither
database, namespace nor dispatch lock is automatically bootstrapped. This applies
to both venue controller adapters, not every native trading caller.

Explicit provisioning writes a prepared marker, commits the namespace and UUID,
then atomically publishes ready enrollment. New namespaces start paused with
Stop requested at epoch 1. Legacy adoption requires the exact inspected valid
record and preserves existing state and Stop history. Supported Polymarket v1
pause migration retains its transaction, hash and concurrency checks.

A prepared marker remains blocked after interruption. Explicit recovery requires
the same store ID, resources and exact current control record. A committed
enrollment may finish without resetting Stop; an exit before database commit
cannot become a fabricated recovery success. Missing independent enrollment
does not authorize rebinding an enrolled database or recreating its locks.

Owned closing retains its separate enrolled lock and can serialize without the
opening record/database. A configured enrollment is reused through the actual
runtime interlock; a mismatched ledger rejects closing. Missing close
serialization never becomes a new lock domain. This supplies neither missing
position evidence nor permission to reduce foreign inventory.

## Operator and resource changes

Local Polymarket CLI actions now inspect, provision and recover enrollment
without constructing an authenticated client. The configured enrollment path is
used consistently through status, pause, autonomous operation and Stop.
Inspection supplies the prepared store ID and current digest needed by recovery.
Status retains unavailable control as unknown and cannot open. Generated Windows
command declarations were regenerated and checked against the parser; no native
binary rebuild, rendered UI acceptance or release follows from those checks.

The original mutex dictionary retained 128 entries after 128 owners were
discarded in the retained original-source counterexample. Weak ownership now
releases unused path entries while live peers share the same mutex. Reentering
one lock instance rejects before replacing its cleanup. Post-acquisition identity
failures release the actual native lock and handle.

Both original atomic-write helpers left a temporary file after an injected
KeyboardInterrupt at replacement. Cancellation cleanup now covers BaseException
and preserves the pre-existing destination in the tested paths. Original leaked
bodies were renamed into inert private evidence. These are bounded resource
lifetime repairs, not proof that the entire application is leak-free.

## Verification and limits

The [publication binding](publication-binding.json) distinguishes exact tested
working bytes from Git's CRLF-to-LF normalization. Original source-phase hashes
remain unchanged; staged blob identities are recorded separately.

The [evidence](evidence.json) binds original source, counterexamples, pre-launch
snapshots, complete logs, actual semantic navigation and current production.
The 321-case combined lane passes. A later 121-case diagnostic/recovery lane
passes after changing only inspection diagnostics; AST comparison verifies all
remaining shared-core definitions unchanged. Those lanes overlap. Four additional
CLI/native declaration checks pass. Every stated lane has zero failures, errors
and skips.

Four contained child processes actually exit before/after enrollment commit,
covering both venue adapters. Committed prepared stores recover only with exact
evidence; incomplete stores remain blocked. Other controls cover file/namespace
loss, byte-identical replacement, malformed records, missing locks, migration,
Stop ordering, custom-path closing, SQLite URI escaping and reserved sidecars.
Earlier failure logs and the initial test-path preflight error remain retained.
No assertion was suppressed or historical result rewritten.

File identities are private local enrollment evidence, not hardcoded application
settings. Verified relocation/replacement migration, full in-place rollback
resistance, adversarial alteration of all local authority, physical reboot or
power-loss acceptance, independent process/credential fencing and universal
caller coverage remain incomplete. Binance's native roundtrip still needs
journal/runtime integration and qualified admission.

No real credential, venue account/order request, protected partial capture,
market sample, model fit, GPU benchmark, scanner retry, service or persistent
worker was used. Financial counts remain **205 observations / 65 hypotheses /
37 scoped mechanisms / zero qualified stable edges**. The full goal stays active.
