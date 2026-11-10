# Independent Binance execution-store enrollment

The offline regression at base `7e200edf275c749383eb859f7f6bc4a65a0bb0d4`
reproduced lost exposure memory: the first synthetic BUY filled and lost its
response, leaving FIRST_UNKNOWN. After retaining that journal under another
filename, a new operation created a fresh namespace, bought again and completed
its own SELL. Three synthetic native writes occurred; the original acquisition
remained outside the new operation. No real account or order was accessed.
The [evidence](evidence.json) binds the original sources and retained probe/result.

Paired opening now requires an independent execution enrollment. The implementation
reuses the strict runtime enrollment record and filesystem-identity checks. It
binds the journal's file identity and committed UUID to the controller's existing
opening and closing interlocks. The journal's existing scope binding still
qualifies endpoint, product and credential fingerprint. Missing, replaced,
truncated or unrecognized storage cannot be bootstrapped by native dispatch.

The gate runs before account preflight, under opening admission and immediately
before dispatch. First-leg preparation explicitly forbids creating the journal.
The second BUY has the same opening checks. A failure after UNKNOWN commits
preserves that obligation even if no native write subsequently occurs; it does
not fabricate a not-found result or authorize a replacement order.

Exact-order recovery and the receipt-owned second SELL keep their separate
scope, ownership, quantity and closing-interlock checks. Loss of the opening
enrollment does not grant authority, but does not prevent those operations when
their own evidence remains valid. Neither two reconciled legs nor enrollment
proves flat assets, native capital adequacy or profit.

## Explicit setup and interruption

Local setup requires paused, ownerless, Stop-latched control. It assigns the
execution UUID in the scoped journal transaction, writes a prepared independent
record before commit, verifies committed storage and then marks it ready.
Setup never clears Stop, acquires a lease or sends an account/order request.
The configured client is constructed and closed only to validate local scope.

An existing journal requires exact snapshot adoption. Inspection uses a bounded
SQLite read transaction and a 64 MiB serialization ceiling; its digest includes
the database's logical snapshot rather than assuming the main file contains
all WAL state. Scope and paired-position validation still apply. Adoption
preserves existing UNKNOWN obligations. An assigned UUID or existing marker
cannot be silently replaced by provisioning again.

Interrupted committed preparation needs the exact UUID and digest of the
prepared record, plus matching committed scope/resources. An uncommitted or
conflicting preparation remains blocked for reviewed restoration. These controls
detect loss/replacement; they do not resist coordinated rollback of every local
authority file. Verified relocation and universal writer fencing remain open.
See the [operator procedure](../../../SPOT_ROUNDTRIP.md#execution-storage-setup).

## Verification and limits

The final source-bound lane passes **170/170 cases**, with zero skips/errors.
It covers explicit setup/adoption, committed and uncommitted interruption,
journal/UUID/marker/interlock loss, opening-admission races, missing-store
noncreation, second-BUY rejection and preserved recovery/receipt-owned selling.
Three command/native-declaration checks pass. The generated declaration was
updated; no native executable was rebuilt or accepted visually.

The original 160-case lane had nine stale fixture assertions expecting no
journal before dispatch; explicit setup now creates it first. They were updated
to check absence of new obligations, while missing-store recovery still removes
the fixture journal and verifies noncreation. An intermediate 162-case lane
passed. The final result is a clean combined run, not a union of those lanes.

The existing contained 24-cycle cancellation/recovery lifecycle check also passes
with the new enrollment reads. After four warmup cycles, native handles remain
282, local mutex entries remain zero and private bytes remain 1,726,693,376 at
first, maximum and final samples. This bounds that tested lifecycle only; it
does not establish application-wide absence of leaks or speed gains.

No dependency environment, account, historical research result, market capture,
training target, benchmark, release or persistent registration changed. This is
a demonstrated capital-safety repair, not completion of the supervisor, whole-app
review or institutional readiness. Financial counts remain **205 observations /
65 hypotheses / 37 scoped mechanisms / 0 qualified stable edges**. Continue with
eligible structural economics, causal cost/capital evidence and the remaining
execution/supervision integration; do not expand training on inadequate labels.
