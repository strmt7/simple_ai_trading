# Local recovery operator workflow

`recover-close` now exposes retained close inspection and explicit local
application through the CLI. The generated native expert-command taxonomy
registers it under **Risk → Owned close recovery**. The reviewed base is
`8d648a464542dd6269d657f104106127a8dc204a`.

## Behavior

Inspection is the default. An explicit existing execution directory and exact
closing client ID are required. The command reads the original journal scope;
it does not load runtime credentials, construct a venue client or infer ownership
from the machine's current account configuration. The displayed scope digest
binds the original endpoint/product/key-identity record. It is not an account UID.
Raw credential fingerprints and workstation paths are not included in the result.

The read-only inspector validates the retained order/fill evidence in one SQLite
snapshot. It reports the local intent state and whether a matching application
receipt committed. It does not publish pending paired redo or inspect current
ledger projections. Source validation and an application receipt therefore remain
distinct from current application admission, account balances and flatness.

Apply requires both exact digests returned by inspection. Missing or changed
digests reject before application. The existing core then rechecks enrollment,
original scope, native receipts, inventory and ownerless Stop under its interlocks.
Custom controller and independent enrollment paths remain explicit options.
Application does not submit an order, query an account, acquire a lease or rearm.

Error results distinguish an application that was not started from an unknown
outcome after entering the application boundary. Human output explicitly explains
that a latter error may be post-commit: inspect and retry the same local
application, never resubmit its order. A committed receipt can be inspected even
while ledger publication still requires the existing recoverable-pair mechanism.

## Verification

**102 affected cases passed**, including the CLI parser/handler, application
backend, new-command regressions and native command-contract/launcher parity.
Inspection left all existing fixture files byte-identical. Tests reject malformed
scope, invalid selection, absent/mismatched digests and reflected filesystem-error
text. A directory containing spaces and custom control/execution enrollments work.

A real owned subprocess ran inspect, apply and repeat with socket connections,
runtime credential loading and client construction blocked. It applied the
retained cumulative close once and preserved a single closed-trade projection.
Another case interrupted after the durable application commit: inspection reported
the committed receipt without publishing redo; retry finished the intended pair.
No actual user ledger or account was accessed.

The native Release executable was rebuilt with two workers against the new
generated command header. Its presentation checks passed. The final Python
handler's additional status text does not change native compiler inputs or the
generated contract; those actual native dependency hashes remained unchanged.
No GUI was launched or activated. This establishes registration, build and contract
parity, not a newly exercised visual/keyboard workflow or a guided recovery form.

The [sealed evidence](evidence.json) binds the final source/archive, raw-log
digests, native compiler inputs and executable, and the actual source-bound
semantic-search receipts. Ruff passed; only authored ranges of legacy files were
formatted. No dependency or global host configuration was changed.

## Remaining boundaries

Authenticated terminal evidence must already have been retained under its separate
authority; this command adds no collection endpoint. Native futures/legacy
application, missing fill lineage, unvalued fees, account UID/availability,
instrument/dust qualification, lost-owner recovery, guided native forms, explicit
rearm and independent supervision remain unfinished or rejected as documented.

There was no market/source capture, production fit, benchmark or order. The
financial frontier stays 206 observations / 65 hypotheses / 37 historical scoped
mechanisms / zero qualified stable edges. Protected partials and old results remain
unchanged. This is an operator-integration checkpoint within the active full goal.
