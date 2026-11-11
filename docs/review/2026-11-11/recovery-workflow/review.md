# Explicit operator collection and recovery

`recover-close` now connects pending-request inspection, separately requested
bounded collection and retained Spot application. Inspection and application
remain offline. A pending close no longer needs terminal evidence merely to be
inspected. This is a partial `EXEC-01` implementation, not complete trading or
capital-protection qualification. The source baseline is
`40bd11cd86770b2a87c50d7cf245f327e952cff8`.

The [procedure](../../../RUNTIME_CONTROL_STORAGE.md#local-operator-command) and
[evidence](evidence.json) bind the operator contract and tested source phases.

## Authority and state transitions

The command binds the original endpoint/product/key scope and exact request
digest. Current private configuration loads only inside admitted collection,
after existing execution enrollment and ownerless Stop checks. Configured
endpoint/product/key changes reject before any account read. Storage is checked
again after configuration loading. The collector rechecks the request digest
before querying; configuration callbacks cannot replace the inspected request.

One exact-order read and at most one matching trade page retain the existing
shared deadline, no-replay and hidden-process boundaries. Already retained
evidence follows a local no-read path without requiring credentials or the HTTP
optional runtime. A disappearing cached record fails instead of falling through
to an authenticated query. Collection neither applies inventory nor releases
UNKNOWN, Stop or ownership obligations. Apply remains a separate digest-bound
local action and retains its existing Spot cash, partial-fill and redo controls.

Read attempts, received responses, cached reuse, retention and application are
separate output states. No response is inferred from a timeout. A failure after
retention remains uncertain until local inspection; it does not claim that
evidence is absent or the original order was never sent. The command creates
no new order, account-wide query, credential argument, implicit setup or rearm.
Actual account reads still require their separate scope authority.

The generated native expert command is synchronized with the real parser,
including collection, request digest and bounded read-timeout options. Guided
forms and complete native operational acceptance remain outstanding.

## Verification and corrections

The final affected-domain run passes **205 cases** covering the real
inspect/collect/cache/apply parser and handler; zero, partial and full Spot fills;
original scope and request changes; missing enrollment and unresolved ownership;
timeouts, worker exit/rejection, post-retention uncertainty, unavailable private
configuration, redacted diagnostics and unchanged durable Stop. Native responses
are synthetic and exercise real owned hidden children. No real account or market
was queried. Existing worker lifetime/resource checks are included; no speed,
GPU, physical reboot or whole-product claim follows.

The first focused phase had 20 passes and one invalid owner-state fixture setup.
The subsequent domain phase retained 199 passes and found a genuine CLI defect:
`RuntimeControlBlocked` derives from `RuntimeError` and escaped the old local-error
handling. The command now returns a structured, sanitized block for that exact
control-error family. Added assertions cover both collect and apply. Original
failures, corrected phases and complete source/log bindings are preserved.

## Remaining work

`EXEC-01` remains partial: futures/legacy inventory application, guided forms,
account UID/capability qualification, lost-owner migration and controlled rearm
are unfinished. Independent full gateway/supervisor containment and durable risk
policy remain open. Next local slice is `SAFE-02`, subject to financial priority
and demonstrated capital/security failures. No historical result, financial
ledger, edge census, scanner quota or release authority changed.
