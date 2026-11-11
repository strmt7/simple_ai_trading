# Order admission after HTTP queue and startup

Baseline `5257185fe8b52b157286eaa29e43398a537e60e1` checked model source
admission before calling the venue adapter. Its owned SDK transport could then
wait for serialized HTTP capacity or initialize its child. The retained offline
reproduction revoked the source during child startup and observed an order POST
inside that child. This was a real gap between local approval and IPC dispatch.

The coordinator now supplies one transient admission scope for its synchronous
submission. The pinned SDK transport consults that scope for the exact order
POST. After capacity acquisition and child readiness, the parent rechecks the
intent's current age/expiry, runtime submission authority and supplied source
assertion before sending any order frame. The explicitly injected inline HTTP
fixture path checks the same assertion immediately before its native exchange.
Context-local state avoids a mutable callback on a shared SDK client; nested
scopes and failures restore their caller's context. No callback or authority is
serialized to the child or journal.

A callback denial carries separate evidence that transmission has not started
in this scope. Only that evidence, together with an unchanged submitting record
without matched quantity, permits the terminal journal code
`admission_revoked_before_http_dispatch`. The caller receives a blocked action.
The healthy child remains available. The transmission marker is set before a
send begins: partial sends, later failures, unproven adapter exceptions and any
earlier send in the same scope cannot produce an unsent rejection. Authenticated
order evidence is not overwritten by a conflicting local denial.

Reads and cancels do not consume this model-order assertion. Deterministic owned
closing remains independent of model source admission. Existing runtime,
ownership, reconciliation and closing gates remain controlling.

## Evidence and limits

[Sealed evidence](evidence.json) retains the failing baseline, corrected fixtures,
verification lanes and exact source hashes. Private source archives and logs are
preserved; the public summary does not reconstruct their bytes independently.
All 326 affected-domain checks passed. Targeted Ruff checks passed for the seven
changed Python modules. These checks are not whole-product qualification.
The authority fixture initially omitted the coordinator's reconciliation methods;
that fixture error and its failed lane remain recorded separately.

This boundary is the parent's IPC send. Source events can still occur after it,
including before the child reaches the venue. It does not establish atomic
source admission at remote exchange execution. The inspected pinned SDK uses a
synchronous order POST; arbitrary injected or independently threaded adapters
do not inherit this transport guarantee. Child hangs and ambiguous sends retain
their existing UNKNOWN/recovery obligations. Startup/capacity transport failures
remain conservative and do not become proven unsent merely because they appear
local. No automatic retry or order replay was added.

Verification uses local SDK/child fixtures, not exchange orders or credentials.
No external requests, financial observations, model training, benchmarks,
dependency changes, services, recurring tasks or release were part of this
repair. POSIX execution, independent gateway supervision, signing containment,
complete crash/reboot recovery and profitability remain unqualified.
