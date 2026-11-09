# Runtime interlock resource cleanup

Interlock setup and unlock failures could leave their file handle open. A
cancelled contended acquisition could also leave the process-local mutex held.
That resource leak can obstruct later Stop/control calls even after the original
operation fails. The old implementation's three regression controls reproduce
the missing handle closure and the blocked peer after cancellation.

The shared `_CrossProcessFileLock` now owns acquired resources through a cleanup
stack. It registers the local-mutex release and file close during acquisition,
then retains OS unlock, close and mutex release for the held context. Exit
attempts every registered release even when another cleanup raises. The tested
acquisition cancellation also unwinds those resources. Normal locking and the
existing ordered Stop/Pause/close behavior are preserved.

All **29 focused cleanup/runtime checks pass**, without a skip. They include
write/flush/fsync failure, unlock seek failure, combined unlock/close failure,
acquisition cancellation, normal exclusion and body failure. A second thread
reopens and locks the same real file after the injected errors. Cancellation's
peer check independently demonstrates mutex release because no OS lock had been
acquired. Tests retain the error traceback while checking unlock-failure cleanup.
The existing runtime suite also exercises persistent Stop, pause/resume, lease
expiry, fresh-interpreter visibility and cross-process owned-close ordering.

The [source-bound evidence](evidence.json) retains implementation/test hashes,
old/new regression results and private log bindings. Other production definitions
are AST-identical to the parent revision; no ledger schema, state law, budget or
order-authority gate changed. Ruff lint and formatting checks pass. All runtime
tests used new task-owned temporary directories; no actual user state, account,
credential, order or protected capture was touched.

That statement describes this exact phase. Its [source reconstruction](source-reconstruction.json)
recreates the module byte-for-byte from the parent commit and retained patch.
The following [storage repair](../../2026-11-10/runtime-control-storage/review.md)
changes initialization helpers; its 61-check combined result is the current
verification. The original hashes and 29-check evidence remain unchanged.

This is a scoped resource-lifetime repair. It does not implement an independent
trading supervisor, make a hanging native order call bounded, or prove abrupt
death, power-loss or reboot recovery. Actual supported-OS process containment,
durable policy/rearm, model-health quarantine and complete final review remain
open in the [capital architecture](../../../CAPITAL_PROTECTION_ARCHITECTURE.md).
The target supervisor is an application-session process; its durable state
persists separately. No machine service, recurring task, autostart hook or
persistent helper was installed. Non-Windows branches have not been executed in
this checkpoint. Financial counts remain **204 / 65 / 37 / 0 qualified stable
edges**; this repair adds no market observation.
