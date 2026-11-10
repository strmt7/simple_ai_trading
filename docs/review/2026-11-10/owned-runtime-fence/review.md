# Owned Binance runtime admission

The ordinary owned-position gateway previously accepted a synthetic native
opening while its persisted controller was paused and Stop was latched. The
retained probe made one synthetic write and recorded owned inventory without a
running lease. It did not access an account or venue. The
[evidence](evidence.json) binds the original source hashes and private raw result,
including the earlier Windows environment-import failure rather than hiding it.

## Repair

`submit_durable_open` now requires the existing Binance runtime lease and both
controller and execution-journal enrollment. The default resolves the controller
beside the position store and obtains the caller's existing lease from the
shared environment variable. An explicit interlock and execution-marker path
support configured programmatic callers. No gateway setup, lease acquisition,
Stop completion, heartbeat worker or restart installation is implicit.

The submission guard spans UNKNOWN precommit, native dispatch, exact response
validation, inventory recording and journal completion. Controller and execution
identities are rechecked before dispatch; journal preparation uses noncreating
access. Loss after precommit preserves uncertainty. A private copy of the caller's
position prevents later caller mutation from changing the journaled request.

Owned closing holds the separately enrolled closing lock through its own UNKNOWN,
dispatch and paired accounting. It snapshots the position/trade and retains
scope, ownership, partial-fill and acknowledgement checks. Stop, controller
database loss and opening-marker loss do not themselves disable owned reduction.
Loss of the closing lock rejects rather than creates a new serialization domain.
The two legacy nondurable position helpers now reject signed use and retain
their paper behavior.

The [operator procedure](../../../RUNTIME_CONTROL_STORAGE.md#ordinary-owned-binance-orders)
describes default and configured setup. The actual ordinary CLI and autonomous
call sites already route through this shared gateway. These checks fence those
paths; low-level transport primitives still require a complete caller inventory.

## Verification and scope

The execution lane passes 484 cases, including the actual CLI/autonomous paths,
lost acknowledgements, incomplete/partial fills, contradictory receipts,
cancellation and abrupt child-process exits. The supplementary lane covers
execution/controller enrollment, shared runtime control, paired intents and
Spot roundtrip execution/CLI. Its exact count, test-source hashes and native
result digests are retained in the evidence. Both lanes preserve their tested
source bytes before and after execution.

The initial integration lane had twelve fixture failures because existing
synthetic native cases did not supply the newly required authority. Those cases
now explicitly enroll real local control/journal storage and a lease, or only
closing authority where appropriate. Their original safety and accounting
assertions remain. Invalid-request cases still prove noncreation; no general
autouse authority fixture bypasses the new boundary.

Twenty observations after four warmup cycles in a 24-cycle synthetic owned
open/close run reported 282 handles throughout, zero retained local mutexes and
1,734,365,184 private bytes throughout. These are current-process observations
under the stated workload, not timing benchmarks or proof against indefinite
memory growth. The lease owner explicitly heartbeats between cycles. Abrupt-exit
helpers now use bounded hidden process capture with suspended Windows Job
enrollment and kill-on-close ownership. All verification helpers returned.
Unrelated host tasks were neither stopped nor reconfigured.

Changed Python scopes pass Ruff. Unrelated formatting churn in the large CLI
test module was removed with whole-module AST equivalence and exact preservation
of unaffected source text; only three intended definitions changed there.
Private logs, native JUnit host metadata and physical paths remain untracked.

No new market capture, account request, real native order, training run or
financial observation followed. Canonical ledgers and protected captures remain
unchanged; zero stable edges are qualified. This is a demonstrated capital-risk
repair, not financial or enterprise-readiness acceptance.

## Remaining work

An independent session-owned trading supervisor must still own heartbeat,
generation fencing, credential isolation and hard process deadlines. The guard
serializes admission and Stop; it cannot preempt an already admitted frozen
native call. Do not interpret this repair as universal writer fencing, verified
venue reduction, anti-rollback protection, capital qualification or reboot proof.
Native caller coverage, real account reconciliation, durable risk/model policy,
structural economics, the whole-repository semantic review and final bug hunt
remain active. No release or persistent job is authorized by this checkpoint.
